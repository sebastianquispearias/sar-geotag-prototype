"""
Visor de mapa open-source con Leaflet/OpenStreetMap.

Genera un archivo HTML y lo sirve via un mini servidor HTTP local
para que los tiles del mapa carguen correctamente (los navegadores
bloquean peticiones HTTPS desde file://).

Muestra:
  - La posición de la cámara (marcador azul)
  - Las posiciones estimadas de las personas detectadas (marcadores rojos)
"""

from __future__ import annotations

import http.server
import os
import socket
import tempfile
import threading
import webbrowser
from functools import partial
from pathlib import Path
from typing import List, Optional

from src.types import GeoPoint

# ── Mini servidor HTTP reutilizable ──
_server_instance: Optional[http.server.HTTPServer] = None
_server_port: int = 0


def _ensure_http_server(directory: str) -> int:
    """
    Levanta (una sola vez) un servidor HTTP sirviendo *directory*.
    Retorna el puerto asignado.
    """
    global _server_instance, _server_port

    if _server_instance is not None:
        return _server_port

    # Buscar un puerto libre
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _server_port = s.getsockname()[1]

    handler = partial(
        http.server.SimpleHTTPRequestHandler,
        directory=directory,
    )
    _server_instance = http.server.HTTPServer(("127.0.0.1", _server_port), handler)

    thread = threading.Thread(target=_server_instance.serve_forever, daemon=True)
    thread.start()
    return _server_port


def _generate_map_html(
    camera_lat: float,
    camera_lon: float,
    camera_altitude_m: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    geo_points: List[GeoPoint],
    image_path: Optional[str] = None,
) -> str:
    """
    Genera HTML puro con Leaflet.js (sin depender de folium instalado)
    para máxima portabilidad.
    """
    # Centro del mapa: promedio entre cámara y puntos, o solo cámara
    all_lats = [camera_lat] + [p.lat for p in geo_points]
    all_lons = [camera_lon] + [p.lon for p in geo_points]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Generar marcadores de personas (rojos)
    person_markers = ""
    for i, pt in enumerate(geo_points):
        person_markers += f"""
        L.circleMarker([{pt.lat}, {pt.lon}], {{
            radius: 10,
            fillColor: '#e74c3c',
            color: '#c0392b',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.85
        }}).addTo(map).bindPopup(
            '<b>🔴 Persona #{i+1}</b><br>' +
            '<b>{pt.label}</b><br>' +
            '<hr style="margin:4px 0">' +
            'Lat: {pt.lat:.7f}<br>' +
            'Lon: {pt.lon:.7f}<br>' +
            'Dist. suelo: {pt.distance_m:.1f} m<br>' +
            'Dist. directa: {pt.slant_distance_m:.1f} m<br>' +
            'Bearing: {pt.bearing_deg:.1f}°<br>' +
            'Confianza: {(pt.confidence or 0)*100:.0f}%'
        );

        L.polyline(
            [[{camera_lat}, {camera_lon}], [{pt.lat}, {pt.lon}]],
            {{color: '#e74c3c', weight: 2, dashArray: '5,5', opacity: 0.6}}
        ).addTo(map);
        """

    # Zoom adaptativo
    zoom = 18 if geo_points else 16

    image_note = ""
    if image_path:
        image_note = f"<br><small>Captura: {Path(image_path).name}</small>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' https: data:;">
    <title>SAR Geotag - Mapa de Detecciones</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin="anonymous" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
            crossorigin="anonymous"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; }}
        #header {{
            background: #2c3e50; color: white; padding: 12px 20px;
            display: flex; align-items: center; gap: 12px;
        }}
        #header h1 {{ font-size: 18px; font-weight: 600; }}
        #header .badge {{
            background: #e74c3c; padding: 3px 10px; border-radius: 12px;
            font-size: 13px;
        }}
        #map {{ height: calc(100vh - 50px); width: 100%; }}
        .info-panel {{
            background: white; padding: 10px 14px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 13px;
            line-height: 1.6;
        }}
        .leaflet-control-layers {{
            font-size: 13px;
        }}
        #loading {{
            position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
            background: #333; color: #fff; padding: 8px 18px; border-radius: 20px;
            font-size: 13px; z-index: 9999; display: none;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>📍 SAR Geotag - Mapa de Detecciones</h1>
        <span class="badge">{len(geo_points)} persona(s)</span>
    </div>
    <div id="map"></div>
    <div id="loading">⏳ Cargando mapa...</div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});

        // --- Múltiples proveedores de tiles ---
        var osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        }});

        var cartoDB = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}@2x.png', {{
            maxZoom: 20,
            subdomains: 'abcd',
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'
        }});

        var cartoLight = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
            maxZoom: 20,
            subdomains: 'abcd',
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'
        }});

        var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 19,
            attribution: '&copy; Esri, Maxar, Earthstar Geographics'
        }});

        // Intentar CartoDB Voyager primero (más confiable), fallback a OSM
        cartoDB.addTo(map);

        var baseLayers = {{
            "🗺️ CartoDB Voyager": cartoDB,
            "🗺️ CartoDB Claro": cartoLight,
            "🗺️ OpenStreetMap": osm,
            "🛰️ Satélite (Esri)": esriSat
        }};
        L.control.layers(baseLayers, null, {{position: 'topright'}}).addTo(map);

        // Indicador de carga
        var loadingEl = document.getElementById('loading');
        var tilesLoading = 0;
        map.on('loading', function() {{ loadingEl.style.display = 'block'; }});
        map.on('load', function() {{ loadingEl.style.display = 'none'; }});

        // Fallback: si CartoDB falla, intentar OSM
        var tileErrors = 0;
        cartoDB.on('tileerror', function() {{
            tileErrors++;
            if (tileErrors > 3) {{
                map.removeLayer(cartoDB);
                osm.addTo(map);
                console.log('CartoDB falló, usando OpenStreetMap');
            }}
        }});

        // Marcador de la cámara (azul)
        L.circleMarker([{camera_lat}, {camera_lon}], {{
            radius: 8,
            fillColor: '#3498db',
            color: '#2980b9',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.85
        }}).addTo(map).bindPopup(
            '<div class="info-panel">' +
            '<b>📷 Posición de la Cámara</b><br>' +
            'Lat: {camera_lat:.7f}<br>' +
            'Lon: {camera_lon:.7f}<br>' +
            'Altitud: {camera_altitude_m:.0f} m.s.n.m.<br>' +
            'Heading: {camera_yaw_deg:.0f}°  Pitch: {camera_pitch_deg:.1f}°' +
            '{image_note}' +
            '</div>'
        ).openPopup();

        {person_markers}

        // Ajustar vista para ver todos los puntos
        var allPoints = [[{camera_lat}, {camera_lon}]];
        {"".join(f"allPoints.push([{p.lat}, {p.lon}]);" for p in geo_points)}
        if (allPoints.length > 1) {{
            map.fitBounds(allPoints, {{padding: [50, 50]}});
        }}
    </script>
</body>
</html>"""
    return html


def show_map(
    camera_lat: float,
    camera_lon: float,
    geo_points: List[GeoPoint],
    camera_altitude_m: float = 0.0,
    camera_yaw_deg: float = 0.0,
    camera_pitch_deg: float = 0.0,
    image_path: Optional[str] = None,
    output_dir: str = "runs",
) -> str:
    """
    Genera un mapa HTML y lo abre en el navegador via servidor HTTP local.

    Retorna la ruta del archivo HTML generado.
    """
    html = _generate_map_html(
        camera_lat, camera_lon,
        camera_altitude_m, camera_yaw_deg, camera_pitch_deg,
        geo_points, image_path,
    )

    # Guardar en el directorio de salida
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_path = out_dir / "last_capture_map.html"
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Servir via HTTP local para que los tiles carguen correctamente
    abs_dir = str(out_dir.resolve())
    port = _ensure_http_server(abs_dir)
    url = f"http://127.0.0.1:{port}/{map_path.name}"
    webbrowser.open(url)

    print(f"[MAP] Mapa servido en: {url}")
    return str(map_path.resolve())


def show_triangulation_map(
    object_lat: float,
    object_lon: float,
    object_alt: float,
    camera_positions: List[Tuple[float, float, float]],
    camera_images: List[str] = None,  # Lista de imágenes base64
    info: dict = None,
    output_dir: str = "runs",
) -> str:
    """
    Muestra el resultado de triangulación con puntos y fotos.
    """
    if info is None:
        info = {}
    if camera_images is None:
        camera_images = [""] * len(camera_positions)
    
    # Centro del mapa
    all_lats = [object_lat] + [c[0] for c in camera_positions]
    all_lons = [object_lon] + [c[1] for c in camera_positions]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    algorithm = info.get("algorithm", "ray_intersection")
    ray_dist = info.get("ray_distance_m", 0)
    dist1 = info.get("distance_from_cam1_m", 0)
    dist2 = info.get("distance_from_cam2_m", 0)

    cam_colors = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800']

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>SAR Geotag - Triangulación</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; }}
        
        #header {{
            background: linear-gradient(135deg, #16213e, #1a1a2e);
            color: white; padding: 12px 20px;
            display: flex; align-items: center; gap: 15px;
            border-bottom: 2px solid #e94560;
        }}
        #header h1 {{ font-size: 18px; }}
        .badge {{ padding: 4px 10px; border-radius: 15px; font-size: 11px; font-weight: bold; }}
        .badge-green {{ background: #4CAF50; }}
        .badge-purple {{ background: #9C27B0; }}
        
        #map {{ height: calc(100vh - 50px); width: 100%; }}
        
        .info-panel {{
            position: absolute; top: 65px; right: 10px;
            background: rgba(22, 33, 62, 0.95); color: white;
            padding: 12px; border-radius: 10px; z-index: 1000;
            min-width: 200px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            border: 1px solid #e94560; font-size: 12px;
        }}
        .info-panel h3 {{ margin-bottom: 10px; color: #e94560; font-size: 13px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
        .info-row {{ display: flex; justify-content: space-between; margin: 5px 0; }}
        .info-row .label {{ color: #aaa; }}
        .info-row .value {{ color: #fff; font-weight: bold; }}
        .info-row .value.hl {{ color: #4CAF50; }}
        
        .toggle-panel {{
            position: absolute; bottom: 20px; left: 10px;
            background: rgba(22, 33, 62, 0.95); color: white;
            padding: 10px 12px; border-radius: 10px; z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4); border: 1px solid #444;
        }}
        .toggle-panel h4 {{ margin-bottom: 8px; font-size: 11px; color: #888; }}
        .toggle-btn {{
            display: flex; align-items: center; gap: 6px;
            margin: 4px 0; cursor: pointer; padding: 4px 6px;
            border-radius: 4px; font-size: 12px;
        }}
        .toggle-btn:hover {{ background: rgba(255,255,255,0.1); }}
        .toggle-btn input {{ cursor: pointer; }}
        .color-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
        
        .photo-panel {{
            position: absolute; bottom: 20px; right: 10px;
            background: rgba(22, 33, 62, 0.95); color: white;
            padding: 10px; border-radius: 10px; z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4); border: 1px solid #444;
            max-width: 450px;
        }}
        .photo-panel h4 {{ margin-bottom: 8px; font-size: 11px; color: #888; }}
        .photo-grid {{ display: flex; gap: 8px; }}
        .photo-item {{ text-align: center; }}
        .photo-item img {{ 
            width: 180px; height: auto; border-radius: 6px;
            border: 2px solid #444; cursor: pointer;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .photo-item img:hover {{ transform: scale(1.05); border-color: #e94560; }}
        .photo-item .label {{ font-size: 10px; margin-top: 4px; color: #aaa; }}
        
        .leaflet-popup-content-wrapper {{ background: rgba(22, 33, 62, 0.95); color: white; border-radius: 8px; }}
        .leaflet-popup-tip {{ background: rgba(22, 33, 62, 0.95); }}
        .leaflet-popup-content b {{ color: #e94560; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🎯 SAR Geotag - Triangulación</h1>
        <span class="badge badge-green">{len(camera_positions)} mediciones</span>
        <span class="badge badge-purple">{algorithm.replace('_', ' ').title()}</span>
    </div>
    
    <div id="map"></div>
    
    <div class="info-panel">
        <h3>📍 OBJETO DETECTADO</h3>
        <div class="info-row"><span class="label">Latitud:</span><span class="value">{object_lat:.7f}°</span></div>
        <div class="info-row"><span class="label">Longitud:</span><span class="value">{object_lon:.7f}°</span></div>
        <div class="info-row"><span class="label">Altitud:</span><span class="value">{object_alt:.1f} m</span></div>
        <hr style="border-color:#333; margin:8px 0;">
        <div class="info-row"><span class="label">Dist. Cam 1:</span><span class="value hl">{dist1:.2f} m</span></div>
        <div class="info-row"><span class="label">Dist. Cam 2:</span><span class="value hl">{dist2:.2f} m</span></div>
        <div class="info-row"><span class="label">Error:</span><span class="value">{ray_dist:.3f} m</span></div>
    </div>
    
    <div class="toggle-panel">
        <h4>MOSTRAR/OCULTAR</h4>
        <label class="toggle-btn"><input type="checkbox" id="toggleTarget" checked onchange="toggleLayer('target')">
            <span class="color-dot" style="background:#e94560"></span> Objeto</label>
        <label class="toggle-btn"><input type="checkbox" id="toggleCam1" checked onchange="toggleLayer('cam1')">
            <span class="color-dot" style="background:{cam_colors[0]}"></span> Cámara 1</label>
        <label class="toggle-btn"><input type="checkbox" id="toggleCam2" checked onchange="toggleLayer('cam2')">
            <span class="color-dot" style="background:{cam_colors[1]}"></span> Cámara 2</label>
        <label class="toggle-btn"><input type="checkbox" id="toggleLines" checked onchange="toggleLayer('lines')">
            <span class="color-dot" style="background:#fff"></span> Líneas</label>
    </div>
    
    <div class="photo-panel" id="photoPanel">
        <h4>📷 FOTOS CAPTURADAS</h4>
        <div class="photo-grid">
"""
    
    # Agregar thumbnails de fotos
    for i, img_b64 in enumerate(camera_images[:2]):
        if img_b64:
            html += f"""
            <div class="photo-item">
                <img src="data:image/jpeg;base64,{img_b64}" onclick="zoomToCamera({i+1})" title="Click para centrar en Cámara {i+1}">
                <div class="label">Cámara {i+1}</div>
            </div>
"""
        else:
            html += f"""
            <div class="photo-item">
                <div style="width:180px;height:120px;background:#333;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#666;">Sin foto</div>
                <div class="label">Cámara {i+1}</div>
            </div>
"""
    
    html += f"""
        </div>
    </div>
    
    <script>
        var map = L.map('map', {{maxZoom: 22}}).setView([{center_lat}, {center_lon}], 20);
        
        var cartoDB = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png', {{
            maxZoom: 22, maxNativeZoom: 20, subdomains: 'abcd'
        }}).addTo(map);
        var cartoLight = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}@2x.png', {{ maxZoom: 22, maxNativeZoom: 20, subdomains: 'abcd' }});
        var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ maxZoom: 22, maxNativeZoom: 19 }});
        L.control.layers({{ "🌙 Oscuro": cartoDB, "☀️ Claro": cartoLight, "🛰️ Satélite": esriSat }}).addTo(map);
        
        var layers = {{ target: L.layerGroup(), cam1: L.layerGroup(), cam2: L.layerGroup(), lines: L.layerGroup() }};
        Object.values(layers).forEach(l => l.addTo(map));
        
        var cameraCoords = [];
        
        // OBJETO - Punto rojo grande
        L.circleMarker([{object_lat}, {object_lon}], {{
            radius: 12, color: '#fff', fillColor: '#e94560', fillOpacity: 1, weight: 3
        }}).addTo(layers.target).bindPopup('<b>🎯 Objeto</b><br>Lat: {object_lat:.7f}<br>Lon: {object_lon:.7f}');
        
        // Círculo de error
        L.circle([{object_lat}, {object_lon}], {{
            radius: Math.max({ray_dist}, 0.5), color: '#e94560', fillColor: '#e94560', fillOpacity: 0.15, weight: 1, dashArray: '4,4'
        }}).addTo(layers.target);
"""

    # Agregar cámaras como PUNTOS
    for i, (lat, lon, alt) in enumerate(camera_positions):
        color = cam_colors[i % len(cam_colors)]
        dist = info.get(f"distance_from_cam{i+1}_m", 0)
        layer = f"cam{i+1}"
        
        html += f"""
        // Cámara {i+1} - Punto
        cameraCoords.push([{lat}, {lon}]);
        L.circleMarker([{lat}, {lon}], {{
            radius: 10, color: '#fff', fillColor: '{color}', fillOpacity: 1, weight: 3
        }}).addTo(layers.{layer}).bindPopup(
            '<b>📷 Cámara {i+1}</b><br>Lat: {lat:.7f}<br>Lon: {lon:.7f}<br>Alt: {alt:.1f}m<br><b style="color:#4CAF50">Dist: {dist:.2f}m</b>'
        );
        // Etiqueta
        L.marker([{lat}, {lon}], {{
            icon: L.divIcon({{ className: '', html: '<div style="background:{color};color:#fff;padding:2px 6px;border-radius:10px;font-size:10px;font-weight:bold;white-space:nowrap;transform:translateY(-20px)">Cam {i+1}</div>', iconAnchor: [20, 0] }})
        }}).addTo(layers.{layer});
        
        // Línea
        L.polyline([[{lat}, {lon}], [{object_lat}, {object_lon}]], {{
            color: '{color}', weight: 2, opacity: 0.7, dashArray: '8,6'
        }}).addTo(layers.lines);
        // Distancia en línea
        L.marker([({lat}+{object_lat})/2, ({lon}+{object_lon})/2], {{
            icon: L.divIcon({{ className: '', html: '<div style="background:rgba(0,0,0,0.8);color:{color};padding:2px 5px;border-radius:8px;font-size:10px;font-weight:bold">{dist:.1f}m</div>', iconAnchor: [20, 10] }})
        }}).addTo(layers.lines);
"""

    html += f"""
        function toggleLayer(name) {{
            var cb = document.getElementById('toggle' + name.charAt(0).toUpperCase() + name.slice(1));
            cb.checked ? map.addLayer(layers[name]) : map.removeLayer(layers[name]);
        }}
        
        function zoomToCamera(num) {{
            if (cameraCoords[num-1]) {{ map.setView(cameraCoords[num-1], 20); }}
        }}
        
        var bounds = L.latLngBounds([[{object_lat}, {object_lon}], {"".join(f"[{c[0]}, {c[1]}]," for c in camera_positions)}]);
        map.fitBounds(bounds, {{ padding: [80, 80], maxZoom: 21 }});
    </script>
</body>
</html>"""

    # Guardar y servir
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    map_path = out_dir_path / "triangulation_result.html"
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_dir = str(out_dir_path.resolve())
    port = _ensure_http_server(abs_dir)
    url = f"http://127.0.0.1:{port}/{map_path.name}"
    webbrowser.open(url)
    print(f"[MAP] Triangulación: {url}")
    return str(map_path.resolve())
