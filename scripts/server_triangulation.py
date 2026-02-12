"""
SAR Geotag - Servidor para recibir fotos del celular Android.

Recibe fotos con metadatos GPS/orientación via HTTP POST,
ejecuta YOLO para detectar personas, y permite al operador
hacer click para asignar mediciones.

Cuando se tienen 2 mediciones de la misma persona, ejecuta
Ray Intersection para calcular la posición.

Uso:
    python scripts/server_triangulation.py

El servidor escucha en http://0.0.0.0:5000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Agregar el directorio raíz al path para imports
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import base64
import json
import threading
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.vision.ultralytics_detector import UltralyticsDetector
from src.geometry.pinhole import create_measurement, Measurement
from src.geometry.geolocalization import geolocalize_from_measurements
from src.geo.map_viewer import show_triangulation_map

# ═══════════════════════════════════════════════════════════════════════════════
# Configuración
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = str(_root / "yolo11n.pt")
CONF_THRESHOLD = 0.5
SERVER_PORT = 5000

# Carpeta para guardar fotos recibidas (fuera del proyecto)
SAVE_DIR = Path.home() / "SAR_Geotag_Capturas"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Conversión de orientación Android → Cámara
# ═══════════════════════════════════════════════════════════════════════════════

def convert_android_orientation(yaw_deg: float, pitch_deg: float, roll_deg: float,
                                 img_width: int, img_height: int) -> dict:
    """
    Convierte orientación del sensor Android a orientación de la cámara.
    
    Problema: Android SensorManager.getOrientation() devuelve la orientación
    del DISPOSITIVO, no de la cámara:
    
    - Cuando sostienes el teléfono vertical (tomando foto normal):
      Android reporta pitch ≈ -90° (dispositivo vertical)
      Pero la cámara mira HORIZONTAL → camera pitch ≈ 0°
    
    - Cuando el teléfono está plano (pantalla arriba):
      Android reporta pitch ≈ 0° (dispositivo horizontal)
      Pero la cámara mira ARRIBA → camera pitch ≈ 90°
    
    Conversión: camera_pitch = -(device_pitch + 90°)
    
    También detecta si la foto es portrait (height > width) para
    intercambiar HFOV/VFOV.
    """
    is_portrait = img_height > img_width
    
    # --- Convertir Pitch ---
    # Android pitch: 0=plano, -90=vertical apuntando arriba
    # Camera pitch: 0=horizontal, negativo=mirando abajo
    camera_pitch = -(pitch_deg + 90.0)
    
    # --- Convertir Roll ---
    # En portrait, roll necesita ajuste
    camera_roll = roll_deg
    if is_portrait:
        # Normalizar roll a -180..180
        if camera_roll > 90:
            camera_roll = camera_roll - 180.0
        elif camera_roll < -90:
            camera_roll = camera_roll + 180.0
    
    # --- Yaw se mantiene igual (0=Norte, 90=Este) ---
    camera_yaw = yaw_deg
    
    return {
        "yaw": camera_yaw,
        "pitch": camera_pitch,
        "roll": camera_roll,
        "is_portrait": is_portrait,
    }


def get_camera_fov(hfov: float, vfov: float, img_width: int, img_height: int) -> tuple:
    """
    Devuelve el FOV correcto según la orientación de la imagen.
    
    Los FOV hardcodeados (67°/52°) son para LANDSCAPE.
    En PORTRAIT (height > width) se intercambian.
    """
    if img_height > img_width:
        # Portrait: el ancho de la imagen corresponde al lado corto del sensor
        return vfov, hfov  # intercambiar
    return hfov, vfov


# ═══════════════════════════════════════════════════════════════════════════════
# Estado global
# ═══════════════════════════════════════════════════════════════════════════════

# Diccionario de mediciones por persona: {"Persona A": [Measurement, ...], ...}
measurements_by_person: Dict[str, List[dict]] = {}

# Cola de fotos pendientes para procesar
pending_photos: List[dict] = []

# Foto actual mostrada en OpenCV
current_photo: Optional[dict] = None
current_detections: List = []

# Resultados de triangulación por persona
triangulation_results: Dict[str, dict] = {}  # {"Persona A": {"lat": ..., "lon": ..., "dist1": ..., ...}}

# Lock para thread safety
lock = threading.Lock()

# Detector YOLO
detector: Optional[UltralyticsDetector] = None

# ═══════════════════════════════════════════════════════════════════════════════
# Flask App
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)


@app.route("/capture", methods=["POST"])
def capture():
    """
    Endpoint para recibir fotos del celular Android.
    
    Espera JSON con:
    - image: string base64 de la imagen JPEG
    - timestamp: ISO 8601
    - gps: {latitude, longitude, altitude, accuracy}
    - orientation: {yaw, pitch, roll}
    - camera: {hfov, vfov, width, height}
    """
    global pending_photos
    
    try:
        data = request.get_json()
        
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400
        
        # Decodificar imagen
        image_b64 = data["image"]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extraer metadatos
        gps = data.get("gps", {})
        orientation = data.get("orientation", {})
        camera = data.get("camera", {})
        
        # Crear thumbnail para mostrar en el mapa
        thumb_size = (200, 150)
        image_thumb = image.copy()
        image_thumb.thumbnail(thumb_size)
        thumb_buffer = BytesIO()
        image_thumb.save(thumb_buffer, format='JPEG', quality=70)
        thumb_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # --- Obtener dimensiones de imagen ---
        img_w = camera.get("width", image_np.shape[1])
        img_h = camera.get("height", image_np.shape[0])
        
        # --- Convertir orientación Android → Cámara ---
        raw_yaw = orientation.get("yaw", 0)
        raw_pitch = orientation.get("pitch", 0)
        raw_roll = orientation.get("roll", 0)
        
        cam_orient = convert_android_orientation(
            raw_yaw, raw_pitch, raw_roll, img_w, img_h
        )
        
        # --- Corregir FOV para portrait/landscape ---
        raw_hfov = camera.get("hfov", 67)
        raw_vfov = camera.get("vfov", 52)
        corrected_hfov, corrected_vfov = get_camera_fov(raw_hfov, raw_vfov, img_w, img_h)
        
        photo_data = {
            "image": image_np,
            "image_b64": thumb_b64,  # Thumbnail para el mapa
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "gps": {
                "latitude": gps.get("latitude", 0),
                "longitude": gps.get("longitude", 0),
                "altitude": gps.get("altitude", 0),
                "accuracy": gps.get("accuracy", 0),
            },
            "orientation": {
                "yaw": cam_orient["yaw"],
                "pitch": cam_orient["pitch"],
                "roll": cam_orient["roll"],
            },
            "camera": {
                "hfov": corrected_hfov,
                "vfov": corrected_vfov,
                "width": img_w,
                "height": img_h,
            },
        }
        
        with lock:
            pending_photos.append(photo_data)
        
        print(f"\n{'='*60}")
        print(f"[RECIBIDO] FOTO #{len(pending_photos)}")
        print(f"{'='*60}")
        print(f"  GPS: ({photo_data['gps']['latitude']:.6f}, {photo_data['gps']['longitude']:.6f})")
        print(f"  Altitud: {photo_data['gps']['altitude']:.1f} m, Precisión: {photo_data['gps']['accuracy']:.1f} m")
        print(f"  Orientación ANDROID (raw):")
        print(f"    YAW={raw_yaw:.1f}°  PITCH={raw_pitch:.1f}°  ROLL={raw_roll:.1f}°")
        print(f"  Orientación CÁMARA (corregida):")
        print(f"    YAW={cam_orient['yaw']:.1f}°  PITCH={cam_orient['pitch']:.1f}°  ROLL={cam_orient['roll']:.1f}°")
        print(f"  {'📱 PORTRAIT' if cam_orient['is_portrait'] else '🖥️ LANDSCAPE'}")
        print(f"  FOV: H={corrected_hfov:.1f}° V={corrected_vfov:.1f}° (raw: H={raw_hfov:.1f}° V={raw_vfov:.1f}°)")
        print(f"  Resolución: {img_w}x{img_h}")
        
        # --- Guardar foto y JSON en carpeta externa ---
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_save_path = SAVE_DIR / f"foto_{ts}.jpg"
            json_save_path = SAVE_DIR / f"foto_{ts}.json"
            image.save(str(photo_save_path), "JPEG", quality=90)
            meta = {
                "timestamp": photo_data["timestamp"],
                "gps": photo_data["gps"],
                "orientation_raw": {"yaw": raw_yaw, "pitch": raw_pitch, "roll": raw_roll},
                "orientation_camera": photo_data["orientation"],
                "camera": photo_data["camera"],
            }
            with open(json_save_path, "w") as jf:
                json.dump(meta, jf, indent=2)
            print(f"  💾 Guardado: {photo_save_path.name}")
        except Exception as save_err:
            print(f"  [WARN] No se pudo guardar: {save_err}")
        
        # Contar mediciones totales
        total_measurements = sum(len(m) for m in measurements_by_person.values())
        
        response = {
            "status": "ok",
            "message": "Foto recibida, esperando asignación en laptop",
            "measurement_count": total_measurements,
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    """Estado actual del servidor, incluyendo persona seleccionada."""
    person_key = f"Persona {selected_person}"
    current_count = len(measurements_by_person.get(person_key, []))
    
    return jsonify({
        "status": "running",
        "pending_photos": len(pending_photos),
        "selected_person": selected_person,
        "measurement_count": current_count,
        "persons": {k: len(v) for k, v in measurements_by_person.items()},
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reinicia todas las mediciones."""
    global measurements_by_person, pending_photos, triangulation_results
    with lock:
        measurements_by_person = {}
        pending_photos = []
        triangulation_results = {}
    print("[RESET] Todas las mediciones eliminadas")
    return jsonify({"status": "ok"})


# ═══════════════════════════════════════════════════════════════════════════════
# Interfaz OpenCV
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_NAME = "SAR Geotag - Servidor"
selected_person = "A"  # Persona actualmente seleccionada
waiting_accept = False  # True cuando mostramos detección esperando [S/N]
pending_measurement = None  # Medición pendiente de aceptar


def auto_assign_detection(photo: dict, detections: list) -> None:
    """
    Auto-asigna la primera detección de persona encontrada.
    Muestra el punto amarillo y espera confirmación con [S].
    """
    global waiting_accept, pending_measurement
    
    if not detections:
        print("[AUTO] No se detectaron personas en esta foto")
        return
    
    det = detections[0]
    x1, y1, x2, y2 = det.xyxy
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    scale = photo.get("scale", 1.0)
    cx_original = int(cx / scale) if scale < 1.0 else cx
    cy_original = int(cy / scale) if scale < 1.0 else cy
    
    person_key = f"Persona {selected_person}"
    
    pending_measurement = {
        "pixel_x": cx_original,
        "pixel_y": cy_original,
        "pixel_display": (cx, cy),
        "gps": photo["gps"],
        "orientation": photo["orientation"],
        "camera": photo["camera"],
        "timestamp": photo["timestamp"],
        "image_b64": photo.get("image_b64", ""),
        "person_key": person_key,
    }
    waiting_accept = True
    
    count = len(measurements_by_person.get(person_key, []))
    print(f"\n[AUTO] Persona detectada en ({cx}, {cy})")
    print(f"  {person_key}: medición {count+1}/2")
    print(f"  Presiona [S] para aceptar, [N] para rechazar")


def accept_measurement() -> None:
    """Acepta la medición pendiente."""
    global waiting_accept, pending_measurement
    
    if not waiting_accept or pending_measurement is None:
        return
    
    person_key = pending_measurement["person_key"]
    
    current_count = len(measurements_by_person.get(person_key, []))
    if current_count >= 2:
        print(f"[WARN] {person_key} ya tiene 2 mediciones. Presiona [R] para reiniciar.")
        waiting_accept = False
        pending_measurement = None
        return
    
    measurement_data = {
        "pixel_x": pending_measurement["pixel_x"],
        "pixel_y": pending_measurement["pixel_y"],
        "gps": pending_measurement["gps"],
        "orientation": pending_measurement["orientation"],
        "camera": pending_measurement["camera"],
        "timestamp": pending_measurement["timestamp"],
        "image_b64": pending_measurement["image_b64"],
    }
    
    with lock:
        if person_key not in measurements_by_person:
            measurements_by_person[person_key] = []
        measurements_by_person[person_key].append(measurement_data)
    
    count = len(measurements_by_person[person_key])
    print(f"\n✅ [ACEPTADO] {person_key} - Medición #{count}")
    print(f"  Pixel: ({measurement_data['pixel_x']}, {measurement_data['pixel_y']})")
    print(f"  GPS: {measurement_data['gps']['latitude']:.6f}, {measurement_data['gps']['longitude']:.6f}")
    
    waiting_accept = False
    pending_measurement = None
    
    if count == 1:
        print(f"\n📷 Ahora envía la FOTO 2 desde otra posición")
    elif count >= 2:
        do_triangulation(person_key)


def reject_measurement() -> None:
    """Rechaza la medición pendiente."""
    global waiting_accept, pending_measurement
    waiting_accept = False
    pending_measurement = None
    print("❌ [RECHAZADO] Medición descartada. Envía otra foto.")


def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    """Callback del mouse - click manual en detección si no hay auto-detección."""
    global current_photo, current_detections, measurements_by_person
    
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    if waiting_accept:
        print("[INFO] Presiona [S] para aceptar o [N] para rechazar")
        return
    
    with lock:
        photo = current_photo
        detections = current_detections.copy() if current_detections else []
    
    if photo is None or not detections:
        return
    
    person_key = f"Persona {selected_person}"
    current_count = len(measurements_by_person.get(person_key, []))
    if current_count >= 2:
        print(f"[WARN] {person_key} ya tiene 2 mediciones. Presiona [R] para reiniciar.")
        return
    
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        if x1 <= x <= x2 and y1 <= y <= y2:
            auto_assign_detection(photo, [det])
            return
    
    print(f"[CLICK] No hay persona en ({x}, {y})")


def do_triangulation(person_key: str) -> None:
    """Ejecuta Ray Intersection para una persona."""
    global measurements_by_person
    
    data_list = measurements_by_person[person_key]
    if len(data_list) < 2:
        return
    
    print(f"\n{'='*60}")
    print(f"EJECUTANDO RAY INTERSECTION - {person_key}")
    print(f"{'='*60}")
    
    # Crear objetos Measurement
    measurements = []
    camera_positions = []
    camera_images = []  # Thumbnails para el mapa
    
    for i, data in enumerate(data_list[:2]):  # Usar las 2 primeras
        m = create_measurement(
            camera_lat=data["gps"]["latitude"],
            camera_lon=data["gps"]["longitude"],
            camera_alt=data["gps"]["altitude"] + 1.5,  # altura cámara
            pixel_x=data["pixel_x"],
            pixel_y=data["pixel_y"],
            image_width=data["camera"]["width"],
            image_height=data["camera"]["height"],
            hfov_deg=data["camera"]["hfov"],
            vfov_deg=data["camera"]["vfov"],
            yaw_deg=data["orientation"]["yaw"],
            pitch_deg=data["orientation"]["pitch"],
            roll_deg=data["orientation"]["roll"],
        )
        measurements.append(m)
        camera_positions.append((
            data["gps"]["latitude"],
            data["gps"]["longitude"],
            data["gps"]["altitude"] + 1.5,
        ))
        camera_images.append(data.get("image_b64", ""))
        print(f"\n  📷 Medición {i+1}:")
        print(f"     GPS: ({data['gps']['latitude']:.6f}, {data['gps']['longitude']:.6f})")
        print(f"     Pixel click: ({data['pixel_x']}, {data['pixel_y']}) en imagen {data['camera']['width']}x{data['camera']['height']}")
        print(f"     FOV: H={data['camera']['hfov']:.1f}°, V={data['camera']['vfov']:.1f}°")
        print(f"     YAW={data['orientation']['yaw']:.1f}°, PITCH={data['orientation']['pitch']:.1f}°")
        print(f"     Vector dirección ECEF: [{m.direction[0]:.4f}, {m.direction[1]:.4f}, {m.direction[2]:.4f}]")
    
    # Ejecutar triangulación
    result_point, info = geolocalize_from_measurements(measurements)
    
    if result_point is None:
        print(f"\n[ERROR] Triangulación fallida: {info.get('error', 'unknown')}")
        return
    
    # Guardar resultado para mostrar en UI
    triangulation_results[person_key] = {
        "lat": result_point.lat_deg,
        "lon": result_point.lon_deg,
        "alt": result_point.alt_m,
        "dist1": info.get("distance_from_cam1_m", 0),
        "dist2": info.get("distance_from_cam2_m", 0),
        "ray_error": info.get("ray_distance_m", 0),
        "algorithm": info.get("algorithm", "ray_intersection"),
    }
    
    print(f"\n🎯 {person_key.upper()} GEOLOCALIZADA:")
    print(f"   Latitud:  {result_point.lat_deg:.10f}°")
    print(f"   Longitud: {result_point.lon_deg:.10f}°")
    print(f"   Altitud:  {result_point.alt_m:.2f} m")
    
    if "distance_from_cam1_m" in info:
        print(f"   Distancia desde cámara 1: {info['distance_from_cam1_m']:.2f} m")
    if "distance_from_cam2_m" in info:
        print(f"   Distancia desde cámara 2: {info['distance_from_cam2_m']:.2f} m")
    if "ray_distance_m" in info:
        print(f"   Error rayos: {info['ray_distance_m']:.4f} m")
    
    print(f"{'='*60}\n")
    
    # Mostrar en mapa (con imágenes)
    show_triangulation_map(
        object_lat=result_point.lat_deg,
        object_lon=result_point.lon_deg,
        object_alt=result_point.alt_m,
        camera_positions=camera_positions,
        camera_images=camera_images,
        info=info,
        output_dir=str(_root / "runs"),
    )


def draw_ui(frame: np.ndarray) -> np.ndarray:
    """Dibuja la interfaz sobre el frame."""
    global current_detections, selected_person, measurements_by_person, triangulation_results
    
    h, w = frame.shape[:2]
    
    # Dibujar detecciones YOLO
    for det in current_detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Persona {det.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
    
    # Punto de medición pendiente (anillo rojo grande)
    if waiting_accept and pending_measurement:
        px, py = pending_measurement["pixel_display"]
        cv2.circle(frame, (px, py), 14, (0, 0, 255), 3)
        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
    
    # Panel superior
    cv2.rectangle(frame, (0, 0), (w, 85), (0, 0, 0), -1)
    cv2.putText(frame, "SAR Geotag - Servidor", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (76, 175, 80), 2)
    
    person_key = f"Persona {selected_person}"
    count = len(measurements_by_person.get(person_key, []))
    
    if waiting_accept:
        cv2.putText(frame, f"Persona {selected_person} | [S] Aceptar  [N] Rechazar",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Medicion {count+1}/2 pendiente de confirmar",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
    else:
        cv2.putText(frame, f"Persona: {selected_person} ({count}/2) | "
                    f"[A-Z] cambiar | [F] adjuntar | [R] reset | [Q] salir", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if count == 0:
            cv2.putText(frame, "Esperando FOTO 1...",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        elif count == 1:
            cv2.putText(frame, "Foto 1 OK. Esperando FOTO 2 desde otra posicion...",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        elif count >= 2:
            cv2.putText(frame, "Triangulacion completada",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    # Panel lateral derecho - mediciones (semitransparente)
    if measurements_by_person:
        overlay = frame.copy()
        panel_h = 30 + len(measurements_by_person) * 25
        cv2.rectangle(overlay, (w - 200, 90), (w, 90 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "Mediciones:", (w - 190, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset = 135
        for person, meas in measurements_by_person.items():
            color = (0, 255, 0) if len(meas) >= 2 else (255, 255, 0)
            text = f"{person}: {len(meas)}/2"
            if person == person_key:
                text = f"> {text}"
            cv2.putText(frame, text, (w - 190, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 25
    
    # Panel lateral izquierdo - RESULTADOS
    if triangulation_results:
        overlay = frame.copy()
        panel_h = 50 + len(triangulation_results) * 110
        cv2.rectangle(overlay, (0, 100), (300, 100 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "RESULTADOS:", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos = 155
        for person, result in triangulation_results.items():
            cv2.putText(frame, f"{person}:", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 22
            cv2.putText(frame, f"  Dist cam1: {result['dist1']:.2f} m", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            y_pos += 18
            cv2.putText(frame, f"  Dist cam2: {result['dist2']:.2f} m", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            y_pos += 18
            cv2.putText(frame, f"  Error: {result['ray_error']:.4f} m", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y_pos += 20
            cv2.putText(frame, f"  Lat: {result['lat']:.7f}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_pos += 16
            cv2.putText(frame, f"  Lon: {result['lon']:.7f}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_pos += 25
    
    # Instrucción central
    if not current_detections and not waiting_accept:
        cv2.putText(frame, "Esperando foto del celular... [F] para adjuntar desde PC", 
                    (w // 2 - 250, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Carpeta de guardado
    cv2.putText(frame, f"Guardando en: {SAVE_DIR}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    
    return frame

def process_pending_photo() -> bool:
    """Procesa la siguiente foto pendiente. Retorna True si había foto."""
    global pending_photos, current_photo, current_detections, detector
    
    # No procesar si hay medición esperando aceptar
    if waiting_accept:
        return False
    
    with lock:
        if not pending_photos:
            return False
        photo = pending_photos.pop(0)
    
    image = photo["image"]
    
    h, w = image.shape[:2]
    max_width = 1100
    max_height = 700
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        photo["image"] = image
        photo["scale"] = scale
        print(f"[RESIZE] Imagen redimensionada de {w}x{h} a {new_w}x{new_h} (escala: {scale:.2f})")
    else:
        photo["scale"] = 1.0
    
    if detector is not None:
        detections = detector.detect(image)
        new_detections = [d for d in detections if d.cls_name == "person"]
        print(f"[YOLO] Detectadas {len(new_detections)} personas")
    else:
        new_detections = []
    
    with lock:
        current_photo = photo
        current_detections = new_detections
    
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass
    
    # Auto-asignar primera detección
    if new_detections:
        auto_assign_detection(photo, new_detections)
    
    return True


def load_photo_from_file() -> None:
    """Abre un diálogo de Windows para adjuntar una foto desde el PC."""
    global pending_photos
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar foto",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")],
        )
        root.destroy()
        
        if not file_path:
            print("[FILE] Cancelado")
            return
        
        json_path = Path(file_path).with_suffix(".json")
        
        image = Image.open(file_path)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Crear thumbnail
        thumb = image.copy()
        thumb.thumbnail((200, 150))
        buf = BytesIO()
        thumb.save(buf, format='JPEG', quality=70)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        if json_path.exists():
            with open(json_path) as jf:
                meta = json.load(jf)
            photo_data = {
                "image": image_np,
                "image_b64": thumb_b64,
                "timestamp": meta.get("timestamp", datetime.now().isoformat()),
                "gps": meta.get("gps", {"latitude": 0, "longitude": 0, "altitude": 0, "accuracy": 0}),
                "orientation": meta.get("orientation_camera", meta.get("orientation", {"yaw": 0, "pitch": 0, "roll": 0})),
                "camera": meta.get("camera", {"hfov": 67, "vfov": 52, "width": image_np.shape[1], "height": image_np.shape[0]}),
            }
            print(f"[FILE] Cargada: {Path(file_path).name} + {json_path.name}")
        else:
            photo_data = {
                "image": image_np,
                "image_b64": thumb_b64,
                "timestamp": datetime.now().isoformat(),
                "gps": {"latitude": 0, "longitude": 0, "altitude": 0, "accuracy": 0},
                "orientation": {"yaw": 0, "pitch": 0, "roll": 0},
                "camera": {"hfov": 67, "vfov": 52, "width": image_np.shape[1], "height": image_np.shape[0]},
            }
            print(f"[FILE] Cargada: {Path(file_path).name} (sin JSON de metadatos)")
            print(f"  [WARN] Sin GPS/orientación - los resultados no serán precisos")
        
        with lock:
            pending_photos.append(photo_data)
        
    except Exception as e:
        print(f"[ERROR] Al cargar foto: {e}")


def run_opencv_ui():
    """Loop principal de la interfaz OpenCV."""
    global selected_person, detector, waiting_accept, pending_measurement
    
    # Cargar detector YOLO
    print(f"[YOLO] Cargando modelo: {MODEL_PATH}")
    try:
        detector = UltralyticsDetector(
            model_name=MODEL_PATH,
            conf=CONF_THRESHOLD,
        )
        print("[YOLO] Modelo cargado correctamente")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar YOLO: {e}")
        detector = None
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print(f"\n{'='*60}")
    print(f"  SERVIDOR INICIADO")
    print(f"  Escuchando en: http://0.0.0.0:{SERVER_PORT}")
    print(f"  Fotos guardadas en: {SAVE_DIR}")
    print(f"{'='*60}")
    print(f"\nCONTROLES:")
    print(f"  [A-Z]   = Seleccionar persona")
    print(f"  [S]     = Aceptar medición")
    print(f"  [N]     = Rechazar medición")
    print(f"  [F]     = Adjuntar foto desde PC")
    print(f"  [R]     = Reiniciar mediciones")
    print(f"  [Q]     = Salir")
    print("-"*60 + "\n")
    
    while True:
        process_pending_photo()
        
        if current_photo is not None:
            frame = current_photo["image"].copy()
        else:
            frame = blank_frame.copy()
        
        frame = draw_ui(frame)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            if waiting_accept:
                accept_measurement()
        elif key == ord('n') or key == ord('N'):
            if waiting_accept:
                reject_measurement()
        elif key == ord('f') or key == ord('F'):
            if not waiting_accept:
                load_photo_from_file()
        elif key == ord('r') or key == ord('R'):
            if not waiting_accept:
                with lock:
                    measurements_by_person.clear()
                    pending_photos.clear()
                    triangulation_results.clear()
                print("[RESET] Mediciones reiniciadas")
        elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
            if not waiting_accept:
                selected_person = chr(key).upper()
                print(f"[SELECT] Persona {selected_person}")
    
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Iniciar Flask en un thread separado
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True),
        daemon=True
    )
    flask_thread.start()
    
    # Ejecutar UI de OpenCV en el thread principal
    run_opencv_ui()


if __name__ == "__main__":
    main()
