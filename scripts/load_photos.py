#!/usr/bin/env python3
"""
Script para cargar fotos desde el PC y calcular triangulación.

Extrae metadatos EXIF de las fotos:
- GPS (latitud, longitud, altitud)
- Orientación (si está disponible)

Uso:
    python scripts/load_photos.py foto1.jpg foto2.jpg
    python scripts/load_photos.py --folder ./fotos/
    python scripts/load_photos.py --interactive
"""

from __future__ import annotations

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Agregar path del proyecto
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.vision.ultralytics_detector import UltralyticsDetector
from src.geometry.pinhole import create_measurement
from src.geometry.geolocalization import geolocalize_from_measurements
from src.geo.map_viewer import show_triangulation_map


# ═══════════════════════════════════════════════════════════════════════════════
# Extracción de EXIF
# ═══════════════════════════════════════════════════════════════════════════════

def get_exif_data(image: Image.Image) -> dict:
    """Extrae todos los datos EXIF de una imagen."""
    exif_data = {}
    try:
        exif = image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
    except Exception:
        pass
    return exif_data


def get_gps_info(exif_data: dict) -> Optional[dict]:
    """Extrae información GPS del EXIF."""
    if "GPSInfo" not in exif_data:
        return None
    
    gps_info = {}
    gps_data = exif_data["GPSInfo"]
    
    for tag_id, value in gps_data.items():
        tag = GPSTAGS.get(tag_id, tag_id)
        gps_info[tag] = value
    
    # Convertir a formato decimal
    def convert_to_degrees(value):
        """Convierte coordenadas GPS a grados decimales."""
        try:
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        except Exception:
            return None
    
    result = {}
    
    # Latitud
    if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if lat and gps_info["GPSLatitudeRef"] == "S":
            lat = -lat
        result["latitude"] = lat
    
    # Longitud
    if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if lon and gps_info["GPSLongitudeRef"] == "W":
            lon = -lon
        result["longitude"] = lon
    
    # Altitud
    if "GPSAltitude" in gps_info:
        try:
            alt = float(gps_info["GPSAltitude"])
            if "GPSAltitudeRef" in gps_info and gps_info["GPSAltitudeRef"] == 1:
                alt = -alt
            result["altitude"] = alt
        except Exception:
            result["altitude"] = 0
    else:
        result["altitude"] = 0
    
    # Dirección (heading/yaw)
    if "GPSImgDirection" in gps_info:
        try:
            result["heading"] = float(gps_info["GPSImgDirection"])
        except Exception:
            pass
    
    return result if "latitude" in result and "longitude" in result else None


def load_photo_with_metadata(path: str) -> Optional[dict]:
    """Carga una foto y extrae sus metadatos."""
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] No existe: {path}")
        return None
    
    try:
        image = Image.open(path)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extraer EXIF
        exif_data = get_exif_data(image)
        gps_info = get_gps_info(exif_data)
        
        if gps_info is None:
            print(f"[WARN] Sin GPS en: {path.name}")
            return None
        
        # Crear thumbnail
        thumb = image.copy()
        thumb.thumbnail((200, 150))
        thumb_buffer = BytesIO()
        thumb.save(thumb_buffer, format='JPEG', quality=70)
        thumb_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        photo_data = {
            "path": str(path),
            "image": image_np,
            "image_b64": thumb_b64,
            "width": image_np.shape[1],
            "height": image_np.shape[0],
            "gps": {
                "latitude": gps_info["latitude"],
                "longitude": gps_info["longitude"],
                "altitude": gps_info.get("altitude", 0),
            },
            "orientation": {
                "yaw": gps_info.get("heading", 0),
                "pitch": 0,
                "roll": 0,
            },
        }
        
        print(f"[OK] {path.name}: GPS=({gps_info['latitude']:.6f}, {gps_info['longitude']:.6f})")
        
        return photo_data
        
    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Interfaz interactiva para seleccionar persona
# ═══════════════════════════════════════════════════════════════════════════════

class PhotoSelector:
    """Interfaz para seleccionar una persona en las fotos."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(_root / "yolo11n.pt")
        self.detector = None
        self.selected_points = []
        self.current_image = None
        self.current_detections = []
        
    def load_detector(self):
        """Carga el detector YOLO."""
        if self.detector is None:
            print(f"[YOLO] Cargando modelo...")
            self.detector = UltralyticsDetector(model_name=self.model_path, conf=0.5)
            print(f"[YOLO] Modelo cargado")
    
    def select_person(self, photo_data: dict, photo_num: int) -> Optional[Tuple[int, int]]:
        """
        Muestra la foto con detecciones YOLO y permite seleccionar una persona.
        
        Returns:
            (pixel_x, pixel_y) del centro de la persona seleccionada, o None.
        """
        self.load_detector()
        
        image = photo_data["image"].copy()
        
        # Redimensionar para mostrar
        h, w = image.shape[:2]
        max_size = 900
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        # Detectar personas
        detections = self.detector.detect(image)
        person_detections = [d for d in detections if d.cls_name == "person"]
        
        self.current_image = image
        self.current_detections = person_detections
        self.current_scale = scale
        self.selected_point = None
        
        # Dibujar detecciones
        display = image.copy()
        for i, det in enumerate(person_detections):
            x1, y1, x2, y2 = det.xyxy
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(display, f"#{i+1} {det.conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # UI
        cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(display, f"Foto {photo_num} - Click en la persona a triangular",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Detectadas: {len(person_detections)} personas | [Q] Cancelar",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        window_name = f"Foto {photo_num} - Seleccionar persona"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_click)
        
        cv2.imshow(window_name, display)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            if self.selected_point is not None:
                break
        
        cv2.destroyWindow(window_name)
        
        if self.selected_point:
            # Convertir a coordenadas originales
            px, py = self.selected_point
            original_x = int(px / scale)
            original_y = int(py / scale)
            return (original_x, original_y)
        
        return None
    
    def _on_click(self, event, x, y, flags, param):
        """Callback del mouse."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        # Buscar si el click está dentro de alguna detección
        for det in self.current_detections:
            x1, y1, x2, y2 = det.xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.selected_point = (cx, cy)
                print(f"[CLICK] Persona seleccionada en ({cx}, {cy})")
                return
        
        print(f"[CLICK] No hay persona en ({x}, {y})")


# ═══════════════════════════════════════════════════════════════════════════════
# Triangulación
# ═══════════════════════════════════════════════════════════════════════════════

def triangulate_from_photos(
    photos: List[dict],
    pixels: List[Tuple[int, int]],
    hfov: float = 67,
    vfov: float = 52,
) -> None:
    """
    Ejecuta triangulación con los datos de las fotos.
    """
    if len(photos) < 2 or len(pixels) < 2:
        print("[ERROR] Se necesitan al menos 2 fotos con selección")
        return
    
    print(f"\n{'='*60}")
    print("EJECUTANDO RAY INTERSECTION")
    print(f"{'='*60}")
    
    measurements = []
    camera_positions = []
    camera_images = []
    
    for i, (photo, pixel) in enumerate(zip(photos[:2], pixels[:2])):
        m = create_measurement(
            camera_lat=photo["gps"]["latitude"],
            camera_lon=photo["gps"]["longitude"],
            camera_alt=photo["gps"]["altitude"] + 1.5,
            pixel_x=pixel[0],
            pixel_y=pixel[1],
            image_width=photo["width"],
            image_height=photo["height"],
            hfov_deg=hfov,
            vfov_deg=vfov,
            yaw_deg=photo["orientation"]["yaw"],
            pitch_deg=photo["orientation"]["pitch"],
            roll_deg=photo["orientation"]["roll"],
        )
        measurements.append(m)
        camera_positions.append((
            photo["gps"]["latitude"],
            photo["gps"]["longitude"],
            photo["gps"]["altitude"] + 1.5,
        ))
        camera_images.append(photo.get("image_b64", ""))
        
        print(f"  Foto {i+1}: GPS=({photo['gps']['latitude']:.6f}, {photo['gps']['longitude']:.6f})")
        print(f"           Pixel=({pixel[0]}, {pixel[1]}), Yaw={photo['orientation']['yaw']:.1f}°")
    
    # Ejecutar triangulación
    result_point, info = geolocalize_from_measurements(measurements)
    
    if result_point is None:
        print(f"\n[ERROR] Triangulación fallida: {info.get('error', 'unknown')}")
        return
    
    print(f"\n🎯 OBJETO GEOLOCALIZADO:")
    print(f"   Latitud:  {result_point.lat_deg:.10f}°")
    print(f"   Longitud: {result_point.lon_deg:.10f}°")
    print(f"   Altitud:  {result_point.alt_m:.2f} m")
    
    if "distance_from_cam1_m" in info:
        print(f"   Distancia desde foto 1: {info['distance_from_cam1_m']:.2f} m")
    if "distance_from_cam2_m" in info:
        print(f"   Distancia desde foto 2: {info['distance_from_cam2_m']:.2f} m")
    if "ray_distance_m" in info:
        print(f"   Error rayos: {info['ray_distance_m']:.4f} m")
    
    print(f"{'='*60}\n")
    
    # Mostrar en mapa
    show_triangulation_map(
        object_lat=result_point.lat_deg,
        object_lon=result_point.lon_deg,
        object_alt=result_point.alt_m,
        camera_positions=camera_positions,
        camera_images=camera_images,
        info=info,
        output_dir=str(_root / "runs"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Modo interactivo
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_mode():
    """Modo interactivo para seleccionar fotos y personas."""
    print("\n" + "="*60)
    print("  SAR GEOTAG - Carga de Fotos desde PC")
    print("="*60)
    print("\nEste modo permite cargar 2 fotos con GPS y calcular")
    print("la posición de una persona usando triangulación.\n")
    
    # Solicitar rutas de fotos
    photos = []
    pixels = []
    
    selector = PhotoSelector()
    
    for i in range(2):
        while True:
            path = input(f"\n📷 Ruta de la foto {i+1} (o 'q' para salir): ").strip()
            
            if path.lower() == 'q':
                print("Cancelado.")
                return
            
            # Remover comillas si las hay
            path = path.strip('"').strip("'")
            
            photo = load_photo_with_metadata(path)
            if photo:
                # Si no tiene yaw, preguntar
                if photo["orientation"]["yaw"] == 0:
                    try:
                        yaw = input(f"   → Ingrese el yaw/heading (0-360, 0=Norte): ").strip()
                        if yaw:
                            photo["orientation"]["yaw"] = float(yaw)
                    except ValueError:
                        pass
                
                # Seleccionar persona
                pixel = selector.select_person(photo, i+1)
                if pixel:
                    photos.append(photo)
                    pixels.append(pixel)
                    break
                else:
                    print("   → No se seleccionó ninguna persona")
            else:
                print(f"   → No se pudo cargar la foto o no tiene GPS")
    
    # Ejecutar triangulación
    if len(photos) >= 2:
        # Preguntar FOV
        print("\n📐 Campo de visión de la cámara:")
        try:
            hfov = input("   HFOV horizontal (default 67°): ").strip()
            hfov = float(hfov) if hfov else 67
            vfov = input("   VFOV vertical (default 52°): ").strip()
            vfov = float(vfov) if vfov else 52
        except ValueError:
            hfov, vfov = 67, 52
        
        triangulate_from_photos(photos, pixels, hfov, vfov)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Carga fotos desde PC y calcula triangulación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    python scripts/load_photos.py --interactive
    python scripts/load_photos.py foto1.jpg foto2.jpg
    python scripts/load_photos.py --folder ./fotos/
        """
    )
    
    parser.add_argument("photos", nargs="*", help="Rutas a las fotos (mínimo 2)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Modo interactivo")
    parser.add_argument("--folder", "-f", type=str,
                        help="Cargar todas las fotos de una carpeta")
    parser.add_argument("--hfov", type=float, default=67,
                        help="Campo de visión horizontal (default: 67°)")
    parser.add_argument("--vfov", type=float, default=52,
                        help="Campo de visión vertical (default: 52°)")
    
    args = parser.parse_args()
    
    # Modo interactivo
    if args.interactive or (not args.photos and not args.folder):
        interactive_mode()
        return
    
    # Cargar fotos desde argumentos
    photo_paths = []
    
    if args.folder:
        folder = Path(args.folder)
        if folder.exists():
            photo_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG"))
            photo_paths = sorted(photo_paths)[:2]
    
    if args.photos:
        photo_paths = [Path(p) for p in args.photos[:2]]
    
    if len(photo_paths) < 2:
        print("[ERROR] Se necesitan al menos 2 fotos")
        print("Usa --interactive para modo interactivo")
        return
    
    # Cargar fotos
    photos = []
    for path in photo_paths:
        photo = load_photo_with_metadata(str(path))
        if photo:
            photos.append(photo)
    
    if len(photos) < 2:
        print("[ERROR] No se pudieron cargar 2 fotos con GPS válido")
        return
    
    # Seleccionar personas
    selector = PhotoSelector()
    pixels = []
    
    for i, photo in enumerate(photos):
        pixel = selector.select_person(photo, i+1)
        if pixel:
            pixels.append(pixel)
        else:
            print(f"[ERROR] No se seleccionó persona en foto {i+1}")
            return
    
    # Triangular
    triangulate_from_photos(photos, pixels, args.hfov, args.vfov)


if __name__ == "__main__":
    main()
