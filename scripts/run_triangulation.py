"""
SAR Geotag - Modo Triangulación (Ray Intersection del Paper)

Este script implementa el algoritmo Ray Intersection del paper
"Object Geolocalization Using Consumer-Grade Devices"

Flujo:
1. Seleccionar cámara
2. Configurar GPS y orientación
3. Tomar primera medición (clic en objeto)
4. MOVER la cámara a otra posición
5. Actualizar GPS/orientación
6. Tomar segunda medición (clic en el MISMO objeto)
7. Ejecutar triangulación (Ray Intersection)
8. Ver resultado en mapa

Controles:
- Click: Añadir medición del punto seleccionado
- [G]: Geolocalizar (ejecutar Ray Intersection)
- [U]: Actualizar posición GPS (después de moverte)
- [R]: Reiniciar mediciones
- [Z]: Deshacer última medición
- [Q]: Salir
"""

from __future__ import annotations

import sys
from pathlib import Path

# Agregar el directorio raíz al path para imports
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from src.viz.ui import show_camera_selector, enumerate_cameras
from src.viz.gps_dialog import show_gps_input_dialog, GpsSetup
from src.geo.triangulation import (
    TriangulationSession,
    draw_measurement_overlay,
    save_triangulation_result,
)
from src.geo.map_viewer import show_triangulation_map
from src.vision.ultralytics_detector import UltralyticsDetector

WINDOW_NAME = "SAR Geotag - Triangulación (Paper)"


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Ruta por defecto al config
_DEFAULT_CONFIG = str(_root / "configs" / "default.yaml")


class TriangulationApp:
    """Aplicación principal de triangulación."""

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        self.cfg = _load_config(config_path)
        self.session = TriangulationSession()
        
        # Estado del mouse
        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.pending_click: Optional[Tuple[int, int]] = None

        # Configuración de cámara
        cam_cfg = self.cfg.get("camera", {})
        self.hfov_deg = float(cam_cfg.get("hfov_deg", 78.0))
        self.vfov_deg = float(cam_cfg.get("vfov_deg", 44.0))
        self.desired_w = int(cam_cfg.get("width", 0) or 0)
        self.desired_h = int(cam_cfg.get("height", 0) or 0)

        # GPS/Orientación actual - valores por defecto de la ubicación del usuario
        self.camera_lat = -12.02964
        self.camera_lon = -77.08645
        self.camera_alt = 63.7888  # altitud + altura cámara
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.camera_roll = 0.0

        # Output
        log_cfg = self.cfg.get("logging", {})
        self.out_dir = str(log_cfg.get("out_dir", "runs"))
        
        # Detector YOLO para detección de personas
        self.detector: Optional[UltralyticsDetector] = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Inicializa el detector YOLO."""
        try:
            model_cfg = self.cfg.get("model", {})
            model_path = str(_root / model_cfg.get("path", "yolo11n.pt"))
            conf_threshold = float(model_cfg.get("conf_threshold", 0.5))
            
            self.detector = UltralyticsDetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
            )
            print(f"[YOLO] Detector cargado: {model_path}")
        except Exception as e:
            print(f"[WARN] No se pudo cargar detector YOLO: {e}")
            self.detector = None

    def _detect_persons(self, frame: np.ndarray) -> list:
        """Detecta personas en el frame usando YOLO."""
        if self.detector is None:
            return []
        
        detections = self.detector.detect(frame)
        # Filtrar solo personas (clase 0 en YOLO)
        persons = [d for d in detections if d.class_id == 0]
        return persons

    def _draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Dibuja las detecciones de personas en el frame."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Rectángulo verde
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Etiqueta
            label = f"Persona {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Centro de la detección (para hacer click)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        
        return frame

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Callback del mouse para capturar clics."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.pending_click = (x, y)

    def _update_gps_dialog(self) -> bool:
        """
        Muestra el diálogo para actualizar la posición GPS.
        Retorna True si se actualizó, False si se canceló.
        """
        gps_cfg = self.cfg.get("gps", {})
        
        # Calcular la altura de cámara para mostrar en el diálogo
        # (camera_alt incluye altitud + altura, necesitamos separarlo)
        base_altitude = 62.7888  # altitud base por defecto
        default_height = 1.0     # altura cámara por defecto
        
        result = show_gps_input_dialog(
            default_lat=self.camera_lat,
            default_lon=self.camera_lon,
            default_altitude=base_altitude,
            default_height=default_height,
            default_yaw=self.camera_yaw,
            default_pitch=self.camera_pitch,
            default_roll=self.camera_roll,
        )

        if result is None:
            return False

        self.camera_lat = result.lat
        self.camera_lon = result.lon
        self.camera_alt = result.altitude_m + result.height_m  # Altitud total de la cámara
        self.camera_yaw = result.yaw_deg
        self.camera_pitch = result.pitch_deg
        self.camera_roll = result.roll_deg

        print(f"[GPS] Actualizado: lat={self.camera_lat:.6f}, lon={self.camera_lon:.6f}, "
              f"alt={self.camera_alt:.1f}m, yaw={self.camera_yaw:.0f}°")
        return True

    def _add_measurement(self, frame: np.ndarray, pixel_x: int, pixel_y: int) -> None:
        """Añade una medición en el punto especificado."""
        h, w = frame.shape[:2]

        result = self.session.add_measurement(
            frame=frame,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=w,
            image_height=h,
            hfov_deg=self.hfov_deg,
            vfov_deg=self.vfov_deg,
            camera_lat=self.camera_lat,
            camera_lon=self.camera_lon,
            camera_alt=self.camera_alt,
            camera_yaw=self.camera_yaw,
            camera_pitch=self.camera_pitch,
            camera_roll=self.camera_roll,
        )

        print(f"[MEDICIÓN #{result['measurement_num']}] Pixel: ({pixel_x}, {pixel_y})")
        print(f"  Cámara: lat={self.camera_lat:.6f}, lon={self.camera_lon:.6f}, "
              f"alt={self.camera_alt:.1f}m")
        print(f"  Orientación: yaw={self.camera_yaw:.0f}°, pitch={self.camera_pitch:.1f}°")

        if "validation" in result:
            v = result["validation"]
            print(f"  Separación de orígenes: {v['origin_separation_m']:.2f}m")
            print(f"  Ángulo entre rayos: {v['ray_angle_deg']:.1f}°")
            for warn in v.get("warnings", []):
                print(f"  ⚠️ {warn}")

    def _do_geolocalization(self) -> None:
        """Ejecuta la geolocalización con las mediciones actuales."""
        if not self.session.can_geolocalize():
            print("[ERROR] Se necesitan al menos 2 mediciones para triangular.")
            print("  1. Haz clic en el objeto")
            print("  2. Muévete y presiona [U] para actualizar GPS")
            print("  3. Haz clic en el MISMO objeto")
            return

        print("\n" + "="*60)
        print("EJECUTANDO RAY INTERSECTION (Paper Algorithm)")
        print("="*60)

        result_point, info = self.session.geolocalize()

        if result_point is None:
            print(f"[ERROR] Geolocalización falló: {info.get('error', 'unknown')}")
            return

        print(f"\n🎯 OBJETO GEOLOCALIZADO:")
        print(f"   Latitud:  {result_point.lat_deg:.10f}°")
        print(f"   Longitud: {result_point.lon_deg:.10f}°")
        print(f"   Altitud:  {result_point.alt_m:.2f} m (sobre elipsoide WGS84)")
        print(f"\n   Algoritmo: {info.get('algorithm', 'unknown')}")
        
        if "distance_from_cam1_m" in info:
            print(f"   Distancia desde cámara 1: {info['distance_from_cam1_m']:.2f} m")
        if "distance_from_cam2_m" in info:
            print(f"   Distancia desde cámara 2: {info['distance_from_cam2_m']:.2f} m")
        if "ray_distance_m" in info:
            print(f"   Distancia entre rayos (calidad): {info['ray_distance_m']:.4f} m")
            if info['ray_distance_m'] > 1.0:
                print(f"   ⚠️ Rayos muy separados - precisión reducida")

        # Guardar resultado
        txt_path = save_triangulation_result(
            self.session, result_point, info, self.out_dir
        )
        print(f"\n   Resultado guardado: {txt_path}")

        # Mostrar en mapa
        camera_positions = [
            (c.camera_lat, c.camera_lon, c.camera_alt)
            for c in self.session.captures
        ]
        show_triangulation_map(
            object_lat=result_point.lat_deg,
            object_lon=result_point.lon_deg,
            object_alt=result_point.alt_m,
            camera_positions=camera_positions,
            info=info,
            output_dir=self.out_dir,
        )

        print("="*60 + "\n")

    def run(self) -> None:
        """Ejecuta la aplicación de triangulación."""
        print("\n" + "="*60)
        print("  SAR GEOTAG - MODO TRIANGULACIÓN (Paper)")
        print("  Algoritmo: Ray Intersection")
        print("="*60)

        # Paso 1: Seleccionar cámara
        print("\nPaso 1: Seleccionar cámara...")
        webcam_index = show_camera_selector()
        print(f"Cámara seleccionada: {webcam_index}")

        # Paso 2: Configurar GPS inicial
        print("\nPaso 2: Configurar posición GPS inicial...")
        if not self._update_gps_dialog():
            print("Cancelado. Saliendo.")
            return

        # Abrir cámara
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            raise RuntimeError(f"No pude abrir la webcam (index={webcam_index})")

        if self.desired_w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_w)
        if self.desired_h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_h)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        print("\n" + "-"*55)
        print("  CONTROLES:")
        print("  [Click] = Añadir medición en el punto")
        print("  [U]     = Actualizar GPS (después de moverte)")
        print("  [G]     = Geolocalizar (Ray Intersection)")
        print("  [R]     = Reiniciar todas las mediciones")
        print("  [Z]     = Deshacer última medición")
        print("  [Q]     = Salir")
        print("-"*55)
        print("\nINSTRUCCIONES:")
        print("1. Haz clic en el objeto que quieres geolocalizar")
        print("2. MUÉVETE a otra posición (1-3 metros de lado)")
        print("3. Presiona [U] para actualizar tu GPS y orientación")
        print("4. Haz clic en el MISMO objeto")
        print("5. Presiona [G] para ejecutar Ray Intersection")
        print("-"*55 + "\n")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Detectar personas con YOLO
                detections = self._detect_persons(frame)

                # Procesar click pendiente
                if self.pending_click is not None:
                    px, py = self.pending_click
                    self._add_measurement(frame, px, py)
                    self.pending_click = None

                # Dibujar detecciones YOLO
                display_frame = frame.copy()
                display_frame = self._draw_detections(display_frame, detections)

                # Dibujar overlay de mediciones
                display_frame = draw_measurement_overlay(
                    display_frame, self.session, self.mouse_pos
                )

                # Mostrar info de GPS actual
                h, w = display_frame.shape[:2]
                gps_text = f"GPS: {self.camera_lat:.6f}, {self.camera_lon:.6f} | Yaw: {self.camera_yaw:.0f}"
                cv2.putText(
                    display_frame, gps_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                )
                
                # Mostrar contador de detecciones
                det_text = f"Personas: {len(detections)}"
                cv2.putText(
                    display_frame, det_text,
                    (w - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                )

                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    break

                elif key == ord("u") or key == ord("U"):
                    # Actualizar GPS
                    print("\n[GPS] Abriendo diálogo de actualización...")
                    self._update_gps_dialog()

                elif key == ord("g") or key == ord("G"):
                    # Geolocalizar
                    self._do_geolocalization()

                elif key == ord("r") or key == ord("R"):
                    # Reiniciar
                    self.session.clear()
                    print("[RESET] Todas las mediciones eliminadas.")

                elif key == ord("z") or key == ord("Z"):
                    # Deshacer
                    if self.session.remove_last():
                        print(f"[UNDO] Medición eliminada. Quedan: {len(self.session.captures)}")
                    else:
                        print("[UNDO] No hay mediciones que eliminar.")

        finally:
            cap.release()
            cv2.destroyAllWindows()


def run_triangulation(config_path: str = _DEFAULT_CONFIG) -> None:
    """Punto de entrada para el modo triangulación."""
    app = TriangulationApp(config_path)
    app.run()


if __name__ == "__main__":
    run_triangulation()
