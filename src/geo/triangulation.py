"""
Gestor de mediciones para triangulación según el paper.

Implementa el flujo de trabajo para capturar 2+ mediciones del mismo objeto
y geolocalizarlo usando Ray Intersection o Gradient Descent.

Flujo (del paper, Figure 3):
1. Primera medición: capturar imagen + seleccionar objeto
2. Mover el dispositivo a otra posición
3. Segunda medición: capturar imagen + seleccionar el mismo objeto
4. Ejecutar Ray Intersection para obtener geolocalización
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.geometry.pinhole import Measurement, create_measurement
from src.geometry.geolocalization import (
    geolocalize_from_measurements,
    validate_measurements,
)
from src.geometry.ecef import WGS84Point


@dataclass
class MeasurementCapture:
    """Datos capturados en una medición."""
    measurement: Measurement
    frame: np.ndarray
    timestamp: str
    pixel_x: int
    pixel_y: int
    camera_lat: float
    camera_lon: float
    camera_alt: float
    camera_yaw: float
    camera_pitch: float
    camera_roll: float


@dataclass
class TriangulationSession:
    """
    Sesión de triangulación para un objeto.
    
    Almacena las mediciones y ejecuta la geolocalización cuando hay suficientes.
    """
    captures: List[MeasurementCapture] = field(default_factory=list)
    target_label: str = "Persona"

    def add_measurement(
        self,
        frame: np.ndarray,
        pixel_x: int,
        pixel_y: int,
        image_width: int,
        image_height: int,
        hfov_deg: float,
        vfov_deg: float,
        camera_lat: float,
        camera_lon: float,
        camera_alt: float,
        camera_yaw: float,
        camera_pitch: float,
        camera_roll: float,
    ) -> dict:
        """
        Añade una medición a la sesión.
        
        Retorna un diccionario con información sobre la medición y validación.
        """
        # Crear la medición según el paper
        measurement = create_measurement(
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            camera_alt=camera_alt,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=image_width,
            image_height=image_height,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            yaw_deg=camera_yaw,
            pitch_deg=camera_pitch,
            roll_deg=camera_roll,
        )

        capture = MeasurementCapture(
            measurement=measurement,
            frame=frame.copy(),
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            camera_alt=camera_alt,
            camera_yaw=camera_yaw,
            camera_pitch=camera_pitch,
            camera_roll=camera_roll,
        )

        self.captures.append(capture)

        result = {
            "measurement_num": len(self.captures),
            "pixel": (pixel_x, pixel_y),
            "camera_pos": (camera_lat, camera_lon, camera_alt),
        }

        # Validar si ya tenemos 2+ mediciones
        if len(self.captures) >= 2:
            validation = validate_measurements(
                self.captures[-2].measurement,
                self.captures[-1].measurement,
            )
            result["validation"] = validation

        return result

    def can_geolocalize(self) -> bool:
        """Retorna True si hay suficientes mediciones válidas."""
        return len(self.captures) >= 2

    def geolocalize(self) -> Tuple[Optional[WGS84Point], dict]:
        """
        Ejecuta la geolocalización usando las mediciones capturadas.
        
        Usa Ray Intersection (2 mediciones) o Gradient Descent (3+).
        """
        if not self.can_geolocalize():
            return None, {"error": "Se necesitan al menos 2 mediciones"}

        measurements = [c.measurement for c in self.captures]
        return geolocalize_from_measurements(measurements)

    def clear(self) -> None:
        """Limpia todas las mediciones de la sesión."""
        self.captures.clear()

    def remove_last(self) -> bool:
        """Elimina la última medición. Retorna True si había algo que eliminar."""
        if self.captures:
            self.captures.pop()
            return True
        return False


def draw_measurement_overlay(
    frame: np.ndarray,
    session: TriangulationSession,
    current_pixel: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Dibuja información sobre las mediciones en el frame.
    
    Muestra:
    - Número de mediciones capturadas
    - Instrucciones para el usuario
    - Punto seleccionado actual (si hay)
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Fondo semi-transparente para el panel de estado
    panel_h = 80
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)

    # Información de mediciones
    num_captures = len(session.captures)
    
    if num_captures == 0:
        status = "PASO 1: Haz clic en el objeto a geolocalizar"
        color = (100, 200, 255)  # Naranja claro
    elif num_captures == 1:
        status = "PASO 2: Muevete y haz clic en el MISMO objeto"
        color = (100, 255, 200)  # Verde claro
    else:
        status = f"{num_captures} mediciones - Presiona [G] para geolocalizar o sigue agregando"
        color = (100, 255, 100)  # Verde

    # Texto de estado
    cv2.putText(
        overlay, status,
        (15, h - panel_h + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
    )

    # Controles
    controls = "[Click]=Medir  [G]=Geolocalizar  [R]=Reiniciar  [Z]=Deshacer"
    cv2.putText(
        overlay, controls,
        (15, h - panel_h + 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )

    # Indicador de mediciones en esquina superior
    badge_text = f"Mediciones: {num_captures}/2+"
    cv2.rectangle(overlay, (w - 160, 5), (w - 5, 35), (60, 60, 60), -1)
    cv2.putText(
        overlay, badge_text,
        (w - 155, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    # Dibujar punto seleccionado actual
    if current_pixel is not None:
        px, py = current_pixel
        # Cruz de mira
        cv2.line(overlay, (px - 20, py), (px + 20, py), (0, 255, 255), 2)
        cv2.line(overlay, (px, py - 20), (px, py + 20), (0, 255, 255), 2)
        cv2.circle(overlay, (px, py), 8, (0, 255, 255), 2)

    # Dibujar marcadores de mediciones previas (si el frame es el mismo tamaño)
    for i, cap in enumerate(session.captures):
        px, py = cap.pixel_x, cap.pixel_y
        # Círculo con número
        cv2.circle(overlay, (px, py), 12, (0, 200, 0), -1)
        cv2.circle(overlay, (px, py), 12, (255, 255, 255), 2)
        cv2.putText(
            overlay, str(i + 1),
            (px - 5, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    return overlay


def save_triangulation_result(
    session: TriangulationSession,
    result_point: WGS84Point,
    info: dict,
    output_dir: str,
) -> str:
    """
    Guarda los resultados de la triangulación en disco.
    
    Retorna la ruta del archivo de resultados.
    """
    out_path = Path(output_dir) / "triangulation"
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar imágenes de cada medición
    for i, cap in enumerate(session.captures):
        img_path = out_path / f"measurement_{timestamp}_{i+1}.png"
        # Dibujar el punto en la imagen
        frame_annotated = cap.frame.copy()
        cv2.circle(frame_annotated, (cap.pixel_x, cap.pixel_y), 10, (0, 0, 255), -1)
        cv2.circle(frame_annotated, (cap.pixel_x, cap.pixel_y), 10, (255, 255, 255), 2)
        cv2.imwrite(str(img_path), frame_annotated)

    # Guardar archivo de texto con resultados
    txt_path = out_path / f"result_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Resultado de Triangulación (Ray Intersection) ===\n\n")
        f.write(f"Fecha: {datetime.now().isoformat()}\n")
        f.write(f"Algoritmo: {info.get('algorithm', 'unknown')}\n\n")
        
        f.write("--- Geolocalización del Objeto ---\n")
        f.write(f"Latitud:  {result_point.lat_deg:.10f}°\n")
        f.write(f"Longitud: {result_point.lon_deg:.10f}°\n")
        f.write(f"Altitud:  {result_point.alt_m:.2f} m (sobre elipsoide)\n\n")

        if "distance_from_cam1_m" in info:
            f.write(f"Distancia desde cámara 1: {info['distance_from_cam1_m']:.2f} m\n")
        if "distance_from_cam2_m" in info:
            f.write(f"Distancia desde cámara 2: {info['distance_from_cam2_m']:.2f} m\n")
        if "ray_distance_m" in info:
            f.write(f"Distancia entre rayos: {info['ray_distance_m']:.4f} m (calidad)\n")

        f.write("\n--- Mediciones ---\n")
        for i, cap in enumerate(session.captures):
            f.write(f"\nMedición {i+1}:\n")
            f.write(f"  Timestamp: {cap.timestamp}\n")
            f.write(f"  Cámara GPS: ({cap.camera_lat:.8f}, {cap.camera_lon:.8f})\n")
            f.write(f"  Cámara Alt: {cap.camera_alt:.2f} m\n")
            f.write(f"  Orientación: yaw={cap.camera_yaw:.1f}° pitch={cap.camera_pitch:.1f}° roll={cap.camera_roll:.1f}°\n")
            f.write(f"  Pixel: ({cap.pixel_x}, {cap.pixel_y})\n")

    return str(txt_path)
