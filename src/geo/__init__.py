"""
Módulo de geolocalización.

Combina la distancia estimada (BBoxSizeEstimator) con la geometría
para producir coordenadas GPS aproximadas de cada persona detectada.
"""

from __future__ import annotations

import math
from typing import List

from src.types import CameraModel, Detection, Estimate, GeoPoint, Pose
from src.geometry.transforms import (
    bearing_to_object,
    destination_point,
    slant_to_ground_distance,
    ground_distance_with_altitude,
)


def geolocate_detections(
    detections: List[Detection],
    estimates: List[Estimate],
    cam: CameraModel,
    pose: Pose,
    camera_lat: float,
    camera_lon: float,
    camera_yaw_deg: float,
) -> List[GeoPoint]:
    """
    Para cada detección con distancia estimada, calcula las coordenadas
    geográficas aproximadas usando:
      1. Bearing = yaw_cámara + offset_angular (pinhole)
      2. Distancia horizontal = corrección por altura + pitch
      3. Destination point formula (esfera terrestre)

    Los datos necesarios para la inferencia (según el paper):
    - Posición GPS de la cámara (lat, lon)
    - Altitud / altura sobre el suelo
    - Orientación: yaw (heading), pitch, roll
    - FOV horizontal de la cámara
    - Bounding box del objeto detectado → centro horizontal
    - Distancia estimada (slant) al objeto

    Parámetros
    ----------
    detections : List[Detection]
        Detecciones de personas.
    estimates : List[Estimate]
        Estimaciones de distancia correspondientes.
    cam : CameraModel
        Modelo de cámara (resolución + FOV).
    pose : Pose
        Pose de la cámara (altura, pitch, etc.).
    camera_lat, camera_lon : float
        Posición GPS de la cámara en grados.
    camera_yaw_deg : float
        Heading de la cámara (0=Norte, 90=Este).

    Retorna
    -------
    List[GeoPoint]
        Puntos geográficos de cada persona detectada.
    """
    geo_points: List[GeoPoint] = []

    for det, est in zip(detections, estimates):
        if det.cls_name != "person":
            continue
        if est.distance_m is None or est.distance_m <= 0:
            continue

        # 1. Bearing absoluto desde la cámara al objeto
        bearing = bearing_to_object(
            camera_yaw_deg=camera_yaw_deg,
            obj_center_x_px=det.center_x,
            image_width_px=cam.width_px,
            hfov_deg=cam.hfov_deg,
        )

        # 2. Distancia horizontal en el plano del suelo
        #    Corrige la distancia slant usando la altura de la cámara
        #    (Pitágoras) y el ángulo de pitch.
        ground_dist = ground_distance_with_altitude(
            slant_distance_m=est.distance_m,
            camera_height_m=pose.height_m,
            pitch_deg=pose.pitch_deg,
        )

        # 3. Punto destino en la superficie terrestre
        lat2, lon2 = destination_point(
            lat_deg=camera_lat,
            lon_deg=camera_lon,
            bearing_deg=bearing,
            distance_m=ground_dist,
        )

        geo_points.append(GeoPoint(
            lat=lat2,
            lon=lon2,
            distance_m=ground_dist,
            bearing_deg=bearing,
            slant_distance_m=est.distance_m,
            camera_altitude_m=pose.altitude_m,
            confidence=est.confidence,
            label=f"Persona ({ground_dist:.1f}m)",
        ))

    return geo_points
