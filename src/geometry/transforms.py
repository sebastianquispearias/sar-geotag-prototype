"""
Funciones de geometría para convertir detecciones en coordenadas geográficas.

Implementa:
  1. Cálculo del bearing (azimut) desde la cámara al objeto detectado
     usando la posición horizontal en píxeles y el FOV de la cámara.
  2. Fórmula de "destination point" (punto destino) sobre esfera terrestre
     para obtener lat/lon del objeto a partir de la posición GPS de la cámara,
     el bearing y la distancia estimada.

Referencia: "Object Geolocalization Using Consumer-Grade Devices" paper.
"""

from __future__ import annotations

import math
from typing import Tuple

# Radio medio de la Tierra (metros)
EARTH_RADIUS_M = 6_371_000.0


def bearing_to_object(
    camera_yaw_deg: float,
    obj_center_x_px: int,
    image_width_px: int,
    hfov_deg: float,
) -> float:
    """
    Calcula el bearing absoluto (grados, sentido horario desde el Norte)
    hacia un objeto detectado en la imagen.

    Parámetros
    ----------
    camera_yaw_deg : float
        Heading/yaw de la cámara (grados, 0 = Norte, 90 = Este).
    obj_center_x_px : int
        Posición horizontal (u) del centro del objeto en píxeles.
    image_width_px : int
        Ancho de la imagen en píxeles.
    hfov_deg : float
        Campo de visión horizontal de la cámara (grados).

    Retorna
    -------
    float
        Bearing absoluto en grados [0, 360).
    """
    # Focal length horizontal (píxeles) — modelo pinhole
    fx = (image_width_px / 2.0) / math.tan(math.radians(hfov_deg / 2.0))

    # Offset desde el centro de la imagen
    delta_u = obj_center_x_px - (image_width_px / 2.0)

    # Ángulo relativo (positivo = derecha)
    delta_theta_deg = math.degrees(math.atan2(delta_u, fx))

    # Bearing absoluto
    bearing = (camera_yaw_deg + delta_theta_deg) % 360.0
    return bearing


def destination_point(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> Tuple[float, float]:
    """
    Calcula el punto destino sobre la superficie terrestre (esfera)
    dado un punto de partida, un bearing y una distancia.

    Fórmula estándar de "destination point given distance and bearing".

    Parámetros
    ----------
    lat_deg, lon_deg : float
        Coordenadas del punto de partida (grados).
    bearing_deg : float
        Bearing en grados (0 = Norte, 90 = Este).
    distance_m : float
        Distancia en metros.

    Retorna
    -------
    (lat2_deg, lon2_deg) : Tuple[float, float]
        Coordenadas del punto destino en grados.
    """
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    theta = math.radians(bearing_deg)
    d_over_R = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_over_R)
        + math.cos(lat1) * math.sin(d_over_R) * math.cos(theta)
    )

    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(d_over_R) * math.cos(lat1),
        math.cos(d_over_R) - math.sin(lat1) * math.sin(lat2),
    )

    return (math.degrees(lat2), math.degrees(lon2))


def slant_to_ground_distance(slant_distance_m: float, pitch_deg: float) -> float:
    """
    Convierte distancia oblicua (slant) a distancia horizontal en el suelo.

    Si la cámara tiene un ángulo de pitch, la distancia medida es la
    línea de visión; la distancia en el suelo es la proyección horizontal.
    """
    return slant_distance_m * math.cos(math.radians(abs(pitch_deg)))


def ground_distance_with_altitude(
    slant_distance_m: float,
    camera_height_m: float,
    pitch_deg: float,
) -> float:
    """
    Calcula la distancia horizontal en el suelo teniendo en cuenta
    la altura de la cámara sobre el suelo.

    Cuando la cámara está elevada (dron, trípode, edificio), la distancia
    medida (slant) incluye el componente vertical. La distancia real en
    el plano del suelo se obtiene con Pitágoras:

        d_ground = sqrt(d_slant² - h²)

    Si d_slant < h (objeto más cerca que la altura), se usa la proyección
    por pitch como fallback.

    Parámetros
    ----------
    slant_distance_m : float
        Distancia directa (línea de visión) cámara→objeto.
    camera_height_m : float
        Altura de la cámara sobre el suelo (metros).
    pitch_deg : float
        Ángulo de pitch de la cámara (negativo = mirando hacia abajo).
    """
    if camera_height_m <= 0 or slant_distance_m <= camera_height_m:
        # Sin elevación significativa, usar proyección simple
        return slant_to_ground_distance(slant_distance_m, pitch_deg)

    # Pitágoras: d_ground = sqrt(d_slant² - h²)
    d_ground = math.sqrt(slant_distance_m**2 - camera_height_m**2)
    return d_ground
