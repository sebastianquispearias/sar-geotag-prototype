"""
Modelo de cámara pinhole y cálculo de vectores line-of-sight.

Implementación basada en el paper "Object Geolocalization Using Consumer-Grade Devices"
Sección 3: Reference Frames and Coordinate Systems (Figure 2b)
Sección 4: Proposed Geolocalization Algorithms

Del paper:
"A simple pinhole camera model represents the viewport of a device and has a local 
camera reference frame, where the optical axis is Z_img, axis X_img is along the 
width of the viewport, and Y_img is in downward direction of the viewport height."

"Selecting a point/pixel at image coordinates (ix, iy) is considered a geolocalization 
measurement and translated to a line-of-sight unit vector based on hfov and vfov."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.geometry.ecef import (
    ECEFPoint,
    WGS84Point,
    wgs84_to_ecef,
    ned_to_ecef_matrix,
)


@dataclass
class Measurement:
    """
    Una medición de geolocalización según el paper.
    
    Del paper, Sección 4:
    "A measurement is a position vector (p, v), where p is the origin 
    (location of the device) and v is a unit vector in the direction 
    of an object, both in the ECEF coordinate system."
    """
    origin: ECEFPoint  # p: posición del dispositivo en ECEF
    direction: np.ndarray  # v: vector unitario hacia el objeto en ECEF

    def __repr__(self) -> str:
        return (
            f"Measurement(origin=({self.origin.x:.1f}, {self.origin.y:.1f}, {self.origin.z:.1f}), "
            f"dir=({self.direction[0]:.4f}, {self.direction[1]:.4f}, {self.direction[2]:.4f}))"
        )


def pixel_to_line_of_sight_ned(
    pixel_x: int,
    pixel_y: int,
    image_width: int,
    image_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> np.ndarray:
    """
    Convierte coordenadas de pixel a vector line-of-sight en el frame de cámara NED.
    
    Del paper (Figure 2b):
    - Z_img es el eje óptico (hacia adelante)
    - X_img es horizontal (derecha positiva)
    - Y_img es vertical hacia abajo (compatible con NED)
    
    El vector retornado apunta desde la cámara hacia el punto seleccionado.
    """
    # Centro de la imagen
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Focal lengths (modelo pinhole)
    fx = cx / math.tan(math.radians(hfov_deg / 2.0))
    fy = cy / math.tan(math.radians(vfov_deg / 2.0))

    # Offset desde el centro
    dx = pixel_x - cx
    dy = pixel_y - cy

    # Vector en el frame de la cámara (Z hacia adelante)
    # [dx/fx, dy/fy, 1] y luego normalizar
    vec = np.array([dx / fx, dy / fy, 1.0])
    vec_normalized = vec / np.linalg.norm(vec)

    return vec_normalized


def camera_frame_to_ned(
    vec_camera: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """
    Transforma un vector del frame de cámara al frame NED local.
    
    El frame de cámara tiene:
    - X hacia la derecha
    - Y hacia abajo
    - Z hacia adelante (eje óptico)
    
    El frame NED tiene:
    - X hacia el Norte
    - Y hacia el Este
    - Z hacia abajo
    
    Parámetros:
    - yaw_deg: rotación alrededor de Z (0=Norte, 90=Este)
    - pitch_deg: inclinación vertical (0=horizontal, negativo=mirando abajo)
    - roll_deg: rotación alrededor del eje óptico
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # Matrices de rotación
    # Rz (yaw) - rotación alrededor de eje vertical
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1],
    ])

    # Ry (pitch) - rotación alrededor de eje lateral
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ])

    # Rx (roll) - rotación alrededor de eje frontal
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)],
    ])

    # Primero transformar de frame cámara a frame "mirando al norte horizontal"
    # Frame cámara: X=derecha, Y=abajo, Z=adelante
    # Frame NED inicial (yaw=0, pitch=0): X=norte, Y=este, Z=abajo
    # Cuando la cámara mira al norte horizontal:
    #   - adelante (Z_cam) = norte (X_ned)
    #   - derecha (X_cam) = este (Y_ned)
    #   - abajo (Y_cam) = abajo (Z_ned)
    
    # Matriz para pasar de frame cámara a NED (cuando yaw=pitch=roll=0, mirando al norte)
    cam_to_ned_base = np.array([
        [0, 0, 1],  # X_ned (norte) = Z_cam (adelante)
        [1, 0, 0],  # Y_ned (este) = X_cam (derecha)
        [0, 1, 0],  # Z_ned (abajo) = Y_cam (abajo)
    ])

    # Rotación completa: primero orientar la cámara, luego aplicar yaw-pitch-roll
    # R = Rz @ Ry @ cam_to_ned_base (pitch negativo = mirar abajo)
    R = Rz @ Ry @ Rx @ cam_to_ned_base

    return R @ vec_camera


def create_measurement(
    camera_lat: float,
    camera_lon: float,
    camera_alt: float,
    pixel_x: int,
    pixel_y: int,
    image_width: int,
    image_height: int,
    hfov_deg: float,
    vfov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> Measurement:
    """
    Crea una medición completa (origin + direction) en coordenadas ECEF.
    
    Este es el proceso completo del paper:
    1. Convertir pixel a line-of-sight en frame de cámara
    2. Transformar a frame NED local
    3. Transformar a frame ECEF global
    
    Parámetros
    ----------
    camera_lat, camera_lon, camera_alt : float
        Posición GPS del dispositivo (grados, metros sobre elipsoide)
    pixel_x, pixel_y : int
        Coordenadas del punto seleccionado en la imagen
    image_width, image_height : int
        Dimensiones de la imagen
    hfov_deg, vfov_deg : float
        Campo de visión horizontal y vertical (grados)
    yaw_deg, pitch_deg, roll_deg : float
        Orientación de la cámara (grados)
    """
    # 1. Pixel -> Line-of-sight en frame de cámara
    los_camera = pixel_to_line_of_sight_ned(
        pixel_x, pixel_y,
        image_width, image_height,
        hfov_deg, vfov_deg,
    )

    # 2. Frame cámara -> NED local
    los_ned = camera_frame_to_ned(los_camera, yaw_deg, pitch_deg, roll_deg)

    # 3. NED local -> ECEF global
    ned_to_ecef = ned_to_ecef_matrix(camera_lat, camera_lon)
    los_ecef = ned_to_ecef @ los_ned
    los_ecef = los_ecef / np.linalg.norm(los_ecef)  # Re-normalizar por seguridad

    # Origen en ECEF
    origin_wgs84 = WGS84Point(lat_deg=camera_lat, lon_deg=camera_lon, alt_m=camera_alt)
    origin_ecef = wgs84_to_ecef(origin_wgs84)

    return Measurement(origin=origin_ecef, direction=los_ecef)
