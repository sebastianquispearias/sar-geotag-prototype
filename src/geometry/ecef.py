"""
Transformaciones entre WGS84 (lat/lon/alt) y ECEF (Earth-Centered Earth-Fixed).

Implementación basada en el paper "Object Geolocalization Using Consumer-Grade Devices"
Sección 3: Reference Frames and Coordinate Systems

Referencias:
- Koks [4]: Using Rotations to Build Aerospace Coordinate Systems
- Osen [17]: Accurate Conversion of Earth-Fixed Earth-Centered Coordinates
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Constantes WGS84
# ══════════════════════════════════════════════════════════════════════════════

WGS84_A = 6_378_137.0  # Semi-eje mayor (metros)
WGS84_B = 6_356_752.314245  # Semi-eje menor (metros)
WGS84_F = 1 / 298.257223563  # Aplanamiento
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Excentricidad al cuadrado


@dataclass
class ECEFPoint:
    """Punto en coordenadas ECEF (metros)."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ECEFPoint":
        return cls(x=arr[0], y=arr[1], z=arr[2])


@dataclass
class WGS84Point:
    """Punto en coordenadas geodésicas WGS84."""
    lat_deg: float  # Latitud (grados)
    lon_deg: float  # Longitud (grados)
    alt_m: float  # Altitud sobre elipsoide HAE (metros)


# ══════════════════════════════════════════════════════════════════════════════
# Conversiones WGS84 <-> ECEF
# ══════════════════════════════════════════════════════════════════════════════

def wgs84_to_ecef(point: WGS84Point) -> ECEFPoint:
    """
    Convierte coordenadas geodésicas (lat, lon, alt) a ECEF (x, y, z).
    
    Basado en la sección 3 del paper y las fórmulas estándar de Koks [4].
    """
    lat = math.radians(point.lat_deg)
    lon = math.radians(point.lon_deg)
    h = point.alt_m

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Radio de curvatura en el primer vertical
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + h) * sin_lat

    return ECEFPoint(x=x, y=y, z=z)


def ecef_to_wgs84(point: ECEFPoint) -> WGS84Point:
    """
    Convierte coordenadas ECEF (x, y, z) a geodésicas (lat, lon, alt).
    
    Usa el método iterativo de Osen [17] para precisión numérica.
    """
    x, y, z = point.x, point.y, point.z

    # Longitud (directa)
    lon = math.atan2(y, x)

    # Latitud y altitud (iterativo)
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - WGS84_E2))  # Aproximación inicial

    for _ in range(10):  # Iteraciones para convergencia
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)
        lat_new = math.atan2(z + WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) / abs(sin_lat) - N * (1 - WGS84_E2)

    return WGS84Point(
        lat_deg=math.degrees(lat),
        lon_deg=math.degrees(lon),
        alt_m=alt,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Base NED (North-East-Down) en ECEF
# ══════════════════════════════════════════════════════════════════════════════

def compute_ned_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la base NED {n, e, d} en coordenadas ECEF para una posición dada.
    
    Del paper, Sección 3:
    "A NED basis {n, e, d} is defined by three ECEF vectors, representing 
    north, east, and down... Intuitively, vectors n and e span a plane, 
    tangential to the ellipsoid, at a particular latitude and longitude."
    
    Proceso (del paper):
    1. Definir NED en λ=φ=0: n=[0,0,1]^T (paralelo a Z), e=[0,1,0]^T (paralelo a Y)
    2. Rotar e alrededor de n por λ grados (longitud)
    3. Rotar n alrededor del nuevo e por -φ grados (latitud)
    4. d = n × e (cross product)
    
    Retorna
    -------
    (n, e, d) : Tuple de 3 vectores unitarios numpy en ECEF
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Vector Norte (tangente al meridiano, apuntando al polo norte)
    n = np.array([
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
    ])

    # Vector Este (tangente al paralelo, apuntando al este)
    e = np.array([
        -sin_lon,
        cos_lon,
        0.0,
    ])

    # Vector Abajo (apuntando al centro de la Tierra)
    d = np.array([
        -cos_lat * cos_lon,
        -cos_lat * sin_lon,
        -sin_lat,
    ])

    return n, e, d


def ned_to_ecef_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Matriz de rotación para transformar vectores de NED local a ECEF global.
    
    Las columnas son los vectores de la base NED expresados en ECEF.
    """
    n, e, d = compute_ned_basis(lat_deg, lon_deg)
    # Cada columna es un vector de la base NED
    return np.column_stack([n, e, d])


def ecef_to_ned_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Matriz de rotación para transformar vectores de ECEF global a NED local.
    
    Es la transpuesta (inversa) de ned_to_ecef_matrix.
    """
    return ned_to_ecef_matrix(lat_deg, lon_deg).T


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def distance_ecef(p1: ECEFPoint, p2: ECEFPoint) -> float:
    """Distancia euclidiana entre dos puntos ECEF (metros)."""
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2
    )
