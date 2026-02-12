from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Detection:
    """Una detección en la imagen."""
    cls_name: str                  # e.g. "person"
    conf: float                    # 0..1
    xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) en pixeles


    @property
    def width_px(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1)

    @property
    def height_px(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, y2 - y1)

    @property
    def center_x(self) -> int:
        """Centro horizontal de la bbox."""
        x1, _, x2, _ = self.xyxy
        return int((x1 + x2) / 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Punto típico para B (pies): centro abajo de la bbox."""
        x1, y1, x2, y2 = self.xyxy
        u = int((x1 + x2) / 2)
        v = int(y2)
        return (u, v)


@dataclass(frozen=True)
class CameraModel:
    """Parámetros necesarios para convertir pixeles <-> rayos."""
    width_px: int
    height_px: int
    hfov_deg: float
    vfov_deg: float


@dataclass(frozen=True)
class Pose:
    """
    Pose de la cámara/dron.
    En prototipo: valores simulados.
    En dron real: vienen de telemetría.
    """
    height_m: float          # altura sobre el suelo (metros)
    altitude_m: float = 0.0  # altitud sobre el nivel del mar (metros)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass(frozen=True)
class Estimate:
    """Resultado del estimador (A o B)."""
    method: str                    # "bbox_size" o "ray_plane"
    distance_m: Optional[float] = None   # Alternativa A
    ground_xy_m: Optional[Tuple[float, float]] = None  # Alternativa B (x,y local)
    confidence: Optional[float] = None


@dataclass(frozen=True)
class GeoPoint:
    """Coordenada geográfica de un objeto detectado."""
    lat: float
    lon: float
    distance_m: float          # distancia horizontal en el suelo
    bearing_deg: float         # azimut desde la cámara (0=N, 90=E)
    slant_distance_m: float = 0.0   # distancia directa (línea de visión)
    camera_altitude_m: float = 0.0  # altitud de la cámara al capturar
    confidence: Optional[float] = None
    label: str = "person"


@dataclass(frozen=True)
class CaptureResult:
    """Resultado de una captura (foto) con todas las geolocalizaciones."""
    timestamp_iso: str
    camera_lat: float
    camera_lon: float
    camera_altitude_m: float
    camera_yaw_deg: float
    camera_pitch_deg: float
    image_path: str
    geo_points: List[GeoPoint] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.geo_points is None:
            object.__setattr__(self, "geo_points", [])
    