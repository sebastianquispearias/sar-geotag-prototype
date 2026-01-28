from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


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
    height_m: float
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
    