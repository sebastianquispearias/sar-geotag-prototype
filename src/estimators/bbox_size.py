from __future__ import annotations
import math

from src.types import CameraModel, Detection, Estimate


class BBoxSizeEstimator:
    def __init__(self, person_height_m: float = 1.70):
        self.person_height_m = person_height_m

    def estimate(self, det: Detection, cam: CameraModel) -> Estimate:
        h_px = det.height_px
        if h_px <= 0:
            return Estimate(method="bbox_size", distance_m=None, confidence=0.0)

        vfov = math.radians(cam.vfov_deg)
        fy = (cam.height_px / 2.0) / math.tan(vfov / 2.0)
        dist_m = (fy * self.person_height_m) / float(h_px)

        return Estimate(method="bbox_size", distance_m=float(dist_m), confidence=1.0)
