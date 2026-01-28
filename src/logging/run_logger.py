from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.types import CameraModel, Detection, Estimate


class RunLogger:
    """
    Logger simple a CSV por ejecución.
    Crea un archivo en runs/run_YYYYmmdd_HHMMSS.csv
    """

    def __init__(self, out_dir: str = "runs", run_name: Optional[str] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.csv_path = self.out_dir / f"{run_name}.csv"
        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "timestamp_iso",
                "frame_idx",
                "cls_name",
                "conf",
                "x1",
                "y1",
                "x2",
                "y2",
                "bbox_h_px",
                "bbox_w_px",
                "u_bottom",
                "v_bottom",
                "method",
                "distance_m",
                "confidence_est",
                "cam_width_px",
                "cam_height_px",
                "cam_hfov_deg",
                "cam_vfov_deg",
            ],
        )
        self._writer.writeheader()
        self._fh.flush()

    def log(self, timestamp_iso: str, frame_idx: int, det: Detection, est: Estimate, cam: CameraModel) -> None:
        x1, y1, x2, y2 = det.xyxy
        u, v = det.bottom_center

        row = {
            "timestamp_iso": timestamp_iso,
            "frame_idx": frame_idx,
            "cls_name": det.cls_name,
            "conf": det.conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bbox_h_px": det.height_px,
            "bbox_w_px": det.width_px,
            "u_bottom": u,
            "v_bottom": v,
            "method": est.method,
            "distance_m": "" if est.distance_m is None else float(est.distance_m),
            "confidence_est": "" if est.confidence is None else float(est.confidence),
            "cam_width_px": cam.width_px,
            "cam_height_px": cam.height_px,
            "cam_hfov_deg": cam.hfov_deg,
            "cam_vfov_deg": cam.vfov_deg,
        }

        self._writer.writerow(row)

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()
