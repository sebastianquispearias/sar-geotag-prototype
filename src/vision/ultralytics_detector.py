from __future__ import annotations
from typing import List, Optional

from ultralytics import YOLO

from src.types import Detection


class UltralyticsDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.25):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        r0 = results[0]

        detections: List[Detection] = []
        if r0.boxes is None:
            return detections

        names = r0.names  # dict id->name

        for b in r0.boxes:
            cls_id = int(b.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            det = Detection(
                cls_name=cls_name,
                conf=conf,
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
            detections.append(det)

        return detections
