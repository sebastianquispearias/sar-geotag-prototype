from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import yaml

from src.types import CameraModel
from src.vision.ultralytics_detector import UltralyticsDetector
from src.estimators.bbox_size import BBoxSizeEstimator
from src.logging.run_logger import RunLogger


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str = "configs/default.yaml") -> None:
    cfg = _load_config(config_path)

    # --- source ---
    source_cfg = cfg.get("source", {})
    webcam_index = int(source_cfg.get("webcam_index", 0))

    # --- detector ---
    det_cfg = cfg.get("detector", {})
    model_path = det_cfg.get("model_path", "") or "yolo11n.pt"
    conf_thres = float(det_cfg.get("conf_thres", 0.25))

    # --- camera ---
    cam_cfg = cfg.get("camera", {})
    desired_w = int(cam_cfg.get("width", 0) or 0)
    desired_h = int(cam_cfg.get("height", 0) or 0)
    hfov_deg = float(cam_cfg.get("hfov_deg", 78.0))
    vfov_deg = float(cam_cfg.get("vfov_deg", 44.0))

    # --- estimation ---
    est_cfg = cfg.get("estimation", {})
    method = est_cfg.get("method", "bbox_size")
    person_height_m = float(est_cfg.get("person_height_m", 1.70))

    # --- logging ---
    log_cfg = cfg.get("logging", {})
    log_enabled = bool(log_cfg.get("enabled", True))
    out_dir = str(log_cfg.get("out_dir", "runs"))

    # Create components
    detector = UltralyticsDetector(model_name=model_path, conf=conf_thres)

    if method != "bbox_size":
        raise ValueError(f"Por ahora este main solo implementa method=bbox_size. Recibí: {method}")

    estimator = BBoxSizeEstimator(person_height_m=person_height_m)
    logger = RunLogger(out_dir=out_dir) if log_enabled else None

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la webcam (index={webcam_index}). Prueba 0/1/2.")

    # Intentar setear resolución (si está en config). Si la cámara no lo soporta, igual funciona.
    if desired_w > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
    if desired_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

    if logger:
        print(f"Logging en: {logger.csv_path}")
    print("Presiona 'q' para salir.")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_iso = datetime.now().isoformat(timespec="milliseconds")

            # Usamos la resolución REAL del frame (no asumimos 720p)
            h, w = frame.shape[:2]
            cam = CameraModel(width_px=w, height_px=h, hfov_deg=hfov_deg, vfov_deg=vfov_deg)

            dets = detector.detect(frame)

            for det in dets:
                if det.cls_name != "person":
                    continue

                est = estimator.estimate(det, cam)

                x1, y1, x2, y2 = det.xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if est.distance_m is not None:
                    txt = f"dist~ {est.distance_m:.1f} m"
                    cv2.putText(
                        frame,
                        txt,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                if logger:
                    logger.log(timestamp_iso, frame_idx, det, est, cam)

            cv2.imshow("proto A: bbox->distance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logger:
            logger.close()
