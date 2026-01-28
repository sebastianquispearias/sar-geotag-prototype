import cv2
from datetime import datetime

from src.types import CameraModel
from src.vision.ultralytics_detector import UltralyticsDetector
from src.estimators.bbox_size import BBoxSizeEstimator
from src.logging.run_logger import RunLogger


def main():
    detector = UltralyticsDetector("yolo11n.pt", conf=0.25)
    estimator = BBoxSizeEstimator(person_height_m=1.70)
    logger = RunLogger(out_dir="runs")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam. Prueba con índice 1 o 2.")

    print(f"Logging en: {logger.csv_path}")
    print("Presiona 'q' para salir.")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_iso = datetime.now().isoformat(timespec="milliseconds")

            h, w = frame.shape[:2]
            cam = CameraModel(width_px=w, height_px=h, hfov_deg=78.0, vfov_deg=44.0)

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

                # Log por detección
                logger.log(timestamp_iso, frame_idx, det, est, cam)

            cv2.imshow("proto A: bbox->distance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()


if __name__ == "__main__":
    main()
