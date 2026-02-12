from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import yaml

from src.types import CameraModel, Detection, Estimate, GeoPoint, Pose
from src.vision.ultralytics_detector import UltralyticsDetector
from src.estimators.bbox_size import BBoxSizeEstimator
from src.logging.run_logger import RunLogger
from src.viz.ui import show_camera_selector, enumerate_cameras, show_camera_switcher, show_help_dialog
from src.viz.gps_dialog import show_gps_input_dialog, GpsSetup
from src.viz.overlay import MenuBar, draw_data_bar, draw_capture_flash, draw_camera_switch_notification
from src.geo import geolocate_detections
from src.geo.map_viewer import show_map

WINDOW_NAME = "SAR Geotag - Live"


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_capture(frame: np.ndarray, out_dir: str) -> str:
    """Guarda el frame actual como imagen PNG y retorna la ruta."""
    cap_dir = Path(out_dir) / "captures"
    cap_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = cap_dir / f"capture_{timestamp}.png"
    cv2.imwrite(str(path), frame)
    return str(path)


def _switch_camera(
    cap: cv2.VideoCapture,
    new_index: int,
    desired_w: int,
    desired_h: int,
) -> cv2.VideoCapture:
    """Libera la cámara actual y abre una nueva."""
    cap.release()
    new_cap = cv2.VideoCapture(new_index)
    if desired_w > 0:
        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
    if desired_h > 0:
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)
    return new_cap


def _do_capture(
    frame: np.ndarray,
    current_person_dets: List[Detection],
    current_person_ests: List[Estimate],
    cam: CameraModel,
    pose: Pose,
    camera_lat: float,
    camera_lon: float,
    camera_altitude: float,
    camera_yaw: float,
    camera_pitch: float,
    out_dir: str,
) -> None:
    """Ejecuta la captura de foto, geolocalización y apertura de mapa."""
    if not current_person_dets:
        print("[CAPTURA] No hay personas detectadas en este frame.")
        return
    if camera_lat == 0.0 and camera_lon == 0.0:
        print("[CAPTURA] Advertencia: GPS en (0,0). Configura tu ubicación.")
        return

    # Guardar imagen
    image_path = _save_capture(frame, out_dir)
    print(f"[CAPTURA] Foto guardada: {image_path}")

    # Efecto flash
    flash_frame = draw_capture_flash(frame)
    cv2.imshow(WINDOW_NAME, flash_frame)
    cv2.waitKey(200)

    # Geolocalizar cada persona detectada
    geo_points = geolocate_detections(
        detections=current_person_dets,
        estimates=current_person_ests,
        cam=cam,
        pose=pose,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        camera_yaw_deg=camera_yaw,
    )

    if geo_points:
        print(f"[GEO] {len(geo_points)} persona(s) geolocalizada(s):")
        for i, gp in enumerate(geo_points):
            print(f"  #{i+1}: lat={gp.lat:.7f}, lon={gp.lon:.7f}, "
                  f"dist_suelo={gp.distance_m:.1f}m, "
                  f"dist_slant={gp.slant_distance_m:.1f}m, "
                  f"bearing={gp.bearing_deg:.1f}°")

        show_map(
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            geo_points=geo_points,
            camera_altitude_m=camera_altitude,
            camera_yaw_deg=camera_yaw,
            camera_pitch_deg=camera_pitch,
            image_path=image_path,
            output_dir=out_dir,
        )
    else:
        print("[GEO] No se pudieron geolocalizar las detecciones.")


def run(config_path: str = "configs/default.yaml") -> None:
    cfg = _load_config(config_path)

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

    # --- gps defaults ---
    gps_cfg = cfg.get("gps", {})
    default_lat = float(gps_cfg.get("lat", 0.0))
    default_lon = float(gps_cfg.get("lon", 0.0))
    default_altitude = float(gps_cfg.get("altitude_m", 0.0))
    default_height = float(gps_cfg.get("height_m", 1.5))
    default_yaw = float(gps_cfg.get("yaw_deg", 0.0))
    default_pitch = float(gps_cfg.get("pitch_deg", 0.0))
    default_roll = float(gps_cfg.get("roll_deg", 0.0))

    # --- logging ---
    log_cfg = cfg.get("logging", {})
    log_enabled = bool(log_cfg.get("enabled", True))
    out_dir = str(log_cfg.get("out_dir", "runs"))

    # =============================================
    # PASO 1: Seleccionar cámara
    # =============================================
    print("=== SAR Geotag Prototype ===")
    print("Buscando dispositivos de cámara...")
    available_cameras = enumerate_cameras()
    webcam_index = show_camera_selector()
    print(f"Cámara seleccionada: index {webcam_index}")

    # =============================================
    # PASO 2: Posición GPS + orientación + altitud
    # =============================================
    print("Ingresando parámetros de posición y orientación...")
    gps_result: Optional[GpsSetup] = show_gps_input_dialog(
        default_lat=default_lat,
        default_lon=default_lon,
        default_altitude=default_altitude,
        default_height=default_height,
        default_yaw=default_yaw,
        default_pitch=default_pitch,
        default_roll=default_roll,
    )

    if gps_result is None:
        print("[INFO] Diálogo cancelado. Usando valores por defecto.")
        gps_result = GpsSetup(
            lat=default_lat, lon=default_lon,
            altitude_m=default_altitude, height_m=default_height,
            yaw_deg=default_yaw, pitch_deg=default_pitch, roll_deg=default_roll,
        )

    camera_lat = gps_result.lat
    camera_lon = gps_result.lon
    camera_altitude = gps_result.altitude_m
    camera_height = gps_result.height_m
    camera_yaw = gps_result.yaw_deg
    camera_pitch = gps_result.pitch_deg
    camera_roll = gps_result.roll_deg

    pose = Pose(
        height_m=camera_height,
        altitude_m=camera_altitude,
        yaw_deg=camera_yaw,
        pitch_deg=camera_pitch,
        roll_deg=camera_roll,
    )

    print(f"GPS: lat={camera_lat:.6f}, lon={camera_lon:.6f}")
    print(f"Altitud: {camera_altitude:.0f}m  |  Altura cámara: {camera_height:.1f}m")
    print(f"Orientación: yaw={camera_yaw:.0f}° pitch={camera_pitch:.1f}° roll={camera_roll:.1f}°")

    # =============================================
    # PASO 3: Iniciar componentes
    # =============================================
    detector = UltralyticsDetector(model_name=model_path, conf=conf_thres)

    if method != "bbox_size":
        raise ValueError(f"Por ahora solo method=bbox_size. Recibí: {method}")

    estimator = BBoxSizeEstimator(person_height_m=person_height_m)
    logger = RunLogger(out_dir=out_dir) if log_enabled else None

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la webcam (index={webcam_index}).")

    if desired_w > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
    if desired_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

    if logger:
        print(f"Logging en: {logger.csv_path}")

    # ── Barra de menú + mouse callback ──
    menu_bar = MenuBar()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, menu_bar.on_mouse)

    # Label de la cámara activa
    active_cam_label = f"Cam {webcam_index}"
    for ci, cl in available_cameras:
        if ci == webcam_index:
            active_cam_label = cl
            break

    print("─" * 55)
    print("  Menú Cámara  = Cambiar dispositivo (con Actualizar)")
    print("  Menú Captura / [C] = Foto + geolocalizar + mapa")
    print("  [Q] = Salir")
    print("─" * 55)

    frame_idx = 0
    current_person_dets: List[Detection] = []
    current_person_ests: List[Estimate] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_iso = datetime.now().isoformat(timespec="milliseconds")

            h, w = frame.shape[:2]
            cam = CameraModel(width_px=w, height_px=h, hfov_deg=hfov_deg, vfov_deg=vfov_deg)

            dets = detector.detect(frame)

            current_person_dets.clear()
            current_person_ests.clear()

            for det in dets:
                if det.cls_name != "person":
                    continue

                est = estimator.estimate(det, cam)
                current_person_dets.append(det)
                current_person_ests.append(est)

                x1, y1, x2, y2 = det.xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if est.distance_m is not None:
                    txt = f"dist~ {est.distance_m:.1f} m"
                    cv2.putText(
                        frame, txt,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                if logger:
                    logger.log(timestamp_iso, frame_idx, det, est, cam)

            # ── Dibujar menú + datos ──
            menu_bar.draw(frame)
            draw_data_bar(
                frame,
                camera_lat=camera_lat,
                camera_lon=camera_lon,
                camera_yaw=camera_yaw,
                camera_altitude=camera_altitude,
                camera_height=camera_height,
                camera_pitch=camera_pitch,
                camera_roll=camera_roll,
                active_cam_label=active_cam_label,
                person_count=len(current_person_dets),
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Procesar clics del menú ──
            action = menu_bar.consume_click()

            if action == "camera":
                # Abrir diálogo de cambio de cámara (con Actualizar)
                switch_result = show_camera_switcher(current_index=webcam_index)
                if switch_result is not None:
                    new_index, new_label, updated_cams = switch_result
                    if new_index != webcam_index:
                        print(f"[CAM] Cambiando a: {new_label}")
                        cap = _switch_camera(cap, new_index, desired_w, desired_h)
                        webcam_index = new_index
                        active_cam_label = new_label
                        available_cameras = updated_cams
                        if cap.isOpened():
                            ok2, temp_frame = cap.read()
                            if ok2:
                                draw_camera_switch_notification(temp_frame, new_label)
                                cv2.imshow(WINDOW_NAME, temp_frame)
                                cv2.waitKey(600)
                        else:
                            print(f"[WARN] No se pudo abrir cámara {new_index}")
                    else:
                        # Actualizar la lista aunque no cambie
                        available_cameras = updated_cams

            elif action == "capture":
                _do_capture(
                    frame, current_person_dets, current_person_ests,
                    cam, pose, camera_lat, camera_lon,
                    camera_altitude, camera_yaw, camera_pitch, out_dir,
                )

            elif action == "help":
                show_help_dialog()

            # ── Atajos de teclado ──
            if key == ord("c") or key == ord("C"):
                _do_capture(
                    frame, current_person_dets, current_person_ests,
                    cam, pose, camera_lat, camera_lon,
                    camera_altitude, camera_yaw, camera_pitch, out_dir,
                )

            elif key == ord("q") or key == ord("Q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logger:
            logger.close()
