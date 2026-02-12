"""
Funciones de overlay para dibujar información en el frame de video.

Implementa una barra de menú estilo VLC en la parte superior del frame,
clickeable con el mouse, y una barra de datos GPS/orientación debajo.

Estructura visual:
  ┌──────────────────────────────────────────────────┐
  │  Cámara   Captura   Ver   Ayuda                  │  ← menú (click)
  ├──────────────────────────────────────────────────┤
  │  GPS: ...  |  Alt: ...  |  Yaw: ...  |  Cam: .. │  ← datos
  ├──────────────────────────────────────────────────┤
  │                                                  │
  │               (video feed)                       │
  │                                                  │
  └──────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ───────────────────── Constantes de layout ─────────────────────
MENU_BAR_H = 26          # altura de la barra de menú
DATA_BAR_H = 40          # altura de la barra de datos
MENU_BG = (240, 240, 240)  # gris claro (como Windows)
MENU_TEXT = (30, 30, 30)    # texto oscuro
MENU_HOVER = (200, 215, 255)  # fondo hover (azul claro)
MENU_FONT = cv2.FONT_HERSHEY_SIMPLEX
MENU_FONT_SCALE = 0.45
MENU_FONT_THICK = 1


@dataclass
class MenuButton:
    """Representa un botón en la barra de menú."""
    label: str
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    action_key: str = ""     # identificador de la acción


class MenuBar:
    """
    Barra de menú clickeable dibujada sobre el frame de OpenCV.
    Detecta clics del mouse para disparar acciones.
    """

    def __init__(self):
        self.buttons: List[MenuButton] = []
        self.hovered_idx: int = -1
        self.last_click: Optional[str] = None  # action_key del último clic
        self._setup_buttons()

    def _setup_buttons(self):
        items = [
            ("Camara", "camera"),
            ("Captura", "capture"),
            ("Ayuda", "help"),
        ]
        x = 5
        for label, key in items:
            ts = cv2.getTextSize(f" {label} ", MENU_FONT, MENU_FONT_SCALE, MENU_FONT_THICK)[0]
            btn_w = ts[0] + 16
            self.buttons.append(MenuButton(
                label=label,
                x1=x, y1=2,
                x2=x + btn_w, y2=MENU_BAR_H - 2,
                action_key=key,
            ))
            x += btn_w + 2

    def draw(self, frame: np.ndarray) -> None:
        """Dibuja la barra de menú en la parte superior del frame."""
        h, w = frame.shape[:2]

        # Fondo de la barra
        cv2.rectangle(frame, (0, 0), (w, MENU_BAR_H), MENU_BG, -1)
        # Línea divisoria
        cv2.line(frame, (0, MENU_BAR_H), (w, MENU_BAR_H), (180, 180, 180), 1)

        for i, btn in enumerate(self.buttons):
            # Hover effect
            if i == self.hovered_idx:
                cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2),
                              MENU_HOVER, -1)
                cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2),
                              (150, 170, 220), 1)

            # Texto
            text_x = btn.x1 + 8
            text_y = btn.y1 + 17
            cv2.putText(frame, btn.label, (text_x, text_y),
                        MENU_FONT, MENU_FONT_SCALE, MENU_TEXT, MENU_FONT_THICK)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Callback de mouse para OpenCV."""
        # Hover
        self.hovered_idx = -1
        if y <= MENU_BAR_H:
            for i, btn in enumerate(self.buttons):
                if btn.x1 <= x <= btn.x2 and btn.y1 <= y <= btn.y2:
                    self.hovered_idx = i
                    break

        # Click
        if event == cv2.EVENT_LBUTTONDOWN and y <= MENU_BAR_H:
            for btn in self.buttons:
                if btn.x1 <= x <= btn.x2 and btn.y1 <= y <= btn.y2:
                    self.last_click = btn.action_key
                    break

    def consume_click(self) -> Optional[str]:
        """Retorna y limpia la última acción clickeada."""
        action = self.last_click
        self.last_click = None
        return action


def draw_data_bar(
    frame: np.ndarray,
    camera_lat: float,
    camera_lon: float,
    camera_yaw: float,
    camera_altitude: float = 0.0,
    camera_height: float = 1.5,
    camera_pitch: float = 0.0,
    camera_roll: float = 0.0,
    active_cam_label: str = "Cam 0",
    person_count: int = 0,
) -> None:
    """
    Dibuja la barra de datos (GPS, orientación, cámara activa)
    justo debajo de la barra de menú.
    """
    h, w = frame.shape[:2]
    y_top = MENU_BAR_H
    y_bot = MENU_BAR_H + DATA_BAR_H

    # Fondo semitransparente oscuro
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top), (w, y_bot), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Línea 1: GPS + altitud
    line1 = (f"GPS: {camera_lat:.6f}, {camera_lon:.6f}  |  "
             f"Alt: {camera_altitude:.0f}m  |  "
             f"Altura: {camera_height:.1f}m")
    cv2.putText(frame, line1, (10, y_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

    # Línea 2: orientación + cámara
    line2 = (f"Yaw: {camera_yaw:.0f}\u00b0  |  "
             f"Pitch: {camera_pitch:.1f}\u00b0  |  "
             f"Roll: {camera_roll:.1f}\u00b0  |  "
             f"{active_cam_label}")
    cv2.putText(frame, line2, (10, y_top + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)

    # Badge de personas detectadas
    if person_count > 0:
        count_text = f"{person_count} persona(s)"
        ts = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        bx = w - ts[0] - 20
        by = y_top + 8
        cv2.rectangle(frame, (bx - 5, by - 2), (w - 5, by + ts[1] + 8), (0, 0, 180), -1)
        cv2.putText(frame, count_text, (bx, by + ts[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Indicador de controles en la parte inferior
    bot_h = 24
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bot_h), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "[C] Capturar    [Q] Salir    |  Menu: click arriba",
                (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1)


def draw_capture_flash(frame: np.ndarray) -> np.ndarray:
    """Efecto visual de 'flash' al tomar una foto."""
    flash = np.full_like(frame, 255)
    blended = cv2.addWeighted(frame, 0.3, flash, 0.7, 0)
    return blended


def draw_camera_switch_notification(frame: np.ndarray, cam_label: str) -> None:
    """Muestra un aviso temporal de que se cambió de cámara."""
    h, w = frame.shape[:2]
    text = f">> Cambiando a: {cam_label} <<"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cx = (w - ts[0]) // 2
    cy = h // 2

    cv2.rectangle(frame, (cx - 15, cy - 30), (cx + ts[0] + 15, cy + 10),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
