"""
Diálogo para ingresar todos los parámetros necesarios para la
geolocalización según el paper:
  - Posición GPS (lat, lon)
  - Altitud sobre el nivel del mar (m)
  - Altura de la cámara sobre el suelo (m)
  - Orientación: Heading (yaw), Pitch, Roll
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GpsSetup:
    """Resultado del diálogo GPS con todos los parámetros."""
    lat: float
    lon: float
    altitude_m: float       # altitud sobre nivel del mar
    height_m: float         # altura de la cámara sobre el suelo
    yaw_deg: float          # heading: 0=Norte, 90=Este
    pitch_deg: float        # inclinación: 0=horizontal, negativo=mirando abajo
    roll_deg: float         # alabeo: 0=nivelado


def show_gps_input_dialog(
    default_lat: float = -12.02964,
    default_lon: float = -77.08645,
    default_altitude: float = 62.7888,
    default_height: float = 1.0,
    default_yaw: float = 0.0,
    default_pitch: float = 0.0,
    default_roll: float = 0.0,
) -> Optional[GpsSetup]:
    """
    Muestra una ventana tkinter para ingresar todos los parámetros
    de posición y orientación de la cámara.

    Retorna GpsSetup o None si se cancela.
    """
    try:
        import tkinter as tk
        from tkinter import ttk

        result: list = [None]

        root = tk.Tk()
        root.title("SAR Geotag - Configuración de Cámara")
        root.geometry("520x520")
        root.resizable(False, False)

        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - 260
        y = (root.winfo_screenheight() // 2) - 260
        root.geometry(f"+{x}+{y}")

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Configuración de Posición y Orientación",
                  font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 3))
        ttk.Label(frame, text="Datos necesarios para la geolocalización (paper).\n"
                  "Usa Google Maps para lat/lon. Usa brújula para heading.",
                  font=("Segoe UI", 9), foreground="#555").pack(anchor=tk.W, pady=(0, 12))

        fields = ttk.Frame(frame)
        fields.pack(fill=tk.X)

        # ── Sección: Posición GPS ──
        ttk.Label(fields, text="── Posición GPS ──",
                  font=("Segoe UI", 9, "bold"), foreground="#2980b9"
                  ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))

        row = 1
        def add_field(r, label_text, default_val, hint=""):
            ttk.Label(fields, text=label_text).grid(row=r, column=0, sticky=tk.W, pady=3)
            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(fields, textvariable=var, width=20)
            entry.grid(row=r, column=1, padx=(10, 5), pady=3)
            if hint:
                ttk.Label(fields, text=hint, font=("Segoe UI", 8), foreground="#888"
                          ).grid(row=r, column=2, sticky=tk.W, pady=3)
            return var, entry

        lat_var, lat_entry = add_field(row, "Latitud:", default_lat, "-90 a 90")
        row += 1
        lon_var, _ = add_field(row, "Longitud:", default_lon, "-180 a 180")
        row += 1
        alt_var, _ = add_field(row, "Altitud (m):", default_altitude, "sobre nivel del mar")
        row += 1
        height_var, _ = add_field(row, "Altura cámara (m):", default_height, "sobre el suelo")

        # ── Sección: Orientación ──
        row += 1
        ttk.Label(fields, text="── Orientación ──",
                  font=("Segoe UI", 9, "bold"), foreground="#27ae60"
                  ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(12, 4))

        row += 1
        yaw_var, _ = add_field(row, "Heading / Yaw (°):", default_yaw, "0=N, 90=E, 180=S")
        row += 1
        pitch_var, _ = add_field(row, "Pitch (°):", default_pitch, "0=horiz, -=abajo")
        row += 1
        roll_var, _ = add_field(row, "Roll (°):", default_roll, "0=nivelado")

        error_label = ttk.Label(frame, text="", foreground="red")
        error_label.pack(pady=(8, 0))

        def on_ok():
            try:
                lat = float(lat_var.get())
                lon = float(lon_var.get())
                alt = float(alt_var.get())
                height = float(height_var.get())
                yaw = float(yaw_var.get())
                pitch = float(pitch_var.get())
                roll = float(roll_var.get())

                if not (-90 <= lat <= 90):
                    error_label.config(text="Latitud debe estar entre -90 y 90")
                    return
                if not (-180 <= lon <= 180):
                    error_label.config(text="Longitud debe estar entre -180 y 180")
                    return
                if height < 0:
                    error_label.config(text="Altura de cámara no puede ser negativa")
                    return

                result[0] = GpsSetup(
                    lat=lat, lon=lon,
                    altitude_m=alt, height_m=height,
                    yaw_deg=yaw % 360.0,
                    pitch_deg=pitch, roll_deg=roll,
                )
                root.destroy()
            except ValueError:
                error_label.config(text="Ingresa valores numéricos válidos")

        def on_cancel():
            root.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="✓ Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✗ Cancelar", command=on_cancel).pack(side=tk.LEFT, padx=5)

        lat_entry.focus()
        root.mainloop()
        return result[0]

    except ImportError:
        # Fallback: consola
        print("\n=== Configuración de Cámara ===")
        try:
            lat = float(input(f"  Latitud [{default_lat}]: ") or default_lat)
            lon = float(input(f"  Longitud [{default_lon}]: ") or default_lon)
            alt = float(input(f"  Altitud m.s.n.m [{default_altitude}]: ") or default_altitude)
            height = float(input(f"  Altura cámara sobre suelo [{default_height}]: ") or default_height)
            yaw = float(input(f"  Heading [{default_yaw}]: ") or default_yaw)
            pitch = float(input(f"  Pitch [{default_pitch}]: ") or default_pitch)
            roll = float(input(f"  Roll [{default_roll}]: ") or default_roll)
            return GpsSetup(lat=lat, lon=lon, altitude_m=alt, height_m=height,
                            yaw_deg=yaw % 360.0, pitch_deg=pitch, roll_deg=roll)
        except (ValueError, EOFError):
            print("[WARN] Valores inválidos. Usando defaults.")
            return GpsSetup(lat=default_lat, lon=default_lon,
                            altitude_m=default_altitude, height_m=default_height,
                            yaw_deg=default_yaw, pitch_deg=default_pitch,
                            roll_deg=default_roll)
