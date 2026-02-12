"""
UI helpers: selector de cámara y diálogos auxiliares.

Incluye:
  - enumerate_cameras: detectar dispositivos de cámara disponibles
  - show_camera_selector: diálogo inicial de selección
  - show_camera_switcher: diálogo en ejecución con botón "Actualizar"
"""

from __future__ import annotations

import cv2
from typing import List, Optional, Tuple


def enumerate_cameras(max_index: int = 10) -> List[Tuple[int, str]]:
    """
    Enumera las cámaras disponibles probando índices 0..max_index.

    Retorna
    -------
    List[(index, label)]
        Lista de tuplas (índice, descripción).
    """
    cameras: List[Tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            label = f"Camara {i}  ({w}x{h})"
            cameras.append((i, label))
            cap.release()
    return cameras


def show_camera_selector() -> int:
    """
    Muestra una ventana con tkinter para seleccionar la cámara.
    Retorna el índice seleccionado (o 0 si se cancela).
    """
    cameras = enumerate_cameras()
    if not cameras:
        print("[WARN] No se encontraron cámaras. Usando index 0.")
        return 0
    if len(cameras) == 1:
        print(f"[INFO] Solo una cámara encontrada: {cameras[0][1]}")
        return cameras[0][0]

    try:
        import tkinter as tk
        from tkinter import ttk

        selected_index = [cameras[0][0]]

        root = tk.Tk()
        root.title("SAR Geotag - Seleccionar Cámara")
        root.geometry("400x200")
        root.resizable(False, False)

        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - 200
        y = (root.winfo_screenheight() // 2) - 100
        root.geometry(f"+{x}+{y}")

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Dispositivos de cámara detectados:",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        combo_var = tk.StringVar()
        labels = [c[1] for c in cameras]
        combo = ttk.Combobox(frame, textvariable=combo_var, values=labels,
                             state="readonly", width=40)
        combo.current(0)
        combo.pack(pady=5)

        def on_ok():
            idx = combo.current()
            selected_index[0] = cameras[idx][0]
            root.destroy()

        def on_cancel():
            root.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Usar cámara", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=on_cancel).pack(side=tk.LEFT, padx=5)

        root.mainloop()
        return selected_index[0]

    except ImportError:
        print("\n=== Cámaras disponibles ===")
        for idx, label in cameras:
            print(f"  [{idx}] {label}")
        try:
            choice = int(input("Selecciona el índice de la cámara: "))
            if any(c[0] == choice for c in cameras):
                return choice
        except (ValueError, EOFError):
            pass
        print(f"Usando cámara por defecto: {cameras[0][0]}")
        return cameras[0][0]


def show_camera_switcher(
    current_index: int = 0,
) -> Optional[Tuple[int, str, List[Tuple[int, str]]]]:
    """
    Diálogo para cambiar de cámara *durante la ejecución*.

    Muestra:
    - Lista de cámaras disponibles (con la actual marcada)
    - Botón "Actualizar" para re-escanear dispositivos
    - Botón "Cambiar" para usar la cámara seleccionada
    - Botón "Cancelar"

    Retorna
    -------
    (new_index, new_label, updated_cameras) o None si se cancela.
    """
    try:
        import tkinter as tk
        from tkinter import ttk

        result: list = [None]
        cameras_ref: list = [enumerate_cameras()]

        root = tk.Tk()
        root.title("SAR Geotag - Cambiar Cámara")
        root.geometry("480x320")
        root.resizable(False, False)

        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - 240
        y = (root.winfo_screenheight() // 2) - 160
        root.geometry(f"+{x}+{y}")

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Dispositivos de cámara",
                  font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 3))
        ttk.Label(frame,
                  text=f"Cámara actual: Camara {current_index}",
                  font=("Segoe UI", 9), foreground="#555"
                  ).pack(anchor=tk.W, pady=(0, 10))

        # ── Listbox con cámaras ──
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            font=("Consolas", 10),
            height=6,
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            activestyle="dotbox",
        )
        scrollbar.config(command=listbox.yview)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        status_var = tk.StringVar(value="")

        def populate_list():
            listbox.delete(0, tk.END)
            cameras = cameras_ref[0]
            if not cameras:
                listbox.insert(tk.END, "  (no se encontraron cámaras)")
                status_var.set("No se encontraron dispositivos de cámara.")
                return

            current_sel = 0
            for i, (idx, lbl) in enumerate(cameras):
                marker = " ◄ actual" if idx == current_index else ""
                listbox.insert(tk.END, f"  {lbl}{marker}")
                if idx == current_index:
                    current_sel = i

            listbox.selection_set(current_sel)
            listbox.see(current_sel)
            status_var.set(f"{len(cameras)} dispositivo(s) encontrado(s)")

        populate_list()

        status_label = ttk.Label(frame, textvariable=status_var,
                                 font=("Segoe UI", 8), foreground="#888")
        status_label.pack(anchor=tk.W, pady=(0, 8))

        # ── Botones ──
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        def on_refresh():
            status_var.set("Buscando dispositivos...")
            root.update()
            cameras_ref[0] = enumerate_cameras()
            populate_list()

        def on_change():
            sel = listbox.curselection()
            cameras = cameras_ref[0]
            if sel and cameras:
                idx = sel[0]
                if idx < len(cameras):
                    new_index = cameras[idx][0]
                    new_label = cameras[idx][1]
                    result[0] = (new_index, new_label, cameras)
            root.destroy()

        def on_cancel():
            root.destroy()

        ttk.Button(btn_frame, text="🔄 Actualizar",
                   command=on_refresh).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="✓ Cambiar cámara",
                   command=on_change).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="✗ Cancelar",
                   command=on_cancel).pack(side=tk.RIGHT, padx=4)

        root.mainloop()
        return result[0]

    except ImportError:
        # Fallback consola
        cameras = enumerate_cameras()
        if not cameras:
            print("[WARN] No se encontraron cámaras.")
            return None
        print("\n=== Cámaras disponibles ===")
        for idx, label in cameras:
            marker = " ◄ actual" if idx == current_index else ""
            print(f"  [{idx}] {label}{marker}")
        try:
            choice = int(input("Nuevo índice (Enter=cancelar): "))
            for idx, lbl in cameras:
                if idx == choice:
                    return (idx, lbl, cameras)
        except (ValueError, EOFError):
            pass
        return None


def show_help_dialog() -> None:
    """Muestra un diálogo de ayuda con los controles del programa."""
    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("SAR Geotag - Ayuda")
        root.geometry("440x350")
        root.resizable(False, False)

        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - 220
        y = (root.winfo_screenheight() // 2) - 175
        root.geometry(f"+{x}+{y}")

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="SAR Geotag Prototype",
                  font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(frame, text="Detección y geolocalización de personas",
                  font=("Segoe UI", 9), foreground="#555").pack(anchor=tk.W, pady=(0, 15))

        help_text = (
            "Controles:\n\n"
            "  Menú Cámara    →  Cambiar dispositivo de cámara\n"
            "                       (con botón Actualizar para re-escanear)\n\n"
            "  Menú Captura   →  Tomar foto + geolocalizar + abrir mapa\n"
            "  Tecla [C]        →  (atajo rápido para lo mismo)\n\n"
            "  Tecla [Q]        →  Salir del programa\n\n"
            "Algoritmo:\n"
            "  1. YOLO detecta personas en el frame\n"
            "  2. Se estima la distancia con el tamaño de la bbox\n"
            "  3. Al capturar, se calcula bearing + distancia\n"
            "  4. Se proyecta la posición GPS en un mapa OpenStreetMap"
        )

        text_widget = tk.Text(frame, font=("Consolas", 9), wrap=tk.WORD,
                              height=12, bg="#f8f8f8", relief=tk.FLAT)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="Cerrar", command=root.destroy).pack(pady=(10, 0))

        root.mainloop()

    except ImportError:
        print("\n=== Ayuda ===")
        print("  Menú Cámara: cambiar dispositivo")
        print("  Menú Captura / [C]: tomar foto y geolocalizar")
        print("  [Q]: salir")
