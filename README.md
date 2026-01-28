# SAR Geotag Prototype (Prototipo A)

Este repositorio implementa un prototipo para SAR: detectar personas con YOLO y estimar una distancia aproximada usando la altura de la bbox (Alternativa A). El objetivo es tener un pipeline reproducible en laptop y luego migrarlo a Raspberry Pi/dron cambiando el backend de inferencia.

## Requisitos
- Windows 10/11
- Python 3.10 o 3.11
- Cámara/webcam
- Git

## Instalación (Windows PowerShell)
Clona el repo y crea un entorno virtual:

```powershell
git clone https://github.com/sebastianquispearias/sar-geotag-prototype.git
cd sar-geotag-prototype
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
