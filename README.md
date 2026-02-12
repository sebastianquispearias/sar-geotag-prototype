# SAR Geotag Prototype (Prototipo A)

Este repositorio implementa un prototipo para SAR: detectar personas con YOLO y estimar una distancia aproximada usando la altura de la bbox (Alternativa A). Además, permite capturar fotos y geolocalizar las personas detectadas en un mapa open-source (OpenStreetMap/Leaflet).

## Funcionalidades
- **Selector de cámara**: Al iniciar, se muestran los dispositivos de cámara disponibles para elegir cuál usar.
- **Posición GPS**: Diálogo para ingresar la latitud, longitud y heading de la cámara.
- **Detección de personas**: YOLO detecta personas en tiempo real y muestra la distancia estimada.
- **Captura de foto + geolocalización**: Presionar `C` captura una foto, geolocaliza cada persona usando bearing + distancia, y abre un mapa interactivo.
- **Mapa open-source**: Visualiza las posiciones estimadas con marcadores rojos sobre OpenStreetMap.

## Requisitos
- Windows 10/11
- Python 3.10 o 3.11
- Cámara/webcam
- Git
- Conexión a internet (para cargar tiles de OpenStreetMap en el mapa)

## Instalación (Windows PowerShell)
Clona el repo y crea un entorno virtual:

```powershell
git clone https://github.com/sebastianquispearias/sar-geotag-prototype.git
cd sar-geotag-prototype
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src
```

## Controles en tiempo real
| Tecla | Acción |
|-------|--------|
| `C`   | Capturar foto → geolocalizar personas → abrir mapa |
| `Q`   | Salir del programa |

## Algoritmo de Geolocalización
Basado en el paper "Object Geolocalization Using Consumer-Grade Devices":

1. **Bearing**: Se calcula el ángulo desde la cámara hacia cada persona detectada usando su posición horizontal en la imagen y el FOV de la cámara (modelo pinhole).
2. **Distancia horizontal**: La distancia estimada (slant) se proyecta al plano del suelo corrigiendo por el ángulo de pitch.
3. **Destination Point**: Con la fórmula esférica se calcula la latitud/longitud del objeto a partir de la posición GPS de la cámara, el bearing y la distancia.
