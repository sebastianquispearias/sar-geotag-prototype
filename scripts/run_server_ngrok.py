"""
Lanzar servidor SAR Geotag para uso con ngrok.

Uso:
    1. Iniciar ngrok:   ngrok http 5000
    2. Ejecutar esto:   python scripts/run_server_ngrok.py
    3. En la app Android, poner la URL de ngrok como servidor
       Ejemplo: https://xxxxx.ngrok-free.dev

Si ya tienes ngrok en otro puerto (ej: 80), puedes pasar el puerto:
    python scripts/run_server_ngrok.py --port 80
    (requiere ejecutar como Administrador si es puerto < 1024)
"""

import sys
import os
import argparse
from pathlib import Path

# Asegurar que se importa desde la raíz del proyecto
_root = Path(__file__).resolve().parent.parent
os.chdir(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Importar el servidor
import server_triangulation as server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor SAR Geotag con ngrok")
    parser.add_argument("--port", "-p", type=int, default=5000,
                        help="Puerto del servidor (default: 5000)")
    args = parser.parse_args()
    
    server.SERVER_PORT = args.port
    
    print("\n" + "=" * 60)
    print("  MODO NGROK")
    print(f"  Servidor en puerto {args.port}")
    print(f"  Asegúrate que ngrok apunte a este puerto:")
    print(f"    ngrok http {args.port}")
    print("=" * 60 + "\n")
    
    server.main()
