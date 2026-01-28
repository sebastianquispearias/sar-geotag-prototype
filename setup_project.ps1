# setup_project.ps1
# Crea estructura de proyecto + archivos base para sar-geotag-prototype (Windows PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "==> Creating folders..." -ForegroundColor Cyan

$dirs = @(
  "configs",
  "src",
  "src\io",
  "src\pose",
  "src\vision",
  "src\geometry",
  "src\estimators",
  "src\viz",
  "src\logging",
  "scripts",
  "weights",
  "runs",
  "data",
  "assets"
)

$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

Write-Host "==> Creating placeholder files..." -ForegroundColor Cyan

$files = @(
  "README.md",
  ".gitignore",
  "requirements.txt",
  "configs\default.yaml",

  "src\main.py",
  "src\types.py",

  "src\io\framesource.py",
  "src\pose\pose_provider.py",

  "src\vision\detector_base.py",
  "src\vision\ultralytics_detector.py",
  "src\vision\ncnn_detector.py",
  "src\vision\tflite_detector.py",

  "src\geometry\camera_model.py",
  "src\geometry\transforms.py",
  "src\geometry\intersections.py",

  "src\estimators\estimator_base.py",
  "src\estimators\bbox_size.py",
  "src\estimators\ray_plane.py",
  "src\estimators\multiview.py",

  "src\viz\overlay.py",
  "src\viz\ui.py",

  "src\logging\run_logger.py",
  "src\logging\metrics.py",

  "scripts\run_webcam.py",
  "scripts\run_video.py",
  "scripts\evaluate_distances.py"
)

$files | ForEach-Object {
  if (!(Test-Path $_)) { New-Item -ItemType File -Path $_ -Force | Out-Null }
}

Write-Host "==> Adding __init__.py for packages..." -ForegroundColor Cyan

$initDirs = @(
  "src",
  "src\io",
  "src\pose",
  "src\vision",
  "src\geometry",
  "src\estimators",
  "src\viz",
  "src\logging"
)

$initDirs | ForEach-Object {
  $p = Join-Path $_ "__init__.py"
  if (!(Test-Path $p)) { New-Item -ItemType File -Path $p -Force | Out-Null }
}

Write-Host "==> Writing .gitignore / requirements / default config..." -ForegroundColor Cyan

$gitignore = @"
# Python
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.log

# VSCode
.vscode/

# Data / outputs
weights/
runs/
data/
assets/demo_videos/

# OS
Thumbs.db
.DS_Store
"@

Set-Content -Path ".gitignore" -Value $gitignore -Encoding UTF8

$req = @"
ultralytics
opencv-python
numpy
pyyaml
"@

Set-Content -Path "requirements.txt" -Value $req -Encoding UTF8

$cfg = @"
source:
  type: webcam       # webcam | video
  webcam_index: 0
  video_path: ""

detector:
  backend: ultralytics   # ultralytics | ncnn | tflite
  model_path: ""         # opcional: si vacío, ultralytics usa default/pretrained
  conf_thres: 0.25
  iou_thres: 0.45
  classes: ["person"]    # para SAR: persona primero

camera:
  width: 1280
  height: 720
  hfov_deg: 78.0         # aproximado; ajusta luego
  vfov_deg: 44.0         # aproximado; ajusta luego

pose:
  provider: simulated
  height_m: 10.0         # simulado por ahora
  yaw_deg: 0.0
  pitch_deg: -30.0
  roll_deg: 0.0

estimation:
  method: bbox_size      # bbox_size | ray_plane
  person_height_m: 1.70  # para Alternativa A
  ground_plane_z: 0.0    # para Alternativa B (suelo plano)

logging:
  enabled: true
  out_dir: runs
  save_video: false
"@

Set-Content -Path "configs\default.yaml" -Value $cfg -Encoding UTF8

Write-Host "==> Done. Project skeleton created." -ForegroundColor Green
Write-Host "Next: create a venv and install deps:" -ForegroundColor Yellow
Write-Host "  py -m venv .venv" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
