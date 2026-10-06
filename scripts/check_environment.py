#!/usr/bin/env python3
"""Lightweight Path-YOLO environment checker."""
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]
checks = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("tifffile", "tifffile"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
    ("PIL", "Pillow"),
    ("shapely", "shapely"),
    ("openpyxl", "openpyxl"),
    ("imagecodecs", "imagecodecs"),
    ("ultralytics", "ultralytics"),
]

print(f"Python: {sys.version.split()[0]}")
print(f"Repository: {ROOT}")
print("\nDependencies:")
missing = []
for module, package in checks:
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [OK] {package}: {ver}")
    except Exception as exc:
        print(f"  [MISSING/ERROR] {package}: {exc}")
        missing.append(package)

for label, path in [
    ("Path-YOLO script", ROOT / "src" / "path_yolo.py"),
    ("Model weights", ROOT / "models" / "best.pt"),
    ("Synthetic demo", ROOT / "sample_data" / "synthetic_demo.tif"),
]:
    print(f"  [{'OK' if path.exists() else 'MISSING'}] {label}: {path}")

try:
    import openslide
    print(f"  [OPTIONAL OK] openslide-python: {getattr(openslide, '__version__', 'installed')}")
except Exception:
    print("  [OPTIONAL] OpenSlide not available; TIFF fallback paths remain available.")

if missing:
    print("\nSome required packages are missing. Run: pip install -r requirements.txt")
    raise SystemExit(1)
print("\nEnvironment check passed.")
