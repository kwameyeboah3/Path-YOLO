# Installation Guide

This guide installs Path-YOLO in a clean Python environment.

## 1. System requirements

Recommended:

- 64-bit Windows or Linux
- Python 3.10 or 3.11
- Enough free disk space for WSI tiles and output files
- Substantial RAM for large whole-slide images
- NVIDIA GPU optional; CPU execution is supported but can be slow

## 2. Create the environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, you can run the Python executable directly from `.venv\Scripts\python.exe` or follow your institution's PowerShell policy.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Check the environment

```bash
python scripts/check_environment.py
```

The checker verifies the main Python packages, the Path-YOLO source file, and the default `models/best.pt` location.

## 4. Optional OpenSlide support

Path-YOLO can use OpenSlide for efficient WSI thumbnail operations when available. The code also includes TIFF-based fallbacks, so OpenSlide is not required for the basic installation.

Python binding:

```bash
pip install -r requirements-optional.txt
```

The `openslide-python` package may also require the native OpenSlide library for your operating system. Follow the OpenSlide project's installation instructions if import errors occur.

## 5. GPU notes

Ultralytics uses PyTorch. If a compatible CUDA-enabled PyTorch installation is available, YOLO inference can use the GPU. Otherwise it runs on CPU. For institution/HPC environments, install the PyTorch build that matches the local CUDA configuration before or together with Ultralytics.

## 6. Verify command-line access

```bash
python src/path_yolo.py --help
```

If that command prints the Path-YOLO options, installation is ready.
