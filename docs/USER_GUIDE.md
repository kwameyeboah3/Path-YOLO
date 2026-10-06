# Path-YOLO User Guide

This guide is written for a new user who wants to go from a downloaded GitHub repository to a viewable Path-YOLO result.

## Workflow overview

**Download Path-YOLO → install Python packages → obtain a WSI → run the model → inspect CSV/Excel outputs → open the overlay in ImageScope.**

## Step 1 — Get Path-YOLO

Clone the repository or download the repository ZIP from GitHub and extract it. Open a terminal in the repository root. You should see `README.md`, `src/`, `models/`, `docs/`, and `examples/`.

## Step 2 — Install the software

Follow `docs/INSTALLATION.md`. A minimal Windows setup is:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

## Step 3 — Confirm the model

Check that this file exists:

```text
models/best.pt
```

If the model is stored elsewhere, pass its path with `--model`.

## Step 4 — Get an input WSI

Path-YOLO accepts `.tif`, `.tiff`, and `.svs` files.

For real public amyloid pathology images, see `docs/PUBLIC_WSI_DATA.md`. For a basic software smoke test, use the included `sample_data/synthetic_demo.tif`.

## Step 5 — Know the WSI calibration

Path-YOLO converts pixels to micrometers. The supplied paper workflow uses:

```text
0.40 µm/pixel
```

If your WSI has a different level-0 resolution, provide the correct value using `--um_per_pixel`. Do not assume 0.40 for an external dataset without confirming the scanner/image metadata.

## Step 6 — Process one WSI

```bash
python src/path_yolo.py \
  --wsi "/path/to/slide.svs" \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
```

For Windows paths containing spaces, keep the WSI path in quotes:

```bat
python src\path_yolo.py --wsi "C:\Data\Amyloid Slides\slide01.svs" --model models\best.pt --output_root outputs --microns_output_dir outputs\microns --um_per_pixel 0.40
```

## Step 7 — Watch the run

Path-YOLO prints progress for tiling, color-based tile sorting, YOLO prediction, WSI post-processing, COVER40 merging, brownness/tissue filtering, and output generation.

WSIs are large, so processing may take a long time and may create many temporary/output files. Run from a disk with sufficient free space.

## Step 8 — Find the final detections

Suppose the WSI is called `slide01.svs`. Path-YOLO creates a folder similar to:

```text
outputs/
└── slide01/
    ├── plaque_locations_tilelevel.csv
    ├── plaque_locations_wsi_postprocessed.csv
    ├── plaque_locations_cover40_boxes.csv
    ├── cover40_brownness_scores.csv
    ├── slide01_Default_Extended_plaque_results.xlsx
    ├── slide01_COVER40_POSTFILTER_overlay.tif
    ├── slide01_COVER40_POSTFILTER_thumb_overlay.png
    └── status.json
```

The main final plaque table is:

```text
plaque_locations_cover40_boxes.csv
```

See `docs/OUTPUTS.md` for detailed descriptions.

## Step 9 — View the WSI result

### Aperio ImageScope

Download ImageScope from Leica Biosystems:

https://www.leicabiosystems.com/us/digital-pathology/manage/aperio-imagescope/

Then:

1. start ImageScope;
2. open `*_COVER40_POSTFILTER_overlay.tif`;
3. zoom into tissue regions;
4. inspect the blue boxes marking Path-YOLO final detections;
5. optionally open the original WSI separately for comparison.

The thumbnail PNG is useful for a quick review when the full TIFF is too large.

## Step 10 — Process multiple WSIs

Put the slides under one directory and run:

```bash
python src/path_yolo.py \
  --input_dir "/path/to/wsi_folder" \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
```

The program recursively searches for supported WSI extensions.

## Step 11 — Record reproducibility information

For research use, record at least:

- Path-YOLO repository version/commit
- model weights/checksum
- WSI filenames and source
- `µm/pixel` calibration
- software environment
- any configuration values changed from the defaults

This makes it possible to reproduce a run later.
