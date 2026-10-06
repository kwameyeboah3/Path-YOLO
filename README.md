# Path-YOLO

**Path-YOLO** is a whole-slide-image (WSI) inference pipeline for automated detection of amyloid-β plaques in immunohistochemistry whole-slide images. The repository contains the portable release of the inference workflow used for the Path-YOLO study, together with the trained model weights, a smoke-test image, runnable examples, and step-by-step documentation.

> **Research use only.** Path-YOLO is not a clinical diagnostic system and is not intended for patient-care decisions.

## What this repository does

Path-YOLO processes `.tif`, `.tiff`, and `.svs` whole-slide images by:

1. dividing the WSI into overlapping **1024 × 1024** tiles;
2. applying a color-based candidate-tile filter;
3. running YOLO plaque detection on candidate tiles;
4. mapping tile detections back to WSI coordinates;
5. applying WSI-level post-processing with border-aware weighted box fusion;
6. merging overlapping detections with the **COVER40** rule;
7. applying post-COVER40 brownness and tissue filtering; and
8. exporting final plaque coordinates, micrometer-based measurements, summary tables, plots, and optional WSI overlays.

The public script is `src/path_yolo.py`. Institution-specific paths from the research version were replaced by repository-relative defaults and command-line options; the scientific detection thresholds and post-processing logic were retained.

---

## Start here

If you have never used the code before, follow these four steps.

### 1. Download the repository

Either clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Path-YOLO
```

or use **Code → Download ZIP** on GitHub, extract it, and open a terminal/PowerShell inside the extracted `Path-YOLO` folder.

### 2. Create a Python environment and install dependencies

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then check the installation:

```bash
python scripts/check_environment.py
```

A GPU is optional. Ultralytics/PyTorch will use an available compatible GPU; otherwise inference can run on CPU, but WSI processing may be much slower.

### 3. Confirm the model weights

The default model location is:

```text
models/best.pt
```

The public command examples below assume that file exists. A different model can be supplied with `--model`.

### 4. Run Path-YOLO

For the included synthetic smoke-test image:

```bash
python src/path_yolo.py \
  --wsi sample_data/synthetic_demo.tif \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
```

On Windows Command Prompt, the same command can be written on one line:

```bat
python src\path_yolo.py --wsi sample_data\synthetic_demo.tif --model models\best.pt --output_root outputs --microns_output_dir outputs\microns --um_per_pixel 0.40
```

The synthetic image only checks that the software workflow can start and process an image. **It is not amyloid pathology data and must not be used to evaluate model accuracy.**

For a real WSI, replace the `--wsi` path with the slide you want to process:

```bash
python src/path_yolo.py \
  --wsi "/path/to/your_slide.svs" \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
```

> `0.40 µm/pixel` is the calibration used in the supplied research workflow. Use the correct calibration for your own WSI if it differs.

---

## Public amyloid WSI data for trying the software

A public Alzheimer's disease pathology dataset is available from Zenodo:

- **Zenodo record:** https://zenodo.org/records/1470797
- **DOI:** https://doi.org/10.5281/zenodo.1470797
- **Suggested smaller starting archive:** `Dataset 1b Development_validation.zip` (~3.9 GB)
- User-provided preview/download page: https://zenodo.org/records/1470797?preview_file=Dataset+1b+Development_validation.zip

The Zenodo record describes 63 WSIs from 63 unique cases and includes segmented tiles plus approximately 80,000 expert amyloid-β pathology annotations. The repository does **not** redistribute these WSIs. Download them directly from Zenodo and review the dataset's citation/data-use information before publication or redistribution.

See **[docs/PUBLIC_WSI_DATA.md](docs/PUBLIC_WSI_DATA.md)** for a guided download-and-run workflow.

---

## Run an entire folder of WSIs

Path-YOLO can recursively process `.tif`, `.tiff`, and `.svs` files:

```bash
python src/path_yolo.py \
  --input_dir "/path/to/wsi_folder" \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
```

Use **either** `--wsi` or `--input_dir`, not both.

---

## What to look at after a run

For each WSI, Path-YOLO creates a folder under `outputs/`. The most important files are:

| Output | Purpose |
|---|---|
| `plaque_locations_cover40_boxes.csv` | Final Path-YOLO plaque boxes after COVER40 + post-filtering |
| `plaque_locations_wsi_postprocessed.csv` | WSI-level post-processed detections before final COVER40 filtering |
| `plaque_locations_tilelevel.csv` | Raw tile-level YOLO detections mapped to WSI coordinates |
| `cover40_brownness_scores.csv` | Brownness/tissue filtering information |
| `*_plaque_results.xlsx` | Plaque locations converted to micrometers plus summary statistics |
| `*_COVER40_POSTFILTER_overlay.tif` | Full-resolution overlay with final plaque boxes, when enabled |
| `*_COVER40_POSTFILTER_thumb_overlay.png` | Lightweight preview of the final overlay |
| `status.json` | Processing status and step-level counts |

For a field-by-field explanation, see **[docs/OUTPUTS.md](docs/OUTPUTS.md)**.

---

## Viewing results in Aperio ImageScope

For Windows users, **Aperio ImageScope** is the recommended WSI viewer for inspecting the original slide and Path-YOLO's large TIFF overlay output.

Official Leica Biosystems page:

https://www.leicabiosystems.com/us/digital-pathology/manage/aperio-imagescope/

After processing a WSI:

1. open the corresponding folder under `outputs/`;
2. locate `*_COVER40_POSTFILTER_overlay.tif`;
3. open that TIFF in ImageScope;
4. pan and zoom through the slide to inspect the final blue Path-YOLO plaque boxes; and
5. optionally open the original `.svs`/`.tif` alongside it for comparison.

If you do not need the large full-resolution overlay, inspect `*_COVER40_POSTFILTER_thumb_overlay.png` instead.

ImageScope is third-party software and is not included in this repository.

---

## Repository layout

```text
Path-YOLO/
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── VERSION
├── MODEL_CARD.md
├── LICENSE_NOTICE.md
├── CHANGELOG.md
├── src/
│   └── path_yolo.py
├── models/
│   ├── best.pt
│   └── README.md
├── sample_data/
│   ├── synthetic_demo.tif
│   └── README.md
├── examples/
│   ├── run_single_wsi.sh
│   ├── run_folder.sh
│   └── run_single_wsi.bat
├── scripts/
│   └── check_environment.py
├── docs/
│   ├── USER_GUIDE.md
│   ├── INSTALLATION.md
│   ├── OUTPUTS.md
│   ├── TROUBLESHOOTING.md
│   ├── PUBLIC_WSI_DATA.md
│   ├── PORTABILITY_CHANGES.md
│   └── PUBLIC_RELEASE_CHECKLIST.md
└── outputs/
    └── .gitkeep
```

---

## Important defaults from the paper workflow

| Setting | Default |
|---|---:|
| Tile size | 1024 px |
| Tile overlap | 50% |
| Tile confidence threshold | 0.10 |
| Tile NMS IoU threshold | 0.80 |
| WSI post-processing | WBF |
| WSI clustering IoU | 0.50 |
| COVER40 threshold | 0.40 |
| Border-aware WBF | Enabled |
| Post-COVER40 brownness/tissue filter | Enabled |
| Default spatial calibration | 0.40 µm/pixel |

These remain defined in `src/path_yolo.py` to preserve the paper workflow.

---

## Command-line help

```bash
python src/path_yolo.py --help
```

Key options:

```text
--wsi PATH                  process one WSI
--input_dir PATH            process all supported WSIs in a folder
--model PATH                YOLO .pt model file
--output_root PATH          per-WSI output directory
--microns_output_dir PATH   micron tables/plots output directory
--um_per_pixel FLOAT        physical WSI calibration
```

---

## Detailed guides

- **[User guide](docs/USER_GUIDE.md)** — complete beginner-friendly workflow from download to viewing results.
- **[Installation](docs/INSTALLATION.md)** — environment setup, optional OpenSlide, and GPU notes.
- **[Outputs](docs/OUTPUTS.md)** — explanation of generated CSV, Excel, image, and status files.
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — common errors and fixes.
- **[Public WSI data](docs/PUBLIC_WSI_DATA.md)** — how to obtain public amyloid WSI data from Zenodo.
- **[Model card](MODEL_CARD.md)** — intended use and limitations of the supplied model.

---

## Reproducibility and limitations

Whole-slide images can be extremely large. This implementation creates overlapping tiles and may construct large WSI-sized output canvases, so processing can require substantial RAM, disk space, and time. The supplied synthetic TIFF is a software smoke test only. Reproducing manuscript-level analyses requires the appropriate study data, model weights, calibration, and the settings reported in the paper.

Path-YOLO detections should be interpreted as research outputs. External validation is recommended before applying the model to images from different laboratories, scanners, staining protocols, magnifications, or populations.

---

## Citation

Please cite the Path-YOLO manuscript when it becomes publicly available. The final journal citation/DOI should be added here after publication.

For the public WSI dataset linked above, follow the citation instructions on the Zenodo record and associated Tang et al. publication.

## License

A public software license has **not yet been selected for this release candidate**.
