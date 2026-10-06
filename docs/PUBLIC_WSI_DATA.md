# Public WSI Data for Testing Path-YOLO

Path-YOLO does not bundle patient/research WSIs. Users can obtain public amyloid pathology data separately.

## Tang et al. dataset on Zenodo

**Record:** https://zenodo.org/records/1470797  
**DOI:** https://doi.org/10.5281/zenodo.1470797

The Zenodo record describes **63 whole-slide images from 63 unique decedent cases**, together with segmented 256 × 256 pixel tiles and approximately **80,000 tile-level amyloid-β pathology expert annotations**. The slides were digitized with an Aperio AT2 scanner at up to 40× magnification.

The record contains:

| Archive | Approx. size |
|---|---:|
| `Dataset 1a Development_train.zip` | 35.3 GB |
| `Dataset 1b Development_validation.zip` | 3.9 GB |
| `Dataset 2 Hold-out.zip` | 41.2 GB |
| `Dataset 3 CERAD-like hold-out.zip` | 26.3 GB |
| `Tiles.zip` | 3.3 GB |

For a first real-data test, `Dataset 1b Development_validation.zip` is the smallest WSI archive in the record and is a practical starting point.

User-provided Zenodo page:

https://zenodo.org/records/1470797?preview_file=Dataset+1b+Development_validation.zip

## Suggested workflow

1. Open the Zenodo record.
2. Download `Dataset 1b Development_validation.zip`.
3. Extract the archive to a local folder with sufficient disk space.
4. Identify the WSI files you want to process.
5. Confirm the level-0 spatial calibration for those files before setting `--um_per_pixel`.
6. Process one WSI first:

```bash
python src/path_yolo.py \
  --wsi "/path/to/extracted/slide.svs" \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
```

7. If the first run works as expected, use `--input_dir` for a batch.

## Important calibration note

`0.40 µm/pixel` is the default from the Path-YOLO research workflow. Do not assume the same calibration for every external WSI. Confirm the source slide's level-0 microns-per-pixel value and use the appropriate `--um_per_pixel` value for physical measurements.

## Data citation and rights

Before publishing analyses or redistributing files, review the Zenodo record, its associated publication, and any data-use/citation information provided by the dataset creators. Path-YOLO does not redistribute the Zenodo WSIs.
