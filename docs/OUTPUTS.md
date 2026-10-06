# Path-YOLO Outputs

## Per-WSI output folder

Each processed WSI receives its own subfolder under `--output_root`.

### `plaque_locations_tilelevel.csv`

Tile-level YOLO detections after local coordinates have been mapped to WSI-level coordinates. Useful for debugging the detector before WSI-level merging.

Typical fields include:

- `global_x_min`, `global_y_min`, `global_x_max`, `global_y_max`: level-0 WSI bounding-box coordinates in pixels
- `score`: YOLO confidence
- `border_weight`: weight used to reduce the influence of detections near tile borders
- `effective_score`: confidence multiplied by border weight
- `class`: model class ID
- `tile_row`, `tile_col`: source tile position
- `tile_rel_path`: source tile filename
- `detection_idx`: tile-level detection identifier

### `plaque_locations_wsi_postprocessed.csv`

Detections after WSI-level post-processing and before the final COVER40 stage.

### `plaque_locations_cover40_boxes.csv`

**Primary final plaque-location table.** These boxes have passed the WSI post-processing, COVER40 merging, and post-COVER40 filtering stages.

### `cover40_brownness_scores.csv`

Records the brownness/tissue measurements and keep/drop decision used by the post-COVER40 filter.

### `*_Default_Extended_plaque_results.xlsx`

Excel workbook containing the final plaque locations converted to physical units and summary statistics. Measurements can include width, height, bounding-box area, equivalent circular diameter, and maximum diameter in micrometers.

### `*_COVER40_POSTFILTER_overlay.tif`

Full-resolution WSI overlay with final Path-YOLO detections drawn as blue boxes. This can be large. Aperio ImageScope is recommended for viewing it on Windows.

### `*_COVER40_POSTFILTER_thumb_overlay.png`

Downsampled preview of the overlay. Use this for quick QC before opening the full TIFF.

### `status.json`

Machine-readable information about processing steps, counts, output paths, and errors when available.

## Micron-output directory

The directory supplied with `--microns_output_dir` can contain additional Excel workbooks, plaque-size histograms, and overlay outputs.

## Which file should I use?

- For **final plaque coordinates**: `plaque_locations_cover40_boxes.csv`
- For **physical size measurements**: `*_plaque_results.xlsx`
- For **visual inspection**: `*_COVER40_POSTFILTER_overlay.tif`
- For **quick visual QC**: `*_COVER40_POSTFILTER_thumb_overlay.png`
- For **debugging earlier detection stages**: tile-level and WSI-postprocessed CSVs
