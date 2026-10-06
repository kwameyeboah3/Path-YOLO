# Path-YOLO Model Card

## Model

The repository contains trained YOLO weights at:

```text
models/best.pt
```

The weights are used by `src/path_yolo.py` to detect amyloid-β plaques on candidate WSI tiles.

## Intended use

- research analysis of amyloid-β immunohistochemistry whole-slide images;
- reproducibility of the Path-YOLO research workflow;
- method development and external research evaluation.

## Not intended for

- clinical diagnosis;
- treatment decisions;
- unsupervised deployment in patient care;
- claims of performance on scanners, stains, laboratories, or populations not evaluated by the authors.

## Input domain

The pipeline is designed for pathology WSIs and accepts `.tif`, `.tiff`, and `.svs` files. The inference code applies color-based candidate filtering before YOLO detection and additional WSI/post-COVER40 filtering afterward.

## Output

The model contributes bounding-box detections that are mapped to WSI coordinates and subsequently post-processed by Path-YOLO. The primary final detection table is `plaque_locations_cover40_boxes.csv`.

## Limitations

Performance can change with stain intensity, scanner characteristics, magnification, image calibration, tissue preparation, artifacts, and domain shift. The supplied synthetic image is not suitable for model evaluation. External validation is recommended for new datasets.

## Training and evaluation details

Refer to the Path-YOLO manuscript and its supplementary material for the authoritative description of model development, training, and reported performance. This repository documentation does not replace the paper.
