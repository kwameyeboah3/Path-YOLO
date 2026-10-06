#!/usr/bin/env bash
set -euo pipefail

# Replace /path/to/wsi_folder with your folder.
python src/path_yolo.py \
  --input_dir "/path/to/wsi_folder" \
  --model models/best.pt \
  --output_root outputs \
  --microns_output_dir outputs/microns \
  --um_per_pixel 0.40
