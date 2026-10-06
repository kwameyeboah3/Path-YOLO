# Model Weights

The default Path-YOLO model is:

```text
best.pt
```

`src/path_yolo.py` uses this file automatically when the repository layout is unchanged.

To use a different weight file:

```bash
python src/path_yolo.py --wsi /path/to/slide.svs --model /path/to/model.pt --output_root outputs --microns_output_dir outputs/microns --um_per_pixel 0.40
```