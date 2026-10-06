# Troubleshooting

## `Model not found`

Confirm that `models/best.pt` exists or pass the correct file:

```bash
python src/path_yolo.py --wsi slide.svs --model /path/to/best.pt ...
```

## `ModuleNotFoundError`

Activate the environment and reinstall dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python scripts/check_environment.py
```

## OpenSlide import/DLL error

OpenSlide is optional in this release. If you want OpenSlide support, install `requirements-optional.txt` and the native OpenSlide library required by your operating system. The code contains TIFF-based thumbnail fallbacks when OpenSlide is unavailable.

## LZW TIFF compression error

Install/reinstall `imagecodecs`:

```bash
pip install --upgrade imagecodecs
```

## Very slow processing

WSIs are large and the pipeline uses overlapping 1024-pixel tiles. CPU inference and full-resolution overlay writing can be slow. Use a compatible GPU when available and ensure the output drive has sufficient free space.

## Memory error / process killed

The research implementation may allocate large arrays for WSI-sized canvases. Try a machine with more RAM, use a smaller WSI for initial testing, or disable optional large full-resolution outputs in `src/path_yolo.py` if the scientific use case permits.

## No detections

Check all of the following:

1. the correct model weights are loaded;
2. the image is a compatible amyloid-β IHC WSI;
3. the slide opens correctly and is not blank/corrupt;
4. the color/stain distribution is reasonably compatible with the model domain;
5. the synthetic demo is not being mistaken for a performance test.

No detection does not necessarily indicate a software error.

## `Already locked` / WSI skipped

The pipeline uses a per-WSI `.lock` directory to prevent two jobs from writing the same output simultaneously. Make sure no other Path-YOLO process is working on that slide. If a previous job crashed, inspect the output folder before removing a stale `.lock` directory manually.

## Permission denied

Choose an output folder where you have write permission. On shared/HPC systems, confirm filesystem permissions and available quota.

## Windows paths with spaces

Wrap paths in quotation marks:

```bat
--wsi "C:\Users\Name\My Slides\slide01.svs"
```

## Full overlay does not open in a normal image viewer

Large TIFF/BigTIFF files can exceed the capabilities of standard photo viewers. Use Aperio ImageScope or another WSI-capable viewer.
