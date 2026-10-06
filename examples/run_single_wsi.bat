@echo off
python src\path_yolo.py --wsi sample_data\synthetic_demo.tif --model models\best.pt --output_root outputs --microns_output_dir outputs\microns --um_per_pixel 0.40
