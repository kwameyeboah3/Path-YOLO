#!/usr/bin/env python3
"""
Path-YOLO public inference pipeline (portable release)

Portable release derived from Path-YOLO inference workflow; supports .tif / .tiff / .svs

Keeps:
  - Overlapping tiling (stride < tile)
  - Color-based tile filtering
  - YOLO prediction
  - WSI-level postprocessing (WBF/support)  [optional]
  - COVER40 merge rule -> plaque_locations_cover40_boxes.csv  [FINAL boxes]
  - Microns outputs (area, ECD, max-diameter)
  - side_by_side.tif overlay (BLUE ONLY; single panel)

Removes:
  - Clustering from COVER40 (grid hash)
  - Green clusters-from-cover40 overlay
  - Cluster CSVs + cluster sheets in Excel
"""

import json
import argparse
import math
import os
import shutil
import getpass
import atexit
import socket
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt  # required by histogram saving

# Optional: for fast WSI thumbnail overlays
try:
    import openslide
    _HAS_OPENSLIDE = True
except Exception:
    openslide = None
    _HAS_OPENSLIDE = False

# Optional: for union/overlap area stats (microns stats)
try:
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    shapely_box = None
    unary_union = None
    _HAS_SHAPELY = False

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["YOLO_NO_HUB"] = "1"

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False


# ============================================================
# CONFIG
# ============================================================

# Portable repository-relative defaults. These may be overridden from the command line.
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_WSI_DIR = str(REPO_ROOT / "sample_data")
OUTPUT_ROOT = str(REPO_ROOT / "outputs")

# Final microns + plots + side-by-side overlay outputs
MICRONS_OUTPUT_DIR = str(REPO_ROOT / "outputs" / "microns")
UM_PER_PIXEL = 0.40

# Faster overlay for thousands of boxes
THUMB_MAX_SIDE = 2000            # 1500–2500 recommended
MAX_PLAQUE_BOXES_TO_DRAW = 8000  # draw at most this many cover40 boxes on the overlay
DRAW_TOP_BY_SCORE = True         # if 'score' exists, draw top boxes; else random sample
# --- COVER40 overlay saving (full-res BigTIFF can be VERY slow) ---
SAVE_FULLRES_COVER40_OVERLAY_TIF = True   # set True only when you really need the BigTIFF
COVER40_OVERLAY_COMPRESSION = "lzw"         # None (fast) or "lzw" (slow)
SAVE_COVER40_THUMB_OVERLAY_PNG = True      # lightweight QC overlay
COVER40_THUMB_SIDE = 2000
SAVE_FULLRES_MERGED_RAW_TIF = False          # save full merged raw WSI mosaic from all tiles
MERGED_RAW_TIF_COMPRESSION = "lzw"


# Use the retrained 1024 model (edit if needed)
YOLO_MODEL_PATH = str(REPO_ROOT / "models" / "best.pt")

# --- Single WSI test mode ---
RUN_SINGLE_WSI_ONLY = False
SINGLE_WSI_PATH = str(REPO_ROOT / "sample_data" / "synthetic_demo.tif")

# --- Tiling ---
TILE_SIZE = 1024

# Overlap: 0.25 (25%) or 0.50 (50%)
TILE_OVERLAP = 0.50
TILE_STRIDE = int(TILE_SIZE * (1.0 - TILE_OVERLAP))  # e.g. 512 for 50% overlap
TILE_STRIDE = max(64, TILE_STRIDE)

# Sorting thresholds (tile heuristic)
MIN_SAT = 10
MIN_BROWN_FRACTION = 0.01



# --- Post-COVER40 brownness filter (box-level) ---
POST_COVER40_BROWN_FILTER = True

# Brown hue window similar to tile sorting: brown ~ H in [5,30] with saturation constraint
POST_BROWN_MIN_SAT = 25
POST_BROWN_H_LO = 5
POST_BROWN_H_HI = 30

# Fraction of pixels inside the COVER40 box that must look "brown"
POST_BROWN_MIN_FRACTION = 0.015   # tune (0.008–0.03); raised to kill non-brown edge FPs   # tune (0.002–0.01)

# If True, apply brownness requirement to ALL boxes (even high-confidence).
POST_BROWN_REQUIRE_FOR_ALL = True

# Optional safety: keep extremely high-confidence boxes even if brownness is low
POST_BROWN_KEEP_IF_SCORE_GE = 1.00  # set 1.0 to disable override; tune down only if needed

# --- Post-COVER40 tissue gating (box-level) ---
# Helps remove false positives on glass/background/slide edges.
POST_TISSUE_GRAY_THRESH = 235          # pixels darker than this are considered "tissue"
POST_BOX_MIN_TISSUE_FRACTION = 0.12    # min tissue fraction inside box to keep (tune 0.08–0.25)

# Stricter tissue requirement for very large COVER40 unions (these are often edge artifacts)
POST_LARGE_BOX_AREA_PIX = 1024 * 1024
POST_LARGE_BOX_MIN_TISSUE_FRACTION = 0.25

# Large-box stricter rule (COVER40 unions can create huge non-plaque boxes at slide edges)
POST_LARGE_BOX_AREA_PIX = 1024 * 1024   # if box area at level-0 exceeds this, require stronger brownness
POST_LARGE_BOX_MIN_BROWN_FRACTION = 0.03

# Thumbnail scoring size (fast)
POST_BROWN_THUMB_MAX_SIDE = 2000

# Prediction thresholds (tile-level)
CONF_THRES = 0.10          # ignore <= 0.1
TILE_NMS_IOU_THRES = 0.80  # Ultralytics tile-local NMS IoU

# --- WSI-level post-processing ---
WSI_POSTPROCESS = True

# Choose:
#   "softnms" -> Soft-NMS across whole WSI
#   "wbf"     -> cluster + weighted box fusion (NOT only highest prob)  [DEFAULT]
WSI_METHOD = "wbf"

WSI_CLUSTER_IOU = 0.50

# Soft-NMS parameters
SOFTNMS_METHOD = "gaussian"     # "linear" or "gaussian"
SOFTNMS_SIGMA = 0.5
SOFTNMS_IOU_THRESH = 0.50
SOFTNMS_SCORE_THRESH = 0.001

# Containment removal (drop small boxes mostly inside bigger boxes)
CONTAINMENT_FRAC = 0.90

# Composite scoring: score + lambda*support_count
SUPPORT_LAMBDA = 0.10

# Merge output canvas dtype
CANVAS_DTYPE = np.uint8

# --- COVER40 RULE SETTINGS
DRAW_BLUE_OVERLAY_COVER40 = True
COVER40_THRESH = 0.40          # (intersection / smaller_box_area) >= 0.40
COVER40_SOURCE = "postprocessed"   # "tilelevel" or "postprocessed"

# --- Border-aware weighting before WBF ---
BORDER_AWARE_WBF = True
EDGE_MARGIN_OUTER_PX = 64      # very edge: strongest downweight
EDGE_MARGIN_INNER_PX = 128     # near edge: mild downweight
BORDER_WEIGHT_OUTER = 0.60
BORDER_WEIGHT_INNER = 0.80
BORDER_WEIGHT_CENTER = 1.00
COVER40_SCORE_USE_MAX = True   # merged union box score = max(score) vs mean(score)

BLUE_THICKNESS = 2
BLUE_FONT_SCALE = 0.5


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TileMeta:
    path: str
    rel_path: str
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    cls: int
    tile_row: int
    tile_col: int
    idx: int
    support: int = 1
    border_weight: float = 1.0


# ============================================================
# UTILS
# ============================================================

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied while creating directory: {path}\n"
            f"Check ownership/ACLs or choose a writable OUTPUT_ROOT. Original error: {e}"
        ) from e


def list_wsis(input_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith((".tif", ".tiff", ".svs")):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def stem_from_path(path: str) -> str:
    base = os.path.basename(path)
    for ext in [".tif", ".tiff", ".svs"]:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def border_weight_from_local_box(x0: float, y0: float, x1: float, y1: float, tile_w: int, tile_h: int) -> float:
    """Downweight detections whose centers are too close to the tile border.
    The same plaque is usually seen more completely in a neighboring overlapping tile.
    """
    cx = 0.5 * (float(x0) + float(x1))
    cy = 0.5 * (float(y0) + float(y1))
    dist_border = min(cx, cy, float(tile_w) - cx, float(tile_h) - cy)

    if not BORDER_AWARE_WBF:
        return 1.0
    if dist_border < EDGE_MARGIN_OUTER_PX:
        return float(BORDER_WEIGHT_OUTER)
    if dist_border < EDGE_MARGIN_INNER_PX:
        return float(BORDER_WEIGHT_INNER)
    return float(BORDER_WEIGHT_CENTER)


def effective_det_score(b: "Box") -> float:
    return float(b.score) * float(getattr(b, "border_weight", 1.0))


def get_positions(length: int, tile: int, stride: int) -> List[int]:
    """Generate tile start positions with overlap, ensuring last tile covers the end."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def to_uint8_rgb(tile: np.ndarray) -> np.ndarray:
    """Fix dark tiles from uint16/other ranges -> uint8 RGB using percentile scaling."""
    if tile is None:
        return tile
    if tile.dtype == np.uint8:
        return tile

    t = tile.astype(np.float32)
    lo = np.percentile(t, 1.0)
    hi = np.percentile(t, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    t = (t - lo) / (hi - lo)
    t = np.clip(t, 0, 1) * 255.0
    return t.astype(np.uint8)


# ============================================================
# WSI LOCKING (safe parallel jobs)
# ============================================================

def _lock_dir_for_wsi(output_root: str, wsi_stem: str) -> str:
    return os.path.join(output_root, wsi_stem, ".lock")


def acquire_wsi_lock(output_root: str, wsi_stem: str) -> Optional[str]:
    """Create an atomic lock directory. Returns lock_path if acquired else None."""
    lock_dir = _lock_dir_for_wsi(output_root, wsi_stem)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)

    try:
        os.mkdir(lock_dir)  # atomic on POSIX
    except FileExistsError:
        return None

    try:
        info = {
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(lock_dir, "owner.json"), "w") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass

    try:
        atexit.register(release_wsi_lock, lock_dir)
    except Exception:
        pass

    return lock_dir


def release_wsi_lock(lock_path: Optional[str]):
    if not lock_path:
        return
    try:
        shutil.rmtree(lock_path)
    except Exception:
        pass


# ============================================================
# STEP 1 – OVERLAPPING TILING
# ============================================================

def tile_wsi_to_pngs(wsi_path: str, out_dir: str, wsi_stem: str) -> List[TileMeta]:
    ensure_dir(out_dir)

    print(f"[TILE] Reading WSI: {wsi_path}")
    img = tiff.imread(wsi_path)
    h, w = img.shape[:2]
    print(f"[TILE] WSI shape: {w} x {h} | tile={TILE_SIZE} stride={TILE_STRIDE} overlap={TILE_OVERLAP:.2f}")

    xs = get_positions(w, TILE_SIZE, TILE_STRIDE)
    ys = get_positions(h, TILE_SIZE, TILE_STRIDE)

    tile_metas: List[TileMeta] = []
    row = 0
    for y in ys:
        col = 0
        for x in xs:
            tile = img[y: y + TILE_SIZE, x: x + TILE_SIZE]
            tile_h, tile_w = tile.shape[:2]
            if tile_h == 0 or tile_w == 0:
                col += 1
                continue

            tile_u8 = to_uint8_rgb(tile)
            fn = f"{wsi_stem}__r{row:05d}_c{col:05d}__x{x}_y{y}.png"
            tile_path = os.path.join(out_dir, fn)

            ok = cv2.imwrite(tile_path, cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR))
            if not ok or (not os.path.exists(tile_path)):
                print(f"[TILE][WARN] Failed to write tile: {tile_path} (ok={ok}). Skipping.")
                col += 1
                continue

            tile_metas.append(TileMeta(
                path=tile_path,
                rel_path=fn,
                row=row,
                col=col,
                x=x,
                y=y,
                w=tile_w,
                h=tile_h,
            ))
            col += 1
        row += 1

    print(f"[TILE] Wrote {len(tile_metas)} tiles to {out_dir}")
    return tile_metas


# ============================================================
# STEP 2 – SIMPLE COLOR-BASED SORTING
# ============================================================

def is_plaque_candidate_tile(tile_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    mask_sat = s >= MIN_SAT
    mask_brown = (h >= 5) & (h <= 30) & mask_sat
    frac_brown = mask_brown.sum() / float(tile_bgr.size / 3)
    return frac_brown >= MIN_BROWN_FRACTION

def get_wsi_thumbnail_bgr_and_scale(
    wsi_path: str,
    thumb_max_side: int,
) -> Tuple[np.ndarray, float, float, int, int]:
    """
    Return (thumb_bgr, sx, sy, w0, h0) where:
      - thumb_bgr is a downsampled BGR image of the WSI
      - sx = w0 / thumb_width, sy = h0 / thumb_height map level-0 coords -> thumb coords
    Works with OpenSlide when available; falls back to tifffile pyramid/first page.
    """
    # 1) Prefer OpenSlide (best for SVS and many pyramidal TIFFs)
    if _HAS_OPENSLIDE:
        slide = openslide.OpenSlide(wsi_path)
        w0, h0 = slide.dimensions
        thumb = slide.get_thumbnail((thumb_max_side, thumb_max_side))
        thumb_rgb = np.array(thumb.convert("RGB"))
        thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
        th, tw = thumb_bgr.shape[:2]
        sx = w0 / float(tw)
        sy = h0 / float(th)
        slide.close()
        return thumb_bgr, sx, sy, int(w0), int(h0)

    # 2) Fallback: tifffile (for .tif/.tiff pyramidal WSIs or normal tiffs)
    try:
        with tiff.TiffFile(wsi_path) as tf:
            # Determine level-0 size
            # Prefer first series/page as level 0
            series = tf.series[0]
            # series.levels exists for pyramidal; otherwise use series.asarray()
            w0 = int(series.pages[0].imagewidth)
            h0 = int(series.pages[0].imagelength)

            # Choose the best level (largest that is <= thumb_max_side)
            level_arr = None
            level_w = None
            level_h = None

            levels = getattr(series, "levels", None)
            if levels:
                # levels are ordered large->small
                chosen = levels[-1]
                for lv in levels[::-1]:  # small->large
                    try:
                        lw = int(lv.pages[0].imagewidth)
                        lh = int(lv.pages[0].imagelength)
                    except Exception:
                        continue
                    if max(lw, lh) <= thumb_max_side:
                        chosen = lv
                        break
                arr = chosen.asarray()
                level_arr = arr
                level_w = int(chosen.pages[0].imagewidth)
                level_h = int(chosen.pages[0].imagelength)
            else:
                # Not pyramidal: read full then resize (may be large; use a safe approach)
                arr0 = series.asarray()
                if arr0.ndim == 2:
                    arr0 = np.stack([arr0]*3, axis=-1)
                if arr0.shape[-1] >= 3:
                    arr0 = arr0[..., :3]
                # resize to thumb_max_side
                h, w = arr0.shape[:2]
                scale = float(thumb_max_side) / float(max(w, h))
                if scale < 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    arr0 = cv2.resize(arr0, (new_w, new_h), interpolation=cv2.INTER_AREA)
                level_arr = arr0
                level_h, level_w = level_arr.shape[:2]

            # Normalize dtype -> uint8
            arr = level_arr
            if arr.dtype != np.uint8:
                # percentile scaling to uint8
                arr = arr.astype(np.float32)
                lo = np.percentile(arr, 1)
                hi = np.percentile(arr, 99)
                if hi <= lo:
                    hi = lo + 1.0
                arr = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0
                arr = arr.astype(np.uint8)

            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.shape[-1] >= 3:
                arr = arr[..., :3]

            thumb_rgb = arr
            thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)

            th, tw = thumb_bgr.shape[:2]
            sx = w0 / float(tw)
            sy = h0 / float(th)
            return thumb_bgr, sx, sy, int(w0), int(h0)
    except Exception as e:
        raise RuntimeError(f"Failed to create thumbnail for brownness filter from: {wsi_path} ({e})")


def _brown_fraction_bgr(img_bgr: np.ndarray) -> float:
    """Return fraction of pixels that look 'brown' using HSV hue window + saturation."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    mask_sat = s >= POST_BROWN_MIN_SAT
    mask_brown = (h >= POST_BROWN_H_LO) & (h <= POST_BROWN_H_HI) & mask_sat
    return float(mask_brown.sum() / (img_bgr.shape[0] * img_bgr.shape[1]))


def filter_cover40_boxes_by_brownness_thumbnail(
    wsi_path: str,
    boxes: List[Box],
) -> Tuple[List[Box], pd.DataFrame]:
    """
    Scores each COVER40 box on a thumbnail and filters out low-brown boxes.
    Works even without OpenSlide (tifffile fallback).
    Returns (kept_boxes, score_df with brown_fraction per box).
    """
    if not boxes:
        return boxes, pd.DataFrame()

    thumb_bgr, sx, sy, w0, h0 = get_wsi_thumbnail_bgr_and_scale(wsi_path, POST_BROWN_THUMB_MAX_SIDE)
    th, tw = thumb_bgr.shape[:2]

    # Tissue mask on thumbnail (True=tissue) for box-level gating
    gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
    tissue_mask = gray < int(POST_TISSUE_GRAY_THRESH)

    kept: List[Box] = []
    rows = []

    for i, b in enumerate(boxes):
        # Map level-0 coords -> thumb coords
        x0 = int(max(0, min(tw - 1, b.x0 / sx)))
        y0 = int(max(0, min(th - 1, b.y0 / sy)))
        x1 = int(max(0, min(tw,     b.x1 / sx)))
        y1 = int(max(0, min(th,     b.y1 / sy)))

        if x1 <= x0 or y1 <= y0:
            rows.append({"idx": i, "score": b.score, "brown_fraction": 0.0, "kept": False, "reason": "bad_crop"})
            continue

        patch = thumb_bgr[y0:y1, x0:x1]
        bf = _brown_fraction_bgr(patch)
        tf = float(tissue_mask[y0:y1, x0:x1].mean())

        # Decision logic:
        # - If requiring brownness for all: keep only if bf>=threshold OR score>=very-high keep threshold
        # - Else: keep high-conf always, filter low/mid by brownness
        if POST_BROWN_REQUIRE_FOR_ALL:
            # thresholds
            min_brown = float(POST_BROWN_MIN_FRACTION)
            min_tissue = float(POST_BOX_MIN_TISSUE_FRACTION)

            # Stricter for very large boxes (edge/background unions)
            area0 = float((b.x1 - b.x0) * (b.y1 - b.y0))
            if area0 >= float(POST_LARGE_BOX_AREA_PIX):
                min_brown = max(min_brown, float(POST_LARGE_BOX_MIN_BROWN_FRACTION))
                min_tissue = max(min_tissue, float(POST_LARGE_BOX_MIN_TISSUE_FRACTION))

            pass_brown = bf >= min_brown
            pass_tissue = tf >= min_tissue

            if (pass_brown and pass_tissue):
                keep = True
                reason = "brown+tissue_pass"
            elif (float(b.score) >= float(POST_BROWN_KEEP_IF_SCORE_GE)) and pass_tissue:
                keep = True
                reason = "score_keep+tissue_pass"
            else:
                keep = False
                reason = "tissue_fail" if not pass_tissue else "brown_fail"
        else:
            # If not requiring brownness for all, still require tissue to avoid glass edge artifacts.
            min_tissue = float(POST_BOX_MIN_TISSUE_FRACTION)
            area0 = float((b.x1 - b.x0) * (b.y1 - b.y0))
            if area0 >= float(POST_LARGE_BOX_AREA_PIX):
                min_tissue = max(min_tissue, float(POST_LARGE_BOX_MIN_TISSUE_FRACTION))

            pass_tissue = tf >= min_tissue

            if float(b.score) >= float(POST_BROWN_KEEP_IF_SCORE_GE):
                keep = bool(pass_tissue)
                reason = "score_keep+tissue_pass" if keep else "score_keep+tissue_fail"
            else:
                keep = (bf >= float(POST_BROWN_MIN_FRACTION)) and pass_tissue
                reason = "brown+tissue_pass" if keep else ("tissue_fail" if not pass_tissue else "brown_fail")

        if keep:
            kept.append(b)

        rows.append({"idx": i, "score": b.score, "brown_fraction": bf, "tissue_fraction": tf, "kept": keep, "reason": reason})

    df = pd.DataFrame(rows)
    return kept, df



def sort_tiles_by_plaque_content(tile_metas: List[TileMeta], out_root: str) -> Tuple[List[TileMeta], List[TileMeta]]:
    cand_dir = os.path.join(out_root, "plaque_candidates")
    neg_dir = os.path.join(out_root, "plaque_negatives")
    ensure_dir(cand_dir)
    ensure_dir(neg_dir)

    with_paths: List[TileMeta] = []
    without_paths: List[TileMeta] = []

    print("[SORT] Sorting tiles by color heuristic...")
    for meta in tqdm(tile_metas, desc="Sort tiles"):
        tile_bgr = cv2.imread(meta.path, cv2.IMREAD_COLOR)
        if tile_bgr is None:
            continue

        if is_plaque_candidate_tile(tile_bgr):
            dst = os.path.join(cand_dir, meta.rel_path)
            cv2.imwrite(dst, tile_bgr)
            with_paths.append(TileMeta(**{**meta.__dict__, "path": dst}))
        else:
            dst = os.path.join(neg_dir, meta.rel_path)
            cv2.imwrite(dst, tile_bgr)
            without_paths.append(TileMeta(**{**meta.__dict__, "path": dst}))

    print(f"[SORT] {len(with_paths)} candidate tiles, {len(without_paths)} negatives.")
    return with_paths, without_paths


# ============================================================
# STEP 3 – YOLO PREDICTION
# ============================================================

def load_yolo_model() -> Optional["YOLO"]:
    if not _HAS_YOLO:
        print("[YOLO] ultralytics not available.")
        return None
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"[YOLO] Model not found: {YOLO_MODEL_PATH}")
        return None
    print(f"[YOLO] Loading model: {YOLO_MODEL_PATH}")
    return YOLO(YOLO_MODEL_PATH)


def run_yolo_on_tiles(
    model: "YOLO",
    with_paths: List[TileMeta],
    tile_meta_by_name: Dict[str, TileMeta],
    annot_dir: str,
    plaque_csv_path: str,
) -> List[Box]:
    ensure_dir(annot_dir)
    detections: List[Box] = []
    rows_for_csv: List[Dict[str, Any]] = []

    print(f"[PREDICT] Running YOLO on {len(with_paths)} candidate tiles.")
    idx_counter = 0

    for meta in tqdm(with_paths, desc="YOLO"):
        tile_path = meta.path
        fn = meta.rel_path
        tile_meta = tile_meta_by_name.get(fn, meta)

        res = model(tile_path, verbose=False, conf=CONF_THRES, iou=TILE_NMS_IOU_THRES)[0]

        # Save annotated tile (optional debug)
        annot_path = os.path.join(annot_dir, fn)
        plotted = res.plot()
        if plotted is not None:
            cv2.imwrite(annot_path, plotted)

        if res.boxes is None or len(res.boxes) == 0:
            continue

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        for (x0, y0, x1, y1), score, cls_id in zip(boxes_xyxy, scores, cls_ids):
            if float(score) <= CONF_THRES:
                continue

            gx0 = int(tile_meta.x + x0)
            gy0 = int(tile_meta.y + y0)
            gx1 = int(tile_meta.x + x1)
            gy1 = int(tile_meta.y + y1)

            bw = border_weight_from_local_box(x0, y0, x1, y1, tile_meta.w, tile_meta.h)

            detections.append(Box(
                x0=gx0, y0=gy0, x1=gx1, y1=gy1,
                score=float(score), cls=int(cls_id),
                tile_row=tile_meta.row, tile_col=tile_meta.col,
                idx=idx_counter,
                support=1,
                border_weight=float(bw),
            ))

            rows_for_csv.append({
                "global_x_min": gx0,
                "global_y_min": gy0,
                "global_x_max": gx1,
                "global_y_max": gy1,
                "score": float(score),
                "border_weight": float(bw),
                "effective_score": float(score) * float(bw),
                "class": int(cls_id),
                "tile_row": tile_meta.row,
                "tile_col": tile_meta.col,
                "tile_rel_path": tile_meta.rel_path,
                "detection_idx": idx_counter,
            })
            idx_counter += 1

    pd.DataFrame(rows_for_csv).to_csv(plaque_csv_path, index=False)
    print(f"[PREDICT] Saved {len(detections)} detections -> {plaque_csv_path}")
    return detections


# ============================================================
# STEP 4 – GATHER MERGE INPUTS
# ============================================================

def gather_merge_inputs(wsi_out_dir: str, with_paths: List[TileMeta], without_paths: List[TileMeta], annot_dir: str) -> Tuple[str, str]:
    merge_raw = os.path.join(wsi_out_dir, "merge_input_raw")
    merge_annot = os.path.join(wsi_out_dir, "merge_input_annotated")
    ensure_dir(merge_raw)
    ensure_dir(merge_annot)

    print("[GATHER] Collecting raw tiles...")
    for meta in tqdm(with_paths + without_paths, desc="Raw tiles"):
        dst = os.path.join(merge_raw, meta.rel_path)
        tile_bgr = cv2.imread(meta.path, cv2.IMREAD_COLOR)
        if tile_bgr is not None:
            cv2.imwrite(dst, tile_bgr)

    print("[GATHER] Collecting annotated tiles...")
    for meta in tqdm(with_paths + without_paths, desc="Annotated tiles"):
        annot_tile_path = os.path.join(annot_dir, meta.rel_path)
        src = annot_tile_path if os.path.exists(annot_tile_path) else meta.path
        dst = os.path.join(merge_annot, meta.rel_path)
        tile_bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if tile_bgr is not None:
            cv2.imwrite(dst, tile_bgr)

    return merge_raw, merge_annot


# ============================================================
# STEP 5 – MERGE TILES BACK INTO A CANVAS (OVERLAP SAFE)
# ============================================================

def _feather_window(h: int, w: int, border: int = 64) -> np.ndarray:
    border = int(max(8, min(border, min(h, w) // 4)))
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    ramp_y = np.linspace(0, 1, border, dtype=np.float32)
    ramp_x = np.linspace(0, 1, border, dtype=np.float32)

    wy[:border] = ramp_y
    wy[-border:] = ramp_y[::-1]
    wx[:border] = ramp_x
    wx[-border:] = ramp_x[::-1]

    win = wy[:, None] * wx[None, :]
    win = np.clip(win, 1e-3, 1.0)
    return win


def merge_tiles_to_canvas_overlap(tiles_dir: str, out_tiff_path: Optional[str], tile_stride: int) -> np.ndarray:
    """
    Merge tiles back into a full-resolution canvas.

    Uses BLEND when feasible; falls back to OVERWRITE if blending would be too large.
    Returns canvas_bgr uint8 (memmap-backed).
    """
    files = [f for f in os.listdir(tiles_dir) if f.lower().endswith(".png")]
    if not files:
        raise RuntimeError(f"No .png tiles in {tiles_dir}")

    max_x1 = 0
    max_y1 = 0
    meta_xy: Dict[str, Tuple[int, int]] = {}

    for fn in files:
        base = os.path.splitext(fn)[0]
        if "__x" in base and "_y" in base:
            x_part = base.split("__x")[-1]
            x_str, y_str = x_part.split("_y")
            x = int(x_str)
            y = int(y_str)
        else:
            parts = base.split("_")
            r_str = next(p for p in parts if p.startswith("r"))
            c_str = next(p for p in parts if p.startswith("c"))
            r = int(r_str[1:])
            c = int(c_str[1:])
            x = c * tile_stride
            y = r * tile_stride

        meta_xy[fn] = (x, y)
        max_x1 = max(max_x1, x + TILE_SIZE)
        max_y1 = max(max_y1, y + TILE_SIZE)

    canvas_h = int(max_y1)
    canvas_w = int(max_x1)

    est_blend_bytes = canvas_h * canvas_w * (3 * 4 + 4)
    BLEND_MAX_BYTES = int(24 * (1024**3))
    do_blend = est_blend_bytes <= BLEND_MAX_BYTES

    print(f"[MERGE] Canvas {canvas_w} x {canvas_h} | tiles={len(files)} | mode={'BLEND' if do_blend else 'OVERWRITE'}")

    tmp_canvas_path = os.path.join(os.path.dirname(tiles_dir), f"._tmp_canvas_{os.getpid()}.dat")
    canvas = np.memmap(tmp_canvas_path, dtype=np.uint8, mode="w+", shape=(canvas_h, canvas_w, 3))
    canvas[:] = 0

    if do_blend:
        tmp_acc_path = os.path.join(os.path.dirname(tiles_dir), f"._tmp_acc_{os.getpid()}.dat")
        tmp_wsum_path = os.path.join(os.path.dirname(tiles_dir), f"._tmp_wsum_{os.getpid()}.dat")

        acc = np.memmap(tmp_acc_path, dtype=np.float16, mode="w+", shape=(canvas_h, canvas_w, 3))
        wsum = np.memmap(tmp_wsum_path, dtype=np.float16, mode="w+", shape=(canvas_h, canvas_w))
        acc[:] = 0
        wsum[:] = 0

        win = _feather_window(TILE_SIZE, TILE_SIZE, border=64).astype(np.float16)

        for fn in tqdm(files, desc=f"Merge {os.path.basename(tiles_dir)}"):
            x, y = meta_xy[fn]
            tile_path = os.path.join(tiles_dir, fn)
            tile_bgr = cv2.imread(tile_path, cv2.IMREAD_COLOR)
            if tile_bgr is None:
                continue
            h, w = tile_bgr.shape[:2]
            win_hw = win[:h, :w]
            tile_f = tile_bgr.astype(np.float16)

            acc[y:y+h, x:x+w, :] += tile_f * win_hw[:, :, None]
            wsum[y:y+h, x:x+w] += win_hw

        chunk = 2048
        for y0 in tqdm(range(0, canvas_h, chunk), desc="Normalize"):
            y1 = min(canvas_h, y0 + chunk)
            wsum_chunk = np.maximum(wsum[y0:y1, :].astype(np.float32), 1e-6)
            acc_chunk = acc[y0:y1, :, :].astype(np.float32)
            out = (acc_chunk / wsum_chunk[:, :, None]).clip(0, 255).astype(np.uint8)
            canvas[y0:y1, :, :] = out

        try:
            del acc, wsum
            os.remove(tmp_acc_path)
            os.remove(tmp_wsum_path)
        except Exception:
            pass
    else:
        for fn in tqdm(files, desc=f"Merge {os.path.basename(tiles_dir)}"):
            x, y = meta_xy[fn]
            tile_path = os.path.join(tiles_dir, fn)
            tile_bgr = cv2.imread(tile_path, cv2.IMREAD_COLOR)
            if tile_bgr is None:
                continue
            h, w = tile_bgr.shape[:2]
            canvas[y:y+h, x:x+w, :] = tile_bgr

    if out_tiff_path:
        print(f"[MERGE] Writing lossless BigTIFF: {out_tiff_path}")
        tiff.imwrite(
            out_tiff_path,
            cv2.cvtColor(np.asarray(canvas), cv2.COLOR_BGR2RGB),
            compression="lzw",
            bigtiff=True,
            photometric="rgb",
        )

    return canvas


# ============================================================
# STEP 6 – WSI-LEVEL POST-PROCESSING
# ============================================================

def box_area(b: Box) -> float:
    return float(max(0, b.x1 - b.x0) * max(0, b.y1 - b.y0))


def iou_xyxy(a: Box, b: Box) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = box_area(a)
    ub = box_area(b)
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0


def containment_fraction(inner: Box, outer: Box) -> float:
    ix0 = max(inner.x0, outer.x0)
    iy0 = max(inner.y0, outer.y0)
    ix1 = min(inner.x1, outer.x1)
    iy1 = min(inner.y1, outer.y1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    a_in = box_area(inner)
    return float(inter / a_in) if a_in > 0 else 0.0


def remove_contained_boxes(boxes: List[Box], frac: float = 0.90) -> List[Box]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (box_area(b), effective_det_score(b)), reverse=True)
    keep: List[Box] = []
    for b in boxes:
        drop = False
        for k in keep:
            if containment_fraction(b, k) >= frac:
                drop = True
                break
        if not drop:
            keep.append(b)
    return keep


def soft_nms(boxes: List[Box]) -> List[Box]:
    if not boxes:
        return []
    boxes = [Box(**b.__dict__) for b in boxes]
    boxes.sort(key=lambda b: effective_det_score(b), reverse=True)
    out: List[Box] = []

    while boxes:
        best = boxes.pop(0)
        out.append(best)

        new_boxes = []
        for b in boxes:
            ov = iou_xyxy(best, b)
            if ov >= SOFTNMS_IOU_THRESH:
                if SOFTNMS_METHOD == "linear":
                    b.score = b.score * (1.0 - ov)
                else:
                    b.score = b.score * math.exp(-(ov * ov) / SOFTNMS_SIGMA)
            if effective_det_score(b) >= SOFTNMS_SCORE_THRESH:
                new_boxes.append(b)

        new_boxes.sort(key=lambda b: effective_det_score(b), reverse=True)
        boxes = new_boxes

    return out


def cluster_boxes_wsi(boxes: List[Box], iou_thresh: float) -> List[List[int]]:
    n = len(boxes)
    if n == 0:
        return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if iou_xyxy(boxes[i], boxes[j]) >= iou_thresh:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        clusters.append(comp)
    return clusters


def wbf_fuse_cluster(boxes: List[Box], idxs: List[int]) -> Box:
    ws = []
    xs0 = []
    ys0 = []
    xs1 = []
    ys1 = []
    cls = boxes[idxs[0]].cls

    support_sum = 0
    score_mean = 0.0
    for i in idxs:
        b = boxes[i]
        wgt = float(effective_det_score(b) + SUPPORT_LAMBDA * b.support)
        ws.append(wgt)
        xs0.append(b.x0 * wgt)
        ys0.append(b.y0 * wgt)
        xs1.append(b.x1 * wgt)
        ys1.append(b.y1 * wgt)
        support_sum += int(b.support)
        score_mean += float(effective_det_score(b))

    W = float(sum(ws)) if sum(ws) > 0 else 1.0
    x0 = int(round(sum(xs0) / W))
    y0 = int(round(sum(ys0) / W))
    x1 = int(round(sum(xs1) / W))
    y1 = int(round(sum(ys1) / W))

    score_mean = score_mean / max(1, len(idxs))

    return Box(
        x0=x0, y0=y0, x1=x1, y1=y1,
        score=float(score_mean),
        cls=int(cls),
        tile_row=-1, tile_col=-1,
        idx=-1,
        support=int(max(1, support_sum)),
        border_weight=1.0,
    )


def wsi_postprocess_boxes(boxes: List[Box]) -> List[Box]:
    if not boxes:
        return []

    n = len(boxes)
    for i in range(n):
        sup = 1
        for j in range(n):
            if i == j:
                continue
            if iou_xyxy(boxes[i], boxes[j]) >= WSI_CLUSTER_IOU:
                sup += 1
        boxes[i].support = sup

    clusters = cluster_boxes_wsi(boxes, iou_thresh=WSI_CLUSTER_IOU)
    fused: List[Box] = [wbf_fuse_cluster(boxes, comp) for comp in clusters]

    fused = remove_contained_boxes(fused, frac=CONTAINMENT_FRAC)

    if WSI_METHOD == "softnms":
        fused = soft_nms(fused)
        fused = remove_contained_boxes(fused, frac=CONTAINMENT_FRAC)

    fused.sort(key=lambda b: (effective_det_score(b) + SUPPORT_LAMBDA * b.support), reverse=True)
    return fused


def write_boxes_csv(boxes: List[Box], path: str):
    rows = []
    for i, b in enumerate(boxes):
        rows.append({
            "global_x_min": b.x0,
            "global_y_min": b.y0,
            "global_x_max": b.x1,
            "global_y_max": b.y1,
            "score": b.score,
            "border_weight": getattr(b, "border_weight", 1.0),
            "effective_score": effective_det_score(b),
            "class": b.cls,
            "support": b.support,
            "idx": i,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# COVER40 MERGE RULE
# ============================================================

def _intersection_area(a: Box, b: Box) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    return float(iw * ih)


def cover_fraction_of_smaller(a: Box, b: Box) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    aa = box_area(a)
    ab = box_area(b)
    small = min(aa, ab)
    if small <= 0:
        return 0.0
    return float(inter / small)


def merge_boxes_by_cover40_union(boxes: List[Box], thresh: float = 0.40) -> List[Box]:
    if not boxes:
        return []

    n = len(boxes)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if cover_fraction_of_smaller(boxes[i], boxes[j]) >= thresh:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    out: List[Box] = []
    new_idx = 0

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)

        if len(comp) == 1:
            b = boxes[comp[0]]
            out.append(Box(
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                score=b.score, cls=b.cls,
                tile_row=b.tile_row, tile_col=b.tile_col,
                idx=new_idx,
                support=b.support,
            ))
            new_idx += 1
            continue

        x0 = min(boxes[k].x0 for k in comp)
        y0 = min(boxes[k].y0 for k in comp)
        x1 = max(boxes[k].x1 for k in comp)
        y1 = max(boxes[k].y1 for k in comp)

        scores = [float(boxes[k].score) for k in comp]
        score = float(max(scores)) if COVER40_SCORE_USE_MAX else float(sum(scores) / max(1, len(scores)))

        cls = int(boxes[comp[0]].cls)
        support_sum = int(max(1, sum(int(boxes[k].support) for k in comp)))

        out.append(Box(
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            score=score, cls=cls,
            tile_row=-1, tile_col=-1,
            idx=new_idx,
            support=support_sum,
        ))
        new_idx += 1

    out.sort(key=lambda b: b.score, reverse=True)
    return out


# ============================================================
# OVERLAY + MICRONS
# ============================================================

def draw_blue_boxes(canvas_bgr: np.ndarray, boxes: List[Box], thickness: int = 2, font_scale: float = 0.5):
    if canvas_bgr is None or not boxes:
        return
    h, w = canvas_bgr.shape[:2]
    for b in boxes:
        x0 = max(0, min(w - 1, int(b.x0)))
        y0 = max(0, min(h - 1, int(b.y0)))
        x1 = max(0, min(w - 1, int(b.x1)))
        y1 = max(0, min(h - 1, int(b.y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(canvas_bgr, (x0, y0), (x1, y1), (255, 0, 0), thickness)

        label = f"plaque {float(b.score):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y_text = max(0, y0 - 2)
        y_bg0 = max(0, y_text - th - 6)
        x_bg1 = min(w - 1, x0 + tw + 6)
        cv2.rectangle(canvas_bgr, (x0, y_bg0), (x_bg1, y_text), (255, 0, 0), -1)
        cv2.putText(canvas_bgr, label, (x0 + 3, max(0, y_text - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def boxes_df_to_microns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["width_px"] = (df["global_x_max"] - df["global_x_min"]).clip(lower=0)
    df["height_px"] = (df["global_y_max"] - df["global_y_min"]).clip(lower=0)
    df["area_px2"] = df["width_px"] * df["height_px"]

    df["x_min_um"] = df["global_x_min"] * UM_PER_PIXEL
    df["y_min_um"] = df["global_y_min"] * UM_PER_PIXEL
    df["x_max_um"] = df["global_x_max"] * UM_PER_PIXEL
    df["y_max_um"] = df["global_y_max"] * UM_PER_PIXEL

    df["width_um"] = df["width_px"] * UM_PER_PIXEL
    df["height_um"] = df["height_px"] * UM_PER_PIXEL
    df["area_um2"] = df["area_px2"] * (UM_PER_PIXEL ** 2)

    df["ecd_px"] = 2.0 * np.sqrt(df["area_px2"] / math.pi)
    df["ecd_um"] = df["ecd_px"] * UM_PER_PIXEL

    df["diameter_px"] = np.maximum(df["width_px"], df["height_px"])
    df["diameter_um"] = df["diameter_px"] * UM_PER_PIXEL
    return df


def stats_microns(df_um: pd.DataFrame) -> pd.DataFrame:
    if df_um is None or df_um.empty:
        return pd.DataFrame([{
            "n_boxes": 0,
            "sum_area_um2": 0.0,
            "union_area_um2": np.nan,
            "overlap_area_um2": np.nan,
            "pct_overlap": np.nan,
            "area_um2_mean": np.nan,
            "area_um2_median": np.nan,
            "ecd_um_mean": np.nan,
            "ecd_um_median": np.nan,
            "diameter_um_mean": np.nan,
            "diameter_um_median": np.nan,
        }])

    sum_area = float(df_um["area_um2"].sum())
    union_area = np.nan
    overlap = np.nan
    pct = np.nan

    if _HAS_SHAPELY:
        geoms = [shapely_box(r.x_min_um, r.y_min_um, r.x_max_um, r.y_max_um) for r in df_um.itertuples()]
        union_area = float(unary_union(geoms).area)
        overlap = max(0.0, sum_area - union_area)
        pct = (overlap / sum_area * 100.0) if sum_area > 0 else 0.0

    return pd.DataFrame([{
        "n_boxes": int(len(df_um)),
        "sum_area_um2": sum_area,
        "union_area_um2": union_area,
        "overlap_area_um2": overlap,
        "pct_overlap": pct,
        "area_um2_mean": float(df_um["area_um2"].mean()),
        "area_um2_median": float(df_um["area_um2"].median()),
        "ecd_um_mean": float(df_um["ecd_um"].mean()),
        "ecd_um_median": float(df_um["ecd_um"].median()),
        "diameter_um_mean": float(df_um["diameter_um"].mean()),
        "diameter_um_median": float(df_um["diameter_um"].median()),
    }])


def save_hist_area(df_um: pd.DataFrame, out_png: str, title: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    plt.figure()
    if df_um is not None and not df_um.empty:
        plt.hist(df_um["area_um2"].values, bins=30)
    plt.title(title)
    plt.xlabel("Area (µm²)")
    plt.ylabel("Count")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def make_side_by_side_tif_blue_only(wsi_path: str, df_cover40: pd.DataFrame, out_tif: str) -> None:
    """
    side_by_side.tif but BLUE-only (single panel).
    Kept name for compatibility with your workflow.
    """
    if not _HAS_OPENSLIDE:
        print("[OVERLAY] openslide not available; skipping side_by_side.tif")
        return
    if df_cover40 is None or df_cover40.empty:
        print("[OVERLAY] no cover40 boxes; skipping side_by_side.tif")
        return

    slide = openslide.OpenSlide(wsi_path)
    w0, h0 = slide.dimensions

    thumb = slide.get_thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))  # PIL
    img = np.array(thumb.convert("RGB"))
    th, tw = img.shape[:2]
    sx = w0 / float(tw)
    sy = h0 / float(th)

    df_draw = df_cover40
    if len(df_draw) > MAX_PLAQUE_BOXES_TO_DRAW:
        if DRAW_TOP_BY_SCORE and ("score" in df_draw.columns):
            df_draw = df_draw.sort_values("score", ascending=False).head(MAX_PLAQUE_BOXES_TO_DRAW)
        else:
            df_draw = df_draw.sample(MAX_PLAQUE_BOXES_TO_DRAW, random_state=0)

    blue = img.copy()
    for r in df_draw.itertuples():
        x0 = int(r.global_x_min / sx); y0 = int(r.global_y_min / sy)
        x1 = int(r.global_x_max / sx); y1 = int(r.global_y_max / sy)
        cv2.rectangle(blue, (x0, y0), (x1, y1), (255, 0, 0), 1)

    ensure_dir(os.path.dirname(out_tif))
    cv2.imwrite(out_tif, cv2.cvtColor(blue, cv2.COLOR_RGB2BGR))
    slide.close()


def postprocess_cover40_to_microns_and_overlay(wsi_path: str, wsi_out_dir: str, wsi_stem: str) -> Dict[str, Any]:
    """Create the final 2-sheet Excel workbook + plots + side_by_side.tif using COVER40 as plaque_locations."""
    ensure_dir(MICRONS_OUTPUT_DIR)

    cover40_csv = os.path.join(wsi_out_dir, "plaque_locations_cover40_boxes.csv")
    if not os.path.exists(cover40_csv):
        return {"enabled": False, "reason": "missing_cover40_csv", "path": cover40_csv}

    df_cover40 = pd.read_csv(cover40_csv)
    df_loc_um = boxes_df_to_microns(df_cover40)
    stats_raw = stats_microns(df_loc_um)

    final_excel = os.path.join(wsi_out_dir, f"{wsi_stem}_Default_Extended_plaque_results.xlsx")
    with pd.ExcelWriter(final_excel) as writer:
        df_loc_um.to_excel(writer, sheet_name="plaque_locations_um", index=False)
        stats_raw.to_excel(writer, sheet_name="stats_raw_microns", index=False)

    microns_xlsx = os.path.join(MICRONS_OUTPUT_DIR, f"{wsi_stem}_Default_Extended_plaque_results_MICRONS.xlsx")
    with pd.ExcelWriter(microns_xlsx) as writer:
        df_loc_um.to_excel(writer, sheet_name="plaque_locations_um", index=False)
        stats_raw.to_excel(writer, sheet_name="stats_raw_microns", index=False)

    plots_dir = os.path.join(MICRONS_OUTPUT_DIR, "plots", f"{wsi_stem}_Default_Extended_plaque_results")
    save_hist_area(df_loc_um, os.path.join(plots_dir, "plaque_size_hist_raw.png"), f"{wsi_stem}: Cover40 plaque areas")

    overlay_dir = os.path.join(MICRONS_OUTPUT_DIR, "overlays", f"{wsi_stem}_Default_Extended_plaque_results")
    out_tif = os.path.join(overlay_dir, "side_by_side.tif")  # kept name
    make_side_by_side_tif_blue_only(wsi_path, df_cover40, out_tif)

    return {
        "enabled": True,
        "final_excel": final_excel,
        "microns_xlsx": microns_xlsx,
        "plots_dir": plots_dir,
        "side_by_side_tif": out_tif if os.path.exists(out_tif) else None,
        "n_cover40": int(len(df_cover40)),
        "thumb_max_side": THUMB_MAX_SIDE,
        "max_plaque_boxes_drawn": MAX_PLAQUE_BOXES_TO_DRAW,
        "has_shapely": _HAS_SHAPELY,
        "has_openslide": _HAS_OPENSLIDE,
    }


# ============================================================
# PER-WSI PIPELINE
# ============================================================

def process_single_wsi(wsi_path: str, model: Optional["YOLO"]):
    wsi_stem = stem_from_path(wsi_path)
    wsi_out_dir = os.path.join(OUTPUT_ROOT, wsi_stem)
    ensure_dir(wsi_out_dir)

    status = {"wsi": wsi_path, "stem": wsi_stem, "steps": {}}

    tiles_dir = os.path.join(wsi_out_dir, "tiles")
    annot_dir = os.path.join(wsi_out_dir, "annotated_tiles")

    plaque_csv_tile = os.path.join(wsi_out_dir, "plaque_locations_tilelevel.csv")
    plaque_csv_post = os.path.join(wsi_out_dir, "plaque_locations_wsi_postprocessed.csv")

    # 1) Tiling
    tile_metas = tile_wsi_to_pngs(wsi_path, tiles_dir, wsi_stem=wsi_stem)
    status["steps"]["tiling"] = {"n_tiles": len(tile_metas), "tile": TILE_SIZE, "stride": TILE_STRIDE, "overlap": TILE_OVERLAP}

    tile_meta_by_name = {tm.rel_path: tm for tm in tile_metas}

    # 2) Sorting
    with_paths, without_paths = sort_tiles_by_plaque_content(tile_metas=tile_metas, out_root=wsi_out_dir)
    status["steps"]["sorting"] = {"n_candidates": len(with_paths), "n_negatives": len(without_paths)}

    # 3) Prediction (tile-level)
    detections: List[Box] = []
    if model is not None and len(with_paths) > 0:
        detections = run_yolo_on_tiles(
            model=model,
            with_paths=with_paths,
            tile_meta_by_name=tile_meta_by_name,
            annot_dir=annot_dir,
            plaque_csv_path=plaque_csv_tile,
        )
    else:
        print("[WARN] Model not available or no candidate tiles, skipping prediction.")
    status["steps"]["prediction"] = {"n_tile_detections": len(detections)}

    # 3b) WSI-level postprocess
    boxes_for_cover40 = detections
    if WSI_POSTPROCESS and detections:
        print(f"[WSI] Post-processing tile detections: method={WSI_METHOD}")
        post = wsi_postprocess_boxes(detections)
        write_boxes_csv(post, plaque_csv_post)
        status["steps"]["wsi_postprocess"] = {"enabled": True, "method": WSI_METHOD, "n_boxes_after": len(post)}
        if COVER40_SOURCE.lower() == "postprocessed":
            boxes_for_cover40 = post
    else:
        status["steps"]["wsi_postprocess"] = {"enabled": False, "n_boxes_after": len(boxes_for_cover40)}

    # 4) Gather merge inputs
    merge_raw_dir, _merge_annot_dir = gather_merge_inputs(wsi_out_dir, with_paths, without_paths, annot_dir)
    status["steps"]["gather"] = {
        "merge_input_raw": len([x for x in os.listdir(merge_raw_dir) if x.lower().endswith(".png")]),
    }

    # 5) Merge RAW tiles back into a canvas (from ALL tiles, not just candidates)
    merged_raw_tif = os.path.join(wsi_out_dir, f"{wsi_stem}_merged_raw_all_tiles.tif") if SAVE_FULLRES_MERGED_RAW_TIF else None
    canvas_raw = merge_tiles_to_canvas_overlap(merge_raw_dir, out_tiff_path=merged_raw_tif, tile_stride=TILE_STRIDE)
    h, w = canvas_raw.shape[:2]
    status["steps"]["merge"] = {"width": int(w), "height": int(h), "merged_raw_tif": merged_raw_tif}

    # 6) COVER40 overlay + CSV
    if DRAW_BLUE_OVERLAY_COVER40:
        merged_cover40 = merge_boxes_by_cover40_union(boxes_for_cover40, thresh=COVER40_THRESH)


        # Optional post-COVER40 brownness filter (box-level, thumbnail-based)
        if POST_COVER40_BROWN_FILTER and merged_cover40:
            merged_cover40, df_brown = filter_cover40_boxes_by_brownness_thumbnail(wsi_path, merged_cover40)
            brown_csv = os.path.join(wsi_out_dir, "cover40_brownness_scores.csv")
            try:
                df_brown.to_csv(brown_csv, index=False)
                print(f"[BROWN] Post-COVER40 filter kept {len(merged_cover40)} boxes; scores -> {brown_csv}")
            except Exception as e:
                print(f"[BROWN] Failed saving brownness scores CSV: {e}")
        # Draw ONCE on raw canvas (in-place)
        draw_blue_boxes(canvas_raw, merged_cover40, thickness=BLUE_THICKNESS, font_scale=BLUE_FONT_SCALE)

        # Always write COVER40 CSV first (so you get results even if overlay writing is slow)
        cover40_csv = os.path.join(wsi_out_dir, "plaque_locations_cover40_boxes.csv")
        write_boxes_csv(merged_cover40, cover40_csv)

        # Full-res BigTIFF overlay can be VERY slow; default is OFF
        blue_overlay_tiff = os.path.join(wsi_out_dir, f"{wsi_stem}_COVER40_POSTFILTER_overlay.tif")
        if SAVE_FULLRES_COVER40_OVERLAY_TIF:
            print(f"[OVERLAY] Writing FULL BigTIFF overlay (may be slow): {blue_overlay_tiff}", flush=True)
            try:
                tiff.imwrite(
                    blue_overlay_tiff,
                    cv2.cvtColor(np.asarray(canvas_raw), cv2.COLOR_BGR2RGB),
                    compression=COVER40_OVERLAY_COMPRESSION,
                    bigtiff=True,
                    photometric="rgb",
                )
                print("[OVERLAY] Done writing FULL BigTIFF", flush=True)
            except Exception as e:
                print(f"[OVERLAY][WARN] Failed writing FULL BigTIFF: {e}")
        else:
            blue_overlay_tiff = None
            print("[OVERLAY] Skipping full-resolution COVER40 BigTIFF overlay", flush=True)

        # Lightweight thumbnail QC overlay
        thumb_png = None
        if SAVE_COVER40_THUMB_OVERLAY_PNG:
            try:
                hh, ww = canvas_raw.shape[:2]
                scale = float(COVER40_THUMB_SIDE) / float(max(hh, ww))
                if scale < 1.0:
                    small = cv2.resize(canvas_raw, (int(ww * scale), int(hh * scale)), interpolation=cv2.INTER_AREA)
                else:
                    small = canvas_raw
                thumb_png = os.path.join(wsi_out_dir, f"{wsi_stem}_COVER40_POSTFILTER_thumb_overlay.png")
                cv2.imwrite(thumb_png, small)
                print(f"[OVERLAY] Wrote thumbnail overlay: {thumb_png}", flush=True)
            except Exception as e:
                print(f"[OVERLAY][WARN] Failed writing thumbnail overlay: {e}")

        status["steps"]["cover40"] = {
            "enabled": True,
            "thresh": COVER40_THRESH,
            "source": COVER40_SOURCE,
            "border_aware_wbf": BORDER_AWARE_WBF,
            "edge_margin_outer_px": EDGE_MARGIN_OUTER_PX,
            "edge_margin_inner_px": EDGE_MARGIN_INNER_PX,
            "n_before": int(len(boxes_for_cover40)),
            "n_after": int(len(merged_cover40)),
            "csv": cover40_csv,
            "overlay": blue_overlay_tiff,
        }
    else:
        status["steps"]["cover40"] = {"enabled": False}

    # 7) Final microns outputs + BLUE-only side_by_side.tif + final 2-sheet Excel workbook
    try:
        post_info = postprocess_cover40_to_microns_and_overlay(wsi_path, wsi_out_dir, wsi_stem)
        status["steps"]["microns_outputs"] = post_info
        if post_info.get("enabled"):
            print(f"[EXCEL] Wrote: {post_info.get('final_excel')}" )
            print(f"[MICRONS] Wrote: {post_info.get('microns_xlsx')}" )
            if post_info.get("side_by_side_tif"):
                print(f"[OVERLAY] side_by_side.tif (BLUE-only): {post_info.get('side_by_side_tif')}")
    except Exception as e:
        print(f"[MICRONS][WARN] Failed microns/overlay step: {e}")
        status["steps"]["microns_outputs"] = {"enabled": False, "error": str(e)}

    status_path = os.path.join(wsi_out_dir, "status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"[STATUS] Saved: {status_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="All-in-one WSI plaque pipeline (COVER40 + post-COVER40 brownness filter). "
                    "Run on a single WSI or on all WSIs in a folder."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--wsi",
        type=str,
        default=None,
        help="Path to a single WSI file (.tif/.tiff/.svs). If provided, runs single-WSI mode.",
    )
    group.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Folder containing WSIs. If provided, runs folder mode over all WSIs in this directory.",
    )

    # Optional convenience overrides (keeps backwards compatibility with your CONFIG defaults)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Override OUTPUT_ROOT (where per-WSI outputs go). Defaults to OUTPUT_ROOT in the script.",
    )
    parser.add_argument(
        "--microns_output_dir",
        type=str,
        default=None,
        help="Override MICRONS_OUTPUT_DIR. Defaults to MICRONS_OUTPUT_DIR in the script.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model weights. Defaults to models/best.pt in the repository.",
    )
    parser.add_argument(
        "--um_per_pixel",
        type=float,
        default=None,
        help="Pixel size in micrometers/pixel used for physical measurements. Default: 0.40.",
    )

    args = parser.parse_args()

    # Apply optional overrides
    global OUTPUT_ROOT, MICRONS_OUTPUT_DIR, YOLO_MODEL_PATH, UM_PER_PIXEL
    if args.output_root:
        OUTPUT_ROOT = args.output_root
    if args.microns_output_dir:
        MICRONS_OUTPUT_DIR = args.microns_output_dir
    if args.model:
        YOLO_MODEL_PATH = args.model
    if args.um_per_pixel is not None:
        UM_PER_PIXEL = float(args.um_per_pixel)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(MICRONS_OUTPUT_DIR)

    model = load_yolo_model()

    # --- Mode selection ---
    if args.wsi is not None:
        wsi_path = args.wsi
        if not os.path.exists(wsi_path):
            print(f"[MAIN] WSI not found: {wsi_path}")
            return

        wsi_stem = stem_from_path(wsi_path)
        lock_path = acquire_wsi_lock(OUTPUT_ROOT, wsi_stem)
        if lock_path is None:
            print(f"[LOCK] Already locked: {wsi_stem}")
            return

        try:
            print("[MAIN] Single-WSI mode (--wsi)")
            print(f"[MAIN] Processing WSI: {wsi_path}")
            try:
                process_single_wsi(wsi_path, model=model)
            except Exception as e:
                print(f"[MAIN][ERROR] WSI failed: {wsi_path} -> {e}")
                err = traceback.format_exc()
                try:
                    ensure_dir(os.path.join(OUTPUT_ROOT, wsi_stem))
                    status_path = os.path.join(OUTPUT_ROOT, wsi_stem, "status.json")
                    with open(status_path, "w") as f:
                        json.dump({"wsi": wsi_path, "stem": wsi_stem, "error": str(e), "traceback": err}, f, indent=2)
                    print(f"[STATUS] Saved error status: {status_path}")
                except Exception as _e2:
                    print(f"[STATUS][WARN] Could not write error status: {_e2}")
                raise

            print("[MAIN] Done.")
        finally:
            release_wsi_lock(lock_path)
            print(f"[LOCK] Released: {wsi_stem}")
        return

    # Folder mode via CLI, else fall back to legacy CONFIG flags
    if args.input_dir is not None:
        input_dir = args.input_dir
        wsis = list_wsis(input_dir)
        if not wsis:
            print(f"[MAIN] No WSIs found in {input_dir}")
            return

        print(f"[MAIN] Found {len(wsis)} WSIs in {input_dir}.")
        for wsi_path in wsis:
            wsi_stem = stem_from_path(wsi_path)

            lock_path = acquire_wsi_lock(OUTPUT_ROOT, wsi_stem)
            if lock_path is None:
                print(f"[LOCK] Skipping (already locked): {wsi_stem}")
                continue

            try:
                print("=" * 80)
                print(f"[MAIN] Processing WSI: {wsi_path}")
                try:
                    process_single_wsi(wsi_path, model=model)
                except Exception as e:
                    print(f"[MAIN][ERROR] WSI failed (skipping): {wsi_path} -> {e}")
                    err = traceback.format_exc()
                    try:
                        ensure_dir(os.path.join(OUTPUT_ROOT, wsi_stem))
                        status_path = os.path.join(OUTPUT_ROOT, wsi_stem, "status.json")
                        with open(status_path, "w") as f:
                            json.dump({"wsi": wsi_path, "stem": wsi_stem, "error": str(e), "traceback": err}, f, indent=2)
                        print(f"[STATUS] Saved error status: {status_path}")
                    except Exception as _e2:
                        print(f"[STATUS][WARN] Could not write error status: {_e2}")

            finally:
                release_wsi_lock(lock_path)
                print(f"[LOCK] Released: {wsi_stem}")

        print("[MAIN] Done.")
        return

    # --- Legacy behavior (no CLI flags) ---
    if RUN_SINGLE_WSI_ONLY:
        wsi_path = SINGLE_WSI_PATH
        if not os.path.exists(wsi_path):
            print(f"[MAIN] WSI not found: {wsi_path}")
            return

        wsi_stem = stem_from_path(wsi_path)
        lock_path = acquire_wsi_lock(OUTPUT_ROOT, wsi_stem)
        if lock_path is None:
            print(f"[LOCK] Already locked: {wsi_stem}")
            return

        try:
            print("[MAIN] Single-WSI test mode (RUN_SINGLE_WSI_ONLY=True)")
            print(f"[MAIN] Processing WSI: {wsi_path}")
            try:
                process_single_wsi(wsi_path, model=model)
            except Exception as e:
                print(f"[MAIN][ERROR] WSI failed: {wsi_path} -> {e}")
                err = traceback.format_exc()
                try:
                    ensure_dir(os.path.join(OUTPUT_ROOT, wsi_stem))
                    status_path = os.path.join(OUTPUT_ROOT, wsi_stem, "status.json")
                    with open(status_path, "w") as f:
                        json.dump({"wsi": wsi_path, "stem": wsi_stem, "error": str(e), "traceback": err}, f, indent=2)
                    print(f"[STATUS] Saved error status: {status_path}")
                except Exception as _e2:
                    print(f"[STATUS][WARN] Could not write error status: {_e2}")
                raise

            print("[MAIN] Done.")
        finally:
            release_wsi_lock(lock_path)
            print(f"[LOCK] Released: {wsi_stem}")
        return

    wsis = list_wsis(INPUT_WSI_DIR)
    if not wsis:
        print(f"[MAIN] No WSIs found in {INPUT_WSI_DIR}")
        return

    print(f"[MAIN] Found {len(wsis)} WSIs in {INPUT_WSI_DIR}.")
    for wsi_path in wsis:
        wsi_stem = stem_from_path(wsi_path)

        lock_path = acquire_wsi_lock(OUTPUT_ROOT, wsi_stem)
        if lock_path is None:
            print(f"[LOCK] Skipping (already locked): {wsi_stem}")
            continue

        try:
            print("=" * 80)
            print(f"[MAIN] Processing WSI: {wsi_path}")
            try:
                process_single_wsi(wsi_path, model=model)
            except Exception as e:
                print(f"[MAIN][ERROR] WSI failed (skipping): {wsi_path} -> {e}")
                err = traceback.format_exc()
                try:
                    ensure_dir(os.path.join(OUTPUT_ROOT, wsi_stem))
                    status_path = os.path.join(OUTPUT_ROOT, wsi_stem, "status.json")
                    with open(status_path, "w") as f:
                        json.dump({"wsi": wsi_path, "stem": wsi_stem, "error": str(e), "traceback": err}, f, indent=2)
                    print(f"[STATUS] Saved error status: {status_path}")
                except Exception as _e2:
                    print(f"[STATUS][WARN] Could not write error status: {_e2}")

        finally:
            release_wsi_lock(lock_path)
            print(f"[LOCK] Released: {wsi_stem}")

    print("[MAIN] Done.")


if __name__ == "__main__":
    main()
