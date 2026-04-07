# -*- coding: utf-8 -*-
"""
features.py — Forensic feature extraction: SRM and ELA.

Covers Section 3.1 of the paper.

    SRM  — Spatial Rich Models: amplifies low-level noise fingerprints.
           R_k = I * F_k  (convolution with high-pass filters)

    ELA  — Error Level Analysis: exposes compression history inconsistencies.
           I_ELA = α · |I_RGB − I_JPEG(q)|

Usage:
    python features.py          # generates SRM + ELA for rgb_dataset/
"""

import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy.signal import convolve2d
from tqdm import tqdm

import config


# ── SRM ───────────────────────────────────────────────────────────────────

def get_srm_filters() -> list:
    """
    Returns the three high-pass SRM filter kernels used in the paper.
    Applied to the grayscale image to amplify noise residual patterns.
    """
    h1 = np.array([[-1,  2, -1],
                   [ 2, -4,  2],
                   [-1,  2, -1]], dtype=np.float32)

    h2 = np.array([[-1,  2, -2,  2, -1],
                   [ 2, -6,  8, -6,  2],
                   [-2,  8,-12,  8, -2],
                   [ 2, -6,  8, -6,  2],
                   [-1,  2, -2,  2, -1]], dtype=np.float32) / 12.0

    h3 = np.array([[0,  0, 0],
                   [0, -1, 1],
                   [0,  0, 0]], dtype=np.float32)

    return [h1, h2, h3]


def create_srm_image(rgb_path: str, filters: list) -> np.ndarray:
    """
    Generates a 3-channel SRM feature image from an RGB image file.

    Args:
        rgb_path: Path to the source RGB image.
        filters:  List of SRM filter kernels (from get_srm_filters()).

    Returns:
        uint8 NumPy array of shape (H, W, 3), or None on error.
    """
    img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    residuals = [convolve2d(img.astype("float32"), f, mode="same", boundary="symm")
                 for f in filters]
    srm = np.stack(residuals, axis=-1)
    srm = cv2.normalize(srm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return srm


def save_srm(rgb_path: str, out_dir: str, filters: list) -> None:
    """Computes SRM and saves as <basename>_srm.png in out_dir."""
    try:
        srm = create_srm_image(rgb_path, filters)
        if srm is None:
            return
        fname = os.path.splitext(os.path.basename(rgb_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{fname}_srm.png"), srm)
    except Exception:
        pass


# ── ELA ───────────────────────────────────────────────────────────────────

def create_ela_image(
    rgb_path: str,
    quality: int = config.ELA_QUALITY,
    scale: int   = config.ELA_SCALE,
) -> Image.Image:
    """
    Generates an ELA feature image.

    The image is re-saved at JPEG quality q; regions with inconsistent
    compression history will show elevated error levels.

    Args:
        rgb_path: Path to the source RGB image.
        quality:  JPEG re-save quality (paper: 90).
        scale:    Brightness scaling factor (paper: 15).

    Returns:
        PIL Image in RGB mode, or None on error.
    """
    try:
        orig = Image.open(rgb_path).convert("RGB")
        tmp_path = "_tmp_ela.jpg"
        orig.save(tmp_path, "JPEG", quality=quality)
        resaved = Image.open(tmp_path)
        ela = ImageChops.difference(orig, resaved)
        extrema = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        if max_diff == 0:
            max_diff = 1
        ela = ImageEnhance.Brightness(ela).enhance(scale / max_diff)
        os.remove(tmp_path)
        return ela
    except Exception:
        return None


def save_ela(
    rgb_path: str,
    out_dir: str,
    quality: int = config.ELA_QUALITY,
    scale: int   = config.ELA_SCALE,
) -> None:
    """Computes ELA and saves as <basename>_ela.png in out_dir."""
    try:
        ela = create_ela_image(rgb_path, quality, scale)
        if ela is None:
            return
        fname = os.path.splitext(os.path.basename(rgb_path))[0]
        ela.save(os.path.join(out_dir, f"{fname}_ela.png"))
    except Exception:
        pass


# ── Batch generation ──────────────────────────────────────────────────────

def generate_forensic_features(
    auth_dir: str = config.AUTH_DIR,
    fake_dir: str = config.FAKE_DIR,
    srm_dir: str  = config.SRM_DATASET_DIR,
    ela_dir: str  = config.ELA_DATASET_DIR,
) -> None:
    """
    Generates SRM and ELA feature maps for all images in auth_dir and
    fake_dir, saving them to the corresponding subdirectories of srm_dir
    and ela_dir.
    """
    import shutil

    filters = get_srm_filters()

    for directory in [srm_dir, ela_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(os.path.join(directory, "authentic"))
        os.makedirs(os.path.join(directory, "fake"))

    for label, src_dir in [("authentic", auth_dir), ("fake", fake_dir)]:
        paths = glob.glob(os.path.join(src_dir, "*.*"))
        for path in tqdm(paths, desc=f"SRM+ELA [{label}]"):
            save_srm(path, os.path.join(srm_dir, label), filters)
            save_ela(path, os.path.join(ela_dir, label))

    print(f"Forensic features saved → {srm_dir}/  {ela_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_forensic_features()
