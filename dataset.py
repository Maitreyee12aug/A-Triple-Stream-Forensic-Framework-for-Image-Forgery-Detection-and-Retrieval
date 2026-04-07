# -*- coding: utf-8 -*-
"""
dataset.py — Dataset preparation for both experiments.

Experiment A — PGF (Programmatically Generated Forgeries):
    Generates copy-move and splicing forgeries from CASIA v2.0 authentic images.

Experiment B — GTF (Ground Truth Forgeries):
    Parses the CASIA v2.0 tampered images and maps each to its authentic source.

Also defines the PyTorch Dataset classes used by all training scripts.

Usage:
    python dataset.py --experiment pgf    # PGF experiment
    python dataset.py --experiment gtf    # GTF experiment
"""

import argparse
import glob
import os
import random
import re
import shutil
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import config
from features import get_srm_filters, save_srm, save_ela


# ── Forgery generation (PGF) ───────────────────────────────────────────────

def create_copy_move_forgery(image_path: str, output_path: str) -> bool:
    """
    Copies a random patch from within the image and pastes it at another
    random location (copy-move forgery). Patch occupies 10–25% of image area.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        pw = random.randint(int(w * config.PATCH_MIN_RATIO), int(w * config.PATCH_MAX_RATIO))
        ph = random.randint(int(h * config.PATCH_MIN_RATIO), int(h * config.PATCH_MAX_RATIO))
        sx, sy = random.randint(0, w - pw), random.randint(0, h - ph)
        patch = img.crop((sx, sy, sx + pw, sy + ph))
        dx, dy = random.randint(0, w - pw), random.randint(0, h - ph)
        img.paste(patch, (dx, dy))
        img.save(output_path)
        return True
    except Exception:
        return False


def create_splicing_forgery(src_path1: str, src_path2: str, output_path: str) -> bool:
    """
    Pastes a random patch from src_path2 into src_path1 (splicing forgery).
    """
    try:
        img1 = Image.open(src_path1).convert("RGB")
        img2 = Image.open(src_path2).convert("RGB").resize(img1.size)
        w, h = img1.size
        pw = random.randint(int(w * config.PATCH_MIN_RATIO), int(w * config.PATCH_MAX_RATIO))
        ph = random.randint(int(h * config.PATCH_MIN_RATIO), int(h * config.PATCH_MAX_RATIO))
        sx, sy = random.randint(0, w - pw), random.randint(0, h - ph)
        patch = img2.crop((sx, sy, sx + pw, sy + ph))
        dx, dy = random.randint(0, w - pw), random.randint(0, h - ph)
        img1.paste(patch, (dx, dy))
        img1.save(output_path)
        return True
    except Exception:
        return False


def prepare_pgf_dataset() -> List[Dict]:
    """
    Builds the PGF dataset:
      1. Copies all CASIA authentic images to rgb_dataset/authentic/.
      2. Generates copy-move + splicing forgeries in rgb_dataset/fake/.
      3. Returns a ground_truth_map: list of {anchor_path, fake_path} dicts.
    """
    base = config.CASIA_BASE_PATH
    if not os.path.exists(base):
        base = config.CASIA_DATASET_DIR

    casia_auth_dir = os.path.join(base, "Au")
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    casia_auth_paths = [
        p for p in glob.glob(os.path.join(casia_auth_dir, "**", "*.*"), recursive=True)
        if os.path.splitext(p)[1].lower() in extensions
    ]

    # Reset directories
    for d in [config.AUTH_DIR, config.FAKE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print("Copying authentic images …")
    for src in tqdm(casia_auth_paths, desc="Authentic"):
        shutil.copy(src, config.AUTH_DIR)

    auth_paths = glob.glob(os.path.join(config.AUTH_DIR, "*.*"))
    n = len(auth_paths) // 2
    ground_truth_map = []

    print(f"Generating {n * 2} forgeries …")
    for i in tqdm(range(n), desc="Forgeries"):
        anchor = random.choice(auth_paths)

        # Copy-move
        cm_name = f"cm_{i}_{os.path.basename(anchor)}"
        cm_path = os.path.join(config.FAKE_DIR, cm_name)
        if create_copy_move_forgery(anchor, cm_path):
            ground_truth_map.append({"anchor_path": anchor, "fake_path": cm_path})

        # Splicing
        splice_src = random.choice(auth_paths)
        sp_name = f"sp_{i}_{os.path.basename(anchor)}"
        sp_path = os.path.join(config.FAKE_DIR, sp_name)
        if create_splicing_forgery(anchor, splice_src, sp_path):
            ground_truth_map.append({"anchor_path": anchor, "fake_path": sp_path})

    print(f"PGF dataset ready — {len(os.listdir(config.AUTH_DIR))} authentic, "
          f"{len(os.listdir(config.FAKE_DIR))} fake, "
          f"{len(ground_truth_map)} mapped pairs.")
    return ground_truth_map


# ── Ground truth parsing (GTF) ────────────────────────────────────────────

def prepare_gtf_dataset() -> List[Dict]:
    """
    Builds the GTF dataset from CASIA v2.0 ground truth:
      1. Copies authentic and tampered images to rgb_dataset/{authentic,fake}/.
      2. Parses filenames to map each tampered image to its authentic source.
      3. Returns a ground_truth_map: list of {anchor_path, fake_path} dicts.
    """
    base = config.CASIA_BASE_PATH
    if not os.path.exists(base):
        base = config.CASIA_DATASET_DIR

    auth_dir    = os.path.join(base, "Au")
    tampered_dir = os.path.join(base, "Tp")
    extensions  = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}

    authentic_paths = [
        p for p in glob.glob(os.path.join(auth_dir, "**", "*.*"), recursive=True)
        if os.path.splitext(p)[1].lower() in extensions
    ]
    tampered_paths = [
        p for p in glob.glob(os.path.join(tampered_dir, "**", "*.*"), recursive=True)
        if os.path.splitext(p)[1].lower() in extensions
    ]

    authentic_map = {os.path.splitext(os.path.basename(p))[0]: p for p in authentic_paths}

    ground_truth_map = []
    unmapped = 0
    print(f"Mapping {len(tampered_paths)} tampered images to authentic sources …")
    for fake_path in tqdm(tampered_paths, desc="Mapping"):
        basename = os.path.splitext(os.path.basename(fake_path))[0]
        matches  = re.findall(r"([a-zA-Z]{2,3}\d{5})", basename)
        found    = False
        if matches:
            matches.sort(key=len, reverse=True)
            for key in matches:
                auth_key = f"Au_{key[:3]}_{key[3:]}"
                if auth_key in authentic_map:
                    ground_truth_map.append({
                        "fake_path":   fake_path,
                        "anchor_path": authentic_map[auth_key],
                    })
                    found = True
                    break
        if not found:
            unmapped += 1

    print(f"Mapped {len(ground_truth_map)} pairs ({unmapped} unmapped).")

    # Reset and populate directories
    for d in [config.AUTH_DIR, config.FAKE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    for src in tqdm(authentic_paths, desc="Copying authentic"):
        shutil.copy(src, config.AUTH_DIR)
    for src in tqdm(tampered_paths, desc="Copying tampered"):
        shutil.copy(src, config.FAKE_DIR)

    print(f"GTF dataset ready — {len(os.listdir(config.AUTH_DIR))} authentic, "
          f"{len(os.listdir(config.FAKE_DIR))} fake.")
    return ground_truth_map


# ── Transforms ────────────────────────────────────────────────────────────

def get_transforms(phase: str = "val") -> Dict:
    """Returns the standard RGB and forensic transform dicts for a given phase."""
    augment = (phase == "train")
    rgb_t = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        *([ transforms.RandomHorizontalFlip() ] if augment else []),
        transforms.ToTensor(),
        transforms.Normalize(config.RGB_MEAN, config.RGB_STD),
    ])
    forensic_t = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        *([ transforms.RandomHorizontalFlip() ] if augment else []),
        transforms.ToTensor(),
        transforms.Normalize(config.FORENSIC_MEAN, config.FORENSIC_STD),
    ])
    return {"rgb": rgb_t, "srm": forensic_t, "ela": forensic_t}


# ── Detection Dataset ─────────────────────────────────────────────────────

class TripleStreamDataset(Dataset):
    """
    PyTorch Dataset for the Triple-Stream Detection Model.
    Returns (rgb_tensor, srm_tensor, ela_tensor), label for each image.
    Label: 0 = authentic, 1 = fake.
    """
    def __init__(
        self,
        file_list:  List[str],
        label_list: List[int],
        srm_dir:    str,
        ela_dir:    str,
        transforms: Dict,
    ):
        self.file_list  = file_list
        self.label_list = label_list
        self.srm_dir    = srm_dir
        self.ela_dir    = ela_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        rgb_path = self.file_list[index]
        label    = self.label_list[index]
        try:
            rgb    = self.transforms["rgb"](Image.open(rgb_path).convert("RGB"))
            fname  = os.path.splitext(os.path.basename(rgb_path))[0]
            subdir = "authentic" if label == 0 else "fake"
            srm    = self.transforms["srm"](
                Image.open(os.path.join(self.srm_dir, subdir, f"{fname}_srm.png")).convert("RGB")
            )
            ela    = self.transforms["ela"](
                Image.open(os.path.join(self.ela_dir, subdir, f"{fname}_ela.png")).convert("RGB")
            )
            return (rgb, srm, ela), label
        except Exception:
            return self.__getitem__((index + 1) % len(self))


# ── Retrieval Dataset (Triplet) ───────────────────────────────────────────

class TripleStreamTripletDataset(Dataset):
    """
    PyTorch Dataset for the Triple-Stream Retrieval Model.
    Returns (anchor_triplet, positive_triplet, negative_triplet) for each pair,
    where the negative is a randomly sampled different authentic image.
    """
    def __init__(
        self,
        ground_truth_map: List[Dict],
        auth_paths:       List[str],
        transforms:       Dict,
        srm_dir:          str = config.SRM_DATASET_DIR,
        ela_dir:          str = config.ELA_DATASET_DIR,
    ):
        self.ground_truth_map = ground_truth_map
        self.auth_paths       = auth_paths
        self.transforms       = transforms
        self.srm_map, self.ela_map = self._build_path_maps(srm_dir, ela_dir)

    @staticmethod
    def _build_path_maps(srm_dir: str, ela_dir: str) -> Tuple[Dict, Dict]:
        srm_map, ela_map = {}, {}
        for d, suffix, target in [
            (srm_dir, "_srm.png", srm_map),
            (ela_dir, "_ela.png", ela_map),
        ]:
            for subdir in ["authentic", "fake"]:
                subpath = os.path.join(d, subdir)
                if not os.path.exists(subpath):
                    continue
                for f in os.listdir(subpath):
                    if f.endswith(suffix):
                        target[f.replace(suffix, "")] = os.path.join(subpath, f)
        return srm_map, ela_map

    def _load_triplet(self, rgb_path: str):
        rgb   = self.transforms["rgb"](Image.open(rgb_path).convert("RGB"))
        fname = os.path.splitext(os.path.basename(rgb_path))[0]
        srm   = self.transforms["srm"](Image.open(self.srm_map[fname]).convert("RGB"))
        ela   = self.transforms["ela"](Image.open(self.ela_map[fname]).convert("RGB"))
        return rgb, srm, ela

    def __len__(self) -> int:
        return len(self.ground_truth_map)

    def __getitem__(self, idx: int):
        try:
            pair        = self.ground_truth_map[idx]
            anchor_path = pair["anchor_path"]
            pos_path    = pair["fake_path"]

            # Mine a hard negative: a different authentic image with forensic features
            while True:
                neg_path  = random.choice(self.auth_paths)
                neg_fname = os.path.splitext(os.path.basename(neg_path))[0]
                if neg_path != anchor_path and neg_fname in self.srm_map and neg_fname in self.ela_map:
                    break

            return (
                self._load_triplet(anchor_path),
                self._load_triplet(pos_path),
                self._load_triplet(neg_path),
            )
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for either experiment.")
    parser.add_argument(
        "--experiment", choices=["pgf", "gtf"], required=True,
        help="'pgf' for Programmatic Forgeries, 'gtf' for CASIA Ground Truth."
    )
    args = parser.parse_args()

    if args.experiment == "pgf":
        gt_map = prepare_pgf_dataset()
    else:
        gt_map = prepare_gtf_dataset()

    print(f"\nRunning forensic feature generation …")
    from features import generate_forensic_features
    generate_forensic_features()
    print(f"\nDataset ready. {len(gt_map)} ground-truth pairs.")
