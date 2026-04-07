# -*- coding: utf-8 -*-
"""
inference.py — End-to-end forensic pipeline for a single query image.

Step 1 : Compute SRM + ELA for the query image on-the-fly.
Step 2 : Run Triple-Stream Detection Model → AUTHENTIC / FAKE + confidence.
Step 3 : If FAKE, run Triple-Stream Retrieval Model + Faiss → Top-K source matches.
Step 4 : Visualise results.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --k 5
"""

import argparse
import io
import os
import pickle

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.signal import convolve2d
from torchvision import transforms

import config
from features import get_srm_filters
from models import TripleStreamForgeryDetector, TripleStreamRetrievalNet


class ForensicPipeline:
    """
    End-to-end forensic analysis pipeline (Section 3, inference path).

    Attributes:
        det_model    : Loaded TripleStreamForgeryDetector.
        ret_model    : Loaded TripleStreamRetrievalNet.
        index        : Faiss index of authentic-image embeddings.
        indexed_paths: Authentic image paths corresponding to Faiss rows.
        srm_filters  : SRM high-pass filter kernels.
    """

    def __init__(
        self,
        det_path:  str = config.DETECTION_MODEL_PATH,
        ret_path:  str = config.RETRIEVAL_MODEL_PATH,
        idx_path:  str = config.FAISS_INDEX_PATH,
        pkl_path:  str = config.FAISS_PATHS_PKL,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rgb_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.RGB_MEAN, config.RGB_STD),
        ])
        self.forensic_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.FORENSIC_MEAN, config.FORENSIC_STD),
        ])

        self.det_model = TripleStreamForgeryDetector().to(self.device)
        self.det_model.load_state_dict(torch.load(det_path, map_location=self.device))
        self.det_model.eval()

        self.ret_model = TripleStreamRetrievalNet().to(self.device)
        self.ret_model.load_state_dict(torch.load(ret_path, map_location=self.device))
        self.ret_model.eval()

        self.index = faiss.read_index(idx_path)
        with open(pkl_path, "rb") as f:
            self.indexed_paths = pickle.load(f)

        self.srm_filters = get_srm_filters()
        print(f"Pipeline ready. Faiss index: {self.index.ntotal} vectors.")

    # ── On-the-fly forensic feature computation ───────────────────────────

    def _compute_srm(self, pil_img: Image.Image) -> Image.Image:
        """Computes SRM from a PIL Image directly (no disk I/O)."""
        import cv2
        gray = np.array(pil_img.convert("L"))
        residuals = [
            convolve2d(gray.astype("float32"), f, mode="same", boundary="symm")
            for f in self.srm_filters
        ]
        srm = np.stack(residuals, axis=-1)
        srm = cv2.normalize(srm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return Image.fromarray(srm, "RGB")

    def _compute_ela(
        self,
        pil_img: Image.Image,
        quality: int = config.ELA_QUALITY,
        scale:   int = config.ELA_SCALE,
    ) -> Image.Image:
        """Computes ELA from a PIL Image using an in-memory buffer."""
        from PIL import ImageChops, ImageEnhance
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf)
        ela = ImageChops.difference(pil_img.convert("RGB"), resaved)
        extrema  = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        if max_diff == 0:
            max_diff = 1
        ela = ImageEnhance.Brightness(ela).enhance(scale / max_diff)
        return ela

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, img_path: str, k: int = config.RETRIEVAL_TOP_K) -> None:
        """
        Analyses a single image:
          1. Detects whether it is AUTHENTIC or FAKE.
          2. If FAKE, retrieves the top-K candidate source images from the Faiss index.
          3. Displays a visualisation.

        Args:
            img_path : Path to the query image.
            k        : Number of source candidates to retrieve.
        """
        print(f"\n{'='*60}")
        print(f"  Analysing: {os.path.basename(img_path)}")
        print(f"{'='*60}")

        try:
            rgb_img = Image.open(img_path).convert("RGB")
            srm_img = self._compute_srm(rgb_img)
            ela_img = self._compute_ela(rgb_img)
        except Exception as e:
            print(f"Error loading/processing image: {e}"); return

        rgb_t = self.rgb_transform(rgb_img).unsqueeze(0).to(self.device)
        srm_t = self.forensic_transform(srm_img).unsqueeze(0).to(self.device)
        ela_t = self.forensic_transform(ela_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits      = self.det_model(rgb_t, srm_t, ela_t)
            probs       = F.softmax(logits, dim=1)
            fake_conf   = probs[0][1].item()
            prediction  = "FAKE" if fake_conf >= config.FAKE_THRESHOLD else "AUTHENTIC"

        print(f"  DETECTION  : {prediction}")
        print(f"  Confidence : {fake_conf:.2%}  (fake probability)")

        if prediction == "AUTHENTIC":
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(rgb_img); ax.set_title(f"AUTHENTIC ({1-fake_conf:.2%})"); ax.axis("off")
            plt.tight_layout(); plt.show()
            return

        # Retrieval
        with torch.no_grad():
            emb = self.ret_model(rgb_t, srm_t, ela_t).cpu().numpy()

        dists, idxs = self.index.search(emb, k)

        fig, axes = plt.subplots(1, k + 1, figsize=(5 * (k + 1), 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Query\nFAKE ({fake_conf:.2%})", fontsize=11)
        axes[0].axis("off")

        for i in range(k):
            ret_img = Image.open(self.indexed_paths[idxs[0][i]])
            axes[i + 1].imshow(ret_img)
            axes[i + 1].set_title(f"Match {i+1}\nDist: {dists[0][i]:.3f}", fontsize=11)
            axes[i + 1].axis("off")

        plt.suptitle(f"Forensic Retrieval: {os.path.basename(img_path)}", fontsize=13)
        plt.tight_layout()
        os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
        out_path = os.path.join(config.OUTPUTS_DIR, f"retrieval_{os.path.basename(img_path)}.png")
        plt.savefig(out_path, dpi=150); plt.show()
        print(f"Result saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Triple-Stream Forensic Pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Path to the query image.")
    parser.add_argument("--k",     type=int, default=config.RETRIEVAL_TOP_K,
                        help=f"Number of source candidates to retrieve (default: {config.RETRIEVAL_TOP_K}).")
    args = parser.parse_args()

    pipeline = ForensicPipeline()
    pipeline.run(args.image, k=args.k)
