# A Triple-Stream Forensic Framework for Image Forgery Detection and Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Accepted-green)](.)

---

> ⚠️ **Citation Notice:** This repository is the official implementation of
> *"A Triple-Stream Forensic Framework for Image Forgery Detection and Retrieval"*
> (Maitreyee Ganguly, Paramita Dey, Soumik Pal — Government College of Engineering and Ceramic Technology, Kolkata).
> If you use this code or build upon this work, please cite our paper using the BibTeX entry at the bottom of this README.

---

## Overview

We propose a unified forensic framework that simultaneously performs **image forgery detection** and **source image retrieval** using three complementary forensic modalities — the first framework to combine both tasks in a single architecture.

Key contributions:

- **Triple-Stream Fusion** — Three parallel ResNet-50 backbones processing RGB, SRM (noise fingerprint), and ELA (compression history) streams simultaneously
- **Two-Phase Fine-Tuning** — Frozen backbone head training followed by end-to-end fine-tuning for stable convergence
- **Forensic-Aware Retrieval** — Multi-Modal Triplet Margin Loss forces the embedding space to capture forensic similarity, not just visual similarity
- **Faiss Indexing** — Millisecond-level source image retrieval via L2 nearest-neighbour search
- **Dual Evaluation** — Validated on both controlled programmatic forgeries (PGF) and real-world CASIA ground truth forgeries (GTF)

### Key Results

| Task | Dataset | Score |
|---|---|---|
| Detection | PGF (copy-move + splicing) | **99.2% accuracy** |
| Detection | GTF (CASIA v2.0 in-the-wild) | **94.0% accuracy** |
| Retrieval Recall@1 | PGF | **94.86%** |
| Retrieval Recall@5 | PGF | **98.87%** |
| Retrieval Recall@10 | PGF | **99.00%** |
| Retrieval Recall@1 | GTF | **90.09%** |
| Retrieval Recall@5 | GTF | **94.80%** |
| Retrieval Recall@10 | GTF | **97.03%** |

---

## Architecture

```
Query Image
     │
     ├─────────────────────────────────────────────┐
     │   SRM (noise fingerprint)                   │ ELA (compression history)
     ▼                                             ▼
┌────────────┐   ┌────────────┐   ┌────────────────────────┐
│ RGB Stream │   │ SRM Stream │   │      ELA Stream         │
│ ResNet-50  │   │ ResNet-50  │   │      ResNet-50           │
│ f_rgb∈R²⁰⁴⁸│   │ f_srm∈R²⁰⁴⁸│   │      f_ela∈R²⁰⁴⁸         │
└─────┬──────┘   └─────┬──────┘   └──────────┬─────────────┘
      └─────────────────┴────────────────────┘
                        │ concat [f_rgb ‖ f_srm ‖ f_ela] ∈ R⁶¹⁴⁴
                        ▼
         ┌──────────────────────────────┐
         │  Detection Path              │  Retrieval Path
         │  MLP Fusion Head             │  Projection Head
         │  Linear(6144→1024)           │  Linear(6144→2048)→ReLU
         │  BN → ReLU → Dropout(0.5)   │  →Linear(2048→512)
         │  → Linear(1024→2)            │  → L2-normalise
         │  [authentic / fake]          │  → embedding ∈ R⁵¹²
         └──────────────────────────────┘
                                        │
                               ┌────────▼────────┐
                               │   Faiss Index    │
                               │ IndexFlatL2(512) │
                               │  L2 NNS search   │
                               └─────────────────┘
                                        │
                               Top-K Source Matches
```

---

## Repository Structure

```
triple-stream-forensics/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── config.py        ← All hyperparameters and path settings
│
├── features.py      ← Section 3.1: SRM and ELA forensic feature extraction
├── dataset.py       ← Section 4.1: PGF forgery generation + CASIA GTF parsing
│                       PyTorch Dataset classes (Detection + Triplet)
├── models.py        ← Sections 3.2–3.3: TripleStreamForgeryDetector +
│                       TripleStreamRetrievalNet
├── train.py         ← Section 4.2: Two-phase detection training + retrieval
│                       training + Faiss index construction (Algorithm 1)
├── evaluate.py      ← Section 5: Confusion matrix + Recall@K evaluation
├── visualize.py     ← Section 5: All paper figures (Figures 2, 4, 7)
├── inference.py     ← Sections 3.2–3.4: End-to-end inference pipeline
│
├── data/            ← (not tracked) Dataset and feature directories
├── checkpoints/     ← (not tracked) Saved model weights + Faiss index
└── outputs/         ← (not tracked) Figures and inference results
```

---

## Requirements

### System Requirements

- Python >= 3.8
- CUDA >= 11.3 recommended (3× ResNet-50 backbones require ~8 GB VRAM for training)
- CPU inference supported but significantly slower

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Maitreyee12aug/triple-stream-forensics.git
cd triple-stream-forensics

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU users) Replace faiss-cpu with faiss-gpu
pip install faiss-gpu
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| PyTorch | >= 1.12.0 | Deep learning framework |
| torchvision | >= 0.13.0 | ResNet-50 backbones |
| faiss-cpu | >= 1.7.3 | Faiss nearest-neighbour indexing |
| opencv-python | >= 4.6.0 | SRM feature computation |
| Pillow | >= 9.0.0 | ELA feature computation |
| scipy | >= 1.9.0 | 2D convolution for SRM filters |
| grad-cam | >= 1.4.6 | Grad-CAM interpretability |
| scikit-learn | >= 1.0.0 | Train/val split, confusion matrix |

---

## Dataset

This work uses the **CASIA v2.0 Image Tampering Detection Dataset**, publicly available on Kaggle:

```
https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
```

Two experimental setups are used:

| Dataset | Authentic | Fake | Forgery Type | Total |
|---|---|---|---|---|
| A: PGF (Programmatic) | 7,491 | 7,490 | Copy-Move + Splicing | 14,981 |
| B: GTF (Ground Truth) | 7,491 | 5,123 | Manual, diverse tools | 12,614 |

Both use an 80/20 train/val split with leak-free partitioning (each authentic image and all its forgeries stay in the same split).

---

## Usage

### Step 1 — Download the Dataset

```bash
# Using Kaggle API
kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset
unzip casia-20-image-tampering-detection-dataset.zip -d CASIA_2.0_Dataset
```

### Step 2 — Train (PGF Experiment)

```bash
python train.py --experiment pgf
```

This runs the full Algorithm 1 pipeline:
- Generates copy-move and splicing forgeries
- Computes SRM and ELA feature maps
- Trains the detection model (Phase 1 + Phase 2)
- Trains the retrieval model (50 epochs, Triplet Margin Loss)
- Builds the Faiss index

Checkpoints saved to `checkpoints/`.

### Step 2 — Train (GTF Experiment)

```bash
python train.py --experiment gtf
```

Parses CASIA v2.0 ground truth tampered images instead of generating forgeries.

### Step 3 — Evaluate

```bash
python evaluate.py --experiment pgf    # or --experiment gtf
```

Reports detection accuracy, confusion matrix, and Recall@1/5/10.

### Step 4 — Inference on a Single Image

```bash
python inference.py --image /path/to/image.jpg
python inference.py --image /path/to/image.jpg --k 5
```

Example output:
```
════════════════════════════════════════════════════════════
  Analysing: suspicious_image.jpg
════════════════════════════════════════════════════════════
  DETECTION  : FAKE
  Confidence : 93.12%  (fake probability)
  Result saved → outputs/retrieval_suspicious_image.jpg.png
```

### Step 5 — Reproduce Paper Figures

```bash
# All figures for one experiment
python visualize.py --experiment pgf

# Grad-CAM only
python visualize.py --experiment pgf --gradcam

# Distance distribution plot
python visualize.py --experiment pgf --distance_plot

# Combined PGF vs GTF distance distribution (Figure 7)
python visualize.py --combined_dist
```

---

## Hyperparameters

All hyperparameters are centralised in `config.py`.

| Parameter | Value | Description |
|---|---|---|
| `IMG_SIZE` | `224` | Input image resolution |
| `BATCH_SIZE` | `16` | Batch size for all dataloaders |
| `PHASE1_EPOCHS` | `5` | Head-only training epochs |
| `PHASE1_LR` | `1e-3` | Phase 1 learning rate |
| `PHASE2_EPOCHS` | `20` | End-to-end fine-tuning epochs |
| `PHASE2_LR` | `1e-5` | Phase 2 learning rate |
| `RETRIEVAL_EPOCHS` | `50` | Retrieval model training epochs |
| `RETRIEVAL_LR` | `1e-4` | Retrieval model learning rate |
| `TRIPLET_MARGIN` | `0.5` | Triplet Margin Loss α |
| `EMBEDDING_DIM` | `512` | Forensic embedding dimension |
| `ELA_QUALITY` | `90` | JPEG re-save quality for ELA |
| `ELA_SCALE` | `15` | ELA brightness scaling factor |
| `RETRIEVAL_TOP_K` | `3` | Default top-K matches at inference |
| `RECALL_K_VALUES` | `[1, 5, 10]` | K values for Recall@K evaluation |

---

## Baseline Comparison (Recall@K)

| Method | Dataset | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) |
|---|---|---|---|---|
| Baseline (RGB-only) | PGF | 74.5 | 85.3 | 91.0 |
| Baseline (RGB-only) | GTF | 68.2 | 79.1 | 84.4 |
| Baseline (SRM-only) | PGF | 58.4 | 71.2 | 79.5 |
| Baseline (ELA-only) | PGF | 65.2 | 76.4 | 83.7 |
| **Proposed Triple-Stream** | **PGF** | **94.9** | **98.9** | **99.0** |
| **Proposed Triple-Stream** | **GTF** | **90.1** | **94.8** | **97.0** |

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ganguly2025triplestream,
  title     = {A Triple-Stream Forensic Framework for Image Forgery Detection and Retrieval},
  author    = {Ganguly, Maitreyee and Dey, Paramita and Pal, Soumik},
  journal   = {[Venue]},
  year      = {2025},
  institution = {Government College of Engineering and Ceramic Technology, Kolkata}
}
```

---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- **Maitreyee Ganguly** — maitreyee12aug@gmail.com
- **Paramita Dey** — dey.paramita77@gmail.com
- **Soumik Pal** — soumik.kms@gmail.com

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [CASIA v2.0 Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
- [Facebook AI Similarity Search (Faiss)](https://github.com/facebookresearch/faiss)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [HuggingFace torchvision](https://github.com/pytorch/vision)
