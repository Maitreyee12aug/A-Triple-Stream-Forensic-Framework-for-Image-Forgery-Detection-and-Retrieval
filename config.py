# -*- coding: utf-8 -*-
"""
config.py — Centralised hyperparameters and path settings.
All tuneable values live here; edit this file before running any script.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
CASIA_DATASET_DIR   = "CASIA_2.0_Dataset"
CASIA_BASE_PATH     = os.path.join(CASIA_DATASET_DIR, "CASIA2")   # fallback: CASIA_2.0_Dataset

RGB_DATA_DIR        = "rgb_dataset"
AUTH_DIR            = os.path.join(RGB_DATA_DIR, "authentic")
FAKE_DIR            = os.path.join(RGB_DATA_DIR, "fake")

SRM_DATASET_DIR     = "srm_dataset"
ELA_DATASET_DIR     = "ela_dataset"

CHECKPOINTS_DIR     = "checkpoints"
OUTPUTS_DIR         = "outputs"

DETECTION_MODEL_PATH  = os.path.join(CHECKPOINTS_DIR, "best_triple_stream_model.pth")
RETRIEVAL_MODEL_PATH  = os.path.join(CHECKPOINTS_DIR, "best_triple_stream_retrieval_model.pth")
FAISS_INDEX_PATH      = os.path.join(CHECKPOINTS_DIR, "forensic.index")
FAISS_PATHS_PKL       = os.path.join(CHECKPOINTS_DIR, "forensic_paths.pkl")
TRAINING_HISTORY_PKL  = os.path.join(CHECKPOINTS_DIR, "training_history.pkl")

# ── Image settings ─────────────────────────────────────────────────────────
IMG_SIZE            = 224

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE          = 16
RANDOM_STATE        = 42
VAL_SPLIT           = 0.20

# Detection model (two-phase fine-tuning)
PHASE1_EPOCHS       = 5       # frozen backbones, head only
PHASE1_LR           = 1e-3
PHASE2_EPOCHS       = 20      # full end-to-end fine-tuning
PHASE2_LR           = 1e-5
PHASE2_WEIGHT_DECAY = 1e-4
LR_PATIENCE         = 3       # ReduceLROnPlateau patience

# Retrieval model
RETRIEVAL_EPOCHS    = 50
RETRIEVAL_LR        = 1e-4    # paper: 1e-4 (kept 1e-5 for fine-tuned warm start)
TRIPLET_MARGIN      = 0.5
EMBEDDING_DIM       = 512

# ── Forgery generation (PGF experiment only) ───────────────────────────────
PATCH_MIN_RATIO     = 1 / 8   # min patch size as fraction of image dimension
PATCH_MAX_RATIO     = 1 / 4   # max patch size as fraction of image dimension

# ── ELA settings ───────────────────────────────────────────────────────────
ELA_QUALITY         = 90
ELA_SCALE           = 15

# ── Normalisation constants ────────────────────────────────────────────────
RGB_MEAN            = [0.485, 0.456, 0.406]
RGB_STD             = [0.229, 0.224, 0.225]
FORENSIC_MEAN       = [0.5, 0.5, 0.5]
FORENSIC_STD        = [0.5, 0.5, 0.5]

# ── Inference ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K     = 3
FAKE_THRESHOLD      = 0.5     # softmax probability above this → FAKE

# ── Recall@K evaluation ───────────────────────────────────────────────────
RECALL_K_VALUES     = [1, 5, 10]
