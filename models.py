# -*- coding: utf-8 -*-
"""
models.py — Triple-Stream model architectures.

Covers Sections 3.2 and 3.3 of the paper.

    TripleStreamForgeryDetector  — Detection model (Section 3.2)
        Three parallel ResNet-50 backbones (RGB, SRM, ELA).
        Late fusion via MLP head → binary classification logits.
        Trained with Binary Cross-Entropy loss (two-phase fine-tuning).

    TripleStreamRetrievalNet     — Retrieval model (Section 3.3)
        Same triple-stream backbone, projection head → L2-normalised embedding.
        Trained with Multi-Modal Triplet Margin Loss.
        Embeddings indexed with Faiss for millisecond-level retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


# ── Detection Model ───────────────────────────────────────────────────────

class TripleStreamForgeryDetector(nn.Module):
    """
    Triple-Stream Forgery Detection Model (Section 3.2).

    Architecture:
        RGB Stream  : ResNet-50 → global avg pool → f_rgb  ∈ R^2048
        SRM Stream  : ResNet-50 → global avg pool → f_srm  ∈ R^2048
        ELA Stream  : ResNet-50 → global avg pool → f_ela  ∈ R^2048
        Fusion      : concat → [f_rgb ‖ f_srm ‖ f_ela] ∈ R^6144
        Head        : Linear(6144→1024) → BN → ReLU → Dropout(0.5)
                      → Linear(1024→2)  [logits for authentic / fake]
    """

    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT

        self.rgb_stream = models.resnet50(weights=weights)
        self.srm_stream = models.resnet50(weights=weights)
        self.ela_stream = models.resnet50(weights=weights)

        num_ftrs = self.rgb_stream.fc.in_features  # 2048

        # Remove classification heads — use as feature extractors
        self.rgb_stream.fc = nn.Identity()
        self.srm_stream.fc = nn.Identity()
        self.ela_stream.fc = nn.Identity()

        self.fusion_head = nn.Sequential(
            nn.Linear(num_ftrs * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_stream(rgb)
        f_srm = self.srm_stream(srm)
        f_ela = self.ela_stream(ela)
        fused = torch.cat([f_rgb, f_srm, f_ela], dim=1)
        return self.fusion_head(fused)

    def freeze_backbones(self) -> None:
        """Phase 1: freeze all backbone parameters, leave fusion head trainable."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fusion_head.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Phase 2: unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


# ── Retrieval Model ───────────────────────────────────────────────────────

class TripleStreamRetrievalNet(nn.Module):
    """
    Triple-Stream Forensic-Aware Retrieval Model (Section 3.3).

    Architecture:
        RGB Stream  : ResNet-50 → global avg pool → f_rgb  ∈ R^2048
        SRM Stream  : ResNet-50 → global avg pool → f_srm  ∈ R^2048
        ELA Stream  : ResNet-50 → global avg pool → f_ela  ∈ R^2048
        Fusion      : concat → [f_rgb ‖ f_srm ‖ f_ela] ∈ R^6144
        Proj Head   : Linear(6144→2048) → BN → ReLU → Linear(2048→emb_dim)
        Output      : L2-normalised embedding ∈ R^emb_dim

    Trained with Triplet Margin Loss:
        L = max(‖f(A)−f(P)‖² − ‖f(A)−f(N)‖² + α, 0)
    """

    def __init__(self, emb_dim: int = config.EMBEDDING_DIM):
        super().__init__()

        self.rgb_stream = models.resnet50(weights=None)
        self.srm_stream = models.resnet50(weights=None)
        self.ela_stream = models.resnet50(weights=None)

        num_ftrs = self.rgb_stream.fc.in_features  # 2048

        self.rgb_stream.fc = nn.Identity()
        self.srm_stream.fc = nn.Identity()
        self.ela_stream.fc = nn.Identity()

        self.proj_head = nn.Sequential(
            nn.Linear(num_ftrs * 3, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, emb_dim),
        )

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_stream(rgb)
        f_srm = self.srm_stream(srm)
        f_ela = self.ela_stream(ela)
        fused = torch.cat([f_rgb, f_srm, f_ela], dim=1)
        return F.normalize(self.proj_head(fused), p=2, dim=1)

    @classmethod
    def from_detection_weights(cls, det_model: TripleStreamForgeryDetector, emb_dim: int = config.EMBEDDING_DIM):
        """
        Initialises the retrieval model by transferring backbone weights from a
        trained detection model (Algorithm 1, Step 3.20 of the paper).
        """
        retrieval = cls(emb_dim=emb_dim)
        retrieval.rgb_stream.load_state_dict(det_model.rgb_stream.state_dict())
        retrieval.srm_stream.load_state_dict(det_model.srm_stream.state_dict())
        retrieval.ela_stream.load_state_dict(det_model.ela_stream.state_dict())
        return retrieval
