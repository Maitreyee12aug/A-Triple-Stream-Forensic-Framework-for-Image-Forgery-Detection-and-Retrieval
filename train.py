# -*- coding: utf-8 -*-
"""
train.py — Trains both models for one experiment (PGF or GTF).

Algorithm 1 from the paper:
    Phase 1 : Freeze backbones, train fusion head (5 epochs, lr=1e-3)
    Phase 2 : Unfreeze all, end-to-end fine-tuning (up to 20 epochs, lr=1e-5)
    Step 3  : Init retrieval model from detection weights, train with
              Triplet Margin Loss (50 epochs, lr=1e-4)
    Step 4  : Build Faiss index over all authentic embeddings

Usage:
    python train.py --experiment pgf
    python train.py --experiment gtf
"""

import argparse
import copy
import os
import pickle

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import (
    TripleStreamDataset,
    TripleStreamTripletDataset,
    get_transforms,
    prepare_pgf_dataset,
    prepare_gtf_dataset,
)
from models import TripleStreamForgeryDetector, TripleStreamRetrievalNet


def build_detection_splits(ground_truth_map):
    """Builds balanced train/val file+label lists from the ground truth map."""
    train_map, val_map = train_test_split(
        ground_truth_map, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE
    )
    train_files, train_labels = [], []
    val_files,   val_labels   = [], []
    for pair in train_map:
        train_files  += [pair["anchor_path"], pair["fake_path"]]
        train_labels += [0, 1]
    for pair in val_map:
        val_files  += [pair["anchor_path"], pair["fake_path"]]
        val_labels += [0, 1]
    return train_files, train_labels, val_files, val_labels, train_map, val_map


# ── Detection training ────────────────────────────────────────────────────

def train_detection_model(ground_truth_map) -> TripleStreamForgeryDetector:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_files, train_labels, val_files, val_labels, train_map, _ = \
        build_detection_splits(ground_truth_map)

    train_t = get_transforms("train")
    val_t   = get_transforms("val")

    train_ds = TripleStreamDataset(train_files, train_labels, config.SRM_DATASET_DIR, config.ELA_DATASET_DIR, train_t)
    val_ds   = TripleStreamDataset(val_files,   val_labels,   config.SRM_DATASET_DIR, config.ELA_DATASET_DIR, val_t)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model     = TripleStreamForgeryDetector().to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Head only ──
    print("\n=== Phase 1: Training fusion head (backbones frozen) ===")
    model.freeze_backbones()
    opt1 = optim.Adam(model.fusion_head.parameters(), lr=config.PHASE1_LR)
    for epoch in range(config.PHASE1_EPOCHS):
        model.train()
        for (rgb, srm, ela), labels in tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}/{config.PHASE1_EPOCHS}"):
            opt1.zero_grad()
            loss = criterion(model(rgb.to(device), srm.to(device), ela.to(device)), labels.to(device))
            loss.backward()
            opt1.step()

    # ── Phase 2: End-to-end ──
    print("\n=== Phase 2: End-to-end fine-tuning ===")
    model.unfreeze_all()
    opt2      = optim.Adam(model.parameters(), lr=config.PHASE2_LR, weight_decay=config.PHASE2_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt2, "min", patience=config.LR_PATIENCE)

    best_val_loss    = float("inf")
    best_weights     = copy.deepcopy(model.state_dict())
    history          = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.PHASE2_EPOCHS):
        model.train()
        run_loss, run_correct = 0.0, 0
        for (rgb, srm, ela), labels in tqdm(train_loader, desc=f"Phase2 Epoch {epoch+1}/{config.PHASE2_EPOCHS}"):
            opt2.zero_grad()
            out  = model(rgb.to(device), srm.to(device), ela.to(device))
            loss = criterion(out, labels.to(device))
            loss.backward(); opt2.step()
            _, preds  = torch.max(out, 1)
            run_loss += loss.item() * rgb.size(0)
            run_correct += (preds == labels.to(device)).sum().item()

        epoch_loss = run_loss  / len(train_ds)
        epoch_acc  = run_correct / len(train_ds)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for (rgb, srm, ela), labels in val_loader:
                out  = model(rgb.to(device), srm.to(device), ela.to(device))
                loss = criterion(out, labels.to(device))
                _, preds = torch.max(out, 1)
                val_loss    += loss.item() * rgb.size(0)
                val_correct += (preds == labels.to(device)).sum().item()
        val_epoch_loss = val_loss    / len(val_ds)
        val_epoch_acc  = val_correct / len(val_ds)
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        print(f"  Train  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
        print(f"  Val    Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

        scheduler.step(val_epoch_loss)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_weights  = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.DETECTION_MODEL_PATH)
    with open(config.TRAINING_HISTORY_PKL, "wb") as f:
        pickle.dump(history, f)
    print(f"\nDetection model saved → {config.DETECTION_MODEL_PATH}")
    return model


# ── Retrieval training ────────────────────────────────────────────────────

def train_retrieval_model(detection_model, ground_truth_map, auth_paths) -> TripleStreamRetrievalNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_map, _ = train_test_split(
        ground_truth_map, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE
    )
    val_t    = get_transforms("val")
    triplet_ds = TripleStreamTripletDataset(train_map, auth_paths, val_t)
    triplet_loader = DataLoader(triplet_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialise from detection backbone weights
    model = TripleStreamRetrievalNet.from_detection_weights(detection_model).to(device)
    loss_fn = nn.TripletMarginLoss(margin=config.TRIPLET_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.RETRIEVAL_LR)

    print("\n=== Training Retrieval Model ===")
    for epoch in range(config.RETRIEVAL_EPOCHS):
        model.train()
        running_loss = 0.0
        for anchor, pos, neg in tqdm(triplet_loader, desc=f"Retrieval Epoch {epoch+1}/{config.RETRIEVAL_EPOCHS}"):
            optimizer.zero_grad()
            e_a = model(anchor[0].to(device), anchor[1].to(device), anchor[2].to(device))
            e_p = model(pos[0].to(device),    pos[1].to(device),    pos[2].to(device))
            e_n = model(neg[0].to(device),    neg[1].to(device),    neg[2].to(device))
            loss = loss_fn(e_a, e_p, e_n)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
        print(f"  Retrieval Loss: {running_loss / len(triplet_loader):.4f}")

    torch.save(model.state_dict(), config.RETRIEVAL_MODEL_PATH)
    print(f"\nRetrieval model saved → {config.RETRIEVAL_MODEL_PATH}")
    return model


# ── Faiss index ───────────────────────────────────────────────────────────

def build_faiss_index(retrieval_model, auth_paths, ground_truth_map) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_t  = get_transforms("val")
    triplet_helper = TripleStreamTripletDataset([], auth_paths, val_t)

    retrieval_model.eval()
    index        = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    all_embeddings, indexed_paths = [], []

    print("\n=== Building Faiss Index ===")
    with torch.no_grad():
        for path in tqdm(auth_paths, desc="Generating embeddings"):
            try:
                rgb, srm, ela = triplet_helper._load_triplet(path)
                emb = retrieval_model(
                    rgb.unsqueeze(0).to(device),
                    srm.unsqueeze(0).to(device),
                    ela.unsqueeze(0).to(device),
                ).cpu().numpy()
                all_embeddings.append(emb)
                indexed_paths.append(path)
            except Exception:
                continue

    if all_embeddings:
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        with open(config.FAISS_PATHS_PKL, "wb") as f:
            pickle.dump(indexed_paths, f)
        print(f"Faiss index saved → {config.FAISS_INDEX_PATH}  ({index.ntotal} vectors)")
    else:
        print("No embeddings generated — Faiss index not built.")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Triple-Stream models.")
    parser.add_argument("--experiment", choices=["pgf", "gtf"], required=True,
                        help="'pgf' = Programmatic Forgeries, 'gtf' = CASIA Ground Truth")
    args = parser.parse_args()

    import glob
    from features import generate_forensic_features

    if args.experiment == "pgf":
        ground_truth_map = prepare_pgf_dataset()
    else:
        ground_truth_map = prepare_gtf_dataset()

    generate_forensic_features()

    auth_paths = glob.glob(os.path.join(config.AUTH_DIR, "*.*"))

    print("\n=== Training Detection Model ===")
    det_model = train_detection_model(ground_truth_map)

    print("\n=== Training Retrieval Model ===")
    ret_model = train_retrieval_model(det_model, ground_truth_map, auth_paths)

    build_faiss_index(ret_model, auth_paths, ground_truth_map)
    print("\nAll training complete.")
