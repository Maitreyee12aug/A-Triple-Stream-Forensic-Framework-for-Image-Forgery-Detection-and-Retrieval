# -*- coding: utf-8 -*-
"""
evaluate.py — Quantitative evaluation of both models.

    Detection : Confusion matrix, accuracy on the validation split.
    Retrieval : Recall@K (K=1, 5, 10) on the validation split.

Usage:
    python evaluate.py --experiment pgf
    python evaluate.py --experiment gtf
"""

import argparse
import glob
import os
import pickle

import faiss
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataset import (
    TripleStreamDataset,
    TripleStreamTripletDataset,
    get_transforms,
    prepare_pgf_dataset,
    prepare_gtf_dataset,
)
from models import TripleStreamForgeryDetector, TripleStreamRetrievalNet


def load_detection_model(device) -> TripleStreamForgeryDetector:
    model = TripleStreamForgeryDetector().to(device)
    model.load_state_dict(torch.load(config.DETECTION_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_retrieval_model(device) -> TripleStreamRetrievalNet:
    model = TripleStreamRetrievalNet().to(device)
    model.load_state_dict(torch.load(config.RETRIEVAL_MODEL_PATH, map_location=device))
    model.eval()
    return model


# ── Detection evaluation ──────────────────────────────────────────────────

def evaluate_detection(ground_truth_map, experiment: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_map = train_test_split(ground_truth_map, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE)

    val_files, val_labels = [], []
    for pair in val_map:
        val_files  += [pair["anchor_path"], pair["fake_path"]]
        val_labels += [0, 1]

    val_t  = get_transforms("val")
    val_ds = TripleStreamDataset(val_files, val_labels, config.SRM_DATASET_DIR, config.ELA_DATASET_DIR, val_t)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = load_detection_model(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for (rgb, srm, ela), labels in tqdm(val_loader, desc="Detection evaluation"):
            out = model(rgb.to(device), srm.to(device), ela.to(device))
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n=== Detection Results ({experiment.upper()}) ===")
    print(f"  Accuracy: {acc*100:.2f}%")

    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Authentic", "Fake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Detection Confusion Matrix — {experiment.upper()}")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUTS_DIR, f"confusion_matrix_{experiment}.png"), dpi=150)
    plt.show()
    print(f"Confusion matrix saved → outputs/confusion_matrix_{experiment}.png")


# ── Retrieval evaluation ──────────────────────────────────────────────────

def evaluate_retrieval(ground_truth_map, auth_paths, experiment: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Faiss index
    if not os.path.exists(config.FAISS_INDEX_PATH):
        print(f"Faiss index not found at {config.FAISS_INDEX_PATH}. Run train.py first.")
        return
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.FAISS_PATHS_PKL, "rb") as f:
        indexed_paths = pickle.load(f)
    print(f"Faiss index loaded: {index.ntotal} vectors.")

    _, val_map = train_test_split(ground_truth_map, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE)

    model = load_retrieval_model(device)
    val_t = get_transforms("val")
    helper = TripleStreamTripletDataset([], auth_paths, val_t)

    query_embeddings, query_anchor_paths = [], []
    print("\nGenerating query embeddings …")
    with torch.no_grad():
        for pair in tqdm(val_map, desc="Query embeddings"):
            try:
                rgb, srm, ela = helper._load_triplet(pair["fake_path"])
                emb = model(
                    rgb.unsqueeze(0).to(device),
                    srm.unsqueeze(0).to(device),
                    ela.unsqueeze(0).to(device),
                ).cpu().numpy()
                query_embeddings.append(emb)
                query_anchor_paths.append(pair["anchor_path"])
            except Exception:
                continue

    if not query_embeddings:
        print("No query embeddings generated.")
        return

    query_embeddings = np.vstack(query_embeddings)
    k_max  = max(config.RECALL_K_VALUES)
    dists, idxs = index.search(query_embeddings, k_max)

    recalls = {k: 0 for k in config.RECALL_K_VALUES}
    total   = len(query_anchor_paths)

    for i in range(total):
        retrieved = [indexed_paths[j] for j in idxs[i]]
        for k in config.RECALL_K_VALUES:
            if query_anchor_paths[i] in retrieved[:k]:
                recalls[k] += 1

    print(f"\n=== Retrieval Results ({experiment.upper()}) ===")
    for k in config.RECALL_K_VALUES:
        pct = recalls[k] / total * 100
        print(f"  Recall@{k:>2}: {pct:.2f}%")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Triple-Stream framework.")
    parser.add_argument("--experiment", choices=["pgf", "gtf"], required=True)
    args = parser.parse_args()

    if args.experiment == "pgf":
        ground_truth_map = prepare_pgf_dataset()
    else:
        ground_truth_map = prepare_gtf_dataset()

    auth_paths = glob.glob(os.path.join(config.AUTH_DIR, "*.*"))

    evaluate_detection(ground_truth_map, args.experiment)
    evaluate_retrieval(ground_truth_map, auth_paths, args.experiment)
