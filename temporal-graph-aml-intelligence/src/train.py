"""
train.py — Training loop, metrics, and ranking utilities for AML risk scoring.

Everything here is framing-agnostic: the metrics (PR-AUC, ROC-AUC, Recall@K,
Precision@K) take score arrays, so they work identically for the graph model, the
tabular baselines, the anomaly branch, and the hybrid meta-model — which keeps the
comparison honest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of all positives captured in the top-k highest-scored items."""
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    n_pos = int(y_true.sum())
    return float(y_true[order].sum() / n_pos) if n_pos else float("nan")


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k flagged items that are truly positive."""
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, ks=(50, 100, 200, 500)) -> dict:
    """Standard score-based metrics used everywhere in the notebook."""
    out = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(y_true, scores, k)
        out[f"precision@{k}"] = precision_at_k(y_true, scores, k)
    return out


def metrics_by_time(y_true, scores, time_steps) -> "list[dict]":
    """Per-time-step PR-AUC / ROC-AUC — surfaces temporal degradation."""
    rows = []
    for ts in sorted(np.unique(time_steps)):
        m = time_steps == ts
        if m.sum() < 5 or y_true[m].sum() == 0:
            continue
        rows.append({
            "time_step": int(ts),
            "n": int(m.sum()),
            "n_illicit": int(y_true[m].sum()),
            "pr_auc": float(average_precision_score(y_true[m], scores[m])),
            "roc_auc": float(roc_auc_score(y_true[m], scores[m])),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# GNN training (full-batch, transductive)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 5e-3
    weight_decay: float = 5e-4
    pos_weight: float | None = None     # set from class imbalance; weights the illicit class
    patience: int = 30                  # early stopping on val PR-AUC
    log_every: int = 20


@torch.no_grad()
def _scores(model, x, edge_index, mask):
    model.eval()
    logits = model(x, edge_index)
    p = torch.sigmoid(logits[mask]).float().cpu().numpy()
    return p


def train_gnn(model, data, device, cfg: TrainConfig, verbose: bool = True):
    """Train a node-classification GNN with early stopping on validation PR-AUC.

    Returns (best_state_dict, history) where history is a list of per-epoch dicts.
    """
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)

    pw = None if cfg.pos_weight is None else torch.tensor([cfg.pos_weight], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    y_val_np = data.y[data.val_mask].numpy()
    history, best_pr, best_state, since = [], -1.0, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask], pos_weight=pw)
        loss.backward()
        opt.step()

        val_scores = _scores(model, x, edge_index, val_mask)
        val_pr = float(average_precision_score(y_val_np, val_scores))
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_pr_auc": val_pr})

        if val_pr > best_pr:
            best_pr, since = val_pr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1

        if verbose and (epoch % cfg.log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  val_PR-AUC={val_pr:.4f}  best={best_pr:.4f}")
        if since >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val PR-AUC={best_pr:.4f})")
            break

    return best_state, history


@torch.no_grad()
def gnn_predict(model, data, device, mask):
    """Return (probabilities, embeddings) for the masked nodes."""
    model = model.to(device).eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    logits = model(x, edge_index)
    emb = model.embed(x, edge_index)
    p = torch.sigmoid(logits[mask]).float().cpu().numpy()
    e = emb[mask].float().cpu().numpy()
    return p, e
