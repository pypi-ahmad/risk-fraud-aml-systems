"""
data_loader.py — Reusable loaders for the Elliptic Bitcoin dataset.

The Elliptic dataset is a temporal transaction graph:
  * 203,769 nodes (Bitcoin transactions), each tagged with a discrete time step 1..49
  * 234,355 directed edges (a payment flow between transactions)
  * 165 anonymised node features (local + aggregated), plus the time step
  * labels: 1 = illicit, 2 = licit, "unknown" = unlabelled

We map labels to {illicit: 1, licit: 0, unknown: -1} and keep unknown nodes in the
graph (they still carry message-passing signal) but exclude them from loss / metrics.

Temporal split (no future leakage): train ts <= 30, val 31..34, test >= 35.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_ROOT = Path("data/raw/elliptic/elliptic_bitcoin_dataset")

ILLICIT, LICIT, UNKNOWN = 1, 0, -1


@dataclass
class EllipticData:
    """Container for the assembled graph + temporal masks."""
    x: torch.Tensor              # [N, F] float node features (time step excluded)
    edge_index: torch.Tensor     # [2, E] long, directed edges (remapped to 0..N-1)
    y: torch.Tensor              # [N] long in {0,1} for labelled, -1 for unknown
    time_step: torch.Tensor      # [N] long, 1..49
    feature_names: list[str]
    train_mask: torch.Tensor     # [N] bool, labelled nodes with ts <= train_max_ts
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        return self.x.size(1)


def load_elliptic_frames(root: Path | str = DEFAULT_ROOT):
    """Load the three raw CSVs and return (features, classes, edges) DataFrames."""
    root = Path(root)
    feats = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")

    # Column 0 = txId, column 1 = time step, columns 2..166 = 165 features.
    feat_cols = [f"feat_{i}" for i in range(feats.shape[1] - 2)]
    feats.columns = ["txId", "time_step"] + feat_cols
    return feats, classes, edges, feat_cols


def build_graph(
    root: Path | str = DEFAULT_ROOT,
    train_max_ts: int = 30,
    val_max_ts: int = 34,
) -> EllipticData:
    """Assemble the Elliptic graph as tensors with temporal train/val/test masks."""
    feats, classes, edges, feat_cols = load_elliptic_frames(root)

    # ── Map labels: '1'->illicit(1), '2'->licit(0), 'unknown'->-1 ───────────────
    label_map = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    classes = classes.copy()
    classes["y"] = classes["class"].astype(str).map(label_map)
    feats = feats.merge(classes[["txId", "y"]], on="txId", how="left")
    feats["y"] = feats["y"].fillna(UNKNOWN).astype(int)

    # ── Contiguous node id remapping (txId -> 0..N-1) ──────────────────────────
    feats = feats.reset_index(drop=True)
    id_to_idx = {tx: i for i, tx in enumerate(feats["txId"].to_numpy())}

    x = torch.tensor(feats[feat_cols].to_numpy(dtype=np.float32))
    y = torch.tensor(feats["y"].to_numpy(dtype=np.int64))
    time_step = torch.tensor(feats["time_step"].to_numpy(dtype=np.int64))

    # ── Edges: drop any endpoint not in the node set, remap to indices ─────────
    e = edges.copy()
    e = e[e["txId1"].isin(id_to_idx) & e["txId2"].isin(id_to_idx)]
    src = e["txId1"].map(id_to_idx).to_numpy()
    dst = e["txId2"].map(id_to_idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # ── Temporal masks over LABELLED nodes only ────────────────────────────────
    labelled = y != UNKNOWN
    train_mask = labelled & (time_step <= train_max_ts)
    val_mask = labelled & (time_step > train_max_ts) & (time_step <= val_max_ts)
    test_mask = labelled & (time_step > val_max_ts)

    return EllipticData(
        x=x, edge_index=edge_index, y=y, time_step=time_step,
        feature_names=feat_cols,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Add reverse edges so message passing flows both ways (common for Elliptic)."""
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)


def tabular_split(data: EllipticData):
    """Return (X, y) numpy arrays per split for non-graph baselines."""
    x = data.x.numpy()
    y = data.y.numpy()
    out = {}
    for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        m = mask.numpy()
        out[name] = (x[m], y[m], data.time_step.numpy()[m])
    return out
