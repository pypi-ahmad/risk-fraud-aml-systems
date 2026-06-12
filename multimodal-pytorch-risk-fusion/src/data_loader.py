"""
data_loader.py — CFPB Consumer Complaint Database loaders for multimodal modelling.

The bulk CSV is ~8 GB / ~3.8M complaints with narratives, so we stream it in chunks and take a
reproducible stratified sample for tractable fine-tuning. We then build two modalities and two
targets, with explicit leakage controls.

Modalities
  * Text       : the consumer complaint narrative.
  * Structured : product (consolidated), sub-product, company, state, tags, and date parts —
                 all known at submission time.

Targets
  * Task A (multi-class) : Issue, restricted to the top-N most frequent issues.
  * Task B (binary)      : relief_granted — did the company give monetary or non-monetary relief?

Leakage controls
  * Post-outcome fields ("Company response to consumer", "Timely response?", "Company public
    response", "Date sent to company") are NEVER used as input features — Task B's label is *derived*
    from the response field but that field is excluded from X.
  * "Submitted via" is dropped: for narrative complaints it is ~constant ("Web") and carries no signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


CSV_PATH = "data/raw/cfpb/complaints.csv"

USECOLS = ["Date received", "Product", "Sub-product", "Issue", "Company",
           "State", "Tags", "Consumer complaint narrative", "Company response to consumer"]

# Collapse CFPB's evolving product taxonomy into stable canonical categories.
PRODUCT_MAP = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Payday loan, title loan, or personal loan": "Payday/personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Payday loan": "Payday/personal loan",
    "Consumer Loan": "Vehicle/consumer loan",
    "Vehicle loan or lease": "Vehicle/consumer loan",
    "Money transfer, virtual currency, or money service": "Money transfer/virtual currency",
    "Money transfers": "Money transfer/virtual currency",
    "Virtual currency": "Money transfer/virtual currency",
    "Credit card or prepaid card": "Credit/prepaid card",
    "Credit card": "Credit/prepaid card",
    "Prepaid card": "Credit/prepaid card",
    "Bank account or service": "Bank account",
    "Checking or savings account": "Bank account",
}

RELIEF_POSITIVE = {"Closed with monetary relief", "Closed with non-monetary relief"}

CAT_FEATURES = ["product", "sub_product", "company", "state", "tags", "year", "month", "dow"]


def load_cfpb_sample(path: str = CSV_PATH, n_target: int = 60_000, seed: int = 42,
                     chunksize: int = 200_000, narr_total_est: int = 3_800_000) -> pd.DataFrame:
    """Stream the CSV and return a reproducible random sample of narrative complaints."""
    frac = min(1.0, (n_target / narr_total_est) * 1.2)   # slight oversample, trimmed later
    parts = []
    for ch in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize, dtype=str):
        narr = ch["Consumer complaint narrative"]
        ch = ch[narr.notna() & (narr.str.strip() != "")]
        if len(ch):
            parts.append(ch.sample(frac=frac, random_state=seed))
    df = pd.concat(parts, ignore_index=True)
    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=seed).reset_index(drop=True)
    return df


def _cap_categories(s: pd.Series, allowed: set, other: str = "OTHER") -> pd.Series:
    return s.where(s.isin(allowed), other)


def clean_and_engineer(df: pd.DataFrame, top_issues: int = 15) -> pd.DataFrame:
    """Consolidate products, derive targets, engineer date parts, drop rows outside top issues."""
    df = df.copy()
    df["narrative"] = df["Consumer complaint narrative"].str.strip()

    df["product"] = df["Product"].map(PRODUCT_MAP).fillna(df["Product"]).fillna("UNK")
    df["sub_product"] = df["Sub-product"].fillna("UNK")
    df["company"] = df["Company"].fillna("UNK")
    df["state"] = df["State"].fillna("UNK")
    df["tags"] = df["Tags"].fillna("None")

    dt = pd.to_datetime(df["Date received"], errors="coerce")
    df["year"] = dt.dt.year.fillna(0).astype(int).astype(str)
    df["month"] = dt.dt.month.fillna(0).astype(int).astype(str)
    df["dow"] = dt.dt.dayofweek.fillna(0).astype(int).astype(str)

    # Task B target: relief granted (binary). Derived from response field, which is NOT a feature.
    df["label_b"] = df["Company response to consumer"].isin(RELIEF_POSITIVE).astype(int)

    # Task A target: Issue restricted to the top-N most frequent issues in the sample.
    df = df[df["Issue"].notna()].copy()
    keep = df["Issue"].value_counts().head(top_issues).index
    df = df[df["Issue"].isin(keep)].reset_index(drop=True)
    return df


@dataclass
class PreparedData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    encoder: OneHotEncoder
    issue_classes: list           # index -> issue name (Task A)
    struct_dim: int
    company_topk: set
    subproduct_topk: set


def prepare(df: pd.DataFrame, seed: int = 42, top_companies: int = 100,
            top_subproducts: int = 40, test_size: float = 0.15,
            val_size: float = 0.15) -> PreparedData:
    """Stratified split + fit a one-hot structured encoder on TRAIN only (no leakage)."""
    classes = sorted(df["Issue"].unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["label_a"] = df["Issue"].map(cls_to_idx)

    # Stratified train / val / test on Task A
    train, tmp = train_test_split(df, test_size=test_size + val_size,
                                  stratify=df["label_a"], random_state=seed)
    rel = test_size / (test_size + val_size)
    val, test = train_test_split(tmp, test_size=rel, stratify=tmp["label_a"], random_state=seed)

    # Cap high-cardinality categoricals using TRAIN frequencies only
    comp_top = set(train["company"].value_counts().head(top_companies).index)
    sub_top = set(train["sub_product"].value_counts().head(top_subproducts).index)
    for part in (train, val, test):
        part["company"] = _cap_categories(part["company"], comp_top)
        part["sub_product"] = _cap_categories(part["sub_product"], sub_top)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(train[CAT_FEATURES])
    struct_dim = encoder.transform(train[CAT_FEATURES][:1]).shape[1]

    return PreparedData(train.reset_index(drop=True), val.reset_index(drop=True),
                        test.reset_index(drop=True), encoder, classes, struct_dim,
                        comp_top, sub_top)


def structured_matrix(df: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
    return encoder.transform(df[CAT_FEATURES])


class CFPBDataset(Dataset):
    """Holds tokenised narratives + structured vectors + both task labels."""
    def __init__(self, encodings, structured, label_a, label_b):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.structured = torch.as_tensor(np.asarray(structured), dtype=torch.float32)
        self.label_a = torch.as_tensor(np.asarray(label_a), dtype=torch.long)
        self.label_b = torch.as_tensor(np.asarray(label_b), dtype=torch.long)

    def __len__(self):
        return len(self.label_a)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "structured": self.structured[i],
            "label_a": self.label_a[i],
            "label_b": self.label_b[i],
        }


def make_collate(tokenizer):
    """Dynamic padding for text; plain stacking for structured + labels."""
    def collate(batch):
        feats = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        padded = tokenizer.pad(feats, return_tensors="pt")
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "structured": torch.stack([b["structured"] for b in batch]),
            "label_a": torch.stack([b["label_a"] for b in batch]),
            "label_b": torch.stack([b["label_b"] for b in batch]),
        }
    return collate


def class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
