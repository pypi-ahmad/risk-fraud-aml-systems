#!/usr/bin/env python
"""
inference.py — Predict Issue (Task A) + relief probability (Task B) for CFPB-style complaints
with the trained multimodal fusion model.

Reuses the exact preprocessing pickled by the notebook (one-hot encoder + category caps), so a raw
complaint (narrative + structured fields) is featurised identically to training.

Usage
-----
Single complaint:
    uv run python inference.py \\
        --narrative "I was charged a late fee even though I paid on time..." \\
        --product "Credit card or prepaid card" --company "BIG BANK" --state CA

Batch (CSV with columns: narrative/Consumer complaint narrative, Product, Sub-product, Company,
State, Tags, Date received):
    uv run python inference.py --input complaints.csv --output preds.csv

Train the model first by running multimodal_pytorch_risk_fusion.ipynb.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))
from src.models import MultiModalRiskNet
from src.data_loader import PRODUCT_MAP


def _engineer_structured(df: pd.DataFrame, pre: dict) -> pd.DataFrame:
    """Reproduce the notebook's structured feature engineering for raw input rows."""
    def col(name, default):
        # Robust to a missing column in a batch CSV (df.get would return the scalar default).
        if name in df.columns:
            return df[name].fillna(default)
        return pd.Series([default] * len(df), index=df.index)

    out = pd.DataFrame(index=df.index)
    prod = col("Product", "UNK")
    out["product"] = prod.map(PRODUCT_MAP).fillna(prod).fillna("UNK")
    out["sub_product"] = col("Sub-product", "UNK")
    out["company"] = col("Company", "UNK")
    out["state"] = col("State", "UNK")
    out["tags"] = col("Tags", "None")
    dt = pd.to_datetime(col("Date received", None), errors="coerce")
    out["year"] = dt.dt.year.fillna(0).astype(int).astype(str)
    out["month"] = dt.dt.month.fillna(0).astype(int).astype(str)
    out["dow"] = dt.dt.dayofweek.fillna(0).astype(int).astype(str)
    # Apply the same top-K caps learned on train
    out["company"] = out["company"].where(out["company"].isin(pre["company_topk"]), "OTHER")
    out["sub_product"] = out["sub_product"].where(out["sub_product"].isin(pre["subproduct_topk"]), "OTHER")
    return out


def load(model_dir: Path, device):
    ckpt_path, pre_path = model_dir / "fusion_best.pt", model_dir / "preprocess.pkl"
    if not ckpt_path.exists() or not pre_path.exists():
        sys.exit(f"Missing {ckpt_path} or {pre_path}. Run the notebook to train + save first.")
    pre = pickle.load(open(pre_path, "rb"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiModalRiskNet(ckpt["struct_dim"], ckpt["num_a"], 2, mode="fusion")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tok, pre, ckpt


@torch.no_grad()
def predict(df, model, tok, pre, ckpt, device, max_len=128, batch_size=32, top_k=3):
    narr_col = "narrative" if "narrative" in df.columns else "Consumer complaint narrative"
    narratives = df[narr_col].fillna("").astype(str).tolist()
    struct = pre["encoder"].transform(_engineer_structured(df, pre)[pre["cat_features"]])
    classes = ckpt["issue_classes"]

    results = []
    for s in range(0, len(df), batch_size):
        chunk_txt = narratives[s:s + batch_size]
        enc = tok(chunk_txt, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                 "structured": torch.tensor(struct[s:s + batch_size], dtype=torch.float32, device=device)}
        out = model(batch)
        pa = F.softmax(out["logits_a"].float(), 1).cpu().numpy()
        pb = F.softmax(out["logits_b"].float(), 1).cpu().numpy()[:, 1]
        gate = out["gate"].float().mean(1).cpu().numpy() if out["gate"] is not None else [None] * len(pa)
        for i in range(len(pa)):
            topk = pa[i].argsort()[::-1][:top_k]
            results.append({
                "issue_top": [(classes[j], float(pa[i][j])) for j in topk],
                "relief_prob": float(pb[i]),
                "text_reliance": None if gate[i] is None else float(gate[i]),
            })
    return results


def main():
    ap = argparse.ArgumentParser(description="Multimodal complaint inference (Issue + relief).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--narrative", help="Single complaint narrative text")
    g.add_argument("--input", help="CSV of complaints (narrative + structured columns)")
    ap.add_argument("--product"); ap.add_argument("--sub_product")
    ap.add_argument("--company"); ap.add_argument("--state"); ap.add_argument("--tags")
    ap.add_argument("--date", help="Date received (YYYY-MM-DD)")
    ap.add_argument("--model-dir", default="checkpoints")
    ap.add_argument("--output", default=None, help="CSV path for batch predictions")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    model, tok, pre, ckpt = load(Path(args.model_dir), device)

    if args.narrative:
        df = pd.DataFrame([{"narrative": args.narrative, "Product": args.product,
                            "Sub-product": args.sub_product, "Company": args.company,
                            "State": args.state, "Tags": args.tags, "Date received": args.date}])
    else:
        df = pd.read_csv(args.input)
    preds = predict(df, model, tok, pre, ckpt, device, max_len=ckpt["max_len"], top_k=args.top_k)

    for i, p in enumerate(preds):
        print(f"\n[{i}] relief probability: {p['relief_prob']*100:.1f}%"
              + (f"   (text-reliance gate={p['text_reliance']:.2f})" if p['text_reliance'] is not None else ""))
        print("    predicted issue (top-k):")
        for name, prob in p["issue_top"]:
            print(f"      {prob*100:5.1f}%  {name}")

    if args.output:
        out = pd.DataFrame([{
            "predicted_issue": p["issue_top"][0][0],
            "issue_confidence": round(p["issue_top"][0][1], 4),
            "relief_prob": round(p["relief_prob"], 4),
            "text_reliance": None if p["text_reliance"] is None else round(p["text_reliance"], 4),
        } for p in preds])
        out.to_csv(args.output, index=False)
        print(f"\nWrote {len(out)} predictions → {args.output}")


if __name__ == "__main__":
    main()
