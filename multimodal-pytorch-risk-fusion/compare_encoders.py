#!/usr/bin/env python
"""
compare_encoders.py — Fair head-to-head: DistilBERT vs MrBERT as the text encoder
in the multimodal fusion model.

Trains, on ONE identical CFPB sample and split, the modality ablation (tabular / text / fusion)
for each text encoder, and prints an after-vs-before table. The tabular-only model is
encoder-independent, so it is trained once.

    uv run python compare_encoders.py

Results are saved to figures/encoder_comparison.csv.

The 307M MrBERT needs a few GB of VRAM, so we wait for the (often shared) GPU to free up before
starting, and scale batch size to the available headroom.
"""

import sys, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(".")
from src.data_loader import (load_cfpb_sample, clean_and_engineer, prepare, structured_matrix,
                             CFPBDataset, make_collate, class_weights)
from src.models import MultiModalRiskNet
from src.train import set_seed, train_multitask, predict, compute_metrics, TrainConfig

# name -> (hf_id, trust_remote_code)
ENCODERS = {
    "distilbert": ("distilbert-base-uncased", False),
    "modernbert": ("answerdotai/ModernBERT-base", False),
    "neobert":    ("chandar-lab/NeoBERT", True),
    "mrbert":     ("BSC-LT/MrBERT", False),
}
SEED = 42


def wait_for_gpu(min_free_gb=4.0, timeout_s=43200, poll_s=30):
    """Poll until the GPU has enough free memory; return (device, batch_size).

    We WAIT for the (shared) GPU rather than force a doomed run — MrBERT (307M) cannot train in
    <~4 GB. If the GPU never frees up within the (long) timeout, exit cleanly with a message so the
    user can simply re-run later; we never kick off an attempt that would OOM or hog the card.
    """
    if not torch.cuda.is_available():
        print("No CUDA available — exiting; this comparison needs a GPU for MrBERT.")
        sys.exit(0)
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free >= min_free_gb:
            bs = 32 if free >= 7 else (16 if free >= 5 else 8)
            print(f"GPU has {free:.1f} GB free — using cuda, batch={bs}", flush=True)
            return torch.device("cuda"), bs
        if time.time() - t0 > timeout_s:
            print(f"GPU still busy ({free:.1f} GB free) after {timeout_s//3600}h — exiting cleanly. "
                  f"Re-run `uv run python compare_encoders.py` when the GPU is free.", flush=True)
            sys.exit(0)
        print(f"  only {free:.1f} GB free (<{min_free_gb}); waiting {poll_s}s for the shared GPU…", flush=True)
        time.sleep(poll_s)


def build_loaders(frame_dict, tokenizer, encoder_obj, struct_cache, batch, device, max_len):
    def enc(texts): return tokenizer(list(texts), truncation=True, max_length=max_len)
    loaders = {}
    for split, frame in frame_dict.items():
        ds = CFPBDataset(enc(frame["narrative"]), struct_cache[split], frame["label_a"], frame["label_b"])
        loaders[split] = DataLoader(ds, batch_size=batch, shuffle=(split == "train"),
                                    collate_fn=make_collate(tokenizer), num_workers=2,
                                    pin_memory=(device.type == "cuda"),
                                    drop_last=(split == "train"))   # avoid size-1 BatchNorm crash at small batch
    return loaders


def train_one(mode, model_name, loaders, prep, num_a, cw_a, cw_b, device, epochs,
              lr=3e-5, unfreeze=2, grad_checkpoint=False, trust_remote_code=False,
              force_fp32=False):
    set_seed(SEED)
    model = MultiModalRiskNet(prep.struct_dim, num_a, 2, mode=mode,
                              model_name=model_name, fusion_dim=256,
                              trust_remote_code=trust_remote_code)
    if unfreeze == -1:
        pass  # full fine-tune: leave the whole backbone trainable
    else:
        model.freeze_text_backbone(unfreeze)
    if grad_checkpoint and hasattr(model, "text_enc"):
        # Trade compute for memory so a full fine-tune fits in 8 GB. Best-effort: some custom
        # backbones (e.g. NeoBERT) don't implement checkpointing — fall back to a smaller batch.
        try:
            model.text_enc.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.text_enc.backbone.config.use_cache = False
        except Exception as e:
            print(f"    (grad-checkpointing unavailable for {model_name}: {type(e).__name__})", flush=True)
    cfg = TrainConfig(epochs=epochs, lr=lr, aux_weight=0.3, patience=2,
                      head_b_metric="auroc", force_fp32=force_fp32)
    t0 = time.time()
    best_state, _, amp = train_multitask(model, loaders["train"], loaders["val"], device, cfg,
                                         class_weights_a=cw_a, class_weights_b=cw_b, verbose=False)
    model.load_state_dict(best_state)
    res = predict(model, loaders["test"], device, amp[0], amp[1])
    m = compute_metrics(res, "auroc")
    m["minutes"] = round((time.time() - t0) / 60, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="sample size")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--unfreeze", type=int, default=-1, help="top layers to train; -1 = full fine-tune")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--encoders", default="distilbert,modernbert,neobert",
                    help="comma-separated subset of: " + ",".join(ENCODERS))
    args = ap.parse_args()
    selected = [e.strip() for e in args.encoders.split(",") if e.strip()]
    for e in selected:
        assert e in ENCODERS, f"unknown encoder '{e}' (choose from {list(ENCODERS)})"
    regime = "full fine-tune" if args.unfreeze == -1 else f"top-{args.unfreeze} layers"
    print(f"REGIME: {regime} | epochs={args.epochs} | max_len={args.max_len} | "
          f"lr={args.lr} | grad_checkpoint={args.grad_checkpoint}")

    set_seed(SEED)
    print(f"Loading CFPB sample (n={args.n}) …")
    df = clean_and_engineer(load_cfpb_sample(n_target=args.n, seed=SEED), top_issues=15)
    prep = prepare(df, seed=SEED, top_companies=100, top_subproducts=40)
    num_a = len(prep.issue_classes)
    frames = {"train": prep.train, "val": prep.val, "test": prep.test}
    struct_cache = {s: structured_matrix(f, prep.encoder) for s, f in frames.items()}
    cw_a = class_weights(prep.train["label_a"], num_a)
    cw_b = class_weights(prep.train["label_b"], 2)
    print(f"classes={num_a}  struct_dim={prep.struct_dim}  "
          f"train/val/test={len(prep.train)}/{len(prep.val)}/{len(prep.test)}")

    device, batch = wait_for_gpu(min_free_gb=args.min_free_gb)
    if args.unfreeze == -1:
        batch = min(batch, 8)   # full fine-tune of 307M needs a small batch even at long context
        print(f"Full fine-tune: capping batch to {batch} (grad-checkpointing on).")

    common = dict(unfreeze=args.unfreeze, grad_checkpoint=args.grad_checkpoint, lr=args.lr)
    rows = []
    # tabular-only once (encoder-independent); reuse the distilbert tokenizer just to build loaders
    tok0 = AutoTokenizer.from_pretrained(ENCODERS["distilbert"][0])
    loaders0 = build_loaders(frames, tok0, None, struct_cache, batch, device, args.max_len)
    m = train_one("tabular", ENCODERS["distilbert"][0], loaders0, prep, num_a, cw_a, cw_b, device, args.epochs, **common)
    rows.append({"encoder": "—", "mode": "tabular", **m})
    print(f"[tabular] A_macroF1={m['A_macro_f1']:.4f} A_acc={m['A_accuracy']:.4f} B_auroc={m['B_auroc']:.4f} ({m['minutes']}m)", flush=True)

    # NeoBERT's custom xformers attention NaNs under bf16 autocast and can't gradient-checkpoint,
    # so we run it in fp32 at a small batch for stability.
    FP32_ENCODERS = {"neobert"}
    # NeoBERT can't grad-checkpoint: full FT needs batch 2 (unstable); partial FT is fine at batch 8.
    BATCH_OVERRIDE = {"neobert": 2 if args.unfreeze == -1 else 8}

    for key in selected:
        name, trc = ENCODERS[key]
        fp32 = key in FP32_ENCODERS
        eb = min(BATCH_OVERRIDE.get(key, batch), batch)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trc)
        loaders = build_loaders(frames, tok, None, struct_cache, eb, device, args.max_len)
        if eb != batch or fp32:
            print(f"  ({key}: batch={eb}, fp32={fp32} — stability settings for this arch)", flush=True)
        for mode in ("text", "fusion"):
            try:
                m = train_one(mode, name, loaders, prep, num_a, cw_a, cw_b, device, args.epochs,
                              trust_remote_code=trc, force_fp32=fp32, **common)
                rows.append({"encoder": key, "mode": mode, **m})
                print(f"[{key}/{mode}] A_macroF1={m['A_macro_f1']:.4f} A_acc={m['A_accuracy']:.4f} "
                      f"B_auroc={m['B_auroc']:.4f} ({m['minutes']}m)", flush=True)
            except Exception as e:
                # Never let one encoder abort the whole comparison (OOM, NaN, custom-arch quirks).
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[{key}/{mode}] FAILED ({type(e).__name__}: {str(e)[:120]}) — recording NaN.", flush=True)
                rows.append({"encoder": key, "mode": mode, "A_macro_f1": float("nan"),
                             "A_weighted_f1": float("nan"), "A_accuracy": float("nan"),
                             "B_auroc": float("nan"), "B_macro_f1": float("nan"), "minutes": 0})
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    out = pd.DataFrame(rows)[["encoder", "mode", "A_macro_f1", "A_weighted_f1", "A_accuracy",
                              "B_auroc", "B_macro_f1", "minutes"]].round(4)
    csv_name = "figures/encoder_comparison_deep.csv" if args.unfreeze == -1 else "figures/encoder_comparison.csv"
    out.to_csv(csv_name, index=False)
    print("\n================ ENCODER COMPARISON (test set) ================")
    print(out.to_string(index=False))

    def get(enc, mode, col):
        r = out[(out.encoder == enc) & (out["mode"] == mode)]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    print("\n──────── each encoder vs DistilBERT baseline ────────")
    for key in selected:
        if key == "distilbert":
            continue
        print(f"  {key} vs distilbert:")
        for col, lbl in [("A_macro_f1", "Issue macro-F1"), ("A_accuracy", "Issue acc"), ("B_auroc", "Relief AUROC")]:
            for mode in ("text", "fusion"):
                d, e = get("distilbert", mode, col), get(key, mode, col)
                print(f"    {mode:6s} {lbl:16s}: {d:.4f} -> {e:.4f}  (Δ {e-d:+.4f})")
    print(f"\nSaved {csv_name}")


if __name__ == "__main__":
    main()
