"""
train.py — Multi-task training loop, metrics, and calibration utilities.

Shared by the full fusion model and the text-only / tabular-only ablations: they all expose the
same `forward(batch) -> {logits_a, logits_b, gate}` interface, so the loop and metrics are written
once and reused — which keeps the modality comparison apples-to-apples.

Training details:
  * joint multi-task loss = CE(head_A) + aux_weight * CE(head_B), both class-weighted,
  * bf16/fp16 mixed precision (autocast) with an optional GradScaler for fp16,
  * early stopping on validation macro-F1 of the primary task (Head A).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def amp_policy(device: torch.device):
    """Pick a stable mixed-precision dtype for the device (bf16 on Ada+, else fp16, else off)."""
    if device.type != "cuda":
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False     # dtype, use_amp, use_scaler
    return torch.float16, True, True


@dataclass
class TrainConfig:
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    aux_weight: float = 0.3          # weight on Head B (secondary task)
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 2
    head_b_metric: str = "auroc"     # 'auroc' if Head B is binary else 'macro_f1'
    force_fp32: bool = False         # disable AMP (some custom backbones NaN under bf16/fp16)


def move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device, amp_dtype, use_amp):
    """Collect probabilities + labels for both heads across a loader."""
    model.eval()
    pa, pb, ya, yb, gates = [], [], [], [], []
    for batch in loader:
        batch = move(batch, device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
        pa.append(F.softmax(out["logits_a"].float(), dim=1).cpu())
        pb.append(F.softmax(out["logits_b"].float(), dim=1).cpu())
        ya.append(batch["label_a"].cpu())
        yb.append(batch["label_b"].cpu())
        if out["gate"] is not None:
            gates.append(out["gate"].float().mean(dim=1).cpu())   # mean gate per example
    res = {
        "prob_a": torch.cat(pa).numpy(), "prob_b": torch.cat(pb).numpy(),
        "y_a": torch.cat(ya).numpy(), "y_b": torch.cat(yb).numpy(),
    }
    res["gate"] = torch.cat(gates).numpy() if gates else None
    return res


def compute_metrics(res, head_b_metric="auroc"):
    pa, ya = res["prob_a"], res["y_a"]
    pb, yb = res["prob_b"], res["y_b"]
    pred_a = pa.argmax(1)
    out = {
        "A_accuracy": accuracy_score(ya, pred_a),
        "A_macro_f1": f1_score(ya, pred_a, average="macro"),
        "A_weighted_f1": f1_score(ya, pred_a, average="weighted"),
    }
    if head_b_metric == "auroc" and pb.shape[1] == 2:
        # Guard against NaN predictions (e.g. a backbone that diverged) so eval doesn't crash.
        out["B_auroc"] = roc_auc_score(yb, pb[:, 1]) if np.isfinite(pb).all() else float("nan")
    out["B_macro_f1"] = f1_score(yb, pb.argmax(1), average="macro")
    return out


def train_multitask(model, train_loader, val_loader, device, cfg: TrainConfig,
                    class_weights_a=None, class_weights_b=None, verbose=True):
    from transformers import get_linear_schedule_with_warmup

    if cfg.force_fp32:
        amp_dtype, use_amp, use_scaler = torch.float32, False, False
    else:
        amp_dtype, use_amp, use_scaler = amp_policy(device)
    model = model.to(device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grouped, lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = GradScaler(enabled=use_scaler)

    cw_a = class_weights_a.to(device) if class_weights_a is not None else None
    cw_b = class_weights_b.to(device) if class_weights_b is not None else None

    history, best, best_state, since = [], -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss_a = F.cross_entropy(out["logits_a"], batch["label_a"], weight=cw_a)
                loss_b = F.cross_entropy(out["logits_b"], batch["label_b"], weight=cw_b)
                loss = loss_a + cfg.aux_weight * loss_b
            if not torch.isfinite(loss):
                continue   # skip a non-finite step rather than poisoning all weights with NaN
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
            sched.step()
            running += loss.item()

        res = predict(model, val_loader, device, amp_dtype, use_amp)
        m = compute_metrics(res, cfg.head_b_metric)
        m["epoch"] = epoch; m["train_loss"] = running / len(train_loader)
        history.append(m)
        if verbose:
            extra = f"val_B_auroc={m.get('B_auroc', float('nan')):.4f}"
            print(f"epoch {epoch}: loss={m['train_loss']:.4f}  val_A_macroF1={m['A_macro_f1']:.4f}  "
                  f"val_A_acc={m['A_accuracy']:.4f}  {extra}")

        score = m["A_macro_f1"]
        if score > best:
            best, since = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best val A_macroF1={best:.4f})")
                break
    return best_state, history, (amp_dtype, use_amp)
