"""
models.py — Multimodal (text + structured) fusion network with multi-task heads.

Architecture
------------
                 ┌──────────────┐
   narrative ──► │  DistilBERT  │──► [CLS] ─► proj ─┐
                 └──────────────┘                   │   gated fusion
                 ┌──────────────┐                   ├──► fused ─┬─► Head A (issue class)
   structured ─► │  Tabular MLP │──► h_struct ──────┘           └─► Head B (ops/risk target)
                 └──────────────┘

* **Text encoder**: DistilBERT, mean-pooled over tokens (mask-aware), projected to `fusion_dim`.
* **Structured encoder**: an MLP over engineered tabular features → `fusion_dim`.
* **Gated fusion**: a learned gate `g = σ(W[h_text; h_struct])` blends the two modalities
  `fused = g * h_text + (1 - g) * h_struct`, so the model can *learn* how much to trust each
  modality per example (and we can read the gate back out for interpretability).
* **Multi-task heads**: Head A (multi-class issue) and Head B (secondary target) share the fused
  representation — the auxiliary task regularises the shared trunk.

The encoders are deliberately swappable so the notebook can run **text-only** and **tabular-only**
ablations by reusing the same submodules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask (ignores padding)."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 fusion_dim: int = 256, dropout: float = 0.2,
                 trust_remote_code: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        hid = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean(out.last_hidden_state, attention_mask)
        return self.proj(pooled)


class StructuredEncoder(nn.Module):
    def __init__(self, in_dim: int, fusion_dim: int = 256,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """Learned per-example gate that blends two modality vectors."""
    def __init__(self, fusion_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * fusion_dim, fusion_dim)

    def forward(self, h_text, h_struct):
        g = torch.sigmoid(self.gate(torch.cat([h_text, h_struct], dim=-1)))
        fused = g * h_text + (1 - g) * h_struct
        return fused, g            # return gate for interpretability


class MultiModalRiskNet(nn.Module):
    """
    mode = 'fusion'  -> text + structured + gated fusion (full model)
           'text'    -> text encoder only (ablation baseline)
           'tabular' -> structured encoder only (ablation baseline)
    """
    def __init__(self, struct_dim: int, num_classes_a: int, num_classes_b: int,
                 model_name: str = "distilbert-base-uncased", fusion_dim: int = 256,
                 dropout: float = 0.2, mode: str = "fusion",
                 trust_remote_code: bool = False):
        super().__init__()
        assert mode in ("fusion", "text", "tabular")
        self.mode = mode
        if mode in ("fusion", "text"):
            self.text_enc = TextEncoder(model_name, fusion_dim, dropout, trust_remote_code)
        if mode in ("fusion", "tabular"):
            self.struct_enc = StructuredEncoder(struct_dim, fusion_dim, dropout=dropout)
        if mode == "fusion":
            self.fusion = GatedFusion(fusion_dim)

        self.head_a = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes_a))
        self.head_b = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes_b))

    def encode(self, batch):
        """Return (fused_representation, gate_or_None)."""
        if self.mode == "text":
            return self.text_enc(batch["input_ids"], batch["attention_mask"]), None
        if self.mode == "tabular":
            return self.struct_enc(batch["structured"]), None
        h_text = self.text_enc(batch["input_ids"], batch["attention_mask"])
        h_struct = self.struct_enc(batch["structured"])
        return self.fusion(h_text, h_struct)

    def forward(self, batch):
        fused, gate = self.encode(batch)
        return {"logits_a": self.head_a(fused), "logits_b": self.head_b(fused), "gate": gate}

    def freeze_text_backbone(self, n_trainable_top_layers: int = 2):
        """Freeze the text backbone's embeddings + all but the top-k transformer layers.

        On an 8 GB GPU (often shared) this cuts activation/gradient memory and speeds training a
        lot, while still letting the top layers adapt to complaint language. No-op for the
        tabular-only model. Architecture-agnostic: locates the transformer-layer ModuleList for
        DistilBERT (`transformer.layer`), BERT/RoBERTa (`encoder.layer`), and
        ModernBERT/MrBERT (`layers`).
        """
        if not hasattr(self, "text_enc"):
            return
        bb = self.text_enc.backbone
        for p in bb.parameters():
            p.requires_grad = False

        layers = None
        for attr in ("transformer.layer", "encoder.layer", "layers", "transformer_encoder"):
            obj = bb
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, nn.ModuleList):
                layers = obj
                break
        if layers is None:
            # Unknown layout: unfreeze the whole backbone rather than silently train nothing.
            for p in bb.parameters():
                p.requires_grad = True
            return
        for layer in layers[len(layers) - n_trainable_top_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
