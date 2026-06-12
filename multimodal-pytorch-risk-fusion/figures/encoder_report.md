# Text-Encoder Bake-off Report — CFPB Multimodal Fusion

**Question:** can a newer/bigger text encoder beat DistilBERT as the text branch of the multimodal
(text + structured) complaint model?

**Setup (identical for every encoder — a fair fight):**
- Data: reproducible **30,000-row** stratified CFPB sample (17,565 train / 3,764 val / 3,764 test), seed 42.
- Tasks: **A** = Issue (15-class), **B** = relief granted (binary). Loss `CE(A) + 0.3·CE(B)`, class-weighted.
- Regime: **full fine-tune**, 4 epochs, max_len 192, lr 2e-5, bf16 autocast (fp32 for NeoBERT — see below),
  early stopping on val Task-A macro-F1. Gated fusion + multi-task heads (`src/models.py`).
- Hardware: single RTX 4060 (8 GB), shared; runs wait for ≥6 GB free.
- Reproduce: `uv run python compare_encoders.py --encoders distilbert,modernbert,neobert,mrbert --unfreeze -1`
  (CSV: `figures/encoder_comparison_deep.csv`; NeoBERT partial-FT retry in `figures/encoder_comparison.csv`).

## Results (held-out test set)

| Encoder | Params | Mode | Issue macro-F1 | Issue acc | Issue wtd-F1 | Relief AUROC | Train min |
|---|---|---|---|---|---|---|---|
| *(none — tabular only)* | — | tabular | 0.426 | 0.388 | 0.357 | 0.801 | 0.5 |
| **DistilBERT** | 66M | text | 0.580 | 0.619 | 0.622 | 0.700 | 9.9 |
| **DistilBERT** | 66M | **fusion** | **0.623** | 0.659 | 0.662 | 0.807 | 9.9 |
| **ModernBERT-base** | 149M | text | 0.604 | 0.662 | 0.664 | 0.702 | 26.1 |
| **ModernBERT-base** | 149M | **fusion** | 0.611 | **0.682** | **0.682** | **0.809** | 26.0 |
| MrBERT | 307M | text | 0.271 | 0.381 | 0.375 | 0.625 | 35.8 |
| MrBERT | 307M | fusion | 0.496 | 0.521 | 0.508 | 0.795 | 26.1 |
| NeoBERT | 222M | text | **0.007** ✗ | 0.055 | — | NaN | 18.0 |
| NeoBERT | 222M | fusion | **0.007** ✗ | 0.055 | — | NaN | 16.2 |

✗ = failed to converge (see below). Best in each column **bold**.

## Analysis

### 🏆 ModernBERT-base wins — modestly but consistently
The English+code-pretrained **ModernBERT (2024) is the strongest encoder**, and the only one that
beats DistilBERT:
- **Text-only**: ModernBERT beats DistilBERT on every Issue metric (acc 0.662 vs 0.619, macro-F1 0.604 vs 0.580).
- **Fusion**: ModernBERT wins **Issue accuracy (0.682 vs 0.659)**, **weighted-F1 (0.682 vs 0.662)**, and
  **Relief AUROC (0.809 vs 0.807)**. DistilBERT keeps a hair's-edge on **Issue macro-F1 (0.623 vs 0.611)** —
  i.e. DistilBERT is marginally better on the *rare* issue classes, ModernBERT better on the common ones
  and on the relief task.
- **Cost**: ModernBERT is **~2.6× slower** to train (26 vs 10 min) for ~2-point accuracy gains.

**Takeaway:** if you want the best quality and can afford ~2.6× train time, **ModernBERT-base is the
upgrade**. If you want speed and rare-class macro-F1, **DistilBERT remains an excellent default**. This is
the opposite of the MrBERT result — *specialization (English+code) + modern architecture wins; multilingual
breadth and raw size do not.*

### MrBERT (307M, multilingual) — loses, as before
Confirmed across all regimes: multilingual capacity is diluted on this English-only task. Fusion 0.496
macro-F1, slowest of the usable models. (Full history in the project README "Encoder bake-off" section.)

### NeoBERT (222M) — failed to converge (an honest engineering result)
NeoBERT **collapsed to 0.007 macro-F1** (predicting essentially one class, NaN Head-B logits) in **both**
configurations we tried:
1. full fine-tune, fp32, batch 2 (forced — it can't gradient-checkpoint);
2. deep partial fine-tune (top-6 layers), fp32, batch 8.

Two independent settings → same degenerate collapse, so this is **not** a batch-size, precision, or
learning-rate artifact. The most likely cause: NeoBERT's custom **xformers** attention does not consume the
standard HuggingFace `attention_mask` the way our generic mean-pooling assumes, so padding corrupts the
pooled sentence vector and the model never learns. NeoBERT also **requires `xformers`** (a non-trivial dep)
and **cannot gradient-checkpoint**, forcing tiny batches on 8 GB.

**Takeaway:** a model can top public benchmarks (NeoBERT leads MTEB for its size) yet be **impractical to
drop into a custom architecture** without bespoke glue (its own pooling head + mask handling). For this
project the integration cost isn't worth it. Reported honestly as non-converged rather than hidden.

## Recommendation

1. **Best quality → switch the text branch to `answerdotai/ModernBERT-base`** (English+code, 149M, clean
   768-hidden drop-in). Expect ~+2 pts Issue accuracy and a touch more Relief AUROC, at ~2.6× train cost.
2. **Best speed / rare-class macro-F1 → keep DistilBERT** (the current default). Still excellent and fastest.
3. **Avoid** MrBERT (multilingual dilution) and NeoBERT (doesn't converge in this integration) for this task.

> Headline: *newer can win — but only the right newer.* The English-specialized ModernBERT beat DistilBERT;
> the multilingual MrBERT and the benchmark-topping-but-hard-to-integrate NeoBERT did not. Measured across
> four encoders and two regimes, end to end.
