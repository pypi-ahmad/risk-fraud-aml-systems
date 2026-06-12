# Multimodal PyTorch Risk Fusion (CFPB Complaints)

A **multimodal** PyTorch model that fuses the **complaint narrative** (DistilBERT) with **structured
metadata** (an MLP) via a **learned gate**, and predicts **two tasks at once**: the complaint *Issue*
(multi-class) and whether the company grants *relief* (binary). Built to be more capable — and more
honestly evaluated — than a text-only or tabular-only model.

## Dataset & download

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
— ~3.8M complaints with a free-text narrative plus structured fields.

```bash
mkdir -p data/raw/cfpb
wget -O data/raw/cfpb/complaints.csv.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip
unzip data/raw/cfpb/complaints.csv.zip -d data/raw/cfpb
```
The notebook streams the 8 GB CSV in chunks and takes a reproducible **stratified ~50k sample** for
tractable fine-tuning on an 8 GB GPU. (Optional intent benchmark: `PolyAI/banking77` via 🤗 `datasets`.)

## Preprocessing pipeline (`src/data_loader.py`)

1. **Stream + sample** narrative complaints (chunked read, fixed seed).
2. **Consolidate** CFPB's evolving product taxonomy into stable canonical products.
3. **Targets:** Task A = *Issue* (top-15); Task B = *relief_granted* (monetary or non-monetary relief).
4. **Structured features** (submission-time only): product, sub-product, company (top-100 + OTHER),
   state, tags, and date parts (year/month/day-of-week), one-hot encoded (fit on **train only**).
5. **Leakage controls:** post-outcome fields (*Company response to consumer*, *Timely response?*,
   *Company public response*, *Date sent to company*) are **excluded from features**; *Submitted via*
   is dropped (≈constant "Web" for narratives). Splits are **stratified on Task A**.

## Model architecture (`src/models.py`)

```
  narrative ─► DistilBERT (mask-mean-pooled) ─► proj(256) ─┐
                                                           ├─ gated fusion ─► fused(256) ─┬─► Head A: Issue (15-way)
  structured ─► MLP ─────────────────────────► h(256) ────┘   g=σ(W[h_t;h_s])             └─► Head B: relief (binary)
                                                              fused = g·text + (1−g)·struct
```
- **Gated fusion** learns a per-example blend; the gate is read back out for interpretability.
- **Multi-task**: shared trunk, two heads; loss = `CE(A) + 0.3·CE(B)`, both class-weighted.
- **8 GB budget**: DistilBERT embeddings + lower layers frozen; top-2 transformer layers fine-tuned;
  **bf16** mixed precision; batch size auto-scaled to free VRAM (falls back to CPU if the GPU is busy).

## Training & evaluation summary

Three models trained with identical loop/loss/data for a fair **ablation**: `fusion`, `text`-only,
`tabular`-only. Reported on the held-out test split:

- Task A: **macro-F1 / weighted-F1 / accuracy**; Task B: **AUROC / macro-F1**.
- **Calibration**: reliability curve + Expected Calibration Error for the fusion model.
- **Error slices**: Task-A macro-F1 by **product, company, state** (weakest groups surfaced).
- **Gate interpretability**: average text-reliance per product.

> **Key finding:** fusion is the best model on **both** tasks, driven by *complementarity*. On
> *Issue* (Task A), text-only and tabular-only tie alone (~0.51 macro-F1) but fusion reaches ~0.60.
> On *Relief* (Task B), **structured metadata dominates** (AUROC ~0.80 vs ~0.68 for text) — relief
> depends more on *who* the complaint targets than on its wording — and fusion only edges it out
> (~0.81). The learned gate's text-reliance varies sensibly by product. Exact numbers regenerate when
> you run the notebook (see `figures/ablation.csv`).

## Setup & run

```bash
git clone https://github.com/pypi-ahmad/multimodal-pytorch-risk-fusion.git
cd multimodal-pytorch-risk-fusion
```


```bash
cd multimodal-pytorch-risk-fusion
uv sync
# download the data (above), then:
uv run python generate_notebook.py            # builds the notebook
uv run jupyter lab multimodal_pytorch_risk_fusion.ipynb
# or headless end-to-end:
uv run jupyter nbconvert --to notebook --execute --inplace \
    multimodal_pytorch_risk_fusion.ipynb
```

Inference with the trained model (`inference.py`):
```bash
# single complaint
uv run python inference.py \
    --narrative "I was charged a late fee even though I paid on time." \
    --product "Credit card or prepaid card" --company "BIG BANK" --state CA

# batch (CSV with narrative + structured columns)
uv run python inference.py --input complaints.csv --output preds.csv
```

## Encoder bake-off: DistilBERT vs ModernBERT vs MrBERT vs NeoBERT

`compare_encoders.py` benchmarks alternative text encoders in the fusion model under one identical
regime (30k sample, full fine-tune, 4 epochs, max_len 192). **Fusion** results on the test set:

| Encoder | Params | Issue macro-F1 | Issue acc | Relief AUROC | Train min | |
|---|---|---|---|---|---|---|
| **ModernBERT-base** (2024, En+code) | 149M | 0.611 | **0.682** | **0.809** | 26 | 🏆 best quality |
| **DistilBERT** (default) | 66M | **0.623** | 0.659 | 0.807 | 10 | ⚡ best speed / rare-class |
| MrBERT (2026, multilingual) | 307M | 0.496 | 0.521 | 0.795 | 26 | dilution → loses |
| NeoBERT (2025) | 222M | 0.007 ✗ | — | NaN | 18 | failed to converge |

**Findings (full write-up: [`figures/encoder_report.md`](encoder_report.md)):**

1. **ModernBERT-base wins overall** — best Issue *accuracy* (0.682 vs 0.659), weighted-F1, and Relief
   AUROC; DistilBERT keeps a hair's-edge on *macro*-F1 (rare classes). English+code specialization +
   modern architecture beats DistilBERT, at ~2.6× train cost. **The right newer model can win.**
2. **MrBERT (multilingual, 307M) loses** — capacity diluted across 35 languages on this English-only
   task; lost in both frozen and full-FT regimes.
3. **NeoBERT failed to converge** — collapsed to 0.007 macro-F1 (one-class, NaN Head-B) in *two*
   independent configs. Its custom **xformers** attention doesn't consume the standard HF
   `attention_mask` as our generic mean-pooling assumes; it also can't gradient-checkpoint and needs
   `xformers`. Benchmark-topping ≠ drop-in-able.

**Recommendation:** keep **DistilBERT** as the fast default; switch to **`answerdotai/ModernBERT-base`**
for best quality. Avoid MrBERT/NeoBERT here. *(Wiring is additive — `src/models.py` gained an optional
`trust_remote_code` flag; the notebook/inference default is unchanged.)*

## Layout

```
multimodal-pytorch-risk-fusion/
├── multimodal_pytorch_risk_fusion.ipynb   # main deliverable (executed)
├── generate_notebook.py                    # builds the notebook
├── inference.py                            # single + batch prediction (Issue + relief + gate)
├── src/
│   ├── data_loader.py   # CFPB streaming sample, cleaning, targets, encoders, Dataset
│   ├── models.py        # TextEncoder, StructuredEncoder, GatedFusion, MultiModalRiskNet
│   └── train.py         # multi-task loop, bf16 AMP, metrics, calibration helpers
├── checkpoints/         # fusion_best.pt + preprocess.pkl  [gitignored]
├── figures/             # generated plots + ablation.csv
└── pyproject.toml
```

## Caveats

- *Issue* is partly determined by *Product*, so Task A is easier than pure text classification — the
  ablation makes this explicit instead of hiding it.
- ~50k sample, frozen lower DistilBERT layers, single seed; narratives are redacted (`XXXX`) and
  English-only. Production needs full fine-tuning, multi-seed CIs, a temporal split with drift
  monitoring, and temperature scaling for calibration.
