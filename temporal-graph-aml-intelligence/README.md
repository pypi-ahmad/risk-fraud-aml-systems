# Temporal-Graph AML Intelligence (Elliptic)

Anti-money-laundering as a **temporal transaction-graph** risk problem on the
[Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — combining
**graph neural networks** (GraphSAGE / GAT in PyTorch Geometric), an **unsupervised anomaly branch**
(Isolation Forest), and a **hybrid risk score**, evaluated honestly under a strict no-future-leakage
temporal split with an analyst-queue simulation.

## Project objective

Score each Bitcoin transaction's risk of being **illicit**, modelling the data as the temporal graph
it actually is rather than as flat tabular rows. The project deliberately compares **graph vs
non-graph** models on equal footing, exposes **concept drift** under a time-based split, and frames
results the way an AML operations team would consume them (Recall@budget, top-N case review).

## Dataset & download

**Elliptic** — 203,769 transaction nodes, 234,355 directed edges, 49 time steps, 165 anonymised
features. Labels: illicit (≈2%), licit (≈21%), unknown (≈77%, kept in-graph for message passing).

```bash
# requires Kaggle API credentials (~/.kaggle/ token or KAGGLE_API_TOKEN env var)
uv run kaggle datasets download -d ellipticco/elliptic-data-set \
    -p data/raw/elliptic --unzip
```

Optional tabular benchmark (IEEE-CIS, requires joining the competition first):
```bash
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee
unzip data/raw/ieee/ieee-fraud-detection.zip -d data/raw/ieee
```

## Setup & run

```bash
git clone https://github.com/pypi-ahmad/temporal-graph-aml-intelligence.git
cd temporal-graph-aml-intelligence
```


Requires [`uv`](https://docs.astral.sh/uv/). Python 3.12.10 and a CUDA 12.8 PyTorch build are pinned;
the notebook **auto-falls back to CPU** if the GPU lacks free memory (the graph is small enough to
train on CPU in a few minutes).

```bash
cd temporal-graph-aml-intelligence
uv sync
# download the data (above), then:
uv run python generate_notebook.py        # builds the notebook
uv run jupyter lab temporal_graph_aml_intelligence.ipynb
# or headless end-to-end:
uv run jupyter nbconvert --to notebook --execute --inplace \
    temporal_graph_aml_intelligence.ipynb
```

## What the notebook covers

1. Problem framing (AML as temporal graph risk) · 2. Data loading & schema checks · 3. Temporal
split · 4. Tabular baselines (LogReg / RandomForest / XGBoost) · 5. GraphSAGE & GAT (PyG) ·
6. Isolation Forest anomaly branch (raw features **and** learned embeddings) · 7. Hybrid risk score
(weighted blend + learned meta-model) · 8. Evaluation (PR-AUC, ROC-AUC, Recall@K / Precision@K,
**per-time-step** drift) · 9. Explainability (global feature importance + local ego-subgraph
evidence) · 10. Analyst queue simulation · 11. Recommendation & limitations.

## Key results (held-out future test set, ts ≥ 35)

| Model | PR-AUC | ROC-AUC | Recall@500 | Precision@500 |
|---|---|---|---|---|
| **XGBoost** (tabular) | **~0.79** | ~0.91 | ~0.46 | ~1.00 |
| Hybrid (meta-model) | ~0.785 | ~0.90 | ~0.46 | ~1.00 |
| GraphSAGE | ~0.43 | ~0.89 | ~0.24 | ~0.51 |
| GAT | ~0.25 | ~0.83 | ~0.10 | ~0.22 |
| IsoForest (GNN embeddings) | ~0.28 | ~0.87 | — | — |
| IsoForest (raw features) | ~0.04 | ~0.18 | — | — |

**Honest headline:** on Elliptic's temporal split, a **tuned XGBoost beats the GNNs** — the features
already encode 1-hop neighbourhood aggregates, so most graph signal is pre-baked, and boosting
exploits it better than a 2-layer GNN under drift. The graph's genuine contribution is its
**learned embedding**, which turns an otherwise-useless Isolation Forest into a usable ranker. The
**hybrid** matches the best single model and is more robust. Operated as a fixed-budget analyst
queue, the top ~250 alerts are ~100% precise and the top 500 capture ~46% of all illicit activity.

## Caveats

- **Concept drift is the dominant effect:** GraphSAGE goes from ~0.96 validation PR-AUC to ~0.43 on
  the future test set — a dark-market shutdown around time step 43 shifts the illicit distribution
  and every model degrades afterwards. Per-time-step PR-AUC makes this explicit.
- Features are **anonymised** (limited semantic explainability); setup is **transductive** full-batch
  (an inductive neighbour-sampling variant is needed for streaming scale).
- Single temporal split / single seed; ~77% of nodes are unlabelled. Production needs rolling-window
  retraining, drift monitoring, and multi-seed confidence intervals.

## Layout

```
temporal-graph-aml-intelligence/
├── temporal_graph_aml_intelligence.ipynb   # main deliverable (executed)
├── generate_notebook.py                     # builds the notebook
├── src/
│   ├── data_loader.py   # Elliptic loaders, graph assembly, temporal masks
│   ├── models.py        # GraphSAGE & GAT (PyTorch Geometric)
│   └── train.py         # training loop, metrics (PR-AUC/ROC-AUC/Recall@K), ranking utils
├── figures/             # generated plots + summary_metrics.csv
├── checkpoints/         # saved GNN weights
└── pyproject.toml
```
