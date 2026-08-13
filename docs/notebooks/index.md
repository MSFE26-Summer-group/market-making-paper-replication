# Notebooks

Exploratory notebooks live in the `notebooks/` folder at the repo root.
They are rendered here automatically via `mkdocs-jupyter`.

## Conventions

- Notebooks call into `src/paper_replication` — keep heavy logic there, not in cells.
- **Never commit cell outputs.** `nbstripout` runs as a pre-commit hook and strips them automatically.
- Name notebooks with a number prefix so they sort logically: `01_data_exploration.ipynb`, `02_baseline.ipynb`, etc.
- Add a Markdown cell at the top of each notebook with a title and one-line description.

## Index

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Initial look at the dataset |
| `02_feature_engineering.ipynb` | Builds and sanity-checks the Attn-LOB pretraining dataset — see [Feature Engineering Pipeline](../feature_engineering.md) |
| `04_attn_lob_model.ipynb` | Builds the Attn-LOB model and runs one forward pass on real data — architecture wiring check only, no training |
| `05_attn_lob_training.ipynb` | Trains the Attn-LOB pretrain head on real data, plots loss curves, evaluates on the held-out test split, and saves/reloads a checkpoint |
| `06_attn_lob_hyperparameter_search.ipynb` | Searches dropout/weight_decay/learning_rate against `val_loss` only, then evaluates the winner on `test` exactly once — see [Exp 1b](../experiments.md#exp-1b-attn-lob-pretrain-classification-tuned-btcusdt) |
| `07_baseline_comparison.ipynb` | Trains AttnLOB, FCLOB, ConvLOB, and DeepLOB on identical data with identical untuned defaults and compares all four against the paper's Table I — see [Exp 2](../experiments.md#exp-2-baseline-comparison-attnlob-vs-fclob-vs-convlob-vs-deeplob-btcusdt) |
| `08_lob_depth_ablation.ipynb` | Builds a deeper (`n_levels=50`) LOB snapshot from `rds_transformed.parquet`'s full order-book depth and compares Attn-LOB at `n_levels=10` vs. `n_levels=50` on the same single-day data/split/hyperparameters — see [Exp 3](../experiments.md#exp-3-attn-lob-lob-depth-ablation-btcusdt-2022-10-20-only) |
