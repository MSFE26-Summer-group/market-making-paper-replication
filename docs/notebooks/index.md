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
