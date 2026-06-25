# [Paper Title] — Replication

[![CI](https://github.com/<your-org>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/<your-repo>/actions/workflows/ci.yml)
[![Docs](https://github.com/<your-org>/<your-repo>/actions/workflows/docs.yml/badge.svg)](https://<your-org>.github.io/<your-repo>/)

Replication of **[Paper Title]** by [Authors], [Venue Year].

📖 **[Full documentation →](https://<your-org>.github.io/<your-repo>/)**

---

## Setup

```bash
git clone https://github.com/MSFE26-Summer-group/market-making-paper-replication.git
cd market-making-paper-replication
pip install uv        # one-time, installs uv globally
uv sync --extra dev --extra docs
uv run pre-commit install
```

## Running Tests

```bash
pytest
```

## Code Standards

All PRs to `main` must pass:

| Tool | Purpose | Hard block? |
|------|---------|-------------|
| `black` | Formatting | ✅ Yes |
| `isort` | Import order | ✅ Yes |
| `mypy` | Type checking | ✅ Yes |
| `pytest` | Tests + 70% coverage | ✅ Yes |
| `ruff` | Linting | ⚠️ Reported only |

Run all checks locally before pushing:

```bash
black src/ tests/
isort src/ tests/
mypy src/
pytest
```

Or install pre-commit hooks (done once via `pre-commit install`) to have them run automatically on every commit — this includes `nbstripout` (strips notebook outputs before committing) and `nbqa` (runs black/isort/mypy inside notebooks).

## Branch Strategy

- `main` — protected; requires a passing CI and at least 1 approved review to merge.
- `feature/<name>` — all work happens here; open a PR against `main`.

## Docs

Documentation lives in `docs/` and is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
It auto-deploys to GitHub Pages on every merge to `main`.

```bash
# Preview locally
mkdocs serve
```

## Repository Layout

```
.
├── src/paper_replication/   # Source code (imported by notebooks & tests)
├── notebooks/               # Jupyter notebooks — exploration & analysis
├── tests/                   # pytest tests
├── docs/                    # MkDocs pages
│   ├── index.md
│   ├── paper_overview.md
│   ├── experiments.md
│   ├── progress.md          # Overseer-facing progress tracker
│   └── api/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml           # Quality gate on every PR
│   │   └── docs.yml         # Deploy docs on merge to main
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── pyproject.toml           # All tool config lives here
├── mkdocs.yml
└── .pre-commit-config.yaml
```
