# Paper Replication

Welcome to the documentation for our replication of **Market Making with Deep Reinforcement Learning from Limit Order Books by Guo, Lin, and Huang, 2023** ([link to paper](papers/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books.pdf)).

## Goals

- Faithfully reproduce the key experiments and results from the original paper.
- Maintain a clean, well-tested codebase so that overseers and collaborators can follow along.
- Document deviations or improvements clearly.

## Quick Start

```bash
git clone https://github.com/MSFE26-Summer-group/market-making-paper-replication.git
cd market-making-paper-replication
pip install uv        # one-time, installs uv globally
uv sync --extra dev --extra docs
uv run pre-commit install
```

## Team

| Name | Role |
|------|------|
| Alberto Munoz | Student |
| ChiaChun Hsu  | Student |
| Devminda Abeynayake | Student |
