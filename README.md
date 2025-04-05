# ML-Template

## Structure

```shell
├── README.md
├── app.log
├── data
│   ├── input
│   ├── output
│   └── raw
├── justfile
├── logging.conf
├── logs
├── models
├── notebooks
├── pyproject.toml
├── scripts
├── src
│   ├── __init__.py
│   ├── config.py
│   └── tests
│       └── base.txt
├── uv.lock
└── visualization
```

## Installation

1. Install `uv` and `just`

https://docs.astral.sh/uv/getting-started/installation/#installation-methods
https://github.com/casey/just

2. Install dependencies

```shell
# uv add ruff tqdm torch lightning jupyter tensorboard torchinfo torchvision dotenv
just start
```

## Useful Snippets

```jupyter
%load_ext autoreload
%autoreload 2
```

```shell
uv run filename.py
```

```python
np.random.seed(42)
torch.manual_seed(42)
```

