"""Save/load checkpoints for any of this replication's pretrain models.

The checkpoint payload is deliberately plain (a dict of primitives + tensors,
no pickled custom classes) so it loads under torch's `weights_only=True`
default -- the safe mode that refuses to unpickle arbitrary objects. The
model's config dataclass is stored as a plain dict via `dataclasses.asdict`
and reconstructed on load, rather than pickling the dataclass instance
itself. Which model/config classes to reconstruct is looked up from
`_MODEL_REGISTRY` by class name, saved alongside the config.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn

from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig
from paper_replication.models.conv_lob import ConvLOB, ConvLOBConfig
from paper_replication.models.deeplob import DeepLOB, DeepLOBConfig
from paper_replication.models.fc_lob import FCLOB, FCLOBConfig

_MODEL_REGISTRY: dict[str, tuple[type[nn.Module], type[Any]]] = {
    "AttnLOB": (AttnLOB, AttnLOBConfig),
    "FCLOB": (FCLOB, FCLOBConfig),
    "ConvLOB": (ConvLOB, ConvLOBConfig),
    "DeepLOB": (DeepLOB, DeepLOBConfig),
}


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Saves `model`. Its class must be one of `_MODEL_REGISTRY`'s keys."""
    class_name = type(model).__name__
    if class_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Don't know how to checkpoint a {class_name!r} -- add it to "
            f"checkpoint._MODEL_REGISTRY first."
        )
    # model.config isn't part of nn.Module's own interface -- mypy resolves
    # it via nn.Module.__getattr__'s stub (typed Tensor | Module, its real
    # purpose being submodule/parameter lookup), hence the explicit Any here.
    config: Any = model.config
    payload = {
        "model_class": class_name,
        "config": dataclasses.asdict(config),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> nn.Module:
    """Loads a checkpoint saved by `save_checkpoint` onto `map_location`.

    `map_location` only controls where `torch.load` places the *state dict's*
    tensors while unpickling -- `load_state_dict` then copies those values
    into the freshly-constructed model's own parameters in place, preserving
    whatever device that fresh model started on (CPU), not the source
    tensors' device. The explicit `.to(map_location)` below is what actually
    puts the returned model on the requested device.
    """
    payload: dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=True
    )
    model_cls, config_cls = _MODEL_REGISTRY[payload["model_class"]]
    config = config_cls(**payload["config"])
    model = model_cls(config)
    model.load_state_dict(payload["state_dict"])
    model = model.to(map_location)
    return model
