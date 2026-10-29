"""Shared helpers: configuration loading, seeding, I/O and logging."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER_NAME = "svdnn_maig"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike = "config.yaml", mode: str = "full") -> dict:
    """Load config.yaml and apply the overrides of the requested run mode."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    modes = cfg.pop("modes", {})
    if mode not in modes:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(modes)}")
    cfg = _deep_update(copy.deepcopy(cfg), modes[mode] or {})
    cfg["mode"] = mode
    cores = os.cpu_count() or 1
    if not cfg.get("n_workers"):
        cfg["n_workers"] = max(1, cores // 2)
    if not cfg.get("n_threads"):
        cfg["n_threads"] = max(1, cores // cfg["n_workers"])
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / mode)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def derive_seed(master: int, *keys: Any) -> int:
    """Deterministic sub-seed from a master seed and arbitrary keys."""
    text = f"{master}|" + "|".join(str(k) for k in keys)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=NumpyEncoder)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "pandas", "scipy", "sklearn", "torch", "xgboost", "optuna", "matplotlib"]:
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[name] = "not installed"
    return info
