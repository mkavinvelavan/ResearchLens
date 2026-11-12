"""Shared helpers: config loading, seeding, run directories, hardware stats."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

# Default Google Drive locations (Colab mounts Drive at /content/drive).
DRIVE_ROOT = Path(os.environ.get("RL_DRIVE_ROOT", "/content/drive/MyDrive/ResearchLens"))
LOCAL_DATA = Path(os.environ.get("RL_DATA_DIR", "/content/data"))


def load_config(path: str | None) -> dict:
    """Load a YAML config file into a dict (empty dict if no path)."""
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Values in --config YAML become defaults; explicit CLI flags still win."""
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    pre, _ = parser.parse_known_args()
    cfg = load_config(pre.config)
    known = {a.dest for a in parser._actions}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"Unknown keys in config {pre.config}: {sorted(unknown)}")
    parser.set_defaults(**cfg)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_dirs(model_name: str, run_name: str, seed: int, drive_root: Path | str = DRIVE_ROOT) -> dict[str, Path]:
    """Checkpoint + results directories on Drive for one (model, run, seed).

    results/<model_name>/<run_name>/seed_<seed>/  -> metrics.json, history.csv, predictions.jsonl
    checkpoints/<model_name>/<run_name>/seed_<seed>/  -> HF Trainer checkpoints (resumable)
    """
    root = Path(drive_root)
    d = {
        "results": root / "results" / model_name / run_name / f"seed_{seed}",
        "checkpoints": root / "checkpoints" / model_name / run_name / f"seed_{seed}",
        "final_model": root / "models" / model_name / run_name / f"seed_{seed}",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        else:
            info["gpu"] = "cpu"
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def latest_checkpoint(ckpt_dir: Path) -> str | None:
    """Return the newest checkpoint-* folder so an interrupted Colab run can resume."""
    ckpts = sorted(Path(ckpt_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")
