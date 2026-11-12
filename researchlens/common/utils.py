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
    args = parser.parse_args()
    print("Fine-tuning settings:")
    for k, v in vars(args).items():
        print(f"  {k:<24} {v}")
    return args


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
        # checkpoints are big (~1.3 GB each for BERT-size models) -> local Colab disk by default, deleted after the run.
        # Set RL_CKPT_ON_DRIVE=1 to keep them on Drive (survives disconnects, needs space).
        "checkpoints": (root if os.environ.get("RL_CKPT_ON_DRIVE") == "1" else Path(os.environ.get("RL_CKPT_DIR", "/content/checkpoints")))
        / ("checkpoints" if os.environ.get("RL_CKPT_ON_DRIVE") == "1" else "") / model_name / run_name / f"seed_{seed}",
        # final model: written to the local Colab disk, uploaded to the Hugging Face Hub, then copied to Drive if
        # there is room (Drive is a backup, the Hub is the permanent store)
        "final_model": Path(os.environ.get("RL_MODEL_DIR", "/content/final_models")) / model_name / run_name / f"seed_{seed}",
        "drive_model": root / "models" / model_name / run_name / f"seed_{seed}",
    }
    for k, p in d.items():
        if k not in ("final_model", "drive_model"):   # created only when the model is actually kept
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


def make_training_args(n_train: int, warmup_ratio: float = 0.0, args_cls=None, **kwargs):
    """Build transformers.TrainingArguments that works on both transformers v4 and v5.

    - warmup is passed as an explicit number of steps (v5 removed `warmup_ratio`);
    - any argument the installed version no longer accepts is dropped with a printed note;
    - args_cls: e.g. Seq2SeqTrainingArguments (default TrainingArguments).
    """
    import inspect
    import math

    from transformers import TrainingArguments

    TrainingArguments = args_cls or TrainingArguments
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    bs = kwargs.get("per_device_train_batch_size", 8) * kwargs.get("gradient_accumulation_steps", 1)
    total = math.ceil(n_train / max(bs, 1)) * float(kwargs.get("num_train_epochs", 1))
    kwargs["warmup_steps"] = int(round(total * warmup_ratio))
    # older v4 releases call it evaluation_strategy
    if "eval_strategy" in kwargs and "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in accepted]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"Note: this transformers version does not accept {dropped}; ignored.")
    print(f"Warm-up: {kwargs['warmup_steps']} of ~{int(total)} training steps ({warmup_ratio:.0%})")
    return TrainingArguments(**kwargs)


# ------------------------------------------------------------------ anti-over-fitting helpers
def add_regularisation_args(p) -> None:
    """Same regularisation switches for every transformer we fine-tune (BERT/SciBERT/ViT)."""
    import argparse

    p.add_argument("--hidden_dropout", type=float, default=None, help="dropout inside the encoder (None = model default)")
    p.add_argument("--attention_dropout", type=float, default=None, help="dropout on attention weights")
    p.add_argument("--freeze_embeddings", action=argparse.BooleanOptionalAction, default=False,
                   help="keep the pre-trained embedding layer fixed")
    p.add_argument("--freeze_layers", type=int, default=0, help="keep the lowest N encoder layers fixed")


def dropout_overrides(model_name: str, hidden: float | None, attention: float | None) -> dict:
    """Config overrides for dropout, only for attributes the model actually has."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    out = {}
    if hidden is not None and hasattr(cfg, "hidden_dropout_prob"):
        out["hidden_dropout_prob"] = hidden
    if attention is not None and hasattr(cfg, "attention_probs_dropout_prob"):
        out["attention_probs_dropout_prob"] = attention
    return out


def apply_freezing(model, freeze_embeddings: bool, freeze_layers: int) -> None:
    base = model.base_model
    if freeze_embeddings and hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
    layers = getattr(getattr(base, "encoder", None), "layer", None)
    if freeze_layers and layers is not None:
        for layer in layers[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_tr:,} of {n_all:,} ({n_tr / n_all:.0%})")


def add_storage_args(p) -> None:
    import argparse

    p.add_argument("--keep_model", action=argparse.BooleanOptionalAction, default=None,
                   help="save the final model to Drive (default: only for seed 42 or when pushing to the Hub)")


def should_keep_model(args) -> bool:
    if args.keep_model is not None:
        return bool(args.keep_model)
    return args.seed == 42 or bool(getattr(args, "push_to_hub", False))


def cleanup_checkpoints(ckpt_dir) -> None:
    """Training finished and results are saved -> intermediate checkpoints are no longer needed."""
    import shutil

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"Removed intermediate checkpoints: {ckpt_dir}")


def verify_model_saved(model_dir) -> None:
    """Fail loudly if the weights file is missing (e.g. disk full) instead of silently losing the model."""
    model_dir = Path(model_dir)
    weights = [p for p in model_dir.glob("*") if p.suffix in (".safetensors", ".bin", ".pt") and p.name != "training_args.bin"]
    if not weights or max(p.stat().st_size for p in weights) < 1_000_000:
        raise RuntimeError(f"Model weights were NOT saved in {model_dir} (disk full?). Checkpoints kept for recovery.")
    print(f"Model weights saved: {', '.join(f'{p.name} ({p.stat().st_size/1e6:.0f} MB)' for p in weights)}")


def backup_to_drive(local_dir, drive_dir) -> None:
    """Copy the final model to Drive only if there is enough free space; never fail the run because of Drive."""
    import shutil

    local_dir, drive_dir = Path(local_dir), Path(drive_dir)
    size = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    try:
        free = shutil.disk_usage(drive_dir.parents[3] if len(drive_dir.parents) > 3 else drive_dir.parent).free
    except Exception:
        free = 0
    if free < size * 1.2:
        print(f"Drive backup skipped: needs {size/1e6:.0f} MB, {free/1e6:.0f} MB free. (The model is on the Hub.)")
        return
    try:
        shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
        print(f"Drive backup: {drive_dir}")
    except Exception as e:
        print(f"Drive backup failed ({e}); the model is on the Hub.")
