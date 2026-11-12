"""Standard result files every model writes, so one results notebook can plot everything.

results/<model>/<run>/seed_<n>/
    metrics.json       final test metrics + run info (params, time, GPU)
    history.csv        one row per log/eval event: step, epoch, loss, eval_*, learning_rate
    predictions.jsonl  one JSON object per test example (gold + prediction + extras)
    config.json        all arguments used for the run
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from transformers import TrainerCallback

from .utils import write_json


class HistoryCallback(TrainerCallback):
    """Collect every Trainer log (train loss, LR, eval metrics) and save history.csv.

    The CSV is re-written after every log event, so the curves survive a Colab disconnect.
    """

    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.rows: list[dict[str, Any]] = []
        self.max_epochs, self.best_epoch, self.best_value, self.last_train_loss = 0, None, None, None
        if self.out_path.exists():  # resumed run: keep earlier history
            with open(self.out_path, newline="", encoding="utf-8") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]

    # ---- readable progress in the notebook ----
    def on_train_begin(self, args, state, control, **kwargs):
        self.max_epochs = args.num_train_epochs
        self.best_epoch, self.best_value, self.last_train_loss = None, None, None
        print(f"\n=== Training: up to {int(self.max_epochs)} epochs, {state.max_steps} steps "
              f"(early stopping may end it sooner) ===", flush=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n--- Epoch {int(round(state.epoch or 0)) + 1}/{int(self.max_epochs)} started ---", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
        key = "eval_" + (args.metric_for_best_model or "loss").removeprefix("eval_")
        if key not in metrics:          # e.g. the training-set evaluation: keep it, print with the val line
            self._pending = {**getattr(self, "_pending", {}), **metrics}
            return
        metrics = {**getattr(self, "_pending", {}), **metrics}
        self._pending = {}
        val = metrics.get(key)
        better = val is not None and (self.best_value is None or
                                      (val > self.best_value if args.greater_is_better else val < self.best_value))
        if better:
            self.best_value, self.best_epoch = val, round(state.epoch or 0)
        parts = [f"Epoch {round(state.epoch or 0)}/{int(self.max_epochs)} done"]
        if self.last_train_loss is not None:
            parts.append(f"train loss {self.last_train_loss:.4f}")
        if "eval_loss" in metrics:
            parts.append(f"val loss {metrics['eval_loss']:.4f}")
        show = ("loss", "f1", "macro_f1", "accuracy")
        for split in ("train", "val"):
            for m in show:
                v = metrics.get(f"eval_{split}_{m}")
                if isinstance(v, (int, float)):
                    parts.append(f"{split} {m} {v:.4f}")
        if not any(k.startswith(("eval_train_", "eval_val_")) for k in metrics):
            for k, v in metrics.items():
                if k.startswith("eval_") and k[5:] in show[1:] and isinstance(v, float):
                    parts.append(f"val {k[5:]} {v:.4f}")
        parts.append("<- new best" if better else f"(best so far: epoch {self.best_epoch})")
        print(" | ".join(parts), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        ran = round(state.epoch or 0, 2)
        stopped = " (stopped early: validation score stopped improving)" if ran < self.max_epochs - 0.01 else ""
        print(f"\n=== Finished: trained {ran} of {int(self.max_epochs)} epochs{stopped}. "
              f"Best epoch = {self.best_epoch} (val score {self.best_value}) - this checkpoint is used for testing ===",
              flush=True)

    def summary(self, state) -> dict:
        return {"epochs_trained": round(state.epoch or 0, 2), "max_epochs": self.max_epochs,
                "best_epoch": self.best_epoch, "total_steps": state.global_step}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]
        if not logs or not state.is_world_process_zero:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4)}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        # drop rows from a crashed run that are ahead of the resumed step
        self.rows = [r for r in self.rows if int(float(r.get("step", 0))) <= state.global_step]
        self.rows.append(row)
        self._write()

    def _write(self):
        keys: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def save_predictions(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_metrics(metrics: dict, path: Path) -> None:
    write_json(metrics, path)


def save_config(args, path: Path) -> None:
    write_json(vars(args) if not isinstance(args, dict) else args, path)


def print_split_table(split_scores: dict, keys: list[str]) -> None:
    """Train / validation / test side by side - the over-fitting check at a glance."""
    print("\n" + f"{'metric':<24}" + "".join(f"{s:>10}" for s in split_scores))
    for k in keys:
        print(f"{k:<24}" + "".join(f"{split_scores[s].get(k, float('nan'))*100:>9.2f}%" for s in split_scores))
    if "train" in split_scores and "test" in split_scores:
        main = keys[0]
        gap = (split_scores["train"][main] - split_scores["test"][main]) * 100
        print(f"train-test gap in {main}: {gap:.2f} points")
