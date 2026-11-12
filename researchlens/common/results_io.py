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
        if self.out_path.exists():  # resumed run: keep earlier history
            with open(self.out_path, newline="", encoding="utf-8") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]

    def on_log(self, args, state, control, logs=None, **kwargs):
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
