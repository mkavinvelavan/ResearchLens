"""Push a trained model + its results to the Hugging Face Hub with a model card."""
from __future__ import annotations

import json
import os
from pathlib import Path


def hf_login_from_colab() -> None:
    """Log in using the HF_TOKEN stored in Colab's Secrets panel (never hard-code tokens)."""
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        print("WARNING: HF_TOKEN not found (Colab -> Secrets, notebook access on). Skipping Hub upload.")
        return
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        print(f"WARNING: Hugging Face login failed: {e}")


def _card(repo_id: str, title: str, base_model: str, dataset: str, task: str, metrics: dict, extra: str = "") -> str:
    rows = "\n".join(
        f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
        for k, v in metrics.items()
        if isinstance(v, (int, float, str)) and not k.startswith("_")
    )
    return f"""---
library_name: transformers
base_model: {base_model}
tags:
- researchlens
- {task}
---

# {title}

Part of **ResearchLens**, a multimodal research-paper assistant (MSc Applied AI project).

- **Base model:** `{base_model}`
- **Training data:** {dataset}
- **Task:** {task}

## Test results

| Metric | Value |
|---|---|
{rows}

{extra}

## Usage

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("{repo_id}")
```
"""


def push_model(*args, **kwargs):
    """Upload, but never crash a finished training run: results are already saved on Drive."""
    try:
        return _push_model(*args, **kwargs)
    except Exception as e:
        print(f"WARNING: Hugging Face upload failed ({type(e).__name__}: {e}). "
              "Results are saved on Drive; you can push later with researchlens.common.hub.push_model.")
        return None


def _push_model(
    model_dir: Path,
    results_dir: Path,
    repo_name: str,
    title: str,
    base_model: str,
    dataset: str,
    task: str,
    private: bool = True,
) -> str:
    """Upload saved model files + metrics/history to <HF_USERNAME>/<repo_name>."""
    from huggingface_hub import HfApi

    user = os.environ.get("HF_USERNAME")
    api = HfApi()
    if not user:
        user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"
    api.create_repo(repo_id, private=private, exist_ok=True)

    metrics_path = Path(results_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    (Path(model_dir) / "README.md").write_text(_card(repo_id, title, base_model, dataset, task, flat), encoding="utf-8")

    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, commit_message="Upload model")
    for name in ("metrics.json", "history.csv", "config.json"):
        p = Path(results_dir) / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"results/{name}", repo_id=repo_id)
    print(f"Pushed to https://huggingface.co/{repo_id}")
    return repo_id
