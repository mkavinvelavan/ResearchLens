"""Cross-domain test of a figure-type classifier: SPIQA test (computer science) vs biomedical test (BIOMEDICA, PubMed).

    python -m researchlens.vision.eval_figcls_crossdomain --model Kavinvelavan/researchlens-vit-figure-type \
        --data_dir /content/data/figs_multidomain --run_name vit_spiqa_only
Writes results/v2_figcls_crossdomain/<run_name>/seed_42/metrics.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from researchlens.common.results_io import save_metrics
from researchlens.common.utils import run_dirs
from researchlens.vision.figdata import CLASSES, build_transforms, classification_metrics, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="/content/data/figs_multidomain")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from researchlens.common.hub import hf_login_from_colab
    hf_login_from_colab()
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    tf = build_transforms(proc, train=False)
    test = read_manifest(Path(args.data_dir))["test"]

    metrics = {"model": args.model, "run_name": args.run_name}
    for d in sorted({it["domain"] for it in test}):
        items = [it for it in test if it["domain"] == d]
        probs = []
        for i in range(0, len(items), args.batch_size):
            x = torch.stack([tf(Image.open(it["path"])) for it in items[i:i + args.batch_size]]).to(dev)
            with torch.no_grad():
                probs.append(torch.softmax(model(pixel_values=x).logits.float(), -1).cpu().numpy())
        probs = np.concatenate(probs)
        y = [it["label"] for it in items]
        m = classification_metrics(y, probs, prefix="test")
        per_class = m.pop("per_class")
        m.pop("classes", None)
        metrics.update({k.replace("test_", f"test_{d}_", 1): v for k, v in m.items()})
        metrics[f"test_{d}_per_class"] = per_class
        metrics[f"test_{d}_n"] = len(items)
        print(f"[{d}] n={len(items)}  accuracy {metrics[f'test_{d}_accuracy']:.4f}  macro-F1 {metrics[f'test_{d}_f1_macro']:.4f}"
              "  per class F1: " + ", ".join(f"{c} {per_class[c]['f1-score']:.2f} (n={per_class[c]['support']})"
                                            for c in CLASSES))
    if "test_spiqa_f1_macro" in metrics and "test_biomedical_f1_macro" in metrics:
        metrics["domain_drop_f1_macro"] = metrics["test_spiqa_f1_macro"] - metrics["test_biomedical_f1_macro"]
        metrics["domain_drop_accuracy"] = metrics["test_spiqa_accuracy"] - metrics["test_biomedical_accuracy"]
        print(f"Drop from CS figures to biomedical figures: macro-F1 {metrics['domain_drop_f1_macro'] * 100:.1f} pts, "
              f"accuracy {metrics['domain_drop_accuracy'] * 100:.1f} pts")
    out = run_dirs("v2_figcls_crossdomain", args.run_name, 42)["results"] / "metrics.json"
    save_metrics(metrics, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
