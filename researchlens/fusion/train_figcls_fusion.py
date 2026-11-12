"""V2 fusion - classify figure type from IMAGE + CAPTION together (gated late fusion).

    image  -> ViT     -> [CLS] -> Linear -> h_img ┐
                                                  ├─ gate g = softmax(W[h_img; h_txt])  (learned per figure)
    caption-> SciBERT -> [CLS] -> Linear -> h_txt ┘
    fused = g_img * h_img + g_txt * h_txt  -> classifier

Both encoders are fine-tuned end to end. "Modality dropout" randomly hides one modality during
training so the model cannot ignore the image and rely only on caption keywords (or vice versa).
The gate values are saved per test figure -> the report plots how much each class relies on vision vs text.

Example (Colab, after prepare_spiqa_figures):
    python -m researchlens.fusion.train_figcls_fusion --config configs/v2_fusion.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from researchlens.common.results_io import HistoryCallback, save_config, save_metrics, save_predictions
from researchlens.common.utils import (
    make_training_args,
    LOCAL_DATA, Timer, count_parameters, hardware_info, latest_checkpoint, parse_args_with_config, run_dirs, set_seed,
)
from researchlens.vision.figdata import CLASSES, build_transforms, class_weights, classification_metrics, read_manifest


def get_args():
    p = argparse.ArgumentParser(description="ViT + SciBERT gated fusion figure classifier")
    p.add_argument("--vision_model", default="google/vit-base-patch16-224-in21k",
                   help="HF id, or a Drive folder of the fine-tuned V2 ViT to start from it")
    p.add_argument("--text_model", default="allenai/scibert_scivocab_uncased",
                   help="HF id, or a Drive folder of the fine-tuned caption SciBERT")
    p.add_argument("--model_tag", default="v2_figcls")
    p.add_argument("--run_name", default="fusion_vit_scibert")
    p.add_argument("--data_dir", default=str(LOCAL_DATA / "spiqa_figs"))
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--modality_dropout", type=float, default=0.15)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", default="researchlens-fusion-figure-type")
    return parse_args_with_config(p)


def build_model(args, weights):
    import torch
    from torch import nn
    from transformers import AutoModel

    class GatedFusionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = AutoModel.from_pretrained(args.vision_model)
            self.text = AutoModel.from_pretrained(args.text_model)
            self.proj_img = nn.Linear(self.vision.config.hidden_size, args.hidden)
            self.proj_txt = nn.Linear(self.text.config.hidden_size, args.hidden)
            self.gate = nn.Linear(2 * args.hidden, 2)
            self.head = nn.Sequential(nn.LayerNorm(args.hidden), nn.Dropout(args.dropout), nn.Linear(args.hidden, len(CLASSES)))
            self.register_buffer("class_w", weights.clone())

        def features(self, pixel_values, input_ids, attention_mask, token_type_ids=None):
            h_img = torch.tanh(self.proj_img(self.vision(pixel_values=pixel_values).last_hidden_state[:, 0]))
            t = self.text(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            h_txt = torch.tanh(self.proj_txt(t.last_hidden_state[:, 0]))
            if self.training and args.modality_dropout > 0:
                b = h_img.size(0)
                r = torch.rand(b, device=h_img.device)
                drop_img = (r < args.modality_dropout / 2).float().unsqueeze(1)
                drop_txt = ((r >= args.modality_dropout / 2) & (r < args.modality_dropout)).float().unsqueeze(1)
                h_img, h_txt = h_img * (1 - drop_img), h_txt * (1 - drop_txt)
            g = torch.softmax(self.gate(torch.cat([h_img, h_txt], -1)), -1)
            fused = g[:, :1] * h_img + g[:, 1:] * h_txt
            return fused, g

        def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None, labels=None):
            fused, g = self.features(pixel_values, input_ids, attention_mask, token_type_ids)
            logits = self.head(fused)
            out = {"logits": logits, "gates": g}
            if labels is not None:
                out["loss"] = nn.functional.cross_entropy(
                    logits.float(), labels, weight=self.class_w, label_smoothing=args.label_smoothing)
            return out

    return GatedFusionClassifier()


def main():
    args = get_args()
    import torch
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import Dataset
    from transformers import AutoImageProcessor, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")

    splits = read_manifest(Path(args.data_dir))
    processor = AutoImageProcessor.from_pretrained(args.vision_model)
    tok = AutoTokenizer.from_pretrained(args.text_model)

    class FigCapDS(Dataset):
        def __init__(self, items, train):
            self.items, self.tf = items, build_transforms(processor, train)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            enc = tok(it["caption"], truncation=True, max_length=args.max_length)
            return {"pixel_values": self.tf(Image.open(it["path"])), "labels": it["label"], **enc}

    def collate(batch):
        text = tok.pad([{k: b[k] for k in b if k in ("input_ids", "attention_mask", "token_type_ids")} for b in batch],
                       return_tensors="pt")
        return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch]), **text}

    ds = {k: FigCapDS(v, train=(k == "train")) for k, v in splits.items()}
    model = build_model(args, class_weights(splits["train"]))

    def compute_metrics(ep):
        preds = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        y = ep.label_ids
        p = preds.argmax(-1)
        return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}

    targs = make_training_args(
        len(ds["train"]), args.warmup_ratio,
        output_dir=str(dirs["checkpoints"]), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2, weight_decay=args.weight_decay, lr_scheduler_type="cosine",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, logging_steps=20,
        fp16=args.fp16 and torch.cuda.is_available(), dataloader_num_workers=args.num_workers,
        remove_unused_columns=False, label_names=["labels"],
        report_to=["wandb"] if args.wandb else "none", run_name=f"{args.model_tag}-{args.run_name}-s{args.seed}",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"], data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=[HistoryCallback(dirs["results"] / "history.csv"),
                   EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    resume = latest_checkpoint(dirs["checkpoints"])
    with Timer() as t:
        trainer.train(resume_from_checkpoint=resume)

    # ---- test: predictions, gates, fused embeddings ----
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_gates, all_emb = [], [], []
    with Timer() as ti, torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for batch in trainer.get_test_dataloader(ds["test"]):
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            fused, g = model.features(**batch)
            all_logits.append(model.head(fused).float().cpu())
            all_gates.append(g.float().cpu())
            all_emb.append(fused.float().cpu())
    probs = torch.softmax(torch.cat(all_logits), -1).numpy()
    gates = torch.cat(all_gates).numpy()
    y = np.array([it["label"] for it in splits["test"]])

    metrics = {
        "model": f"{args.vision_model} + {args.text_model}", "run_name": args.run_name, "seed": args.seed,
        "modality": "image + caption (gated fusion)",
        **classification_metrics(y, probs),
        "mean_gate_image": float(gates[:, 0].mean()), "mean_gate_text": float(gates[:, 1].mean()),
        "best_val_macro_f1": trainer.state.best_metric, "train_seconds": round(t.seconds, 1),
        "test_samples_per_second": round(len(y) / ti.seconds, 2),
        **count_parameters(model), **hardware_info(),
    }
    save_metrics(metrics, dirs["results"] / "metrics.json")
    save_predictions(
        ({"path": it["rel_path"], "caption": it["caption"], "gold": CLASSES[it["label"]],
          "pred": CLASSES[int(pr.argmax())], "probs": [round(float(x), 5) for x in pr],
          "gate_image": round(float(g[0]), 4), "gate_text": round(float(g[1]), 4)}
         for it, pr, g in zip(splits["test"], probs, gates)),
        dirs["results"] / "predictions.jsonl",
    )
    np.save(dirs["results"] / "embeddings.npy", torch.cat(all_emb).numpy())

    final = dirs["final_model"]
    torch.save(model.state_dict(), final / "fusion_model.pt")
    (final / "fusion_config.json").write_text(json.dumps({k: v for k, v in vars(args).items()
                                                          if k not in ("push_to_hub", "wandb")}, indent=2))
    tok.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"Fusion: accuracy = {metrics['test_accuracy']:.4f} | macro-F1 = {metrics['test_f1_macro']:.4f} | "
          f"mean gate image/text = {metrics['mean_gate_image']:.2f}/{metrics['mean_gate_text']:.2f}")

    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(final, dirs["results"], args.hub_repo,
                   title="ViT + SciBERT gated-fusion figure-type classifier (SPIQA)",
                   base_model=args.vision_model, dataset="SPIQA figures + captions", task="image-classification")


if __name__ == "__main__":
    main()
