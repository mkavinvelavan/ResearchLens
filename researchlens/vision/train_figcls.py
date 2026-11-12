"""V2 - Scientific figure-type classification: fine-tune a ViT on SPIQA figures.

Classes: plot, schematic, photo, table, other  (see prepare_spiqa_figures.py)

Example (Colab):
    python -m researchlens.vision.train_figcls --config configs/v2_figcls.yaml --seed 42
    # baselines for the report
    python -m researchlens.vision.train_figcls --config configs/v2_figcls.yaml --freeze_backbone --run_name vit_linear_probe
    python -m researchlens.vision.train_figcls --config configs/v2_figcls.yaml --model_name microsoft/resnet-50 --run_name resnet50

Writes results/<model_tag>/<run_name>/seed_<seed>/{metrics.json, history.csv, predictions.jsonl,
config.json, embeddings.npy}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from researchlens.common.results_io import HistoryCallback, print_split_table, save_config, save_metrics, save_predictions
from researchlens.common.utils import (
    backup_to_drive, verify_model_saved,
    add_storage_args, cleanup_checkpoints, should_keep_model,
    add_regularisation_args, apply_freezing, dropout_overrides,
    make_training_args,
    LOCAL_DATA,
    Timer,
    count_parameters,
    hardware_info,
    latest_checkpoint,
    parse_args_with_config,
    run_dirs,
    set_seed,
)
from researchlens.vision.figdata import epoch_metrics, split_view, train_eval_subset, CLASSES, build_transforms, class_weights, classification_metrics, read_manifest


def get_args():
    p = argparse.ArgumentParser(description="ViT figure-type classifier")
    p.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    p.add_argument("--model_tag", default="v2_figcls")
    p.add_argument("--run_name", default="vit_base")
    p.add_argument("--data_dir", default=str(LOCAL_DATA / "spiqa_figs"))
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    add_regularisation_args(p)
    p.add_argument("--class_weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_backbone", action="store_true", help="linear-probe baseline")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    add_storage_args(p)
    p.add_argument("--hub_repo", default="researchlens-vit-figure-type")
    return parse_args_with_config(p)


def main():
    args = get_args()
    import torch
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import Dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")

    data_dir = Path(args.data_dir)
    splits = read_manifest(data_dir)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    class FigureDS(Dataset):
        def __init__(self, items, train):
            self.items, self.tf = items, build_transforms(processor, train)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            return {"pixel_values": self.tf(Image.open(it["path"])), "labels": it["label"]}

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
        }

    ds = {k: FigureDS(v, train=(k == "train")) for k, v in splits.items()}

    id2label = dict(enumerate(CLASSES))
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name, num_labels=len(CLASSES), id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True,
        **dropout_overrides(args.model_name, args.hidden_dropout, args.attention_dropout),
    )
    apply_freezing(model, args.freeze_embeddings, args.freeze_layers)
    if args.freeze_backbone:
        for n, p_ in model.named_parameters():
            p_.requires_grad = n.startswith("classifier")

    weights = class_weights(splits["train"])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=weights.to(outputs.logits.device) if args.class_weights else None,
                label_smoothing=args.label_smoothing,
            )
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    # fixed 2,000-example slice of the training set, scored every epoch (no augmentation) -> over-fitting curves
    train_eval_items = train_eval_subset(splits["train"], 2000, args.seed)
    train_eval_ds = FigureDS(train_eval_items, train=False)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = logits.argmax(-1)
        return epoch_metrics(labels, pred)

    targs = make_training_args(
        len(ds["train"]), args.warmup_ratio,
        output_dir=str(dirs["checkpoints"]),
        num_train_epochs=args.epochs,
        learning_rate=args.lr if not args.freeze_backbone else args.lr * 20,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="val_macro_f1",
        greater_is_better=True,
        logging_steps=20,
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=["wandb"] if args.wandb else "none",
        run_name=f"{args.model_tag}-{args.run_name}-s{args.seed}",
        seed=args.seed,
    )
    hist_cb = HistoryCallback(dirs["results"] / "history.csv", resume=latest_checkpoint(dirs["checkpoints"]) is not None)
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset={"train": train_eval_ds, "val": ds["validation"]},
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=[
            hist_cb,
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    resume = latest_checkpoint(dirs["checkpoints"])
    if resume:
        print(f"Resuming from {resume}")
    with Timer() as t:
        trainer.train(resume_from_checkpoint=resume)

    # ---- final test evaluation on the best checkpoint ----
    out = trainer.predict(ds["test"], metric_key_prefix="test")
    logits = out.predictions
    probs = torch.softmax(torch.tensor(out.predictions, dtype=torch.float), -1).numpy()
    metrics = {
        "model": args.model_name,
        "run_name": args.run_name,
        "seed": args.seed,
        "modality": "image",
        "freeze_backbone": args.freeze_backbone,
        **classification_metrics(out.label_ids, probs),
        **hist_cb.summary(trainer.state),
        "best_val_macro_f1": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_seconds": round(t.seconds, 1),
        "test_samples_per_second": out.metrics.get("test_samples_per_second"),
        **count_parameters(model),
        **hardware_info(),
    }
    # ---- best checkpoint on the training slice and validation set too (train / val / test gap) ----
    for split, d in (("train", train_eval_ds), ("val", ds["validation"])):
        o = trainer.predict(d, metric_key_prefix=split)
        lg = o.predictions[0] if isinstance(o.predictions, tuple) else o.predictions
        pr_ = torch.softmax(torch.tensor(lg, dtype=torch.float), -1).numpy()
        metrics.update(classification_metrics(o.label_ids, pr_, prefix=split))
    metrics["gap_train_test_f1_macro"] = metrics["train_f1_macro"] - metrics["test_f1_macro"]
    metrics["gap_train_test_accuracy"] = metrics["train_accuracy"] - metrics["test_accuracy"]
    print_split_table(split_view(metrics), ["f1_macro", "accuracy", "balanced_accuracy", "precision_macro",
                                            "recall_macro", "top2_accuracy", "cohen_kappa"])
    save_metrics(metrics, dirs["results"] / "metrics.json")
    save_predictions(
        (
            {"path": it["rel_path"], "caption": it["caption"], "gold": CLASSES[it["label"]],
             "pred": CLASSES[int(pr.argmax())], "probs": [round(float(x), 5) for x in pr]}
            for it, pr in zip(splits["test"], probs)
        ),
        dirs["results"] / "predictions.jsonl",
    )

    # ---- image embeddings of the test set (for the t-SNE / UMAP plot) ----
    model.eval()
    device = next(model.parameters()).device
    embs = []
    loader = trainer.get_test_dataloader(ds["test"])
    with torch.no_grad():
        for batch in loader:
            o = model.base_model(pixel_values=batch["pixel_values"].to(device))
            pooled = getattr(o, "pooler_output", None)
            feat = pooled.flatten(1) if pooled is not None else o.last_hidden_state[:, 0]
            embs.append(feat.float().cpu().numpy())
    np.save(dirs["results"] / "embeddings.npy", np.concatenate(embs))

    if should_keep_model(args):
        dirs["final_model"].mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(dirs["final_model"]))
        processor.save_pretrained(str(dirs["final_model"]))
    if should_keep_model(args):
        verify_model_saved(dirs["final_model"])
    cleanup_checkpoints(dirs["checkpoints"])
    print(f"Test accuracy = {metrics['test_accuracy']:.4f} | macro-F1 = {metrics['test_f1_macro']:.4f}")
    print(f"Results: {dirs['results']}")

    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(
            dirs["final_model"], dirs["results"], args.hub_repo,
            title="ViT for Scientific Figure-Type Classification (SPIQA)", base_model=args.model_name,
            dataset="SPIQA figures (Pramanick et al., 2024)", task="image-classification",
        )

    if should_keep_model(args):
        backup_to_drive(dirs["final_model"], dirs["drive_model"])

if __name__ == "__main__":
    main()
