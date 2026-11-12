"""V2 text branch - classify the figure type from its CAPTION only, with SciBERT.

Same SPIQA figures, labels and test set as the ViT (image-only) classifier, so the report can compare
text-only vs image-only vs image+caption fusion on identical examples.

Example (Colab, after prepare_spiqa_figures):
    python -m researchlens.nlp.train_caption_cls --config configs/v2_caption.yaml --seed 42

Writes results/v2_figcls/scibert_caption/seed_<n>/{metrics.json, history.csv, predictions.jsonl, embeddings.npy}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from researchlens.common.results_io import HistoryCallback, print_split_table, save_config, save_metrics, save_predictions
from researchlens.common.utils import (
    add_storage_args, cleanup_checkpoints, should_keep_model,
    add_regularisation_args, apply_freezing, dropout_overrides,
    make_training_args,
    LOCAL_DATA, Timer, count_parameters, hardware_info, latest_checkpoint, parse_args_with_config, run_dirs, set_seed,
)
from researchlens.vision.figdata import epoch_metrics, split_view, train_eval_subset, CLASSES, classification_metrics, read_manifest


def get_args():
    p = argparse.ArgumentParser(description="SciBERT caption -> figure type")
    p.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--model_tag", default="v2_figcls")
    p.add_argument("--run_name", default="scibert_caption")
    p.add_argument("--data_dir", default=str(LOCAL_DATA / "spiqa_figs"))
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    add_regularisation_args(p)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    add_storage_args(p)
    p.add_argument("--hub_repo", default="researchlens-scibert-caption-figure-type")
    return parse_args_with_config(p)


def main():
    args = get_args()
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")

    data_dir = Path(args.data_dir)
    splits = read_manifest(data_dir)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    def to_ds(items):
        ds = Dataset.from_list([{"text": it["caption"] or "", "labels": it["label"]} for it in items])
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_length), batched=True,
                      remove_columns=["text"])

    ds = {k: to_ds(v) for k, v in splits.items()}
    id2label = dict(enumerate(CLASSES))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(CLASSES), id2label=id2label, label2id={v: k for k, v in id2label.items()},
        **dropout_overrides(args.model_name, args.hidden_dropout, args.attention_dropout))
    apply_freezing(model, args.freeze_embeddings, args.freeze_layers)

    # fixed 2,000-example slice of the training set, scored every epoch (no augmentation) -> over-fitting curves
    train_eval_items = train_eval_subset(splits["train"], 2000, args.seed)
    train_eval_ds = to_ds(train_eval_items)

    def compute_metrics(ep):
        logits, labels = ep
        pred = logits.argmax(-1)
        return epoch_metrics(labels, pred)

    targs = make_training_args(
        len(ds["train"]), args.warmup_ratio,
        output_dir=str(dirs["checkpoints"]), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2, weight_decay=args.weight_decay, lr_scheduler_type="linear", label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="val_macro_f1", greater_is_better=True, logging_steps=20,
        fp16=args.fp16 and torch.cuda.is_available(), report_to=["wandb"] if args.wandb else "none",
        run_name=f"{args.model_tag}-{args.run_name}-s{args.seed}", seed=args.seed,
    )
    hist_cb = HistoryCallback(dirs["results"] / "history.csv")
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset={"train": train_eval_ds, "val": ds["validation"]},
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        callbacks=[hist_cb,
                   EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    resume = latest_checkpoint(dirs["checkpoints"])
    with Timer() as t:
        trainer.train(resume_from_checkpoint=resume)

    out = trainer.predict(ds["test"], metric_key_prefix="test")
    probs = torch.softmax(torch.tensor(out.predictions, dtype=torch.float), -1).numpy()
    metrics = {
        "model": args.model_name, "run_name": args.run_name, "seed": args.seed, "modality": "text (caption)",
        **classification_metrics(out.label_ids, probs),
        **hist_cb.summary(trainer.state),
        "best_val_macro_f1": trainer.state.best_metric, "train_seconds": round(t.seconds, 1),
        "test_samples_per_second": out.metrics.get("test_samples_per_second"),
        **count_parameters(model), **hardware_info(),
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
        ({"path": it["rel_path"], "caption": it["caption"], "gold": CLASSES[it["label"]],
          "pred": CLASSES[int(pr.argmax())], "probs": [round(float(x), 5) for x in pr]}
         for it, pr in zip(splits["test"], probs)),
        dirs["results"] / "predictions.jsonl",
    )

    # caption embeddings ([CLS]) for the t-SNE plot
    model.eval()
    device = next(model.parameters()).device
    embs = []
    with torch.no_grad():
        for batch in trainer.get_test_dataloader(ds["test"]):
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            embs.append(model.base_model(**batch).last_hidden_state[:, 0].float().cpu().numpy())
    np.save(dirs["results"] / "embeddings.npy", np.concatenate(embs))

    if should_keep_model(args):
        dirs["final_model"].mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(dirs["final_model"]))
        tok.save_pretrained(str(dirs["final_model"]))
    cleanup_checkpoints(dirs["checkpoints"])
    print(f"Caption-only: accuracy = {metrics['test_accuracy']:.4f} | macro-F1 = {metrics['test_f1_macro']:.4f}")

    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(dirs["final_model"], dirs["results"], args.hub_repo,
                   title="SciBERT caption-based figure-type classifier (SPIQA)", base_model=args.model_name,
                   dataset="SPIQA captions (Pramanick et al., 2024)", task="text-classification")


if __name__ == "__main__":
    main()
