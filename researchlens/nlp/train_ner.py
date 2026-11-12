"""N1 - Scientific Named Entity Recognition: fine-tune SciBERT on SciERC.

Example (Colab):
    python -m researchlens.nlp.train_ner --config configs/n1_ner.yaml --seed 42
    # baseline for the report: general-domain BERT
    python -m researchlens.nlp.train_ner --config configs/n1_ner.yaml --model_name bert-base-uncased --run_name bert_base

Writes results/<model_tag>/<run_name>/seed_<seed>/{metrics.json, history.csv, predictions.jsonl, config.json}
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
from researchlens.nlp.data_scierc import BIO_LABELS, load_ner_splits


def get_args():
    p = argparse.ArgumentParser(description="SciBERT NER on SciERC")
    p.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--model_tag", default="n1_ner", help="folder name under results/")
    p.add_argument("--run_name", default="scibert")
    p.add_argument("--data_dir", default=str(LOCAL_DATA / "scierc"))
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    add_regularisation_args(p)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    p.add_argument("--push_to_hub", action="store_true")
    add_storage_args(p)
    p.add_argument("--hub_repo", default="researchlens-scibert-ner")
    return parse_args_with_config(p)


def main():
    args = get_args()
    import torch
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")

    label2id = {l: i for i, l in enumerate(BIO_LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    splits = load_ner_splits(Path(args.data_dir))

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def encode(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_length)
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            prev, labels = None, []
            for w in word_ids:
                # label only the first sub-token of each word; others are ignored (-100)
                labels.append(-100 if w is None or w == prev else label2id[tags[w]])
                prev = w
            all_labels.append(labels)
        enc["labels"] = all_labels
        return enc

    ds = {}
    for split, exs in splits.items():
        raw = Dataset.from_list([{"id": e["id"], "tokens": e["tokens"], "ner_tags": e["ner_tags"]} for e in exs])
        ds[split] = raw.map(encode, batched=True, remove_columns=["id", "tokens", "ner_tags"])

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(BIO_LABELS), id2label=id2label, label2id=label2id,
        **dropout_overrides(args.model_name, args.hidden_dropout, args.attention_dropout),
    )
    apply_freezing(model, args.freeze_embeddings, args.freeze_layers)

    def decode(pred_ids, label_ids):
        gold, pred = [], []
        for p_row, l_row in zip(pred_ids, label_ids):
            g, pr = [], []
            for p_, l_ in zip(p_row, l_row):
                if l_ != -100:
                    g.append(id2label[int(l_)])
                    pr.append(id2label[int(p_)])
            gold.append(g)
            pred.append(pr)
        return gold, pred

    def scores(gold, pred) -> dict:
        """Every NER metric: entity-level P/R/F1 (strict, micro/macro/weighted) + token accuracies."""
        rep = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
        g = np.array([t for s in gold for t in s])
        p_ = np.array([t for s in pred for t in s])
        ent = g != "O"
        out = {"accuracy": float((g == p_).mean()),                       # token accuracy (all tokens)
               "entity_token_accuracy": float((g[ent] == p_[ent]).mean()) if ent.any() else 0.0,
               "sentence_exact_match": float(np.mean([a == b for a, b in zip(gold, pred)]))}
        for avg in ("micro", "macro", "weighted"):
            for m, name in (("precision", "precision"), ("recall", "recall"), ("f1-score", "f1")):
                out[f"{name}_{avg}"] = rep[f"{avg} avg"][m]
        out["f1"] = out["f1_micro"]
        return out, rep

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        gold, pred = decode(np.argmax(logits, -1), labels)
        return scores(gold, pred)[0]

    targs = make_training_args(
        len(ds["train"]), args.warmup_ratio,
        output_dir=str(dirs["checkpoints"]),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="val_f1",
        greater_is_better=True,
        logging_steps=20,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=["wandb"] if args.wandb else "none",
        run_name=f"{args.model_tag}-{args.run_name}-s{args.seed}",
        seed=args.seed,
    )
    hist_cb = HistoryCallback(dirs["results"] / "history.csv", resume=latest_checkpoint(dirs["checkpoints"]) is not None)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        # evaluated every epoch: training set (to measure over-fitting) and validation set (for model selection)
        eval_dataset={"train": ds["train"], "val": ds["validation"]},
        data_collator=DataCollatorForTokenClassification(tok),
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

    # ---- final evaluation of the best checkpoint on train / val / test (test is used only here) ----
    split_scores, outputs = {}, {}
    for split, d in (("train", ds["train"]), ("val", ds["validation"]), ("test", ds["test"])):
        o = trainer.predict(d, metric_key_prefix=split)
        g, pr = decode(np.argmax(o.predictions, -1), o.label_ids)
        sc, rep = scores(g, pr)
        split_scores[split] = sc
        outputs[split] = (o, g, pr, rep)
    out, gold, pred, report = outputs["test"]

    metrics = {
        "model": args.model_name,
        "run_name": args.run_name,
        "seed": args.seed,
        **{f"{split}_{k}": v for split, sc in split_scores.items() for k, v in sc.items() if k != "f1"},
        "gap_train_test_f1": split_scores["train"]["f1_micro"] - split_scores["test"]["f1_micro"],
        "gap_val_test_f1": split_scores["val"]["f1_micro"] - split_scores["test"]["f1_micro"],
        "per_class": {
            k: {m: v[m] for m in ("precision", "recall", "f1-score", "support")}
            for k, v in report.items()
            if "avg" not in k
        },
        **hist_cb.summary(trainer.state),
        "best_val_f1": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_seconds": round(t.seconds, 1),
        "test_samples_per_second": out.metrics.get("test_samples_per_second"),
        **count_parameters(model),
        **hardware_info(),
    }
    save_metrics(metrics, dirs["results"] / "metrics.json")

    test_exs = splits["test"]
    save_predictions(
        (
            {"id": ex["id"], "tokens": ex["tokens"][: len(g)], "gold": g, "pred": p}
            for ex, g, p in zip(test_exs, gold, pred)
        ),
        dirs["results"] / "predictions.jsonl",
    )

    if should_keep_model(args):
        dirs["final_model"].mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(dirs["final_model"]))
        tok.save_pretrained(str(dirs["final_model"]))
    if should_keep_model(args):
        verify_model_saved(dirs["final_model"])
    cleanup_checkpoints(dirs["checkpoints"])
    print_split_table(split_scores, ["f1_micro", "f1_macro", "precision_micro", "recall_micro", "accuracy",
                                     "entity_token_accuracy"])
    print(f"Results: {dirs['results']}")

    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(
            dirs["final_model"], dirs["results"], args.hub_repo,
            title="SciBERT for Scientific NER (SciERC)", base_model=args.model_name,
            dataset="SciERC (Luan et al., 2018)", task="token-classification",
        )

    if should_keep_model(args):
        backup_to_drive(dirs["final_model"], dirs["drive_model"])

if __name__ == "__main__":
    main()
