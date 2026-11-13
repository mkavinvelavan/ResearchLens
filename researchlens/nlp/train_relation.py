"""N2 - Relation extraction with SciBERT on SciERC: given two entities in a sentence, which relation links them?

Input  : a sentence with two entities marked:  "we train @ BERT @ on # SQuAD # and report F1"
         (@ ... @ = first entity, # ... # = second entity; plain characters, so no new tokens are added and the
          embeddings can stay frozen. Typed markers are NOT used, so the model works with any NER label scheme.)
Output : USED-FOR, FEATURE-OF, HYPONYM-OF, PART-OF, COMPARE, CONJUNCTION, EVALUATE-FOR or NONE
Model  : SciBERT; the vectors at the two opening markers are concatenated -> dropout -> linear layer
         (the "entity start" representation of Baldini Soares et al., 2019).

Data   : every ordered pair of gold entities in a sentence. Annotated pairs get their relation type, all other
         pairs get NONE (that is the real situation at inference time, so NONE pairs are kept in val/test).
Metric : micro P/R/F1 over the 7 relation types (NONE = "no relation", not scored as a class), as in SciERC papers;
         also macro-F1, per-type scores and the train / val / test gap.

    python -m researchlens.nlp.train_relation --config configs/n2_relation.yaml --push_to_hub
Writes results/n2_relation/<run>/seed_42/{metrics.json, history.csv, predictions.jsonl, config.json}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from researchlens.common.results_io import HistoryCallback, print_split_table, save_config, save_metrics, save_predictions
from researchlens.common.utils import (
    LOCAL_DATA, Timer, add_regularisation_args, add_storage_args, apply_freezing, backup_to_drive, cleanup_checkpoints,
    count_parameters, dropout_overrides, hardware_info, latest_checkpoint, make_training_args, parse_args_with_config,
    run_dirs, set_seed, should_keep_model, verify_model_saved,
)
from researchlens.nlp.data_scierc import RELATION_TYPES, download_scierc, read_docs, sentence_examples

LABELS = ["NONE"] + RELATION_TYPES
E1, E2 = "@", "#"


# ------------------------------------------------------------------ data
def mark(tokens: list[str], a: tuple[int, int], b: tuple[int, int]) -> tuple[list[str], int, int]:
    """Insert markers around span a (E1) and span b (E2); returns words + indices of the two opening markers."""
    out, i1, i2 = [], -1, -1
    for i, w in enumerate(tokens):
        if i == a[0]:
            i1 = len(out); out.append(E1)
        if i == b[0]:
            i2 = len(out); out.append(E2)
        out.append(w)
        if i == a[1]:
            out.append(E1)
        if i == b[1]:
            out.append(E2)
    return out, i1, i2


def build_pairs(sentences: list[dict]) -> list[dict]:
    """All ordered pairs of (non-identical) gold entities per sentence, labelled with the relation or NONE."""
    pairs = []
    for ex in sentences:
        gold = {(r[0], r[1], r[2], r[3]): r[4] for r in ex["relations"]}
        ents = sorted({(s, e) for s, e, _ in ex["entities"]})
        types = {(s, e): t for s, e, t in ex["entities"]}
        for a in ents:
            for b in ents:
                if a == b or (a[0] <= b[1] and b[0] <= a[1]):   # same or overlapping span -> skip
                    continue
                words, i1, i2 = mark(ex["tokens"], a, b)
                pairs.append({
                    "id": ex["id"], "words": words, "m1": i1, "m2": i2,
                    "e1": " ".join(ex["tokens"][a[0]:a[1] + 1]), "e2": " ".join(ex["tokens"][b[0]:b[1] + 1]),
                    "t1": types[a], "t2": types[b],
                    "label": LABELS.index(gold.get((a[0], a[1], b[0], b[1]), "NONE")),
                })
    return pairs


def load_relation_splits(data_dir: Path) -> dict[str, list[dict]]:
    folder = download_scierc(data_dir)
    splits = {}
    for split, fname in [("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")]:
        sents = sentence_examples(read_docs(folder / fname))
        splits[split] = build_pairs(sents)
        n_pos = sum(p["label"] > 0 for p in splits[split])
        n_gold = sum(len(s["relations"]) for s in sents)
        print(f"{split:<10} {len(sents):>5} sentences  {len(splits[split]):>6} entity pairs  "
              f"{n_pos} with a relation (gold relations: {n_gold})")
    return splits


# ------------------------------------------------------------------ metrics
def relation_metrics(y_true, y_pred, prefix: str | None = None) -> dict:
    """Micro P/R/F1 over the relation types (NONE excluded), macro-F1 over types, accuracy over all pairs."""
    from sklearn.metrics import precision_recall_fscore_support

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos = list(range(1, len(LABELS)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=pos, average="micro", zero_division=0)
    _, _, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=pos, average="macro", zero_division=0)
    # "is there any relation?" regardless of type
    bt, bp = y_true > 0, y_pred > 0
    tp = int((bt & bp).sum())
    det_p = tp / max(int(bp.sum()), 1)
    det_r = tp / max(int(bt.sum()), 1)
    out = {"f1": f, "precision": p, "recall": r, "macro_f1": fm, "accuracy": float((y_true == y_pred).mean()),
           "detection_f1": 2 * det_p * det_r / max(det_p + det_r, 1e-9)}
    return {f"{prefix}_{k}": v for k, v in out.items()} if prefix else out


def per_type(y_true, y_pred) -> dict:
    from sklearn.metrics import classification_report

    rep = classification_report(y_true, y_pred, labels=list(range(1, len(LABELS))), target_names=LABELS[1:],
                                output_dict=True, zero_division=0)
    return {t: {m: rep[t][m] for m in ("precision", "recall", "f1-score", "support")} for t in LABELS[1:]}


# ------------------------------------------------------------------ model
def build_model(args):
    import torch
    from torch import nn
    from transformers import AutoModel

    class RelationClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                args.model_name, **dropout_overrides(args.model_name, args.hidden_dropout, args.attention_dropout))
            h = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(args.head_dropout)
            self.classifier = nn.Linear(2 * h, len(LABELS))
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        @property
        def base_model(self):          # so apply_freezing() finds the embeddings
            return self.encoder

        def forward(self, input_ids, attention_mask, m1, m2, token_type_ids=None, labels=None):
            hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
            idx = torch.arange(hs.size(0), device=hs.device)
            rep = torch.cat([hs[idx, m1], hs[idx, m2]], -1)
            logits = self.classifier(self.dropout(rep))
            out = {"logits": logits}
            if labels is not None:
                out["loss"] = self.loss_fn(logits.float(), labels)
            return out

    model = RelationClassifier()
    for t in list(model.parameters()) + list(model.buffers()):   # safetensors needs contiguous tensors (v5)
        t.data = t.data.contiguous()
    return model


def get_args():
    p = argparse.ArgumentParser(description="SciBERT relation extraction on SciERC")
    p.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--model_tag", default="n2_relation")
    p.add_argument("--run_name", default="scibert_relation")
    p.add_argument("--data_dir", default=str(LOCAL_DATA / "scierc"))
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=160)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--head_dropout", type=float, default=0.1)
    add_regularisation_args(p)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--push_to_hub", action="store_true")
    add_storage_args(p)
    p.add_argument("--hub_repo", default="researchlens-scibert-relation")
    return parse_args_with_config(p)


def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")
    splits = load_relation_splits(Path(args.data_dir))
    tok = AutoTokenizer.from_pretrained(args.model_name)

    def encode(p):
        enc = tok(p["words"], is_split_into_words=True, truncation=True, max_length=args.max_length)
        wid = enc.word_ids()
        first = {}
        for i, w in enumerate(wid):
            if w is not None and w not in first:
                first[w] = i
        m1, m2 = first.get(p["m1"]), first.get(p["m2"])
        if m1 is None or m2 is None:            # a marker was cut off by truncation -> use [CLS]
            m1, m2 = m1 or 0, m2 or 0
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                **({"token_type_ids": enc["token_type_ids"]} if "token_type_ids" in enc else {}),
                "m1": m1, "m2": m2, "labels": p["label"]}

    enc = {k: [encode(p) for p in v] for k, v in splits.items()}
    rng = random.Random(args.seed)
    train_eval = rng.sample(enc["train"], min(3000, len(enc["train"])))   # fixed slice -> over-fitting curve

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": torch.zeros(len(batch), n, dtype=torch.long),
               "attention_mask": torch.zeros(len(batch), n, dtype=torch.long)}
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = torch.zeros(len(batch), n, dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            out["input_ids"][i, :L] = torch.tensor(b["input_ids"])
            out["attention_mask"][i, :L] = 1
            if "token_type_ids" in out:
                out["token_type_ids"][i, :L] = torch.tensor(b["token_type_ids"])
        out["input_ids"][out["attention_mask"] == 0] = tok.pad_token_id
        out["m1"] = torch.tensor([b["m1"] for b in batch])
        out["m2"] = torch.tensor([b["m2"] for b in batch])
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    model = build_model(args)
    apply_freezing(model, args.freeze_embeddings, args.freeze_layers)

    def compute_metrics(ep):
        logits = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return relation_metrics(ep.label_ids, logits.argmax(-1))

    targs = make_training_args(
        len(enc["train"]), args.warmup_ratio,
        output_dir=str(dirs["checkpoints"]), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay, lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="val_f1", greater_is_better=True, logging_steps=50, remove_unused_columns=False,
        label_names=["labels"], fp16=args.fp16 and torch.cuda.is_available(), report_to="none",
        save_safetensors=False, seed=args.seed,
    )
    hist_cb = HistoryCallback(dirs["results"] / "history.csv", resume=latest_checkpoint(dirs["checkpoints"]) is not None)
    trainer = Trainer(model=model, args=targs, train_dataset=enc["train"],
                      eval_dataset={"train": train_eval, "val": enc["validation"]}, data_collator=collate,
                      compute_metrics=compute_metrics,
                      callbacks=[hist_cb, EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
    with Timer() as t:
        trainer.train(resume_from_checkpoint=latest_checkpoint(dirs["checkpoints"]))

    metrics = {"model": args.model_name, "run_name": args.run_name, "seed": args.seed,
               "labels": LABELS, "n_pairs": {k: len(v) for k, v in enc.items()},
               "n_positive_pairs": {k: sum(p["labels"] > 0 for p in v) for k, v in enc.items()}}
    preds = {}
    for split, d in (("train", train_eval), ("val", enc["validation"]), ("test", enc["test"])):
        o = trainer.predict(d, metric_key_prefix=split)
        lg = o.predictions[0] if isinstance(o.predictions, tuple) else o.predictions
        preds[split] = (o.label_ids, lg)
        metrics.update(relation_metrics(o.label_ids, lg.argmax(-1), prefix=split))
    y, lg = preds["test"]
    metrics["test_per_type"] = per_type(y, lg.argmax(-1))
    metrics.update(hist_cb.summary(trainer.state))
    metrics.update({"best_val_f1": trainer.state.best_metric, "train_seconds": round(t.seconds, 1),
                    "gap_train_test_f1": metrics["train_f1"] - metrics["test_f1"],
                    **count_parameters(model), **hardware_info()})
    print_split_table({s: {k: metrics[f"{s}_{k}"] for k in ("f1", "precision", "recall", "macro_f1", "detection_f1")}
                       for s in ("train", "val", "test")}, ["f1", "precision", "recall", "macro_f1", "detection_f1"])
    print("Per relation type (test):")
    for tname, m in metrics["test_per_type"].items():
        print(f"  {tname:<14} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1-score']:.2f}  (n={m['support']})")
    save_metrics(metrics, dirs["results"] / "metrics.json")
    probs = torch.softmax(torch.tensor(lg, dtype=torch.float), -1).numpy()
    save_predictions(({"id": p["id"], "e1": p["e1"], "e2": p["e2"], "t1": p["t1"], "t2": p["t2"],
                       "gold": LABELS[p["label"]], "pred": LABELS[int(pr.argmax())], "conf": round(float(pr.max()), 4)}
                      for p, pr in zip(splits["test"], probs)), dirs["results"] / "predictions.jsonl")

    if should_keep_model(args):
        final = dirs["final_model"]
        final.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(str(final))      # encoder: loadable with AutoModel
        tok.save_pretrained(str(final))
        torch.save(model.classifier.state_dict(), final / "relation_head.pt")
        (final / "relation_config.json").write_text(json.dumps(
            {"labels": LABELS, "markers": {"e1": E1, "e2": E2}, "max_length": args.max_length,
             "head": "concat(h[first @], h[first #]) -> dropout -> Linear(2*hidden, n_labels)",
             "base_model": args.model_name}, indent=1))
        verify_model_saved(final)
    cleanup_checkpoints(dirs["checkpoints"])
    print(f"Relation extraction (test): micro-F1 {metrics['test_f1']:.4f} | P {metrics['test_precision']:.4f} | "
          f"R {metrics['test_recall']:.4f} | train-test gap {metrics['gap_train_test_f1'] * 100:.1f} pts")

    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(dirs["final_model"], dirs["results"], args.hub_repo,
                   title="SciBERT relation extraction (SciERC, entity markers)", base_model=args.model_name,
                   dataset="SciERC (Luan et al., 2018)", task="text-classification")
    if should_keep_model(args):
        backup_to_drive(dirs["final_model"], dirs["drive_model"])


if __name__ == "__main__":
    main()
