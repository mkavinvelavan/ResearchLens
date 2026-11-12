"""V3 - Formula recognition: fine-tune TrOCR-small to turn formula images into LaTeX (im2latex-100k).

Example (Kaggle / Colab):
    python -m researchlens.vision.train_formula_ocr --config configs/v3_formula.yaml --seed 42 --push_to_hub

Writes results/v3_formula/<run_name>/seed_<seed>/{metrics.json, history.csv, predictions.jsonl, config.json}
Metrics: exact match, character error rate (CER), token edit similarity, BLEU-4 - on train subset / val / test.
"""
from __future__ import annotations

import argparse

import numpy as np

from researchlens.common.results_io import HistoryCallback, print_split_table, save_config, save_metrics, save_predictions
from researchlens.common.utils import (
    Timer, add_regularisation_args, add_storage_args, backup_to_drive, cleanup_checkpoints, count_parameters,
    hardware_info, latest_checkpoint, make_training_args, parse_args_with_config, run_dirs, set_seed,
    should_keep_model, verify_model_saved,
)
from researchlens.vision.formula_metrics import formula_scores

KEYS = ["exact_match", "bleu4", "token_edit_sim", "cer"]


def get_args():
    p = argparse.ArgumentParser(description="TrOCR formula image -> LaTeX")
    p.add_argument("--model_name", default="microsoft/trocr-small-printed")
    p.add_argument("--dataset", default="yuntian-deng/im2latex-100k")
    p.add_argument("--model_tag", default="v3_formula")
    p.add_argument("--run_name", default="trocr_small")
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_train_eval", type=int, default=500, help="fixed train slice scored every epoch (over-fitting)")
    p.add_argument("--max_target_len", type=int, default=160)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_beams", type=int, default=1, help="beam size for the final test predictions")
    add_regularisation_args(p)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    add_storage_args(p)
    p.add_argument("--hub_repo", default="researchlens-trocr-formula")
    return parse_args_with_config(p)


def _columns(ds):
    """Find the image column and the LaTeX text column whatever they are called."""
    from datasets import Image as HFImage

    img = next((k for k, f in ds.features.items() if isinstance(f, HFImage)), None)
    txt = next((k for k in ("formula", "latex", "text", "label", "latex_formula") if k in ds.features), None)
    if img is None or txt is None:
        raise ValueError(f"Cannot find image/text columns in {list(ds.features)}")
    return img, txt


def main():
    args = get_args()
    import torch
    from datasets import load_dataset
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor,
                              VisionEncoderDecoderModel)

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    tok = processor.tokenizer

    # ---- data: fixed random subsets (seeded) of the official splits ----
    raw = load_dataset(args.dataset)
    split_names = {"train": "train", "val": "val" if "val" in raw else "validation", "test": "test"}
    img_col, txt_col = _columns(raw["train"])
    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    def short_enough(texts):
        return [len(ids) <= args.max_target_len for ids in tok(texts)["input_ids"]]

    data = {}
    for s, name in split_names.items():
        d = raw[name].shuffle(seed=args.seed)
        d = d.select(range(min(len(d), int(sizes[s] * 1.2))))
        before = len(d)
        d = d.filter(short_enough, batched=True, input_columns=[txt_col])
        d = d.select(range(min(len(d), sizes[s])))
        print(f"{s}: {len(d)} formulas (dropped {before - len(d)} longer than {args.max_target_len} tokens or surplus)")
        data[s] = d
    data["train_eval"] = data["train"].select(range(min(args.n_train_eval, len(data["train"]))))

    def transform(b):
        pix = processor(images=[im.convert("RGB") for im in b[img_col]], return_tensors="pt").pixel_values
        lab = tok(b[txt_col], padding="max_length", max_length=args.max_target_len, truncation=True).input_ids
        lab = [[t if t != tok.pad_token_id else -100 for t in seq] for seq in lab]
        return {"pixel_values": pix, "labels": torch.tensor(lab)}

    ds = {k: v.with_transform(transform) for k, v in data.items()}

    # ---- model ----
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        cfg.pad_token_id = tok.pad_token_id
        cfg.eos_token_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    model.generation_config.max_length = args.max_target_len
    model.generation_config.num_beams = 1
    if args.hidden_dropout is not None:
        model.config.decoder.dropout = args.hidden_dropout
        if hasattr(model.decoder.config, "dropout"):
            model.decoder.config.dropout = args.hidden_dropout
    with torch.no_grad():  # safetensors refuses non-contiguous weights (transformers v5)
        for t in model.parameters():
            if not t.data.is_contiguous():
                t.data = t.data.contiguous()

    def decode(ids):
        ids = np.where(ids != -100, ids, tok.pad_token_id)
        return [s.strip() for s in tok.batch_decode(ids, skip_special_tokens=True)]

    def compute_metrics(ep):
        preds = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return formula_scores(decode(preds), decode(ep.label_ids))

    def collate(batch):
        pix = [b["pixel_values"] if b["pixel_values"].dim() == 3 else b["pixel_values"][0] for b in batch]
        lab = [b["labels"] if b["labels"].dim() == 1 else b["labels"][0] for b in batch]
        return {"pixel_values": torch.stack(pix), "labels": torch.stack(lab)}

    targs = make_training_args(
        len(ds["train"]), args.warmup_ratio, args_cls=Seq2SeqTrainingArguments,
        output_dir=str(dirs["checkpoints"]), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay, label_smoothing_factor=args.label_smoothing, lr_scheduler_type="cosine",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="val_exact_match", greater_is_better=True, logging_steps=50,
        predict_with_generate=True, generation_max_length=args.max_target_len, generation_num_beams=1,
        fp16=args.fp16 and torch.cuda.is_available(), dataloader_num_workers=args.num_workers,
        remove_unused_columns=False, report_to=["wandb"] if args.wandb else "none",
        run_name=f"{args.model_tag}-{args.run_name}-s{args.seed}", seed=args.seed,
    )
    hist_cb = HistoryCallback(dirs["results"] / "history.csv", resume=latest_checkpoint(dirs["checkpoints"]) is not None)
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ds["train"],
        eval_dataset={"train": ds["train_eval"], "val": ds["val"]},
        data_collator=collate, compute_metrics=compute_metrics,
        callbacks=[hist_cb, EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    resume = latest_checkpoint(dirs["checkpoints"])
    with Timer() as t:
        trainer.train(resume_from_checkpoint=resume)

    # ---- final evaluation of the best checkpoint: train slice / val / test ----
    split_scores, test_preds = {}, None
    for split, d, beams in (("train", ds["train_eval"], 1), ("val", ds["val"], 1), ("test", ds["test"], args.num_beams)):
        o = trainer.predict(d, metric_key_prefix=split, num_beams=beams, max_length=args.max_target_len)
        preds = o.predictions[0] if isinstance(o.predictions, tuple) else o.predictions
        p_txt, r_txt = decode(preds), decode(o.label_ids)
        split_scores[split] = formula_scores(p_txt, r_txt)
        if split == "test":
            test_preds = (p_txt, r_txt)
    print_split_table(split_scores, KEYS)

    metrics = {
        "model": args.model_name, "dataset": args.dataset, "run_name": args.run_name, "seed": args.seed,
        **{f"{s}_{k}": v for s, sc in split_scores.items() for k, v in sc.items()},
        "gap_train_test_exact_match": split_scores["train"]["exact_match"] - split_scores["test"]["exact_match"],
        "n_train": len(ds["train"]), "n_val": len(ds["val"]), "n_test": len(ds["test"]),
        **hist_cb.summary(trainer.state), "best_val_exact_match": trainer.state.best_metric,
        "train_seconds": round(t.seconds, 1), **count_parameters(model), **hardware_info(),
    }
    save_metrics(metrics, dirs["results"] / "metrics.json")
    save_predictions(({"gold": r, "pred": p, "exact": "".join(p.split()) == "".join(r.split())}
                      for p, r in zip(*test_preds)), dirs["results"] / "predictions.jsonl")

    if should_keep_model(args):
        dirs["final_model"].mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(dirs["final_model"]))
        processor.save_pretrained(str(dirs["final_model"]))
        verify_model_saved(dirs["final_model"])
    cleanup_checkpoints(dirs["checkpoints"])
    print(f"Formula OCR: exact match = {metrics['test_exact_match']:.4f} | CER = {metrics['test_cer']:.4f} | "
          f"BLEU-4 = {metrics['test_bleu4']:.4f}")
    print(f"Results: {dirs['results']}")

    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(dirs["final_model"], dirs["results"], args.hub_repo,
                   title="TrOCR for formula image to LaTeX (im2latex-100k)", base_model=args.model_name,
                   dataset="im2latex-100k (Deng et al., 2017)", task="image-to-text")

    if should_keep_model(args):
        backup_to_drive(dirs["final_model"], dirs["drive_model"])


if __name__ == "__main__":
    main()
