# Fine-tuning settings (hyper-parameters)

All values live in `configs/*.yaml` and can be overridden on the command line (e.g. `--lr 5e-5`).
Each run also prints them at the start and saves them to `results/.../config.json`.

Shared by every model: optimizer **AdamW**, mixed precision **fp16** on GPU, evaluation + checkpoint **every epoch**,
the **best validation checkpoint** is kept and used for the single test evaluation, **early stopping** when the
validation metric stops improving, **3 seeds** (42, 43, 44).

## N1 · SciBERT NER (SciERC) — `configs/n1_ner.yaml` (regularised)

The first run (lr 3e-5, 10 epochs) showed validation loss rising from epoch 3 while training loss went to ~0.02,
so the defaults were changed to reduce over-fitting. The original settings are kept as the `scibert_noreg` run.

| Setting | Before (`scibert_noreg`) | Now (`scibert`) | Why |
|---|---|---|---|
| Pre-trained model | SciBERT | SciBERT | BERT pre-trained on 1.14M scientific papers |
| Epochs (max) | 10 | **8** | best epochs were 8–10; fewer passes over 1,861 sentences |
| Learning rate | 3e-5 | **2e-5** | smaller updates, less memorisation |
| Weight decay | 0.01 | **0.05** | stronger penalty on large weights |
| Encoder dropout | 0.1 | **0.2** | randomly drops more activations while training |
| Attention dropout | 0.1 | 0.1 | |
| Word embeddings | trainable | **frozen** | ~24M fewer trainable weights; rare words stay as pre-trained |
| Early stopping | 3 epochs | **2 epochs** without val-F1 gain | stop sooner |
| Batch size / max length | 16 / 256 | 16 / 256 | |
| LR schedule | linear, 10% warm-up | linear, 10% warm-up | |
| Best-model metric | validation entity F1 | validation entity F1 | |
| Baseline | — | `bert-base-uncased`, same regularised settings | value of scientific pre-training |

## V2 · ViT figure classifier (SPIQA images) — `configs/v2_figcls.yaml`

| Setting | Value | Why |
|---|---|---|
| Pre-trained model | `google/vit-base-patch16-224-in21k` | ViT-Base pre-trained on ImageNet-21k (14M images) |
| Epochs (max) | 8 | |
| Learning rate | 5e-5 (×20 for the linear probe) | frozen backbone needs a larger LR for the head |
| Batch size | 32 | |
| Image size | 224×224, padded to square with white | keeps wide plots undistorted |
| Augmentation | small random crop (85–100%), colour jitter; **no flips** | flipping would mirror text and axes |
| LR schedule | cosine, 10% warm-up | |
| Weight decay | 0.05 | ViTs benefit from stronger decay |
| Encoder dropout | 0.1 (ViT default 0.0) | regularisation |
| Label smoothing | 0.1 | reduces over-confidence |
| Class weights | inverse frequency | classes are imbalanced (many plots, few photos) |
| Early stopping | 3 epochs without val macro-F1 gain | |
| Best-model metric | validation macro-F1 | fair across imbalanced classes |
| Baselines | frozen-ViT linear probe; `microsoft/resnet-50` (CNN) | |

## N2 · SciBERT caption classifier (SPIQA captions) — `configs/v2_caption.yaml`

| Setting | Value |
|---|---|
| Pre-trained model | `allenai/scibert_scivocab_uncased` |
| Epochs (max) / LR / batch | 5 / 3e-5 / 32 |
| Max caption length | 128 tokens |
| Regularisation | weight decay 0.05, dropout 0.2, frozen embeddings, label smoothing 0.1 |
| Schedule | linear, 10% warm-up; early stopping 2 |
| Best-model metric | validation macro-F1 |

## M2 · ViT + SciBERT gated fusion — `configs/v2_fusion.yaml`

| Setting | Value | Why |
|---|---|---|
| Encoders | ViT-Base (image) + SciBERT (caption), both fine-tuned | |
| Fusion | 512-d projections + learned 2-way softmax gate | gate shows reliance on image vs text |
| Epochs (max) / LR / batch | 6 / 3e-5 / 24 | two encoders → smaller batch on a T4 |
| Modality dropout | 15% (hide image or caption at random) | forces the model to use both inputs |
| Dropout / label smoothing | 0.2 / 0.1 | |
| Schedule | cosine, 10% warm-up; early stopping 2 | |

## Planned experiment: learning-rate sensitivity

For the report, run each main model once more with LR ×0.5 and ×2 (seed 42) to show the results are not tuned by
luck: e.g. `--lr 1.5e-5 --run_name scibert_lr1.5e-5`.

# Metrics recorded for every model

Logged **every epoch** on the training set (fixed 2,000-example slice for the figure models) and the validation set,
and computed for the **best checkpoint on train, validation and test** (`train_*`, `val_*`, `test_*` in metrics.json).

| NER (N1) | Classifiers (V2 image, N2 caption, M2 fusion) |
|---|---|
| entity precision / recall / F1 (micro, macro, weighted; strict) | accuracy, balanced accuracy, top-2 accuracy |
| token accuracy (all tokens) | macro precision / recall / F1, weighted F1 |
| entity-token accuracy (tokens inside gold entities) | Cohen's kappa, ROC-AUC (one-vs-rest), log loss |
| sentence exact match | per-class precision / recall / F1, confusion matrix |
| per-entity-type P / R / F1, confusion matrix | |

Over-fitting evidence: `train_vs_val_*` curves (per epoch), `generalisation_gap` chart (train vs val vs test bars),
`gap_train_test_*` values, loss curves (running training loss, end-of-epoch training-set loss, validation loss).
`all_metrics.csv` lists every number for every run and seed.
