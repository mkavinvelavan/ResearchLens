# Fine-tuning settings (hyper-parameters)

All values live in `configs/*.yaml` and can be overridden on the command line (e.g. `--lr 5e-5`).
Each run also prints them at the start and saves them to `results/.../config.json`.

Shared by every model: optimizer **AdamW**, mixed precision **fp16** on GPU, evaluation + checkpoint **every epoch**,
the **best validation checkpoint** is kept and used for the single test evaluation, **early stopping** when the
validation metric stops improving, **3 seeds** (42, 43, 44).

## N1 · SciBERT NER (SciERC) — `configs/n1_ner.yaml`

| Setting | Value | Why |
|---|---|---|
| Pre-trained model | `allenai/scibert_scivocab_uncased` | BERT pre-trained on 1.14M scientific papers |
| Epochs (max) | 10 | small dataset; early stopping usually ends at 5–8 |
| Learning rate | 3e-5 | standard range for BERT fine-tuning (2e-5 – 5e-5) |
| Batch size | 16 | fits a T4 with sequences of 256 tokens |
| Max sequence length | 256 tokens | covers almost every SciERC sentence |
| LR schedule | linear decay, 10% warm-up | stabilises the first steps |
| Weight decay | 0.01 | light regularisation |
| Early stopping | 3 epochs without val-F1 gain | avoids overfitting |
| Best-model metric | validation entity F1 (strict, seqeval) | official NER metric |
| Labelling | BIO tags, first sub-word of each word | standard for token classification |
| Baseline | `bert-base-uncased`, same settings | shows the value of scientific pre-training |

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
