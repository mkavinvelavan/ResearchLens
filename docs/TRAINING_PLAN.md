# Training plan (lean): 5 fine-tuned models, one seed each

Time budget is limited, so ResearchLens fine-tunes **5 models** (balanced: 2 NLP, 2 Vision, 1 Multimodal) and uses
pre-trained models for everything else. Each new run uses **one seed (42)** with settings taken from published
fine-tuning recipes and the N1 experiments. Reliability is shown with **bootstrap 95% confidence intervals** on the
test set and **McNemar tests** between models (computed from saved predictions, no extra GPU time).

## Fine-tuned models

| # | Model | Track | Data | Key settings | Runs | Est. T4 time |
|---|---|---|---|---|---|---|
| 1 | SciBERT NER | NLP | SciERC | done (before / regularised / BERT baseline); + balanced run: lr 3e-5, ≤10 ep, dropout 0.2, wd 0.05, embeddings trainable, patience 2 | 1 extra | ~4 min |
| 2 | SciBERT caption → figure type | NLP | SPIQA captions | lr 3e-5, ≤4 ep, batch 32, len 128, wd 0.05, dropout 0.2, label smoothing 0.1 | 1 | ~15 min |
| 3 | ViT-B/16 figure classifier | Vision | SPIQA images (≤3k/class) | lr 5e-5, ≤6 ep, batch 32, wd 0.05, dropout 0.1, label smoothing 0.1, light augmentation | 1 | ~30 min |
| 4 | TrOCR-small formula → LaTeX | Vision | im2latex-100k (20k subset) | lr 5e-5, 3 ep, batch 16, beam 4 at test | 1 | ~1 h |
| 5 | ViT + SciBERT gated fusion | Multimodal | SPIQA figures + captions | lr 3e-5, ≤5 ep, batch 24, modality dropout 0.15 | 1 | ~45 min |
| – | ResNet-50 baseline (for #3) | Vision | SPIQA images | same as #3 | 1 | ~20 min |

One-time SPIQA download + preparation: ~0.5–1 h (cached on Drive as a zip afterwards).

**Remaining GPU time: ~3.5–4 h ≈ 2 Colab sessions.**

| Session | Runs |
|---|---|
| 1 | SPIQA prep → ViT (#3) → caption SciBERT (#2) → ResNet-50 baseline |
| 2 | Fusion (#5) → formula TrOCR (#4) → N1 balanced run |

## Pre-trained components (not trained)

| Component | Model |
|---|---|
| PDF parsing / layout / tables | MinerU |
| OCR inside figures | PaddleOCR |
| Relations between entities | Qwen3-VL prompted with SciBERT entities |
| Text embeddings for Qdrant | pre-trained sentence-embedding model |
| Image embeddings for Qdrant | SigLIP (zero-shot) |
| Answers, overview, comparison, Gap Map, figure explanations | Qwen3-VL |

## Evaluation without training (system level)

Retrieval Recall@5 and answer accuracy on QASPER / SPIQA question subsets, text-only vs text+image retrieval
ablation, Evidence Trace citation accuracy. Inference only.
