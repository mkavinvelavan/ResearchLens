# Training plan: models, settings and time on a free Colab T4

Settings are chosen from published fine-tuning recipes for each model and from what N1 taught us
(regularise, but don't under-train), so each model is trained **once with the right settings** rather than tuned
by trial and error. Early stopping protects every run. Times are estimates for a T4 GPU with mixed precision and will
be replaced by measured times after each run.

Seed policy (to save GPU time): **3 seeds** when one run takes under ~20 min, **1 seed** for expensive runs
(reported as a single run; significance via McNemar / bootstrap on the test set instead of seed variance).

## NLP

| Model | Data | Main settings | Runs | Est. time |
|---|---|---|---|---|
| **N1 NER** (SciBERT) ✅ | SciERC | done: before / regularised; + **balanced** run: lr 3e-5, ≤10 ep, wd 0.05, dropout 0.2, embeddings trainable, patience 2 | 3 seeds | ~12 min |
| **N2 Caption classifier** (SciBERT) | SPIQA captions (~15k) | lr 3e-5, ≤4 ep, batch 32, len 128, wd 0.05, dropout 0.2, label smoothing 0.1 | 3 seeds | ~40 min |
| **N3 Relation extraction** (SciBERT) | SciERC | entity markers, lr 3e-5, ≤10 ep, batch 16, wd 0.05, dropout 0.2, patience 2 | 3 seeds | ~20 min |
| **N4 Evidence reranker** (SciBERT cross-encoder) | QASPER (~2.6k questions, 1 positive : 7 hard negatives) | lr 2e-5, 3 ep, batch 16, len 384, warm-up 10% | 1 seed | ~45 min |
| BERT-base baselines | as above | same settings | 1 seed each | ~45 min |

## Vision

| Model | Data | Main settings | Runs | Est. time |
|---|---|---|---|---|
| **V2 Figure classifier** (ViT-B/16) | SPIQA figures (≤3k / class) | lr 5e-5, ≤6 ep, batch 32, wd 0.05, dropout 0.1, label smoothing 0.1, light aug. | 3 seeds | ~1.5 h (+ ~0.5–1 h one-time dataset download/prep) |
| V2 baselines | same | ViT linear probe; ResNet-50 | 1 seed each | ~30 min |
| **V1 Layout detection** (YOLOS-small, ViT detector) | DocLayNet subset (~8k pages, 11 classes) | from COCO weights, lr 5e-5, 12 ep, img 512, batch 8 | 1 seed | ~2.5–3 h |
| **V3 Table structure** (Table Transformer, DETR) | FinTabNet / PubTables subset (~5k tables) | lr 5e-5, 10 ep, batch 4 | 1 seed | ~2 h |
| **V4 Formula → LaTeX** (TrOCR-small: ViT encoder + text decoder) | im2latex-100k subset (~50k) | lr 5e-5, 5 ep, batch 16, beam 4 at test | 1 seed | ~2–2.5 h |

## Multimodal

| Model | Data | Main settings | Runs | Est. time |
|---|---|---|---|---|
| **M2 Gated fusion** (ViT + SciBERT) | SPIQA figures + captions | lr 3e-5, ≤5 ep, batch 24, modality dropout 0.15 | 1 seed | ~45 min |
| **M3 SigLIP retriever** | SPIQA question/caption ↔ figure pairs | siglip-base, lr 1e-5, 3 ep, batch 32 | 1 seed | ~1 h |
| **M1 LayoutLMv3** page regions | DocLayNet subset (text + boxes + image) | lr 3e-5, 5 ep, batch 8 | 1 seed | ~1.5 h |
| **M4 Multimodal reranker** | QASPER + SPIQA | reuses N4 + M3 encoders; lr 2e-5, 2 ep | 1 seed | ~45 min |
| **M5 Qwen3-VL-2B QLoRA** (4-bit) | SPIQA QA subset (~3k) | LoRA r 16, lr 1e-4, 1 ep, batch 1 × grad-acc 8 | 1 seed | ~2–3 h |

## Totals

| Track | GPU time (est.) |
|---|---|
| NLP (N1 extra, N2, N3, N4 + baselines) | ~2.7 h |
| Vision (V1–V4 + baselines + data prep) | ~9–10 h |
| Multimodal (M1–M5) | ~6–7 h |
| **Total** | **~18–20 GPU hours** |

Free Colab sessions usually last up to ~3.5 h on a T4 and daily GPU access is not guaranteed, so this is roughly
**6–8 free sessions**. Long runs (V1, V3, V4, M5) each fit in one session. A paid Colab plan would allow
back-to-back sessions and faster GPUs.

## Order (keeps NLP and Vision in step, cheapest first)

1. N1 balanced run · N2 caption · V2 ViT (+ baselines) · M2 fusion  — one SPIQA dataset serves all four
2. N3 relation extraction · V4 formula recognition
3. N4 evidence reranker · M3 SigLIP retriever
4. V1 layout detection · M1 LayoutLMv3 (same DocLayNet subset)
5. V3 table structure · M4 multimodal reranker
6. M5 Qwen3-VL, then the application
