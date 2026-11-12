# ResearchLens model plan: equal NLP and Vision

Rule: **NLP and Vision get the same number of fine-tuned models, the same evaluation template, the same number of
baselines and the same explainability depth.** Multimodal models are built on top of both, and every one is
reported as *text only* vs *image only* vs *both*.

## 1. Mirrored model pairs (4 NLP + 4 Vision)

| Role | NLP model (SciBERT family) | Vision model (ViT family) |
|---|---|---|
| **Detect** | **N1 NER**: find entity spans in text (Task, Method, Metric, Material…) · *SciERC* · entity F1 | **V1 Layout detection**: find regions on a page (text, title, figure, table, equation, caption) · *DocLayNet* · mAP |
| **Classify** | **N2 Caption classifier**: figure type from caption text · *SPIQA captions* · accuracy, macro-F1 | **V2 Figure classifier**: figure type from pixels · *SPIQA images* · accuracy, macro-F1 |
| **Structure** | **N3 Relation extraction**: links between entities (USED-FOR, EVALUATE-FOR…) · *SciERC* · relation F1 | **V3 Table structure recognition**: rows, columns, header cells of a table image · *PubTables-1M (subset)* · GriTS / mAP |
| **Read & find** | **N4 Evidence reranker**: question → paragraph that answers it · *QASPER* · evidence F1, MRR, Recall@k | **V4 Formula recognition**: equation image → LaTeX · *im2latex-100k* · exact match, edit distance, BLEU |

N2 and V2 use the **same figures and labels**, so they compare text vs vision directly.

## 2. Identical evaluation template (every NLP and every Vision model)

| Item | NLP | Vision |
|---|---|---|
| Seeds | 3, mean ± std | 3, mean ± std |
| Baseline 1 (general-domain model) | BERT-base | ImageNet ResNet-50 / non-document backbone |
| Baseline 2 (cheap / frozen) | BM25 or frozen-encoder linear probe | frozen-ViT linear probe |
| Training curves | loss, val metric, LR | loss, val metric, LR |
| Per-class scores + confusion matrix | yes | yes |
| Explainability | Integrated-Gradients word heatmaps | attention rollout / Grad-CAM heatmaps |
| Error analysis | worst-case sentence table | misclassified image grid |
| Embedding plot | t-SNE of [CLS] text embeddings | t-SNE of [CLS] image embeddings |
| Efficiency | params, train time, GPU memory, samples/s | same |
| Hub upload + model card | yes | yes |

## 3. Multimodal layer (built from the pairs)

| Model | Uses | Ablation reported |
|---|---|---|
| **M1 LayoutLMv3** page understanding | N1-style text + V1-style page image | text-only vs image-only vs LayoutLMv3 |
| **M2 Gated fusion** figure classifier *(built)* | N2 + V2 | N2 vs V2 vs fusion (+ McNemar test, gate analysis) |
| **M3 SigLIP cross-modal retriever** | question text ↔ figure image | text-only (BM25 on captions) vs image-only vs both |
| **M4 Multimodal evidence reranker** | N4 for paragraphs + figure scorer | N4 alone vs multimodal |
| **M5 Qwen3-VL-4B QLoRA** answer generator | retrieved text + figures + tables + formulas | text-only context vs text + visual context |

## 4. Report structure (equal chapters)

1. NLP methods & results: N1–N4
2. Vision methods & results: V1–V4
3. Multimodal fusion & system: M1–M5, three-way ablations, end-to-end QA (QASPER + SPIQA), Evidence Trace accuracy
4. Application: overview, comparison, Gap Map, idea generator

## 5. Build order (keeps both sides in step)

| Step | NLP | Vision | Multimodal |
|---|---|---|---|
| 1 | N1 NER ✅ | V2 figure classifier ✅ | M2 fusion ✅ |
| 2 | N2 caption classifier ✅ | V1 layout detection | — |
| 3 | N3 relation extraction | V4 formula recognition | M3 SigLIP retriever |
| 4 | N4 evidence reranker | V3 table structure | M1 LayoutLMv3, M4 reranker |
| 5 | — | — | M5 Qwen3-VL, application |
