# ResearchLens

A multimodal research assistant for scientific papers (MSc Applied AI project). It reads text, figures, tables
and formulas, answers questions with an **Evidence Trace**, compares two papers, and maps research gaps.

The project has three tracks with matching weight: **NLP**, **Vision**, and **Multimodal Fusion**.

| Track | Model | Data | Main metric | Status |
|---|---|---|---|---|
| NLP | N1 SciBERT NER | SciERC | entity F1 | ✅ ready to train |
| NLP | N2 SciBERT relation extraction | SciERC | relation F1 | next |
| NLP | N3 SciBERT evidence reranker | QASPER | evidence F1, MRR | planned |
| Vision | V1 ViT layout detector | DocLayNet | mAP | planned |
| Vision | V2 ViT figure-type classifier (image) | SPIQA | accuracy, macro-F1 | ✅ ready to train |
| NLP | V2-T SciBERT figure-type classifier (caption) | SPIQA | accuracy, macro-F1 | ✅ ready to train |
| Fusion | V2-F ViT + SciBERT gated fusion (image + caption) | SPIQA | accuracy, macro-F1, McNemar | ✅ ready to train |
| Vision | V3 ViT formula recognizer | im2latex-100k | exact match, edit distance | planned |
| Fusion | F1 SigLIP cross-modal retriever | SPIQA | Recall@k | planned |
| Fusion | F2 Qwen3-VL QLoRA (optional) | SPIQA | answer accuracy | stretch |

## Workflow

```
VS Code (write code) ──push──► GitHub ──clone──► Colab GPU (train)
                                                   │  checkpoints + results ──► Google Drive
                                                   └─ final model ───────────► Hugging Face Hub
```

1. Write code in VS Code, commit and push to GitHub (Source Control panel).
2. Open a notebook from `notebooks/` in Colab (**File → Open notebook → GitHub**, paste the repo URL),
   or run it from VS Code with the Google Colab extension. If `drive.mount()` doesn't work through the
   VS Code extension, use the browser version for training runs.
3. In Colab: set the runtime to **T4 GPU** and add `HF_TOKEN` (write access) in **Secrets** (key icon).
4. Edit `REPO_URL` and `HF_USERNAME` in the first cell, then run all cells.

Training resumes automatically from the latest checkpoint on Drive if Colab disconnects. Just rerun the cell.

## Repository layout

```
configs/                 YAML hyper-parameters per model (CLI flags override them)
notebooks/               Colab launchers: N1_*, V2_*, and R0_results_all_models
researchlens/
  common/                seeding, Drive paths, result files, Hugging Face upload
  evaluation/            plot style, plotting functions, one report() per model
  nlp/                   SciERC loader, NER training, caption classifier
  vision/                SPIQA figure dataset builder, shared figure data, ViT training, attention maps
  fusion/                ViT + SciBERT gated fusion; cross-modal retrieval (coming)
app/                     ResearchLens application (Part 3, coming)
```

## Result files (same for every model)

```
Drive: ResearchLens/results/<model_tag>/<run_name>/seed_<n>/
  metrics.json        final test metrics + params, training time, GPU, speed
  history.csv         per-step train loss + learning rate, per-epoch validation metrics
  predictions.jsonl   every test prediction with ground truth
  config.json         exact arguments of the run
  embeddings.npy      (vision) test-set embeddings for t-SNE
```

`researchlens.evaluation.reports` turns these into every graph and table in `results/<model_tag>/report/`
(PNG at 300 dpi + vector PDF, plus `summary.csv` / `summary.tex` with mean ± std over seeds).
Graphs can be regenerated at any time without retraining.

### Graphs produced so far

V2 compares **text vs vision vs fusion on the same SPIQA figures**: caption-only SciBERT, image-only ViT,
probability averaging (no training) and a trained gated fusion. Extra V2 outputs: `modality_comparison`,
`per_class_by_modality`, `fusion_gate_by_class` (how much each class relies on image vs caption) and
`significance_mcnemar.csv` (is fusion significantly better than each single modality?).

| N1 NER | V2 figure classifier |
|---|---|
| loss curves, validation F1 per epoch, LR schedule | loss curves, validation macro-F1 per epoch, LR schedule |
| SciBERT vs BERT baseline bars | ViT vs linear-probe vs ResNet-50 bars |
| per-entity-type P/R/F1 | per-class P/R/F1 |
| entity-type confusion matrix | confusion matrix (normalised + counts) |
| recall vs entity length | ROC + precision-recall curves |
| error examples (CSV) | t-SNE of embeddings, confidence histogram |
| | attention rollout maps, misclassified grid |

## Run commands (inside Colab)

```bash
# N1 NER
python -m researchlens.nlp.train_ner --config configs/n1_ner.yaml --seed 42 --push_to_hub
python -m researchlens.nlp.train_ner --config configs/n1_ner.yaml --model_name bert-base-uncased --run_name bert_base

# V2 figure classifier
python -m researchlens.vision.prepare_spiqa_figures --out_dir /content/data/spiqa_figs --max_per_class 3000
python -m researchlens.vision.train_figcls --config configs/v2_figcls.yaml --seed 42 --push_to_hub
python -m researchlens.vision.train_figcls --config configs/v2_figcls.yaml --freeze_backbone --run_name vit_linear_probe
python -m researchlens.vision.train_figcls --config configs/v2_figcls.yaml --model_name microsoft/resnet-50 --run_name resnet50
python -m researchlens.nlp.train_caption_cls --config configs/v2_caption.yaml --seed 42
python -m researchlens.fusion.train_figcls_fusion --config configs/v2_fusion.yaml --seed 42
```

## Datasets

- **SciERC**: Luan et al., 2018. *Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction.*
  Downloaded automatically from the authors' processed release.
- **SPIQA**: Pramanick et al., 2024. *SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers.* `google/spiqa` on the Hub.
