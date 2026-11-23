# Notebooks

The notebooks are thin launchers. All of the logic lives in the `researchlens/` package, so each notebook clones this
repository, installs the requirements and runs `python -m researchlens....` with a config from `configs/`.
The folders follow the project order: train the models, evaluate them, then run the application.

| Platform | How to open |
|---|---|
| Google Colab | **File → Open notebook → GitHub**, paste the repo URL. Runtime: **T4 GPU**. Add `HF_TOKEN` in **Secrets**. |
| Kaggle | **New notebook → File → Import notebook**. Accelerator: **GPU T4 x2** or **P100**. Add `HF_TOKEN` (and `GEMINI_API_KEY` for evaluation and app notebooks) under **Add-ons → Secrets**. |

## 1_training: fine-tune the models (Part 1)

| Notebook | Platform | What it trains | Dataset |
|---|---|---|---|
| `N1_train_ner_scibert.ipynb` | Colab | SciBERT NER, with a BERT-base baseline | SciERC |
| `V2_train_figcls_vit.ipynb` | Colab | ViT figure classifier, ResNet-50 and linear-probe baselines, SciBERT caption classifier, gated fusion | SPIQA |
| `K1_kaggle_trocr_formula.ipynb` | Kaggle | TrOCR formula reader (image → LaTeX) | im2latex-100k |
| `K2_kaggle_ner_multidomain.ipynb` | Kaggle | Multi-domain SciBERT NER (final NER model) | SciERC + ScienceIE |
| `K3_kaggle_vit_multidomain.ipynb` | Kaggle | Multi-domain ViT figure classifier (final ViT model) | SPIQA + BIOMEDICA |
| `K4_kaggle_relation_evidence.ipynb` | Kaggle | SciBERT relation extractor and SciBERT evidence finder | SciERC, QASPER |

## 2_evaluation: results, graphs and the full-system test

| Notebook | Platform | What it does |
|---|---|---|
| `R0_results_all_models.ipynb` | Colab | First results notebook (N1 and V2 runs saved on Google Drive) |
| `U1_drive_results_to_hf.ipynb` | Colab / Kaggle | Copies the Colab-era result files from Google Drive to the Hugging Face results dataset |
| `R1_results_all_models.ipynb` | Colab / Kaggle | **Final results**: every graph and table for every model, rebuilt from the saved result files (no retraining) |
| `K8_kaggle_system_eval.ipynb` | Kaggle | Full-system evaluation: QASPER retrieval and answers with and without RAG, SPIQA figure retrieval and answers with and without images, bootstrap confidence intervals |
| `K9_kaggle_retrieval_retest.ipynb` | Kaggle | Retrieval re-test after the re-ranking fix (paragraphs only); no Gemini key needed |
| `K10_kaggle_final_eval.ipynb` | Kaggle | Final evaluation: retrieval ablation, speed measurements and answers of the final system |

## 3_application: the ResearchLens app (Parts 2 and 3)

| Notebook | Platform | What it does |
|---|---|---|
| `K5_kaggle_end_to_end.ipynb` | Kaggle | End-to-end test on a GPU: parse a paper, run every model, build the Qdrant index, answer questions |
| `K6_kaggle_app.ipynb` | Kaggle | Starts the Streamlit app on Kaggle with a temporary public link |
| `K7_kaggle_demo_papers.ipynb` | Kaggle | Pre-processes the demo papers once on a GPU so they open in seconds on the website |

Prefixes: `N` = NLP model, `V` = vision model, `K` = run on Kaggle, `R` = results, `U` = upload utility.
