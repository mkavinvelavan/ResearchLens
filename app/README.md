# ResearchLens application (Part 3)

`streamlit_app.py` — upload one or two papers, get an automatic overview, ask questions, explain figures/tables/formulas,
compare two papers, Research Gap Map and research ideas, every answer with an Evidence Trace.

* Our fine-tuned models (SciBERT NER / relations / evidence finder / caption classifier, ViT, TrOCR) load from the private
  Hugging Face repos with `HF_TOKEN`.
* Answers come from **Gemini** through the free Google AI Studio API (`GEMINI_API_KEY`). If one Gemini model is busy the
  app tries the next one (`GEMINI_MODELS`).
* Runs on a CPU. Without a GPU, low-memory mode is on: each processing model is loaded for its step and freed afterwards.

## Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub (allow access to the `ResearchLens` repo).
2. **Create app → Deploy a public app from GitHub**
   * Repository: `mkavinvelavan/ResearchLens`, branch `main`
   * Main file path: `app/streamlit_app.py`
   * App URL: choose a name, e.g. `researchlens`
3. **Advanced settings** → Python version **3.12**, and in **Secrets** paste (with your own values):

   ```toml
   HF_TOKEN = "your Hugging Face read token"
   GEMINI_API_KEY = "your Google AI Studio key"
   ```
4. Deploy. The first start installs packages (~5 min); the first paper downloads the models (~1–2 min) and processing
   a paper takes a few minutes on the free CPU.

Free-tier limits: about 2.7 GB RAM; the app sleeps after a few days without visitors (anyone can wake it with one
click); processed papers are lost when the app restarts. Only upload public papers (the free Gemini tier may use the
content to improve Google's products).

## Run locally / on Kaggle

```
pip install -r requirements-app.txt
HF_TOKEN=... GEMINI_API_KEY=... streamlit run app/streamlit_app.py
```
