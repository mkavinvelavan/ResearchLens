"""Loads the ResearchLens models (our fine-tuned ones from the private Hugging Face repos + pre-trained ones) lazily,
once per process, so the Streamlit app and the notebooks share the same code.

Our fine-tuned models (Part 1)                          Pre-trained, used as-is
  ner        SciBERT NER, multi-domain                    text_embedder  BAAI/bge-small-en-v1.5 (384-d)
  relation   SciBERT relation extractor (SciERC)          siglip         google/siglip-base-patch16-224 (768-d, text<->image)
  evidence   SciBERT evidence finder (QASPER)             answers        Gemini via API (pipeline/answer.py)
  figtype    ViT figure classifier, multi-domain          ocr            PaddleOCR (optional)
  captype    SciBERT caption classifier
  formula    TrOCR formula reader
"""
from __future__ import annotations

import json
import os
from functools import cached_property

HF_USER = os.environ.get("RL_HF_USER", "Kavinvelavan")
REPOS = {
    "ner": f"{HF_USER}/researchlens-scibert-ner-multidomain",
    "relation": f"{HF_USER}/researchlens-scibert-relation",
    "evidence": f"{HF_USER}/researchlens-scibert-evidence",
    "figtype": f"{HF_USER}/researchlens-vit-figure-type-multidomain",
    "captype": f"{HF_USER}/researchlens-scibert-caption-figure-type",
    "formula": f"{HF_USER}/researchlens-trocr-formula",
}
TEXT_EMBEDDER = os.environ.get("RL_TEXT_EMBEDDER", "BAAI/bge-small-en-v1.5")
SIGLIP = os.environ.get("RL_SIGLIP", "google/siglip-base-patch16-224")


def cpu_limit() -> int:
    """CPU cores this container may really use. Hosted containers (Streamlit Cloud) often see the host's many cores but
    are limited to ~2 by a cgroup quota; using more threads than that makes PyTorch much slower."""
    n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]       # cgroup v2: "200000 100000"
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        try:                                                                     # cgroup v1
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = min(n, max(1, q // per))
        except Exception:
            pass
    return int(os.environ.get("RL_THREADS", n))


class ModelBank:
    def __init__(self, device: str | None = None, token: str | None = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token or os.environ.get("HF_TOKEN")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        # Low-memory mode (free hosting, ~2.7 GB RAM): a processing model is loaded for its step and freed right after,
        # so only the search models (bge, SigLIP, evidence finder) stay in memory. On by default without a GPU.
        self.low_memory = os.environ.get("RL_LOW_MEMORY", "0" if self.device == "cuda" else "1") == "1"
        if self.device == "cpu":
            torch.set_num_threads(cpu_limit())
            print(f"[models] CPU mode: {torch.get_num_threads()} threads, low_memory={self.low_memory}", flush=True)

    def release(self, *names: str):
        """Free models that are no longer needed (only in low-memory mode)."""
        if not self.low_memory:
            return
        import gc

        freed = [n for n in names if self.__dict__.pop(n, None) is not None]
        if freed:
            gc.collect()
            print(f"[models] freed {', '.join(freed)}", flush=True)

    def _kw(self):
        return {"token": self.token} if self.token else {}

    # ------------------------------------------------------------------ our fine-tuned models
    @cached_property
    def ner(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REPOS["ner"], **self._kw())
        m = AutoModelForTokenClassification.from_pretrained(REPOS["ner"], **self._kw()).to(self.device).eval()
        return tok, m

    @cached_property
    def relation(self):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REPOS["relation"], **self._kw())
        enc = AutoModel.from_pretrained(REPOS["relation"], **self._kw()).to(self.device).eval()
        cfg = json.load(open(hf_hub_download(REPOS["relation"], "relation_config.json", **self._kw())))
        head = torch.nn.Linear(2 * enc.config.hidden_size, len(cfg["labels"]))
        head.load_state_dict(torch.load(hf_hub_download(REPOS["relation"], "relation_head.pt", **self._kw()),
                                        map_location="cpu"))
        return tok, enc, head.to(self.device).eval(), cfg

    @cached_property
    def evidence(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REPOS["evidence"], **self._kw())
        m = AutoModelForSequenceClassification.from_pretrained(REPOS["evidence"], **self._kw()).to(self.device).eval()
        return tok, m

    @cached_property
    def figtype(self):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(REPOS["figtype"], **self._kw())
        m = AutoModelForImageClassification.from_pretrained(REPOS["figtype"], **self._kw()).to(self.device).eval()
        return proc, m

    @cached_property
    def captype(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REPOS["captype"], **self._kw())
        m = AutoModelForSequenceClassification.from_pretrained(REPOS["captype"], **self._kw()).to(self.device).eval()
        return tok, m

    @cached_property
    def formula(self):
        from transformers import AutoProcessor, VisionEncoderDecoderModel

        proc = AutoProcessor.from_pretrained(REPOS["formula"], **self._kw())
        m = VisionEncoderDecoderModel.from_pretrained(REPOS["formula"], **self._kw()).to(self.device).eval()
        return proc, m

    # ------------------------------------------------------------------ pre-trained
    @cached_property
    def text_embedder(self):
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TEXT_EMBEDDER)
        m = AutoModel.from_pretrained(TEXT_EMBEDDER).to(self.device).eval()
        return tok, m

    @cached_property
    def siglip(self):
        from transformers import AutoModel, AutoProcessor

        proc = AutoProcessor.from_pretrained(SIGLIP)
        m = AutoModel.from_pretrained(SIGLIP).to(self.device).eval()
        return proc, m

    @cached_property
    def ocr(self):
        try:
            from paddleocr import PaddleOCR

            return PaddleOCR(lang="en")
        except Exception as e:  # optional dependency
            print(f"PaddleOCR not available ({type(e).__name__}); figure text comes from the PDF text layer only.")
            return None
