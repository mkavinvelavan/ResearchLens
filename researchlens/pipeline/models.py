"""Loads the ResearchLens models (our fine-tuned ones from the private Hugging Face repos + pre-trained ones) lazily,
once per process, so the Streamlit app and the notebooks share the same code.

Our fine-tuned models (Part 1)                          Pre-trained, used as-is
  ner        SciBERT NER, multi-domain                    text_embedder  BAAI/bge-small-en-v1.5 (384-d)
  relation   SciBERT relation extractor (SciERC)          siglip         google/siglip-base-patch16-224 (768-d, text<->image)
  evidence   SciBERT evidence finder (QASPER)             vlm            Qwen/Qwen3-VL-4B-Instruct
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
VLM = os.environ.get("RL_VLM", "Qwen/Qwen3-VL-4B-Instruct")


class ModelBank:
    def __init__(self, device: str | None = None, token: str | None = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token or os.environ.get("HF_TOKEN")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

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

    @cached_property
    def vlm(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(VLM)
        # bf16 only on GPUs that really have it (Ampere+); a T4 "supports" bf16 only by slow emulation -> fp16
        native_bf16 = self.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
        kw = {"dtype": torch.bfloat16 if native_bf16 else self.dtype}
        if self.device == "cuda":
            # 2 GPUs (Kaggle 2 x T4): Qwen3-VL gets GPU 1 alone, our small models stay on GPU 0
            kw["device_map"] = {"": 1} if torch.cuda.device_count() > 1 else "auto"
            # 16 GB GPUs: 4-bit weights (~3 GB instead of ~9 GB) leave room for long multimodal prompts
            small_gpu = torch.cuda.get_device_properties(0).total_memory < 20e9
            if os.environ.get("RL_VLM_4BIT", "1" if small_gpu else "0") == "1":
                from transformers import BitsAndBytesConfig

                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if native_bf16 else torch.float16)
        m = AutoModelForImageTextToText.from_pretrained(VLM, **kw).eval()
        if self.device != "cuda":
            m.to(self.device)
        return proc, m
