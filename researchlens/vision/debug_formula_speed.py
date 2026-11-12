"""Quick speed check for the TrOCR pipeline: where does the time go (data vs GPU)?

    python -m researchlens.vision.debug_formula_speed
"""
import time

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

T0 = time.time()


def lap(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


name = "microsoft/trocr-small-printed"
proc = TrOCRProcessor.from_pretrained(name)
tok = proc.tokenizer
print("image processor class:", type(proc.image_processor).__name__)
raw = load_dataset("yuntian-deng/im2latex-100k", split="train")
lap(f"dataset loaded: {len(raw)} rows, columns {list(raw.features)}")
sub = raw.select(range(64))
ims = [sub[i]["image"] for i in range(16)]
lap(f"16 images decoded, sizes {[im.size for im in ims[:4]]}")
t = time.time()
pix = proc(images=[im.convert("RGB") for im in ims], return_tensors="pt").pixel_values
lap(f"processor on 16 images: {time.time() - t:.2f}s -> {tuple(pix.shape)}")
txt_col = next(k for k in ("formula", "latex", "text") if k in sub.features)
lab = tok([sub[i][txt_col] for i in range(16)], padding="max_length", max_length=160, truncation=True,
          return_tensors="pt").input_ids
model = VisionEncoderDecoderModel.from_pretrained(name).cuda()
model.config.decoder_start_token_id = tok.cls_token_id
model.config.pad_token_id = tok.pad_token_id
lap("model on GPU")
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
scaler = torch.amp.GradScaler()
pix, lab = pix.cuda(), lab.cuda()
lab[lab == tok.pad_token_id] = -100
for i in range(6):
    torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(pixel_values=pix, labels=lab).loss
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
    torch.cuda.synchronize()
    lap(f"train step {i}: {time.time() - t:.2f}s  loss {loss.item():.3f}")
model.eval()
t = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    out = model.generate(pixel_values=pix[:16], max_length=160, num_beams=1)
torch.cuda.synchronize()
lap(f"generate 16 formulas: {time.time() - t:.2f}s, output len {out.shape[1]}")
print("GPU:", torch.cuda.get_device_name(0), "| torch threads:", torch.get_num_threads())
