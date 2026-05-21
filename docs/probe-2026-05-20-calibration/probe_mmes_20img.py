#!/usr/bin/env python3
"""
Python reference probe for multi-modal-embed-small (mmes) against the
20-image calibration corpus on the same 24 anchors as candle-binding.

Reproduces the production-path mmes embedding pipeline per the
2026-05-14 probe (docs/probe-2026-05-14/probe_mmes.py):

  - Image: SigLIP-base-patch16-512 vision_model.pooler_output ->
           trained Linear(768->384) projection -> L2-normalize
  - Text:  MiniLM-L6-v2 last_hidden_state mean-pool [384] -> L2-normalize
  - Cosine: dot product in 384-dim aligned space

Mirrors candle-binding's test_calibration_image_routing_pack output schema
so the resulting CSVs are directly diff-able against the candle-binding
pre/post-norm CSVs in this directory.

Anchors: 24 from config/signal/embedding/image-routing.yaml (verbatim).
Images: 20 from this directory's fixtures/ subdir.

Run:
  ~/vllm-semantic-router-multimodal-testing/.venv/bin/python3 \\
      docs/probe-2026-05-20-calibration/probe_mmes_20img.py
"""

import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipModel,
    SiglipProcessor,
)

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("probe_mmes_20img")

PROBE_DIR = Path(__file__).parent.resolve()
FIXTURE_DIR = PROBE_DIR / "fixtures"
OUTPUT_STEM = PROBE_DIR / "mmes_2026_05_21"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32

# Anchors verbatim from config/signal/embedding/image-routing.yaml
ANCHORS = {
    "identifier_document_imagery": [
        ("passport", "photograph of a passport page"),
        ("drivers_license", "photograph of a driver's license or national ID card"),
        ("credit_card", "photograph of a credit card or payment card"),
        ("badge", "photograph of an employee or visitor badge"),
        ("insurance_card", "photograph of a healthcare insurance card"),
        ("wristband", "photograph of a patient wristband or hospital identifier"),
        ("tax_doc", "photograph of a tax document or paystub with personal details"),
        ("paper_form", "photograph of a paper form filled in with personal information"),
    ],
    "code_or_terminal_imagery": [
        ("vscode_code", "screenshot of source code in an editor"),
        ("terminal", "terminal or shell window screenshot"),
        ("stacktrace", "screenshot of a log file or stack trace"),
        ("git_diff", "git diff or pull-request review screenshot"),
        ("debugger", "screenshot of a debugger paused at a breakpoint"),
        ("build_output", "command-line build output or test runner output"),
        ("api_response", "screenshot of an API response or HTTP request body"),
        ("db_client", "screenshot of a database client or SQL query result"),
    ],
    "ambient_office_imagery": [
        ("whiteboard", "photograph of a whiteboard with handwritten notes"),
        ("conference_room", "photograph of a conference room or meeting space"),
        ("desk_surface", "photograph of an office workspace or desk surface"),
        ("lobby", "photograph of a building lobby or interior hallway"),
        ("printer", "photograph of an office printer or shared equipment"),
        ("coffee_notebook", "photograph of a coffee cup or notebook on a desk"),
        ("plants", "photograph of indoor potted plants or office decor"),
        ("factory_floor", "photograph of a wide factory floor or warehouse aisle"),
    ],
}

ANCHOR_FLAT_ORDER = [
    short for rule_anchors in ANCHORS.values() for (short, _) in rule_anchors
]
ANCHOR_TEXT = {
    short: long for anchors in ANCHORS.values() for (short, long) in anchors
}


class MultiModalEmbedder(nn.Module):
    """Production-path mmes architecture, per the model card."""

    def __init__(self):
        super().__init__()
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.image_processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-512"
        )
        self.image_encoder = SiglipModel.from_pretrained(
            "google/siglip-base-patch16-512"
        ).vision_model
        self.image_proj = nn.Linear(768, 384)

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {
            k: v.to(next(self.parameters()).device) for k, v in inputs.items()
        }
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_image(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {
            k: v.to(next(self.parameters()).device) for k, v in inputs.items()
        }
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)


def load_weights(model):
    ckpt_path = hf_hub_download(
        repo_id="llm-semantic-router/multi-modal-embed-small",
        filename="model.pt",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    text_sd = {
        k.replace("text_encoder.encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("text_encoder.encoder.")
    }
    miss_t, unexp_t = model.text_encoder.load_state_dict(text_sd, strict=False)
    log.info(
        f"text_encoder loaded: {len(text_sd)} keys, "
        f"missing={len(miss_t)}, unexpected={len(unexp_t)}"
    )

    image_sd = {
        k.replace("image_encoder.vision_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("image_encoder.vision_encoder.")
    }
    miss_i, unexp_i = model.image_encoder.load_state_dict(image_sd, strict=False)
    log.info(
        f"image_encoder loaded: {len(image_sd)} keys, "
        f"missing={len(miss_i)}, unexpected={len(unexp_i)}"
    )

    proj_sd = {
        k.replace("image_encoder.projection.", ""): v
        for k, v in state_dict.items()
        if k.startswith("image_encoder.projection.")
    }
    miss_p, unexp_p = model.image_proj.load_state_dict(proj_sd, strict=False)
    log.info(
        f"image_proj loaded: {len(proj_sd)} keys, "
        f"missing={len(miss_p)}, unexpected={len(unexp_p)}"
    )


def bucket_from_filename(fname):
    if fname.startswith("inrule_"):
        return "inrule"
    if fname.startswith("adversarial_"):
        return "adversarial"
    if fname.startswith("ood_"):
        return "ood"
    return "unknown"


def main():
    log.info(f"Device: {DEVICE}, dtype: {DTYPE}")

    t0 = time.time()
    log.info("Constructing MultiModalEmbedder")
    model = MultiModalEmbedder()
    log.info("Loading trained checkpoint")
    load_weights(model)
    model = model.to(DEVICE).eval()
    log.info(f"Model ready in {time.time() - t0:.1f}s")

    log.info(f"Encoding {sum(len(a) for a in ANCHORS.values())} anchors")
    anchor_emb = {}
    with torch.no_grad():
        for short in ANCHOR_FLAT_ORDER:
            emb = model.encode_text(ANCHOR_TEXT[short])
            anchor_emb[short] = emb[0]

    images = sorted(
        p for p in FIXTURE_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    log.info(f"Found {len(images)} images in {FIXTURE_DIR}")

    summary_rows = []
    full_rows = []

    print("\n=== mmes (Python production path) Calibration: 20 images x 24 anchors ===\n")

    for img_path in images:
        fname = img_path.name
        bucket = bucket_from_filename(fname)

        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            img_emb = model.encode_image(img)
        img_emb = img_emb[0]

        cosines = {
            short: float((anchor_emb[short] @ img_emb).item())
            for short in ANCHOR_FLAT_ORDER
        }

        max_per_rule = {}
        for rule_name, rule_anchors in ANCHORS.items():
            best_short, best_cos = max(
                ((short, cosines[short]) for short, _ in rule_anchors),
                key=lambda x: x[1],
            )
            max_per_rule[rule_name] = (best_short, best_cos)

        top_rule = max(max_per_rule, key=lambda r: max_per_rule[r][1])
        top_anchor, top_cosine = max_per_rule[top_rule]

        print(
            f"  {fname:50s} | bucket={bucket:11s} "
            f"top={top_rule:35s} cos={top_cosine:.4f}"
        )

        summary_rows.append({
            "image": fname,
            "bucket": bucket,
            "max_identifier": f"{max_per_rule['identifier_document_imagery'][1]:.4f}",
            "max_code": f"{max_per_rule['code_or_terminal_imagery'][1]:.4f}",
            "max_ambient": f"{max_per_rule['ambient_office_imagery'][1]:.4f}",
            "top_rule": top_rule,
            "top_anchor": top_anchor,
            "top_cosine": f"{top_cosine:.4f}",
        })

        full_row = {"image": fname, "bucket": bucket}
        for short in ANCHOR_FLAT_ORDER:
            full_row[short] = f"{cosines[short]:.4f}"
        full_rows.append(full_row)

    summary_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_summary.csv")
    full_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_full.csv")

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with open(full_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {full_path}")

    passport_row = next(
        (r for r in full_rows if r["image"] == "inrule_identifier_passport.jpg"),
        None,
    )
    if passport_row is not None:
        print(
            f"\n*** Python reference for inrule_identifier_passport.jpg, "
            f"passport anchor: {passport_row['passport']}"
        )


if __name__ == "__main__":
    main()
