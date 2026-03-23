# vision pipeline: generate caption + tags from an image using BLIP.

import os
import logging
from functools import lru_cache
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VISION_MODEL = os.getenv("VISION_MODEL", "Salesforce/blip-image-captioning-base")


@lru_cache(maxsize=1)
def _load_model():
    # load BLIP processor and model once
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    logger.info("Loading vision model: %s", VISION_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(VISION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL).to(device)
    return processor, model, device


def describe_image(image_bytes: bytes) -> dict:
    """
    Returns:
        {"caption": str, "tags": list[str]}
    """
    processor, model, device = _load_model()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # unconditional caption
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()

    # conditional prompts for tags
    tag_prompts = ["a photo of", "the color of the image is", "the background shows"]
    tags = []
    for prompt in tag_prompts:
        inp = processor(image, prompt, return_tensors="pt").to(device)
        out = model.generate(**inp, max_new_tokens=15)
        tag = processor.decode(out[0], skip_special_tokens=True)
        # strip the prompt prefix from output
        tag = tag.replace(prompt, "").strip().rstrip(".")
        if tag:
            tags.append(tag)

    return {"caption": caption, "tags": tags[:3]}
