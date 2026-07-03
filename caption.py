"""Offline VLM captioning for remote-sensing hazy images.

Run this ONCE per dataset split before training/testing. It loads a
vision-language model, prompts it to describe the haziness of each
image, and caches the results into ``captions.json`` next to the input
directory. At train/test time only the cached JSON is read (the VLM is
never loaded again), so this adds zero runtime cost to training.

Usage:
    python caption.py --input_dir ./Haze1k-thick/train/hazy
    python caption.py --input_dir ./data/hazy --vlm_model Qwen/Qwen2-VL-2B-Instruct
    python caption.py --input_dir ./data/hazy --api_key sk-... --api openai

The script is idempotent: re-running it skips images already present in
``captions.json`` so you can safely interrupt and resume.

The VLM runs in fp16, batch=1, with images resized so the longest side
is <= 512 px, which keeps peak VRAM well under 15 GB on a Colab T4/L4.
"""
import os
import json
import argparse
from PIL import Image


HAZE_PROMPT = (
    "You are analyzing a remote sensing image taken from above. "
    "Rate the haziness of this image on a scale: clear, lightly hazy, "
    "moderately hazy, heavily hazy. Reply with EXACTLY one short "
    "sentence of the form: '<level>. <brief visibility description>'."
)

LEVEL_KEYWORDS = [
    ("heavily hazy", 3),
    ("moderately hazy", 2),
    ("lightly hazy", 1),
    ("clear", 0),
]


def parse_level(caption: str) -> int:
    """Extract a discrete haze level (0..3) from a VLM caption."""
    text = caption.lower()
    for kw, lvl in LEVEL_KEYWORDS:
        if kw in text:
            return lvl
    if "heavy" in text or "severe" in text or "dense" in text:
        return 3
    if "moderate" in text:
        return 2
    if "light" in text or "thin" in text or "slight" in text:
        return 1
    return 0


def list_images(directory: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(exts))


def load_existing(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache_path: str, cache: dict):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def resize_for_vlm(image: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = image.size
    scale = max_side / float(max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return image


# ---------------------------------------------------------------------------
# Captioning backends
# ---------------------------------------------------------------------------

def _caption_local_vlm(image_path, processor, model, device, vlm_model_id, max_side):
    import torch

    image = Image.open(image_path).convert("RGB")
    image = resize_for_vlm(image, max_side=max_side)

    is_qwen = "qwen" in vlm_model_id.lower() and "vl" in vlm_model_id.lower()

    if is_qwen:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": HAZE_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    else:
        prompt = f"User: <image>\n{HAZE_PROMPT}\nAssistant:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
    prompt_len = inputs["input_ids"].shape[1]
    generated = out[0][prompt_len:]
    caption = processor.decode(generated, skip_special_tokens=True).strip()
    return caption


def _caption_openai(image_path, api_key, model="gpt-4o", max_side=512):
    import base64
    import io
    import requests

    image = Image.open(image_path).convert("RGB")
    image = resize_for_vlm(image, max_side=max_side)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": HAZE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 80,
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def build_captioner(vlm_model_id, api_key, api, device, max_side):
    """Return a callable ``caption_fn(image_path) -> str`` and a cleanup fn."""
    if api_key and api in ("openai", "gemini"):
        if api == "openai":
            return (lambda p: _caption_openai(p, api_key=api_key, max_side=max_side)), lambda: None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gm = genai.GenerativeModel("gemini-1.5-flash")

            def gemini_caption(p):
                image = Image.open(p).convert("RGB")
                image = resize_for_vlm(image, max_side=max_side)
                return gm.generate_content([HAZE_PROMPT, image]).text.strip()

            return gemini_caption, lambda: None
        except ImportError:
            raise SystemExit("Install google-generativeai for --api gemini (pip install google-generativeai)")

    # Local VLM
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        vlm_model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    def caption_fn(p):
        return _caption_local_vlm(p, processor, model, device, vlm_model_id, max_side)

    def cleanup():
        del model
        del processor
        torch.cuda.empty_cache()

    return caption_fn, cleanup


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def main():
    parser = argparse.ArgumentParser(description="Caption hazy images with a VLM (offline cache).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing hazy images to caption.")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="HF model id of a local VLM.")
    parser.add_argument("--api", type=str, default="local", choices=["local", "openai", "gemini"])
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--max_side", type=int, default=512,
                        help="Resize longest image side to this before VLM (VRAM control).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-caption images already present in captions.json.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    cache_path = os.path.join(args.input_dir, "captions.json")
    cache = {} if args.overwrite else load_existing(cache_path)
    if not args.overwrite:
        print(f"Loaded {len(cache)} existing captions from {cache_path}")

    images = list_images(args.input_dir)
    todo = [f for f in images if args.overwrite or f not in cache]
    print(f"Total images: {len(images)} | To caption: {len(todo)}")
    if not todo:
        print("Nothing to do. Exiting.")
        return

    device = "cuda" if _cuda_available() else "cpu"
    if device == "cpu" and args.api == "local":
        print("WARNING: no CUDA detected; local VLM will be very slow on CPU.")

    caption_fn, cleanup = build_captioner(args.vlm_model, args.api_key, args.api, device, args.max_side)

    try:
        from tqdm import tqdm
        iterator = tqdm(todo, desc="Captioning")
    except ImportError:
        iterator = todo

    try:
        for i, fname in enumerate(iterator):
            ipath = os.path.join(args.input_dir, fname)
            try:
                caption = caption_fn(ipath)
            except Exception as e:
                print(f"\n[WARN] failed on {fname}: {e}")
                continue
            cache[fname] = {"caption": caption, "level": parse_level(caption)}
            if (i + 1) % 20 == 0 or (i + 1) == len(todo):
                save_cache(cache_path, cache)
    finally:
        save_cache(cache_path, cache)
        cleanup()
        print(f"Done. {len(cache)} captions in {cache_path}")


if __name__ == "__main__":
    main()