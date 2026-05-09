from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from .config import load_config
from .model import LiteLlavaCaptioner, resolve_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Directory produced by jicl-train.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    cfg = load_config(checkpoint / "config.yaml")
    prompt = args.prompt or cfg.data.prompt

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation. Pass --device cpu --allow-cpu to run on CPU.")
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU generation is disabled by default. Pass --allow-cpu to opt in.")
    device = torch.device(args.device)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "cpu":
            dtype = torch.float32
    else:
        dtype = resolve_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(cfg.model.vision_model)
    cfg.model.torch_dtype = args.dtype

    model = LiteLlavaCaptioner(cfg.model)
    model.load_lite(checkpoint, map_location=device)
    model.to(device=device, dtype=dtype)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"].to(
        device=device,
        dtype=dtype,
    )
    tokens = tokenizer(prompt.strip() + "\n", return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            pixel_values=pixel_values,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(tokenizer.decode(generated[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
