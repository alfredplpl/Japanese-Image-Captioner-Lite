from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from .config import load_config, save_config
from .data import CaptionCollator, CaptionJsonlDataset
from .model import LiteLlavaCaptioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainable_parameters(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(cfg, parameters):
    if cfg.train.optimizer == "adamw":
        return AdamW(
            parameters,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    if cfg.train.optimizer == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise ImportError("Install bitsandbytes or run `uv sync --extra bnb` to use train.optimizer: adamw8bit.") from exc
        return AdamW8bit(
            parameters,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    raise ValueError("train.optimizer must be one of: adamw, adamw8bit")


def build_scheduler(cfg, optimizer, warmup_steps: int, total_steps: int):
    if cfg.train.lr_scheduler == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if cfg.train.lr_scheduler == "constant_with_warmup":
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    raise ValueError("train.lr_scheduler must be one of: cosine, constant_with_warmup")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.train.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this config. Set train.require_cuda: false to allow CPU.")
    set_seed(cfg.train.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision if cfg.train.mixed_precision != "none" else None,
    )
    output_dir = Path(cfg.train.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.language_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(cfg.model.vision_model)

    train_dataset = CaptionJsonlDataset(cfg.data.train_jsonl, cfg.data)
    collator = CaptionCollator(image_processor, tokenizer, cfg.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    model = LiteLlavaCaptioner(cfg.model)
    optimizer = build_optimizer(cfg, trainable_parameters(model))
    updates_per_epoch = math.ceil(len(train_loader) / cfg.train.gradient_accumulation_steps)
    total_steps = cfg.train.epochs * updates_per_epoch
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    scheduler = build_scheduler(cfg, optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        scheduler,
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "config.yaml")
        tokenizer.save_pretrained(output_dir / "tokenizer")

    global_step = 0
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process)
    model.train()

    for epoch in range(cfg.train.epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters(model), cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if accelerator.is_main_process and global_step % cfg.train.log_every == 0:
                    progress.set_description(f"epoch={epoch + 1} loss={loss.item():.4f}")
                if (
                    accelerator.is_main_process
                    and cfg.train.save_every_steps > 0
                    and global_step % cfg.train.save_every_steps == 0
                ):
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_lite(output_dir / f"step-{global_step}")

    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_lite(output_dir)


if __name__ == "__main__":
    main()
