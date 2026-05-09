from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DataConfig


class CaptionJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, cfg: DataConfig):
        self.jsonl_path = Path(jsonl_path)
        self.cfg = cfg
        self.image_root = Path(cfg.image_root) if cfg.image_root else self.jsonl_path.parent
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if self.cfg.image_key not in item or self.cfg.caption_key not in item:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_no} needs "
                        f"'{self.cfg.image_key}' and '{self.cfg.caption_key}' keys"
                    )
                records.append(item)
        if not records:
            raise ValueError(f"No records found in {self.jsonl_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record[self.cfg.image_key])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "caption": str(record[self.cfg.caption_key]),
            "image_path": str(image_path),
        }


class CaptionCollator:
    def __init__(self, image_processor, tokenizer, cfg: DataConfig):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        prompt = self.cfg.prompt.strip()
        prefix = f"{prompt}\n"
        texts = [prefix + caption for caption in captions]

        prompt_ids = self.tokenizer(
            [prefix] * len(batch),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_length,
            return_tensors="pt",
        )
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100

        for row, prompt_len in enumerate(prompt_ids["attention_mask"].sum(dim=1).tolist()):
            labels[row, :prompt_len] = -100

        pixel_values = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
