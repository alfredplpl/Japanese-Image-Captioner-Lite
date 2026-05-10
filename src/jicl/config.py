from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vision_model: str = "google/siglip2-so400m-patch16-512"
    language_model: str = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
    torch_dtype: str = "auto"
    projector_type: str = "mlp"
    projector_layer_norm: bool = False
    projector_dropout: float = 0.0
    num_image_tokens: int = 64
    freeze_vision: bool = True
    freeze_language_model: bool = True
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DataConfig:
    train_jsonl: str = "data/train.jsonl"
    val_jsonl: str | None = None
    image_root: str | None = None
    caption_key: str = "caption"
    image_key: str = "image"
    prompt: str = "画像を日本語で簡潔に説明してください。"
    max_text_length: int = 96
    max_train_samples: int | None = None
    num_workers: int = 2


@dataclass
class TrainConfig:
    output_dir: str = "outputs/lite-captioner"
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    require_cuda: bool = True
    log_every: int = 20
    save_every_steps: int = 1000
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    field_names = {f.name for f in instance.__dataclass_fields__.values()}
    for key, value in values.items():
        if key in field_names:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    _merge_dataclass(cfg.model, raw.get("model", {}))
    _merge_dataclass(cfg.data, raw.get("data", {}))
    _merge_dataclass(cfg.train, raw.get("train", {}))
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    data = {
        "model": cfg.model.__dict__,
        "data": cfg.data.__dict__,
        "train": cfg.train.__dict__,
    }
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
