from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config import ModelConfig


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str | None:
    if dtype_name == "auto":
        return "auto"
    if dtype_name in {"", "none", "None"}:
        return None
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError("torch_dtype must be one of: auto, bf16, fp16, fp32, none")


class VisionProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        lm_dim: int,
        projector_type: str,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("projector_dropout must be in the range [0.0, 1.0)")
        input_norm = nn.LayerNorm(vision_dim) if layer_norm else nn.Identity()
        output_norm = nn.LayerNorm(lm_dim) if layer_norm else nn.Identity()
        dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if projector_type == "linear":
            self.net = nn.Sequential(
                input_norm,
                nn.Linear(vision_dim, lm_dim),
                output_norm,
            )
        elif projector_type == "mlp":
            self.net = nn.Sequential(
                input_norm,
                nn.Linear(vision_dim, lm_dim),
                nn.GELU(),
                dropout_layer,
                nn.Linear(lm_dim, lm_dim),
                output_norm,
            )
        else:
            raise ValueError("projector_type must be 'linear' or 'mlp'")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LiteLlavaCaptioner(nn.Module):
    def __init__(self, cfg: ModelConfig, init_lora: bool = True):
        super().__init__()
        self.cfg = cfg
        torch_dtype = resolve_torch_dtype(cfg.torch_dtype)
        vision_model = AutoModel.from_pretrained(cfg.vision_model, torch_dtype=torch_dtype)
        self.vision_model = getattr(vision_model, "vision_model", vision_model)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            cfg.language_model,
            torch_dtype=torch_dtype,
        )

        if cfg.freeze_vision:
            self.vision_model.requires_grad_(False)
        if cfg.freeze_language_model:
            self.language_model.requires_grad_(False)

        if cfg.use_lora and init_lora:
            self._enable_lora(cfg)

        vision_dim = self._vision_hidden_size(cfg.vision_model)
        lm_dim = self.language_model.get_input_embeddings().embedding_dim
        self.projector = VisionProjector(
            vision_dim,
            lm_dim,
            cfg.projector_type,
            layer_norm=cfg.projector_layer_norm,
            dropout=cfg.projector_dropout,
        )

    @staticmethod
    def _vision_hidden_size(model_name: str) -> int:
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "vision_config"):
            return int(config.vision_config.hidden_size)
        return int(config.hidden_size)

    def _enable_lora(self, cfg: ModelConfig) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("Install with `pip install -e .[lora]` to use LoRA.") from exc

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        self.language_model = get_peft_model(self.language_model, lora_cfg)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.freeze_vision:
            with torch.no_grad():
                vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        else:
            vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        features = getattr(vision_outputs, "last_hidden_state", None)
        if features is None:
            features = vision_outputs[0]

        if features.size(1) > self.cfg.num_image_tokens:
            features = features[:, : self.cfg.num_image_tokens, :]
        return self.projector(features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        image_embeds = self.encode_images(pixel_values)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_attention = torch.ones(
            image_embeds.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([image_attention, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            image_labels = torch.full(
                image_embeds.shape[:2],
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([image_labels, labels], dim=1)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> torch.Tensor:
        image_embeds = self.encode_images(pixel_values)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_attention = torch.ones(
            image_embeds.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def save_lite(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), output_dir / "projector.pt")
        if hasattr(self.language_model, "save_pretrained") and self.cfg.use_lora:
            self.language_model.save_pretrained(output_dir / "lora")

    def load_lite(self, checkpoint_dir: str | Path, map_location: str | torch.device = "cpu") -> None:
        checkpoint_dir = Path(checkpoint_dir)
        state = torch.load(checkpoint_dir / "projector.pt", map_location=map_location)
        self.projector.load_state_dict(state)
        if self.cfg.use_lora:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Install peft to load a LoRA checkpoint.") from exc
            lora_dir = checkpoint_dir / "lora"
            if not lora_dir.exists():
                raise FileNotFoundError(f"LoRA checkpoint directory not found: {lora_dir}")
            self.language_model = PeftModel.from_pretrained(
                self.language_model,
                lora_dir,
                is_trainable=False,
            )
