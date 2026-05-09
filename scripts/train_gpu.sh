#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/lite_llava_caption.yaml}"

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - "$CONFIG_PATH" <<'PY'
import sys

import torch
import yaml

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if cfg.get("train", {}).get("require_cuda", True) and not torch.cuda.is_available():
    raise SystemExit("CUDA is required, but torch.cuda.is_available() is false.")

if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gb = props.total_memory / 1024**3
        print(f"  [{index}] {props.name} ({gb:.1f} GiB)")
PY

MIXED_PRECISION="${MIXED_PRECISION:-$(python - "$CONFIG_PATH" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("train", {}).get("mixed_precision", "bf16"))
PY
)}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"

cd "$ROOT_DIR"
exec accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  -m jicl.train \
  --config "$CONFIG_PATH"
