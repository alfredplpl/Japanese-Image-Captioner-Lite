#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-outputs/lite-captioner}"
IMAGE_PATH=""
PROMPT="${PROMPT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DTYPE="${DTYPE:-auto}"
DEVICE="${DEVICE:-cuda}"

usage() {
  cat <<'EOF'
Usage:
  scripts/generate_gpu.sh --image path/to/image.jpg [options]

Options:
  --checkpoint PATH       Checkpoint directory. Default: outputs/lite-captioner
  --image PATH            Image path. Required.
  --prompt TEXT           Prompt text. Default: prompt from checkpoint config.
  --max-new-tokens N      Max generated tokens. Default: 64
  --dtype auto|bf16|fp16|fp32
                          Default: auto
  --device cuda|cpu       Default: cuda
  --allow-cpu             Required when --device cpu is used.
  -h, --help              Show this help.

Environment overrides:
  CHECKPOINT, PROMPT, MAX_NEW_TOKENS, DTYPE, DEVICE
EOF
}

ALLOW_CPU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --allow-cpu)
      ALLOW_CPU=(--allow-cpu)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$IMAGE_PATH" ]]; then
  echo "--image is required." >&2
  usage >&2
  exit 2
fi

if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="$ROOT_DIR/$CHECKPOINT"
fi
if [[ "$IMAGE_PATH" != /* ]]; then
  IMAGE_PATH="$ROOT_DIR/$IMAGE_PATH"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$DEVICE" == "cuda" ]]; then
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required, but torch.cuda.is_available() is false.")

print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
fi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --image "$IMAGE_PATH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  "${ALLOW_CPU[@]}"
)

if [[ -n "$PROMPT" ]]; then
  ARGS+=(--prompt "$PROMPT")
fi

cd "$ROOT_DIR"
exec python -m jicl.generate "${ARGS[@]}"
