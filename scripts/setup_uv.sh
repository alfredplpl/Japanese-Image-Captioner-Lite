#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_LORA=1
RECREATE_VENV=0

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_uv.sh [options]

Options:
  --python VERSION   Python version for .venv. Default: 3.11
  --no-lora          Install without the LoRA extra.
  --recreate         Remove and recreate .venv before installing.
  -h, --help         Show this help.

Environment overrides:
  PYTHON_VERSION
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --no-lora)
      INSTALL_LORA=0
      shift
      ;;
    --recreate)
      RECREATE_VENV=1
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

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed.

Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then restart your shell or add uv to PATH.
EOF
  exit 1
fi

cd "$ROOT_DIR"

if [[ "$RECREATE_VENV" -eq 1 ]]; then
  rm -rf .venv
fi

uv venv --python "$PYTHON_VERSION"

if [[ "$INSTALL_LORA" -eq 1 ]]; then
  uv sync --extra lora
else
  uv sync
fi

uv run python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

cat <<'EOF'

Environment is ready.

Common commands:
  uv run jicl-train --config configs/lite_llava_caption.yaml
  uv run ./scripts/train_gpu.sh configs/lite_llava_caption.yaml
  uv run ./scripts/generate_gpu.sh --checkpoint outputs/lite-captioner --image path/to/image.jpg
EOF
