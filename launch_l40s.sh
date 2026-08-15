#!/usr/bin/env bash
# Train Creak ~500M on one L40S for a 5-hour GPU rental. Run from this directory.
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

# Optional first arg: total minutes (default 300 = 5 hours), including data prep.
MAX_MINUTES="${1:-300}"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), flush=True)
    assert torch.cuda.is_bf16_supported(), "L40S should support bf16"
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python -m hcrm dash --port 8765 >/tmp/creak_dash.log 2>&1 &
echo "Dashboard: http://127.0.0.1:8765/  (log also in checkpoints/train.log)"
echo "5-hour GPU budget: ${MAX_MINUTES} minutes from now (data prep counts)."

python -m hcrm train --max-minutes "$MAX_MINUTES"
