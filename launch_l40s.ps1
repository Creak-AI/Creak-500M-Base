# Train Creak ~500M on one L40S for a 5-hour GPU rental. Run from this directory.
Set-Location $PSScriptRoot

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "8" }
if (-not $env:HF_HOME) { $env:HF_HOME = (Join-Path (Get-Location) ".cache\huggingface") }

# Optional first arg: total minutes (default 300 = 5 hours), including data prep.
$MaxMinutes = if ($args.Count -ge 1) { $args[0] } else { "300" }

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

Start-Process -NoNewWindow python -ArgumentList "-m","hcrm","dash","--port","8765"
Write-Host "Dashboard: http://127.0.0.1:8765/  (follow checkpoints/train.log, do not copy a live tqdm bar)"
Write-Host "5-hour GPU budget: $MaxMinutes minutes from now (data prep counts)."

python -m hcrm train --max-minutes $MaxMinutes
