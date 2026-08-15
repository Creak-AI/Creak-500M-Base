# Creak 500M — L40S GPU run (5 hours)

Standalone copy of TinyBig / HCRM for a **~500 million parameter** Creak trained on one **NVIDIA L40S (48GB)**. The local 10M CPU tree at the repo root is unchanged. Always run commands **from this folder**.

**GPU rental is 5 hours.** `python -m hcrm train` uses a **300-minute wall clock from process start**, including dataset download and packing. After prep, leftover time goes to chat SFT, then a ~45 minute reason tail if at least ~25 minutes remain. Interleaved `<|think|>` is in the main mix either way. Do not pass `--skip-reason`. RL is a **separate** session — it will not fit in the same 5 hours.

## Layout

| | |
| --- | --- |
| Params | ~500M unique (tied embed), almost all in a **512-row channel bank** |
| Width | d_model **512**, **8** layers, **64** channelites, **512** channels, d_ff **896**, top_k **8** |
| Context | **2048** tokens, RoPE, conv k=7 d=2 |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M-Instruct` + Creak specials (`<\|think\|>`, `<\|user\|>`, …) |
| Data | 5h-sized mix (~160k rows): smol-smoltalk 40k, OpenHermes 20k, UltraChat 10k, MetaMath 20k, Numina 25k, Orca-Math 12k, GSM8K 8k, plus smaller chat slices |
| Train | batch 4 × accum 4, lr 2.5e-4, **bf16** AMP, grad checkpoint, save every 50 steps |

Do not mix this runtime table with the local 10M table. After the first 500M checkpoint, chat with `/forget` if you reused a path.

## Setup on the GPU box

```bash
cd 500M-GPU-RUN
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/verify_500m.py      # should print ~500M
```

L40S is Ada and supports bf16. Confirm with `nvidia-smi` and `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"`.

## Train (5 hours)

Follow **`checkpoints/train.log`** or `python -m hcrm dash` (`http://127.0.0.1:8765/`). Do not copy a live tqdm bar.

```bash
python -m hcrm train                 # 5 hours from now
bash launch_l40s.sh                  # same, starts dash too
bash launch_l40s.sh 240              # 4 hours if the box is already warm
```

Checkpoints write every 50 optimizer steps so a preemption still leaves a usable `checkpoints/hcrm_slm.pt`.

If you have extra GPU time later:

```bash
python -m hcrm rl --max-minutes 90
```

## Chat

```bash
python -m hcrm chat --ckpt checkpoints/hcrm_slm.pt --table runtime/table.jsonl
```

Uses CUDA when available. `--cpu` forces CPU. Encoder stays frozen; new facts go into the runtime table.

## If it OOMs on 48GB

Keep grouped einsums. Try in order:

1. `--batch-size 2` (keep `--grad-accum 4` or raise it)
2. `--seq-len 1024`
3. `--channel-group 16` (smaller slices, more steps)
4. `--amp fp16` only if bf16 is unavailable

Do **not** set `--channel-group 0` — a full `btd,cdf→btcf` with C=512, ff=896, T=2048 will blow VRAM.

## CLI map

```
python -m hcrm dash
python -m hcrm train [--max-minutes 300] [--reason-minutes 45]
python -m hcrm rl     [--max-minutes 90]
python -m hcrm chat   [--ckpt checkpoints/hcrm_slm.pt]
```
