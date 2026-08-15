"""Print checkpoint step/loss to checkpoints/train.log. Use a NEW terminal:

    Get-Content checkpoints\\train.log -Wait
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "hcrm_slm.pt"
LOG = ROOT / "checkpoints" / "train.log"


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    last_mtime = None
    last_step = None
    print(f"watching {CKPT}", flush=True)
    print(f"log      {LOG}", flush=True)
    while True:
        if not CKPT.exists():
            time.sleep(4)
            continue
        mtime = CKPT.stat().st_mtime
        if mtime == last_mtime:
            time.sleep(4)
            continue
        last_mtime = mtime
        try:
            blob = torch.load(CKPT, map_location="cpu", weights_only=False)
        except Exception:
            time.sleep(2)
            continue
        step = blob.get("step")
        if step == last_step:
            time.sleep(4)
            continue
        last_step = step
        loss = float(blob.get("loss") or 0.0)
        ppl = math.exp(min(loss, 20.0))
        line = (
            f"{time.strftime('%H:%M:%S')}  step={step}  "
            f"loss={loss:.3f}  ppl={ppl:.1f}\n"
        )
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="", flush=True)
        time.sleep(6)


if __name__ == "__main__":
    main()
