"""Print unique parameter count for the L40S 500M layout (no GPU required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hcrm.config import HCRMConfig


def count_from_cfg(cfg: HCRMConfig) -> dict[str, int]:
    d, c, f = cfg.d_model, cfg.n_channels, cfg.d_ff
    l, v, cites, k = cfg.n_layers, cfg.vocab_size, cfg.n_channelites, cfg.conv_kernel
    bank = c * d * f + c * f + c * f * d + c * d
    embed = v * d
    mix = l * (d * k + d + 2 * (d * d + d))
    router = (d * 1 + 1) + (d * c + c) + (d * cites + cites)
    norms = (2 * l + 1) * d
    return {
        "channel_bank": bank,
        "tied_embed": embed,
        "local_mix": mix,
        "router": router,
        "rmsnorms": norms,
        "unique_total": bank + embed + mix + router + norms,
    }


def main() -> None:
    cfg = HCRMConfig()
    parts = count_from_cfg(cfg)
    print(
        f"layout d={cfg.d_model} L={cfg.n_layers} cites={cfg.n_channelites} "
        f"C={cfg.n_channels} ff={cfg.d_ff} vocab={cfg.vocab_size} seq={cfg.max_seq_len}"
    )
    for name, n in parts.items():
        if name == "unique_total":
            continue
        print(f"  {name:16s} {n:>12,}")
    total = parts["unique_total"]
    print(f"  unique_total     {total:>12,}  ({total / 1e6:.2f}M)")
    try:
        import torch
        from hcrm.model import HCRM

        with torch.device("meta"):
            model = HCRM(cfg)
            live = model.count_params()
        print(f"  meta count_params {live:>12,}  ({live / 1e6:.2f}M)")
        if live != total:
            print(f"  formula delta {live - total:+,}")
    except Exception as exc:
        print(f"  meta instantiate skipped ({exc})")


if __name__ == "__main__":
    main()
