from __future__ import annotations

import sys


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


USAGE = """\
HCRM — Hierarchical Channel-Routed Memory (Creak 500M, L40S)

  python -m hcrm dash   [--port 8765]   live training dashboard (WebSocket)
  python -m hcrm train              5-hour GPU budget from now (data prep counts)
  python -m hcrm train --max-minutes 300
  python -m hcrm rl     [--max-minutes 90]   extra session; does not fit in the 5h SFT slot
  python -m hcrm chat    [--ckpt checkpoints/hcrm_slm.pt]

This folder is the ~500M GPU package (d=512, 8 layers, 64 channelites,
512 channels, d_ff=896, seq=2048, SmolLM2 tokenizer). Default train is a
**5 hour** L40S rental: download+pack, then chat SFT, then a ~45m reason
tail if time remains. Run it from 500M-GPU-RUN.

Resume is on by default. Follow progress in checkpoints/train.log or
python -m hcrm dash — do not copy a live tqdm bar.
"""


def main() -> None:
    _utf8_stdio()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(USAGE)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd in {"train", "fit"}:
        from hcrm.train import main as train_main

        train_main(rest)
    elif cmd in {"dash", "dashboard", "live"}:
        from hcrm.dash import main as dash_main

        dash_main(rest)
    elif cmd in {"rl", "grpo", "reinforce"}:
        from hcrm.rl import main as rl_main

        rl_main(rest)
    elif cmd in {"chat", "repl"}:
        from hcrm.chat import main as chat_main

        chat_main(rest)
    elif cmd in {"chat32", "repl32", "large"}:
        from hcrm.chat_large import main as chat_large_main

        chat_large_main(rest)
    else:
        print(f"unknown command {cmd!r}\n")
        print(USAGE)
        sys.exit(2)


if __name__ == "__main__":
    main()
