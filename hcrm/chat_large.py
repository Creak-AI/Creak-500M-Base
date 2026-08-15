"""Verbose REPL for the Colab 32k / 17M HCRM checkpoint."""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from hcrm.chat import (
    HELP,
    _enable_ansi,
    c,
    clip_text,
    format_route,
    is_collapsed,
    nucleus,
    penalize_repeats,
    route_summary,
)
from hcrm.config import HCRMConfig
from hcrm.large import LargeHCRM, load_large
from hcrm.table import RuntimeTable

DEFAULT_CKPT = Path("32k_17M/hcrm_smoltalk_32k.pt")
DEFAULT_TABLE = Path("32k_17M/table.jsonl")


def load_hf_tokenizer(tok_id: str):
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def memories_prefix(hits: list) -> str:
    lines = []
    for entry, score in hits[:2]:
        body = clip_text(entry.text, 160)
        if is_collapsed(body):
            continue
        lines.append(f"- {body}")
    if not lines:
        return ""
    return "Known facts:\n" + "\n".join(lines)


def encode_prompt(tok, cfg: HCRMConfig, history: list[tuple[str, str]], user: str, mem_block: str) -> list[int]:
    messages: list[dict[str, str]] = []
    if mem_block:
        messages.append({"role": "system", "content": mem_block})
    for u, a in history[-8:]:
        a = clip_text(a, 800)
        if is_collapsed(a):
            continue
        messages.append({"role": "user", "content": clip_text(u, 600)})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    return list(ids)[-cfg.max_seq_len :]


def decode_token(tok, tid: int) -> str:
    return tok.decode([tid], skip_special_tokens=False).replace("\n", "\\n")


@torch.no_grad()
def generate(
    model: LargeHCRM,
    tok,
    prompt_ids: list[int],
    max_new: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    ctx: int,
) -> tuple[str, dict]:
    cfg = model.cfg
    device = next(model.parameters()).device
    stop = {cfg.eos_id, cfg.pad_id}
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid != tok.unk_token_id:
            stop.add(int(tid))
    if tok.eos_token_id is not None:
        stop.add(int(tok.eos_token_id))
    window_cap = max(32, min(int(ctx), cfg.max_seq_len))
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pieces: list[int] = []
    t0 = time.time()
    last_route = None
    channel_hist: list[int] = []
    out = None

    if verbose:
        out = model(ids[:, -window_cap:])
        last_route = out.route
        print(c("36", format_route("prompt", cfg, last_route)))

    for step in range(max_new):
        window = ids[:, -window_cap:]
        out = model(window)
        last_route = out.route
        logits = penalize_repeats(out.logits[:, -1, :].float(), pieces)
        if temperature <= 0:
            nxt = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = nucleus(logits / max(temperature, 1e-5), top_p)
            nxt = int(torch.multinomial(probs, 1).item())
        if nxt in stop:
            break
        if len(pieces) >= 2 and pieces[-1] == pieces[-2] == nxt:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        pieces.append(nxt)
        top_ch = int(out.route["top_channels"][0, -1, 0].item())
        channel_hist.append(top_ch)
        if verbose:
            print(c("90", format_route(decode_token(tok, nxt), cfg, out.route, step=step)))
            sys.stdout.flush()

    text = tok.decode(pieces, skip_special_tokens=True)
    for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(tag, "")
    stats = {
        "tokens": len(pieces),
        "seconds": time.time() - t0,
        "route": last_route,
        "channels": channel_hist,
        "hidden": None if out is None else out.hidden,
        "ids": ids,
    }
    return text.strip(), stats


def one_turn(
    model: LargeHCRM,
    tok,
    cfg: HCRMConfig,
    table: RuntimeTable,
    history: list[tuple[str, str]],
    user: str,
    temperature: float,
    top_p: float,
    max_new: int,
    verbose: bool,
    ctx: int,
) -> list[tuple[str, str]]:
    probe = encode_prompt(tok, cfg, history, user, "")
    with torch.no_grad():
        probe_out = model(torch.tensor([probe], dtype=torch.long, device=next(model.parameters()).device))
    cite, u, chans = route_summary(cfg, probe_out.route)
    key = model.mean_key(probe_out.hidden)
    hits = [
        (entry, score)
        for entry, score in table.lookup(chans, key[0], u, k=4, neighbor=1)
        if not is_collapsed(entry.text)
    ]
    mem_block = memories_prefix(hits)

    print(c("36", f"  [index] routing value u={u:.3f}  → channelite {cite}  channels {chans}"))
    if hits:
        print(c("33", f"  [table] {len(hits)} hits in nearby channels (locality search):"))
        for entry, score in hits:
            print(c("33", f"           ch{entry.channel}  {score:.2f}  {entry.text[:100]}"))
    else:
        print(c("33", "  [table] empty for this bucket — no prior memories loaded"))

    prompt_ids = encode_prompt(tok, cfg, history, user, mem_block)
    print(c("90", f"  [gen] {len(prompt_ids)} prompt tokens, ctx={min(ctx, cfg.max_seq_len)}  temp={temperature} top_p={top_p}"))
    text, stats = generate(model, tok, prompt_ids, max_new, temperature, top_p, verbose, ctx)
    tps = stats["tokens"] / max(stats["seconds"], 1e-6)
    print()
    print(c("1;37", text) if text else c("31", "(no tokens emitted)"))
    print()
    print(
        c(
            "90",
            f"  [done] {stats['tokens']} tokens in {stats['seconds']:.2f}s ({tps:.1f} tok/s)  "
            f"channel path {stats['channels'][:12]}{'…' if len(stats['channels'])>12 else ''}",
        )
    )
    if text and not is_collapsed(text):
        history.append((user, text))
        blob = f"user: {user}\nassistant: {text}"
        write_ch = chans[0] if chans else 0
        if stats["hidden"] is not None:
            key = model.mean_key(stats["hidden"][:, -min(32, stats["hidden"].size(1)) :])
        entry = table.write(write_ch, u, blob, key[0], role="turn")
        print(
            c(
                "35",
                f"  [learn] wrote runtime row → channelite {entry.channelite} channel {entry.channel} "
                f"(encoder frozen, table grew to {len(table.entries)})",
            )
        )
    elif is_collapsed(text):
        print(c("31", "  [learn] skipped table write — collapsed/looped output"))
    print()
    return history


def repl(args: argparse.Namespace) -> None:
    _enable_ansi()
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"No 32k checkpoint at {ckpt}")
    print(c("90", f"  loading {ckpt} …"))
    model, cfg, tok_id = load_large(ckpt, device)
    tok = load_hf_tokenizer(tok_id)
    cfg.pad_id = tok.pad_token_id if tok.pad_token_id is not None else cfg.pad_id
    cfg.eos_id = tok.eos_token_id if tok.eos_token_id is not None else cfg.eos_id
    table = RuntimeTable(Path(args.table), cfg.n_channels, cfg.n_channelites)

    print(c("1;35", "HCRM chat32  —  32k / ~17M  (SmolLM2 tokenizer, RoPE)"))
    print(
        f"  params={model.count_params():,}  d={cfg.d_model}  layers={cfg.n_layers}  "
        f"channelites={cfg.n_channelites}  channels={cfg.n_channels}  top_k={cfg.top_k}  "
        f"max_seq={cfg.max_seq_len}  gen_ctx={args.ctx}"
    )
    print(f"  checkpoint={ckpt}  table={table.path} ({len(table.entries)} memories)")
    print(f"  tokenizer={tok_id}")
    print("  type /help for commands. The encoder stays frozen; new facts go into the runtime table.")
    print()

    history: list[tuple[str, str]] = []
    verbose = not args.quiet
    temperature = args.temp
    max_new = args.max_new

    if args.once:
        one_turn(model, tok, cfg, table, history, args.once, temperature, args.top_p, max_new, verbose, args.ctx)
        return

    while True:
        try:
            raw = input(c("1;32", "HCRM32> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not raw:
            continue
        if raw.startswith("/"):
            cmd, *rest = raw.split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd in {"/quit", "/exit"}:
                print("bye")
                return
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/verbose":
                if arg in {"off", "0", "false"}:
                    verbose = False
                elif arg in {"on", "1", "true", ""}:
                    verbose = True
                print("verbose =", verbose)
                continue
            if cmd == "/table":
                info = table.summary()
                print(f"  {info['n']} entries")
                for e in table.entries[-12:]:
                    print(f"  [{e.ts[:19]}] cite={e.channelite} ch={e.channel} u={e.u:.3f}  {e.text[:90]}")
                continue
            if cmd == "/forget":
                table.clear()
                print("  runtime table cleared")
                continue
            if cmd == "/reset":
                history.clear()
                print("  conversation reset")
                continue
            if cmd == "/stats":
                print(
                    f"  params={model.count_params():,}  history_turns={len(history)}  "
                    f"table={len(table.entries)}  temp={temperature}  max_new={max_new}  ctx={args.ctx}"
                )
                continue
            if cmd == "/temp":
                temperature = float(arg)
                print("  temp =", temperature)
                continue
            if cmd == "/tokens":
                max_new = int(arg)
                print("  max_new =", max_new)
                continue
            if cmd == "/save":
                print(f"  table already persisted at {table.path}")
                continue
            print("  unknown command — /help")
            continue
        history = one_turn(
            model, tok, cfg, table, history, raw, temperature, args.top_p, max_new, verbose, args.ctx
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verbose REPL for the 32k / 17M HCRM checkpoint.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--table", default=str(DEFAULT_TABLE))
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--ctx", type=int, default=1024, help="CPU generate window (full 32k is too slow on CPU)")
    p.add_argument("--quiet", action="store_true", help="hide per-token routing")
    p.add_argument("--once", default="", help="run one prompt and exit (smoke test)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    repl(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
