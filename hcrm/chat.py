from __future__ import annotations

from pathlib import Path
import argparse
import os
import re
import sys
import time

from typing import Any

import torch
import torch.nn.functional as F

from hcrm.config import SYSTEM_PROMPT, HCRMConfig
from hcrm.data import wrap_turn
from hcrm.model import HCRM
from hcrm.seed_qa import SEED_QA
from hcrm.table import RuntimeTable
from hcrm.tokenizer import apply_config_special_ids, decode, load_matching_tokenizer

HELP = """\
Commands
  /help              this message
  /verbose [on|off]  per-token channel routing
  /table             dump runtime table
  /forget            clear persistent memory
  /reset             clear chat history (keeps table)
  /stats             model + table stats
  /temp <float>      sampling temperature
  /tokens <int>      max new tokens
  /reason [on|off]   allow <|think|> (on by default)
  /save              write table to disk (already auto-saves)
  /quit              exit

Anything else is sent to Creak. Replies come from the model, not seed Q&A.
"""


def _enable_ansi() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if os.name == "nt":
        os.system("")


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def cfg_from_checkpoint(blob: dict) -> HCRMConfig:
    raw = dict(blob.get("config") or {})
    sd = blob.get("model") or {}
    if "pos_emb.weight" in sd:
        raw.setdefault("use_rope", False)
        raw.setdefault("conv_dilation", 1)
    else:
        raw.setdefault("use_rope", True)
    has_gru = any(k.startswith("blocks.0.mix.gru") for k in sd)
    raw.setdefault("use_gru", has_gru)
    if "blocks.0.mix.dw.weight" in sd:
        raw.setdefault("conv_kernel", int(sd["blocks.0.mix.dw.weight"].shape[-1]))
    return HCRMConfig(**{k: v for k, v in raw.items() if k in HCRMConfig.__dataclass_fields__})


def load_checkpoint(ckpt: Path, device: torch.device) -> tuple[HCRM, HCRMConfig, Any]:
    if ckpt.is_dir():
        pt = ckpt / "hcrm_slm.pt"
        tok_root = ckpt
    else:
        pt = ckpt
        tok_root = ckpt.parent
    if not pt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {pt}. Train first: python -m hcrm train"
        )
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(blob)
    weight_vocab = int(blob["model"]["tok_emb.weight"].shape[0])
    cfg.vocab_size = weight_vocab

    tok = load_matching_tokenizer(tok_root, weight_vocab, blob.get("tokenizer_json"))
    if tok is None:
        raise RuntimeError(
            f"Checkpoint vocab={weight_vocab} (d_model={cfg.d_model}) does not match any tokenizer. "
            "Expected checkpoints/hf_tokenizer or data/smollm2_creak_tokenizer. "
            "Retrain: python -m hcrm train"
        )
    cfg = apply_config_special_ids(cfg, tok)
    cfg.vocab_size = weight_vocab
    model = HCRM(cfg)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing:
        print(c("33", f"  [load] missing weights: {missing[:8]}"))
    if unexpected:
        print(c("33", f"  [load] unexpected weights: {unexpected[:8]}"))
    model.to(device)
    model.eval()
    return model, cfg, tok


def nucleus(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    out = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return out


_ARITH_RE = re.compile(
    r"^\s*(?:what(?:'s| is)\s+)?(-?\d+)\s*([+\-*/x×])\s*(-?\d+)\s*[?.!]?\s*$",
    re.I,
)
_WORD_RE = re.compile(r"[a-z0-9']+", re.I)
_GREET = {
    "hi",
    "hello",
    "hey",
    "yo",
    "howdy",
    "hiya",
    "hola",
    "hol",
    "bonjour",
    "hallo",
    "salut",
    "sup",
    "wassup",
    "wazzup",
    "whats up",
    "what up",
    "whats going on",
    "hows it going",
    "how is it going",
    "how are you doing",
    "how have you been",
    "hows things",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
}
_EMAIL_RAMBLE = (
    "i hope you're doing well",
    "i hope you are doing well",
    "touch base",
    "justry",
    "firsthandets",
    "i wanted to touch",
    "the process of our",
    "here are some tips",
    "is a classic question",
    "i'm here to help you with that",
    "im here to help you with that",
)


def clip_text(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text[:n]


def is_collapsed(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    compact = t.replace(" ", "")
    if len(compact) >= 8 and len(set(compact)) <= 2:
        return True
    words = t.split()
    return len(words) >= 6 and len(set(words)) <= 2


def is_garbage(text: str) -> bool:
    """Think-loops, GSM8K calculator junk, workplace-email collapse, and babble."""
    t = (text or "").strip()
    if is_collapsed(t):
        return True
    low = t.lower()
    if "<<" in t and ">>" in t:
        return True
    if low.count("the number of") >= 2:
        return True
    if any(p in low for p in _EMAIL_RAMBLE):
        return True
    if low.count("**") >= 2 and re.search(r"\b1\.", low):
        return True
    words = _WORD_RE.findall(low)
    if len(words) >= 10 and len(set(words)) / len(words) < 0.35:
        return True
    if len(words) >= 8:
        for n in (3, 4):
            grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            if grams and grams.count(max(grams, key=grams.count)) >= 3:
                return True
    return False


def _norm_user(user: str) -> str:
    u = (user or "").lower().replace("'", "")
    u = re.sub(r"[^a-z0-9+\-*/ ]+", " ", u)
    return re.sub(r"\s+", " ", u).strip()


def is_short_prompt(user: str) -> bool:
    return len(_WORD_RE.findall(_norm_user(user))) <= 3


def try_arithmetic(user: str) -> str | None:
    m = _ARITH_RE.match(user or "")
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op in {"x", "×"}:
        op = "*"
    if op == "/" and b == 0:
        return "Division by zero."
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
    if op == "/" and result == int(result):
        result = int(result)
    shown = op
    return f"{a} {shown} {b} = {result}."


def seed_fallback(user: str) -> str | None:
    u = _norm_user(user)
    if not u:
        return None
    for q, a in SEED_QA:
        if _norm_user(q) == u:
            return a
    if u in _GREET or any(u.startswith(g + " ") for g in _GREET):
        if u == "hol" or u.startswith("hola"):
            return "Hola. How can I help you?"
        if u.startswith("bonjour"):
            return "Bonjour. How can I help you?"
        return "Hello! How can I help you today?"
    if u in {"what", "huh", "uh", "umm", "eh", "ok", "okay", "yes", "no"}:
        return "Sure. Ask a question, or tell me what you want to do."
    return None


def ngram_repeating(pieces: list[int], n: int = 3, copies: int = 3) -> bool:
    need = n * copies
    if len(pieces) < need:
        return False
    gram = tuple(pieces[-n:])
    window = pieces[-need:]
    hits = sum(1 for i in range(0, len(window) - n + 1, n) if tuple(window[i : i + n]) == gram)
    return hits >= copies


def penalize_repeats(logits: torch.Tensor, pieces: list[int], penalty: float = 1.25) -> torch.Tensor:
    logits = logits.reshape(-1).clone()
    if not pieces:
        return logits
    for tid in set(pieces[-64:]):
        if logits[tid] > 0:
            logits[tid] /= penalty
        else:
            logits[tid] *= penalty
    if len(pieces) >= 2 and pieces[-1] == pieces[-2]:
        logits[pieces[-1]] = -1e9
    if len(pieces) >= 6 and pieces[-3:] == pieces[-6:-3]:
        logits[pieces[-3]] = -1e9
        logits[pieces[-2]] = -1e9
        logits[pieces[-1]] = -1e9
    return logits


def infer_tau(cfg: HCRMConfig) -> float:
    return float(getattr(cfg, "infer_tau", 0.1) or 0.1)


@torch.no_grad()
def generate(
    model: HCRM,
    tok: Any,
    prompt_ids: list[int],
    max_new: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    repetition_penalty: float = 1.25,
    allow_think: bool = False,
) -> tuple[str, dict]:
    cfg = model.cfg
    device = next(model.parameters()).device
    tau = infer_tau(cfg)
    stop = {cfg.eos_id, getattr(cfg, "end_id", cfg.eos_id)}
    blocked: set[int] = set()
    for name in (
        "<|user|>",
        "<|bos|>",
        "<|system|>",
        "<|assistant|>",
        "<|mem|>",
        "<|eos|>",
        "<|pad|>",
        "<|end|>",
    ):
        tid = tok.token_to_id(name)
        if tid is not None:
            stop.add(tid)
    if not allow_think:
        for name in ("<|think|>", "<|/think|>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                blocked.add(tid)
                stop.add(tid)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pieces: list[int] = []
    t0 = time.time()
    last_route = None
    channel_hist: list[int] = []
    out = None

    if verbose:
        out = model(ids[:, -cfg.max_seq_len :], tau=tau)
        last_route = out.route
        print(c("36", format_route("prompt", cfg, last_route)))

    for step in range(max_new):
        window = ids[:, -cfg.max_seq_len :]
        out = model(window, tau=tau)
        last_route = out.route
        logits = penalize_repeats(out.logits[:, -1, :].float(), pieces, penalty=repetition_penalty)
        for tid in blocked:
            logits[tid] = -1e9
        if temperature <= 0:
            nxt = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = nucleus(logits / max(temperature, 1e-5), top_p)
            nxt = int(torch.multinomial(probs, 1).item())
        if nxt in stop or nxt in blocked:
            break
        if len(pieces) >= 2 and pieces[-1] == pieces[-2] == nxt:
            break
        pieces.append(nxt)
        if ngram_repeating(pieces):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        top_ch = int(out.route["top_channels"][0, -1, 0].item())
        channel_hist.append(top_ch)
        if verbose:
            token_txt = decode(tok, [nxt], skip_special=False).replace("\n", "\\n")
            print(c("90", format_route(token_txt, cfg, out.route, step=step)))
            sys.stdout.flush()

    text = decode(tok, pieces, skip_special=False)
    for tag in (
        "<|eos|>",
        "<|user|>",
        "<|assistant|>",
        "<|bos|>",
        "<|end|>",
        "<|system|>",
        "<|mem|>",
        "<|pad|>",
    ):
        text = text.replace(tag, "")
    if not allow_think:
        text = text.replace("<|think|>", "").replace("<|/think|>", "")
    stats = {
        "tokens": len(pieces),
        "seconds": time.time() - t0,
        "route": last_route,
        "channels": channel_hist,
        "hidden": None if out is None else out.hidden,
        "ids": ids,
    }
    return text.strip(), stats


def format_route(label: str, cfg: HCRMConfig, route: dict, step: int | None = None) -> str:
    u = float(route["u"][0, -1].item())
    cite = int(route["channelite_mass"][0, -1].argmax().item())
    per = cfg.channels_per_channelite
    lo = cite * per
    hi = lo + per - 1
    chans = [int(x) for x in route["top_channels"][0, -1].tolist()]
    weights = [float(x) for x in route["top_weights"][0, -1].tolist()]
    wtxt = ", ".join(f"ch{c}:{w:.2f}" for c, w in zip(chans, weights))
    prefix = f"t{step:03d} " if step is not None else ""
    shown = label[:18] + ("…" if len(label) > 18 else "")
    return (
        f"  [route] {prefix}{shown!r:<20}  u={u:.3f}  "
        f"channelite {cite}  range {lo}–{hi}  {wtxt}"
    )


def memories_prefix(hits: list) -> str:
    chunks = []
    for entry, score in hits[:2]:
        body = clip_text(entry.text, 90)
        if is_garbage(body):
            continue
        chunks.append(f"<|mem|>{body}<|end|>")
    return "".join(chunks)


def history_prefix(history: list[tuple[str, str]], mem_block: str) -> str:
    parts = ["<|bos|>", wrap_turn("system", SYSTEM_PROMPT), mem_block]
    for u, a in history[-6:]:
        a = clip_text(a, 600)
        if is_garbage(a):
            continue
        parts.append(wrap_turn("user", clip_text(u, 300)))
        parts.append(wrap_turn("assistant", a))
    return "".join(parts)


def encode_prompt(
    tok: Any,
    cfg: HCRMConfig,
    history: list[tuple[str, str]],
    user: str,
    mem_block: str,
) -> list[int]:
    """Always keep the Creak system prompt + current user turn."""
    suffix_ids = tok.encode(wrap_turn("user", user) + "<|assistant|>").ids
    budget = max(8, cfg.max_seq_len - len(suffix_ids))
    prefix_ids = tok.encode(history_prefix(history, mem_block)).ids
    if len(prefix_ids) > budget:
        bos = tok.token_to_id("<|bos|>")
        sys_ids = tok.encode("<|bos|>" + wrap_turn("system", SYSTEM_PROMPT)).ids
        keep = budget - len(sys_ids)
        tail = prefix_ids[-max(0, keep) :] if keep > 0 else []
        prefix_ids = sys_ids + tail
        if bos is not None and prefix_ids and prefix_ids[0] != bos:
            prefix_ids = [bos] + prefix_ids
    ids = prefix_ids + suffix_ids
    bos = tok.token_to_id("<|bos|>")
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids[: cfg.max_seq_len]


def encode_query(tok: Any, cfg: HCRMConfig, user: str) -> list[int]:
    text = "<|bos|>" + wrap_turn("system", SYSTEM_PROMPT) + wrap_turn("user", user)
    return tok.encode(text).ids[: cfg.max_seq_len]


def route_summary(cfg: HCRMConfig, route: dict) -> tuple[int, float, list[int]]:
    u = float(route["u"][0, -8:].mean().item())
    cite = int(route["channelite_mass"][0, -8:].mean(dim=0).argmax().item())
    chans = [int(x) for x in route["top_channels"][0, -1].tolist()]
    return cite, u, chans


@torch.no_grad()
def seed_runtime_table(model: HCRM, tok: Any, cfg: HCRMConfig, table: RuntimeTable) -> int:
    if len(table.entries) >= 40:
        return 0
    device = next(model.parameters()).device
    tau = infer_tau(cfg)
    n = 0
    for user, assistant in SEED_QA:
        q_ids = encode_query(tok, cfg, user)
        q_out = model(torch.tensor([q_ids], dtype=torch.long, device=device), tau=tau)
        cite, u, chans = route_summary(cfg, q_out.route)
        q_key = model.mean_key(q_out.hidden)
        blob = f"user: {user}\nassistant: {assistant}"
        full_ids = encode_prompt(tok, cfg, [], user, "")
        a_ids = tok.encode(assistant + "<|end|>").ids
        full = torch.tensor([full_ids + a_ids], dtype=torch.long, device=device)
        full = full[:, -cfg.max_seq_len :]
        a_out = model(full, tau=tau)
        r_key = model.mean_key(a_out.hidden[:, -min(32, a_out.hidden.size(1)) :])
        table.write(
            chans[0] if chans else cite,
            u,
            blob,
            q_key[0],
            role="seed",
            key_query=q_key[0],
            key_response=r_key[0],
        )
        n += 1
    print(c("35", f"  [seed] wrote {n} Creak Q&A rows into the runtime table"))
    return n


def repl(args: argparse.Namespace) -> None:
    _enable_ansi()
    from hcrm.train import pick_device, setup_cuda

    setup_cuda()
    device = torch.device("cpu") if args.cpu else pick_device()
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    ckpt = Path(args.ckpt)
    print(c("36", f"  loading {ckpt} on {device}"))
    model, cfg, tok = load_checkpoint(ckpt, device)
    table = RuntimeTable(Path(args.table), cfg.n_channels, cfg.n_channelites)
    dropped = table.prune(lambda e: e.role != "seed" and is_garbage(e.text))
    if dropped:
        print(c("33", f"  [table] dropped {dropped} looped/think rows (they were poisoning retrieval)"))
    if args.seed_table:
        seed_runtime_table(model, tok, cfg, table)

    print(c("1;35", "Creak  —  Hierarchical Channel-Routed Memory"))
    print(
        f"  params={model.count_params():,}  d={cfg.d_model}  layers={cfg.n_layers}  "
        f"channelites={cfg.n_channelites}  channels={cfg.n_channels}  top_k={cfg.top_k}"
    )
    print(f"  checkpoint={ckpt}  table={table.path} ({len(table.entries)} memories)")
    print("  type /help for commands. Replies are sampled from the model; new facts go into the runtime table.")
    print()

    history: list[tuple[str, str]] = []
    verbose = not args.quiet
    temperature = args.temp
    max_new = args.max_new
    top_p = args.top_p
    rep = args.repetition_penalty
    allow_think = True

    while True:
        try:
            raw = input(c("1;32", "Creak> ")).strip()
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
                    f"table={len(table.entries)}  temp={temperature}  max_new={max_new}"
                )
                continue
            if cmd == "/temp":
                temperature = float(arg)
                print("  temp =", temperature)
                continue
            if cmd == "/reason":
                if arg in {"off", "0", "false"}:
                    allow_think = False
                elif arg in {"on", "1", "true", ""}:
                    allow_think = True
                print("reason/think =", allow_think)
                continue
            if cmd == "/tokens":
                max_new = max(8, min(256, int(arg)))
                print("  max_new =", max_new)
                continue
            if cmd == "/save":
                print(f"  table already persisted at {table.path}")
                continue
            print("  unknown command — /help")
            continue

        user = raw
        probe = encode_query(tok, cfg, user)
        with torch.no_grad():
            probe_out = model(torch.tensor([probe], dtype=torch.long), tau=infer_tau(cfg))
        cite, u, chans = route_summary(cfg, probe_out.route)
        q_key = model.mean_key(probe_out.hidden)
        hits = [
            (entry, score)
            for entry, score in table.lookup(
                chans, q_key[0], u, k=4, neighbor=1, query_text=user
            )
            if not is_garbage(entry.text)
        ]
        # Short prompts retrieve unrelated seeds and pull the tiny model into
        # no_robots email templates ("I hope you're doing well...").
        mem_block = "" if is_short_prompt(user) else memories_prefix(hits)

        print(c("36", f"  [index] routing value u={u:.3f}  → channelite {cite}  channels {chans}"))
        if hits:
            print(c("33", f"  [table] {len(hits)} hits in nearby channels (query-key search):"))
            for entry, score in hits:
                print(c("33", f"           ch{entry.channel}  {score:.2f}  {entry.text[:100]}"))
        else:
            print(c("33", "  [table] empty for this bucket — no prior memories loaded"))

        prompt_ids = encode_prompt(tok, cfg, history, user, mem_block)
        print(
            c(
                "90",
                f"  [gen] {len(prompt_ids)} prompt tokens, sampling temp={temperature} "
                f"top_p={top_p} rep={rep} tau={infer_tau(cfg):.2f}",
            )
        )
        text, stats = generate(
            model,
            tok,
            prompt_ids,
            max_new,
            temperature,
            top_p,
            verbose,
            repetition_penalty=rep,
            allow_think=allow_think,
        )
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

        if text and not is_garbage(text):
            history.append((user, text))
            blob = f"user: {user}\nassistant: {text}"
            write_ch = chans[0] if chans else 0
            r_key = q_key
            if stats["hidden"] is not None:
                r_key = model.mean_key(stats["hidden"][:, -min(32, stats["hidden"].size(1)) :])
            entry = table.write(
                write_ch,
                u,
                blob,
                q_key[0],
                role="turn",
                key_query=q_key[0],
                key_response=r_key[0],
            )
            print(
                c(
                    "35",
                    f"  [learn] wrote runtime row → channelite {entry.channelite} channel {entry.channel} "
                    f"(encoder frozen, table grew to {len(table.entries)})",
                )
            )
        else:
            print(c("31", "  [learn] skipped table write — looped/think garbage"))
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verbose REPL chat for Creak (HCRM).")
    p.add_argument("--ckpt", default="checkpoints/hcrm_slm.pt")
    p.add_argument("--table", default="runtime/table.jsonl")
    p.add_argument("--temp", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.25)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--quiet", action="store_true", help="hide per-token routing")
    p.add_argument("--seed-table", dest="seed_table", action="store_true", help="write seed Q&A into the runtime table")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    p.set_defaults(seed_table=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    repl(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
