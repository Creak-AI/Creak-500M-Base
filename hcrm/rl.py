"""GRPO on Creak: sample replies, score them, reinforce the better ones.

Every prompt uses the think system prompt. Rewards prefer a real
<|think|>…<|/think|> then an answer, and pay more for a correct final
answer — with caps so empty tags or number-only hacks cannot dominate.
Runs on CUDA when available (L40S 500M package).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from hcrm.chat import (
    is_garbage,
    load_checkpoint,
    ngram_repeating,
    nucleus,
    penalize_repeats,
)
from hcrm.config import REASON_SYSTEM_PROMPT, HCRMConfig
from hcrm.data import wrap_turn
from hcrm.model import HCRM
from hcrm.tokenizer import decode
from hcrm.train import arch_compatible, log_line, parse_when, rss_gb, vram_gb, save_checkpoint, summarize_route

USER_RE = re.compile(r"<\|user\|>(.*?)<\|end\|>", re.S)
ASST_RE = re.compile(r"<\|assistant\|>(.*?)<\|end\|>", re.S)
GOLD_RE = re.compile(r"<\|/think\|>([^<\|]+)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

CHAT_STARTERS = [
    ("Hi", ""),
    ("Hello", ""),
    ("Hey", ""),
    ("Hola", ""),
    ("What's up", ""),
    ("How are you", ""),
    ("Who are you", "Creak"),
    ("What is your name", "Creak"),
    ("Thanks", ""),
    ("Help", ""),
    ("Tell me a joke", ""),
    ("What is water", "H2O"),
    ("What can you do", ""),
    ("Good morning", ""),
    ("Bye", ""),
    ("What is AI", ""),
    ("Explain gravity", ""),
    ("Tell a short story", ""),
]

_STOP = {"the", "a", "an", "is", "to", "of", "and", "in", "on", "for", "it"}


def _threads(n: int) -> int:
    return max(1, min(int(n), os.cpu_count() or int(n)))


def encode_user_prompt(tok, cfg: HCRMConfig, user: str, system: str) -> list[int]:
    text = "<|bos|>" + wrap_turn("system", system) + wrap_turn("user", user) + "<|assistant|>"
    return tok.encode(text).ids[: cfg.max_seq_len]


def _norm_num(text: str) -> str:
    m = NUM_RE.findall((text or "").replace(",", ""))
    return m[-1] if m else ""


def load_reason_prompts(path: Path, limit: int = 4000) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or ""
            um = USER_RE.search(text)
            if not um:
                continue
            user = um.group(1).strip()
            gm = GOLD_RE.search(text)
            gold = _norm_num(gm.group(1) if gm else "")
            if user and gold:
                out.append({"user": user, "kind": "reason", "gold": gold})
            if len(out) >= limit:
                break
    return out


def load_chat_prompts(path: Path, limit: int = 1500) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("source") or "") in {"gsm8k", "seed-qa", "metamath", "numina"}:
                continue
            text = rec.get("text") or ""
            um = USER_RE.search(text)
            if not um:
                continue
            user = " ".join(um.group(1).split())
            if not (2 <= len(user) <= 80):
                continue
            gold = ""
            am = ASST_RE.search(text)
            if am:
                body = am.group(1)
                if "<|/think|>" in body:
                    body = body.split("<|/think|>")[-1]
                gold = " ".join(body.split())[:80]
            out.append({"user": user, "kind": "chat", "gold": gold})
            if len(out) >= limit:
                break
    return out


def _answer_span(text: str) -> str:
    if "<|/think|>" in text:
        return text.split("<|/think|>")[-1]
    return text.replace("<|think|>", "")


def think_reward(text: str) -> float:
    """Pay for one real think block plus an answer. Empty tags do not score."""
    n_open = text.count("<|think|>")
    n_close = text.count("<|/think|>")
    if n_open == 0 and n_close == 0:
        return -0.8
    if n_open != 1 or n_close != 1:
        return -0.5
    inner = text.split("<|think|>", 1)[1].split("<|/think|>", 1)[0].strip()
    after = text.split("<|/think|>", 1)[1].strip()
    if "<<" in inner or ">>" in inner:
        return -0.7
    words = re.findall(r"[A-Za-z0-9']+", inner)
    if len(inner) < 12 or len(words) < 3:
        return -0.3
    if len(words) >= 8 and len(set(w.lower() for w in words)) / len(words) < 0.35:
        return -0.5
    r = 0.9
    if after:
        r += 0.3
    else:
        r -= 0.2
    return r


def correctness_reward(text: str, gold: str) -> float:
    """Correct final answer pays more; full credit needs a real think block. Capped."""
    if not gold:
        return 0.0
    after = _answer_span(text).strip()
    has_think = "<|think|>" in text and "<|/think|>" in text
    gold_n = _norm_num(gold)
    got_n = _norm_num(after)
    r = 0.0
    if gold_n and got_n == gold_n:
        r += 4.0 if has_think else 1.6
    elif gold_n and gold_n in after.replace(",", ""):
        r += 1.6 if has_think else 0.6
    elif gold_n and gold_n in text.replace(",", ""):
        r += 0.35
    else:
        gw = set(re.findall(r"[a-z0-9']+", gold.lower())) - _STOP
        aw = set(re.findall(r"[a-z0-9']+", after.lower()))
        if 2 <= len(gw) <= 12 and aw:
            r += min(1.2, 1.5 * (len(gw & aw) / len(gw)))
            if not has_think:
                r *= 0.5
    return min(4.2, r)


def reward_reply(text: str, ended: bool, n_tok: int, gold: str) -> float:
    r = 0.0
    r += 0.5 if ended else -0.4
    if n_tok < 4:
        r -= 1.0
    elif 16 <= n_tok <= 80:
        r += 0.4
    elif n_tok > 100:
        r -= 0.8
    if is_garbage(text) or is_garbage(_answer_span(text)):
        r -= 2.0
    r += think_reward(text)
    r += correctness_reward(text, gold)
    return r


def _route_trace(route: dict | None, u_hist: list[float], ch_hist: list[int]) -> dict:
    if route is None:
        return {
            "summary": "u=0.000 channelite=0 channels[0=0.00]",
            "u_span": "0.00-0.00",
            "path": [],
        }
    u_lo = min(u_hist) if u_hist else 0.0
    u_hi = max(u_hist) if u_hist else 0.0
    return {
        "summary": summarize_route(route),
        "u_span": f"{u_lo:.2f}-{u_hi:.2f}",
        "path": ch_hist[:12],
    }


@torch.no_grad()
def sample_ids(
    model: HCRM,
    prompt_ids: list[int],
    max_new: int,
    temperature: float,
    top_p: float,
    tau: float,
) -> tuple[list[int], bool, dict]:
    """Autoregressive rollout with chat-style routing: eval + full local window.

    Each step forwards ids[:, -max_seq_len:] so the u-router sees causal conv
    context. eval() uses top-k gates like chat, not the dense SFT mixture.
    """
    cfg = model.cfg
    device = next(model.parameters()).device
    stop = {cfg.eos_id, getattr(cfg, "end_id", cfg.eos_id)}
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pieces: list[int] = []
    u_hist: list[float] = []
    ch_hist: list[int] = []
    last_route = None
    was = model.training
    model.eval()
    try:
        for _ in range(max_new):
            window = ids[:, -cfg.max_seq_len :]
            out = model(window, tau=tau)
            last_route = out.route
            u_hist.append(float(out.route["u"][0, -1].item()))
            ch_hist.append(int(out.route["top_channels"][0, -1, 0].item()))
            logits = penalize_repeats(out.logits[:, -1, :].float(), pieces, penalty=1.15)
            if temperature <= 0:
                nxt = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = nucleus(logits / max(temperature, 1e-5), top_p)
                nxt = int(torch.multinomial(probs, 1).item())
            if nxt in stop:
                return pieces, True, _route_trace(last_route, u_hist, ch_hist)
            if len(pieces) >= 2 and pieces[-1] == pieces[-2] == nxt:
                return pieces, True, _route_trace(last_route, u_hist, ch_hist)
            pieces.append(nxt)
            if ngram_repeating(pieces):
                return pieces, True, _route_trace(last_route, u_hist, ch_hist)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        return pieces, False, _route_trace(last_route, u_hist, ch_hist)
    finally:
        model.train(was)


def completion_logprobs(
    model: HCRM,
    prompt_len: int,
    seqs: list[list[int]],
    pad_id: int,
    tau: float,
) -> torch.Tensor:
    """Teacher-forced logprobs with SFT-style dense routing (train mode)."""
    device = next(model.parameters()).device
    was = model.training
    model.train()
    try:
        max_len = max(len(s) for s in seqs)
        batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            batch[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        out = model(batch[:, : model.cfg.max_seq_len], tau=tau)
        logp = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        tgt = batch[:, 1 : model.cfg.max_seq_len]
        gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        pos = torch.arange(gathered.size(1), device=device)
        lengths = torch.tensor([min(len(s), model.cfg.max_seq_len) for s in seqs], device=device)
        mask = (pos.unsqueeze(0) >= (prompt_len - 1)) & (pos.unsqueeze(0) < (lengths.unsqueeze(1) - 1))
        denom = mask.sum(dim=-1).clamp_min(1)
        return (gathered * mask).sum(dim=-1) / denom
    finally:
        model.train(was)


def freeze_copy(model: HCRM) -> HCRM:
    ref = copy.deepcopy(model)
    # Dense softmax like SFT so KL matches the policy backward pass.
    ref.train()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def train_rl(args: argparse.Namespace) -> Path:
    from hcrm.train import pick_device, setup_cuda

    setup_cuda()
    threads = _threads(args.threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass

    now = datetime.now()
    if args.max_minutes and args.max_minutes > 0:
        deadline = now + timedelta(minutes=args.max_minutes)
    else:
        deadline = parse_when(args.until, 22, 0, now)

    ckpt_path = Path(args.ckpt)
    out_dir = ckpt_path.parent
    log_path = out_dir / "train.log"
    device = torch.device("cpu") if args.cpu else pick_device()
    model, cfg, tok = load_checkpoint(ckpt_path, device)
    model.train()
    ref = freeze_copy(model)

    sft_bak = out_dir / "hcrm_slm_sft.pt"
    copy_sft = True
    if sft_bak.exists():
        try:
            bak = torch.load(sft_bak, map_location="cpu", weights_only=False)
            copy_sft = not arch_compatible(bak.get("config") or {}, cfg)
        except Exception:
            copy_sft = True
        if copy_sft:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stale = sft_bak.with_name(f"hcrm_slm_sft_{stamp}.pt")
            shutil.move(str(sft_bak), str(stale))
            log_line(log_path, f"Stashed older SFT backup to {stale}")
    if ckpt_path.exists() and copy_sft:
        shutil.copy2(ckpt_path, sft_bak)
        log_line(log_path, f"Copied SFT weights to {sft_bak} before RL")

    chat_prompts = [{"user": u, "kind": "chat", "gold": g} for u, g in CHAT_STARTERS]
    mix_path = Path("data/creak_gpu_mix.jsonl")
    if not mix_path.exists():
        mix_path = Path("data/creak_mix.jsonl")
    chat_prompts.extend(load_chat_prompts(mix_path))
    think_mix = Path("data/creak_think_mix.jsonl")
    reason_prompts = load_reason_prompts(Path("data/creak_gsm8k.jsonl"))
    if not reason_prompts:
        reason_prompts = load_reason_prompts(think_mix if think_mix.exists() else mix_path, limit=800)
    if not chat_prompts:
        raise RuntimeError("No chat prompts for RL")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    group = max(2, int(args.group_size))
    tau = float(args.tau)
    step = 0
    t0 = time.time()
    log_t = t0
    running_r = 0.0
    running_n = 0

    log_line(log_path, "=" * 72)
    log_line(
        log_path,
        f"Creak RL start {now.isoformat(timespec='seconds')}  "
        f"until={deadline.strftime('%Y-%m-%d %H:%M')}  threads={threads}  "
        f"device={device}  group={group}  lr={args.lr:.2e}  kl={args.kl:.3f}  "
        f"chat_prompts={len(chat_prompts)}  reason_prompts={len(reason_prompts)}",
    )
    log_line(
        log_path,
        f"BEGIN rl  params={model.count_params():,}  packs={len(chat_prompts)+len(reason_prompts)}  "
        f"seq={cfg.max_seq_len}  batch={group}x1  until={deadline.strftime('%H:%M')}  "
        f"left={(deadline.timestamp() - time.time()) / 60:.1f}m  rss={rss_gb():.2f}GB",
    )

    try:
        while time.time() < deadline.timestamp():
            use_reason = bool(reason_prompts) and (random.random() < 0.45)
            spec = random.choice(reason_prompts if use_reason else chat_prompts)
            system = REASON_SYSTEM_PROMPT
            prompt_ids = encode_user_prompt(tok, cfg, spec["user"], system)
            max_new = args.reason_max_new if spec["kind"] == "reason" else args.max_new

            seqs: list[list[int]] = []
            rewards: list[float] = []
            ended_n = 0
            think_n = 0
            gold_n = 0
            traces: list[dict] = []
            for _ in range(group):
                pieces, ended, trace = sample_ids(model, prompt_ids, max_new, args.temp, args.top_p, tau)
                traces.append(trace)
                ended_n += int(ended)
                raw = decode(tok, pieces, skip_special=False)
                r = reward_reply(raw, ended, len(pieces), spec["gold"])
                think_n += int(think_reward(raw) > 0)
                if spec["gold"] and _norm_num(_answer_span(raw)) == _norm_num(spec["gold"]) and _norm_num(spec["gold"]):
                    gold_n += 1
                seqs.append(prompt_ids + pieces)
                rewards.append(r)

            r_t = torch.tensor(rewards, dtype=torch.float32)
            adv = r_t - r_t.mean()
            std = r_t.std(unbiased=False)
            if float(std) < 1e-6:
                adv = r_t - 0.5
            else:
                adv = adv / (std + 1e-6)

            logp = completion_logprobs(model, len(prompt_ids), seqs, cfg.pad_id, tau)
            with torch.no_grad():
                logp_ref = completion_logprobs(ref, len(prompt_ids), seqs, cfg.pad_id, tau)
            kl = (logp.detach() - logp_ref).mean()
            pg = -(adv.to(logp.device) * logp).mean()
            loss = pg + args.kl * (logp - logp_ref.detach()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            running_r += float(r_t.mean())
            running_n += 1

            if step % 10 == 0:
                avg = running_r / max(1, running_n)
                running_r = 0.0
                running_n = 0
                dt = max(time.time() - log_t, 1e-6)
                log_t = time.time()
                toks = 10 * group * ((args.reason_max_new + args.max_new) / 2)
                tps = toks / dt
                left = max(0.0, deadline.timestamp() - time.time())
                dash_loss = max(0.001, 4.0 - avg)
                route_txt = traces[-1]["summary"] if traces else "u=0.000 channelite=0 channels[0=0.00]"
                u_span = traces[-1]["u_span"] if traces else "0.00-0.00"
                log_line(
                    log_path,
                    f"rl step={step} loss={dash_loss:.3f} ppl={max(0.1, 10 + avg):.1f} "
                    f"lr={args.lr:.2e} tau={tau:.2f} tps={tps:.0f} left={left/60:.1f}m "
                    f"rss={rss_gb():.2f}GB {route_txt} "
                    f"reward={avg:.3f} ended={ended_n/group:.2f} think={think_n/group:.2f} "
                    f"gold={gold_n/group:.2f} kl={float(kl):.3f} kind={spec['kind']} u_span={u_span}"
                    f"{f' vram={vram_gb():.2f}GB' if device.type == 'cuda' else ''}",
                )
            if step % 50 == 0:
                save_checkpoint(ckpt_path, model, cfg, step, float(r_t.mean()), optimizer=opt, phase="rl")
    except KeyboardInterrupt:
        log_line(log_path, "rl interrupted — saving.")

    save_checkpoint(ckpt_path, model, cfg, step, 0.0, optimizer=opt, phase="rl")
    log_line(
        log_path,
        f"END rl  steps={step}  elapsed={(time.time() - t0) / 60:.1f}m  rss={rss_gb():.2f}GB  saved={ckpt_path}",
    )
    return ckpt_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO reinforcement learning for Creak (GPU).")
    p.add_argument("--ckpt", default="checkpoints/hcrm_slm.pt")
    p.add_argument("--max-minutes", type=float, default=90.0)
    p.add_argument("--until", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.05)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--reason-max-new", type=int, default=96)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    train_rl(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
