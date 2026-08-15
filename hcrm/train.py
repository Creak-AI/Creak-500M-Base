from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
import traceback

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from hcrm.config import HCRMConfig, TrainConfig
from hcrm.data import (
    PackedLM,
    creak_jsonl_ok,
    iter_jsonl_texts,
    pack_token_ids,
    prepare_gpu_mix,
    prepare_gsm8k,
    prepare_think_mix,
    prepare_reason_think_mix,
    mix_reason_with_chat,
    creak_gpu_mix_ok,
    creak_think_mix_ok,
    creak_gpu_think_mix_ok,
)
from hcrm.model import HCRM
from hcrm.routing import load_balance_loss, locality_loss
from hcrm.tokenizer import apply_config_special_ids, encode, load_smollm_creak_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * float(step + 1) / max(1, warmup)
    if step >= total:
        return base * 0.05
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def anneal_tau(progress: float) -> float:
    """1.2 → 0.5 quickly, then 0.5 → 0.3, then hold."""
    p = min(1.0, max(0.0, progress))
    if p < 0.30:
        return 1.2 + (0.5 - 1.2) * (p / 0.30)
    if p < 0.60:
        return 0.5 + (0.3 - 0.5) * ((p - 0.30) / 0.30)
    return 0.3


def summarize_route(route: dict) -> str:
    u = float(route["u"][:, -8:].detach().mean().item())
    cite = route["channelite_mass"][:, -8:].detach().mean(dim=(0, 1))
    ch = route["channel_probs"][:, -8:].detach().mean(dim=(0, 1))
    top_cite = int(torch.argmax(cite).item())
    top_ch = torch.topk(ch.detach(), k=min(3, ch.numel()))
    chs = ", ".join(f"{int(i)}={float(v):.2f}" for v, i in zip(top_ch.values, top_ch.indices))
    return f"u={u:.3f} channelite={top_cite} channels[{chs}]"


def proc_mem_gb() -> tuple[float, float]:
    """Return (current_commit_gb, peak_commit_gb)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        cur = info.rss / (1024**3)
        peak = getattr(info, "peak_wset", info.rss) / (1024**3)
        return cur, peak
    except Exception:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            cur = float(max(counters.WorkingSetSize, counters.PrivateUsage, counters.PagefileUsage)) / (1024**3)
            peak = float(max(counters.PeakWorkingSetSize, counters.PeakPagefileUsage, cur)) / (1024**3)
            return cur, peak
        return 0.0, 0.0
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            text = f.read()
        def kb(name: str) -> float:
            for line in text.splitlines():
                if line.startswith(name):
                    return float(line.split()[1]) / (1024**2)
            return 0.0
        cur = kb("VmRSS:")
        peak = max(kb("VmHWM:"), kb("VmPeak:"))
        return cur, peak
    except Exception:
        return 0.0, 0.0


def rss_gb() -> float:
    cur, peak = proc_mem_gb()
    return max(cur, peak)


def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated()) / (1024**3)
    except Exception:
        return 0.0


def mem_tag() -> str:
    tag = f"rss={rss_gb():.2f}GB"
    if torch.cuda.is_available():
        tag += f" vram={vram_gb():.2f}GB"
    return tag


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_cuda() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def optimizer_to_device(opt: AdamW, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def log_line(log_path: Path, msg: str) -> None:
    print(msg, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def parse_when(value: str | None, hour: int, minute: int, now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    if not value:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%H:%M"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%H:%M":
                parsed = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
                if parsed <= now:
                    parsed += timedelta(days=1)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"Could not parse time {value!r}; use HH:MM or YYYY-MM-DDTHH:MM")


def arch_compatible(saved: dict, cfg: HCRMConfig) -> bool:
    keys = (
        "d_model",
        "n_layers",
        "n_channels",
        "n_channelites",
        "d_ff",
        "vocab_size",
        "conv_kernel",
        "use_rope",
        "use_gru",
        "max_seq_len",
        "top_k",
    )
    for key in keys:
        if key not in saved:
            return False
        if saved[key] != getattr(cfg, key):
            return False
    return True


def backup_mismatched_ckpt(path: Path, cfg: HCRMConfig, log_path: Path) -> None:
    if not path.exists():
        return
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log_line(log_path, f"Could not read {path} for backup: {exc}")
        return
    saved = blob.get("config") or {}
    if arch_compatible(saved, cfg):
        return
    tag = f"{saved.get('d_model', '?')}d_{saved.get('n_layers', '?')}L"
    bak = path.with_name(f"hcrm_slm_{tag}.pt")
    if not bak.exists():
        shutil.copy2(path, bak)
        log_line(log_path, f"Backed up previous checkpoint to {bak}")


def stash_checkpoint(path: Path, log_path: Path) -> None:
    """Move the latest checkpoint aside so a from-scratch run cannot resume it."""
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_name(f"hcrm_slm_scratchbak_{stamp}.pt")
    shutil.move(str(path), str(bak))
    log_line(log_path, f"Stashed previous checkpoint to {bak} (training from scratch)")


def save_checkpoint(
    path: Path,
    model: HCRM,
    cfg: HCRMConfig,
    opt_step: int,
    loss: float,
    optimizer: AdamW | None = None,
    phase: str = "chat",
    epoch: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tok_json = None
    hf_dir = path.with_name("hf_tokenizer")
    if not hf_dir.exists():
        for candidate in (
            path.with_name("tokenizer.json"),
            Path("data/smollm2_creak_tokenizer") / "tokenizer.json",
            Path("data/creak_tokenizer.json"),
            Path("data/tokenizer.json"),
        ):
            if candidate.exists():
                tok_json = candidate.read_text(encoding="utf-8")
                break
    blob: dict = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "step": opt_step,
        "loss": loss,
        "phase": phase,
        "epoch": int(epoch),
        "tokenizer_json": tok_json,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "torch_rng": torch.random.get_rng_state(),
        "py_rng": random.getstate(),
    }
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    cfg.save(path.with_name("config.json"))
    side = {
        "step": opt_step,
        "phase": phase,
        "epoch": int(epoch),
        "loss": loss,
        "params": model.count_params(),
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_channelites": cfg.n_channelites,
        "n_channels": cfg.n_channels,
        "d_ff": cfg.d_ff,
        "max_seq_len": cfg.max_seq_len,
        "saved_at": blob["saved_at"],
        "path": str(path),
    }
    path.with_name("resume.json").write_text(json.dumps(side, indent=2), encoding="utf-8")


def prune_step_ckpts(out_dir: Path, keep: int) -> None:
    files = sorted(out_dir.glob("hcrm_slm_step*.pt"), key=lambda p: p.stat().st_mtime)
    for stale in files[:-max(0, keep)]:
        try:
            stale.unlink()
        except OSError:
            pass


def creak_corpus_ok(path: Path) -> bool:
    return creak_jsonl_ok(path)


def probe_one(cfg: HCRMConfig, batch_size: int) -> tuple[float, int] | None:
    gc.collect()
    model = HCRM(cfg)
    opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    params = model.count_params()
    ids = torch.randint(0, max(8, cfg.vocab_size), (batch_size, cfg.max_seq_len))
    labels = ids.clone()
    labels[:, :16] = -100
    try:
        out = model(ids, labels=labels, tau=1.0)
        out.loss.backward()
        opt.step()
        rss = rss_gb()
        return rss, params
    except (MemoryError, RuntimeError) as exc:
        print(f"RAM probe failed: {exc}", flush=True)
        return None
    finally:
        del model, opt, ids, labels
        gc.collect()


def fit_to_ram(cfg: HCRMConfig, tcfg: TrainConfig, log_path: Path) -> tuple[HCRMConfig, TrainConfig]:
    """Grow the training footprint until RSS is in [target, max] GB."""
    plan = [
        dict(max_seq_len=2048, batch_size=2, channel_chunk=1024, channel_group=8, d_ff=cfg.d_ff, n_layers=cfg.n_layers),
        dict(max_seq_len=3072, batch_size=2, channel_chunk=1024, channel_group=8, d_ff=cfg.d_ff, n_layers=cfg.n_layers),
        dict(max_seq_len=3584, batch_size=2, channel_chunk=1024, channel_group=8, d_ff=cfg.d_ff, n_layers=cfg.n_layers),
        dict(max_seq_len=4096, batch_size=2, channel_chunk=1024, channel_group=8, d_ff=cfg.d_ff, n_layers=cfg.n_layers),
        dict(max_seq_len=4096, batch_size=2, channel_chunk=2048, channel_group=4, d_ff=cfg.d_ff, n_layers=cfg.n_layers),
        dict(max_seq_len=4096, batch_size=2, channel_chunk=0, channel_group=0, d_ff=cfg.d_ff, n_layers=cfg.n_layers),
    ]
    best: tuple[HCRMConfig, TrainConfig, float] | None = None
    for i, step in enumerate(plan, start=1):
        trial = replace(cfg, **{k: v for k, v in step.items() if k != "batch_size"})
        trial_t = replace(tcfg, batch_size=step["batch_size"])
        log_line(
            log_path,
            f"RAM probe {i}/{len(plan)} seq={trial.max_seq_len} batch={trial_t.batch_size} "
            f"chunk={trial.channel_chunk} group={trial.channel_group} d_ff={trial.d_ff} layers={trial.n_layers}",
        )
        result = probe_one(trial, trial_t.batch_size)
        if result is None:
            log_line(log_path, "  probe OOM/error — keeping previous size")
            break
        rss, params = result
        log_line(log_path, f"  rss={rss:.2f}GB  unique_params={params:,}")
        if rss <= 0.05:
            log_line(log_path, "  RSS unreadable — locking seq=4096 / full channels (expected ~12GB)")
            locked = replace(cfg, max_seq_len=4096, channel_chunk=0, channel_group=0)
            return locked, trial_t
        if rss > tcfg.max_ram_gb:
            log_line(log_path, f"  over max_ram_gb={tcfg.max_ram_gb:.1f} — stopping growth")
            break
        cfg, tcfg = trial, trial_t
        best = (cfg, tcfg, rss)
        if rss >= tcfg.target_ram_gb:
            log_line(log_path, f"  hit target RAM {tcfg.target_ram_gb:.1f}GB")
            break
    if best is None:
        raise RuntimeError("RAM probe failed on the smallest plan — free memory and retry")
    cfg, tcfg, rss = best
    log_line(
        log_path,
        f"Using seq={cfg.max_seq_len} batch={tcfg.batch_size}x{tcfg.grad_accum} "
        f"d={cfg.d_model} L={cfg.n_layers} d_ff={cfg.d_ff} rss~{rss:.2f}GB",
    )
    return cfg, tcfg


def _pack_cache_path(jsonl_path: Path, cfg: HCRMConfig) -> Path:
    return jsonl_path.with_name(
        f"{jsonl_path.stem}.packs_s{cfg.max_seq_len}_v{cfg.vocab_size}.pt"
    )


def load_or_build_packs(jsonl_path: Path, tok, cfg: HCRMConfig, log_path: Path | None = None) -> list:
    cache = _pack_cache_path(jsonl_path, cfg)
    stamp = {
        "mtime": jsonl_path.stat().st_mtime,
        "size": jsonl_path.stat().st_size,
        "seq": cfg.max_seq_len,
        "vocab": cfg.vocab_size,
        "eos": cfg.eos_id,
    }
    if cache.exists():
        try:
            blob = torch.load(cache, map_location="cpu", weights_only=False)
            if blob.get("stamp") == stamp and blob.get("packs"):
                if log_path is not None:
                    log_line(log_path, f"Reusing packed cache {cache} ({len(blob['packs'])} packs)")
                return blob["packs"]
        except Exception as exc:
            if log_path is not None:
                log_line(log_path, f"Pack cache unreadable ({exc}); rebuilding")

    def encoded_chats():
        for text in iter_jsonl_texts(jsonl_path):
            yield encode(tok, text, cfg.max_seq_len, keep_tail=False)

    packs = pack_token_ids(encoded_chats(), cfg.max_seq_len, cfg.eos_id)
    try:
        torch.save({"stamp": stamp, "packs": packs}, cache)
        if log_path is not None:
            log_line(log_path, f"Wrote packed cache {cache}")
    except Exception as exc:
        if log_path is not None:
            log_line(log_path, f"Could not write pack cache ({exc})")
    return packs


def make_loader(
    jsonl_path: Path,
    tok,
    cfg: HCRMConfig,
    tcfg: TrainConfig,
    log_path: Path | None = None,
) -> DataLoader:
    packs = load_or_build_packs(jsonl_path, tok, cfg, log_path=log_path)
    gc.collect()
    ds = PackedLM(
        packs,
        pad_id=cfg.pad_id,
        max_len=cfg.max_seq_len,
        bos_id=cfg.bos_id,
        user_id=tok.token_to_id("<|user|>"),
        asst_id=tok.token_to_id("<|assistant|>"),
        sys_id=tok.token_to_id("<|system|>"),
        end_id=tok.token_to_id("<|end|>") or cfg.end_id,
    )
    if len(ds) < 4:
        raise RuntimeError(f"Not enough packed sequences in {jsonl_path}")
    kwargs: dict = dict(
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    if tcfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def train_loop(
    model: HCRM,
    opt: AdamW,
    loader: DataLoader,
    cfg: HCRMConfig,
    tcfg: TrainConfig,
    deadline: float,
    log_path: Path,
    ckpt_path: Path,
    phase: str,
    opt_step: int = 0,
    base_lr: float | None = None,
    start_epoch: int = 0,
) -> int:
    from contextlib import nullcontext

    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    amp = str(getattr(tcfg, "amp", "bf16") or "fp32").lower()
    amp_dtype = None
    if use_cuda and amp != "fp32":
        if amp == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    scaler = None
    if amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    )
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    steps_per_epoch = max(1, len(loader) // tcfg.grad_accum)
    total_opt_steps = max(steps_per_epoch * tcfg.epochs, opt_step + 1)
    warmup = max(1, int(total_opt_steps * tcfg.warmup_ratio))
    lr0 = tcfg.lr if base_lr is None else base_lr
    model.train()
    global_step = 0
    running = 0.0
    running_n = 0
    best = float("inf")
    t0 = time.time()
    log_t = t0
    stop = False
    last_loss = 0.0
    epoch = start_epoch
    phase_t0 = time.time()
    phase_budget = max(1.0, deadline - phase_t0)
    amp_name = "fp32" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")

    log_line(
        log_path,
        f"BEGIN {phase}  params={model.count_params():,}  packs={len(loader.dataset)}  "
        f"seq={cfg.max_seq_len}  batch={tcfg.batch_size}x{tcfg.grad_accum}  "
        f"d={cfg.d_model} L={cfg.n_layers} cites={cfg.n_channelites} C={cfg.n_channels} "
        f"top_k={cfg.top_k}  device={device} amp={amp_name}  "
        f"resume_step={opt_step} epoch={start_epoch}  "
        f"until={datetime.fromtimestamp(deadline).strftime('%H:%M')}  "
        f"left={(deadline - time.time()) / 60:.1f}m  {mem_tag()}",
    )

    try:
        for epoch in range(start_epoch, tcfg.epochs):
            if stop:
                break
            # Disable the live bar: selecting/copying it on Windows blocks WriteConsole
            # and can freeze the whole train process. Progress goes to train.log.
            bar = tqdm(
                loader,
                desc=f"{phase} {epoch+1}/{tcfg.epochs}",
                leave=False,
                mininterval=5.0,
                disable=True,
            )
            opt.zero_grad(set_to_none=True)
            for batch in bar:
                if time.time() >= deadline:
                    log_line(log_path, f"{phase} time budget reached — saving.")
                    stop = True
                    break
                progress = min(1.0, (time.time() - phase_t0) / phase_budget)
                tau = anneal_tau(progress)
                ids = batch["input_ids"].to(device, non_blocking=use_cuda)
                labels = batch["labels"].to(device, non_blocking=use_cuda)
                with autocast:
                    out = model(ids, labels=labels, tau=tau)
                    bal = load_balance_loss(out.route["channel_probs"], cfg.n_channels)
                    loc = locality_loss(out.route["u"])
                    loss = out.loss + tcfg.aux_balance * bal + tcfg.aux_locality * loc
                if scaler is not None:
                    scaler.scale(loss / tcfg.grad_accum).backward()
                else:
                    (loss / tcfg.grad_accum).backward()
                last_loss = float(out.loss.detach().float().item())
                running += last_loss
                running_n += 1
                global_step += 1

                if global_step % tcfg.grad_accum == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                    lr = cosine_lr(opt_step, warmup, total_opt_steps, lr0)
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    opt_step += 1
                    if opt_step % tcfg.log_every == 0:
                        avg = running / max(1, running_n)
                        running = 0.0
                        running_n = 0
                        dt = max(time.time() - log_t, 1e-6)
                        log_t = time.time()
                        toks = tcfg.log_every * tcfg.grad_accum * tcfg.batch_size * cfg.max_seq_len
                        tps = toks / dt
                        left = max(0.0, deadline - time.time())
                        extra = f" vram={vram_gb():.2f}GB" if use_cuda else ""
                        msg = (
                            f"{phase} step={opt_step} loss={avg:.3f} ppl={math.exp(min(avg, 20)):.1f} "
                            f"lr={lr:.2e} tau={tau:.2f} tps={tps:.0f} left={left/60:.1f}m "
                            f"rss={rss_gb():.2f}GB {summarize_route(out.route)}{extra}"
                        )
                        bar.set_postfix_str(
                            f"loss={avg:.3f} ppl={math.exp(min(avg, 20)):.1f} tau={tau:.2f} left={left/60:.1f}m"
                        )
                        log_line(log_path, msg)
                    if opt_step % tcfg.save_every == 0:
                        save_checkpoint(
                            ckpt_path, model, cfg, opt_step, last_loss,
                            optimizer=opt, phase=phase, epoch=epoch,
                        )
                        step_path = ckpt_path.with_name(f"hcrm_slm_step{opt_step}.pt")
                        save_checkpoint(
                            step_path, model, cfg, opt_step, last_loss,
                            optimizer=None, phase=phase, epoch=epoch,
                        )
                        prune_step_ckpts(ckpt_path.parent, tcfg.keep_step_ckpts)
                        if last_loss < best:
                            best = last_loss
            if not stop:
                save_checkpoint(
                    ckpt_path, model, cfg, opt_step, last_loss,
                    optimizer=opt, phase=phase, epoch=epoch,
                )
    except KeyboardInterrupt:
        log_line(log_path, f"{phase} interrupted — saving.")
    except Exception:
        log_line(log_path, f"{phase} crashed at step={opt_step}:\n{traceback.format_exc()}")
        raise

    save_checkpoint(
        ckpt_path, model, cfg, opt_step, last_loss,
        optimizer=opt, phase=phase, epoch=epoch,
    )
    log_line(
        log_path,
        f"END {phase}  steps={opt_step}  elapsed={(time.time() - t0) / 60:.1f}m  {mem_tag()}  saved={ckpt_path}",
    )
    return opt_step


def prepare_chat_data(cfg: HCRMConfig, tcfg: TrainConfig, log_path: Path):
    data_dir = tcfg.data_dir
    out_dir = tcfg.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = data_dir / "creak_think_mix.jsonl"
    raw_mix = data_dir / "creak_gpu_mix.jsonl"
    smoltalk_cache = data_dir / "creak_smoltalk.jsonl"
    gsm_path = data_dir / "creak_gsm8k.jsonl"
    tok_dir = data_dir / "smollm2_creak_tokenizer"

    if not creak_gpu_mix_ok(raw_mix):
        log_line(
            log_path,
            "Building large GPU mix (SmolTalk slices + UltraChat + Orca-Math + GSM8K + CoT)",
        )
        prepare_gpu_mix(raw_mix, smoltalk_cache, gsm_path)
    else:
        log_line(log_path, f"Reusing GPU mix {raw_mix}")

    if not creak_gpu_think_mix_ok(subset_path):
        log_line(log_path, "Building interleaved-think mix (reason in every reply, math/CoT upsampled)")
        prepare_think_mix(raw_mix, subset_path, reason_copies=2)
    else:
        log_line(log_path, f"Reusing interleaved-think mix {subset_path}")

    tok_id = getattr(cfg, "tokenizer_id", None) or "HuggingFaceTB/SmolLM2-135M-Instruct"
    log_line(log_path, f"Loading SmolLM2 tokenizer + Creak specials from {tok_id}")
    tok = load_smollm_creak_tokenizer(tok_dir, tok_id)
    cfg = apply_config_special_ids(cfg, tok)
    cfg.save(out_dir / "config.json")
    dest = out_dir / "hf_tokenizer"
    if tok_dir.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(tok_dir, dest)
    return subset_path, tok, cfg


def alloc_budget(
    hard_stop: datetime,
    skip_reason: bool,
    reason_minutes: float,
    log_path: Path,
) -> tuple[datetime, datetime]:
    """Spend leftover wall clock after data prep: most of it on chat, a tail on reason."""
    now = datetime.now()
    left_m = max(0.0, (hard_stop - now).total_seconds() / 60.0)
    log_line(
        log_path,
        f"GPU budget remaining {left_m:.1f}m  hard_stop={hard_stop.strftime('%Y-%m-%d %H:%M')}",
    )
    if left_m < 8:
        log_line(log_path, "Almost no GPU time left after data prep — saving whatever we have")
        return now, now
    if skip_reason or left_m < 50:
        log_line(log_path, "Using remaining time for the interleaved chat mix (reason tail skipped)")
        return hard_stop, hard_stop
    tail = min(max(20.0, float(reason_minutes)), left_m * 0.2, left_m - 15.0)
    tail = max(0.0, tail)
    chat_until = now + timedelta(minutes=left_m - tail)
    log_line(
        log_path,
        f"Split remaining: chat until {chat_until.strftime('%H:%M')}  "
        f"reason until {hard_stop.strftime('%H:%M')} ({tail:.0f}m tail)",
    )
    return chat_until, hard_stop


def train(
    cfg: HCRMConfig,
    tcfg: TrainConfig,
    chat_until: datetime,
    reason_until: datetime,
    skip_probe: bool = False,
    force_phase: str | None = None,
    skip_reason: bool = False,
) -> Path:
    set_seed(tcfg.seed)
    setup_cuda()
    threads = max(1, min(int(tcfg.num_threads), os.cpu_count() or int(tcfg.num_threads)))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass

    out_dir = tcfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    ckpt_path = out_dir / "hcrm_slm.pt"
    device = pick_device()
    hard_stop = None
    if tcfg.max_minutes and tcfg.max_minutes > 0:
        hard_stop = datetime.now() + timedelta(minutes=float(tcfg.max_minutes))
        chat_until = hard_stop
        reason_until = hard_stop
    log_line(log_path, "=" * 72)
    log_line(
        log_path,
        f"Creak train start {datetime.now().isoformat(timespec='seconds')}  "
        f"chat_until={chat_until.strftime('%Y-%m-%d %H:%M')}  "
        f"reason_until={reason_until.strftime('%Y-%m-%d %H:%M')}  "
        f"threads={threads}  device={device} amp={tcfg.amp}  "
        f"resume={tcfg.resume}  skip_reason={skip_reason}"
        + (f"  gpu_budget={tcfg.max_minutes:.0f}m" if hard_stop else ""),
    )
    if device.type != "cuda":
        log_line(log_path, "WARNING: CUDA not available — 500M training on CPU will be extremely slow")

    if not tcfg.resume:
        stash_checkpoint(ckpt_path, log_path)
    else:
        backup_mismatched_ckpt(ckpt_path, cfg, log_path)
    subset_path, tok, cfg = prepare_chat_data(cfg, tcfg, log_path)

    reason_path = tcfg.data_dir / "creak_gsm8k.jsonl"

    resume_blob = None
    if tcfg.resume and ckpt_path.exists():
        resume_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = resume_blob.get("config") or {}
        looks_creak = arch_compatible(saved, cfg)
        if looks_creak:
            cfg = HCRMConfig(**{k: v for k, v in saved.items() if k in HCRMConfig.__dataclass_fields__})
            cfg = apply_config_special_ids(cfg, tok)
            log_line(
                log_path,
                f"Resuming {resume_blob.get('phase')} from step {resume_blob.get('step')} "
                f"epoch {resume_blob.get('epoch', 0)}",
            )
        else:
            log_line(log_path, "Checkpoint architecture differs from this run — training from scratch")
            resume_blob = None

    if resume_blob is None and not skip_probe:
        cfg, tcfg = fit_to_ram(cfg, tcfg, log_path)
    else:
        log_line(
            log_path,
            f"Probe off — locking L40S 500M layout seq={cfg.max_seq_len} d={cfg.d_model} "
            f"C={cfg.n_channels} ff={cfg.d_ff} group={cfg.channel_group}",
        )

    loader = make_loader(subset_path, tok, cfg, tcfg, log_path=log_path)
    if hard_stop is not None:
        chat_until, reason_until = alloc_budget(
            hard_stop, skip_reason, getattr(tcfg, "reason_minutes", 45.0), log_path
        )
        if datetime.now() >= hard_stop:
            log_line(log_path, "GPU budget exhausted during data prep — no training steps this session")
            return ckpt_path
    model = HCRM(cfg)
    model.to(device)
    opt = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay, betas=(0.9, 0.95))
    opt_step = 0
    start_epoch = 0
    phase = "chat"
    if resume_blob is not None:
        missing, unexpected = model.load_state_dict(resume_blob["model"], strict=False)
        if missing or unexpected:
            log_line(log_path, f"resume load missing={missing[:6]} unexpected={unexpected[:6]}")
        if resume_blob.get("optimizer"):
            try:
                opt.load_state_dict(resume_blob["optimizer"])
                optimizer_to_device(opt, device)
            except Exception as exc:
                log_line(log_path, f"Could not load optimizer ({exc}); continuing with fresh Adam")
        opt_step = int(resume_blob.get("step") or 0)
        start_epoch = int(resume_blob.get("epoch") or 0)
        phase = str(resume_blob.get("phase") or "chat")
        if resume_blob.get("torch_rng") is not None:
            try:
                torch.random.set_rng_state(resume_blob["torch_rng"])
            except Exception as exc:
                log_line(log_path, f"Could not restore torch RNG ({exc})")
        if resume_blob.get("py_rng") is not None:
            try:
                random.setstate(resume_blob["py_rng"])
            except Exception as exc:
                log_line(log_path, f"Could not restore python RNG ({exc})")
    if force_phase in {"chat", "reason"}:
        phase = force_phase
        log_line(log_path, f"Phase forced to {phase}")
    if resume_blob is None:
        save_checkpoint(ckpt_path, model, cfg, 0, 0.0, optimizer=opt, phase="chat", epoch=0)
        log_line(log_path, f"Wrote initial 500M checkpoint {ckpt_path} ({model.count_params():,} params)")

    now = datetime.now()
    if phase == "chat" and now < chat_until:
        opt_step = train_loop(
            model,
            opt,
            loader,
            cfg,
            tcfg,
            chat_until.timestamp(),
            log_path,
            ckpt_path,
            "chat",
            opt_step=opt_step,
            base_lr=tcfg.lr,
            start_epoch=start_epoch,
        )
        phase = "reason"

    now = datetime.now()
    if (not skip_reason) and now < reason_until and (reason_until - now).total_seconds() >= 12 * 60:
        if not reason_path.exists() or reason_path.stat().st_size < 100:
            log_line(log_path, "Preparing GSM8K reasoning corpus")
            prepare_gsm8k(reason_path)
        mix_path = tcfg.data_dir / "creak_reason_think_mix.jsonl"
        log_line(
            log_path,
            "Building reason tail: interleaved GSM8K/MetaMath/Numina plus a small chat slice",
        )
        prepare_reason_think_mix(subset_path, mix_path, chat_keep=8000)
        if datetime.now() >= reason_until:
            log_line(log_path, "Reason mix used the remaining GPU budget — skipping reason train loop")
        else:
            reason_loader = make_loader(mix_path, tok, cfg, tcfg, log_path=log_path)
            reason_lr = max(5e-5, float(tcfg.lr) * 0.4)
            reason_opt = AdamW(model.parameters(), lr=reason_lr, weight_decay=tcfg.weight_decay, betas=(0.9, 0.95))
            train_loop(
                model,
                reason_opt,
                reason_loader,
                cfg,
                tcfg,
                reason_until.timestamp(),
                log_path,
                ckpt_path,
                "reason",
                opt_step=opt_step,
                base_lr=reason_lr,
            )
    else:
        log_line(log_path, "Separate GSM8K tail skipped — interleaved reasoning is in the main mix")

    log_line(log_path, f"Done. Latest checkpoint {ckpt_path}")
    return ckpt_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Creak HCRM ~500M on an L40S GPU.")
    p.add_argument("--max-minutes", type=float, default=300.0, help="Total GPU wall clock from start, including data prep (default 5 hours)")
    p.add_argument("--until", type=str, default=None, help="Override: stop chat at local HH:MM instead of --max-minutes")
    p.add_argument("--reason-until", type=str, default=None, help="Override: stop reasoning at local HH:MM")
    p.add_argument("--reason-minutes", type=float, default=45.0, help="Minutes reserved at the end for the reason tail (inside the GPU budget)")
    p.add_argument("--max-samples", type=int, default=500000)
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--channels", type=int, default=512)
    p.add_argument("--channelites", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=896)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--channel-group", type=int, default=32)
    p.add_argument("--channel-chunk", type=int, default=512)
    p.add_argument("--ce-chunk", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--target-ram-gb", type=float, default=12.0)
    p.add_argument("--max-ram-gb", type=float, default=14.5)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--fresh-data", action="store_true")
    p.add_argument("--probe-ram", action="store_true", help="CPU RAM growth probe (do not use on L40S)")
    p.add_argument("--skip-probe", action="store_true", help="Deprecated: probe is already off")
    p.add_argument("--phase", choices=("chat", "reason"), default=None, help="Override checkpoint phase")
    p.add_argument("--skip-reason", action="store_true", help="Do not run a separate GSM8K tail stage")
    p.add_argument("--dry-run", action="store_true", help="one random batch, no dataset download")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    now = datetime.now()
    clock_mode = bool(args.until or args.reason_until)
    if clock_mode:
        chat_until = parse_when(args.until, 6, 30, now)
        reason_until = parse_when(args.reason_until, 7, 0, now)
        if reason_until <= chat_until:
            reason_until = chat_until + timedelta(minutes=max(15.0, args.reason_minutes))
        budget_minutes = 0.0
    else:
        budget_minutes = float(args.max_minutes or 300.0)
        chat_until = now + timedelta(minutes=budget_minutes)
        reason_until = chat_until

    cfg = HCRMConfig(
        d_model=args.d_model,
        n_layers=args.layers,
        n_channels=args.channels,
        n_channelites=args.channelites,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        top_k=args.top_k,
        channel_group=args.channel_group,
        channel_chunk=args.channel_chunk,
        ce_chunk=args.ce_chunk,
        grad_checkpoint=not args.no_grad_checkpoint,
        dataset_id="creak-gpu-500m",
    )
    tcfg = TrainConfig(
        max_minutes=budget_minutes,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=max(1, args.grad_accum),
        lr=args.lr,
        save_every=max(10, args.save_every),
        seed=args.seed,
        num_threads=max(1, args.threads),
        num_workers=max(0, args.workers),
        resume=not args.no_resume,
        target_ram_gb=args.target_ram_gb,
        max_ram_gb=args.max_ram_gb,
        amp=args.amp,
        reason_minutes=max(0.0, args.reason_minutes),
    )
    if args.dry_run:
        dry_run(cfg)
        return
    if args.fresh_data:
        for cached in (
            tcfg.data_dir / "creak_smoltalk.jsonl",
            tcfg.data_dir / "creak_mix.jsonl",
            tcfg.data_dir / "creak_gpu_mix.jsonl",
            tcfg.data_dir / "creak_think_mix.jsonl",
            tcfg.data_dir / "creak_reason_think_mix.jsonl",
            tcfg.data_dir / "creak_tokenizer.json",
            tcfg.data_dir / "creak_gsm8k.jsonl",
        ):
            if cached.exists():
                cached.unlink()
        tok_dir = tcfg.data_dir / "smollm2_creak_tokenizer"
        if tok_dir.exists():
            shutil.rmtree(tok_dir)
        for pack in tcfg.data_dir.glob("*.packs_s*.pt"):
            pack.unlink()
    train(
        cfg,
        tcfg,
        chat_until,
        reason_until,
        skip_probe=not args.probe_ram,
        force_phase=args.phase,
        skip_reason=args.skip_reason,
    )


def dry_run(cfg: HCRMConfig) -> None:
    setup_cuda()
    device = pick_device()
    cfg = HCRMConfig(**{**cfg.to_dict(), "max_seq_len": min(cfg.max_seq_len, 256)})
    model = HCRM(cfg).to(device)
    bs = 2
    ids = torch.randint(0, cfg.vocab_size, (bs, cfg.max_seq_len), device=device)
    labels = ids.clone()
    t0 = time.time()
    out = model(ids, labels=labels)
    out.loss.backward()
    dt = time.time() - t0
    print(
        f"dry-run ok  unique_params={model.count_params():,}  "
        f"loss={float(out.loss.item()):.4f}  batch {bs}x{cfg.max_seq_len} fwd+bwd {dt:.2f}s  "
        f"{summarize_route(out.route)}  {mem_tag()}  device={device}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
