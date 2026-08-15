"""Build the Colab HCRM + smol-smoltalk 32k notebook."""

from __future__ import annotations

import json
from pathlib import Path


def cell_md(text: str) -> dict:
    src = [line + "\n" for line in text.strip("\n").split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n")
        if not src[-1].endswith("\n"):
            src[-1] += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def cell_code(text: str) -> dict:
    src = [line + "\n" for line in text.strip("\n").split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n")
        if src[-1] and not src[-1].endswith("\n"):
            src[-1] += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


cells = []

cells.append(
    cell_md(
        """# HCRM on Colab — smol-smoltalk, 32k context

Train **Hierarchical Channel-Routed Memory** (no global attention) on
[`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
with a **32,768-token** packed context window.

| | |
|---|---|
| Hardware | Colab GPU (T4 16GB works; L4/A100 is faster) |
| Time cap | **3 hours** (stops and saves automatically) |
| Context | 32,768 tokens, packed conversations |
| Why HCRM | Local mixing is O(T), so 32k is feasible; Transformer attention is O(T²) |

T4 has ~15GB. Full-sequence CE logits are ~3GB at 16k×vocab, which is why a forward-only (or tight) probe can pass and **backward still OOM**. This notebook chunks the loss, checkpoints blocks, and probes a real AdamW+GradScaler step with headroom.

**Colab setup**

1. Runtime → Change runtime type → **T4 GPU** (or better)
2. If you already hit OOM: **Runtime → Restart session** (clears fragmented VRAM), then Run all
3. Optional: mount Drive in the save cell so the checkpoint survives the VM
"""
    )
)

cells.append(
    cell_code(
        """# GPU + deps — set allocator BEFORE importing torch
import os, subprocess, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Installing packages (1-2 min)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets>=2.20", "transformers>=4.44", "accelerate>=0.33",
    "safetensors", "tqdm",
])

import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available(), "Enable a GPU: Runtime → Change runtime type → T4 GPU"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("bf16", torch.cuda.is_bf16_supported())
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))"""
    )
)

cells.append(
    cell_code(
        """from dataclasses import dataclass, asdict
from pathlib import Path
import json, math, os, random, time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---- knobs (3h T4 budget) ----
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_SEQ = 32768
MAX_MINUTES = 170.0          # leave ~10 min for tokenize + save
MAX_STREAM = 80_000          # conversations to tokenize
D_MODEL = 256
N_LAYERS = 6
N_CHANNELITES = 8
N_CHANNELS = 16
D_FF = 512
TOP_K = 2
BATCH = 1
GRAD_ACCUM = 8
LR = 6e-4
SEED = 42
OUT_DIR = Path("/content/hcrm_smoltalk_32k")
DRIVE_DIR = Path("/content/drive/MyDrive/hcrm_smoltalk_32k")  # used if Drive is mounted

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("MAX_SEQ", MAX_SEQ, "budget_min", MAX_MINUTES)"""
    )
)

cells.append(
    cell_md("## HCRM model (RoPE, 32k, checkpointed, chunked CE)")
)

cells.append(
    cell_code(
        r'''@dataclass
class HCRMConfig:
    vocab_size: int = 49152
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_channelites: int = N_CHANNELITES
    n_channels: int = N_CHANNELS
    d_ff: int = D_FF
    conv_kernel: int = 7
    max_seq_len: int = MAX_SEQ
    top_k: int = TOP_K
    dropout: float = 0.0
    gumbel_tau: float = 1.0
    route_sigma: float = 0.08
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    tie_embeddings: bool = True
    channel_chunk: int = 1024
    channel_group: int = 4
    ce_chunk: int = 256
    grad_checkpoint: bool = True

    def __post_init__(self):
        assert self.n_channels % self.n_channelites == 0

    @property
    def channels_per_channelite(self) -> int:
        return self.n_channels // self.n_channelites


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms).to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE so 32k context does not need a 32k position table."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        pos = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :].to(x.dtype)
        sin = emb.sin()[None, :, :].to(x.dtype)
        return x * cos + rotate_half(x) * sin


class CausalLocalMix(nn.Module):
    def __init__(self, d_model, kernel, dropout):
        super().__init__()
        self.kernel = kernel
        self.dw = nn.Conv1d(d_model, d_model, kernel, groups=d_model, bias=True)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (self.kernel - 1, 0))
        h = self.dw(h).transpose(1, 2)
        return self.drop(self.out(h * torch.sigmoid(self.gate(x))))


def rbf_range_logits(u, n_buckets, sigma):
    centers = (torch.arange(n_buckets, device=u.device, dtype=u.dtype) + 0.5) / n_buckets
    return -((u.unsqueeze(-1) - centers) ** 2) / (2 * sigma * sigma + 1e-8)


class HierarchicalRouter(nn.Module):
    def __init__(self, cfg: HCRMConfig):
        super().__init__()
        self.cfg = cfg
        self.key = nn.Linear(cfg.d_model, 1)
        self.refine = nn.Linear(cfg.d_model, cfg.n_channels)
        self.channelite_boost = nn.Linear(cfg.d_model, cfg.n_channelites)

    def forward(self, x, tau=None):
        cfg = self.cfg
        tau = cfg.gumbel_tau if tau is None else tau
        u = torch.sigmoid(self.key(x).squeeze(-1))
        loc = rbf_range_logits(u, cfg.n_channels, cfg.route_sigma)
        cite = F.softmax(rbf_range_logits(u, cfg.n_channelites, cfg.route_sigma * 1.6) + self.channelite_boost(x), dim=-1)
        parent = cite.repeat_interleave(cfg.channels_per_channelite, dim=-1)
        logits = loc + self.refine(x) + torch.log(parent.clamp_min(1e-8))
        soft = F.softmax(logits / max(tau, 1e-4), dim=-1)
        topv, topi = torch.topk(soft, k=cfg.top_k, dim=-1)
        if self.training:
            gates = soft
        else:
            sparse = torch.zeros_like(soft).scatter_(-1, topi, topv)
            gates = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mass = gates.view(*gates.shape[:-1], cfg.n_channelites, cfg.channels_per_channelite).sum(-1)
        return gates, {"u": u, "channel_probs": soft, "channelite_mass": mass, "top_channels": topi}


class ChannelBank(nn.Module):
    """Mix channels in small groups so we never materialize [B, T, C, F]."""

    def __init__(self, cfg: HCRMConfig):
        super().__init__()
        c, d, f = cfg.n_channels, cfg.d_model, cfg.d_ff
        self.chunk = cfg.channel_chunk
        self.group = cfg.channel_group
        self.w1 = nn.Parameter(torch.empty(c, d, f))
        self.b1 = nn.Parameter(torch.zeros(c, f))
        self.w2 = nn.Parameter(torch.empty(c, f, d))
        self.b2 = nn.Parameter(torch.zeros(c, d))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def _slice(self, x, gates):
        b, t, d = x.shape
        out = x.new_zeros(b, t, d)
        c = self.w1.size(0)
        g = self.group
        for c0 in range(0, c, g):
            w1 = self.w1[c0:c0 + g]
            hidden = torch.einsum("btd,gdf->btgf", x, w1) + self.b1[c0:c0 + g]
            hidden = F.silu(hidden)
            mixed = torch.einsum("btgf,gfd->btgd", hidden, self.w2[c0:c0 + g]) + self.b2[c0:c0 + g]
            out = out + torch.einsum("btg,btgd->btd", gates[:, :, c0:c0 + g], mixed)
        return out

    def forward(self, x, gates):
        t = x.size(1)
        if t <= self.chunk:
            return self._slice(x, gates)
        parts = []
        for i in range(0, t, self.chunk):
            sl = slice(i, i + self.chunk)
            parts.append(self._slice(x[:, sl], gates[:, sl]))
        return torch.cat(parts, dim=1)


class HCRMBlock(nn.Module):
    def __init__(self, cfg, channels, router):
        super().__init__()
        self.cfg = cfg
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.mix = CausalLocalMix(cfg.d_model, cfg.conv_kernel, cfg.dropout)
        self.channels, self.router = channels, router
        self.drop = nn.Dropout(cfg.dropout)

    def _body(self, x):
        x = x + 0.5 * self.mix(self.n1(x))
        h = self.n2(x)
        gates, route = self.router(h)
        x = x + 0.5 * self.drop(self.channels(h, gates))
        return x, route["u"], route["channel_probs"], route["channelite_mass"], route["top_channels"]

    def forward(self, x, tau=None):
        if self.training and self.cfg.grad_checkpoint:
            x, u, probs, mass, topi = torch.utils.checkpoint.checkpoint(
                self._body, x, use_reentrant=False
            )
        else:
            x, u, probs, mass, topi = self._body(x)
        return x, {"u": u, "channel_probs": probs, "channelite_mass": mass, "top_channels": topi}


def chunked_cross_entropy(hidden, lm_head, labels, ignore_index=-100, chunk=256):
    """Never build [B, T, vocab] logits — that is a 3GB fp32 alloc at 16k on SmolLM's vocab."""
    h = hidden[:, :-1]
    y = labels[:, 1:]
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    ntok = (y != ignore_index).sum().clamp(min=1)
    for i in range(0, h.size(1), chunk):
        logits = lm_head(h[:, i:i + chunk]).float()
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, i:i + chunk].reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
    return total / ntok


class HCRM(nn.Module):
    def __init__(self, cfg: HCRMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.rope = RotaryEmbedding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.router = HierarchicalRouter(cfg)
        self.channels = ChannelBank(cfg)
        self.blocks = nn.ModuleList(HCRMBlock(cfg, self.channels, self.router) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None, tau=None, return_logits=False):
        x = self.drop(self.rope(self.tok_emb(input_ids)))
        route = {}
        for block in self.blocks:
            x, route = block(x, tau=tau)
        hidden = self.norm(x)
        loss = None
        if labels is not None:
            loss = chunked_cross_entropy(hidden, self.lm_head, labels, chunk=self.cfg.ce_chunk)
            usage = route["channel_probs"].mean(dim=(0, 1))
            loss = loss + 0.005 * self.cfg.n_channels * (usage * usage).sum()
        if return_logits:
            logits = self.lm_head(hidden)
        else:
            logits = self.lm_head(hidden[:, -1:])
        return logits, loss, route

print("HCRM classes ready")'''
    )
)

cells.append(
    cell_md("## Probe GPU memory with a real train step (forward + backward + AdamW)")
)

cells.append(
    cell_code(
        """from transformers import AutoTokenizer
import gc

tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = HCRMConfig(
    vocab_size=len(tok),
    pad_id=tok.pad_token_id,
    bos_id=tok.bos_token_id or tok.eos_token_id,
    eos_id=tok.eos_token_id,
)
print("tokenizer", TOKENIZER_ID, "vocab", len(tok))

device = torch.device("cuda")
use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
print("amp", amp_dtype)
total_mem = torch.cuda.get_device_properties(0).total_memory


def autocast():
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def try_seq(seq_len: int) -> bool:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = opt = scaler = ids = labels = loss = None
    try:
        m = HCRM(cfg).to(device)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
        labels = ids.clone()
        with autocast():
            _, loss, _ = m(ids, labels=labels)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        peak = torch.cuda.max_memory_allocated()
        ok = peak < total_mem * 0.78
        n = m.count_params()
        tag = "OK" if ok else "TIGHT"
        print(f"{tag}  seq={seq_len:,}  params={n:,}  peak={peak/1e9:.2f}/{total_mem/1e9:.2f} GB")
        return ok
    except torch.cuda.OutOfMemoryError:
        print(f"OOM seq={seq_len:,}")
        return False
    finally:
        del m, opt, scaler, ids, labels, loss
        gc.collect()
        torch.cuda.empty_cache()


seq_len = 4096
for candidate in (32768, 16384, 8192, 4096):
    if try_seq(candidate):
        seq_len = candidate
        break
else:
    raise RuntimeError("Even 4096 OOM — try a larger GPU")

cfg.max_seq_len = seq_len
print("Using max_seq_len =", seq_len)"""
    )
)

cells.append(
    cell_md(
        """## Tokenize and pack `smol-smoltalk` to 32k

Conversations are usually much shorter than 32k. Packing concatenates them so each GPU step actually uses the long window.
"""
    )
)

cells.append(
    cell_code(
        """from datasets import load_dataset

print("Streaming", DATASET_ID)
raw = load_dataset(DATASET_ID, split="train", streaming=True)

def to_text(messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = []
        for m in messages or []:
            parts.append(f"<|{m.get('role','user')}|>{m.get('content','')}")
        return "".join(parts)

tokenized = []
n = 0
t0 = time.time()
for row in raw:
    text = to_text(row.get("messages"))
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=seq_len)["input_ids"]
    if len(ids) >= 8:
        tokenized.append(ids)
        n += 1
    if n >= MAX_STREAM:
        break
    if n % 5000 == 0 and n:
        print(f"  tokenized {n}/{MAX_STREAM}  {time.time()-t0:.0f}s")
print(f"tokenized {len(tokenized)} conversations in {time.time()-t0:.0f}s")

def pack_sequences(rows, max_len, eos_id):
    packs, cur = [], []
    for ids in rows:
        piece = ids if ids[-1] == eos_id else ids + [eos_id]
        if len(piece) > max_len:
            piece = piece[:max_len]
        if len(cur) + len(piece) > max_len:
            if cur:
                packs.append(cur)
            cur = list(piece)
        else:
            cur.extend(piece)
    if cur:
        packs.append(cur)
    return packs

packs = pack_sequences(tokenized, seq_len, tok.eos_token_id)
print("packed sequences", len(packs), "mean_len", sum(map(len, packs)) / max(1, len(packs)))
del tokenized

class PackedLM(Dataset):
    def __init__(self, packs, pad_id, max_len):
        self.packs, self.pad_id, self.max_len = packs, pad_id, max_len

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, i):
        ids = self.packs[i][: self.max_len]
        n = len(ids)
        if n < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - n)
        x = torch.tensor(ids, dtype=torch.long)
        y = x.clone()
        y[x == self.pad_id] = -100
        # do not train on padding; keep full packed tokens
        return {"input_ids": x, "labels": y}

ds = PackedLM(packs, tok.pad_token_id, seq_len)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
print("steps/epoch", len(loader))"""
    )
)

cells.append(
    cell_md("## Train (auto-stops at 170 minutes)")
)

cells.append(
    cell_code(
        """gc.collect()
torch.cuda.empty_cache()
ds = PackedLM(packs, tok.pad_token_id, seq_len)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
model = HCRM(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
print("params", f"{model.count_params():,}")

def cosine_lr(step, warmup, total, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

epochs = 8
total_opt = max(1, (len(loader) // GRAD_ACCUM) * epochs)
warmup = max(50, int(0.03 * total_opt))
deadline = time.time() + MAX_MINUTES * 60
ckpt = OUT_DIR / "hcrm_smoltalk_32k.pt"
best = float("inf")
opt_step = micro = 0
model.train()
opt.zero_grad(set_to_none=True)
t0 = time.time()
stop = False

print(f"training up to {MAX_MINUTES:.0f} min  seq={seq_len}  accum={GRAD_ACCUM}")
try:
    for epoch in range(epochs):
        if stop:
            break
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        running = 0.0
        nrun = 0
        for batch in bar:
            if time.time() >= deadline:
                print("Time budget reached")
                stop = True
                break
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, loss, route = model(ids, labels=labels)
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += float(loss.detach()) * GRAD_ACCUM
            nrun += 1
            micro += 1
            if micro % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = cosine_lr(opt_step, warmup, total_opt, LR)
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    avg = running / max(1, nrun)
                    running = nrun = 0
                    cite = int(route["channelite_mass"][:, -8:].detach().float().mean(0).argmax())
                    left = max(0.0, deadline - time.time())
                    bar.set_postfix(loss=f"{avg:.3f}", ppl=f"{math.exp(min(avg, 20)):.1f}",
                                    lr=f"{lr:.2e}", cite=cite, left=f"{left/60:.1f}m")
                if opt_step % 100 == 0:
                    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                                "step": opt_step, "seq_len": seq_len, "tokenizer": TOKENIZER_ID}, ckpt)
        torch.save({"model": model.state_dict(), "config": asdict(cfg),
                    "step": opt_step, "seq_len": seq_len, "tokenizer": TOKENIZER_ID, "loss": best}, ckpt)
except KeyboardInterrupt:
    print("interrupted — saving")

torch.save({"model": model.state_dict(), "config": asdict(cfg),
            "step": opt_step, "seq_len": seq_len, "tokenizer": TOKENIZER_ID}, ckpt)
cfg_path = OUT_DIR / "config.json"
cfg_path.write_text(json.dumps(asdict(cfg) | {"seq_len": seq_len, "tokenizer": TOKENIZER_ID}, indent=2))
print(f"saved {ckpt}  steps={opt_step}  elapsed={(time.time()-t0)/60:.1f}m")

if Path("/content/drive/MyDrive").exists():
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(ckpt, DRIVE_DIR / ckpt.name)
    shutil.copy2(cfg_path, DRIVE_DIR / "config.json")
    print("copied to", DRIVE_DIR)
else:
    print("Drive not mounted — checkpoint lives in", OUT_DIR)
    print("To persist: from google.colab import drive; drive.mount('/content/drive') then re-run this save block")"""
    )
)

cells.append(
    cell_md("## Optional: mount Drive, then re-run the last few lines of the train cell")
)

cells.append(
    cell_code(
        """# from google.colab import drive
# drive.mount('/content/drive')"""
    )
)

cells.append(
    cell_md("## Chat (32k window, greedy-ish sampling)")
)

cells.append(
    cell_code(
        r'''@torch.no_grad()
def generate(prompt: str, max_new=128, temperature=0.7, top_p=0.9):
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    stop = {tok.eos_token_id}
    last = None
    repeat = 0
    for _ in range(max_new):
        window = ids[:, -cfg.max_seq_len:]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, _, route = model(window)
        logits = logits[:, -1, :].float()
        if last is not None:
            logits[0, last] /= 1.3
        if temperature <= 0:
            nxt = int(logits.argmax(-1))
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_p, dim=-1)
            mask = cdf - sorted_p > top_p
            sorted_p = sorted_p.masked_fill(mask, 0)
            sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
            nxt = int(sorted_i[0, torch.multinomial(sorted_p[0], 1)])
        if nxt in stop:
            break
        if nxt == last:
            repeat += 1
            if repeat >= 3:
                break
        else:
            repeat = 0
        last = nxt
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    out = tok.decode(ids[0, tok(text, return_tensors='pt').input_ids.shape[1]:], skip_special_tokens=True)
    cite = int(route["channelite_mass"][0, -1].argmax())
    chans = [int(x) for x in route["top_channels"][0, -1].tolist()]
    print(f"[route] channelite={cite} channels={chans}  u={float(route['u'][0,-1]):.3f}")
    return out.strip()

print(generate("Write a 4-sentence story about a robot who learns to garden."))'''
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": [],
            "toc_visible": True,
        },
    },
    "cells": cells,
}

out = Path(__file__).resolve().parents[1] / "HCRM_Colab_smol_smoltalk_32k.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(cells))
