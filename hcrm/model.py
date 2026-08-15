from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from hcrm.config import HCRMConfig
from hcrm.routing import HierarchicalRouter, load_balance_loss, locality_loss


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * (x * rms).to(x.dtype))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires even d_model")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None, :, :], emb.sin().to(dtype)[None, :, :]


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin


class CausalLocalMix(nn.Module):
    """O(T) local mixing. Dilated depthwise conv + optional causal GRU."""

    def __init__(
        self,
        d_model: int,
        kernel: int,
        dropout: float,
        dilation: int = 1,
        use_gru: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.dilation = max(1, int(dilation))
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel,
            groups=d_model,
            bias=True,
            dilation=self.dilation,
        )
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, batch_first=True) if use_gru else None

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        pad = self.dilation * (self.kernel - 1)
        h = F.pad(h, (pad, 0))
        h = self.dw(h).transpose(1, 2)
        h = h * torch.sigmoid(self.gate(x))
        h = self.drop(self.out(h))
        if self.gru is not None:
            h, _ = self.gru(h)
        return h


class ChannelBank(nn.Module):
    """
    Channels are ordered rows of matrices. Related topics sit in nearby rows.
    Only the gated (top-k) rows contribute to the residual.
    Sequence-chunked / channel-grouped so a long window can fill RAM on purpose
    without going over the machine limit.
    """

    def __init__(self, cfg: HCRMConfig) -> None:
        super().__init__()
        c, d, f = cfg.n_channels, cfg.d_model, cfg.d_ff
        self.chunk = max(0, int(getattr(cfg, "channel_chunk", 0) or 0))
        self.group = max(0, int(getattr(cfg, "channel_group", 0) or 0))
        self.w1 = nn.Parameter(torch.empty(c, d, f))
        self.b1 = nn.Parameter(torch.zeros(c, f))
        self.w2 = nn.Parameter(torch.empty(c, f, d))
        self.b2 = nn.Parameter(torch.zeros(c, d))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def _slice(self, x: Tensor, gates: Tensor) -> Tensor:
        c = self.w1.size(0)
        group = self.group
        if group <= 0 or group >= c:
            hidden = torch.einsum("btd,cdf->btcf", x, self.w1) + self.b1
            hidden = F.silu(hidden)
            mixed = torch.einsum("btcf,cfd->btcd", hidden, self.w2) + self.b2
            return torch.einsum("btc,btcd->btd", gates, mixed)
        acc = None
        for i in range(0, c, group):
            sl = slice(i, min(i + group, c))
            hidden = torch.einsum("btd,cdf->btcf", x, self.w1[sl]) + self.b1[sl]
            hidden = F.silu(hidden)
            mixed = torch.einsum("btcf,cfd->btcd", hidden, self.w2[sl]) + self.b2[sl]
            part = torch.einsum("btc,btcd->btd", gates[:, :, sl], mixed)
            acc = part if acc is None else acc + part
        return acc

    def forward(self, x: Tensor, gates: Tensor) -> Tensor:
        t = x.size(1)
        chunk = self.chunk
        if chunk <= 0 or t <= chunk:
            return self._slice(x, gates)
        parts = []
        for i in range(0, t, chunk):
            sl = slice(i, i + chunk)
            parts.append(self._slice(x[:, sl], gates[:, sl]))
        return torch.cat(parts, dim=1)


class HCRMBlock(nn.Module):
    def __init__(self, cfg: HCRMConfig, channels: ChannelBank, router: HierarchicalRouter) -> None:
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.mix = CausalLocalMix(
            cfg.d_model,
            cfg.conv_kernel,
            cfg.dropout,
            dilation=int(getattr(cfg, "conv_dilation", 1) or 1),
            use_gru=bool(getattr(cfg, "use_gru", False)),
        )
        self.channels = channels
        self.router = router
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, tau: float | None = None) -> tuple[Tensor, dict[str, Any]]:
        x = x + 0.5 * self.mix(self.n1(x))
        h = self.n2(x)
        gates, route = self.router(h, tau=tau)
        x = x + 0.5 * self.drop(self.channels(h, gates))
        return x, route


@dataclass
class HCRMOutput:
    logits: Tensor
    loss: Tensor | None
    aux_loss: Tensor | None
    hidden: Tensor
    route: dict[str, Any]


class HCRM(nn.Module):
    def __init__(self, cfg: HCRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.use_rope = bool(getattr(cfg, "use_rope", False))
        if self.use_rope:
            self.pos_emb = None
            self.rope = RotaryEmbedding(cfg.d_model)
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.rope = None
        self.drop = nn.Dropout(cfg.dropout)
        self.router = HierarchicalRouter(cfg)
        self.channels = ChannelBank(cfg)
        self.blocks = nn.ModuleList(HCRMBlock(cfg, self.channels, self.router) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def count_params(self) -> int:
        seen: set[int] = set()
        n = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            n += p.numel()
        return n

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        tau: float | None = None,
    ) -> HCRMOutput:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} > max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(input_ids)
        if self.use_rope:
            cos, sin = self.rope(t, input_ids.device, x.dtype)
            x = apply_rope(x, cos, sin)
        else:
            pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        last_route: dict[str, Any] = {}
        use_ckpt = bool(getattr(self.cfg, "grad_checkpoint", False)) and self.training
        for block in self.blocks:
            if use_ckpt:
                x, last_route = checkpoint(block, x, tau, use_reentrant=False)
            else:
                x, last_route = block(x, tau=tau)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        loss = None
        aux = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_chunk = max(0, int(getattr(self.cfg, "ce_chunk", 0) or 0))
            if ce_chunk <= 0 or shift_logits.size(1) <= ce_chunk:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                total = shift_logits.new_zeros(())
                denom = 0
                vocab = shift_logits.size(-1)
                for i in range(0, shift_logits.size(1), ce_chunk):
                    sl = slice(i, i + ce_chunk)
                    chunk_logits = shift_logits[:, sl].reshape(-1, vocab)
                    chunk_labels = shift_labels[:, sl].reshape(-1)
                    total = total + F.cross_entropy(
                        chunk_logits, chunk_labels, ignore_index=-100, reduction="sum"
                    )
                    denom += int((chunk_labels != -100).sum().item())
                loss = total / max(1, denom)
            aux = (
                load_balance_loss(last_route["channel_probs"], self.cfg.n_channels)
                + locality_loss(last_route["u"])
            )
        return HCRMOutput(logits=logits, loss=loss, aux_loss=aux, hidden=hidden, route=last_route)

    @torch.no_grad()
    def mean_key(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        w = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
