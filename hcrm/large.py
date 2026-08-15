"""Colab / 32k HCRM (RoPE, grouped channel bank). Loads hcrm_smoltalk_32k.pt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hcrm.config import HCRMConfig
from hcrm.model import HCRMOutput


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms).to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
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
    def __init__(self, d_model: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.kernel = kernel
        self.dw = nn.Conv1d(d_model, d_model, kernel, groups=d_model, bias=True)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.kernel - 1, 0))
        h = self.dw(h).transpose(1, 2)
        return self.drop(self.out(h * torch.sigmoid(self.gate(x))))


def rbf_range_logits(u: Tensor, n_buckets: int, sigma: float) -> Tensor:
    centers = (torch.arange(n_buckets, device=u.device, dtype=u.dtype) + 0.5) / n_buckets
    return -((u.unsqueeze(-1) - centers) ** 2) / (2 * sigma * sigma + 1e-8)


class HierarchicalRouter(nn.Module):
    def __init__(self, cfg: HCRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.key = nn.Linear(cfg.d_model, 1)
        self.refine = nn.Linear(cfg.d_model, cfg.n_channels)
        self.channelite_boost = nn.Linear(cfg.d_model, cfg.n_channelites)

    def forward(self, x: Tensor, tau: float | None = None) -> tuple[Tensor, dict[str, Any]]:
        cfg = self.cfg
        tau = cfg.gumbel_tau if tau is None else tau
        u = torch.sigmoid(self.key(x).squeeze(-1))
        loc = rbf_range_logits(u, cfg.n_channels, cfg.route_sigma)
        cite = F.softmax(
            rbf_range_logits(u, cfg.n_channelites, cfg.route_sigma * 1.6) + self.channelite_boost(x),
            dim=-1,
        )
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
        return gates, {
            "u": u,
            "channel_probs": soft,
            "channelite_mass": mass,
            "top_channels": topi,
            "top_weights": topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-8),
        }


class ChannelBank(nn.Module):
    def __init__(self, cfg: HCRMConfig) -> None:
        super().__init__()
        c, d, f = cfg.n_channels, cfg.d_model, cfg.d_ff
        self.chunk = cfg.channel_chunk
        self.group = max(1, int(getattr(cfg, "channel_group", 4) or 4))
        self.w1 = nn.Parameter(torch.empty(c, d, f))
        self.b1 = nn.Parameter(torch.zeros(c, f))
        self.w2 = nn.Parameter(torch.empty(c, f, d))
        self.b2 = nn.Parameter(torch.zeros(c, d))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def _slice(self, x: Tensor, gates: Tensor) -> Tensor:
        b, t, d = x.shape
        out = x.new_zeros(b, t, d)
        c = self.w1.size(0)
        g = self.group
        for c0 in range(0, c, g):
            w1 = self.w1[c0 : c0 + g]
            hidden = torch.einsum("btd,gdf->btgf", x, w1) + self.b1[c0 : c0 + g]
            hidden = F.silu(hidden)
            mixed = torch.einsum("btgf,gfd->btgd", hidden, self.w2[c0 : c0 + g]) + self.b2[c0 : c0 + g]
            out = out + torch.einsum("btg,btgd->btd", gates[:, :, c0 : c0 + g], mixed)
        return out

    def forward(self, x: Tensor, gates: Tensor) -> Tensor:
        t = x.size(1)
        if t <= self.chunk:
            return self._slice(x, gates)
        parts = []
        for i in range(0, t, self.chunk):
            sl = slice(i, i + self.chunk)
            parts.append(self._slice(x[:, sl], gates[:, sl]))
        return torch.cat(parts, dim=1)


class HCRMBlock(nn.Module):
    def __init__(self, cfg: HCRMConfig, channels: ChannelBank, router: HierarchicalRouter) -> None:
        super().__init__()
        self.cfg = cfg
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.mix = CausalLocalMix(cfg.d_model, cfg.conv_kernel, cfg.dropout)
        self.channels = channels
        self.router = router
        self.drop = nn.Dropout(cfg.dropout)

    def _body(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = x + 0.5 * self.mix(self.n1(x))
        h = self.n2(x)
        gates, route = self.router(h)
        x = x + 0.5 * self.drop(self.channels(h, gates))
        return x, route["u"], route["channel_probs"], route["channelite_mass"], route["top_channels"]

    def forward(self, x: Tensor, tau: float | None = None) -> tuple[Tensor, dict[str, Any]]:
        _ = tau
        if self.training and self.cfg.grad_checkpoint:
            x, u, probs, mass, topi = torch.utils.checkpoint.checkpoint(self._body, x, use_reentrant=False)
        else:
            x, u, probs, mass, topi = self._body(x)
        topv, _ = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        return x, {
            "u": u,
            "channel_probs": probs,
            "channelite_mass": mass,
            "top_channels": topi,
            "top_weights": topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-8),
        }


class LargeHCRM(nn.Module):
    def __init__(self, cfg: HCRMConfig) -> None:
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

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        tau: float | None = None,
        return_logits: bool = False,
    ) -> HCRMOutput:
        _ = labels
        x = self.drop(self.rope(self.tok_emb(input_ids)))
        last_route: dict[str, Any] = {}
        for block in self.blocks:
            x, last_route = block(x, tau=tau)
        hidden = self.norm(x)
        logits = self.lm_head(hidden if return_logits else hidden[:, -1:])
        return HCRMOutput(logits=logits, loss=None, aux_loss=None, hidden=hidden, route=last_route)

    @torch.no_grad()
    def mean_key(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        w = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def cfg_from_blob(blob: dict[str, Any]) -> HCRMConfig:
    raw = dict(blob.get("config") or {})
    if blob.get("seq_len"):
        raw["max_seq_len"] = int(blob["seq_len"])
    allowed = {k: v for k, v in raw.items() if k in HCRMConfig.__dataclass_fields__}
    cfg = HCRMConfig(**allowed)
    cfg.vocab_size = int(blob["model"]["tok_emb.weight"].shape[0])
    cfg.grad_checkpoint = False
    return cfg


def load_large(ckpt: str | Path, device: torch.device) -> tuple[LargeHCRM, HCRMConfig, str]:
    path = Path(ckpt)
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg_from_blob(blob)
    model = LargeHCRM(cfg)
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    tok_id = blob.get("tokenizer") or "HuggingFaceTB/SmolLM2-135M-Instruct"
    return model, cfg, tok_id
