from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hcrm.config import HCRMConfig


def gumbel_noise(logits: Tensor) -> Tensor:
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))


def gumbel_softmax(logits: Tensor, tau: float, hard: bool = False) -> Tensor:
    y = F.softmax((logits + gumbel_noise(logits)) / max(tau, 1e-4), dim=-1)
    if not hard:
        return y
    index = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
    return y_hard + (y - y.detach())


def rbf_range_logits(u: Tensor, n_buckets: int, sigma: float) -> Tensor:
    """Locality-sensitive range scores: u in (0, 1) vs equally spaced bucket centers."""
    centers = (torch.arange(n_buckets, device=u.device, dtype=u.dtype) + 0.5) / n_buckets
    dist = (u.unsqueeze(-1) - centers) ** 2
    return -dist / (2.0 * sigma * sigma + 1e-8)


class HierarchicalRouter(nn.Module):
    """
    Channelites (coarse ranges) gate Channels (fine rows).

    A learned 1-D routing index is bucketed like LSH. Nearby semantic
    content maps to nearby rows. Discrete choices use Gumbel-Softmax + STE.
    """

    def __init__(self, cfg: HCRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.key = nn.Linear(cfg.d_model, 1)
        self.refine = nn.Linear(cfg.d_model, cfg.n_channels)
        self.channelite_boost = nn.Linear(cfg.d_model, cfg.n_channelites)

    def forward(
        self,
        x: Tensor,
        tau: float | None = None,
        hard: bool | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        cfg = self.cfg
        tau = cfg.gumbel_tau if tau is None else tau
        _ = hard

        u = torch.sigmoid(self.key(x).squeeze(-1))
        loc = rbf_range_logits(u, cfg.n_channels, cfg.route_sigma)
        cite_loc = rbf_range_logits(u, cfg.n_channelites, cfg.route_sigma * 1.6)
        cite_logits = cite_loc + self.channelite_boost(x)
        cite_probs = F.softmax(cite_logits, dim=-1)

        per = cfg.channels_per_channelite
        parent = cite_probs.repeat_interleave(per, dim=-1)
        logits = loc + self.refine(x) + torch.log(parent.clamp_min(1e-8))

        soft = F.softmax(logits / max(tau, 1e-4), dim=-1)
        topv, topi = torch.topk(soft, k=cfg.top_k, dim=-1)
        if self.training:
            # Dense mixture so every channel row gets a gradient. Top-k is inference-only.
            gates = soft
        else:
            sparse = torch.zeros_like(soft).scatter_(-1, topi, topv)
            gates = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        grouped = gates.view(*gates.shape[:-1], cfg.n_channelites, per)
        cite_mass = grouped.sum(dim=-1)
        return gates, {
            "u": u,
            "channel_probs": soft,
            "channelite_probs": cite_probs,
            "channelite_mass": cite_mass,
            "top_channels": topi,
            "top_weights": topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-8),
        }


def load_balance_loss(channel_probs: Tensor, n_channels: int) -> Tensor:
    usage = channel_probs.mean(dim=(0, 1))
    return n_channels * (usage * usage).sum()


def locality_loss(u: Tensor) -> Tensor:
    if u.size(1) < 2:
        return u.new_zeros(())
    return (u[:, 1:] - u[:, :-1]).pow(2).mean()
