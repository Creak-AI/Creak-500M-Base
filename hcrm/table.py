from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import re

import torch
from torch import Tensor


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) >= 2}


@dataclass
class MemoryEntry:
    channel: int
    channelite: int
    u: float
    text: str
    key: list[float]
    ts: str
    role: str = "turn"
    key_query: list[float] = field(default_factory=list)
    key_response: list[float] = field(default_factory=list)

    def query_vec(self) -> list[float]:
        return self.key_query or self.key

    def score(
        self,
        query: Tensor,
        query_u: float,
        query_text: str = "",
        now: datetime | None = None,
    ) -> float:
        vec = self.query_vec()
        k = torch.tensor(vec, device=query.device, dtype=query.dtype).flatten()
        q = query.flatten()
        n = min(int(k.numel()), int(q.numel()))
        if n == 0:
            cos = 0.0
        else:
            cos = float(torch.nn.functional.cosine_similarity(q[:n], k[:n], dim=0))
        loc = math.exp(-((query_u - self.u) ** 2) / 0.02)
        recency = 0.5
        try:
            ts = datetime.fromisoformat(self.ts.replace("Z", "+00:00"))
            age_h = max(0.0, ((now or datetime.now(timezone.utc)) - ts).total_seconds() / 3600.0)
            recency = math.exp(-age_h / 12.0)
        except Exception:
            recency = 0.5
        qw = _words(query_text)
        ew = _words(self.text)
        if qw and ew:
            kw = len(qw & ew) / max(1, len(qw))
        else:
            kw = 0.0
        return 0.50 * cos + 0.20 * loc + 0.15 * recency + 0.15 * kw


class RuntimeTable:
    """
    Persistent, database-style context. New facts are appended to disk under
    the routed channel instead of fine-tuning the encoder.
    Retrieve on user-query keys, with recency and keyword bias.
    """

    def __init__(self, path: str | Path, n_channels: int, n_channelites: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_channels = n_channels
        self.n_channelites = n_channelites
        self.entries: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        fields = MemoryEntry.__dataclass_fields__
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            self.entries.append(MemoryEntry(**{k: v for k, v in raw.items() if k in fields}))

    def write(
        self,
        channel: int,
        u: float,
        text: str,
        key: Tensor,
        role: str = "turn",
        key_query: Tensor | None = None,
        key_response: Tensor | None = None,
    ) -> MemoryEntry:
        per = max(1, self.n_channels // self.n_channelites)
        q = key_query if key_query is not None else key
        q_list = [float(x) for x in q.detach().cpu().flatten().tolist()]
        r_list = (
            [float(x) for x in key_response.detach().cpu().flatten().tolist()]
            if key_response is not None
            else []
        )
        entry = MemoryEntry(
            channel=int(channel),
            channelite=int(channel) // per,
            u=float(u),
            text=text.strip()[:800],
            key=q_list,
            ts=datetime.now(timezone.utc).isoformat(),
            role=role,
            key_query=q_list,
            key_response=r_list,
        )
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
        return entry

    def lookup(
        self,
        channels: list[int],
        query_key: Tensor,
        query_u: float,
        k: int = 4,
        neighbor: int = 1,
        query_text: str = "",
    ) -> list[tuple[MemoryEntry, float]]:
        allowed = set()
        for ch in channels:
            for d in range(-neighbor, neighbor + 1):
                n = ch + d
                if 0 <= n < self.n_channels:
                    allowed.add(n)
        now = datetime.now(timezone.utc)
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self.entries:
            if entry.channel not in allowed:
                continue
            scored.append((entry, entry.score(query_key, query_u, query_text=query_text, now=now)))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    def summary(self) -> dict[str, Any]:
        by_ch: dict[int, int] = {}
        for e in self.entries:
            by_ch[e.channel] = by_ch.get(e.channel, 0) + 1
        return {"n": len(self.entries), "by_channel": by_ch}

    def rewrite(self, keep: list[MemoryEntry]) -> None:
        self.entries = list(keep)
        with self.path.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")

    def prune(self, drop) -> int:
        keep = [e for e in self.entries if not drop(e)]
        n = len(self.entries) - len(keep)
        if n:
            self.rewrite(keep)
        return n
