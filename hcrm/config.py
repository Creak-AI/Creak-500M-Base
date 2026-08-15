from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|think|>",
    "<|/think|>",
    "<|mem|>",
]

SYSTEM_PROMPT = (
    "You are Creak, a helpful AI assistant. Think in short <|think|> blocks as you go. "
    "Plan, write some of the answer, then think again to check or continue. "
    "Interleave thinking with the response. Never leave a think block empty."
)
REASON_SYSTEM_PROMPT = SYSTEM_PROMPT


@dataclass
class HCRMConfig:
    """Creak HCRM ~500M for L40S: channel-heavy bank, interleaved think."""

    vocab_size: int = 49152
    d_model: int = 512
    n_layers: int = 8
    n_channelites: int = 64
    n_channels: int = 512
    d_ff: int = 896
    conv_kernel: int = 7
    conv_dilation: int = 2
    use_gru: bool = False
    use_rope: bool = True
    max_seq_len: int = 2048
    top_k: int = 8
    dropout: float = 0.05
    gumbel_tau: float = 1.2
    infer_tau: float = 0.1
    route_sigma: float = 0.08
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    end_id: int = 2
    tie_embeddings: bool = True
    channel_chunk: int = 512
    channel_group: int = 32
    ce_chunk: int = 256
    grad_checkpoint: bool = True
    dataset_id: str = "creak-gpu-500m"
    dataset_file: str = "train"
    tokenizer_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def __post_init__(self) -> None:
        if self.n_channels % self.n_channelites != 0:
            raise ValueError("n_channels must be divisible by n_channelites")
        if self.top_k < 1 or self.top_k > self.n_channels:
            raise ValueError("top_k must be in [1, n_channels]")

    @property
    def channels_per_channelite(self) -> int:
        return self.n_channels // self.n_channelites

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HCRMConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    output_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    max_samples: int = 500000
    keep_think_chars: int = 0
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 64
    lr: float = 2.5e-4
    num_threads: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 100
    seed: int = 42
    max_minutes: float = 0.0
    num_workers: int = 2
    aux_balance: float = 0.005
    aux_locality: float = 0.0
    resume: bool = True
    phase: str = "chat"
    keep_step_ckpts: int = 2
    target_ram_gb: float = 12.0
    max_ram_gb: float = 14.5
    amp: str = "bf16"
    reason_minutes: float = 45.0
