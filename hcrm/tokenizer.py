from __future__ import annotations

from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from hcrm.config import SPECIAL_TOKENS, HCRMConfig

SMOLLM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class _EncodeOut:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class HFTok:
    """Duck-typed HuggingFace tokenizer so pack/chat/train keep the same API."""

    def __init__(self, raw) -> None:
        self.raw = raw

    def token_to_id(self, name: str) -> int | None:
        if not name:
            return None
        vocab = self.raw.get_vocab()
        if name not in vocab:
            return None
        return int(vocab[name])

    def get_vocab_size(self) -> int:
        return int(len(self.raw))

    def encode(self, text: str, add_special_tokens: bool = False) -> _EncodeOut:
        ids = self.raw.encode(text, add_special_tokens=add_special_tokens)
        return _EncodeOut([int(i) for i in ids])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.raw.decode(ids, skip_special_tokens=skip_special_tokens)


def load_smollm_creak_tokenizer(save_dir: str | Path, tokenizer_id: str = SMOLLM_ID) -> HFTok:
    from transformers import AutoTokenizer

    save_dir = Path(save_dir)
    if (save_dir / "tokenizer.json").exists() or (save_dir / "tokenizer_config.json").exists():
        raw = AutoTokenizer.from_pretrained(str(save_dir), use_fast=True)
        return HFTok(raw)
    raw = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    extra = [t for t in SPECIAL_TOKENS if t not in raw.get_vocab()]
    if extra:
        raw.add_special_tokens({"additional_special_tokens": extra})
    if raw.pad_token is None:
        raw.pad_token = raw.eos_token
    if "<|pad|>" in raw.get_vocab() and raw.pad_token != "<|pad|>":
        raw.pad_token = "<|pad|>"
    raw.padding_side = "right"
    try:
        raw.model_max_length = max(int(raw.model_max_length or 0), 8192)
    except Exception:
        pass
    save_dir.mkdir(parents=True, exist_ok=True)
    raw.save_pretrained(str(save_dir))
    return HFTok(raw)


def train_tokenizer(texts, vocab_size: int, save_path: str | Path) -> Tokenizer:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    bos_id = tok.token_to_id("<|bos|>")
    eos_id = tok.token_to_id("<|eos|>")
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.enable_padding(pad_id=tok.token_to_id("<|pad|>"), pad_token="<|pad|>")
    tok.save(str(save_path))
    _ = bos_id, eos_id
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    tok.enable_padding(pad_id=tok.token_to_id("<|pad|>") or 0, pad_token="<|pad|>")
    return tok


def encode(
    tok: Any,
    text: str,
    max_len: int,
    add_special: bool = False,
    keep_tail: bool = False,
) -> list[int]:
    ids = tok.encode(text, add_special_tokens=add_special).ids
    if len(ids) <= max_len:
        return ids
    if keep_tail:
        bos = tok.token_to_id("<|bos|>")
        ids = ids[-(max_len - (1 if bos is not None else 0)) :]
        if bos is not None:
            ids = [bos] + ids
        return ids[:max_len]
    ids = ids[:max_len]
    eos = tok.token_to_id("<|eos|>")
    if eos is not None:
        ids[-1] = eos
    return ids


def decode(tok: Any, ids: list[int], skip_special: bool = False) -> str:
    if skip_special:
        special = {tok.token_to_id(t) for t in SPECIAL_TOKENS if tok.token_to_id(t) is not None}
        ids = [i for i in ids if i not in special]
    return tok.decode(ids, skip_special_tokens=skip_special)


def _tid(tok: Any, *names: str, default: int | None = None) -> int | None:
    for name in names:
        vid = tok.token_to_id(name)
        if vid is not None:
            return int(vid)
    return default


def apply_config_special_ids(cfg: HCRMConfig, tok: Any) -> HCRMConfig:
    cfg.vocab_size = tok.get_vocab_size()
    pad = _tid(tok, "<|pad|>")
    if pad is None and hasattr(tok, "raw"):
        pad = getattr(tok.raw, "pad_token_id", None)
    cfg.pad_id = int(pad if pad is not None else 0)
    cfg.bos_id = int(_tid(tok, "<|bos|>", "<|im_start|>", "<|endoftext|>", default=1))
    cfg.eos_id = int(_tid(tok, "<|eos|>", "<|im_end|>", "<|endoftext|>", default=2))
    cfg.unk_id = int(_tid(tok, "<|unk|>", default=3))
    end_id = _tid(tok, "<|end|>")
    cfg.end_id = cfg.eos_id if end_id is None else end_id
    return cfg


def load_matching_tokenizer(ckpt: Path, weight_vocab: int, tokenizer_json: str | None = None):
    """Load the SmolLM2+Creak tokenizer that matches a checkpoint vocab."""
    from tokenizers import Tokenizer

    root = ckpt if ckpt.is_dir() else ckpt.parent
    dirs = [
        root / "hf_tokenizer",
        Path("checkpoints/hf_tokenizer"),
        Path("data/smollm2_creak_tokenizer"),
    ]
    for folder in dirs:
        if (folder / "tokenizer_config.json").exists() or (folder / "tokenizer.json").exists():
            tok = load_smollm_creak_tokenizer(folder)
            if tok.get_vocab_size() == weight_vocab:
                return tok
    files = [
        root / "tokenizer.json",
        Path("data/creak_tokenizer.json"),
        Path("data/tokenizer.json"),
    ]
    if tokenizer_json:
        try:
            tok = Tokenizer.from_str(tokenizer_json)
            if tok.get_vocab_size() == weight_vocab:
                return tok
        except Exception:
            pass
    for path in files:
        if not path.exists():
            continue
        try:
            tok = load_tokenizer(path)
        except Exception:
            continue
        if tok.get_vocab_size() == weight_vocab:
            return tok
    return None
