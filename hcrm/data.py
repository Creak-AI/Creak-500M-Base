from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import random
import re
import urllib.request

import torch
from torch.utils.data import Dataset

from hcrm.config import REASON_SYSTEM_PROMPT, SYSTEM_PROMPT, HCRMConfig
from hcrm.seed_qa import SEED_QA
from hcrm.tokenizer import encode, load_tokenizer

DATASET_URL = "https://huggingface.co/datasets/clzoro/Claude-Distills/resolve/main/claude_distill.jsonl"

THINK_RE = re.compile(
    r"(?is)<think>(.*?)</think>|<thinking>(.*?)</thinking>|"
    r"<\|think\|>(.*?)<\|/think\|>"
)


def strip_assistant(content: str, keep_think_chars: int) -> str:
    text = content or ""
    match = THINK_RE.search(text)
    if not match:
        return text.strip()
    think = next((g for g in match.groups() if g), "") or ""
    rest = text[match.end() :].strip()
    think = " ".join(think.split())
    if keep_think_chars > 0 and think:
        clipped = think[:keep_think_chars].rsplit(" ", 1)[0] or think[:keep_think_chars]
        return f"<|think|>{clipped}<|/think|>{rest}".strip()
    return rest or think[: max(keep_think_chars, 80)]


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] or text[:n]


def wrap_turn(role: str, content: str) -> str:
    return f"<|{role}|>{content}<|end|>"


_TURN_RE = re.compile(r"<\|(system|user|assistant)\|>(.*?)<\|end\|>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<\|think\|>(.*?)<\|/think\|>", re.DOTALL)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_PLAN = (
    "The user asked about {topic}. This is {kind}. I'll plan a clear answer, then check it.",
    "Goal: answer '{topic}'. I'll draft the first part, then verify before finishing.",
    "I need to respond to {topic}. Think first, write some of the answer, then think again.",
    "What is being asked: {topic}. I'll reason briefly, reply, then double-check.",
)
_CHECK = (
    "Check: does that match '{topic}'? If yes, finish. If not, correct the rest.",
    "That was the first piece. Next I'll complete the answer and stay on topic.",
    "Pause and check my reasoning. Then write the rest without rambling.",
    "Re-read the question. The next sentences should stay specific.",
)
_VERIFY = (
    "Verify the last steps. The final answer should match the work above.",
    "Re-read the question and confirm the number or claim before stating it.",
    "Check the arithmetic and units. Then give only the confirmed answer.",
)
_REASON_SOURCES = {"gsm8k", "metamath", "numina", "orca-math"}


def _topic(user: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", user or "")
    if not words:
        return "the request"
    return _clip(" ".join(words[:8]), 60)


def _kind(user: str) -> str:
    u = (user or "").lower()
    if re.search(r"\d", u) and re.search(r"[+\-*/=]|how many|what is|percent", u):
        return "a math question"
    if len(u.split()) <= 3:
        return "a short chat turn"
    if "story" in u:
        return "a story request"
    if any(g in u for g in ("hi", "hello", "hey", "thanks", "bye")):
        return "a greeting"
    return "a question"


def _sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _extract_thinks(text: str) -> tuple[list[str], str]:
    thinks = [" ".join(m.split()) for m in _THINK_BLOCK_RE.findall(text or "") if m.strip()]
    rest = _THINK_BLOCK_RE.sub(" ", text or "")
    rest = re.sub(r"<<.*?>>", "", rest)
    rest = " ".join(rest.split())
    return thinks, rest


def _split_cot(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if "####" not in raw:
        return "", raw
    left, right = raw.rsplit("####", 1)
    rationale = " ".join(re.sub(r"<<.*?>>", "", left).split())
    final = " ".join(right.split())
    final = re.sub(r"(?i)^the answer is:?\s*", "", final).strip()
    m = re.search(r"(?i)the answer is:?\s*(.+)$", final)
    if m:
        final = m.group(1).strip()
    return rationale, final or rationale


def weave_assistant(user: str, assistant: str) -> str:
    """Interleave real think blocks with the answer. Empty tags never score as thinking."""
    text = re.sub(r"<<.*?>>", "", assistant or "")
    text = " ".join(text.split())
    rationale, cot_final = _split_cot(text)
    thinks, answer = _extract_thinks(text if not rationale else cot_final)
    if rationale:
        thinks = [rationale] + thinks
        answer = cot_final or answer
    answer = answer.strip()
    if not answer:
        answer = thinks[-1] if thinks else "I am not sure."
        if thinks:
            thinks = thinks[:-1]
    topic = _topic(user)
    kind = _kind(user)
    h = int(hashlib.md5(f"{user or ''}\n{answer}".encode("utf-8")).hexdigest(), 16) & 0xFFFF
    plan = _PLAN[h % len(_PLAN)].format(topic=topic, kind=kind)
    check = _CHECK[(h // 3) % len(_CHECK)].format(topic=topic)
    verify = _VERIFY[(h // 5) % len(_VERIFY)]
    first = _clip(thinks[0], 700) if thinks and len(thinks[0]) >= 12 else plan
    sents = _sentences(answer)
    numeric = bool(re.search(r"\d", answer)) and (bool(rationale) or kind == "a math question")
    if len(sents) <= 1 or len(answer) < 90:
        if numeric:
            return (
                f"<|think|>{first}<|/think|>Working toward the answer. "
                f"<|think|>{verify}<|/think|>{answer}"
            )
        return f"<|think|>{first}<|/think|>{answer}"
    mid = max(1, len(sents) // 2)
    head = " ".join(sents[:mid]).strip()
    tail = " ".join(sents[mid:]).strip()
    if not tail:
        return f"<|think|>{first}<|/think|>{head}"
    second = _clip(thinks[1], 400) if len(thinks) > 1 and len(thinks[1]) >= 12 else (verify if numeric else check)
    return f"<|think|>{first}<|/think|>{head}<|think|>{second}<|/think|>{tail}"


def transform_dialog(text: str, system: str = SYSTEM_PROMPT) -> str:
    turns = _TURN_RE.findall(text or "")
    if not any(role == "assistant" for role, _ in turns):
        return ""
    last_user = ""
    parts = ["<|bos|>"]
    for role, content in turns:
        content = " ".join((content or "").split())
        if role == "system":
            content = system
        elif role == "user":
            last_user = content
        elif role == "assistant":
            content = weave_assistant(last_user, content)
        parts.append(wrap_turn(role, content))
    return "".join(parts)


def format_creak_dialog(
    turns: list[tuple[str, str]],
    system: str = SYSTEM_PROMPT,
    weave: bool = True,
) -> str:
    parts = ["<|bos|>", wrap_turn("system", system)]
    last_user = ""
    for role, content in turns:
        if role == "user":
            last_user = content
        if role == "assistant" and weave:
            content = weave_assistant(last_user, content)
        parts.append(wrap_turn(role, content))
    return "".join(parts)


def format_messages(messages: list[dict[str, Any]], keep_think_chars: int) -> str:
    user = ""
    assistant = ""
    for msg in messages or []:
        role = (msg.get("role") or "user").strip().lower()
        content = str(msg.get("content") or "").strip()
        if role == "user" and content:
            user = content
        elif role == "assistant" and content:
            assistant = content if keep_think_chars else strip_assistant(content, keep_think_chars)
    user = _clip(user, 220)
    assistant = _clip(assistant, 520)
    if len(user) < 12 or len(assistant) < 24:
        return ""
    return format_creak_dialog([("user", user), ("assistant", assistant)])


def _open_dataset_stream(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TinyBig-HCRM/0.1"},
    )
    return urllib.request.urlopen(req, timeout=120)


def iter_claude_distills(url: str = DATASET_URL) -> Iterator[dict[str, Any]]:
    with _open_dataset_stream(url) as resp:
        buf = ""
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


STORY_PROMPTS = (
    "Write a short story.",
    "Tell me a short story.",
    "Tell a story about a child.",
    "Write a simple story.",
)


def format_story(story: str, prompt: str) -> str:
    story = _clip(story, 420)
    if len(story) < 40:
        return ""
    return format_creak_dialog([("user", prompt), ("assistant", story)])


def prepare_tinystories(
    out_path: str | Path,
    max_samples: int,
) -> list[str]:
    from datasets import load_dataset

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming roneneldan/TinyStories -> {max_samples} stories", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts: list[str] = []
    seen = 0
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            seen += 1
            raw = str(row.get("text") or "").strip()
            prompt = STORY_PROMPTS[kept % len(STORY_PROMPTS)]
            text = format_story(raw, prompt)
            if not text:
                continue
            f.write(json.dumps({"text": text, "source": "tinystories"}, ensure_ascii=False) + "\n")
            texts.append(text)
            kept += 1
            if kept >= max_samples:
                break
            if seen % 500 == 0:
                print(f"  scanned {seen} raw rows, kept {kept}/{max_samples}", flush=True)
    print(f"Wrote {kept} TinyStories examples to {out_path} (scanned {seen})", flush=True)
    return texts


def prepare_subset(
    out_path: str | Path,
    max_samples: int,
    keep_think_chars: int,
    url: str = DATASET_URL,
) -> list[str]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    seen = 0
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_claude_distills(url):
            seen += 1
            messages = row.get("messages") or []
            user = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
            if len(str(user).strip()) < 12:
                continue
            text = format_messages(messages, keep_think_chars)
            if not text:
                continue
            rec = {
                "text": text,
                "source": row.get("source") or "claude-distills",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            texts.append(text)
            kept += 1
            if kept >= max_samples:
                break
            if seen % 500 == 0:
                print(f"  scanned {seen} raw rows, kept {kept}/{max_samples}", flush=True)
    print(f"Wrote {kept} examples to {out_path} (scanned {seen})", flush=True)
    return texts


def load_subset_texts(path: str | Path) -> list[str]:
    texts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append(json.loads(line)["text"])
    return texts


def iter_jsonl_texts(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


class JsonlTextIter:
    """Restartable text iterator so BPE training can rescan the file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self):
        return iter_jsonl_texts(self.path)


def format_smoltalk_messages(
    messages: list[dict[str, Any]],
    max_chars: int = 8000,
    system: str = SYSTEM_PROMPT,
    keep_think_chars: int = 0,
) -> str:
    parts = ["<|bos|>", wrap_turn("system", system)]
    n_user = 0
    n_asst = 0
    last_user = ""
    _ = keep_think_chars
    for msg in messages or []:
        role = str(msg.get("role") or "user").strip().lower()
        content = " ".join(str(msg.get("content") or "").split())
        if not content:
            continue
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        if role == "user":
            last_user = content
        elif role == "assistant":
            content = weave_assistant(last_user, content)
        if len(content) > 2800:
            content = content[:2800].rsplit(" ", 1)[0] or content[:2800]
        parts.append(wrap_turn(role, content))
        if role == "user":
            n_user += 1
        elif role == "assistant":
            n_asst += 1
    if n_user < 1 or n_asst < 1:
        return ""
    text = "".join(parts)
    if len(text) > max_chars:
        text = text[: max(80, max_chars - 7)] + "<|end|>"
    return text


def format_seed_qa(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> str:
    return format_creak_dialog([("user", user), ("assistant", assistant)], system=system)


def append_seed_chats(out_path: str | Path, repeats: int = 8) -> int:
    """Mix gold Creak turns into the chat corpus so greetings are not a fluke."""
    out_path = Path(out_path)
    n = 0
    with out_path.open("a", encoding="utf-8") as f:
        for _ in range(max(1, repeats)):
            for user, assistant in SEED_QA:
                rec = {"text": format_seed_qa(user, assistant), "source": "seed-qa"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"Appended {n} Creak seed chats to {out_path}", flush=True)
    return n


def prepare_smoltalk(
    out_path: str | Path,
    max_samples: int,
    keep_think_chars: int = 0,
) -> int:
    from datasets import load_dataset

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming HuggingFaceTB/smol-smoltalk -> {max_samples} Creak chats", flush=True)
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
    seen = 0
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            seen += 1
            text = format_smoltalk_messages(
                row.get("messages") or [],
                keep_think_chars=keep_think_chars,
            )
            if not text:
                continue
            rec = {"text": text, "source": row.get("source") or "smol-smoltalk"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= max_samples:
                break
            if kept % 1000 == 0:
                print(f"  kept {kept}/{max_samples} (scanned {seen})", flush=True)
    print(f"Wrote {kept} smol-smoltalk examples to {out_path} (scanned {seen})", flush=True)
    append_seed_chats(out_path)
    return kept


def format_gsm8k(question: str, answer: str) -> str:
    raw = (answer or "").strip()
    if not question.strip() or not raw:
        return ""
    return format_creak_dialog(
        [("user", _clip(question, 800)), ("assistant", raw)],
        system=REASON_SYSTEM_PROMPT,
    )


def prepare_gsm8k(out_path: str | Path, max_samples: int = 8000) -> int:
    from datasets import load_dataset

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading openai/gsm8k -> {max_samples} reasoning chats", flush=True)
    last_err: Exception | None = None
    ds = None
    for kwargs in (
        {"path": "openai/gsm8k", "name": "main", "split": "train"},
        {"path": "gsm8k", "name": "main", "split": "train"},
    ):
        try:
            ds = load_dataset(**kwargs)
            break
        except Exception as exc:
            last_err = exc
    if ds is None:
        raise RuntimeError(f"Could not load GSM8K: {last_err}")
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = format_gsm8k(str(row.get("question") or ""), str(row.get("answer") or ""))
            if not text:
                continue
            f.write(json.dumps({"text": text, "source": "gsm8k"}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= max_samples:
                break
    print(f"Wrote {kept} GSM8K reasoning examples to {out_path}", flush=True)
    return kept


def mix_reason_with_chat(
    gsm_path: str | Path,
    chat_path: str | Path,
    out_path: str | Path,
    chat_copies: int = 2,
) -> int:
    """Keep greetings as chat. GSM8K-only SFT makes every reply start with think."""
    import random

    gsm_path, chat_path, out_path = Path(gsm_path), Path(chat_path), Path(out_path)
    rows: list[str] = []
    for line in gsm_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(line)
    chat_lines = [line for line in chat_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for _ in range(max(1, chat_copies)):
        rows.extend(chat_lines)
    random.Random(42).shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Mixed {len(rows)} reason+chat rows -> {out_path}", flush=True)
    return len(rows)


def prepare_reason_think_mix(
    think_mix_path: str | Path,
    out_path: str | Path,
    chat_keep: int = 8000,
) -> int:
    """Reason tail: all interleaved math/CoT plus a small chat slice so greetings survive."""
    think_mix_path, out_path = Path(think_mix_path), Path(out_path)
    reason_rows: list[str] = []
    chat_rows: list[str] = []
    with think_mix_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = str(rec.get("source") or "")
            if src in _REASON_SOURCES:
                reason_rows.append(line.rstrip("\n"))
            else:
                chat_rows.append(line.rstrip("\n"))
    rng = random.Random(42)
    rng.shuffle(chat_rows)
    keep = chat_rows[: max(0, int(chat_keep))]
    rows = reason_rows + keep
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    print(
        f"Reason-think mix {len(rows)} rows "
        f"({len(reason_rows)} CoT + {len(keep)} chat, "
        f"{100.0 * len(reason_rows) / max(1, len(rows)):.1f}% math) -> {out_path}",
        flush=True,
    )
    return len(rows)


def _write_example(f, text: str, source: str) -> bool:
    if not text:
        return False
    f.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
    return True


def format_dolly(instruction: str, context: str, response: str) -> str:
    inst = _clip(instruction, 400)
    ctx = _clip(context, 700)
    resp = _clip(response, 500)
    if len(inst) < 8 or len(resp) < 12:
        return ""
    user = inst if not ctx else f"{inst}\n\n{ctx}"
    return format_creak_dialog([("user", user), ("assistant", resp)])


def creak_jsonl_ok(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size < 100:
        return False
    with path.open(encoding="utf-8") as f:
        line = f.readline()
    return "<|end|>" in line and "<|system|>" in line and "Creak" in line


def creak_mix_ok(path: str | Path) -> bool:
    path = Path(path)
    if not creak_jsonl_ok(path) or path.stat().st_size < 1_000_000:
        return False
    sources: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 400:
                break
            if not line.strip():
                continue
            try:
                sources.add(str(json.loads(line).get("source") or ""))
            except json.JSONDecodeError:
                continue
    return bool(sources & {"gsm8k", "metamath", "numina"})


def creak_think_mix_ok(path: str | Path) -> bool:
    path = Path(path)
    if not creak_jsonl_ok(path) or path.stat().st_size < 1_000_000:
        return False
    n = 0
    think_asst = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 80:
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(rec.get("text") or "")
            n += 1
            asst = text.split("<|assistant|>", 1)
            if len(asst) == 2 and "<|think|>" in asst[1]:
                think_asst += 1
    return n >= 20 and think_asst >= max(10, n // 2)


def prepare_think_mix(
    src_path: str | Path,
    dest_path: str | Path,
    reason_copies: int = 2,
) -> int:
    """Wrap every assistant turn with interleaved think; upsample math/CoT rows."""
    src_path, dest_path = Path(src_path), Path(dest_path)
    rows: list[str] = []
    counts: dict[str, int] = {}
    with src_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = transform_dialog(str(rec.get("text") or ""))
            if not text or "<|think|>" not in text.split("<|assistant|>", 1)[-1]:
                continue
            rec["text"] = text
            src = str(rec.get("source") or "")
            copies = max(1, reason_copies) if src in _REASON_SOURCES else 1
            encoded = json.dumps(rec, ensure_ascii=False)
            for _ in range(copies):
                rows.append(encoded)
            counts[src] = counts.get(src, 0) + copies
    random.Random(42).shuffle(rows)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    tmp.replace(dest_path)
    reason_n = sum(counts.get(s, 0) for s in _REASON_SOURCES)
    print(
        f"Think mix {len(rows)} rows (~{100.0 * reason_n / max(1, len(rows)):.1f}% math/CoT) "
        f"{counts} -> {dest_path}",
        flush=True,
    )
    return len(rows)


def _copy_jsonl(src: Path, dest_f, extra_source: str | None = None, skip_sources: tuple[str, ...] = ()) -> int:
    n = 0
    if not src.exists():
        return 0
    skip = set(skip_sources)
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src_name = str(rec.get("source") or "")
            if src_name in skip:
                continue
            if extra_source:
                rec["source"] = extra_source
                dest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                dest_f.write(line if line.endswith("\n") else line + "\n")
            n += 1
    return n


def _stream_smoltalk_config(
    config: str,
    max_samples: int,
    dest_f,
    source: str,
    keep_think: int = 0,
    system: str = SYSTEM_PROMPT,
) -> int:
    from datasets import load_dataset

    print(f"Streaming HuggingFaceTB/smoltalk:{config} -> {max_samples}", flush=True)
    ds = load_dataset("HuggingFaceTB/smoltalk", config, split="train", streaming=True)
    kept = 0
    for row in ds:
        text = format_smoltalk_messages(
            row.get("messages") or [],
            keep_think_chars=keep_think,
            system=system,
        )
        if _write_example(dest_f, text, source):
            kept += 1
            if kept >= max_samples:
                break
            if kept % 1000 == 0:
                print(f"  {source} {kept}/{max_samples}", flush=True)
    print(f"  kept {kept} from {source}", flush=True)
    return kept


def format_lima_row(row: dict[str, Any]) -> str:
    conv = row.get("conversations")
    turns: list[tuple[str, str]] = []
    if isinstance(conv, list) and conv:
        if isinstance(conv[0], str):
            for i, text in enumerate(conv):
                role = "user" if i % 2 == 0 else "assistant"
                turns.append((role, _clip(str(text), 1200)))
        else:
            for msg in conv:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("from") or msg.get("role") or "user").lower()
                if role in {"human", "user", "prompter"}:
                    role = "user"
                elif role in {"gpt", "assistant", "bot"}:
                    role = "assistant"
                content = str(msg.get("value") or msg.get("content") or "")
                if content:
                    turns.append((role, _clip(content, 1200)))
    if not turns:
        inst = str(row.get("instruction") or row.get("prompt") or "")
        out = str(row.get("output") or row.get("response") or "")
        if inst and out:
            turns = [("user", _clip(inst, 800)), ("assistant", _clip(out, 1200))]
    if not any(r == "user" for r, _ in turns) or not any(r == "assistant" for r, _ in turns):
        return ""
    return format_creak_dialog(turns)


def prepare_creak_mix(out_path: str | Path, smoltalk_cache: str | Path, gsm_path: str | Path) -> int:
    """
    One mixed Creak corpus: SmolTalk-grade chat + minority GSM8K think + CoT extras.
    Reasoning is shuffled in so <|think|> is not the default first token.
    """
    from datasets import load_dataset

    out_path = Path(out_path)
    smoltalk_cache = Path(smoltalk_cache)
    gsm_path = Path(gsm_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        if creak_jsonl_ok(smoltalk_cache):
            counts["smol-smoltalk"] = _copy_jsonl(smoltalk_cache, f, skip_sources=("seed-qa",))
            print(f"Reused {counts['smol-smoltalk']} cached smol-smoltalk chats", flush=True)
        else:
            prepare_smoltalk(smoltalk_cache, 20000, keep_think_chars=0)
            counts["smol-smoltalk"] = _copy_jsonl(smoltalk_cache, f, skip_sources=("seed-qa",))

        chat_slices = (
            ("everyday-conversations", 3000, "everyday", SYSTEM_PROMPT),
            ("systemchats-30k", 4000, "systemchats", SYSTEM_PROMPT),
            ("smol-rewrite", 2000, "rewrite", SYSTEM_PROMPT),
            ("smol-summarize", 2000, "summarize", SYSTEM_PROMPT),
            ("openhermes-100k", 4000, "openhermes", SYSTEM_PROMPT),
            ("self-oss-instruct", 2000, "code", SYSTEM_PROMPT),
            ("metamathqa-50k", 8000, "metamath", REASON_SYSTEM_PROMPT),
            ("numina-cot-100k", 2500, "numina", REASON_SYSTEM_PROMPT),
        )
        for config, cap, src, system in chat_slices:
            try:
                counts[src] = _stream_smoltalk_config(
                    config, cap, f, src, keep_think=0, system=system
                )
            except Exception as exc:
                print(f"Skip {src}: {exc}", flush=True)
                counts[src] = 0

        try:
            print("Loading HuggingFaceH4/no_robots", flush=True)
            ds = load_dataset("HuggingFaceH4/no_robots", split="train")
            n = 0
            for row in ds:
                text = format_smoltalk_messages(row.get("messages") or [], keep_think_chars=0)
                if _write_example(f, text, "no_robots"):
                    n += 1
            counts["no_robots"] = n
            print(f"  kept {n} no_robots", flush=True)
        except Exception as exc:
            print(f"Skip no_robots: {exc}", flush=True)
            counts["no_robots"] = 0

        try:
            print("Loading GAIR/lima", flush=True)
            ds = load_dataset("GAIR/lima", split="train")
            n = 0
            for row in ds:
                if _write_example(f, format_lima_row(row), "lima"):
                    n += 1
            counts["lima"] = n
            print(f"  kept {n} lima", flush=True)
        except Exception as exc:
            print(f"Skip lima: {exc}", flush=True)
            counts["lima"] = 0

        if not gsm_path.exists() or gsm_path.stat().st_size < 100:
            try:
                prepare_gsm8k(gsm_path)
            except Exception as exc:
                print(f"Skip gsm8k download: {exc}", flush=True)
        counts["gsm8k"] = _copy_jsonl(gsm_path, f)
        print(f"  kept {counts['gsm8k']} gsm8k reasoning chats", flush=True)

        try:
            print("Streaming roneneldan/TinyStories -> 3000", flush=True)
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            n = 0
            for row in ds:
                prompt = STORY_PROMPTS[n % len(STORY_PROMPTS)]
                text = format_story(str(row.get("text") or ""), prompt)
                if _write_example(f, text, "tinystories"):
                    n += 1
                    if n >= 3000:
                        break
            counts["tinystories"] = n
            print(f"  kept {n} tinystories", flush=True)
        except Exception as exc:
            print(f"Skip tinystories: {exc}", flush=True)
            counts["tinystories"] = 0

    append_seed_chats(out_path, repeats=20)
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(42).shuffle(lines)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = len(lines)
    gsm_n = counts.get("gsm8k", 0)
    print(
        f"Creak mix {total} rows (~{100.0 * gsm_n / max(1, total):.1f}% GSM8K-think) {counts} -> {out_path}",
        flush=True,
    )
    return total


def creak_gpu_mix_ok(path: str | Path) -> bool:
    path = Path(path)
    if not creak_jsonl_ok(path) or path.stat().st_size < 4_000_000:
        return False
    sources: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 800:
                break
            if not line.strip():
                continue
            try:
                sources.add(str(json.loads(line).get("source") or ""))
            except json.JSONDecodeError:
                continue
    return bool(sources & {"gsm8k", "metamath", "numina", "orca-math"})


def creak_gpu_think_mix_ok(path: str | Path) -> bool:
    path = Path(path)
    if not creak_think_mix_ok(path) or path.stat().st_size < 4_000_000:
        return False
    return True


def _stream_ultrachat(max_samples: int, dest_f) -> int:
    from datasets import load_dataset

    print(f"Streaming HuggingFaceH4/ultrachat_200k -> {max_samples}", flush=True)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train", streaming=True)
    kept = 0
    for row in ds:
        text = format_smoltalk_messages(row.get("messages") or [], max_chars=8000)
        if _write_example(dest_f, text, "ultrachat"):
            kept += 1
            if kept >= max_samples:
                break
            if kept % 2000 == 0:
                print(f"  ultrachat {kept}/{max_samples}", flush=True)
    print(f"  kept {kept} from ultrachat", flush=True)
    return kept


def _stream_orca_math(max_samples: int, dest_f) -> int:
    from datasets import load_dataset

    print(f"Streaming microsoft/orca-math-word-problems-200k -> {max_samples}", flush=True)
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
    kept = 0
    for row in ds:
        q = str(row.get("question") or row.get("problem") or "").strip()
        a = str(row.get("answer") or row.get("response") or "").strip()
        text = format_creak_dialog(
            [("user", _clip(q, 1200)), ("assistant", a)],
            system=REASON_SYSTEM_PROMPT,
        )
        if _write_example(dest_f, text, "orca-math"):
            kept += 1
            if kept >= max_samples:
                break
            if kept % 2000 == 0:
                print(f"  orca-math {kept}/{max_samples}", flush=True)
    print(f"  kept {kept} from orca-math", flush=True)
    return kept


def prepare_gpu_mix(out_path: str | Path, smoltalk_cache: str | Path, gsm_path: str | Path) -> int:
    """Interleaved-ready mix sized for a 5-hour L40S rental (~160k rows)."""
    from datasets import load_dataset

    out_path = Path(out_path)
    smoltalk_cache = Path(smoltalk_cache)
    gsm_path = Path(gsm_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        if creak_jsonl_ok(smoltalk_cache) and smoltalk_cache.stat().st_size > 1_500_000:
            counts["smol-smoltalk"] = _copy_jsonl(smoltalk_cache, f, skip_sources=("seed-qa",))
            print(f"Reused {counts['smol-smoltalk']} cached smol-smoltalk chats", flush=True)
        else:
            prepare_smoltalk(smoltalk_cache, 40000, keep_think_chars=0)
            counts["smol-smoltalk"] = _copy_jsonl(smoltalk_cache, f, skip_sources=("seed-qa",))

        chat_slices = (
            ("everyday-conversations", 5000, "everyday", SYSTEM_PROMPT),
            ("systemchats-30k", 8000, "systemchats", SYSTEM_PROMPT),
            ("smol-rewrite", 4000, "rewrite", SYSTEM_PROMPT),
            ("smol-summarize", 5000, "summarize", SYSTEM_PROMPT),
            ("openhermes-100k", 20000, "openhermes", SYSTEM_PROMPT),
            ("self-oss-instruct", 5000, "code", SYSTEM_PROMPT),
            ("metamathqa-50k", 20000, "metamath", REASON_SYSTEM_PROMPT),
            ("numina-cot-100k", 25000, "numina", REASON_SYSTEM_PROMPT),
        )
        for config, cap, src, system in chat_slices:
            try:
                counts[src] = _stream_smoltalk_config(
                    config, cap, f, src, keep_think=0, system=system
                )
            except Exception as exc:
                print(f"Skip {src}: {exc}", flush=True)
                counts[src] = 0

        try:
            counts["ultrachat"] = _stream_ultrachat(10000, f)
        except Exception as exc:
            print(f"Skip ultrachat: {exc}", flush=True)
            counts["ultrachat"] = 0

        try:
            counts["orca-math"] = _stream_orca_math(12000, f)
        except Exception as exc:
            print(f"Skip orca-math: {exc}", flush=True)
            counts["orca-math"] = 0

        try:
            print("Loading HuggingFaceH4/no_robots", flush=True)
            ds = load_dataset("HuggingFaceH4/no_robots", split="train")
            n = 0
            for row in ds:
                text = format_smoltalk_messages(row.get("messages") or [], keep_think_chars=0)
                if _write_example(f, text, "no_robots"):
                    n += 1
            counts["no_robots"] = n
            print(f"  kept {n} no_robots", flush=True)
        except Exception as exc:
            print(f"Skip no_robots: {exc}", flush=True)
            counts["no_robots"] = 0

        try:
            print("Loading GAIR/lima", flush=True)
            ds = load_dataset("GAIR/lima", split="train")
            n = 0
            for row in ds:
                if _write_example(f, format_lima_row(row), "lima"):
                    n += 1
            counts["lima"] = n
            print(f"  kept {n} lima", flush=True)
        except Exception as exc:
            print(f"Skip lima: {exc}", flush=True)
            counts["lima"] = 0

        if not gsm_path.exists() or gsm_path.stat().st_size < 100:
            try:
                prepare_gsm8k(gsm_path, max_samples=8000)
            except Exception as exc:
                print(f"Skip gsm8k download: {exc}", flush=True)
        counts["gsm8k"] = _copy_jsonl(gsm_path, f)
        print(f"  kept {counts['gsm8k']} gsm8k reasoning chats", flush=True)

        try:
            print("Streaming roneneldan/TinyStories -> 5000", flush=True)
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            n = 0
            for row in ds:
                prompt = STORY_PROMPTS[n % len(STORY_PROMPTS)]
                text = format_story(str(row.get("text") or ""), prompt)
                if _write_example(f, text, "tinystories"):
                    n += 1
                    if n >= 5000:
                        break
            counts["tinystories"] = n
            print(f"  kept {n} tinystories", flush=True)
        except Exception as exc:
            print(f"Skip tinystories: {exc}", flush=True)
            counts["tinystories"] = 0

    append_seed_chats(out_path, repeats=12)
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(42).shuffle(lines)
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    total = len(lines)
    print(f"GPU mix {total} rows {counts} -> {out_path}", flush=True)
    return total


def pack_token_ids(sequences: Iterator[list[int]], max_len: int, eos_id: int) -> list[torch.Tensor]:
    packs: list[torch.Tensor] = []
    cur: list[int] = []
    n_in = 0
    for ids in sequences:
        n_in += 1
        piece = list(ids)
        if len(piece) < 8:
            continue
        if piece[-1] not in {eos_id}:
            piece.append(eos_id)
        if len(piece) > max_len:
            piece = piece[:max_len]
            piece[-1] = eos_id
        if len(cur) + len(piece) > max_len:
            if len(cur) >= 16:
                packs.append(torch.tensor(cur, dtype=torch.int32))
            cur = piece
        else:
            cur.extend(piece)
    if len(cur) >= 16:
        packs.append(torch.tensor(cur, dtype=torch.int32))
    print(f"Packed {n_in} chats into {len(packs)} x {max_len} sequences", flush=True)
    return packs


def _assistant_labels(
    ids: list[int],
    pad_id: int,
    bos_id: int,
    user_id: int | None,
    asst_id: int | None,
    sys_id: int | None,
    end_id: int | None,
) -> list[int]:
    labels = [-100 if tid == pad_id else tid for tid in ids]
    skip_ids = {i for i in (bos_id, user_id, sys_id, pad_id) if i is not None}
    if asst_id is None:
        return labels
    train = False
    for i, tid in enumerate(ids):
        if tid == asst_id:
            train = True
            labels[i] = -100
            continue
        if tid in skip_ids:
            train = False
            labels[i] = -100
            continue
        if end_id is not None and tid == end_id:
            if train:
                train = False
            else:
                labels[i] = -100
            continue
        if not train:
            labels[i] = -100
    if all(v == -100 for v in labels):
        return [-100 if tid == pad_id else tid for tid in ids]
    return labels


class PackedLM(Dataset):
    """Packed conversations padded to max_len. Loss is assistant tokens only."""

    def __init__(
        self,
        packs: list[torch.Tensor],
        pad_id: int,
        max_len: int,
        bos_id: int,
        user_id: int | None,
        asst_id: int | None,
        sys_id: int | None,
        end_id: int | None = None,
    ) -> None:
        n = len(packs)
        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        labels = torch.full((n, max_len), -100, dtype=torch.long)
        for i, pack in enumerate(packs):
            ids = pack.tolist()[:max_len]
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            input_ids[i] = torch.tensor(ids, dtype=torch.long)
            labels[i] = torch.tensor(
                _assistant_labels(ids, pad_id, bos_id, user_id, asst_id, sys_id, end_id),
                dtype=torch.long,
            )
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


class ChatDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer_path: str | Path, cfg: HCRMConfig) -> None:
        self.cfg = cfg
        self.tok = load_tokenizer(tokenizer_path)
        self.asst_id = self.tok.token_to_id("<|assistant|>")
        self.rows: list[list[int]] = []
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            ids = encode(self.tok, text, cfg.max_seq_len, keep_tail=False)
            if len(ids) < 8:
                continue
            if self.asst_id is not None and self.asst_id not in ids:
                continue
            self.rows.append(ids)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.rows[idx]
        pad = self.cfg.pad_id
        max_len = self.cfg.max_seq_len
        if len(ids) < max_len:
            ids = ids + [pad] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == pad] = -100
        if self.asst_id is not None:
            hits = (input_ids == self.asst_id).nonzero(as_tuple=False)
            if len(hits) > 0:
                labels[: int(hits[0]) + 1] = -100
        if int((labels != -100).sum()) == 0:
            labels = input_ids.clone()
            labels[labels == pad] = -100
        return {"input_ids": input_ids, "labels": labels}
