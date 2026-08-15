from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import argparse
import base64
import hashlib
import json
import re
import socket
import struct
import sys
import threading
import time
import webbrowser

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
HTML_PATH = Path(__file__).with_name("dash.html")

STEP_RE = re.compile(
    r"(?P<phase>\w+) step=(?P<step>\d+) loss=(?P<loss>[\d.]+) ppl=(?P<ppl>[\d.]+) "
    r"lr=(?P<lr>[\d.eE+-]+) tau=(?P<tau>[\d.]+) tps=(?P<tps>[\d.]+) "
    r"left=(?P<left>[\d.]+)m rss=(?P<rss>[\d.]+)GB u=(?P<u>[\d.]+) "
    r"channelite=(?P<cite>\d+) channels\[(?P<ch>[^\]]*)\]"
)
BEGIN_RE = re.compile(
    r"BEGIN (?P<phase>\w+)\s+params=(?P<params>[0-9,]+)\s+packs=(?P<packs>\d+)\s+"
    r"seq=(?P<seq>\d+)\s+batch=(?P<batch>\S+).*?"
    r"until=(?P<until>\S+)\s+left=(?P<left>[\d.]+)m\s+rss=(?P<rss>[\d.]+)GB"
)
PACKED_RE = re.compile(r"Packed (?P<chats>\d+) chats into (?P<packs>\d+) x (?P<seq>\d+) sequences")
END_RE = re.compile(r"END (?P<phase>\w+)\s+steps=(?P<steps>\d+)\s+elapsed=(?P<elapsed>[\d.]+)m")


def _f(d: dict, key: str) -> float:
    return float(d[key])


class RunState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.train_start = ""
        self.chat_until = ""
        self.reason_until = ""
        self.threads = 0
        self.device = ""
        self.amp = ""
        self.resume = None
        self.skip_reason = None
        self.phase = ""
        self.params = 0
        self.packs = 0
        self.seq = 0
        self.batch = ""
        self.batch_size = 4
        self.grad_accum = 1
        self.until = ""
        self.left0 = 0.0
        self.rss0 = 0.0
        self.chats_packed = 0
        self.points: list[dict] = []
        self.last_lines: list[str] = []
        self.status = "idle"
        self.end: dict | None = None
        self.mix = ""
        self.updated = time.time()

    def note_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        self.last_lines.append(text)
        self.last_lines = self.last_lines[-12:]

    def ingest(self, line: str) -> str | None:
        raw = line.rstrip("\n")
        if not raw.strip():
            return None
        self.note_line(raw)
        self.updated = time.time()

        if raw.startswith("Creak train start"):
            self.reset()
            self.note_line(raw)
            self.status = "starting"
            parts = {}
            for chunk in raw.split("  "):
                if "=" in chunk:
                    k, v = chunk.split("=", 1)
                    parts[k.strip()] = v.strip()
                elif chunk.startswith("Creak train start "):
                    self.train_start = chunk.split(" ", 3)[-1]
            self.chat_until = parts.get("chat_until", "")
            self.reason_until = parts.get("reason_until", "")
            self.threads = int(parts.get("threads") or 0)
            self.device = parts.get("device", "")
            self.amp = parts.get("amp", "")
            if "resume" in parts:
                self.resume = parts["resume"].lower() == "true"
            if "skip_reason" in parts:
                self.skip_reason = parts["skip_reason"].lower() == "true"
            return "reset"

        if raw.startswith("Creak mix ") or raw.startswith("GPU mix "):
            self.mix = raw
            return "meta"

        packed = PACKED_RE.search(raw)
        if packed:
            self.chats_packed = int(packed.group("chats"))
            self.packs = int(packed.group("packs"))
            self.seq = int(packed.group("seq"))
            return "meta"

        begin = BEGIN_RE.search(raw)
        if begin:
            g = begin.groupdict()
            self.phase = g["phase"]
            self.params = int(g["params"].replace(",", ""))
            self.packs = int(g["packs"])
            self.seq = int(g["seq"])
            self.batch = g["batch"]
            if "x" in self.batch:
                a, b = self.batch.split("x", 1)
                self.batch_size = int(a)
                self.grad_accum = int(b)
            self.until = g["until"]
            self.left0 = float(g["left"])
            self.rss0 = float(g["rss"])
            self.status = "running"
            self.end = None
            self.points = []
            return "begin"

        step = STEP_RE.search(raw)
        if step:
            g = step.groupdict()
            pt = {
                "phase": g["phase"],
                "step": int(g["step"]),
                "loss": _f(g, "loss"),
                "ppl": _f(g, "ppl"),
                "lr": _f(g, "lr"),
                "tau": _f(g, "tau"),
                "tps": _f(g, "tps"),
                "left": _f(g, "left"),
                "rss": _f(g, "rss"),
                "u": _f(g, "u"),
                "cite": int(g["cite"]),
                "ch": g["ch"],
            }
            self.points.append(pt)
            self.phase = pt["phase"]
            self.status = "running"
            return "step"

        end = END_RE.search(raw)
        if end:
            self.end = {
                "phase": end.group("phase"),
                "steps": int(end.group("steps")),
                "elapsed": float(end.group("elapsed")),
            }
            self.status = "done"
            return "end"

        if raw.startswith("Done."):
            if self.status == "running":
                self.status = "done"
            return "end"
        return "log"

    def snapshot(self, ckpt: Path, cfg: Path) -> dict:
        last = self.points[-1] if self.points else None
        steps_per_epoch = 0
        if self.packs and self.batch_size:
            steps_per_epoch = max(1, self.packs // max(1, self.batch_size))
        warmup = max(1, int(steps_per_epoch * 64 * 0.03)) if steps_per_epoch else 18914
        cites = [0, 0, 0, 0]
        ch_top = {}
        for p in self.points:
            c = p["cite"]
            if 0 <= c < 4:
                cites[c] += 1
        for p in self.points[-80:]:
            top = (p["ch"].split(",")[0].split("=")[0].strip() if p["ch"] else "")
            if top:
                ch_top[top] = ch_top.get(top, 0) + 1
        losses = [p["loss"] for p in self.points]
        best = None
        if losses:
            i = min(range(len(losses)), key=lambda j: losses[j])
            best = {"step": self.points[i]["step"], "loss": losses[i], "ppl": self.points[i]["ppl"]}
        ckpt_info = None
        if ckpt.exists():
            st = ckpt.stat()
            ckpt_info = {"path": str(ckpt), "bytes": st.st_size, "mtime": st.st_mtime}
        config = None
        if cfg.exists():
            try:
                config = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                config = None
        stale = (time.time() - self.updated) > 45
        status = self.status
        if status == "running" and stale:
            status = "stale"
        return {
            "status": status,
            "train_start": self.train_start,
            "chat_until": self.chat_until,
            "reason_until": self.reason_until,
            "threads": self.threads,
            "device": self.device,
            "amp": self.amp,
            "resume": self.resume,
            "skip_reason": self.skip_reason,
            "phase": self.phase,
            "params": self.params,
            "packs": self.packs,
            "seq": self.seq,
            "batch": self.batch,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "until": self.until,
            "left0": self.left0,
            "rss0": self.rss0,
            "chats_packed": self.chats_packed,
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup,
            "points": self.points,
            "last": last,
            "best": best,
            "cites": cites,
            "ch_top": ch_top,
            "end": self.end,
            "mix": self.mix,
            "lines": self.last_lines,
            "ckpt": ckpt_info,
            "config": config,
            "ts": time.time(),
        }


class Hub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: list[socket.socket] = []
        self.state = RunState()
        self.stop = False

    def add(self, sock: socket.socket) -> None:
        with self.lock:
            self.clients.append(sock)

    def drop(self, sock: socket.socket) -> None:
        with self.lock:
            if sock in self.clients:
                self.clients.remove(sock)
        try:
            sock.close()
        except OSError:
            pass

    def broadcast(self, payload: dict) -> None:
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        frame = ws_frame(blob)
        dead: list[socket.socket] = []
        with self.lock:
            clients = list(self.clients)
        for sock in clients:
            try:
                sock.sendall(frame)
            except OSError:
                dead.append(sock)
        for sock in dead:
            self.drop(sock)


HUB = Hub()
PATHS = {"log": Path("checkpoints/train.log"), "ckpt": Path("checkpoints/hcrm_slm.pt"), "cfg": Path("checkpoints/config.json")}


def ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([n])
    elif n < 65536:
        header += bytes([126]) + struct.pack("!H", n)
    else:
        header += bytes([127]) + struct.pack("!Q", n)
    return header + payload


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf.extend(chunk)
    return bytes(buf)


def read_ws_frame(sock: socket.socket) -> tuple[int, bytes]:
    hdr = recv_exact(sock, 2)
    opcode = hdr[0] & 0x0F
    masked = bool(hdr[1] & 0x80)
    length = hdr[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(sock, 8))[0]
    mask = recv_exact(sock, 4) if masked else b""
    data = bytearray(recv_exact(sock, length))
    if masked:
        for i in range(len(data)):
            data[i] ^= mask[i % 4]
    return opcode, bytes(data)


def accept_ws(handler: BaseHTTPRequestHandler) -> None:
    key = handler.headers.get("Sec-WebSocket-Key", "")
    digest = hashlib.sha1((key + GUID).encode("ascii")).digest()
    accept = base64.b64encode(digest).decode("ascii")
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()
    try:
        handler.wfile.flush()
    except Exception:
        pass
    sock = handler.connection
    sock.settimeout(2.0)
    HUB.add(sock)
    hello = {"type": "hello", "state": HUB.state.snapshot(PATHS["ckpt"], PATHS["cfg"])}
    try:
        sock.sendall(ws_frame(json.dumps(hello).encode("utf-8")))
    except OSError:
        HUB.drop(sock)
        return
    try:
        while not HUB.stop:
            try:
                opcode, data = read_ws_frame(sock)
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                break
            if opcode == 0x8:
                break
            if opcode == 0x9:
                try:
                    sock.sendall(ws_frame(data, opcode=0xA))
                except OSError:
                    break
    finally:
        HUB.drop(sock)


class DashHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[dash] " + (fmt % args) + "\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/ws", "/stream"} or self.headers.get("Upgrade", "").lower() == "websocket":
            accept_ws(self)
            return
        if parsed.path in {"/", "/index.html", "/dash"}:
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/state":
            blob = json.dumps(HUB.state.snapshot(PATHS["ckpt"], PATHS["cfg"])).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)


def tail_log(path: Path, start_pos: int) -> None:
    pos = start_pos
    last_broadcast = 0.0
    pending = None
    while not HUB.stop:
        try:
            if not path.exists():
                time.sleep(0.3)
                continue
            size = path.stat().st_size
            if size < pos:
                HUB.state.reset()
                pos = 0
            if size > pos:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                kind = None
                new_pts: list[dict] = []
                reset = False
                for line in chunk.splitlines():
                    kind = HUB.state.ingest(line)
                    if kind == "reset" or kind == "begin":
                        reset = True
                        new_pts = []
                    elif kind == "step" and HUB.state.points:
                        new_pts.append(HUB.state.points[-1])
                if reset:
                    pending = {"type": "hello", "state": HUB.state.snapshot(PATHS["ckpt"], PATHS["cfg"])}
                elif new_pts:
                    snap = HUB.state.snapshot(PATHS["ckpt"], PATHS["cfg"])
                    pending = {
                        "type": "steps",
                        "points": new_pts,
                        "status": snap["status"],
                        "last": snap["last"],
                        "n": len(HUB.state.points),
                        "best": snap["best"],
                        "cites": snap["cites"],
                        "ch_top": snap["ch_top"],
                        "ckpt": snap["ckpt"],
                        "lines": snap["lines"],
                    }
                elif kind in {"end", "meta"}:
                    pending = {"type": "hello", "state": HUB.state.snapshot(PATHS["ckpt"], PATHS["cfg"])}
            now = time.time()
            if pending and now - last_broadcast >= 0.2:
                HUB.broadcast(pending)
                pending = None
                last_broadcast = now
        except Exception as exc:
            sys.stderr.write(f"[dash] tail error: {exc}\n")
        time.sleep(0.15)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Creak training dashboard (WebSocket).")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--log", default="checkpoints/train.log")
    p.add_argument("--ckpt", default="checkpoints/hcrm_slm.pt")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not HTML_PATH.exists():
        raise SystemExit(f"missing {HTML_PATH}")
    PATHS["log"] = Path(args.log)
    PATHS["ckpt"] = Path(args.ckpt)
    PATHS["cfg"] = Path(args.ckpt).with_name("config.json")
    start_pos = 0
    if PATHS["log"].exists():
        text = PATHS["log"].read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            HUB.state.ingest(line)
        start_pos = PATHS["log"].stat().st_size
        print(f"  parsed {len(HUB.state.points)} log windows from {PATHS['log']}", flush=True)
    t = threading.Thread(target=tail_log, args=(PATHS["log"], start_pos), daemon=True)
    t.start()
    httpd = ThreadingHTTPServer((args.host, args.port), DashHandler)
    local = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    bind_url = f"http://{args.host}:{args.port}/"
    open_url = f"http://{local}:{args.port}/"
    print(f"Creak live dash  {bind_url}", flush=True)
    print(f"  local {open_url}  ws ws://{local}:{args.port}/ws", flush=True)
    print(f"  log {PATHS['log']}", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(open_url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping dash", flush=True)
    finally:
        HUB.stop = True
        httpd.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
