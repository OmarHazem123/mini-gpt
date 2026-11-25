"""Local web UI for trying a trained Mini-GPT in the browser.

Usage::

    python -m mini_gpt.ui                    # serves checkpoints/best.pt on http://127.0.0.1:8000
    python -m mini_gpt.ui --open             # ...and opens the browser
    python -m mini_gpt.ui --checkpoint checkpoints/last.pt --port 8080

Standard library only (``http.server``); the page in ``ui.html`` has no external
dependencies, so it works offline. Endpoints:

* ``GET  /``             the page.
* ``GET  /api/info``     checkpoint and model details as JSON.
* ``POST /api/generate`` JSON ``{prompt, tokens, temperature, top_k, seed}``. Streams the
  continuation as UTF-8 text, one character per write. Bad input gets a JSON ``{"error"}``.

One generation runs at a time. A second request gets ``409`` instead of queueing, and a
client that disconnects (the page's Stop button) ends its generation immediately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import webbrowser
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from mini_gpt.model import MiniGPT
from mini_gpt.sample import DEFAULT_CHECKPOINT, load_model, stream_text
from mini_gpt.utils import CharTokenizer, resolve_device

MAX_BODY_BYTES = 16 * 1024
MAX_PROMPT_CHARS = 2000
MAX_TOKENS = 2000
MAX_TEMPERATURE = 5.0
MAX_SEED = 2**32 - 1
DEFAULTS: dict[str, Any] = {"tokens": 500, "temperature": 0.8, "top_k": None, "seed": 1337}
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class BadRequestError(ValueError):
    """A generation request the server rejects; ``status`` is the HTTP status to send."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def _int_field(data: dict[str, Any], name: str, low: int, high: int) -> int:
    value = data.get(name, DEFAULTS[name])
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise BadRequestError(f"{name} must be a whole number from {low} to {high}")
    return value


def parse_generate_request(body: bytes, vocab_size: int) -> dict[str, Any]:
    """Validate a ``/api/generate`` JSON body and return keyword arguments for sampling.

    An empty or missing prompt means "start from a blank line" (``prompt=None``).
    Characters outside the vocabulary are reported later, by ``stream_text``.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BadRequestError("Request body must be JSON") from None
    if not isinstance(data, dict):
        raise BadRequestError("Request body must be a JSON object")

    prompt = data.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise BadRequestError("prompt must be a string")
    if prompt is not None and len(prompt) > MAX_PROMPT_CHARS:
        raise BadRequestError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")

    temperature = data.get("temperature", DEFAULTS["temperature"])
    if (isinstance(temperature, bool) or not isinstance(temperature, int | float)
            or not math.isfinite(temperature) or not 0 <= temperature <= MAX_TEMPERATURE):
        raise BadRequestError(f"temperature must be a number from 0 to {MAX_TEMPERATURE:g}")

    top_k = data.get("top_k")
    if top_k is not None:
        top_k = _int_field(data, "top_k", 1, vocab_size)

    return {
        "prompt": prompt or None,
        "max_new_tokens": _int_field(data, "tokens", 1, MAX_TOKENS),
        "temperature": float(temperature),
        "top_k": top_k,
        "seed": _int_field(data, "seed", 0, MAX_SEED),
    }


def _page() -> bytes:
    return resources.files("mini_gpt").joinpath("ui.html").read_bytes()


class UIServer(ThreadingHTTPServer):
    """HTTP server that owns one loaded model and serialises generation on it."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        model: MiniGPT,
        tokenizer: CharTokenizer,
        info: dict[str, Any],
    ) -> None:
        super().__init__(address, _Handler)
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.page = _page()
        self.lock = threading.Lock()  # one generation at a time

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"


def create_server(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "auto",
) -> UIServer:
    """Load ``checkpoint`` and bind the UI server (``port=0`` picks a free port)."""
    dev = resolve_device(device)
    model, tokenizer, ckpt = load_model(checkpoint, dev)
    cfg = model.config
    best = float(ckpt.get("best_val_loss", math.inf))
    info = {
        "checkpoint": Path(checkpoint).as_posix(),
        "iter": int(ckpt["iter_num"]),
        "best_val_loss": best if math.isfinite(best) else None,
        "params": model.num_parameters(False),
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "block_size": cfg.block_size,
        "vocab": "".join(tokenizer.chars),
        "device": str(dev),
        "limits": {"tokens": MAX_TOKENS, "prompt_chars": MAX_PROMPT_CHARS,
                   "temperature": MAX_TEMPERATURE, "seed": MAX_SEED},
        "defaults": DEFAULTS,
    }
    return UIServer((host, port), model, tokenizer, info)


class _Handler(BaseHTTPRequestHandler):
    server: UIServer
    server_version = "mini-gpt-ui"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the console quiet; errors are reported to the client

    # ---- helpers ----
    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, obj: Any) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def _host_allowed(self) -> bool:
        """When bound to loopback, only answer requests addressed to loopback.

        This blocks DNS-rebinding pages from talking to the server through another name.
        """
        if self.server.server_address[0] not in ("127.0.0.1", "::1"):
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in _LOOPBACK

    # ---- routes ----
    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        if not self._host_allowed():
            return self._send_json(HTTPStatus.FORBIDDEN, {"error": "Unexpected Host header"})
        if self.path == "/":
            return self._send(HTTPStatus.OK, self.server.page, "text/html; charset=utf-8")
        if self.path == "/api/info":
            return self._send_json(HTTPStatus.OK, self.server.info)
        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib name)
        if not self._host_allowed():
            return self._send_json(HTTPStatus.FORBIDDEN, {"error": "Unexpected Host header"})
        if self.path != "/api/generate":
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        try:
            kwargs = self._read_request()
        except BadRequestError as e:
            return self._send_json(e.status, {"error": str(e)})

        if not self.server.lock.acquire(blocking=False):
            return self._send_json(HTTPStatus.CONFLICT,
                                   {"error": "The model is busy with another generation"})
        try:
            try:
                chars = stream_text(self.server.model, self.server.tokenizer, **kwargs)
            except ValueError as e:  # e.g. a prompt character outside the vocabulary
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            # No Content-Length: the body is streamed and the connection closes at the end.
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            for ch in chars:  # model.generate runs under torch.no_grad()
                self.wfile.write(ch.encode("utf-8"))
                self.wfile.flush()
        except ConnectionError:
            pass  # the client stopped reading (Stop button or closed tab); end quietly
        finally:
            self.server.lock.release()

    def _read_request(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            raise BadRequestError("Content-Length is required",
                                  HTTPStatus.LENGTH_REQUIRED) from None
        if not 0 <= length <= MAX_BODY_BYTES:
            raise BadRequestError("Request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        # Read the body before any other rejection: closing a socket with unread data makes
        # Windows send a reset, and the client may then lose the error response.
        body = self.rfile.read(length)
        # Requiring JSON makes cross-site requests need a CORS preflight, which this
        # server never grants, so other websites cannot start generations.
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if content_type != "application/json":
            raise BadRequestError("Content-Type must be application/json",
                             HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        return parse_generate_request(body, self.server.tokenizer.vocab_size)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try a trained Mini-GPT in your browser.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="checkpoint to load (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind (default: %(default)s, this computer only)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: %(default)s)")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | cuda:N (default: %(default)s)")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).is_file():
        print(f"error: checkpoint {args.checkpoint} not found "
              f"(train first: python tasks.py train)", file=sys.stderr)
        return 1
    try:
        server = create_server(args.checkpoint, host=args.host, port=args.port,
                               device=args.device)
    except OSError as e:
        print(f"error: cannot listen on {args.host}:{args.port} ({e.strerror or e}); "
              f"try another --port", file=sys.stderr)
        return 1
    info = server.info
    print(f"Mini-GPT UI: {server.url}  ({info['checkpoint']}, iter {info['iter']}, "
          f"{info['device']})  Press Ctrl+C to stop.", flush=True)
    if args.host not in _LOOPBACK:
        print("warning: listening beyond this computer; anyone on the network can use it",
              file=sys.stderr)
    if args.open:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
