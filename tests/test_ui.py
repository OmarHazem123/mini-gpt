"""Tests for the local web UI server (``mini_gpt.ui``)."""

from __future__ import annotations

import dataclasses
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mini_gpt.model import MiniGPT, MiniGPTConfig
from mini_gpt.sample import generate_text, load_model
from mini_gpt.ui import UIServer, create_server
from mini_gpt.utils import CharTokenizer, save_checkpoint
from test_training import CORPUS

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never use a proxy


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small random-weight checkpoint in the format ``mini_gpt.train`` writes."""
    tokenizer = CharTokenizer.from_text(CORPUS)
    model = MiniGPT(MiniGPTConfig(vocab_size=tokenizer.vocab_size, block_size=16, n_layer=2,
                                  n_head=2, n_embd=32, dropout=0.0))
    path = tmp_path_factory.mktemp("ui") / "best.pt"
    save_checkpoint(path, model=model, iter_num=7, best_val_loss=2.5, tokenizer=tokenizer,
                    config={"model": dataclasses.asdict(model.config)})
    return path


@pytest.fixture(scope="module")
def server(checkpoint: Path) -> Iterator[UIServer]:
    srv = create_server(checkpoint, port=0, device="cpu")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def get(srv: UIServer, path: str) -> tuple[int, str, bytes]:
    try:
        with _OPENER.open(srv.url + path, timeout=30) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def post(srv: UIServer, payload: Any, content_type: str = "application/json") -> tuple[int, bytes]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(srv.url + "/api/generate", data=data, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with _OPENER.open(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_serves_page_and_model_info(server: UIServer) -> None:
    status, ctype, body = get(server, "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"<title>mini-gpt playground</title>" in body

    status, ctype, body = get(server, "/api/info")
    assert status == 200 and ctype == "application/json"
    info = json.loads(body)
    assert info["iter"] == 7 and info["best_val_loss"] == 2.5
    assert info["vocab"] == "".join(sorted(set(CORPUS)))
    assert info["device"] == "cpu" and info["checkpoint"].endswith("best.pt")

    assert get(server, "/missing")[0] == 404


def test_streamed_text_matches_generate_text(server: UIServer, checkpoint: Path) -> None:
    model, tokenizer, _ = load_model(checkpoint)

    status, body = post(server, {"prompt": "the ", "tokens": 40, "temperature": 0.8,
                                 "top_k": 5, "seed": 3})
    assert status == 200
    expected = generate_text(model, tokenizer, "the ", 40, temperature=0.8, top_k=5, seed=3)
    assert body.decode() == expected[len("the "):]  # only the continuation is streamed

    # An empty prompt starts from token 0, like the CLI's default.
    status, body = post(server, {"prompt": "", "tokens": 25, "seed": 1})
    assert status == 200
    assert body.decode() == generate_text(model, tokenizer, None, 25, seed=1)[1:]


@pytest.mark.parametrize(("payload", "status", "message"), [
    ({"prompt": "Zebra"}, 400, "not in the vocabulary"),
    ({"tokens": 0}, 400, "tokens must be"),
    ({"tokens": 10**6}, 400, "tokens must be"),
    ({"tokens": True}, 400, "tokens must be"),
    ({"temperature": -1}, 400, "temperature must be"),
    ({"top_k": 0}, 400, "top_k must be"),
    ({"seed": "7"}, 400, "seed must be"),
    ({"prompt": 5}, 400, "prompt must be a string"),
    ([1, 2], 400, "JSON object"),
    (b"not json", 400, "must be JSON"),
])
def test_bad_requests_get_clear_errors(
    server: UIServer, payload: Any, status: int, message: str
) -> None:
    got_status, body = post(server, payload)
    assert got_status == status
    assert message in json.loads(body)["error"]


def test_rejects_non_json_oversized_and_foreign_host_requests(server: UIServer) -> None:
    assert post(server, {"tokens": 5}, content_type="text/plain")[0] == 415

    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.putrequest("POST", "/api/generate")  # announce a huge body but send none
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(10**6))
    conn.endheaders()
    assert conn.getresponse().status == 413
    conn.close()

    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/api/info", headers={"Host": "attacker.example"})
    assert conn.getresponse().status == 403  # DNS-rebinding guard
    conn.close()


def test_busy_model_rejects_instead_of_queueing(server: UIServer) -> None:
    assert server.lock.acquire(timeout=30)
    try:
        status, body = post(server, {"tokens": 5})
        assert status == 409 and "busy" in json.loads(body)["error"]
    finally:
        server.lock.release()
    assert post(server, {"tokens": 5})[0] == 200


def test_client_disconnect_ends_generation_and_frees_the_model(server: UIServer) -> None:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("POST", "/api/generate", body=json.dumps({"tokens": 2000}),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200 and len(resp.read(5)) == 5
    resp.close()
    conn.close()  # what the page's Stop button does

    deadline = time.monotonic() + 30
    while not server.lock.acquire(blocking=False):
        assert time.monotonic() < deadline, "generation lock was never released"
        time.sleep(0.05)
    server.lock.release()
    assert post(server, {"tokens": 5})[0] == 200
