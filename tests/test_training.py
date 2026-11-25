"""Tests for the training loop and utilities: LR schedule, config, tokenizer, batching,
gradient accumulation, NaN/Inf handling, clipping, checkpoints, seeding, end-to-end runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from mini_gpt.model import MiniGPT, MiniGPTConfig
from mini_gpt.train import (
    Config,
    OptimizerConfig,
    RunDirectoryError,
    accumulate_gradients,
    build_optimizer,
    get_lr,
    optimizer_step,
    parse_args,
    train,
)
from mini_gpt.utils import (
    CharTokenizer,
    Precision,
    apply_overrides,
    get_batch,
    has_tensor_cores,
    load_checkpoint,
    load_config,
    resolve_amp_dtype,
    save_checkpoint,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 32
CPU_FP32 = Precision(torch.device("cpu"), torch.float32)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def tiny_model(dropout: float = 0.0) -> MiniGPT:
    return MiniGPT(MiniGPTConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2,
                                 n_embd=32, dropout=dropout))


def random_batch(batch_size: int = 8, seq_len: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.randint(0, VOCAB, (batch_size, seq_len)),
            torch.randint(0, VOCAB, (batch_size, seq_len)))


# ======================================================================================
# Learning-rate schedule
# ======================================================================================

LR_KW: dict[str, Any] = dict(max_lr=1e-3, min_lr=1e-4, warmup_iters=10, lr_decay_iters=100)


def test_lr_warmup_is_linear_and_reaches_peak() -> None:
    assert get_lr(0, **LR_KW) == pytest.approx(1e-4)
    assert get_lr(4, **LR_KW) == pytest.approx(5e-4)
    assert get_lr(9, **LR_KW) == pytest.approx(1e-3)


def test_lr_cosine_decay_is_monotonic_and_ends_at_min() -> None:
    lrs = [get_lr(i, **LR_KW) for i in range(10, 101)]
    assert lrs[0] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))
    assert get_lr(55, **LR_KW) == pytest.approx((1e-3 + 1e-4) / 2)  # cosine midpoint
    assert get_lr(100, **LR_KW) == pytest.approx(1e-4)
    assert get_lr(10_000, **LR_KW) == pytest.approx(1e-4)


# ======================================================================================
# Config
# ======================================================================================


def default_raw() -> dict[str, Any]:
    return load_config(ROOT / "configs" / "default.yaml")


def test_default_config_is_valid() -> None:
    cfg = Config.from_dict(default_raw())
    assert cfg.lr_decay_iters == cfg.training.max_iters  # lr_decay_iters: null
    assert cfg.training.batch_size * cfg.training.grad_accum_steps == 64
    assert cfg.training.amp_dtype == "float32"
    assert isinstance(cfg.optimizer.lr, float)


# ======================================================================================
# Precision selection
# ======================================================================================


def fake_gpu(monkeypatch: pytest.MonkeyPatch, name: str, capability: tuple[int, int]) -> None:
    """Make torch report a given CUDA device (works on machines without a GPU)."""
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)


@pytest.mark.parametrize(
    ("name", "capability", "expected"),
    [
        ("NVIDIA A100-SXM4-40GB", (8, 0), torch.bfloat16),
        ("NVIDIA GeForce RTX 4090", (8, 9), torch.bfloat16),
        ("NVIDIA GeForce RTX 2080 Ti", (7, 5), torch.float16),
        ("Tesla T4", (7, 5), torch.float16),
        ("Tesla V100-SXM2-16GB", (7, 0), torch.float16),
        ("NVIDIA GeForce GTX 1650 Ti", (7, 5), torch.float32),
        ("NVIDIA GeForce GTX 1660 SUPER", (7, 5), torch.float32),
        ("NVIDIA GeForce MX450", (7, 5), torch.float32),
        ("NVIDIA T1000", (7, 5), torch.float32),
        ("NVIDIA GeForce GTX 1080 Ti", (6, 1), torch.float32),
    ],
)
def test_auto_precision_follows_tensor_core_support(
    monkeypatch: pytest.MonkeyPatch, name: str, capability: tuple[int, int],
    expected: torch.dtype,
) -> None:
    fake_gpu(monkeypatch, name, capability)
    device = torch.device("cuda")
    assert resolve_amp_dtype("auto", device) == expected
    assert has_tensor_cores(device) == (expected != torch.float32)


def test_auto_precision_on_cpu_is_float32() -> None:
    assert resolve_amp_dtype("auto", torch.device("cpu")) == torch.float32
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_amp_dtype("float16", torch.device("cpu"))


def test_explicit_float16_without_tensor_cores_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_gpu(monkeypatch, "NVIDIA GeForce GTX 1650 Ti", (7, 5))
    with caplog.at_level("WARNING", logger="mini_gpt.utils"):
        assert resolve_amp_dtype("float16", torch.device("cuda")) == torch.float16
    assert "no tensor cores" in caplog.text


def test_config_rejects_unknown_keys() -> None:
    raw = default_raw()
    raw["training"]["batchsize"] = 8
    with pytest.raises(ValueError, match="batchsize"):
        Config.from_dict(raw)


def test_yaml_style_string_float_gets_helpful_error() -> None:
    raw = default_raw()
    raw["optimizer"]["lr"] = "1e-3"  # what PyYAML produces for `lr: 1e-3`
    with pytest.raises(ValueError, match=r"1\.0e-3"):
        Config.from_dict(raw)


def test_overrides_are_typed() -> None:
    raw = default_raw()
    apply_overrides(raw, ["training.max_iters=7", "optimizer.lr=3e-4",
                          "training.device=cpu", "scheduler.lr_decay_iters=null"])
    assert raw["training"]["max_iters"] == 7
    assert raw["optimizer"]["lr"] == pytest.approx(3e-4)
    assert raw["training"]["device"] == "cpu"
    assert raw["scheduler"]["lr_decay_iters"] is None
    with pytest.raises(KeyError):
        apply_overrides(raw, ["training.max_iter=7"])


def test_warmup_must_be_shorter_than_decay_horizon() -> None:
    raw = default_raw()
    apply_overrides(raw, ["training.max_iters=50"])  # warmup_iters is 100
    with pytest.raises(ValueError, match="warmup_iters"):
        Config.from_dict(raw)


# ======================================================================================
# Tokenizer and batching
# ======================================================================================


def test_tokenizer_roundtrip_and_state_dict() -> None:
    text = "hello world\n"
    tok = CharTokenizer.from_text(text)
    assert tok.chars == sorted(set(text))
    assert tok.decode(tok.encode(text)) == text
    assert CharTokenizer.from_state_dict(tok.state_dict()).chars == tok.chars


def test_tokenizer_rejects_unknown_characters() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="'z'"):
        tok.encode("abz")
    with pytest.raises(ValueError):
        tok.decode([3])


def test_get_batch_targets_are_inputs_shifted_by_one() -> None:
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)  # data is arange, so "next token" means +1
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)  # windows are contiguous
    assert int(y.max()) <= 99


# ======================================================================================
# Gradient accumulation, NaN/Inf handling, clipping
# ======================================================================================


def test_gradient_accumulation_matches_full_batch() -> None:
    """Two micro-batches of 4 must give the same gradients as one batch of 8."""
    torch.manual_seed(0)
    model = tiny_model().double()
    scaler = CPU_FP32.grad_scaler()
    x, y = random_batch(8)

    accumulate_gradients(model, [(x, y)], CPU_FP32, scaler)
    full = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    loss = accumulate_gradients(model, [(x[:4], y[:4]), (x[4:], y[4:])], CPU_FP32, scaler)

    assert loss.ndim == 0 and torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None
        torch.testing.assert_close(p.grad, full[name], rtol=1e-12, atol=1e-12, msg=name)


def test_nonfinite_gradient_skips_step_and_names_parameter() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig())
    scaler = CPU_FP32.grad_scaler()
    accumulate_gradients(model, [random_batch()], CPU_FP32, scaler)

    grad = model.blocks[0].attn.c_attn.weight.grad
    assert grad is not None
    grad[0, 0] = float("nan")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    stats = optimizer_step(model, optimizer, scaler, grad_clip=1.0)

    assert stats.skipped and math.isnan(stats.grad_norm)
    assert stats.nonfinite_params == ("blocks.0.attn.c_attn.weight",)
    assert not optimizer.state  # AdamW never stepped
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]), f"{name} changed despite the skipped step"


def test_gradients_are_clipped_to_grad_clip() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig())
    scaler = CPU_FP32.grad_scaler()
    accumulate_gradients(model, [random_batch()], CPU_FP32, scaler)
    for p in model.parameters():
        assert p.grad is not None
        p.grad.mul_(1e3)

    stats = optimizer_step(model, optimizer, scaler, grad_clip=0.5)

    post = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()
                     if p.grad is not None]))
    assert stats.clipped and not stats.skipped and stats.grad_norm > 0.5
    assert post.item() == pytest.approx(0.5, rel=1e-4)


@requires_cuda
def test_fp16_grad_scaler_skips_overflow_then_recovers() -> None:
    """With float16 the scaler must skip an overflowing step, halve the loss scale, and
    let the next clean step through."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    precision = Precision(device, torch.float16)
    model = tiny_model().to(device)
    optimizer = build_optimizer(model, OptimizerConfig())
    scaler = precision.grad_scaler()
    x, y = (t.to(device) for t in random_batch())

    accumulate_gradients(model, [(x, y)], precision, scaler)
    grad = model.lm_head.weight.grad
    assert grad is not None
    grad[0, 0] = float("inf")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    scale_before = scaler.get_scale()

    stats = optimizer_step(model, optimizer, scaler, grad_clip=1.0)
    assert stats.skipped and "wte.weight" in stats.nonfinite_params  # tied with lm_head
    assert scaler.get_scale() == pytest.approx(scale_before * 0.5)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]), name

    optimizer.zero_grad(set_to_none=True)
    accumulate_gradients(model, [(x, y)], precision, scaler)
    stats = optimizer_step(model, optimizer, scaler, grad_clip=1.0)
    assert not stats.skipped and math.isfinite(stats.grad_norm)
    assert not torch.equal(model.lm_head.weight, before["wte.weight"])


# ======================================================================================
# Checkpointing and seeding
# ======================================================================================


def test_checkpoint_roundtrip_restores_everything(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig())
    scaler = CPU_FP32.grad_scaler()
    accumulate_gradients(model, [random_batch()], CPU_FP32, scaler)
    optimizer_step(model, optimizer, scaler, grad_clip=1.0)
    tok = CharTokenizer.from_text("abc\n")
    gen = torch.Generator().manual_seed(5)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, iter_num=7, best_val_loss=1.5,
                    config={"seed": 1}, tokenizer=tok, batch_generator=gen)
    expected_next = torch.randint(100, (4,), generator=gen)  # what gen yields after saving

    torch.manual_seed(99)  # different init, so the restore has to do real work
    model2 = tiny_model()
    optimizer2 = build_optimizer(model2, OptimizerConfig())
    gen2 = torch.Generator().manual_seed(12345)
    ckpt = load_checkpoint(path, model=model2, optimizer=optimizer2, batch_generator=gen2,
                           restore_rng=True)

    assert ckpt["iter_num"] == 7 and ckpt["best_val_loss"] == 1.5
    assert ckpt["config"] == {"seed": 1}
    assert CharTokenizer.from_state_dict(ckpt["tokenizer"]).chars == tok.chars
    assert not any("causal_mask" in k for k in ckpt["model"])  # non-persistent buffer
    for (name, a), (_, b) in zip(model.state_dict().items(), model2.state_dict().items(),
                                 strict=True):
        assert torch.equal(a, b), name
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(optimizer.state[p1]["exp_avg"], optimizer2.state[p2]["exp_avg"])
    assert torch.equal(torch.randint(100, (4,), generator=gen2), expected_next)
    assert not path.with_name("ckpt.pt.tmp").exists()  # atomic write cleaned up
    torch.load(path, weights_only=True)  # safe-loading works


def _three_training_steps(seed: int) -> list[float]:
    set_seed(seed, deterministic=True)
    model = tiny_model(dropout=0.1)  # dropout makes the RNG matter
    optimizer = build_optimizer(model, OptimizerConfig())
    scaler = CPU_FP32.grad_scaler()
    data = torch.randint(0, VOCAB, (2_000,))
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        batches = [get_batch(data, 4, 16, generator=gen) for _ in range(2)]
        losses.append(accumulate_gradients(model, batches, CPU_FP32, scaler).item())
        optimizer_step(model, optimizer, scaler, grad_clip=1.0)
    return losses


def test_set_seed_makes_training_bitwise_reproducible() -> None:
    assert _three_training_steps(0) == _three_training_steps(0)
    assert _three_training_steps(0) != _three_training_steps(1)


# ======================================================================================
# End-to-end training runs (synthetic corpus, no network)
# ======================================================================================

CORPUS = "the quick brown fox jumps over the lazy dog.\n" * 300


def write_corpus(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "input.txt").write_text(CORPUS, encoding="utf-8")


def tiny_raw_config(tmp_path: Path, **training: Any) -> dict[str, Any]:
    return {
        "seed": 0,
        "deterministic": True,
        # vocab_size is deliberately wrong: train() must take it from the tokenizer.
        "model": {"vocab_size": 1, "block_size": 16, "n_layer": 2, "n_head": 2,
                  "n_embd": 32, "dropout": 0.0},
        "data": {"data_dir": str(tmp_path / "data"), "train_split": 0.9},
        "training": {"batch_size": 8, "grad_accum_steps": 2, "max_iters": 40,
                     "eval_interval": 20, "eval_iters": 4, "log_interval": 5,
                     "device": "cpu", "amp_dtype": "float32",
                     "out_dir": str(tmp_path / "ckpt"), "sample_tokens": 20, **training},
        "optimizer": {"lr": 3.0e-3, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.99,
                      "eps": 1.0e-8},
        "scheduler": {"warmup_iters": 5, "lr_decay_iters": None, "min_lr": 3.0e-4},
    }


def test_train_end_to_end_cpu(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    summary = train(Config.from_dict(tiny_raw_config(tmp_path)))

    history = summary["eval_history"]
    assert [h["iter"] for h in history] == [0, 20, 40]
    assert history[-1]["val"] < history[0]["val"] - 0.5, history
    assert summary["skipped_steps"] == 0
    assert len(summary["sample"]) == 21  # prompt token + 20 sampled

    out = tmp_path / "ckpt"
    for name in ("best.pt", "last.pt", "metrics.jsonl", "train.log"):
        assert (out / name).exists(), name
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    train_records = [r for r in records if r["type"] == "train"]
    assert train_records and all(math.isfinite(r["grad_norm"]) for r in train_records)
    ckpt = load_checkpoint(out / "last.pt")
    assert ckpt["iter_num"] == 40
    assert ckpt["config"]["model"]["vocab_size"] == len(set(CORPUS))
    (out / "train.log").unlink()  # fails on Windows if the log handler leaked


def test_train_resume_continues_from_last_checkpoint(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    train(Config.from_dict(tiny_raw_config(tmp_path, max_iters=20)))
    summary = train(Config.from_dict(tiny_raw_config(tmp_path, max_iters=40)), resume=True)
    assert [h["iter"] for h in summary["eval_history"]] == [40]  # no re-eval at iter 20
    assert load_checkpoint(tmp_path / "ckpt" / "last.pt")["iter_num"] == 40


def test_new_run_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    train(Config.from_dict(tiny_raw_config(tmp_path, max_iters=10, eval_interval=5)))
    out = tmp_path / "ckpt"
    snapshot = {name: (out / name).read_bytes() for name in ("last.pt", "best.pt",
                                                              "metrics.jsonl", "train.log")}

    with pytest.raises(RunDirectoryError, match="--resume"):
        train(Config.from_dict(tiny_raw_config(tmp_path, max_iters=10, eval_interval=5)))

    for name, content in snapshot.items():  # nothing was touched, not even the logs
        assert (out / name).read_bytes() == content, name


def test_force_starts_a_clean_run(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    train(Config.from_dict(tiny_raw_config(tmp_path, max_iters=20, eval_interval=5)))
    train(Config.from_dict(tiny_raw_config(tmp_path, max_iters=10, eval_interval=5)),
          force=True)

    out = tmp_path / "ckpt"
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    evals = [r["iter"] for r in records if r["type"] == "eval"]
    assert evals == [0, 5, 10]  # only the forced run; the old run's metrics are gone
    assert load_checkpoint(out / "last.pt")["iter_num"] == 10


def test_resume_requires_an_existing_run(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    cfg = Config.from_dict(tiny_raw_config(tmp_path))
    with pytest.raises(RunDirectoryError, match="Cannot resume"):
        train(cfg, resume=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(cfg, resume=True, force=True)
    with pytest.raises(SystemExit):  # argparse rejects the combination too
        parse_args(["--resume", "--force"])


@requires_cuda
def test_train_end_to_end_cuda_fp16(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    summary = train(Config.from_dict(
        tiny_raw_config(tmp_path, device="cuda", amp_dtype="float16")))
    assert summary["amp_dtype"] == "float16"
    assert summary["final_loss_scale"] is not None
    assert summary["peak_mem_allocated_mib"] > 0
    history = summary["eval_history"]
    assert all(math.isfinite(h["val"]) for h in history)
    assert history[-1]["val"] < history[0]["val"] - 0.5, history
