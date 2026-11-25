"""Tests for text generation from a trained checkpoint (``mini_gpt.sample``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mini_gpt.sample import generate_text, load_model, main
from mini_gpt.train import Config, train
from test_training import tiny_raw_config, write_corpus


@pytest.fixture
def trained_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Train the tiny CPU model (about a second); return its checkpoint dir and summary.

    Function-scoped on purpose: ``train`` sets process-wide determinism flags, and the
    autouse fixture in ``conftest.py`` only restores them around function-scoped setup.
    """
    write_corpus(tmp_path)
    summary = train(Config.from_dict(tiny_raw_config(tmp_path)))  # seed 0, 20 sample tokens
    return tmp_path / "ckpt", summary


def test_sampling_from_last_checkpoint_reproduces_the_training_sample(
    trained_run: tuple[Path, dict[str, Any]],
) -> None:
    out_dir, summary = trained_run
    model, tokenizer, ckpt = load_model(out_dir / "last.pt")
    assert not model.training  # dropout off
    # Same weights, prompt (token 0), temperature 0.8, seed and device as the end-of-run sample.
    text = generate_text(model, tokenizer, None, 20, temperature=0.8, seed=0)
    assert text == summary["sample"]
    assert ckpt["iter_num"] == 40


def test_cli_prints_prompt_and_continuation(
    trained_run: tuple[Path, dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir, _ = trained_run
    argv = ["--checkpoint", str(out_dir / "best.pt"), "--prompt", "the ", "--tokens", "30",
            "--top-k", "5", "--device", "cpu"]
    assert main(argv) == 0
    first = capsys.readouterr()
    assert first.out.startswith("the ") and len(first.out) == len("the ") + 30 + 1  # + "\n"
    assert "best val loss" in first.err  # header goes to stderr, text to stdout

    assert main(argv) == 0
    assert capsys.readouterr().out == first.out  # seeded: identical on repeat


def test_cli_reports_bad_input(
    trained_run: tuple[Path, dict[str, Any]], tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, _ = trained_run
    assert main(["--checkpoint", str(tmp_path / "missing.pt")]) == 1
    assert "not found" in capsys.readouterr().err

    # "Z" never occurs in the lowercase training corpus.
    assert main(["--checkpoint", str(out_dir / "best.pt"), "--prompt", "Zebra",
                 "--device", "cpu"]) == 2
    assert "not in the vocabulary" in capsys.readouterr().err

    assert main(["--checkpoint", str(out_dir / "best.pt"), "--top-k", "0",
                 "--device", "cpu"]) == 2
    assert "top_k must be positive" in capsys.readouterr().err
