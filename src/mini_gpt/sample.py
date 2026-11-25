"""Generate text from a trained Mini-GPT checkpoint.

Usage::

    python -m mini_gpt.sample                                    # checkpoints/best.pt
    python -m mini_gpt.sample --prompt "ROMEO:" --tokens 300 --top-k 40
    python -m mini_gpt.sample --checkpoint checkpoints/last.pt --temperature 0

A checkpoint stores the weights, the model config and the character vocabulary, so no
YAML file or dataset is needed. The generated text goes to stdout and a one-line header
goes to stderr, so ``> out.txt`` captures only the text.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from mini_gpt.model import MiniGPT, MiniGPTConfig
from mini_gpt.utils import CharTokenizer, load_checkpoint, resolve_device

DEFAULT_CHECKPOINT = "checkpoints/best.pt"


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[MiniGPT, CharTokenizer, dict[str, Any]]:
    """Rebuild a model and its tokenizer from a checkpoint written by ``mini_gpt.train``.

    Returns:
        ``(model, tokenizer, checkpoint)``. The model is on ``device`` in eval mode
        (dropout off). ``checkpoint`` is the raw dict (``iter_num``, ``best_val_loss``, ...).
    """
    ckpt = load_checkpoint(path)  # weights_only=True, tensors mapped to CPU
    if ckpt.get("tokenizer") is None:
        raise ValueError(f"{path} has no tokenizer; it was not written by mini_gpt.train")
    model = MiniGPT(MiniGPTConfig.from_dict(dict(ckpt["config"]["model"])))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, CharTokenizer.from_state_dict(ckpt["tokenizer"]), ckpt


def stream_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str | None = None,
    max_new_tokens: int = 500,
    *,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 1337,
) -> Iterator[str]:
    """Yield the continuation of ``prompt`` one sampled character at a time.

    The model should be in eval mode (``load_model`` does this). Sampling uses its own
    seeded generator, so the same arguments give the same text on the same device.

    Arguments are checked immediately, not on the first ``next()``. A caller such as the
    web UI can therefore reject bad input before it starts sending output.

    Args:
        prompt: Text to continue. ``None`` starts from token id 0 (the newline in
            TinyShakespeare), like the sample printed at the end of training.
        temperature: Softmax temperature; ``0.0`` means greedy decoding.
        top_k: If set, sample only from the ``top_k`` most likely characters.
        seed: Seed for the sampling generator.

    Raises:
        ValueError: If the prompt is empty or uses a character outside the vocabulary, or
            if ``max_new_tokens``, ``temperature`` or ``top_k`` is out of range.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if not (math.isfinite(temperature) and temperature >= 0.0):
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if prompt == "":
        raise ValueError("prompt must contain at least one character")
    ids = [0] if prompt is None else tokenizer.encode(prompt)
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T0)
    gen = torch.Generator(device=device).manual_seed(seed)  # must live on the model's device
    return _stream(model, tokenizer, idx, max_new_tokens, temperature, top_k, gen)


def _stream(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    gen: torch.Generator,
) -> Iterator[str]:
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # One step of model.generate. Repeated single steps that share one generator
        # consume the RNG exactly like a single call, so the text is identical.
        idx = model.generate(idx, 1, temperature=temperature, top_k=top_k,
                             generator=gen)  # (1, T) -> (1, T + 1)
        yield tokenizer.decode([int(idx[0, -1])])
        idx = idx[:, -block_size:]  # (1, T + 1) -> (1, <= block_size); only this is attended


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str | None = None,
    max_new_tokens: int = 500,
    *,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 1337,
) -> str:
    """Return ``prompt`` followed by ``max_new_tokens`` sampled characters.

    Arguments are the same as :func:`stream_text`. With ``prompt=None`` the text starts
    with the character for token id 0, exactly like the end-of-training sample.
    """
    chars = stream_text(model, tokenizer, prompt, max_new_tokens, temperature=temperature,
                        top_k=top_k, seed=seed)
    start = tokenizer.decode([0]) if prompt is None else prompt
    return start + "".join(chars)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Mini-GPT.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="checkpoint to load (default: %(default)s)")
    parser.add_argument("--prompt", default=None,
                        help="text to continue (default: start from a newline, like the "
                             "end-of-training sample)")
    parser.add_argument("--tokens", type=int, default=500,
                        help="number of characters to generate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="softmax temperature; 0 = greedy (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="sample only from the k most likely characters (default: off)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="sampling seed (default: %(default)s)")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | cuda:N (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.checkpoint)
    if not path.is_file():
        print(f"error: checkpoint {path} not found (train first: python tasks.py train)",
              file=sys.stderr)
        return 1
    device = resolve_device(args.device)
    model, tokenizer, ckpt = load_model(path, device)
    print(f"# {path} | iter {ckpt['iter_num']} | best val loss {ckpt['best_val_loss']:.4f} | "
          f"device {device}", file=sys.stderr)
    try:
        text = generate_text(model, tokenizer, args.prompt, args.tokens,
                             temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
