"""Shared utilities for Mini-GPT.

* Reproducibility: :func:`set_seed`.
* Device and mixed-precision selection: :func:`resolve_device`,
  :func:`resolve_amp_dtype`, :class:`Precision`.
* Config loading with ``section.key=value`` overrides: :func:`load_config`.
* Checkpointing: :func:`save_checkpoint`, :func:`load_checkpoint`.
* Character-level TinyShakespeare data: :class:`CharTokenizer`,
  :func:`load_char_dataset`, :func:`get_batch`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import re
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device) RNGs.

    With ``deterministic=True`` this also:

    * sets ``cudnn.deterministic = True`` and ``cudnn.benchmark = False``;
    * sets ``CUBLAS_WORKSPACE_CONFIG`` (needed for deterministic cuBLAS matmuls);
    * enables ``torch.use_deterministic_algorithms`` in warn-only mode, so any
      non-deterministic kernel is reported instead of silently used.

    Call this *before* the first CUDA operation: cuBLAS reads
    ``CUBLAS_WORKSPACE_CONFIG`` when its handle is created.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU RNG and every CUDA device
    torch.cuda.manual_seed_all(seed)  # explicit; a no-op without CUDA
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


# --------------------------------------------------------------------------------------
# Device and precision
# --------------------------------------------------------------------------------------

_AMP_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_device(name: str = "auto") -> torch.device:
    """Map ``"auto" | "cuda" | "cuda:N" | "cpu"`` to a device, failing early if unusable."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Install the CUDA build of PyTorch "
            "(`make setup`) or set training.device=cpu."
        )
    return device


def has_native_bf16(device: torch.device) -> bool:
    """True only for GPUs with *native* bfloat16 (compute capability >= 8.0, Ampere+).

    ``torch.cuda.is_bf16_supported()`` can also return True when bf16 is merely emulated
    (e.g. Turing, sm_75), which is correct but slow, so it is not used here.
    """
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0)


# Turing (compute capability 7.5) chips WITHOUT tensor cores: TU116/TU117 in the GTX 16xx
# series, MX450/MX550 laptop GPUs and the T400-T2000 workstation cards. RTX 20xx, Quadro
# RTX and Tesla T4 share the same compute capability but do have tensor cores.
_TURING_WITHOUT_TENSOR_CORES = re.compile(
    r"GTX\s*16\d\d|\bMX\s*[45]50\b|\bT(?:400|500|550|600|1000|1200|2000)\b", re.IGNORECASE
)


def has_tensor_cores(device: torch.device) -> bool:
    """Best-effort check for tensor cores, the units that make fp16/bf16 matmuls fast.

    * Compute capability < 7.0 (Pascal and older): no tensor cores.
    * 7.0/7.2 (Volta) and >= 8.0 (Ampere, Ada, Hopper, Blackwell): tensor cores.
    * 7.5 (Turing): depends on the chip, so the device name is matched against
      ``_TURING_WITHOUT_TENSOR_CORES``.

    Without tensor cores float16 brings no speed-up and can be much slower: on a
    GTX 1650 Ti, cuBLAS fp16 GEMMs measured 0.41 TFLOP/s against 1.8 TFLOP/s for fp32.
    """
    if device.type != "cuda":
        return False
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 0):
        return False
    if capability == (7, 5):
        return _TURING_WITHOUT_TENSOR_CORES.search(torch.cuda.get_device_name(device)) is None
    return True


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    """Pick the autocast dtype.

    ``"auto"`` resolves to:

    * bfloat16 on GPUs with native bf16 (Ampere and newer);
    * float16 with loss scaling on GPUs with tensor cores but no native bf16 (Volta,
      Turing RTX);
    * float32 (autocast off) on GPUs without tensor cores (GTX 16xx, Pascal) and on CPU,
      where reduced precision saves memory but costs speed.
    """
    if name == "auto":
        if has_native_bf16(device):
            return torch.bfloat16
        if has_tensor_cores(device):
            return torch.float16
        return torch.float32
    if name not in _AMP_DTYPES:
        raise ValueError(f"amp_dtype must be one of {['auto', *_AMP_DTYPES]}, got {name!r}")
    dtype = _AMP_DTYPES[name]
    if dtype == torch.float16 and device.type != "cuda":
        raise ValueError("float16 autocast with loss scaling requires CUDA; use bfloat16 or "
                         "float32 on CPU.")
    if dtype == torch.float16 and not has_tensor_cores(device):
        log.warning("float16 was requested but this GPU has no tensor cores; float32 is "
                    "usually faster here (measured ~3.7x on a GTX 1650 Ti).")
    if dtype == torch.bfloat16 and device.type == "cuda" and not has_native_bf16(device):
        log.warning("bfloat16 is emulated on this GPU (compute capability < 8.0) and will be "
                    "slow; use float16 (tensor-core GPUs) or float32.")
    return dtype


@dataclass(frozen=True)
class Precision:
    """Mixed-precision policy for one device.

    float32 means autocast is off. float16 additionally needs a GradScaler, because its
    narrow exponent range underflows small gradients. bfloat16 has fp32's range, so it
    does not.
    """

    device: torch.device
    dtype: torch.dtype

    @property
    def autocast_enabled(self) -> bool:
        return self.dtype != torch.float32

    @property
    def needs_grad_scaler(self) -> bool:
        return self.dtype == torch.float16

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """Autocast context for forward passes; a no-op context when disabled."""
        if not self.autocast_enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.dtype)

    def grad_scaler(self) -> torch.amp.GradScaler:
        """A GradScaler that is active only for float16 (pass-through otherwise)."""
        return torch.amp.GradScaler(self.device.type, enabled=self.needs_grad_scaler)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

_DC = TypeVar("_DC")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a YAML config and apply ``section.key=value`` overrides."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> None:
    """Apply overrides like ``training.max_iters=200`` in place.

    The key must already exist in the config, so typos fail loudly. Values are parsed as
    int, then float (so ``1e-3`` works here, unlike in PyYAML), then YAML (``true``,
    ``null``, strings).
    """
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node: Any = cfg
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                raise KeyError(f"Unknown config section {part!r} in override {item!r}")
        if leaf not in node:
            raise KeyError(f"Unknown config key {key!r} in override {item!r}")
        node[leaf] = _parse_scalar(raw)


def _parse_scalar(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return yaml.safe_load(raw)


def dataclass_from_dict(cls: type[_DC], data: Mapping[str, Any] | None, section: str) -> _DC:
    """Instantiate a config dataclass, rejecting unknown keys (catches YAML typos)."""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}': {unknown}. Valid: {sorted(known)}")
    return cls(**data)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def capture_rng_state(batch_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Snapshot the Python, torch CPU/CUDA and batch-sampler RNG states.

    NumPy's state is left out: it holds an ndarray, which ``torch.load(weights_only=True)``
    refuses, and nothing in training draws from NumPy.
    """
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "batch_generator": batch_generator.get_state() if batch_generator is not None else None,
    }


def restore_rng_state(
    state: Mapping[str, Any], batch_generator: torch.Generator | None = None
) -> None:
    """Inverse of :func:`capture_rng_state`."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if batch_generator is not None and state["batch_generator"] is not None:
        batch_generator.set_state(state["batch_generator"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    iter_num: int,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_val_loss: float = float("inf"),
    tokenizer: CharTokenizer | None = None,
    batch_generator: torch.Generator | None = None,
) -> None:
    """Write a training checkpoint atomically (temp file, then rename).

    A crash mid-write therefore never leaves a truncated checkpoint behind. The file
    contains only tensors and plain Python types, so it loads with
    ``torch.load(weights_only=True)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": dict(config),
        "tokenizer": tokenizer.state_dict() if tokenizer is not None else None,
        "rng": capture_rng_state(batch_generator),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    batch_generator: torch.Generator | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint safely (``weights_only=True``) and restore any given objects.

    Returns the raw checkpoint dict (``iter_num``, ``best_val_loss``, ``config``, ...).
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        restore_rng_state(ckpt["rng"], batch_generator)
    return ckpt


# --------------------------------------------------------------------------------------
# Character-level tokenizer and TinyShakespeare data
# --------------------------------------------------------------------------------------


class CharTokenizer:
    """Bijective mapping between single characters and integer ids ``0..V-1``.

    Args:
        chars: The vocabulary, in id order. Must contain unique single characters.
    """

    def __init__(self, chars: Sequence[str]) -> None:
        chars = list(chars)
        if not chars:
            raise ValueError("Vocabulary must not be empty")
        if any(len(c) != 1 for c in chars):
            raise ValueError("Every vocabulary entry must be a single character")
        if len(set(chars)) != len(chars):
            raise ValueError("Vocabulary contains duplicate characters")
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a vocabulary of the sorted unique characters in ``text``."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"Character {e.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: Iterable[int]) -> str:
        try:
            return "".join(self.itos[int(i)] for i in ids)
        except KeyError as e:
            raise ValueError(f"Token id {e.args[0]} is outside the vocabulary") from None

    def state_dict(self) -> dict[str, Any]:
        return {"chars": "".join(self.chars)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> CharTokenizer:
        return cls(list(state["chars"]))


@dataclass(frozen=True)
class CharDataset:
    """Encoded train/val token streams plus the tokenizer that produced them."""

    train: torch.Tensor  # (N_train,) int64
    val: torch.Tensor  # (N_val,) int64
    tokenizer: CharTokenizer


def download_tinyshakespeare(data_dir: str | Path, url: str = TINY_SHAKESPEARE_URL) -> Path:
    """Return ``<data_dir>/input.txt``, downloading it first if it does not exist."""
    path = Path(data_dir) / "input.txt"
    if path.exists():
        return path
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download from a non-HTTPS URL: {url!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading TinyShakespeare from %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "mini-gpt"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    log.info("Saved %d bytes to %s", len(payload), path)
    return path


def load_char_dataset(
    data_dir: str | Path, url: str = TINY_SHAKESPEARE_URL, train_split: float = 0.9
) -> CharDataset:
    """Load (downloading if needed) the corpus, tokenize it, and split it train/val.

    The vocabulary is built from the full text (standard for char-level models). The
    split is a contiguous prefix/suffix, so validation text is never seen in training.
    """
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    text = download_tinyshakespeare(data_dir, url).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # (N,)
    n_train = int(train_split * data.numel())
    return CharDataset(train=data[:n_train], val=data[n_train:], tokenizer=tokenizer)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous windows and their next-token targets.

    Args:
        data: 1-D token stream of shape ``(N,)``, kept on the CPU.
        batch_size: Number of windows ``B``.
        block_size: Window length ``T``.
        device: Where to put the returned batch.
        generator: CPU RNG for the window offsets (makes the data order reproducible
            and independent of the device).

    Returns:
        ``(x, y)``, both of shape ``(B, T)``, with ``y[:, i] == x[:, i + 1]`` for
        ``i < T - 1``.
    """
    n = data.numel()
    if n <= block_size:
        raise ValueError(f"Split has {n} tokens; need more than block_size={block_size}")
    starts = torch.randint(n - block_size, (batch_size,), generator=generator)  # (B,)
    offsets = torch.arange(block_size)  # (T,)
    positions = starts[:, None] + offsets[None, :]  # (B, 1) + (1, T) -> (B, T)
    x = data[positions]  # (N,)[(B, T)] -> (B, T)
    y = data[positions + 1]  # (B, T), shifted one step right
    return x.to(device), y.to(device)
