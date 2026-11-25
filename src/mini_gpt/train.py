"""Train Mini-GPT on character-level TinyShakespeare.

Usage::

    python -m mini_gpt.train --config configs/default.yaml
    python -m mini_gpt.train -o training.max_iters=500 -o training.batch_size=8
    python -m mini_gpt.train --resume          # continue from <out_dir>/last.pt
    python -m mini_gpt.train --force           # start over in an out_dir that has a run

A new run refuses to start in an ``out_dir`` that already holds checkpoints, so a
finished run is never overwritten by accident.

Training features:

* Mixed precision with ``torch.amp.autocast``: float16 plus ``GradScaler`` on GPUs
  without native bfloat16, bfloat16 on Ampere+, float32 on CPU.
* Gradient accumulation, so the effective batch fits a small GPU.
* Linear warmup followed by cosine learning-rate decay.
* Global-norm gradient clipping with ``clip_grad_norm_``.
* Gradient-norm logging and NaN/Inf detection. A non-finite step is skipped and the
  offending parameters are named. Training aborts after too many consecutive bad
  steps.
* Periodic evaluation, atomic ``best.pt``/``last.pt`` checkpoints, and a
  ``metrics.jsonl`` log.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch

from mini_gpt.model import MiniGPT, MiniGPTConfig
from mini_gpt.utils import (
    TINY_SHAKESPEARE_URL,
    Precision,
    dataclass_from_dict,
    get_batch,
    load_char_dataset,
    load_checkpoint,
    load_config,
    resolve_amp_dtype,
    resolve_device,
    save_checkpoint,
    set_seed,
)

log = logging.getLogger("mini_gpt.train")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def _require_number(section: str, name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        hint = ""
        if isinstance(value, str):
            hint = " (PyYAML reads values like 1e-3 as strings; write 1.0e-3 instead)"
        raise ValueError(f"{section}.{name} must be a number, got {value!r}{hint}")


def _require_positive_int(section: str, name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section}.{name} must be a positive int, got {value!r}")


@dataclass(frozen=True)
class DataConfig:
    dataset_url: str = TINY_SHAKESPEARE_URL
    data_dir: str = "data"
    train_split: float = 0.9

    def __post_init__(self) -> None:
        _require_number("data", "train_split", self.train_split)
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"data.train_split must be in (0, 1), got {self.train_split}")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_iters: int = 5000
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    device: str = "auto"
    amp_dtype: str = "auto"
    out_dir: str = "checkpoints"
    max_nonfinite_steps: int = 25
    sample_tokens: int = 500

    def __post_init__(self) -> None:
        for name in ("batch_size", "grad_accum_steps", "max_iters", "eval_interval",
                     "eval_iters", "log_interval", "max_nonfinite_steps"):
            _require_positive_int("training", name, getattr(self, name))
        _require_number("training", "grad_clip", self.grad_clip)
        if self.grad_clip < 0:
            raise ValueError("training.grad_clip must be >= 0 (0 disables clipping)")
        if self.sample_tokens < 0:
            raise ValueError("training.sample_tokens must be >= 0")


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1.0e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            _require_number("optimizer", f.name, getattr(self, f.name))
        if self.lr <= 0 or self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer: lr and eps must be > 0, weight_decay >= 0")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("optimizer: beta1 and beta2 must be in [0, 1)")


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_iters: int = 100
    lr_decay_iters: int | None = None  # None -> decay over the whole run (max_iters)
    min_lr: float = 1.0e-4

    def __post_init__(self) -> None:
        if isinstance(self.warmup_iters, bool) or not isinstance(self.warmup_iters, int) \
                or self.warmup_iters < 0:
            raise ValueError(f"scheduler.warmup_iters must be an int >= 0, "
                             f"got {self.warmup_iters!r}")
        if self.lr_decay_iters is not None:
            _require_positive_int("scheduler", "lr_decay_iters", self.lr_decay_iters)
        _require_number("scheduler", "min_lr", self.min_lr)
        if self.min_lr < 0:
            raise ValueError("scheduler.min_lr must be >= 0")


@dataclass(frozen=True)
class Config:
    """Validated, immutable view of the full YAML config."""

    model: MiniGPTConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    seed: int = 1337
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.scheduler.min_lr > self.optimizer.lr:
            raise ValueError("scheduler.min_lr must not exceed optimizer.lr")
        if self.scheduler.warmup_iters >= self.lr_decay_iters:
            raise ValueError(
                f"scheduler.warmup_iters ({self.scheduler.warmup_iters}) must be smaller "
                f"than the decay horizon ({self.lr_decay_iters})"
            )

    @property
    def lr_decay_iters(self) -> int:
        """Effective cosine-decay horizon (defaults to ``max_iters``)."""
        return self.scheduler.lr_decay_iters or self.training.max_iters

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Config:
        sections = {"model", "data", "training", "optimizer", "scheduler"}
        unknown = sorted(set(raw) - sections - {"seed", "deterministic"})
        if unknown:
            raise ValueError(f"Unknown top-level config key(s): {unknown}")
        return cls(
            model=MiniGPTConfig.from_dict(dict(raw.get("model") or {})),
            data=dataclass_from_dict(DataConfig, raw.get("data"), "data"),
            training=dataclass_from_dict(TrainingConfig, raw.get("training"), "training"),
            optimizer=dataclass_from_dict(OptimizerConfig, raw.get("optimizer"), "optimizer"),
            scheduler=dataclass_from_dict(SchedulerConfig, raw.get("scheduler"), "scheduler"),
            seed=int(raw.get("seed", 1337)),
            deterministic=bool(raw.get("deterministic", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Learning-rate schedule and optimizer
# --------------------------------------------------------------------------------------


def get_lr(
    it: int, *, max_lr: float, min_lr: float, warmup_iters: int, lr_decay_iters: int
) -> float:
    """Learning rate for optimizer step ``it`` (0-based).

    * ``it < warmup_iters``: linear ramp ``max_lr * (it + 1) / warmup_iters``, reaching
      ``max_lr`` on the last warmup step.
    * ``warmup_iters <= it < lr_decay_iters``: cosine decay from ``max_lr`` to ``min_lr``.
    * ``it >= lr_decay_iters``: constant ``min_lr``.
    """
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it >= lr_decay_iters:
        return min_lr
    progress = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # in [0, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def build_optimizer(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied to matrices only.

    Weights with ``dim >= 2`` (Linear and Embedding) are decayed. Biases and LayerNorm
    gains are not: decaying them only pulls the normalisation towards zero.
    ``model.parameters()`` yields a tied weight once, so it is not double-counted.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


# --------------------------------------------------------------------------------------
# One optimizer step: accumulation, unscale, norm check, clip, step
# --------------------------------------------------------------------------------------


def accumulate_gradients(
    model: torch.nn.Module,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    precision: Precision,
    scaler: torch.amp.GradScaler,
) -> torch.Tensor:
    """Run forward/backward over ``micro_batches`` and accumulate gradients.

    Each micro-batch loss is divided by the number of micro-batches, so the summed
    gradient equals the gradient of the *mean* loss over the effective batch. For
    equal-sized micro-batches this matches one big batch exactly (up to float
    rounding).

    Returns:
        The mean un-scaled loss as a 0-dim tensor. It stays on the device, so there is
        no host sync.
    """
    if not micro_batches:
        raise ValueError("micro_batches must not be empty")
    n = len(micro_batches)
    total = torch.zeros((), device=micro_batches[0][0].device)  # ()
    for x, y in micro_batches:  # x, y: (B, T)
        with precision.autocast():
            _, loss = model(x, y)  # loss: () in >= fp32
        assert loss is not None
        scaler.scale(loss / n).backward()  # scaled only when float16
        total += loss.detach()  # ()
    return total / n


@dataclass(frozen=True)
class StepStats:
    """Outcome of :func:`optimizer_step`."""

    grad_norm: float  # global L2 norm of the true (un-scaled) gradients, before clipping
    clipped: bool  # True if the norm exceeded grad_clip and the gradients were rescaled
    skipped: bool  # True if the update was skipped because of a NaN/Inf gradient
    nonfinite_params: tuple[str, ...]  # names of parameters whose gradient was NaN/Inf


def optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> StepStats:
    """Unscale, check, clip and apply the accumulated gradients.

    Order of operations:

    1. ``scaler.unscale_``: gradients back to true units (a no-op without float16).
    2. Per-parameter L2 norms, then the global norm. This happens *before* clipping,
       because ``clip_grad_norm_`` multiplies every gradient by ``max_norm / total``
       and would spread a single NaN to all parameters, hiding the culprit.
    3. If the norm is finite: ``clip_grad_norm_`` (it recomputes the same norm; the
       extra pass is cheap), then ``scaler.step``, which equals ``optimizer.step()``
       when the scaler is disabled.
    4. If not finite: skip the update and record which parameters were bad.
    5. ``scaler.update()`` on every step. After a non-finite step it halves the float16
       loss scale, using the inf check recorded by ``unscale_``.
    """
    scaler.unscale_(optimizer)
    named = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    per_param = torch.stack([torch.linalg.vector_norm(p.grad) for _, p in named])  # (P,)
    total = torch.linalg.vector_norm(per_param)  # (P,) -> ()
    grad_norm = total.item()  # the one host sync per step
    finite = math.isfinite(grad_norm)

    clipped = False
    nonfinite: tuple[str, ...] = ()
    if finite:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in named], max_norm=grad_clip)
            clipped = grad_norm > grad_clip
        scaler.step(optimizer)
    else:
        bad = (~torch.isfinite(per_param)).tolist()  # (P,) bool
        nonfinite = tuple(n for (n, _), is_bad in zip(named, bad, strict=True) if is_bad)
    scaler.update()
    return StepStats(grad_norm=grad_norm, clipped=clipped, skipped=not finite,
                     nonfinite_params=nonfinite)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    splits: Mapping[str, torch.Tensor],
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    precision: Precision,
    seed: int,
) -> dict[str, float]:
    """Mean loss over ``eval_iters`` batches per split, with dropout disabled.

    A fresh generator with a fixed seed is used for every call, so each evaluation sees
    the *same* batches. That makes successive val losses directly comparable instead of
    adding sampling noise to the curve.
    """
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for name, data in splits.items():
        gen = torch.Generator().manual_seed(seed)
        losses = torch.empty(eval_iters, device=precision.device)  # (eval_iters,)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, precision.device, gen)  # (B, T)
            with precision.autocast():
                _, loss = model(x, y)  # ()
            losses[k] = loss
        out[name] = losses.mean().item()
    model.train(was_training)
    return out


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------


def _write_metrics(stream: TextIO, record: Mapping[str, Any]) -> None:
    stream.write(json.dumps(record) + "\n")
    stream.flush()


def _peak_mem_mib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0


class RunDirectoryError(RuntimeError):
    """``out_dir`` is in the wrong state for the requested start mode."""


def _check_run_directory(out_dir: Path, *, resume: bool, force: bool) -> None:
    """Protect existing runs: no silent overwrite, and no resume from nothing."""
    if resume and force:
        raise ValueError("resume and force are mutually exclusive")
    existing = [name for name in ("last.pt", "best.pt") if (out_dir / name).exists()]
    if resume and "last.pt" not in existing:
        raise RunDirectoryError(f"Cannot resume: {out_dir / 'last.pt'} does not exist.")
    if existing and not (resume or force):
        raise RunDirectoryError(
            f"{out_dir} already holds a run ({', '.join(existing)}). Pass --resume to "
            f"continue it, --force to start over (its checkpoints and logs are replaced), "
            f"or set training.out_dir to a new directory."
        )


def train(cfg: Config, *, resume: bool = False, force: bool = False) -> dict[str, Any]:
    """Run training and return a summary dict (losses, memory, skipped steps, sample).

    Args:
        cfg: Validated configuration.
        resume: Continue the run in ``training.out_dir`` from ``last.pt``. Logs and
            metrics are appended.
        force: Start a fresh run even if ``training.out_dir`` already holds one. Its
            checkpoints are overwritten and its logs and metrics are truncated, so two
            runs never mix in one ``metrics.jsonl``.

    Raises:
        RunDirectoryError: A run already exists and neither ``resume`` nor ``force`` is
            set, or ``resume`` is set but there is nothing to resume.
    """
    out_dir = Path(cfg.training.out_dir)
    _check_run_directory(out_dir, resume=resume, force=force)  # before touching any file
    set_seed(cfg.seed, cfg.deterministic)  # before any CUDA op (see set_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "a" if resume else "w"
    file_handler = logging.FileHandler(out_dir / "train.log", mode=mode, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    log.addHandler(file_handler)
    try:
        with open(out_dir / "metrics.jsonl", mode, encoding="utf-8") as metrics:
            return _train(cfg, out_dir, metrics, resume=resume)
    finally:
        log.removeHandler(file_handler)
        file_handler.close()


def _train(cfg: Config, out_dir: Path, metrics: TextIO, *, resume: bool) -> dict[str, Any]:
    tcfg = cfg.training
    device = resolve_device(tcfg.device)
    precision = Precision(device, resolve_amp_dtype(tcfg.amp_dtype, device))

    # ---- data ----
    dataset = load_char_dataset(cfg.data.data_dir, cfg.data.dataset_url, cfg.data.train_split)
    tokenizer = dataset.tokenizer
    if cfg.model.vocab_size != tokenizer.vocab_size:
        log.warning("model.vocab_size=%d does not match the tokenizer (%d); using the "
                    "tokenizer's value", cfg.model.vocab_size, tokenizer.vocab_size)
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, vocab_size=tokenizer.vocab_size))
    block_size = cfg.model.block_size

    # ---- model, optimizer, scaler ----
    model = MiniGPT(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.optimizer)
    scaler = precision.grad_scaler()
    batch_gen = torch.Generator().manual_seed(cfg.seed)  # CPU RNG for batch offsets

    start_iter, best_val = 0, float("inf")
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume:
        ckpt = load_checkpoint(last_path, model=model, optimizer=optimizer, scaler=scaler,
                               batch_generator=batch_gen, restore_rng=True)
        start_iter, best_val = int(ckpt["iter_num"]), float(ckpt["best_val_loss"])
        log.info("Resumed from %s at iter %d (best val %.4f)", last_path, start_iter, best_val)

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum_steps * block_size
    log.info("device=%s amp_dtype=%s grad_scaler=%s deterministic=%s", device,
             str(precision.dtype).removeprefix("torch."), scaler.is_enabled(), cfg.deterministic)
    log.info("params=%.2fM (non-embedding %.2fM) | vocab=%d | train/val tokens=%d/%d",
             model.num_parameters(False) / 1e6, model.num_parameters() / 1e6,
             tokenizer.vocab_size, dataset.train.numel(), dataset.val.numel())
    log.info("micro-batch=%d x accum=%d x block=%d -> %d tokens/step | lr %.1e->%.1e, "
             "warmup %d, decay to iter %d", tcfg.batch_size, tcfg.grad_accum_steps,
             block_size, tokens_per_step, cfg.optimizer.lr, cfg.scheduler.min_lr,
             cfg.scheduler.warmup_iters, cfg.lr_decay_iters)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save(path: Path, it: int) -> None:
        save_checkpoint(path, model=model, optimizer=optimizer, scaler=scaler, iter_num=it,
                        best_val_loss=best_val, config=cfg.to_dict(), tokenizer=tokenizer,
                        batch_generator=batch_gen)

    eval_history: list[dict[str, float]] = []
    consecutive_bad = skipped_total = 0
    last_train_loss = float("nan")
    model.train()

    for it in range(start_iter, tcfg.max_iters + 1):
        is_last = it == tcfg.max_iters

        # ---- evaluation + checkpointing (before the step, so iter 0 = init loss) ----
        if (it % tcfg.eval_interval == 0 or is_last) and not (resume and it == start_iter):
            losses = estimate_loss(model, {"train": dataset.train, "val": dataset.val},
                                   eval_iters=tcfg.eval_iters, batch_size=tcfg.batch_size,
                                   block_size=block_size, precision=precision,
                                   seed=cfg.seed + 1)
            eval_history.append({"iter": it, **losses})
            _write_metrics(metrics, {"type": "eval", "iter": it, **losses})
            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]
                save(best_path, it)
            save(last_path, it)
            log.info("eval  iter %5d | train %.4f | val %.4f%s", it, losses["train"],
                     losses["val"], " | saved best.pt" if improved else "")
        if is_last:
            break

        # ---- one optimizer step over grad_accum_steps micro-batches ----
        t0 = time.perf_counter()
        lr = get_lr(it, max_lr=cfg.optimizer.lr, min_lr=cfg.scheduler.min_lr,
                    warmup_iters=cfg.scheduler.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batches = [get_batch(dataset.train, tcfg.batch_size, block_size, device, batch_gen)
                   for _ in range(tcfg.grad_accum_steps)]
        loss = accumulate_gradients(model, batches, precision, scaler)
        stats = optimizer_step(model, optimizer, scaler, tcfg.grad_clip)
        dt = time.perf_counter() - t0  # optimizer_step synced via .item()

        # ---- numerical-instability handling ----
        if stats.skipped:
            consecutive_bad += 1
            skipped_total += 1
            reason = ("float16 overflow; the loss scale is lowered to "
                      f"{scaler.get_scale():g}" if scaler.is_enabled() else "update skipped")
            log.warning("iter %5d | non-finite grad norm (%s): %s | params: %s", it,
                        stats.grad_norm, reason, ", ".join(stats.nonfinite_params[:5]) or "-")
            if consecutive_bad > tcfg.max_nonfinite_steps:
                raise FloatingPointError(
                    f"{consecutive_bad} consecutive non-finite gradient steps (last at iter "
                    f"{it}); training has diverged. Lower optimizer.lr, keep grad_clip > 0, "
                    f"or use amp_dtype=float32 to diagnose."
                )
        else:
            consecutive_bad = 0

        # ---- logging ----
        if it % tcfg.log_interval == 0:
            last_train_loss = loss.item()
            loss_scale = scaler.get_scale() if scaler.is_enabled() else None
            record = {
                "type": "train", "iter": it, "loss": last_train_loss, "lr": lr,
                "grad_norm": stats.grad_norm, "clipped": stats.clipped,
                "skipped": stats.skipped, "loss_scale": loss_scale,
                "step_ms": dt * 1e3, "tok_per_s": tokens_per_step / dt,
                "peak_mem_mib": _peak_mem_mib(device),
            }
            _write_metrics(metrics, record)
            log.info("iter  %5d | loss %.4f | lr %.2e | grad_norm %7.3f%s | scale %-7s | "
                     "%6.0f ms | %7.0f tok/s | peak %4.0f MiB", it, last_train_loss, lr,
                     stats.grad_norm, " (clipped)" if stats.clipped else "",
                     "off" if loss_scale is None else f"{loss_scale:g}", dt * 1e3,
                     tokens_per_step / dt, record["peak_mem_mib"])

    # ---- summary + sample ----
    sample = ""
    if tcfg.sample_tokens > 0:
        model.eval()
        gen = torch.Generator(device=device).manual_seed(cfg.seed)  # must match the device
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # (1, 1)
        with precision.autocast():
            ids = model.generate(prompt, tcfg.sample_tokens, temperature=0.8, top_k=None,
                                 generator=gen)  # (1, 1 + sample_tokens)
        sample = tokenizer.decode(ids[0].tolist())
        log.info("sample:\n%s", sample)

    summary = {
        "iters": tcfg.max_iters,
        "best_val_loss": best_val,
        "eval_history": eval_history,
        "last_train_loss": last_train_loss,
        "skipped_steps": skipped_total,
        "final_loss_scale": scaler.get_scale() if scaler.is_enabled() else None,
        "peak_mem_allocated_mib": _peak_mem_mib(device),
        "peak_mem_reserved_mib": (torch.cuda.max_memory_reserved(device) / 2**20
                                  if device.type == "cuda" else 0.0),
        "device": str(device),
        "amp_dtype": str(precision.dtype).removeprefix("torch."),
        "sample": sample,
    }
    log.info("done | best val %.4f | skipped steps %d | peak allocated %.0f MiB, reserved "
             "%.0f MiB", best_val, skipped_total, summary["peak_mem_allocated_mib"],
             summary["peak_mem_reserved_mib"])
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mini-GPT on TinyShakespeare.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. -o training.max_iters=200 "
                             "(repeatable)")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true",
                       help="continue the run in <training.out_dir> from last.pt")
    start.add_argument("--force", action="store_true",
                       help="start over even if <training.out_dir> already holds a run "
                            "(replaces its checkpoints, logs and metrics)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    cfg = Config.from_dict(load_config(args.config, args.override))
    try:
        train(cfg, resume=args.resume, force=args.force)
    except RunDirectoryError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
