"""Decoder-only Transformer (mini-GPT) built from core PyTorch primitives.

Only ``nn.Module``, ``nn.Linear``, ``nn.Embedding``, ``nn.Dropout`` and raw tensor
operations are used. Attention and layer normalisation are written by hand: there is
no ``nn.Transformer``, ``nn.MultiheadAttention``, ``nn.LayerNorm`` or
``F.scaled_dot_product_attention``.

Architecture (GPT-2 style, pre-LayerNorm)::

    idx -> token emb + position emb -> dropout
        -> N x [ x + Attn(LN(x)) ; x + MLP(LN(x)) ]
        -> LN -> lm_head -> logits

Numerical-stability choices:
    * LayerNorm statistics, attention softmax and the cross-entropy loss always run in
      at least float32, even when the surrounding code runs under fp16/bf16 autocast.
      (float64 inputs stay float64.)
    * Queries are scaled by ``1/sqrt(hs)`` *before* the ``q @ k^T`` matmul, so the raw
      fp16 dot products stay small and do not overflow (fp16 max is about 6.5e4).
    * Causal masking fills with ``-inf``. Every query row keeps at least its diagonal
      entry, so softmax never evaluates an all ``-inf`` row (which would give NaN).

Shape legend used in the comments:
    B  = batch size
    T  = sequence length (T <= block_size)
    C  = embedding dimension (n_embd)
    nh = number of attention heads (n_head)
    hs = head size (C // nh)
    V  = vocabulary size
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CausalSelfAttention",
    "FeedForward",
    "LayerNorm",
    "MiniGPT",
    "MiniGPTConfig",
    "TransformerBlock",
]


def _upcast(x: torch.Tensor) -> torch.Tensor:
    """Promote fp16/bf16 to fp32; leave fp32 and fp64 untouched (no copy).

    Unlike ``x.float()``, this never *downcasts* float64, so fp64 reference tests stay
    exact.
    """
    return x.to(torch.promote_types(x.dtype, torch.float32))


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class MiniGPTConfig:
    """Architecture hyperparameters for :class:`MiniGPT`.

    Attributes:
        vocab_size: Number of distinct tokens ``V``.
        block_size: Maximum context length ``T``. Sets the positional-embedding table
            size and the causal-mask size.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads. Must divide ``n_embd``.
        n_embd: Residual-stream width ``C``.
        dropout: Dropout probability for embeddings, attention weights and residual
            branches. Use ``0.0`` for deterministic forward passes.
        bias: Whether Linear layers and LayerNorms carry a bias term.
        tie_weights: Share the token-embedding matrix with the output projection.
    """

    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")

    @property
    def head_size(self) -> int:
        """Per-head dimension ``hs = n_embd // n_head``."""
        return self.n_embd // self.n_head

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> MiniGPTConfig:
        """Build a config from a mapping, such as the ``model:`` section of the YAML.

        Unknown keys raise instead of being ignored, so a typo in the YAML file
        (``n_heads`` for ``n_head``) fails loudly instead of silently using a default.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"Unknown model config keys: {unknown}. Valid keys: {sorted(known)}")
        return cls(**cfg)


# --------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Layer normalisation over the last dimension, written with raw tensor ops.

    Computes ``y = (x - mean) / sqrt(var + eps) * weight + bias`` using the *biased*
    variance (divide by ``C``, not ``C - 1``), which matches ``nn.LayerNorm``.

    The statistics are computed in at least float32 whatever the input dtype, and the
    result is cast back to the input dtype. Under fp16 autocast this avoids the
    precision loss and overflow of computing ``mean((x - mean)^2)`` in half precision.

    Args:
        ndim: Size ``C`` of the normalised (last) dimension.
        bias: Whether to learn an additive bias.
        eps: Constant added to the variance for numerical stability.
    """

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))  # (C,)
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(ndim)) if bias else None  # (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` of shape ``(..., C)`` and return the same shape and dtype."""
        xf = _upcast(x)  # (..., C) -> (..., C), at least fp32
        mean = xf.mean(dim=-1, keepdim=True)  # (..., C) -> (..., 1)
        centered = xf - mean  # (..., C) - (..., 1) -> (..., C)
        var = centered.pow(2).mean(dim=-1, keepdim=True)  # (..., C) -> (..., 1), biased
        x_hat = centered * torch.rsqrt(var + self.eps)  # (..., C) * (..., 1) -> (..., C)
        y = x_hat * self.weight.to(x_hat.dtype)  # (..., C) * (C,) -> (..., C)
        if self.bias is not None:
            y = y + self.bias.to(x_hat.dtype)  # (..., C) + (C,) -> (..., C)
        return y.to(x.dtype)  # (..., C), cast back to the input dtype

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}, bias={self.bias is not None}"


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal (look-ahead) mask.

    Parameter layout matches ``nn.MultiheadAttention`` exactly:

    * ``c_attn`` is one fused ``Linear(C, 3C)`` whose output is laid out as
      ``[q | k | v]`` along the feature axis. Its weight equals MHA's
      ``in_proj_weight = [W_q; W_k; W_v]`` and its bias equals ``in_proj_bias``.
    * Head ``h`` uses features ``[h*hs, (h+1)*hs)`` of each of q, k and v.
    * ``c_proj`` corresponds to MHA's ``out_proj``.

    So weights can be copied 1:1 between the two modules for equivalence tests.

    Args:
        config: Model configuration. Uses ``n_embd``, ``n_head``, ``block_size``,
            ``dropout`` and ``bias``.
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.head_size
        self.scale = 1.0 / math.sqrt(self.head_size)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # C -> 3C
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # C -> C
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Boolean mask where True marks a *future* key (j > i) that query i must not see.
        # This is the same convention as nn.MultiheadAttention's boolean attn_mask.
        # persistent=False: it is derived from block_size, so it stays out of state_dict.
        bs = config.block_size
        future = torch.triu(torch.ones(bs, bs, dtype=torch.bool), diagonal=1)  # (bs, bs)
        self.register_buffer("causal_mask", future.view(1, 1, bs, bs), persistent=False)
        self.causal_mask: torch.Tensor  # (1, 1, bs, bs), declared for type checkers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over ``x`` of shape ``(B, T, C)`` and return shape ``(B, T, C)``."""
        B, T, C = x.shape  # (B, T, C)
        nh, hs = self.n_head, self.head_size
        if C != self.n_embd:
            raise ValueError(f"Expected last dim {self.n_embd}, got {C}")
        if T > self.causal_mask.size(-1):
            raise ValueError(f"Sequence length {T} exceeds block_size {self.causal_mask.size(-1)}")

        # Fused QKV projection, then split into heads.
        qkv = self.c_attn(x)  # (B, T, C) -> (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, 3C) -> 3 x (B, T, C)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)

        # Scale q before the matmul to keep fp16 dot products far from overflow.
        q = q * self.scale  # (B, nh, T, hs)
        att = q @ k.transpose(-2, -1)  # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)

        # Causal mask: hide future keys. The diagonal is never masked, so each row has at
        # least one finite score and softmax stays well defined (no 0/0 -> NaN).
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, bs, bs) -> (1, 1, T, T)
        att = att.masked_fill(mask, float("-inf"))  # (B, nh, T, T), broadcast over (B, nh)

        # Softmax in >= fp32 (max-subtracted internally), then back to the value dtype.
        att = torch.softmax(_upcast(att), dim=-1).to(v.dtype)  # (B, nh, T, T)
        att = self.attn_dropout(att)  # (B, nh, T, T)

        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2)  # (B, nh, T, hs) -> (B, T, nh, hs)
        y = y.contiguous().view(B, T, C)  # (B, T, nh, hs) -> (B, T, C), re-assemble heads
        y = self.resid_dropout(self.c_proj(y))  # (B, T, C) -> (B, T, C)
        return y


class FeedForward(nn.Module):
    """Position-wise MLP: ``Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout``.

    Args:
        config: Model configuration. Uses ``n_embd``, ``dropout`` and ``bias``.
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # C -> 4C
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # 4C -> C
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to ``x`` of shape ``(B, T, C)`` and return shape ``(B, T, C)``."""
        x = self.c_fc(x)  # (B, T, C) -> (B, T, 4C)
        x = F.gelu(x)  # (B, T, 4C), exact erf-based GELU
        x = self.c_proj(x)  # (B, T, 4C) -> (B, T, C)
        x = self.dropout(x)  # (B, T, C)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: ``x + Attn(LN(x))`` then ``x + MLP(LN(x))``.

    Pre-LN keeps an identity path through the residual stream, which gives
    well-behaved gradients at initialisation without needing a long warmup.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``x`` of shape ``(B, T, C)`` and return shape ``(B, T, C)``."""
        x = x + self.attn(self.ln_1(x))  # (B, T, C) + (B, T, C) -> (B, T, C)
        x = x + self.mlp(self.ln_2(x))  # (B, T, C) + (B, T, C) -> (B, T, C)
        return x


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------


class MiniGPT(nn.Module):
    """Decoder-only Transformer language model with learned token and position embeddings.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token table (V, C)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position table (block_size, C)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # C -> V

        if config.tie_weights:
            # Both use a (V, C) matrix: an embedding row lookup and a logit projection.
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 scaled init: each block adds two residual branches, so the variance of
        # the residual stream grows with depth. Scaling the output projections by
        # 1/sqrt(2 * n_layer) keeps it roughly constant at initialisation.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 initialisation: N(0, 0.02) weights, zero biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # LayerNorm initialises itself to weight=1 and bias=0 in its constructor.

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count the trainable parameters (shared tied weights are counted once).

        Args:
            non_embedding: Exclude position embeddings. Token embeddings are also
                excluded when they are *not* tied, because then they are a pure lookup
                table. When tied they double as the output layer, so they are counted.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
            if not self.config.tie_weights:
                n -= self.wte.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute next-token logits and, if targets are given, the mean cross-entropy loss.

        Logits are returned for every position (not just the last). The causal-masking
        tests rely on this.

        Args:
            idx: Token ids of shape ``(B, T)``, dtype long, with ``T <= block_size``.
            targets: Optional next-token ids of shape ``(B, T)``. Positions equal to
                ``-100`` are ignored by the loss.

        Returns:
            ``(logits, loss)``. ``logits`` has shape ``(B, T, V)``. ``loss`` is a scalar in
            at least float32, or ``None`` if no targets were given.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
        B, T = idx.shape  # (B, T)
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        pos = torch.arange(T, dtype=torch.long, device=idx.device)  # (T,)
        tok_emb = self.wte(idx)  # (B, T) -> (B, T, C)
        pos_emb = self.wpe(pos)  # (T,) -> (T, C)
        x = self.drop(tok_emb + pos_emb)  # (B, T, C) + (T, C) -> (B, T, C), broadcast over B

        for block in self.blocks:
            x = block(x)  # (B, T, C) -> (B, T, C)

        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) -> (B, T, V)

        loss: torch.Tensor | None = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must match idx shape {tuple(idx.shape)}"
                )
            # Upcast so log-softmax over V runs in >= fp32 even under fp16 autocast.
            loss = F.cross_entropy(
                _upcast(logits).reshape(B * T, -1),  # (B, T, V) -> (B*T, V)
                targets.reshape(B * T),  # (B, T) -> (B*T,)
                ignore_index=-100,
            )  # -> () scalar
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens after the prompt ``idx``.

        Call ``model.eval()`` first so dropout is disabled.

        Args:
            idx: Prompt token ids of shape ``(B, T0)``.
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature. ``0.0`` means greedy (argmax) decoding.
            top_k: If set, sample only from the ``top_k`` most likely tokens.
            generator: Optional RNG, for reproducible sampling.

        Returns:
            Token ids of shape ``(B, T0 + max_new_tokens)``.
        """
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]  # (B, T) -> (B, T'), T' <= block_size
            logits, _ = self(idx_cond)  # (B, T') -> (B, T', V)
            logits = logits[:, -1, :].float()  # (B, T', V) -> (B, V), last position only

            if temperature == 0.0:
                idx_next = logits.argmax(dim=-1, keepdim=True)  # (B, V) -> (B, 1)
            else:
                logits = logits / temperature  # (B, V)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1:]  # (B, V) -> (B, 1)
                    logits = logits.masked_fill(logits < kth, float("-inf"))  # (B, V)
                probs = torch.softmax(logits, dim=-1)  # (B, V)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T) + (B, 1) -> (B, T+1)
        return idx
