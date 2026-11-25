"""Mini-GPT: a from-scratch decoder-only Transformer in core PyTorch."""

from mini_gpt.model import (
    CausalSelfAttention,
    FeedForward,
    LayerNorm,
    MiniGPT,
    MiniGPTConfig,
    TransformerBlock,
)

__all__ = [
    "CausalSelfAttention",
    "FeedForward",
    "LayerNorm",
    "MiniGPT",
    "MiniGPTConfig",
    "TransformerBlock",
]

__version__ = "0.1.0"
