"""Shared pytest configuration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_global_torch_state() -> Iterator[None]:
    """``set_seed(deterministic=True)`` flips process-wide flags; undo them after each test."""
    det = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_det = torch.backends.cudnn.deterministic
    cudnn_bench = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
