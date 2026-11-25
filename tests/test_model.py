"""Correctness tests for the Mini-GPT model.

1. Output shapes across batch sizes and sequence lengths.
2. Causal masking: changing the token at ``t + 1`` must not change logits at ``<= t``.
3. Numerical stability: no NaN/Inf in loss or gradients (fp32 and mixed precision).
4. Equivalence of ``CausalSelfAttention`` with ``nn.MultiheadAttention``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.model import CausalSelfAttention, LayerNorm, MiniGPT, MiniGPTConfig

VOCAB = 65
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def make_config(**overrides: object) -> MiniGPTConfig:
    base: dict[str, object] = dict(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4,
                                   n_embd=64, dropout=0.0)
    base.update(overrides)
    return MiniGPTConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def model() -> MiniGPT:
    torch.manual_seed(0)
    return MiniGPT(make_config()).eval()


def assert_all_grads_finite(module: nn.Module) -> None:
    for name, p in module.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a NaN/Inf gradient"


# ======================================================================================
# 1. Output shapes
# ======================================================================================


@pytest.mark.parametrize(("batch_size", "seq_len"),
                         [(1, 1), (1, 32), (2, 7), (4, 16), (8, 32), (3, 31)])
def test_logits_and_loss_shapes(model: MiniGPT, batch_size: int, seq_len: int) -> None:
    idx = torch.randint(0, VOCAB, (batch_size, seq_len))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (batch_size, seq_len, VOCAB)
    assert loss is not None and loss.shape == () and loss.dtype == torch.float32


def test_loss_is_none_without_targets(model: MiniGPT) -> None:
    logits, loss = model(torch.randint(0, VOCAB, (2, 5)))
    assert logits.shape == (2, 5, VOCAB) and loss is None


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 1), (3, 5), (2, 32)])
def test_attention_preserves_shape(batch_size: int, seq_len: int) -> None:
    attn = CausalSelfAttention(make_config())
    x = torch.randn(batch_size, seq_len, 64)
    assert attn(x).shape == (batch_size, seq_len, 64)


def test_sequence_longer_than_block_size_raises(model: MiniGPT) -> None:
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_generate_appends_tokens_beyond_block_size(model: MiniGPT) -> None:
    out = model.generate(torch.zeros(2, 1, dtype=torch.long), max_new_tokens=40,
                         temperature=0.0)
    assert out.shape == (2, 41)  # context is cropped to block_size internally


# ======================================================================================
# 2. Causal masking
# ======================================================================================


@pytest.mark.parametrize("t", [0, 1, 7, 15, 30])
def test_changing_token_t_plus_1_does_not_change_logits_up_to_t(model: MiniGPT, t: int) -> None:
    torch.manual_seed(1)
    idx = torch.randint(0, VOCAB, (4, 32))
    perturbed = idx.clone()
    perturbed[:, t + 1] = (idx[:, t + 1] + 1) % VOCAB  # a different token at t+1 in every row

    with torch.no_grad():
        logits, _ = model(idx)
        logits_perturbed, _ = model(perturbed)

    # Positions 0..t cannot see t+1, so their logits must be bit-identical...
    assert torch.equal(logits_perturbed[:, : t + 1], logits[:, : t + 1])
    # ...while position t+1 sees the new token, so the perturbation is not a no-op.
    assert not torch.allclose(logits_perturbed[:, t + 1], logits[:, t + 1])


def test_rewriting_the_entire_future_does_not_leak(model: MiniGPT) -> None:
    torch.manual_seed(2)
    idx = torch.randint(0, VOCAB, (2, 32))
    with torch.no_grad():
        reference, _ = model(idx)
        for t in range(31):
            perturbed = idx.clone()
            perturbed[:, t + 1 :] = torch.randint(0, VOCAB, (2, 31 - t))
            logits, _ = model(perturbed)
            assert torch.equal(logits[:, : t + 1], reference[:, : t + 1]), f"leak at t={t}"


def test_attention_output_has_zero_gradient_wrt_future_inputs() -> None:
    """d out[:, t] / d x[:, j] must be exactly zero for every j > t."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(make_config()).eval()
    x = torch.randn(2, 16, 64, requires_grad=True)
    t = 7
    attn(x)[:, t].sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad[:, t + 1 :] == 0)
    assert torch.any(x.grad[:, : t + 1] != 0)


# ======================================================================================
# 3. Numerical stability
# ======================================================================================


def test_forward_backward_finite_fp32() -> None:
    torch.manual_seed(0)
    model = MiniGPT(make_config(dropout=0.1)).train()
    idx = torch.randint(0, VOCAB, (4, 32))
    targets = torch.randint(0, VOCAB, (4, 32))
    _, loss = model(idx, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert_all_grads_finite(model)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param("cpu", torch.bfloat16, id="cpu-bf16"),
        pytest.param("cuda", torch.float16, id="cuda-fp16", marks=requires_cuda),
    ],
)
def test_forward_backward_finite_under_autocast(device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    model = MiniGPT(make_config(dropout=0.1)).to(device).train()
    idx = torch.randint(0, VOCAB, (4, 32), device=device)
    targets = torch.randint(0, VOCAB, (4, 32), device=device)
    with torch.amp.autocast(device_type=device, dtype=dtype):
        logits, loss = model(idx, targets)
    assert logits.dtype == dtype  # the matmuls really ran in reduced precision
    assert loss is not None and loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert_all_grads_finite(model)


def test_initial_loss_is_close_to_uniform() -> None:
    """With N(0, 0.02) init the logits are near zero, so loss ~ ln(V). A big gap means
    a broken init or a scaling bug, both early warnings of training instability."""
    torch.manual_seed(0)
    model = MiniGPT(make_config(n_layer=4, n_embd=128)).eval()
    idx = torch.randint(0, VOCAB, (8, 32))
    with torch.no_grad():
        _, loss = model(idx, torch.randint(0, VOCAB, (8, 32)))
    assert loss is not None
    assert abs(loss.item() - math.log(VOCAB)) < 0.15


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_layernorm_matches_torch_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    ln = LayerNorm(64).to(dtype)
    with torch.no_grad():
        ln.weight.normal_()
        assert ln.bias is not None
        ln.bias.normal_()
    x = torch.randn(4, 8, 64, dtype=dtype) * 5 + 3
    expected = F.layer_norm(x, (64,), ln.weight, ln.bias, eps=ln.eps)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(ln(x), expected, rtol=tol, atol=tol)


def test_layernorm_fp16_large_activations_stay_finite() -> None:
    """fp16 inputs around 1000: x^2 ~ 1e6 exceeds fp16's max (65504). Statistics must be
    computed in fp32 for this to stay finite and accurate."""
    torch.manual_seed(0)
    x = (torch.randn(4, 8, 64) * 50 + 1000).half()
    y = LayerNorm(64)(x)
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), F.layer_norm(x.float(), (64,)), rtol=1e-2, atol=1e-2)


# ======================================================================================
# 4. Equivalence with nn.MultiheadAttention
# ======================================================================================


def make_attention_pair(
    n_embd: int, n_head: int, bias: bool, seed: int = 0, dtype: torch.dtype = torch.float32
) -> tuple[CausalSelfAttention, nn.MultiheadAttention]:
    """Build both modules from one seed and copy the reference weights into ours.

    The layouts match 1:1: ``in_proj_weight = [W_q; W_k; W_v]`` (3C, C) -> ``c_attn``,
    and ``out_proj`` (C, C) -> ``c_proj``.
    """
    torch.manual_seed(seed)
    ref = nn.MultiheadAttention(n_embd, n_head, dropout=0.0, bias=bias, batch_first=True)
    ours = CausalSelfAttention(make_config(n_embd=n_embd, n_head=n_head, bias=bias))
    with torch.no_grad():
        ours.c_attn.weight.copy_(ref.in_proj_weight)
        ours.c_proj.weight.copy_(ref.out_proj.weight)
        if bias:
            ours.c_attn.bias.copy_(ref.in_proj_bias)
            ours.c_proj.bias.copy_(ref.out_proj.bias)
    return ours.to(dtype).eval(), ref.to(dtype).eval()


def causal_bool_mask(seq_len: int) -> torch.Tensor:
    """(T, T) bool mask; True = blocked (nn.MultiheadAttention convention)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(("n_embd", "n_head"), [(32, 1), (64, 4), (96, 8)])
@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 1), (3, 17), (2, 32)])
def test_matches_nn_multihead_attention(
    bias: bool, n_embd: int, n_head: int, batch_size: int, seq_len: int
) -> None:
    ours, ref = make_attention_pair(n_embd, n_head, bias)
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, n_embd)
    with torch.no_grad():
        expected, _ = ref(x, x, x, attn_mask=causal_bool_mask(seq_len), need_weights=False)
        actual = ours(x)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), (
        f"max abs diff {(actual - expected).abs().max().item():.3e}"
    )


def test_matches_nn_multihead_attention_float64_tight() -> None:
    ours, ref = make_attention_pair(64, 4, bias=True, dtype=torch.float64)
    x = torch.randn(2, 32, 64, dtype=torch.float64)
    with torch.no_grad():
        expected, _ = ref(x, x, x, attn_mask=causal_bool_mask(32), need_weights=False)
        actual = ours(x)
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_gradients_match_nn_multihead_attention() -> None:
    """Backward must agree too: input gradients and every weight gradient."""
    ours, ref = make_attention_pair(64, 4, bias=True, dtype=torch.float64)
    x = torch.randn(2, 16, 64, dtype=torch.float64)
    upstream = torch.randn(2, 16, 64, dtype=torch.float64)

    x_ours = x.clone().requires_grad_(True)
    (ours(x_ours) * upstream).sum().backward()
    x_ref = x.clone().requires_grad_(True)
    out_ref, _ = ref(x_ref, x_ref, x_ref, attn_mask=causal_bool_mask(16), need_weights=False)
    (out_ref * upstream).sum().backward()

    pairs = [
        (x_ours.grad, x_ref.grad),
        (ours.c_attn.weight.grad, ref.in_proj_weight.grad),
        (ours.c_attn.bias.grad, ref.in_proj_bias.grad),
        (ours.c_proj.weight.grad, ref.out_proj.weight.grad),
        (ours.c_proj.bias.grad, ref.out_proj.bias.grad),
    ]
    for actual, expected in pairs:
        assert actual is not None and expected is not None
        assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)
