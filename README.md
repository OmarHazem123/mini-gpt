# Mini-GPT: a from-scratch decoder-only Transformer in PyTorch

A GPT-style character-level language model trained on TinyShakespeare. It is built
from core PyTorch: `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.Dropout` and raw
tensor ops. Multi-head causal self-attention and LayerNorm are written by hand. There is
no `nn.Transformer`, `nn.MultiheadAttention`, `nn.LayerNorm` or
`F.scaled_dot_product_attention`.

The focus is **numerical engineering**:

* precision-safe kernels;
* a mixed-precision training loop with NaN/Inf detection;
* tests that prove causality and equivalence with PyTorch's reference attention;
* benchmarks measured on real hardware that changed a design decision. On a GPU without
  tensor cores, float16 turned out **3.7× slower** than float32, so the default
  precision is now picked per GPU.

---

## Highlights

- **Hand-written attention, verified against PyTorch.**
  - The parameter layout matches `nn.MultiheadAttention` exactly, so weights copy 1:1.
  - Outputs agree to `1e-6` in float32 and `1e-12` in float64, across 18 shape/bias
    cases.
  - Gradients agree as well.
- **Causality proven, not assumed.**
  - Changing token `t+1` leaves logits at every position `≤ t` *bit-identical*.
  - The gradient of the attention output at `t` with respect to any future input is
    exactly zero.
- **Mixed precision done properly.**
  - `torch.amp.autocast` with a `GradScaler` for float16.
  - LayerNorm, softmax and the loss always run in at least float32.
  - Overflowing steps are skipped, the loss scale backs off, and the offending
    parameters are named in the log.
- **Hardware-aware precision.** `amp_dtype: auto` picks:
  - bfloat16 on Ampere and newer;
  - float16 on tensor-core GPUs (Volta, Turing RTX);
  - float32 on GPUs without tensor cores (GTX 16xx, Pascal).
- **Reproducible.**
  - Seeded RNGs, deterministic CUDA kernels, and a device-independent data order.
  - Two fresh CUDA runs give identical eval losses.
- **Safe checkpoints.**
  - Writes are atomic and load with `weights_only=True`.
  - Checkpoints include the RNG state, so a resumed run continues the same random
    sequence.
  - A new run refuses to overwrite an existing one unless you pass `--resume` or
    `--force`.
- **80 pytest tests.** They run on CPU, plus 3 extra CUDA tests, in about 30 s.

---

## Architecture

```
token ids (B, T)
  │
  ├── wte: Embedding(V, C) ─────┐
  └── wpe: Embedding(T_max, C) ─┴── add ── dropout
                                     │
                ┌────────────────────▼─────────────────────┐
                │  TransformerBlock × n_layer              │
                │    x = x + CausalSelfAttention(LN(x))    │
                │    x = x + FeedForward(LN(x))            │
                └────────────────────┬─────────────────────┘
                                     │
                             LayerNorm (ln_f)
                                     │
                   lm_head: Linear(C, V)  (tied to wte)
                                     │
                             logits (B, T, V)
```

**Causal self-attention.** Every op in [`model.py`](src/mini_gpt/model.py) carries a
shape comment.

```
x (B, T, C) ── c_attn: Linear(C, 3C) ── split ── q, k, v: (B, T, C) ── view/transpose ── (B, nh, T, hs)
q · 1/√hs  (scaled before the matmul, keeping fp16 dot products far from overflow)
scores = q @ kᵀ                                   (B, nh, T, T)
scores.masked_fill(future, -inf)                  causal mask; diagonal never masked
softmax in ≥ fp32 ── dropout ── @ v               (B, nh, T, hs)
merge heads ── c_proj: Linear(C, C) ── dropout    (B, T, C)
```

| Default model (`configs/default.yaml`) | Value |
|---|---|
| Layers × heads × width | 6 × 6 × 384 (head size 64) |
| Context length | 256 characters |
| Vocabulary | 65 characters (taken from the tokenizer) |
| Parameters | 10.77 M (10.67 M excluding position embeddings) |
| Style | Pre-LayerNorm, GELU MLP (4×), dropout 0.2, tied embeddings, GPT-2 init with residual projections scaled by `1/√(2·n_layer)` |

### Numerical-engineering decisions

| Risk | Mitigation | Verified by |
|---|---|---|
| fp16 overflow / precision loss in LayerNorm statistics | Mean and variance computed in ≥ fp32, output cast back | fp16 inputs ≈ 1000 (x² > fp16 max) stay finite and match `F.layer_norm` |
| Overflow in `q @ kᵀ` under fp16 | Scale `q` by `1/√hs` *before* the matmul | fp16 autocast forward/backward finite on CUDA |
| `NaN` from a fully masked softmax row | Fill with `-inf`; every row keeps its diagonal | Causal tests (exact equality) |
| Low-precision softmax and cross-entropy | Upcast to ≥ fp32. A `_upcast` helper never *downcasts* fp64, so fp64 reference tests stay exact | fp64 equivalence at `1e-12` |
| Small gradients underflowing in fp16 | `GradScaler`, unscaled before measuring the norm and clipping | Injected `inf`: step skipped, scale halves, next step recovers |
| One NaN hidden by clipping (`clip_grad_norm_` spreads it to every grad) | Per-parameter norms computed *before* clipping; bad parameters named in the log | Injected `NaN`: step skipped, parameter named, weights untouched |
| Silent divergence | Abort after 25 consecutive non-finite steps | — |
| Run-to-run noise in validation curves | Every evaluation uses the same fixed-seed batches | — |

---

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12) and, for GPU
training, an NVIDIA driver that supports CUDA 12.6.

```bash
# 1. Create .venv with the CUDA 12.6 build of PyTorch, the package and dev tools
python tasks.py setup
#    CPU-only machines: python tasks.py setup --torch-index https://download.pytorch.org/whl/cpu
#    macOS:             python tasks.py setup --torch-index https://pypi.org/simple

# 2. Run the test suite
python tasks.py test

# 3. Train (TinyShakespeare is downloaded to data/ on the first run)
python tasks.py train

# Override any config value; resume or restart a run
python tasks.py train -o training.max_iters=500 -o optimizer.lr=6e-4
python tasks.py train --resume        # continue from checkpoints/last.pt
python tasks.py train --force         # start over, replacing checkpoints/
```

`tasks.py` uses only the standard library and works in any shell. The same targets are
in the `Makefile` for GNU Make users (Linux/macOS, or Git Bash on Windows after
`choco install make`):

```bash
make setup    make test ARGS="-k causal -v"    make train ARGS="--resume"    make lint
```

A run writes to `checkpoints/`:

* `best.pt`: lowest validation loss so far.
* `last.pt`: the latest checkpoint, used by `--resume`.
* `metrics.jsonl`: per-step loss, learning rate, gradient norm, loss scale, throughput and
  peak memory, plus eval records.
* `train.log`: the run's log.

---

## Training pipeline ([`train.py`](src/mini_gpt/train.py))

1. **Config.** YAML is validated into frozen dataclasses. Unknown keys and
   PyYAML's `1e-3`-as-string trap raise clear errors.
2. **Precision.** `torch.amp.autocast` plus `GradScaler` for float16, chosen per GPU
   (see above).
3. **Gradient accumulation.** Each micro-batch loss is divided by the number of
   micro-batches, so the gradient equals a full-batch gradient. A test checks this to
   `1e-12` in fp64.
4. **Learning-rate schedule.** Linear warmup, then cosine decay to `min_lr`. The decay
   horizon defaults to `max_iters`.
5. **Optimizer step.** Unscale → measure per-parameter and global norms →
   `clip_grad_norm_` → `scaler.step` → `scaler.update` (the last runs every step).
6. **AdamW.** Weight decay on matrices only; biases and LayerNorm gains are excluded.
7. **Evaluation and checkpoints.** Periodic eval, atomic `best.pt`/`last.pt`, and a text
   sample at the end.

---

## Benchmarks (measured on my hardware)

**Setup:**
- GPU: NVIDIA GeForce GTX 1650 Ti, 4 GB (Turing TU117, compute capability 7.5, **no
  tensor cores**), driver 591.59.
- Software: PyTorch 2.14.1+cu126, Python 3.12, Windows 11.
- Workload: the default model at context 256.
- Memory is from `torch.cuda.max_memory_allocated` / `max_memory_reserved`, evaluation
  included.

### 1. Precision trade-off: float32 wins on a GPU without tensor cores

Micro-batch 16 × 4 accumulation steps = 16,384 tokens per optimizer step, 200 steps:

| | **float32** (default here) | float16 + GradScaler |
|---|---|---|
| Throughput | **18.8k tok/s** | 5.1k tok/s |
| Time per optimizer step | 0.87 s | 3.2 s |
| Peak memory (allocated / reserved) | 1275 / 1320 MiB | **973 / 1018 MiB** |
| Validation loss, step 0 → 100 → 200 | 4.300 → 2.471 → 2.277 | 4.300 → 2.471 → 2.280 |
| Non-finite (skipped) steps | 0 | 0 (loss scale stayed at 65536) |
| Wall time, 200 steps incl. 3 evaluations | 3.3 min | 11.5 min |
| Projected 5000-step run (training steps only) | ≈ 73 min | ≈ 4.4 h |

The loss curves are almost identical, so float16 is numerically sound; it is just slower
on this GPU. The cause is the matmul kernel itself:

| cuBLAS GEMM, model-realistic shapes | float32 | float16 |
|---|---|---|
| QKV projection (4096×384)·(384×1152) | 2.40 ms (1.51 TFLOP/s) | 8.77 ms (0.41 TFLOP/s) |
| MLP up-projection (4096×384)·(384×1536) | 2.66 ms (1.82 TFLOP/s) | 11.73 ms (0.41 TFLOP/s) |
| Square 4096³ | 76.1 ms (1.81 TFLOP/s) | 331.3 ms (0.41 TFLOP/s) |
| Attention scores, batched (96×256×64)·(96×64×256) | 0.46 ms | 2.16 ms |

Without tensor cores, float16 GEMMs hit a flat ~0.41 TFLOP/s ceiling, 4.4× below fp32.
That is why `auto` picks float32 on these GPUs and float16/bfloat16 only where tensor
cores exist.

Deterministic mode costs nothing measurable: 5151 vs 5126 tok/s with it on and off.

### 2. Micro-batch VRAM scaling (float16, 20 steps each, fresh process per size)

| Micro-batch × accumulation | Peak allocated | Peak reserved | Free after run (of 4096 MiB) | Throughput |
|---|---|---|---|---|
| 8 × 8 | 621 MiB | 648 MiB | 2615 MiB | 5.0k tok/s |
| **16 × 4** (default) | **973 MiB** | **1016 MiB** | **2247 MiB** | **5.2k tok/s** |
| 24 × 2 ¹ | 1318 MiB | 1342 MiB | 1921 MiB | 5.2k tok/s |
| 32 × 2 | 1671 MiB | 1718 MiB | 1545 MiB | 5.2k tok/s |
| 48 × 1 ¹ ² | 2333 MiB | 2542 MiB | 721 MiB | 5.3k tok/s |
| 64 × 1 | 3028 MiB | 3178 MiB | **85 MiB** | 5.3k tok/s |

¹ Effective batch 48, since 64 is not divisible by the micro-batch.
² One float16 overflow at step 6. It was detected and skipped, the loss scale went from
65536 to 32768, and training continued.

Memory grows by about 43 MiB per sequence, mostly the explicit `(B, n_head, T, T)`
attention matrices. Throughput is almost flat because the GPU is compute-bound, so the
conservative micro-batch of 16 costs about 3% throughput against 64 and leaves about
2.2 GB of headroom. In float32 the same setting peaks at 1275 MiB.

---

## Test suite ([`tests/`](tests))

```bash
python tasks.py test          # 80 tests: CPU-only machines skip the 3 CUDA tests
```

| Area | What is checked |
|---|---|
| Shapes | Logits `(B, T, V)` and scalar loss over 6 batch/length combinations; attention shape; `T > block_size` rejected |
| Causal masking | Changing token `t+1` leaves logits `≤ t` bit-identical (and does change `t+1`); rewriting the whole future never leaks; `∂out[t]/∂x[j>t] = 0` exactly |
| Numerical stability | Finite loss and gradients for every parameter in fp32, CPU bf16 and CUDA fp16; initial loss ≈ ln(V); LayerNorm stable on large fp16 inputs |
| MHA equivalence | `torch.allclose` against `nn.MultiheadAttention` with identical weights: 18 fp32 cases, fp64 at `1e-12`, and gradients |
| Training mechanics | LR schedule, gradient accumulation = full batch, NaN step skipping, clipping, fp16 scaler back-off and recovery |
| Infrastructure | Config validation and overrides, tokenizer, batching, checkpoint and RNG round-trip, bitwise-reproducible seeding, hardware-aware precision on 10 simulated GPUs |
| End to end | Training on a synthetic corpus (CPU and CUDA fp16), resume, overwrite protection, `--force` |

---

## Project layout

```
configs/default.yaml        Hyperparameters (model / data / training / optimizer / scheduler)
src/mini_gpt/model.py       LayerNorm, CausalSelfAttention, FeedForward, TransformerBlock, MiniGPT
src/mini_gpt/train.py       Config dataclasses, LR schedule, AMP training loop, CLI
src/mini_gpt/utils.py       Seeding, device/precision selection, checkpoints, tokenizer, data
tests/                      pytest suite (test_model.py, test_training.py)
tasks.py / Makefile         setup | test | train | lint | clean
LICENSE                     MIT
```

## Limitations and next steps

- **No full-length result yet.** The numbers above come from 200-step runs. A full
  5000-step run (about 73 min in float32 on this GPU) is the next step.
- **Explicit attention.** It materialises `(B, nh, T, T)`, by design for this project.
  A fused or flash kernel would cut memory roughly linearly in `T`.
- **No `torch.compile`.** Triton is not available on native Windows.
- **Partial `auto` detection.** Turing GPUs without tensor cores are identified by device
  name, which covers the GTX 16xx, MX450/550 and T400–T2000 cards. `amp_dtype` can
  always be set explicitly.

---

## References and acknowledgments

This is an independent implementation, written from scratch for education and
engineering practice. No code was copied from other implementations. The architecture
and the training setup follow the published work below.

- **Architecture: GPT-2.** A. Radford, J. Wu, R. Child, D. Luan, D. Amodei and
  I. Sutskever, *Language Models are Unsupervised Multitask Learners*, OpenAI, 2019.
  The following come from GPT-2:
  - pre-LayerNorm blocks, learned position embeddings and tied input/output embeddings;
  - residual projections initialised with a `1/√(2·n_layer)` scale;
  - the parameter names (`wte`, `wpe`, `c_attn`, `c_proj`, `ln_f`).
- **Attention.** A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. Scaled
  dot-product and multi-head attention.
- **Training setup: nanoGPT.** Andrej Karpathy,
  [nanoGPT](https://github.com/karpathy/nanoGPT). The defaults in
  `configs/default.yaml` follow its `shakespeare_char` configuration:
  - 6 layers, 6 heads, width 384, context 256, dropout 0.2;
  - AdamW at lr `1e-3` with β₂ = 0.99;
  - 100 warmup steps, then cosine decay to `1e-4` over 5000 steps;
  - an effective batch of 64 sequences, split here into 16 × 4 micro-batches to fit a
    4 GB GPU.
- **Dataset: TinyShakespeare.** About 40,000 lines from Shakespeare's plays, compiled by
  Andrej Karpathy for [char-rnn](https://github.com/karpathy/char-rnn). The text is in
  the public domain. It is downloaded to `data/` on the first run and is not stored in
  this repository.
- **PyTorch.** `nn.MultiheadAttention` and `F.layer_norm` are used only in the tests,
  as reference implementations.

## License

MIT. See [LICENSE](LICENSE).
