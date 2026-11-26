"""Draw the README charts as self-contained SVG files (standard library only).

    python scripts/make_figures.py [--metrics checkpoints/metrics.jsonl] [--out docs/images]

* ``loss_curve.svg``: train and validation loss of a training run. Both lines use the
  periodic eval records in ``metrics.jsonl`` (100 batches per split, dropout off), the
  same numbers as the README's results table.
* ``precision.svg``: float32 vs float16 on the GTX 1650 Ti. The values are the measured
  numbers from the README's "Benchmarks" tables, copied here verbatim.

Each SVG has its own light and dark colours (``prefers-color-scheme``) on an opaque
background, so it stays readable on either GitHub theme.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

# Chart colours: surfaces, inks and the first two categorical slots, light and dark.
LIGHT = {"surface": "#ffffff", "border": "rgba(11,11,11,0.10)", "ink": "#191919",
         "ink2": "#666666", "grid": "#e7e7e7", "axis": "#c2c2c2",
         "s1": "#257de0", "s2": "#f47230"}
DARK = {"surface": "#191919", "border": "rgba(255,255,255,0.10)", "ink": "#efefef",
        "ink2": "#9c9c9c", "grid": "#303030", "axis": "#474747",
        "s1": "#338ced", "s2": "#e76624"}

# Measured on the GTX 1650 Ti; README "Benchmarks", section 1 (both tables).
THROUGHPUT_K_TOK_S = {"float32": 18.8, "float16 + GradScaler": 5.1}
GEMM_TFLOPS = {  # shape: (float32, float16)
    "QKV projection": (1.51, 0.41),
    "MLP up-projection": (1.82, 0.41),
    "Square 4096³": (1.81, 0.41),
}


def _css(c: dict[str, str]) -> str:
    return (f".bg{{fill:{c['surface']};stroke:{c['border']}}}"
            f".ink{{fill:{c['ink']}}}.ink2{{fill:{c['ink2']}}}"
            f".grid{{stroke:{c['grid']}}}.axis{{stroke:{c['axis']}}}"
            f".s1{{stroke:{c['s1']}}}.s2{{stroke:{c['s2']}}}"
            f".f1{{fill:{c['s1']}}}.f2{{fill:{c['s2']}}}.ring{{stroke:{c['surface']}}}")


def _svg(width: int, height: int, title: str, desc: str, body: list[str]) -> str:
    style = (f"text{{font-family:{FONT};font-variant-numeric:tabular-nums}}"
             f".t{{font-size:16px;font-weight:600}}.st{{font-size:13px}}"
             f".lb{{font-size:12px}}.pt{{font-size:13px;font-weight:600}}"
             f"{_css(LIGHT)}@media (prefers-color-scheme: dark){{{_css(DARK)}}}")
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{_esc(title)}</title>',
        f'<desc id="desc">{_esc(desc)}</desc>',
        f"<style>{style}</style>",
        f'<rect class="bg" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8"/>',
        *body,
        "</svg>",
        "",
    ])


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x: float, y: float, s: str, cls: str, anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
            f'text-anchor="{anchor}">{_esc(s)}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="{cls}" stroke-width="1"/>')


def _legend(x: float, y: float, items: list[tuple[str, str]], kind: str) -> list[str]:
    """Legend row: a short line (kind="line") or square (kind="box") key, then the label."""
    out = []
    for series, label in items:
        if kind == "line":
            out.append(f'<line x1="{x:.1f}" y1="{y - 4:.1f}" x2="{x + 16:.1f}" y2="{y - 4:.1f}" '
                       f'class="{series}" stroke-width="2" stroke-linecap="round"/>')
        else:
            fill = series.replace("s", "f")
            out.append(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="12" height="12" rx="2" '
                       f'class="{fill}"/>')
        out.append(_text(x + 22, y, label, "lb ink2"))
        x += 22 + 7.2 * len(label) + 20
    return out


def loss_curve(evals: list[dict]) -> str:
    W, H = 760, 430
    x0, x1, y0, y1 = 56, 630, 100, 378  # plot box
    max_iter, y_max = 5000, 4.5
    max_iter = max(max_iter, evals[-1]["iter"])

    def sx(it: float) -> float:
        return x0 + (x1 - x0) * it / max_iter

    def sy(v: float) -> float:
        return y1 - (y1 - y0) * v / y_max

    body = [
        _text(24, 34, "Training and validation loss over the full 5000-step run", "t ink"),
        _text(24, 56, "Mean cross-entropy per character, measured every 250 steps on 100 "
                      "batches per split. Default config, float32, GTX 1650 Ti.", "st ink2"),
        *_legend(24, 82, [("s1", "Train"), ("s2", "Validation")], "line"),
    ]
    for v in range(0, 5):
        body.append(_line(x0, sy(v), x1, sy(v), "axis" if v == 0 else "grid"))
        body.append(_text(x0 - 8, sy(v) + 4, f"{v}", "lb ink2", "end"))
    for it in range(0, max_iter + 1, 1000):
        body.append(_text(sx(it), y1 + 20, f"{it:,}", "lb ink2", "middle"))
    body.append(_text((x0 + x1) / 2, y1 + 42, "Optimizer step", "lb ink2", "middle"))

    for key, series in (("train", "s1"), ("val", "s2")):
        pts = " L".join(f"{sx(r['iter']):.1f},{sy(r[key]):.1f}" for r in evals)
        body.append(f'<path d="M{pts}" fill="none" class="{series}" stroke-width="2" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>')

    best = min(evals, key=lambda r: r["val"])
    last = evals[-1]
    bx, by = sx(best["iter"]), sy(best["val"])
    body.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" class="f2 ring" '
                f'stroke-width="2"/>')
    body.append(_text(bx, by - 30, f"Best validation {best['val']:.3f}", "lb ink", "middle"))
    body.append(_text(bx, by - 15, f"at step {best['iter']:,} (best.pt)", "lb ink2", "middle"))
    body.append(_text(x1 + 10, sy(last["val"]) + 4, f"Validation {last['val']:.3f}", "lb ink"))
    body.append(_text(x1 + 10, sy(last["train"]) + 4, f"Train {last['train']:.3f}", "lb ink"))
    for i, s in enumerate((f"After step {best['iter']:,} the model overfits:",
                           "training loss keeps falling,",
                           "validation loss rises.")):
        body.append(_text(sx(3150), sy(3.1) + 17 * i, s, "lb ink2"))

    desc = (f"Line chart. Train and validation loss fall from about {evals[0]['train']:.1f} "
            f"at step 0. Validation loss is lowest, {best['val']:.3f}, at step "
            f"{best['iter']}, then rises to {last['val']:.3f} at step {last['iter']} while "
            f"train loss falls to {last['train']:.3f}.")
    return _svg(W, H, "Training and validation loss", desc, body)


def _bar(x0: float, x1: float, y: float, h: float, cls: str, r: float = 4.0) -> str:
    """Horizontal bar, square at the baseline, rounded data end."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" '
            f'class="{cls}"/>')


def _bar_panel(top: float, title: str, rows: list[tuple[str, list[tuple[float, str]]]],
               ticks: list[float], fmt: str, unit_max: float) -> tuple[list[str], float]:
    # fmt formats the values at the bar ends; tick labels use "g" (0, 0.5, 1, 1.5).
    """One panel of horizontal bars. rows: (row label, [(value, series fill), ...])."""
    bx0, bx1, bar_h, gap, group_gap = 200, 640, 18, 2, 14
    body = [_text(24, top, title, "pt ink")]
    y = top + 14
    grid_top = y

    def sx(v: float) -> float:
        return bx0 + (bx1 - bx0) * v / unit_max

    bars = []
    for label, values in rows:
        block = len(values) * bar_h + (len(values) - 1) * gap
        body.append(_text(bx0 - 12, y + block / 2 + 4, label, "lb ink", "end"))
        for value, fill in values:
            bars.append(_bar(bx0, sx(value), y, bar_h, fill))
            bars.append(_text(sx(value) + 6, y + bar_h / 2 + 4, format(value, fmt), "lb ink"))
            y += bar_h + gap
        y += group_gap - gap
    grid_bottom = y - group_gap + 6
    for t in ticks:
        body.append(_line(sx(t), grid_top - 4, sx(t), grid_bottom,
                          "axis" if t == 0 else "grid"))
        body.append(_text(sx(t), grid_bottom + 16, format(t, "g"), "lb ink2", "middle"))
    return body + bars, grid_bottom + 16


def precision() -> str:
    W = 760
    body = [
        _text(24, 34, "float32 vs float16 on a GPU without tensor cores (GTX 1650 Ti)",
              "t ink"),
        _text(24, 56, "Higher is better. Measured on this machine; the same numbers are in "
                      "the tables below.", "st ink2"),
        *_legend(24, 82, [("s1", "float32"), ("s2", "float16")], "box"),
    ]
    fp32, fp16 = THROUGHPUT_K_TOK_S["float32"], THROUGHPUT_K_TOK_S["float16 + GradScaler"]
    panel, bottom = _bar_panel(
        118, f"Training throughput, thousand tokens/s: float32 is {fp32 / fp16:.1f}× faster",
        [("Full training step", [(fp32, "f1"), (fp16, "f2")])],
        [0, 5, 10, 15, 20], ".1f", 20)
    body += panel
    rows = [(shape, [(a, "f1"), (b, "f2")]) for shape, (a, b) in GEMM_TFLOPS.items()]
    sq32, sq16 = GEMM_TFLOPS["Square 4096³"]
    panel, bottom = _bar_panel(
        bottom + 40, f"Matrix multiply, TFLOP/s: float16 is stuck near {sq16} "
                     f"({sq32 / sq16:.1f}× below float32)",
        rows, [0, 0.5, 1.0, 1.5, 2.0], ".2f", 2.0)
    body += panel
    H = round(bottom + 22)
    desc = (f"Two bar charts. Training throughput: float32 {fp32}k tokens/s, float16 "
            f"{fp16}k. Matrix multiply TFLOP/s, float32 vs float16: "
            + "; ".join(f"{k} {a} vs {b}" for k, (a, b) in GEMM_TFLOPS.items()) + ".")
    return _svg(W, H, "float32 vs float16 speed", desc, body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=ROOT / "checkpoints/metrics.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/images")
    args = parser.parse_args(argv)
    if not args.metrics.is_file():
        print(f"error: {args.metrics} not found. Train first (python tasks.py train) "
              f"or pass --metrics PATH.", file=sys.stderr)
        return 1
    evals: dict[int, dict] = {}
    for line in args.metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("type") == "eval":
                evals[record["iter"]] = record  # last record per iter wins (resumed runs)
    if len(evals) < 2:
        print(f"error: {args.metrics} has fewer than 2 eval records.", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name, svg in (("loss_curve.svg", loss_curve([evals[k] for k in sorted(evals)])),
                      ("precision.svg", precision())):
        (args.out / name).write_text(svg, encoding="utf-8", newline="\n")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
