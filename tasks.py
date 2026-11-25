"""Make-free task runner that mirrors the Makefile targets (standard library only).

Usage (any shell, including PowerShell and cmd):
    python tasks.py setup [--torch-index URL]
    python tasks.py test  [pytest args...]
    python tasks.py train [trainer args...]      e.g. -o training.max_iters=500,
                                                 --resume, or --force
    python tasks.py sample [sampler args...]     e.g. --prompt "ROMEO:" --tokens 300
    python tasks.py ui [ui args...]              browser playground, e.g. --open
    python tasks.py lint
    python tasks.py clean
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
PYTHON_VERSION = "3.12"
CONFIG = "configs/default.yaml"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


def run(*cmd: str | Path) -> None:
    """Run a command (argument list, no shell) from the project root; exit on failure."""
    args = [str(c) for c in cmd]
    print("+", " ".join(args), flush=True)
    result = subprocess.run(args, cwd=ROOT, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def require_venv() -> None:
    if not PY.exists():
        sys.exit(f"{PY} not found. Run `python tasks.py setup` first.")


def setup(args: list[str]) -> None:
    torch_index = DEFAULT_TORCH_INDEX
    if args[:1] == ["--torch-index"] and len(args) == 2:
        torch_index = args[1]
    elif args:
        sys.exit("usage: python tasks.py setup [--torch-index URL]")
    if shutil.which("uv") is None:
        sys.exit("uv is required: https://docs.astral.sh/uv/getting-started/installation/")
    run("uv", "venv", VENV, "--python", PYTHON_VERSION, "--allow-existing")
    run("uv", "pip", "install", "--python", PY, "torch", "--index-url", torch_index)
    run("uv", "pip", "install", "--python", PY, "-e", ".[dev]")
    run(PY, "-c", "import torch; print('torch', torch.__version__, '| CUDA build', "
                  "torch.version.cuda, '| GPU available', torch.cuda.is_available())")


def test(args: list[str]) -> None:
    require_venv()
    run(PY, "-m", "pytest", *args)


def train(args: list[str]) -> None:
    require_venv()
    run(PY, "-m", "mini_gpt.train", "--config", CONFIG, *args)


def sample(args: list[str]) -> None:
    require_venv()
    run(PY, "-m", "mini_gpt.sample", *args)


def ui(args: list[str]) -> None:
    require_venv()
    try:
        run(PY, "-m", "mini_gpt.ui", *args)
    except KeyboardInterrupt:  # Ctrl+C stops the server; no traceback from this wrapper
        pass


def lint(args: list[str]) -> None:
    require_venv()
    run(PY, "-m", "ruff", "check", "src", "tests", *args)


def clean(args: list[str]) -> None:
    for name in (".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    for path in [*ROOT.glob("src/*.egg-info"), *ROOT.glob("src/**/__pycache__"),
                 *ROOT.glob("tests/**/__pycache__")]:
        shutil.rmtree(path, ignore_errors=True)


TASKS = {"setup": setup, "test": test, "train": train, "sample": sample, "ui": ui,
         "lint": lint, "clean": clean}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        sys.exit(__doc__)
    TASKS[sys.argv[1]](sys.argv[2:])
