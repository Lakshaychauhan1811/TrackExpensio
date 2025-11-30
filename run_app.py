"""
Convenience launcher for the TrackExpensio .

Usage:
    python run_app.py                # installs deps (if needed) + runs uvicorn
    python run_app.py --reload       # runs with reload flag for dev
    python run_app.py --skip-install # skip requirements installation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_SCRIPTS = ROOT / "venv" / ("Scripts" if os.name == "nt" else "bin")
DEFAULT_PYTHON = VENV_SCRIPTS / ("python.exe" if os.name == "nt" else "python")
REQUIREMENTS = ROOT / "requirements.txt"


def run_command(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def ensure_dependencies(python_bin: Path, skip: bool) -> None:
    if skip or not REQUIREMENTS.exists():
        return
    print(f"📦 Installing dependencies with {python_bin} ...")
    run_command([str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def start_server(python_bin: Path, host: str, port: str, reload: bool) -> None:
    cmd = [
        str(python_bin),
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        host,
        "--port",
        port,
    ]
    if reload:
        cmd.append("--reload")
    print(f"🚀 Starting TrackExpensio on http://{host}:{port}")
    run_command(cmd)


def detect_python() -> Path:
    if DEFAULT_PYTHON.exists():
        return DEFAULT_PYTHON
    return Path(sys.executable)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the TrackExpensio API server")
    parser.add_argument(
        "--host",
        default=os.getenv("FASTAPI_HOST", "127.0.0.1"),
        help="Host interface (default: 127.0.0.1 or FASTAPI_HOST env)",
    )
    parser.add_argument(
        "--port",
        default=os.getenv("FASTAPI_PORT", "8080"),
        help="Port number (default: 8080 or FASTAPI_PORT env)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Run uvicorn with auto-reload (useful for development)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install step (assumes deps already installed)",
    )
    args = parser.parse_args(argv)

    python_bin = detect_python()
    ensure_dependencies(python_bin, args.skip_install)
    start_server(python_bin, args.host, args.port, args.reload)


if __name__ == "__main__":
    main()

