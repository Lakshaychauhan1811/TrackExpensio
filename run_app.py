"""
Convenience launcher for the TrackExpensio .

Usage:
    python run_app.py                # installs deps (if needed) + runs uvicorn
    python run_app.py --reload       # runs with reload flag for dev
    python run_app.py --skip-install # skip requirements installation
    python run_app.py --restart      # stop old server on this port, then start fresh
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_SCRIPTS = ROOT / "venv" / ("Scripts" if os.name == "nt" else "bin")
DEFAULT_PYTHON = VENV_SCRIPTS / ("python.exe" if os.name == "nt" else "python")
REQUIREMENTS = ROOT / "requirements.txt"


def default_host() -> str:
    if os.getenv("FASTAPI_HOST"):
        return os.getenv("FASTAPI_HOST")
    return "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"


def default_port() -> str:
    return os.getenv("FASTAPI_PORT") or os.getenv("PORT") or "8080"


def run_command(cmd: list[str], env: dict | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.check_call(cmd, env=merged)


def ensure_dependencies(python_bin: Path, skip: bool) -> None:
    if skip or not REQUIREMENTS.exists():
        return
    print(f"Installing dependencies with {python_bin} ...")
    run_command([str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def server_already_healthy(host: str, port: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "healthy"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def pids_listening_on_port(port: int) -> set[int]:
    """Return process IDs listening on a TCP port (skips system PID 0)."""
    pids: set[int] = set()
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                errors="replace",
            )
            needle = f":{port} "
            for line in out.splitlines():
                if needle not in line or "LISTENING" not in line.upper():
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    pid = int(parts[-1])
                except ValueError:
                    continue
                if pid > 0:
                    pids.add(pid)
        except (OSError, subprocess.SubprocessError):
            pass
        return pids

    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            errors="replace",
        )
        for token in out.split():
            try:
                pid = int(token.strip())
                if pid > 0:
                    pids.add(pid)
            except ValueError:
                continue
    except (OSError, subprocess.SubprocessError):
        pass
    return pids


def stop_processes_on_port(port: int) -> list[int]:
    """Stop processes listening on port. Returns PIDs that were targeted."""
    stopped: list[int] = []
    for pid in sorted(pids_listening_on_port(port)):
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    stopped.append(pid)
            else:
                os.kill(pid, 15)
                stopped.append(pid)
        except OSError:
            continue
    if stopped:
        time.sleep(1.5)
    return stopped


def free_port_help(port: str) -> None:
    if os.name == "nt":
        print(
            "  Safe restart:  python run_app.py --skip-install --restart\n"
            "  Manual kill:   Get-NetTCPConnection -LocalPort "
            f"{port} -ErrorAction SilentlyContinue | "
            "Where-Object { $_.OwningProcess -gt 0 } | "
            "Select-Object -ExpandProperty OwningProcess -Unique | "
            "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
        )
    else:
        print(f"  Safe restart:  python run_app.py --skip-install --restart")
        print(f"  Manual kill:   lsof -ti tcp:{port} | xargs kill -9")


def ensure_port_available(host: str, port: str, *, restart: bool = False) -> None:
    port_int = int(port)
    if not port_in_use(host, port_int):
        return

    url = f"http://{host}:{port}"
    healthy = server_already_healthy(host, port)

    if restart:
        pids = stop_processes_on_port(port_int)
        if pids:
            print(f"Stopped process(es) on port {port}: {', '.join(str(p) for p in pids)}")
        else:
            print(f"No listener found on port {port} (port may clear shortly)...")
        for _ in range(10):
            if not port_in_use(host, port_int):
                return
            time.sleep(0.5)
        print(f"WARNING: Port {port} still in use after restart attempt.")
        free_port_help(port)
        sys.exit(1)

    if healthy:
        print(f"TrackExpensio is already running at {url}")
        print("Open that URL in your browser.")
        print("To reload latest code:  python run_app.py --skip-install --restart")
        sys.exit(0)

    print(f"ERROR: Port {port} is already in use by another program (not TrackExpensio).")
    free_port_help(port)
    print(f"  Or use another port:  python run_app.py --skip-install --port 8081")
    sys.exit(1)


def start_server(python_bin: Path, host: str, port: str, reload: bool, restart: bool) -> None:
    ensure_port_available(host, port, restart=restart)
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
    print(f"Starting TrackExpensio on http://{host}:{port}")
    run_command(cmd, env={"PYTHONIOENCODING": "utf-8"})


def detect_python() -> Path:
    if DEFAULT_PYTHON.exists():
        return DEFAULT_PYTHON
    return Path(sys.executable)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the TrackExpensio API server")
    parser.add_argument(
        "--host",
        default=default_host(),
        help="Host interface (default: 127.0.0.1 local, 0.0.0.0 on Render)",
    )
    parser.add_argument(
        "--port",
        default=default_port(),
        help="Port number (default: 8080 or PORT env on Render)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Run uvicorn with auto-reload (useful for development)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Stop any process listening on the port, then start the server",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install step (assumes deps already installed)",
    )
    args = parser.parse_args(argv)

    python_bin = detect_python()
    ensure_dependencies(python_bin, args.skip_install)
    start_server(python_bin, args.host, args.port, args.reload, args.restart)


if __name__ == "__main__":
    main()
