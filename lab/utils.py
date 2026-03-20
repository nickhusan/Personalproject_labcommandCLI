"""Shared helpers for running, displaying, and copying commands."""

from __future__ import annotations

import shutil
import subprocess
import sys

import click


def run_command(cmd: list[str] | str, shell: bool = False) -> None:
    """Run a command directly, streaming output to the terminal."""
    click.secho(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", fg="cyan")
    try:
        subprocess.run(cmd, shell=shell, check=False)
    except FileNotFoundError:
        click.secho(f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}", fg="red")
    except KeyboardInterrupt:
        pass


def show_command(cmd: str, explanation: str = "") -> None:
    """Display a command with explanation, without running it."""
    click.secho("\nCommand:", fg="yellow", bold=True)
    click.secho(f"  {cmd}", fg="green")
    if explanation:
        click.secho(f"\n  {explanation}", fg="white", dim=True)
    click.echo()


def copy_to_clipboard(cmd: str, explanation: str = "") -> None:
    """Copy a command to clipboard and display it."""
    show_command(cmd, explanation)
    clipboard_cmd = _get_clipboard_cmd()
    if clipboard_cmd:
        try:
            subprocess.run(clipboard_cmd, input=cmd.encode(), check=True)
            click.secho("Copied to clipboard!", fg="green", bold=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    click.secho("(Could not copy to clipboard — copy manually from above)", fg="yellow")


def get_gpu_info() -> list[dict]:
    """Query GPU index, name, memory used/total/free. Returns list of dicts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "used": int(parts[2]),
                "total": int(parts[3]),
                "free": int(parts[4]),
            })
    return gpus


def _get_clipboard_cmd() -> list[str] | None:
    """Get the platform-appropriate clipboard command."""
    if sys.platform == "darwin":
        return ["pbcopy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    return None
