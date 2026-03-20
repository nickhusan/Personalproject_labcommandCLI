"""GPU monitoring commands."""

import subprocess

import click

from lab.utils import run_command, show_command


@click.group()
def gpu():
    """GPU monitoring — check status, memory, and processes."""


@gpu.command()
def status():
    """Show GPU status (nvidia-smi)."""
    run_command(["nvidia-smi"])


@gpu.command()
def watch():
    """Watch GPU status, refreshing every second."""
    run_command(["watch", "-n", "1", "nvidia-smi"])


@gpu.command()
def processes():
    """List processes using GPUs."""
    run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv",
    ])


@gpu.command()
def free():
    """Show which GPUs have free memory."""
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
        click.secho("Could not query GPUs (is nvidia-smi available?)", fg="red")
        return

    click.secho("\nGPU Free Memory:\n", fg="cyan", bold=True)
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            idx, name, used, total, free_mb = parts
            pct_free = float(free_mb) / float(total) * 100 if float(total) > 0 else 0
            color = "green" if pct_free > 50 else "yellow" if pct_free > 10 else "red"
            click.secho(
                f"  GPU {idx} ({name}): {free_mb} MiB free / {total} MiB total ({pct_free:.0f}%)",
                fg=color,
            )
    click.echo()
