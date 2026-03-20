"""GPU monitoring commands."""

import click

from lab.utils import run_command, get_gpu_info


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
    gpus = get_gpu_info()
    if not gpus:
        click.secho("Could not query GPUs (is nvidia-smi available?)", fg="red")
        return

    click.secho("\nGPU Free Memory:\n", fg="cyan", bold=True)
    for g in gpus:
        pct_free = g["free"] / g["total"] * 100 if g["total"] > 0 else 0
        color = "green" if pct_free > 50 else "yellow" if pct_free > 10 else "red"
        click.secho(
            f"  GPU {g['index']} ({g['name']}): {g['free']} MiB free / {g['total']} MiB total ({pct_free:.0f}%)",
            fg=color,
        )
    click.echo()
