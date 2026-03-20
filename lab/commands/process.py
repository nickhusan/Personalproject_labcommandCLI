"""Process management commands."""

import subprocess

import click

from lab.utils import run_command, show_command


@click.group()
def process():
    """Process management — list, kill, and manage GPU processes."""


@process.command("list")
def list_processes():
    """List processes using GPUs."""
    run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv",
    ])


@process.command()
@click.argument("name")
def kill(name):
    """Kill processes matching NAME (pkill -f)."""
    click.secho(f"Killing processes matching '{name}'...", fg="yellow")
    run_command(["pkill", "-f", name])


@process.command()
@click.argument("gpu_id")
def killgpu(gpu_id):
    """Kill all processes on a specific GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        click.secho(f"Could not query GPU {gpu_id}", fg="red")
        return

    pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    if not pids:
        click.secho(f"No processes found on GPU {gpu_id}", fg="green")
        return

    click.secho(f"Killing {len(pids)} process(es) on GPU {gpu_id}: {', '.join(pids)}", fg="yellow")
    for pid in pids:
        run_command(["kill", "-9", pid])
    click.secho("Done.", fg="green")
