"""Main CLI entry point for lab."""

import click

from lab.commands.gpu import gpu
from lab.commands.process import process
from lab.commands.run import run
from lab.commands.env import env
from lab.commands.disk import disk
from lab.commands.model import model


@click.group()
def cli():
    """Lab CLI — quick access to common GPU lab commands.

    Run `lab <category> --help` for commands in each category.
    """


cli.add_command(gpu)
cli.add_command(process)
cli.add_command(run)
cli.add_command(env)
cli.add_command(disk)
cli.add_command(model)


@cli.command()
def help():
    """Browse all command categories."""
    categories = {
        "gpu": "GPU monitoring (nvidia-smi, memory, processes)",
        "process": "Process management (list, kill, killgpu)",
        "run": "Background jobs (nohup, screen sessions)",
        "env": "Environment management (conda, pip)",
        "disk": "Disk usage and cleanup",
        "model": "Model operations (download, serve, chat)",
    }
    click.secho("\nLab CLI — Command Categories\n", fg="cyan", bold=True)
    for name, desc in categories.items():
        click.secho(f"  lab {name:<10}", fg="green", nl=False)
        click.secho(f" {desc}", fg="white")
    click.echo()
    click.secho("Run `lab <category> --help` for details.\n", dim=True)
