"""Environment management commands (conda, pip)."""

import click

from lab.utils import run_command, show_command, copy_to_clipboard


@click.group()
def env():
    """Environment management — conda envs, pip packages."""


@env.command("list")
def list_envs():
    """List conda environments."""
    run_command(["conda", "env", "list"])


@env.command()
@click.argument("name")
def activate(name):
    """Show how to activate a conda environment."""
    show_command(
        f"conda activate {name}",
        "Run this in your shell (can't be activated from a subprocess).",
    )


@env.command()
@click.argument("name")
@click.option("--python", default="3.10", help="Python version (default: 3.10)")
def create(name, python):
    """Show how to create a new conda environment."""
    copy_to_clipboard(
        f"conda create -n {name} python={python} -y",
        f"Creates a new conda env '{name}' with Python {python}.",
    )


@env.command()
def packages():
    """List installed pip packages."""
    run_command(["pip", "list"])
