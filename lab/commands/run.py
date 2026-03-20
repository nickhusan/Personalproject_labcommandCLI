"""Background job commands (nohup, screen)."""

import click

from lab.utils import run_command, show_command, copy_to_clipboard


@click.group()
def run():
    """Background jobs — nohup, screen sessions."""


@run.command()
@click.argument("command", nargs=-1, required=True)
def bg(command):
    """Run a command in the background with nohup."""
    cmd_str = " ".join(command)
    copy_to_clipboard(
        f"nohup {cmd_str} > output.log 2>&1 &",
        "Runs in background, survives disconnect. Output goes to output.log.",
    )


@run.command()
@click.argument("name")
def screen(name):
    """Create a new screen session."""
    run_command(["screen", "-S", name])


@run.command()
@click.argument("name")
def attach(name):
    """Reattach to a screen session."""
    run_command(["screen", "-r", name])


@run.command("list")
def list_sessions():
    """List active screen sessions."""
    run_command(["screen", "-ls"])
