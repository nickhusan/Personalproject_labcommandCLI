"""Disk usage commands."""

import click

from lab.utils import run_command, show_command


@click.group()
def disk():
    """Disk usage — check space, find large files, cleanup tips."""


@disk.command()
def usage():
    """Show disk usage (df -h)."""
    run_command(["df", "-h"])


@disk.command()
def big():
    """Show largest items in current directory."""
    run_command("du -sh * | sort -rh | head -20", shell=True)


@disk.command()
def clean():
    """Show common cleanup commands."""
    commands = [
        ("Clear pip cache", "pip cache purge"),
        ("Clear conda cache", "conda clean --all -y"),
        ("Clear HuggingFace cache", "rm -rf ~/.cache/huggingface/hub/*"),
        ("Find large files (>1GB)", "find ~ -size +1G -type f 2>/dev/null"),
        ("Clear apt cache", "sudo apt-get clean"),
    ]
    click.secho("\nCommon cleanup commands:\n", fg="cyan", bold=True)
    for desc, cmd in commands:
        click.secho(f"  {desc}:", fg="yellow")
        click.secho(f"    {cmd}\n", fg="green")
