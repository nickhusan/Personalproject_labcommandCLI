"""Tests for the main CLI entry point."""

from click.testing import CliRunner

from lab.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Lab CLI" in result.output


def test_cli_has_all_groups():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    for group in ["gpu", "process", "run", "env", "disk", "model", "help"]:
        assert group in result.output


def test_help_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["help"])
    assert result.exit_code == 0
    assert "GPU monitoring" in result.output
    assert "Process management" in result.output
