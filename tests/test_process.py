"""Tests for process commands."""

from unittest.mock import patch

from click.testing import CliRunner

from lab.cli import cli


def test_process_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["process", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "kill" in result.output
    assert "killgpu" in result.output


@patch("lab.commands.process.run_command")
def test_process_list(mock_run):
    runner = CliRunner()
    result = runner.invoke(cli, ["process", "list"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


@patch("lab.commands.process.run_command")
def test_process_kill(mock_run):
    runner = CliRunner()
    result = runner.invoke(cli, ["process", "kill", "vllm"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with(["pkill", "-f", "vllm"])


@patch("lab.commands.process.run_command")
@patch("lab.commands.process.subprocess.run")
def test_process_killgpu_no_processes(mock_subprocess, mock_run):
    mock_subprocess.return_value.stdout = ""
    mock_subprocess.return_value.returncode = 0
    runner = CliRunner()
    result = runner.invoke(cli, ["process", "killgpu", "0"])
    assert result.exit_code == 0
    assert "No processes" in result.output
