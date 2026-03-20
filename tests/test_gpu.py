"""Tests for GPU commands."""

from unittest.mock import patch

from click.testing import CliRunner

from lab.cli import cli


def test_gpu_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["gpu", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "watch" in result.output
    assert "processes" in result.output
    assert "free" in result.output


@patch("lab.commands.gpu.run_command")
def test_gpu_status(mock_run):
    runner = CliRunner()
    result = runner.invoke(cli, ["gpu", "status"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with(["nvidia-smi"])


@patch("lab.commands.gpu.run_command")
def test_gpu_watch(mock_run):
    runner = CliRunner()
    result = runner.invoke(cli, ["gpu", "watch"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with(["watch", "-n", "1", "nvidia-smi"])


@patch("lab.commands.gpu.run_command")
def test_gpu_processes(mock_run):
    runner = CliRunner()
    result = runner.invoke(cli, ["gpu", "processes"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv",
    ])


@patch("lab.commands.gpu.subprocess.run")
def test_gpu_free(mock_run):
    mock_run.return_value.stdout = "0, NVIDIA A100, 1000, 40000, 39000\n1, NVIDIA A100, 20000, 40000, 20000\n"
    mock_run.return_value.returncode = 0
    runner = CliRunner()
    result = runner.invoke(cli, ["gpu", "free"])
    assert result.exit_code == 0
    assert "GPU 0" in result.output
    assert "GPU 1" in result.output
