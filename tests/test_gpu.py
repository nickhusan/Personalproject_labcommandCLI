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


@patch("lab.commands.gpu.get_gpu_info")
def test_gpu_free(mock_info):
    mock_info.return_value = [
        {"index": "0", "name": "NVIDIA A100", "used": 1000, "total": 40000, "free": 39000},
        {"index": "1", "name": "NVIDIA A100", "used": 20000, "total": 40000, "free": 20000},
    ]
    runner = CliRunner()
    result = runner.invoke(cli, ["gpu", "free"])
    assert result.exit_code == 0
    assert "GPU 0" in result.output
    assert "GPU 1" in result.output
