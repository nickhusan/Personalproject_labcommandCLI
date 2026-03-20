"""Tests for model commands."""

from unittest.mock import patch

from click.testing import CliRunner

from lab.cli import cli
from lab.commands.model import _parse_param_count, _guess_dtype, _estimate_vram_gb


def test_parse_param_count():
    assert _parse_param_count("meta-llama/Llama-3-70B") == 70.0
    assert _parse_param_count("mistralai/Mistral-7B-v0.1") == 7.0
    assert _parse_param_count("Qwen/Qwen2.5-1.5B") == 1.5
    assert _parse_param_count("microsoft/phi-3-mini-4b-instruct") == 4.0
    assert _parse_param_count("some-random-model") is None


def test_guess_dtype():
    assert _guess_dtype("TheBloke/Llama-2-70B-AWQ") == "awq"
    assert _guess_dtype("TheBloke/Llama-2-70B-GPTQ") == "gptq"
    assert _guess_dtype("some-model-int8") == "int8"
    assert _guess_dtype("meta-llama/Llama-3-70B") == "fp16"


def test_estimate_vram():
    # 70B fp16 = 70 * 2 * 1.25 = 175 GB
    assert _estimate_vram_gb(70, "fp16") == 175.0
    # 70B int4 = 70 * 0.5 * 1.25 = 43.75 GB
    assert _estimate_vram_gb(70, "int4") == 43.75
    # 7B fp16 = 7 * 2 * 1.25 = 17.5 GB
    assert _estimate_vram_gb(7, "fp16") == 17.5


@patch("lab.commands.model.get_gpu_info")
def test_model_check_fits_single(mock_info):
    mock_info.return_value = [
        {"index": "0", "name": "H100", "used": 1000, "total": 81920, "free": 80920},
    ]
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "check", "meta-llama/Llama-3-7B"])
    assert result.exit_code == 0
    assert "FITS on a single GPU" in result.output


@patch("lab.commands.model.get_gpu_info")
def test_model_check_fits_multi(mock_info):
    mock_info.return_value = [
        {"index": "0", "name": "H100", "used": 10000, "total": 81920, "free": 71920},
        {"index": "1", "name": "H100", "used": 10000, "total": 81920, "free": 71920},
        {"index": "2", "name": "H100", "used": 10000, "total": 81920, "free": 71920},
    ]
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "check", "meta-llama/Llama-3-70B"])
    assert result.exit_code == 0
    assert "FITS across" in result.output


@patch("lab.commands.model.get_gpu_info")
def test_model_check_doesnt_fit(mock_info):
    mock_info.return_value = [
        {"index": "0", "name": "A100", "used": 30000, "total": 40960, "free": 10960},
    ]
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "check", "meta-llama/Llama-3-70B"])
    assert result.exit_code == 0
    assert "DOES NOT FIT" in result.output
    assert "Options to make it fit" in result.output


def test_model_check_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "check", "--help"])
    assert result.exit_code == 0
    assert "--params" in result.output
    assert "--dtype" in result.output
