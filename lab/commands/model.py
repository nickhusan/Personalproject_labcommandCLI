"""Model operations (HuggingFace, vLLM)."""

import os
import re

import click

from lab.utils import show_command, copy_to_clipboard, run_command, get_gpu_info


# Bytes per parameter for each dtype
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
}

# vLLM overhead multiplier (KV cache, activations, CUDA context)
OVERHEAD_MULTIPLIER = 1.25


def _parse_param_count(name):
    """Try to extract parameter count (in billions) from a model name.

    Handles patterns like: 70B, 7b, 8B, 0.5B, 1.5b, 70b-chat, etc.
    """
    match = re.search(r"(\d+\.?\d*)[bB]", name)
    if match:
        return float(match.group(1))
    return None


def _guess_dtype(name):
    """Guess the dtype/quantization from the model name."""
    lower = name.lower()
    if "awq" in lower:
        return "awq"
    if "gptq" in lower:
        return "gptq"
    if "int8" in lower or "8bit" in lower:
        return "int8"
    if "int4" in lower or "4bit" in lower:
        return "int4"
    return "fp16"


def _estimate_vram_gb(params_b, dtype):
    """Estimate total VRAM needed in GB, including overhead."""
    bytes_per = BYTES_PER_PARAM.get(dtype, 2.0)
    model_gb = params_b * bytes_per  # params_b is in billions, bytes_per * 1B = GB
    return model_gb * OVERHEAD_MULTIPLIER


def _display_gpu_table(gpus):
    """Display GPU table with free memory."""
    click.secho("\nAvailable GPUs:\n", fg="cyan", bold=True)
    for g in gpus:
        pct_free = g["free"] / g["total"] * 100 if g["total"] > 0 else 0
        color = "green" if pct_free > 50 else "yellow" if pct_free > 10 else "red"
        click.secho(
            f"  [{g['index']}] {g['name']}  —  "
            f"{g['free']} MiB free / {g['total']} MiB total ({pct_free:.0f}% free)",
            fg=color,
        )
    click.echo()


@click.group()
def model():
    """Model operations — download, serve, list, chat."""


@model.command()
@click.argument("name")
def download(name):
    """Show how to download a model from HuggingFace."""
    copy_to_clipboard(
        f"huggingface-cli download {name}",
        f"Downloads '{name}' to the HuggingFace cache (~/.cache/huggingface/hub/).",
    )


@model.command()
@click.argument("name")
@click.option("--port", default=8000, help="Port to serve on (default: 8000)")
@click.option("--gpus", default=1, help="Number of GPUs (default: 1)")
def serve(name, port, gpus):
    """Show vLLM serve command for a model."""
    cmd = f"python -m vllm.entrypoints.openai.api_server --model {name} --port {port}"
    if gpus > 1:
        cmd += f" --tensor-parallel-size {gpus}"
    copy_to_clipboard(cmd, "Starts a vLLM OpenAI-compatible server.")


@model.command("list")
def list_models():
    """List models in common cache directories."""
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/data/models",
        "/models",
    ]

    click.secho("\nCached models:\n", fg="cyan", bold=True)
    found = False
    for cache_dir in cache_dirs:
        if os.path.isdir(cache_dir):
            entries = sorted(os.listdir(cache_dir))
            model_entries = [e for e in entries if e.startswith("models--")]
            if model_entries:
                click.secho(f"  {cache_dir}/", fg="yellow")
                for entry in model_entries:
                    name = entry.replace("models--", "").replace("--", "/")
                    click.secho(f"    {name}", fg="green")
                found = True
            else:
                # Show all entries for non-HF cache dirs
                regular = [e for e in entries if not e.startswith(".")]
                if regular:
                    click.secho(f"  {cache_dir}/", fg="yellow")
                    for entry in regular:
                        click.secho(f"    {entry}", fg="green")
                    found = True

    if not found:
        click.secho("  No models found in common cache directories.", fg="yellow")
    click.echo()


@model.command()
@click.argument("name")
@click.option("--params", type=float, default=None, help="Parameter count in billions (e.g. 70). Auto-detected from name if omitted.")
@click.option("--dtype", type=click.Choice(list(BYTES_PER_PARAM.keys())), default=None, help="Data type / quantization. Auto-detected from name if omitted.")
def check(name, params, dtype):
    """Check if a model fits on your GPUs and how many you need."""
    # Parse param count
    if params is None:
        params = _parse_param_count(name)
    if params is None:
        click.secho(f"Could not detect param count from '{name}'.", fg="red")
        click.secho("Use --params to specify, e.g.: lab model check my-model --params 70", fg="yellow")
        return

    # Detect dtype
    if dtype is None:
        dtype = _guess_dtype(name)

    vram_needed_gb = _estimate_vram_gb(params, dtype)
    vram_needed_mib = vram_needed_gb * 1024

    click.secho(f"\nModel: {name}", fg="cyan", bold=True)
    click.secho(f"  Parameters:  {params}B", fg="white")
    click.secho(f"  Dtype:       {dtype} ({BYTES_PER_PARAM[dtype]} bytes/param)", fg="white")
    click.secho(f"  Est. VRAM:   {vram_needed_gb:.1f} GB ({vram_needed_mib:.0f} MiB) — includes ~25% overhead for KV cache/activations\n", fg="white")

    # Check GPUs
    gpus = get_gpu_info()
    if not gpus:
        click.secho("Could not detect GPUs (is nvidia-smi available?)", fg="yellow")
        click.secho(f"You need at least {vram_needed_gb:.1f} GB of total GPU memory.", fg="white")
        _show_alternatives(params, dtype, vram_needed_gb, total_free_gb=0, num_gpus=0)
        return

    _display_gpu_table(gpus)

    total_free_mib = sum(g["free"] for g in gpus)
    total_free_gb = total_free_mib / 1024

    # Check single GPU fit
    best_gpu = max(gpus, key=lambda g: g["free"])
    if best_gpu["free"] >= vram_needed_mib:
        click.secho("FITS on a single GPU!", fg="green", bold=True)
        click.secho(
            f"  GPU {best_gpu['index']} ({best_gpu['name']}) has {best_gpu['free']} MiB free — "
            f"model needs {vram_needed_mib:.0f} MiB",
            fg="green",
        )
        click.secho(f"\n  Run: lab model run {name}\n", fg="cyan")
        return

    # Check multi-GPU fit
    sorted_gpus = sorted(gpus, key=lambda g: g["free"], reverse=True)
    cumulative = 0
    needed_gpus = []
    for g in sorted_gpus:
        needed_gpus.append(g)
        cumulative += g["free"]
        if cumulative >= vram_needed_mib:
            break

    if cumulative >= vram_needed_mib:
        gpu_ids = ",".join(g["index"] for g in needed_gpus)
        click.secho(f"FITS across {len(needed_gpus)} GPUs with tensor parallelism!", fg="yellow", bold=True)
        click.secho(f"  Use GPUs: {gpu_ids} ({cumulative:.0f} MiB free combined)", fg="yellow")
        click.secho(f"  Model needs: {vram_needed_mib:.0f} MiB", fg="white")
        click.secho(f"\n  Run: lab model run {name}\n", fg="cyan")
        return

    # Doesn't fit
    click.secho("DOES NOT FIT on this machine!", fg="red", bold=True)
    click.secho(
        f"  Need: {vram_needed_mib:.0f} MiB  |  Available: {total_free_mib:.0f} MiB "
        f"(across {len(gpus)} GPUs)",
        fg="red",
    )
    click.echo()
    _show_alternatives(params, dtype, vram_needed_gb, total_free_gb, len(gpus))


def _show_alternatives(params, current_dtype, vram_needed_gb, total_free_gb, num_gpus):
    """Suggest alternatives when the model doesn't fit."""
    click.secho("Options to make it fit:\n", fg="yellow", bold=True)

    # Suggest quantization if not already quantized
    if current_dtype in ("fp32", "fp16", "bf16"):
        for alt_dtype in ["int8", "int4", "awq", "gptq"]:
            alt_gb = _estimate_vram_gb(params, alt_dtype)
            label = "might fit" if total_free_gb > 0 and alt_gb <= total_free_gb else f"needs {alt_gb:.1f} GB"
            fits = total_free_gb > 0 and alt_gb <= total_free_gb
            color = "green" if fits else "yellow"
            click.secho(f"  Quantize to {alt_dtype}: {alt_gb:.1f} GB — {label}", fg=color)
        click.echo()

    # Suggest more GPUs
    if num_gpus > 0:
        per_gpu_gb = total_free_gb / num_gpus if num_gpus > 0 else 80
        gpus_needed = max(1, int(vram_needed_gb / per_gpu_gb) + 1)
        extra = gpus_needed - num_gpus
        if extra > 0:
            click.secho(f"  Add more GPUs: ~{gpus_needed} GPUs needed at current per-GPU memory", fg="yellow")
            click.secho(f"    You have {num_gpus}, need ~{extra} more", fg="yellow")
            click.echo()

    # Multi-node suggestion
    click.secho("  Multi-node serving: use vLLM with Ray across multiple machines", fg="yellow")
    click.secho("    1. Install Ray on both machines: pip install ray", fg="white", dim=True)
    click.secho("    2. Start Ray head: ray start --head", fg="white", dim=True)
    click.secho("    3. Join from other machine: ray start --address=<head-ip>:6379", fg="white", dim=True)
    click.secho("    4. vLLM will auto-detect Ray workers for tensor parallelism", fg="white", dim=True)
    click.echo()


@model.command("run")
@click.argument("name")
@click.option("--port", default=8000, help="Port to serve on (default: 8000)")
@click.option("--params", type=float, default=None, help="Parameter count in billions (auto-detected from name).")
@click.option("--dtype", type=click.Choice(list(BYTES_PER_PARAM.keys())), default=None, help="Data type (auto-detected from name).")
@click.option("--skip-check", is_flag=True, help="Skip the VRAM pre-flight check.")
def run_model(name, port, params, dtype, skip_check):
    """Pick GPUs interactively, then launch vLLM to serve a model."""
    gpus = get_gpu_info()
    if not gpus:
        click.secho("Could not detect GPUs (is nvidia-smi available?)", fg="red")
        return

    # Pre-flight VRAM check
    if not skip_check:
        if params is None:
            params = _parse_param_count(name)
        if dtype is None:
            dtype = _guess_dtype(name)

        if params is not None:
            vram_needed_gb = _estimate_vram_gb(params, dtype)
            vram_needed_mib = vram_needed_gb * 1024
            total_free_mib = sum(g["free"] for g in gpus)

            click.secho(f"\nModel: {name} ({params}B, {dtype})", fg="cyan", bold=True)
            click.secho(f"  Est. VRAM: {vram_needed_gb:.1f} GB  |  Available: {total_free_mib / 1024:.1f} GB total free\n", fg="white")

            if vram_needed_mib > total_free_mib:
                click.secho("WARNING: Model may not fit on this machine!", fg="red", bold=True)
                click.secho(f"  Need {vram_needed_mib:.0f} MiB but only {total_free_mib:.0f} MiB free across all GPUs.", fg="red")
                _show_alternatives(params, dtype, vram_needed_gb, total_free_mib / 1024, len(gpus))
                if not click.confirm("Try anyway?"):
                    return
                click.echo()

    _display_gpu_table(gpus)

    # Prompt for GPU selection
    selection = click.prompt(
        "Select GPUs (comma-separated, e.g. 0,1,2)",
        type=str,
    )
    gpu_ids = [s.strip() for s in selection.split(",") if s.strip()]

    # Validate
    valid_ids = {g["index"] for g in gpus}
    bad = [g for g in gpu_ids if g not in valid_ids]
    if bad:
        click.secho(f"Invalid GPU id(s): {', '.join(bad)}", fg="red")
        return

    # Check selected GPUs have enough VRAM
    if not skip_check and params is not None:
        selected_free = sum(g["free"] for g in gpus if g["index"] in gpu_ids)
        if selected_free < vram_needed_mib:
            click.secho(
                f"\nWARNING: Selected GPUs have {selected_free:.0f} MiB free "
                f"but model needs ~{vram_needed_mib:.0f} MiB.",
                fg="red",
            )
            if not click.confirm("Launch anyway?"):
                return

    num_gpus = len(gpu_ids)
    cuda_devices = ",".join(gpu_ids)

    # Build and run command
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} python -m vllm.entrypoints.openai.api_server --model {name} --port {port}"
    if num_gpus > 1:
        cmd += f" --tensor-parallel-size {num_gpus}"

    click.secho(f"\nLaunching on GPU(s) {cuda_devices}...\n", fg="green", bold=True)
    run_command(cmd, shell=True)


@model.command()
@click.option("--port", default=8000, help="Port the model is served on (default: 8000)")
def chat(port):
    """Show how to query a running vLLM model."""
    cmd = f"""curl http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "<model-name>",
    "messages": [{{"role": "user", "content": "Hello!"}}]
  }}'"""
    copy_to_clipboard(cmd, "Sends a chat request to a running vLLM server. Replace <model-name> with the actual model.")
