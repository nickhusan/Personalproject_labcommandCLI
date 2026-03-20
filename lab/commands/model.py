"""Model operations (HuggingFace, vLLM)."""

import os

import click

from lab.utils import show_command, copy_to_clipboard


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
