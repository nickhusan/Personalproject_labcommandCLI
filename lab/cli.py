"""Main CLI entry point for lab."""

import json
from urllib.request import urlopen, Request
from urllib.error import URLError

import click

from lab.commands.gpu import gpu
from lab.commands.process import process
from lab.commands.run import run
from lab.commands.env import env
from lab.commands.disk import disk
from lab.commands.model import model


@click.group()
def cli():
    """Lab CLI — quick access to common GPU lab commands.

    Run `lab <category> --help` for commands in each category.
    """


cli.add_command(gpu)
cli.add_command(process)
cli.add_command(run)
cli.add_command(env)
cli.add_command(disk)
cli.add_command(model)


def _detect_model(port):
    """Auto-detect the model name from a running vLLM server."""
    try:
        req = Request(f"http://localhost:{port}/v1/models")
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except (URLError, OSError, json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def _stream_response(port, model_name, messages):
    """Send messages to vLLM and stream the response. Returns the full response text."""
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": True,
    }).encode()

    req = Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full_response = []
    try:
        with urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        click.echo(content, nl=False)
                        full_response.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        click.echo()  # final newline
        return "".join(full_response)
    except (URLError, OSError) as e:
        click.secho(f"\nError connecting to model: {e}", fg="red")
        click.secho(f"Is vLLM running on port {port}?", fg="yellow")
        return None


def _connect_or_fail(port, model_name):
    """Detect model and validate connection. Returns model_name or exits."""
    if model_name is None:
        click.secho("Detecting model...", fg="cyan", dim=True)
        model_name = _detect_model(port)
        if model_name is None:
            click.secho(
                f"No vLLM server found on port {port}. "
                "Start one with: lab model run <name>",
                fg="red",
            )
            return None
        click.secho(f"Using model: {model_name}\n", fg="cyan")
    return model_name


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--port", default=8000, help="vLLM server port (default: 8000)")
@click.option("--model-name", default=None, help="Model name (auto-detected if omitted)")
@click.option("--system", default=None, help="System prompt")
def ask(question, port, model_name, system):
    """Ask a question to your running vLLM model.

    Example: lab ask what is gradient descent
    """
    model_name = _connect_or_fail(port, model_name)
    if model_name is None:
        return

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": " ".join(question)})

    _stream_response(port, model_name, messages)


@cli.command()
@click.option("--port", default=8000, help="vLLM server port (default: 8000)")
@click.option("--model-name", default=None, help="Model name (auto-detected if omitted)")
@click.option("--system", default=None, help="System prompt")
def chat(port, model_name, system):
    """Interactive chat with your running vLLM model.

    Keeps conversation context. Type 'exit' or 'quit' to leave.
    Ctrl+C also exits cleanly.

    Example: lab chat
    """
    model_name = _connect_or_fail(port, model_name)
    if model_name is None:
        return

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    click.secho("Chat started. Type 'exit' or 'quit' to leave.\n", fg="green", dim=True)

    try:
        while True:
            try:
                user_input = click.prompt(click.style("You", fg="cyan", bold=True))
            except click.Abort:
                break

            if user_input.strip().lower() in ("exit", "quit"):
                break

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            click.secho("Model: ", fg="green", bold=True, nl=False)
            response = _stream_response(port, model_name, messages)

            if response is not None:
                messages.append({"role": "assistant", "content": response})
            else:
                # Remove the failed user message so conversation stays clean
                messages.pop()
            click.echo()
    except KeyboardInterrupt:
        pass

    click.secho("\nChat ended.", fg="cyan", dim=True)


@cli.command("help")
def help_cmd():
    """Browse all command categories."""
    categories = {
        "gpu": "GPU monitoring (nvidia-smi, memory, processes)",
        "process": "Process management (list, kill, killgpu)",
        "run": "Background jobs (nohup, screen sessions)",
        "env": "Environment management (conda, pip)",
        "disk": "Disk usage and cleanup",
        "model": "Model operations (download, serve, run, stop, check)",
        "ask": "Ask a one-off question to your running model",
        "chat": "Interactive conversation with your running model",
    }
    click.secho("\nLab CLI — Command Categories\n", fg="cyan", bold=True)
    for name, desc in categories.items():
        click.secho(f"  lab {name:<10}", fg="green", nl=False)
        click.secho(f" {desc}", fg="white")
    click.echo()
    click.secho("Run `lab <category> --help` for details.\n", dim=True)
