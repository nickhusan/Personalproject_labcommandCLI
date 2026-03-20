"""Setup command — check/create venv with vLLM installed."""

import shutil
import subprocess

import click


def _get_conda_envs():
    """Get list of conda environments as (name, path) tuples."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        import json
        data = json.loads(result.stdout)
        envs = []
        for path in data.get("envs", []):
            name = path.rsplit("/", 1)[-1]
            envs.append((name, path))
        return envs
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _env_has_vllm(env_path):
    """Check if an environment has vLLM installed."""
    python = f"{env_path}/bin/python"
    try:
        result = subprocess.run(
            [python, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


@click.command()
@click.option("--name", default="vllm", help="Name for the new conda env (default: vllm)")
@click.option("--python", "python_ver", default="3.10", help="Python version (default: 3.10)")
def setup(name, python_ver):
    """Check for vLLM environment, create one if needed."""

    has_conda = shutil.which("conda") is not None

    if not has_conda:
        click.secho("conda not found!", fg="red")
        click.secho("\nYou can install vLLM with pip instead:", fg="yellow")
        click.secho("  pip install vllm", fg="green")
        return

    # Scan existing envs for vLLM
    click.secho("\nScanning conda environments...\n", fg="cyan")
    envs = _get_conda_envs()

    ready_envs = []
    other_envs = []
    for env_name, env_path in envs:
        vllm_ver = _env_has_vllm(env_path)
        if vllm_ver:
            ready_envs.append((env_name, env_path, vllm_ver))
        else:
            other_envs.append((env_name, env_path))

    # Show envs with vLLM
    if ready_envs:
        click.secho("Environments with vLLM installed:\n", fg="green", bold=True)
        for i, (env_name, env_path, ver) in enumerate(ready_envs):
            click.secho(f"  [{i + 1}] {env_name} (vLLM {ver})", fg="green")
            click.secho(f"      {env_path}", fg="white", dim=True)
        click.echo()
        click.secho("To use one of these, run:", fg="cyan")
        click.secho(f"  conda activate {ready_envs[0][0]}\n", fg="green")

        if not click.confirm("Create a new environment anyway?", default=False):
            return
        click.echo()

    # Show other envs — offer to install vLLM into one
    if other_envs and not ready_envs:
        click.secho("Existing environments (no vLLM):\n", fg="yellow")
        for i, (env_name, env_path) in enumerate(other_envs):
            click.secho(f"  [{i + 1}] {env_name}", fg="yellow")
            click.secho(f"      {env_path}", fg="white", dim=True)
        click.echo()

        choice = click.prompt(
            "Install vLLM into an existing env, or create new?\n"
            "  Enter a number to pick an env, or 'new' to create one",
            type=str,
            default="new",
        )

        if choice != "new":
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(other_envs):
                    env_name, env_path = other_envs[idx]
                    click.secho(f"\nInstalling vLLM into '{env_name}'...\n", fg="cyan")
                    pip_path = f"{env_path}/bin/pip"
                    result = subprocess.run(
                        [pip_path, "install", "vllm"],
                        check=False,
                    )
                    if result.returncode == 0:
                        click.secho(f"\nvLLM installed into '{env_name}'!", fg="green", bold=True)
                        click.secho(f"\nNow run:", fg="cyan")
                        click.secho(f"  conda activate {env_name}", fg="green")
                    else:
                        click.secho("\nInstallation failed. Try manually:", fg="red")
                        click.secho(f"  conda activate {env_name} && pip install vllm", fg="yellow")
                    return
            except ValueError:
                pass
            click.secho("Invalid choice, creating new env.", fg="yellow")

    if not ready_envs and not other_envs:
        click.secho("No conda environments found.\n", fg="yellow")

    # Create new env
    click.secho(f"Creating conda env '{name}' with Python {python_ver}...\n", fg="cyan", bold=True)

    result = subprocess.run(
        ["conda", "create", "-n", name, f"python={python_ver}", "-y"],
        check=False,
    )
    if result.returncode != 0:
        click.secho(f"\nFailed to create env '{name}'.", fg="red")
        return

    # Get the new env path
    envs_after = _get_conda_envs()
    env_path = None
    for env_name, path in envs_after:
        if env_name == name:
            env_path = path
            break

    if env_path is None:
        click.secho(f"\nEnv created but couldn't find path. Run manually:", fg="yellow")
        click.secho(f"  conda activate {name} && pip install vllm", fg="green")
        return

    # Install vLLM
    click.secho(f"\nInstalling vLLM...\n", fg="cyan")
    pip_path = f"{env_path}/bin/pip"
    result = subprocess.run(
        [pip_path, "install", "vllm"],
        check=False,
    )

    if result.returncode == 0:
        click.secho(f"\nAll set! Environment '{name}' is ready with vLLM.", fg="green", bold=True)
    else:
        click.secho(f"\nvLLM installation failed. After activating, try: pip install vllm", fg="red")

    click.echo()
    click.secho("Now run:", fg="cyan", bold=True)
    click.secho(f"  conda activate {name}", fg="green")
    click.secho(f"  lab model run <model-name>", fg="green")
    click.echo()
