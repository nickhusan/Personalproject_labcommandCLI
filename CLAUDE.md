# Lab CLI

GPU lab command reference & runner. A Python CLI tool using `click` that gives quick access to common ML lab commands.

## Project structure
- `lab/` — main package
- `lab/cli.py` — entry point, registers all command groups
- `lab/commands/` — one module per category (gpu, process, run, env, disk, model)
- `lab/utils.py` — shared helpers (run_command, display_command, copy_to_clipboard)
- `tests/` — unit tests, mock subprocess calls

## Conventions
- Use `click` for CLI framework
- Commands either run directly (safe/read-only), show with explanation, or copy to clipboard
- Keep dependencies minimal (only click)
- Tests mock subprocess — never run real system commands in tests
- Python 3.10+
