"""TOML config loader and validation."""

import tomllib
from pathlib import Path


def load_config(repo_path: Path) -> dict | None:
    """Load .clean_room/config.toml. Returns None if the file doesn't exist."""
    config_file = repo_path / ".clean_room" / "config.toml"
    if not config_file.exists():
        return None
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def require_models_config(config: dict | None) -> dict:
    """Extract [models] section or raise a hard error."""
    if config is None:
        raise RuntimeError(
            "No config file found. Run 'cra init' to create .clean_room/config.toml"
        )
    models = config.get("models")
    if models is None:
        raise RuntimeError(
            "Missing [models] section in .clean_room/config.toml. "
            "Run 'cra init' to configure model settings."
        )
    return models


def create_default_config(repo_path: Path) -> Path:
    """Create a default config.toml in .clean_room/. Returns the path."""
    clean_room = repo_path / ".clean_room"
    clean_room.mkdir(parents=True, exist_ok=True)
    config_path = clean_room / "config.toml"
    if config_path.exists():
        raise FileExistsError(f"Config already exists: {config_path}")
    config_path.write_text(
        '[models]\n'
        'provider = "ollama"\n'
        'coding = "qwen2.5-coder:3b-instruct"\n'
        'reasoning = "qwen3:4b-instruct-2507"\n'
        'base_url = "http://localhost:11434"\n'
        'context_window = 32768\n'
        '\n'
        '[models.overrides]\n'
        '# scope = "qwen3:4b-scope-v1"\n'
        '# execute_code = "qwen2.5-coder:3b-code-v1"\n'
        '\n'
        '# [models.temperature]\n'
        '# coding = 0.0\n'
        '# reasoning = 0.0\n'
        '\n'
        '[budget]\n'
        '# context_window: defaults to [models].context_window if omitted\n'
        'reserved_tokens = 4096\n'
        '\n'
        '[stages]\n'
        'default = "scope,precision"\n'
        '\n'
        '[testing]\n'
        'test_command = "pytest tests/"\n'
        '# lint_command = "ruff check src/"\n'
        '# type_check_command = "mypy src/"\n'
        '# timeout = 120\n'
        '\n'
        '[orchestrator]\n'
        'max_retries_per_step = 1\n'
        '# max_adjustment_rounds = 1\n'
    )
    return config_path
