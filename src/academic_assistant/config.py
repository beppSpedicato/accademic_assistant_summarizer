"""Per-machine configuration shared by every academic-assistant subcommand."""

import getpass
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import tomli_w
from platformdirs import user_config_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

APP_NAME = "academic-assistant"
CONFIG_FILENAME = "config.toml"


@dataclass(frozen=True)
class Config:
    base_url: str = "http://localhost:4141/v1"
    model: str = "gpt-5.2"
    api_key: str = ""
    subject: str = "computational linear algebra"


class ConfigError(Exception):
    pass


def config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False)) / CONFIG_FILENAME


def load_config(path: Optional[Path] = None) -> Config:
    path = path or config_path()
    if not path.exists():
        raise ConfigError(
            f"No configuration found at {path}. Run `academic-assistant config init` first."
        )
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid configuration file {path}: {e}") from e

    known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
    return replace(Config(), **{k: str(v) for k, v in known.items()})


def save_config(config: Config, path: Optional[Path] = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # The file holds the API key: create it owner-only and swap it in atomically.
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        tomli_w.dump(asdict(config), f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def resolve_config(
    url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    path: Optional[Path] = None,
) -> Config:
    """Load the saved config and apply per-run CLI overrides on top of it."""
    config = load_config(path)
    overrides = {"base_url": url, "model": model, "api_key": api_key}
    return replace(config, **{k: v for k, v in overrides.items() if v is not None})


def mask_secret(secret: str) -> str:
    if not secret:
        return "(not set)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:3]}...{secret[-4:]}"


def run_wizard(
    path: Optional[Path] = None,
    prompt: Optional[Callable[[str], str]] = None,
    secret_prompt: Optional[Callable[[str], str]] = None,
) -> Path:
    path = path or config_path()
    prompt = prompt or input
    secret_prompt = secret_prompt or getpass.getpass
    try:
        current = load_config(path)
    except ConfigError:
        current = Config()

    print(f"Configuring {APP_NAME} (saved to {path}). Press Enter to keep the value in brackets.")

    def ask(label: str, default: str) -> str:
        return prompt(f"{label} [{default}]: ").strip() or default

    base_url = ask("Base URL of the OpenAI-compatible API", current.base_url)
    model = ask("Model name", current.model)
    hint = f"keep {mask_secret(current.api_key)}" if current.api_key else "none"
    api_key = secret_prompt(f"API key (input hidden) [{hint}]: ").strip() or current.api_key
    subject = ask("Course subject", current.subject)

    saved = save_config(Config(base_url=base_url, model=model, api_key=api_key, subject=subject), path)
    print(f"Configuration saved to {saved}")
    return saved


def show_config(path: Optional[Path] = None) -> str:
    path = path or config_path()
    return format_config(load_config(path), path)


def format_config(config: Config, path: Path) -> str:
    return "\n".join(
        [
            f"Config file: {path}",
            f"base_url = {config.base_url}",
            f"model    = {config.model}",
            f"api_key  = {mask_secret(config.api_key)}",
            f"subject  = {config.subject}",
        ]
    )
