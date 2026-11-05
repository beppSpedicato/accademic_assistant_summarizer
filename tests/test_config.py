import os
import stat
import sys

import pytest

from academic_assistant import config as config_mod
from academic_assistant.config import (
    Config,
    ConfigError,
    load_config,
    mask_secret,
    resolve_config,
    run_wizard,
    save_config,
    show_config,
)


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_missing_config_points_to_config_init():
    with pytest.raises(ConfigError, match="academic-assistant config init"):
        load_config()


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "config.toml"
    cfg = Config(base_url="http://x/v1", model="m", api_key="k", subject="s")
    assert save_config(cfg, path) == path
    assert load_config(path) == cfg
    assert not path.with_suffix(".toml.tmp").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_config_file_is_owner_only(tmp_path):
    path = save_config(Config(api_key="secret"), tmp_path / "config.toml")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_partial_file_uses_defaults_and_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('model = "only-model"\nunknown = 1\n')
    assert load_config(path) == Config(model="only-model")


def test_invalid_toml_raises_config_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("model = \n")
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(path)


def test_resolve_applies_only_given_overrides(config_file):
    cfg = resolve_config(model="other-model")
    assert cfg.model == "other-model"
    assert cfg.base_url == "http://llm.test/v1"
    assert cfg.api_key == "sk-test-0123456789"


def test_resolve_allows_explicit_default_values(config_file):
    # Regression: an explicitly passed value equal to the old built-in default was ignored.
    assert resolve_config(model="gpt-5.2").model == "gpt-5.2"


@pytest.mark.parametrize(
    "secret, expected",
    [("", "(not set)"), ("short", "*****"), ("sk-abcdefghijkl", "sk-...ijkl")],
)
def test_mask_secret(secret, expected):
    assert mask_secret(secret) == expected


def test_wizard_fresh_install_uses_defaults_for_empty_answers(capsys):
    path = run_wizard(prompt=scripted(["", "my-model", ""]), secret_prompt=scripted([""]))
    assert path == config_mod.config_path()
    assert load_config() == Config(model="my-model")
    assert "Configuration saved" in capsys.readouterr().out


def test_wizard_keeps_existing_values_including_api_key(config_file):
    seen = []
    run_wizard(
        prompt=lambda p: (seen.append(p), "")[1],
        secret_prompt=lambda p: (seen.append(p), "")[1],
    )
    cfg = load_config()
    assert cfg.api_key == "sk-test-0123456789"
    assert cfg.model == "test-model"
    assert any("[http://llm.test/v1]" in p for p in seen)
    secret_prompt = next(p for p in seen if "API key" in p)
    assert "sk-...6789" in secret_prompt and "0123456789" not in secret_prompt


def test_wizard_overwrites_with_new_answers(config_file):
    run_wizard(
        prompt=scripted(["http://new/v1", "new-model", "Linear algebra"]),
        secret_prompt=scripted(["sk-new-key-99999"]),
    )
    assert load_config() == Config("http://new/v1", "new-model", "sk-new-key-99999", "Linear algebra")


def test_show_config_masks_key(config_file):
    out = show_config()
    assert "sk-...6789" in out
    assert "sk-test-0123456789" not in out
    assert str(config_file) in out
