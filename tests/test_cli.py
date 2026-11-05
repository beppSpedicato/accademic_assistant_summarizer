import os
import subprocess
import sys
from pathlib import Path

import pytest

from academic_assistant import __version__
from academic_assistant.cli import main
from academic_assistant.config import load_config
from pdf_factory import make_pdf

SRC = Path(__file__).resolve().parents[1] / "src"


def run_module(*args):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(
        [sys.executable, "-m", "academic_assistant", *args], capture_output=True, text=True, env=env
    )


def answer(monkeypatch, lines, secret=""):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": secret)


# --- process-level smoke tests ---------------------------------------------


def test_module_help_lists_subcommands():
    result = run_module("--help")
    assert result.returncode == 0
    for cmd in ("config", "ingest", "summarize", "index", "chat"):
        assert cmd in result.stdout


def test_module_version():
    result = run_module("--version")
    assert result.returncode == 0
    assert __version__ in result.stdout


# --- argument handling -----------------------------------------------------


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: academic-assistant" in capsys.readouterr().out


def test_summarize_requires_file_or_all(config_file, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["summarize"])
    assert exc.value.code == 2
    assert "pass a PDF file or --all" in capsys.readouterr().err


def test_config_requires_action(capsys):
    with pytest.raises(SystemExit):
        main(["config"])


@pytest.mark.parametrize("argv", [["summarize", "x.pdf"], ["summarize", "--all"], ["chat"], ["config", "show"]])
def test_missing_config_fails_with_hint(argv, workdir, capsys):
    assert main(argv) == 1
    assert "academic-assistant config init" in capsys.readouterr().err


def test_missing_config_does_not_change_directory(tmp_path, workdir):
    other = tmp_path / "other"
    other.mkdir()
    assert main(["--workdir", str(other), "chat"]) == 1
    assert Path.cwd().resolve() == workdir.resolve()


# --- config subcommands ------------------------------------------------------


def test_config_init_then_show(monkeypatch, capsys):
    answer(monkeypatch, ["http://llm.local/v1", "local-model", "Optimization"], secret="sk-abcdefghijkl")
    assert main(["config", "init"]) == 0
    assert load_config().model == "local-model"

    capsys.readouterr()
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "local-model" in out and "Optimization" in out
    assert "sk-...ijkl" in out and "sk-abcdefghijkl" not in out


# --- full workflow ---------------------------------------------------------


def test_full_workflow_with_workdir(tmp_path, monkeypatch, fake_llm, capsys):
    course = tmp_path / "course"
    make_pdf(course / "slides/01-power.pdf", [["Power method", "Dominant eigenvalue"]])
    make_pdf(course / "slides/02-krylov.pdf", [["Krylov subspaces", "Arnoldi iteration"]])

    answer(monkeypatch, ["http://llm.local/v1", "local-model", "Numerical linear algebra"], secret="sk-secret-key")
    assert main(["config", "init"]) == 0

    # --workdir works both before and after the subcommand.
    assert main(["--workdir", str(course), "ingest", "--pdf-glob", "slides/*.pdf"]) == 0
    fake_llm.responses = ["# Power\n\nDominant eigenvalue notes.", "# Krylov\n\nArnoldi notes."]
    assert main(["summarize", "--all", "--pdf-glob", "slides/*.pdf", "--workdir", str(course)]) == 0
    assert (course / "summaries/01-power.md").read_text().startswith("# Power")
    assert (course / "summaries/02-krylov.md").read_text().startswith("# Krylov")
    assert all(c == ("local-model", "http://llm.local/v1", "sk-secret-key") for c in fake_llm.connections)
    assert "Numerical linear algebra" in fake_llm.prompts[0]

    assert main(["--workdir", str(course), "index"]) == 0
    assert (course / "chroma_db_summaries").is_dir()
    assert (course / "summaries/manifest.json").is_file()

    fake_llm.responses = ["# Chat answer"]
    answer(monkeypatch, ["What does Arnoldi build?", "/exit"])
    capsys.readouterr()
    assert main(["--workdir", str(course), "chat", "--k", "2"]) == 0
    assert "Chat answer" in capsys.readouterr().out
    assert "What does Arnoldi build?" in fake_llm.prompts[-1]
    assert "SOURCE: summaries/" in fake_llm.prompts[-1]


def test_single_summarize_and_flag_overrides(config_file, workdir, fake_llm):
    make_pdf(workdir / "01-intro.pdf", [["Intro lecture"]])
    assert main(["summarize", "01-intro.pdf", "--model", "flag-model", "--url", "http://flag/v1"]) == 0
    assert fake_llm.connections == [("flag-model", "http://flag/v1", "sk-test-0123456789")]
    assert (workdir / "summaries/01-intro.md").exists()


def test_summarize_failures_exit_non_zero(config_file, workdir, fake_llm):
    assert main(["summarize", "missing.pdf"]) == 1
    assert main(["summarize", "--all"]) == 1  # no PDFs


def test_ingest_without_pdfs_exits_non_zero(workdir):
    assert main(["ingest"]) == 1
