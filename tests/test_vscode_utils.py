from academic_assistant.utils import vscode_utils
from academic_assistant.utils.vscode_utils import (
    clean_tmp_dir,
    is_running_inside_vscode,
    open_markdown_in_vscode_tmp,
)


def test_detects_vscode_terminal(monkeypatch):
    assert not is_running_inside_vscode()
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert is_running_inside_vscode()
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert not is_running_inside_vscode()


def test_does_not_open_outside_vscode(monkeypatch, workdir):
    # Regression: the check used to always return True.
    monkeypatch.setattr(vscode_utils.shutil, "which", lambda name: "/usr/bin/code")
    calls = []
    monkeypatch.setattr(vscode_utils.subprocess, "run", lambda *a, **k: calls.append(a))
    assert open_markdown_in_vscode_tmp("# hi") is None
    assert calls == []
    assert not (workdir / ".tmp").exists()


def test_opens_unique_file_inside_vscode(monkeypatch, workdir):
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    monkeypatch.setattr(vscode_utils.shutil, "which", lambda name: "/usr/bin/code")
    calls = []
    monkeypatch.setattr(vscode_utils.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    first = open_markdown_in_vscode_tmp("# one", filename_hint="answer")
    second = open_markdown_in_vscode_tmp("# two", filename_hint="answer")

    assert first != second
    assert first.suffix == ".md" and first.read_text() == "# one"
    assert calls == [["code", "-r", str(first)], ["code", "-r", str(second)]]

    clean_tmp_dir()
    assert not (workdir / ".tmp").exists()


def test_clean_tmp_dir_missing_is_noop(workdir):
    clean_tmp_dir()
