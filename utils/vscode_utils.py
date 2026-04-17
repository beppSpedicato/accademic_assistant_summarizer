import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def is_code_cli_available() -> bool:
    """Return True if the 'code' CLI is on PATH."""

    return shutil.which("code") is not None


def is_running_inside_vscode() -> bool:
    """Best-effort check whether this process is launched from VS Code."""

    # TODO: This is a very naive check. We might want to check for specific environment variables that VS Code sets, or other heuristics.
    return True


def open_markdown_in_vscode_tmp(
    markdown: str,
    filename_hint: str = "chat.md",
    tmp_dir: str = ".tmp",
    unique: bool = True,
) -> Optional[Path]:
    """Create a temp markdown file under tmp_dir and open it in VS Code.

    If unique=True, the file name is made unique by appending a uuid4 suffix.
    Returns the written path when created, otherwise None.
    """

    if not (is_code_cli_available() and is_running_inside_vscode()):
        return None

    base = Path(tmp_dir)
    base.mkdir(parents=True, exist_ok=True)

    safe_name = filename_hint.strip() or "chat.md"
    if not safe_name.lower().endswith(".md"):
        safe_name = f"{safe_name}.md"

    if unique:
        from uuid import uuid4

        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix or ".md"
        safe_name = f"{stem}_{uuid4().hex}{suffix}"

    path = base / safe_name
    path.write_text(markdown, encoding="utf-8")

    # -r: reuse existing window
    subprocess.run(["code", "-r", str(path)], check=False)
    return path


def clean_tmp_dir(tmp_dir: str = ".tmp") -> None:
    base = Path(tmp_dir)
    if base.exists():
        shutil.rmtree(base)
