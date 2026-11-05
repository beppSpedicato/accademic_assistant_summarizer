import json
from pathlib import Path

import pytest

from academic_assistant import summaries_chat
from academic_assistant.config import Config
from academic_assistant.summaries_chat import (
    _extract_title,
    _split_markdown,
    answer_question,
    chat_repl,
    update_index,
)

CONFIG = Config(base_url="http://llm.test/v1", model="chat-model", api_key="k", subject="numerical methods")


# --- chunking / title (unit) ---------------------------------------------


def test_split_empty_text():
    assert _split_markdown("   \n ") == []


def test_split_short_text_is_one_chunk():
    assert _split_markdown("  hello  ", chunk_size=100) == ["hello"]


def test_split_overlapping_chunks_cover_text():
    text = "".join(chr(ord("a") + i % 26) for i in range(25))
    chunks = _split_markdown(text, chunk_size=10, overlap=3)
    assert chunks == [text[0:10], text[7:17], text[14:24], text[21:25]]
    assert all(len(c) <= 10 for c in chunks)


def test_split_non_positive_chunk_size_returns_whole_text():
    assert _split_markdown("abc", chunk_size=0) == ["abc"]


def test_split_overlap_not_smaller_than_chunk_still_terminates():
    chunks = _split_markdown("abcdef", chunk_size=3, overlap=5)
    assert chunks[0] == "abc" and chunks[-1] == "def"


@pytest.mark.parametrize(
    "md, title",
    [
        ("# Sparse Matrices\n## Key Concepts", "Sparse Matrices"),
        ("intro\n\n  # Indented Title  \n", "Indented Title"),
        ("## Only subheading\n#NoSpace", None),
        ("", None),
    ],
)
def test_extract_title(md, title):
    assert _extract_title(md) == title


# --- index (integration with Chroma) -------------------------------------


@pytest.fixture
def summaries(workdir):
    d = workdir / "summaries"
    d.mkdir()
    (d / "01-power.md").write_text("# Power Method\n\nDominant eigenvalue via repeated multiplication.\n")
    (d / "02-krylov.md").write_text("# Krylov\n\nArnoldi builds an orthonormal basis.\n")
    return d


def _indexed():
    rows = summaries_chat._get_db()._collection.get()["metadatas"]
    return sorted((m["source_path"], m["chunk_index"]) for m in rows)


def _manifest():
    return json.loads(Path(summaries_chat.MANIFEST_PATH).read_text())


def test_index_builds_chunks_with_metadata(summaries, capsys):
    update_index(chunk_size=20, overlap=5)
    assert "Updated: 2, skipped: 0, deleted: 0" in capsys.readouterr().out

    rows = summaries_chat._get_db()._collection.get()["metadatas"]
    titles = {m["source_path"]: m["title"] for m in rows}
    assert titles == {"summaries/01-power.md": "Power Method", "summaries/02-krylov.md": "Krylov"}
    assert len(rows) > 2  # small chunk size -> several chunks per file
    assert set(_manifest()["files"]) == {"summaries/01-power.md", "summaries/02-krylov.md"}


def test_index_is_incremental(summaries, capsys):
    update_index()
    update_index()
    assert "Updated: 0, skipped: 2, deleted: 0" in capsys.readouterr().out.splitlines()[-3]

    (summaries / "01-power.md").write_text("# Power Method v2\n\nChanged.\n")
    (summaries / "02-krylov.md").unlink()
    (summaries / "03-qr.md").write_text("# QR\n\nHouseholder reflections.\n")
    update_index()

    assert "Updated: 2, skipped: 0, deleted: 1" in capsys.readouterr().out
    assert _indexed() == [("summaries/01-power.md", 0), ("summaries/03-qr.md", 0)]
    docs = summaries_chat._get_db()._collection.get(where={"source_path": "summaries/01-power.md"})["documents"]
    assert docs == ["# Power Method v2\n\nChanged."]
    assert set(_manifest()["files"]) == {"summaries/01-power.md", "summaries/03-qr.md"}


def test_index_no_matches(workdir, capsys):
    update_index()
    assert "No markdown files matched" in capsys.readouterr().out
    assert not Path(summaries_chat.MANIFEST_PATH).exists()


# --- Q&A -----------------------------------------------------------------


def test_answer_question_builds_prompt_from_retrieved_chunks(summaries, fake_llm):
    update_index()
    fake_llm.responses = [r"# Answer \(\lambda\)"]
    out = answer_question("What is the power method?", CONFIG, k=2)

    assert out == r"# Answer $\lambda$"
    (prompt,) = fake_llm.prompts
    assert "SOURCE: summaries/01-power.md" in prompt
    assert "SOURCE: summaries/02-krylov.md" in prompt
    assert "What is the power method?" in prompt
    assert "tutor for numerical methods" in prompt
    assert fake_llm.connections == [("chat-model", "http://llm.test/v1", "k")]


def test_answer_question_respects_k(summaries, fake_llm):
    update_index()
    answer_question("q", CONFIG, k=1)
    assert fake_llm.prompts[0].count("SOURCE:") == 1


def _scripted_input(monkeypatch, lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


def test_repl_answers_and_handles_commands(summaries, fake_llm, monkeypatch, capsys):
    update_index()
    (summaries / "03-qr.md").write_text("# QR\n\nHouseholder.\n")
    opened = []
    monkeypatch.setattr(summaries_chat, "open_markdown_in_vscode_tmp", lambda md, **kw: opened.append(md))
    fake_llm.responses = ["# First answer"]
    _scripted_input(monkeypatch, ["", "Explain Krylov", "/reindex", "/clean", "/exit", "never asked"])

    chat_repl(CONFIG, k=2)

    out = capsys.readouterr().out
    assert "First answer" in out
    assert "Updated: 1, skipped: 2" in out
    assert "Cleaned .tmp/" in out
    assert len(fake_llm.prompts) == 1
    assert opened == ["# First answer"]


def test_repl_survives_llm_errors_and_exits_on_eof(summaries, fake_llm, monkeypatch, capsys):
    update_index()
    fake_llm.error = RuntimeError("connection refused")
    _scripted_input(monkeypatch, ["question?"])

    chat_repl(CONFIG)

    assert "Error: connection refused" in capsys.readouterr().out
