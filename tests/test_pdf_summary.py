from pathlib import Path

import pytest

from academic_assistant import pdf_summary
from academic_assistant.config import Config
from academic_assistant.pdf_summary import create_summary, ensure_db, run_all_summaries, source_key
from pdf_factory import make_pdf

CONFIG = Config(base_url="http://llm.test/v1", model="test-model", api_key="k", subject="numerical methods")


@pytest.fixture
def lectures(workdir):
    """Two synthetic lecture PDFs in a subfolder plus one at the top level."""
    make_pdf(workdir / "lectures/01-power.pdf", [["Power method", "Dominant eigenvalue (lambda_1)"], ["Rayleigh quotient"]])
    make_pdf(workdir / "lectures/02-krylov.pdf", [["Krylov subspaces", "Arnoldi iteration"]])
    make_pdf(workdir / "03-top.pdf", [["Top level lecture on QR"]])
    return workdir


def test_load_pdf_pages_joins_all_pages(lectures):
    text = pdf_summary._load_pdf_pages("lectures/01-power.pdf")
    assert "Dominant eigenvalue (lambda_1)" in text
    assert "Rayleigh quotient" in text


@pytest.mark.parametrize("spelling", ["lectures/01-power.pdf", "./lectures/01-power.pdf", "lectures/../lectures/01-power.pdf"])
def test_source_key_normalizes_relative_spellings(workdir, spelling):
    assert source_key(spelling) == "lectures/01-power.pdf"


def test_source_key_accepts_absolute_paths(workdir):
    assert source_key(str(workdir / "lectures" / "01-power.pdf")) == "lectures/01-power.pdf"


def test_ensure_db_no_matches(workdir, capsys):
    assert ensure_db("*.pdf") == (None, [])
    assert "No PDF files matched" in capsys.readouterr().out


def test_ensure_db_skips_blank_pdfs(workdir, capsys):
    make_pdf(workdir / "scanned.pdf", [[]])
    make_pdf(workdir / "text.pdf", [["Some text"]])
    db, files = ensure_db("*.pdf")
    assert files == ["scanned.pdf", "text.pdf"]
    assert db._collection.count() == 1
    assert "no text could be extracted from scanned.pdf" in capsys.readouterr().out


def test_ensure_db_only_blank_pdfs_returns_none(workdir):
    make_pdf(workdir / "scanned.pdf", [[]])
    db, files = ensure_db("*.pdf")
    assert db is None and files == ["scanned.pdf"]


def test_ensure_db_reuses_existing_and_rebuild_replaces(lectures):
    db, _ = ensure_db("lectures/*.pdf")
    assert db._collection.count() == 2
    db, _ = ensure_db("*.pdf")  # reused as-is: glob differs but the DB already exists
    assert db._collection.count() == 2

    db, _ = ensure_db("*.pdf", rebuild=True)
    sources = {m["source"] for m in db._collection.get()["metadatas"]}
    assert sources == {"03-top.pdf"}


def test_summary_from_subfolder_glob_gets_its_own_context(lectures, fake_llm):
    # Regression: the filter used the bare file name while metadata stored the
    # glob path, so subfolder PDFs were summarized from an empty context.
    fake_llm.responses = [r"# Power method \(A x\)"]
    out = create_summary("lectures/01-power.pdf", CONFIG, pdf_glob="lectures/*.pdf")

    assert out == Path("summaries/01-power.md")
    assert out.read_text() == "# Power method $A x$"
    (prompt,) = fake_llm.prompts
    assert "Dominant eigenvalue" in prompt and "Arnoldi" not in prompt
    assert "numerical methods" in prompt
    assert fake_llm.connections == [("test-model", "http://llm.test/v1", "k")]


def test_summary_custom_output_path(lectures, fake_llm):
    out = create_summary("03-top.pdf", CONFIG, output_path="notes")
    assert out == Path("notes/03-top.md")


def test_summary_missing_file(lectures, fake_llm, capsys):
    assert create_summary("nope.pdf", CONFIG) is None
    assert "not found" in capsys.readouterr().out
    assert fake_llm.connections == []


def test_summary_of_pdf_not_in_db_does_not_call_llm(lectures, fake_llm, capsys):
    ensure_db("lectures/*.pdf")
    assert create_summary("03-top.pdf", CONFIG) is None
    assert "not in the RAG database" in capsys.readouterr().out
    assert fake_llm.prompts == []


def test_run_all_summarizes_every_pdf(lectures, fake_llm):
    fake_llm.responses = ["# One", "# Two"]
    assert run_all_summaries(CONFIG, pdf_glob="lectures/*.pdf") is True
    assert Path("summaries/01-power.md").read_text() == "# One"
    assert Path("summaries/02-krylov.md").read_text() == "# Two"
    assert "Power method" in fake_llm.prompts[0] and "Krylov" in fake_llm.prompts[1]


def test_run_all_reports_partial_failure(workdir, fake_llm):
    make_pdf(workdir / "a.pdf", [["Readable"]])
    make_pdf(workdir / "b.pdf", [[]])
    assert run_all_summaries(CONFIG) is False
    assert Path("summaries/a.md").exists()
    assert not Path("summaries/b.md").exists()


def test_run_all_with_no_pdfs(workdir, fake_llm):
    assert run_all_summaries(CONFIG) is False
