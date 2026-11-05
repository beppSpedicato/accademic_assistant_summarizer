# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`academic-assistant`: a pip-installable CLI that summarizes lecture PDFs into Markdown and offers RAG Q&A over those summaries, using local Chroma stores and any **OpenAI-compatible** chat endpoint (via `langchain-openai`). Python >= 3.10. User-facing docs are in `README.md`.

## Commands

```bash
pip install -e ".[dev]"                 # dev install (pytest, build)
pytest                                   # full suite; no network or LLM needed
pytest tests/test_cli.py::test_full_workflow_with_workdir   # single test
python -m build                          # sdist + wheel
academic-assistant --help                # or: python -m academic_assistant --help
```

Subcommands: `config init|show`, `ingest`, `summarize FILE|--all`, `index`, `chat`. `--workdir` works before or after the subcommand.

CI: `.github/workflows/tests.yml` (Linux/macOS/Windows × Python 3.10–3.13, plus a build + clean-venv wheel smoke test).

## Architecture (`src/academic_assistant/`)

- `cli.py`: argparse entry point (`main(argv) -> exit code`). Heavy modules (chromadb/langchain) are imported lazily inside the dispatch. `summarize`/`chat` resolve the config *before* `--workdir` chdir so a missing config fails fast.
- `config.py`: the single per-machine config (`config.toml` in `platformdirs.user_config_dir("academic-assistant")`, written atomically with 0600). Resolution is **CLI flag > config file > dataclass defaults**; environment variables and `.env` are intentionally not read. `config_path()` is the one function tests patch.
- `pdf_summary.py`: `ensure_db` extracts PDF text with `pypdf` (one Chroma document per PDF, `source` metadata = `source_key(path)`, a normalized relative POSIX path) into `./chroma_db`; `create_summary` retrieves with `filter={"source": source_key(file)}` — ingest and retrieval must use the same key. Writes `summaries/<stem>.md`.
- `summaries_chat.py`: `update_index` chunks `summaries/*.md` into `./chroma_db_summaries`, tracking sha256 per file in `summaries/manifest.json` (incremental add/update/delete). `answer_question` / `chat_repl` do the Q&A.
- `llm.py`: `make_chat_model` is the only place `ChatOpenAI` is constructed; call it as `llm.make_chat_model(...)` (module attribute) so tests can swap in a fake.
- `utils/chroma_utils.py`: `get_chroma` (resolves the persist path to absolute because Chroma caches clients by path string), `get_embedding_function` (returns `None` = Chroma's default ONNX embedding; tests replace it), `remove_persist_dir` (clears Chroma's client cache before deleting, otherwise a same-process rebuild fails with "readonly database").
- `utils/prompts.py`: both prompt templates; the course subject comes from config via `.partial(subject=...)`.

Every generated artifact (`chroma_db/`, `chroma_db_summaries/`, `summaries/`, `.tmp/`) is relative to the working directory; only the config is global.

## Summary format convention

The prompts in `utils/prompts.py` require this structure (tests check it):

- `# <Title>`
- `## Key Concepts`
- `## Detailed Notes`
- `## Action Items / Study Questions`

LaTeX `\( \)` / `\[ \]` are normalized to `$` / `$$` by `utils/math_format.py` before writing.

## Tests

`tests/conftest.py` has an autouse fixture that points the config at a tmp path, injects `DeterministicFakeEmbedding`, unsets `TERM_PROGRAM` (so VS Code is never opened) and clears Chroma's client cache. Use the `fake_llm` fixture (records connections/prompts, queued `responses`) and `workdir`/`config_file` fixtures. Synthetic PDFs come from `tests/pdf_factory.make_pdf` (no extra dependency).

## Local environment quirk

This checkout lives in an iCloud-synced Desktop: macOS re-flags new files as hidden, and Python 3.13 skips hidden `.pth` files, so the editable install in `.venv` may not import outside the repo. `pytest` is unaffected (`pythonpath = ["src"]`); for CLI runs use `PYTHONPATH=src python -m academic_assistant ...` or a venv outside iCloud. The first test run after edits can stall while iCloud syncs `__pycache__`; `PYTHONDONTWRITEBYTECODE=1` avoids it.
