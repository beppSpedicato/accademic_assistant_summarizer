# academic-assistant

A command-line tool that turns lecture PDFs into **GitHub-friendly Markdown summaries** and lets you **chat over those summaries**, using a local **Chroma** RAG index and any **OpenAI-compatible** chat endpoint (OpenAI, LM Studio, vLLM, LiteLLM proxy, text-generation-webui, ...).

## Install

Requires **Python >= 3.10**.

```bash
# From GitHub
pip install "git+https://github.com/beppSpedicato/accademic_assistant_summarizer.git"

# From a local clone
pip install .
```

This installs the `academic-assistant` command (also runnable as `python -m academic_assistant`).

> Chroma's default embedding model (~80 MB, ONNX all-MiniLM-L6-v2) is downloaded to `~/.cache/chroma` the first time you ingest or index.

## Configure (once per machine)

```bash
academic-assistant config init
```

The wizard asks for:

| Setting | Example |
|---|---|
| Base URL of the OpenAI-compatible API | `http://localhost:4141/v1` |
| Model name | `gpt-5.2` |
| API key (typed hidden; leave empty if your endpoint doesn't need one) | `sk-...` |
| Course subject (used in the prompts) | `computational linear algebra` |

Press Enter to keep the value shown in brackets, so you can re-run `config init` to change a single setting. The configuration is shared by every subcommand and stored in (on macOS/Linux with owner-only `0600` permissions, since it holds the API key):

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/academic-assistant/config.toml` |
| Linux | `~/.config/academic-assistant/config.toml` |
| Windows | `%LOCALAPPDATA%\academic-assistant\config.toml` |

`academic-assistant config show` prints the current values (API key masked). For a single run you can override the saved values with `--url`, `--model` and `--api-key` on `summarize` and `chat`.

## Use

Run the commands from the folder that contains your PDFs, or point them at it with `--workdir <dir>` (accepted before or after the subcommand). Everything the tool generates is written inside that folder.

```bash
# 1. Build the PDF RAG database (./chroma_db) from *.pdf
academic-assistant ingest
academic-assistant ingest --pdf-glob "slides/*.pdf"

# 2. Summarize one PDF, or rebuild the database and summarize all of them
academic-assistant summarize 01-SparseMatrices.pdf      # -> summaries/01-SparseMatrices.md
academic-assistant summarize --all --pdf-glob "slides/*.pdf"

# 3. Index the summaries for chat (incremental: only changed files are re-embedded)
academic-assistant index                                 # -> ./chroma_db_summaries, summaries/manifest.json

# 4. Ask questions over the summaries
academic-assistant chat
```

Chat commands: `/reindex` updates the summaries index, `/clean` removes `.tmp/`, `/exit` quits. When run from a VS Code terminal with the `code` CLI available, each answer is also opened as a Markdown file under `.tmp/`.

### Options

| Command | Option | Default |
|---|---|---|
| `ingest`, `summarize` | `--pdf-glob <glob>` | `*.pdf` |
| `index` | `--md-glob <glob>` | `summaries/*.md` |
| `index` | `--chunk-size <n>` / `--overlap <n>` | `3000` / `300` characters |
| `chat` | `--k <n>` (retrieved chunks per question) | `8` |
| `summarize`, `chat` | `--url`, `--model`, `--api-key` | saved config |

Each summary follows the same structure: `# Title`, `## Key Concepts`, `## Detailed Notes`, `## Action Items / Study Questions`, with math as `$...$` / `$$...$$`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # whole suite (no network, no LLM needed)
pytest tests/test_config.py::test_save_load_roundtrip   # a single test
python -m build                          # sdist + wheel in dist/
```

CI (`.github/workflows/tests.yml`) runs the suite on Linux, macOS and Windows with Python 3.10–3.13 and smoke-tests the built wheel.
