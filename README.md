# Local RAG Summarizer (OpenAI-Compatible API)

A small CLI utility to:

- build a local **Chroma** knowledge base from all PDFs in this folder
- query an **OpenAI-compatible** chat endpoint to generate **GitHub-friendly Markdown** summaries

## Purpose

This repo folder contains lecture PDFs for a university course. The script `rag_summary.py` helps you quickly produce one Markdown summary per PDF, either for a single file or for the whole directory in one run.

## Requirements

- **Python >= 3.13**
- An OpenAI-compatible API endpoint (examples: LM Studio, vLLM, LiteLLM proxy, text-generation-webui, etc.)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### A) Build the PDFs RAG database (fresh ingest)

This scans `*.pdf`, extracts text with `pypdf`, and writes a Chroma DB to `./chroma_db`.

```bash
python rag_summary.py --init
```

### B) Summarize a single PDF

```bash
python rag_summary.py --summarize "example.pdf" --url "http://localhost:4141/v1" --model "your-model"
```

Outputs a Markdown file under `./summaries/` (e.g. `summaries/01-SparseMatrices.md`).

### C) Full run: rebuild PDF RAG + summarize every PDF

This is the “one command” mode: it rebuilds the DB, then loops through all matching PDFs and writes one `.md` per PDF.

```bash
python rag_summary.py --run-all --url "http://localhost:4141/v1" --model "your-model"
```

### D) Chat over the generated summaries (RAG Q&A)

After you have generated Markdown files under `./summaries/`, you can build a separate RAG index over those summaries and ask questions interactively.

1) Build/update the summaries index (incremental)

```bash
python rag_chat_summaries.py --init
```

This creates/updates:
- `./chroma_db_summaries/` (Chroma DB for summaries)
- `./summaries/manifest.json` (path + sha256, for safe incremental updates)

2) Start interactive chat

```bash
python rag_chat_summaries.py --chat --url "http://localhost:4141/v1" --model "your-model"
```

REPL commands:
- `/reindex` rebuilds/updates the summaries index
- `/exit` quits

### Options (PDF summarizer)

- `--init`: rebuild the RAG database from all PDFs
- `--summarize <filename.pdf>`: summarize one PDF
- `--run-all`: rebuild the RAG database and summarize each PDF
- `--pdf-glob <glob>`: choose PDFs to include (default: `*.pdf`)
- `--url <url>`: base URL for the OpenAI-compatible API (default: `http://localhost:4141/v1`)
- `--model <model>`: model name (default: `gpt-5.2`)
- `--api-key <key>`: API key if your proxy requires one (default: `dummy`)

### Options (summaries chat)

- `--init`: build/update the summaries RAG index + manifest
- `--chat`: start the interactive REPL
- `--md-glob <glob>`: which markdown files to index (default: `summaries/*.md`)
- `--chunk-size <n>` / `--overlap <n>`: chunking settings for indexing
- `--k <n>`: number of retrieved chunks per question
- `--url`, `--model`, `--api-key`: same meaning as above

## Export commands (add to PATH)

This repo includes two wrapper commands under `./bin/`:

- `bin/rag-pdf` → runs `rag_summary.py` (PDF → Markdown summaries)
- `bin/rag-chat` → runs `rag_chat_summaries.py` (chat over `summaries/*.md`)

### Make the commands available anywhere

Add this repo's `bin/` directory to your shell `PATH`.

**zsh** (`~/.zshrc`):

```bash
export PATH="/absolute/path/to/this/repo/bin:$PATH"
```

Then restart your terminal (or `source ~/.zshrc`) and you can run:

```bash
rag-pdf --help
rag-chat --help
```

### Running against any folder (`--workdir`)

Both commands accept `--workdir` to operate on a target directory, even if you run the command from somewhere else.

Examples:

```bash
# Summarize PDFs located in /path/to/lectures
rag-pdf --workdir "/path/to/lectures" --run-all --url "http://localhost:4141/v1" --model "your-model"

# Build the summaries index and start a chat over /path/to/lectures/summaries/*.md
rag-chat --workdir "/path/to/lectures" --init
rag-chat --workdir "/path/to/lectures" --chat --url "http://localhost:4141/v1" --model "your-model"
```

## Notes

- The summaries are written to the `summaries/` folder by default.
- The summaries chat index is stored separately in `./chroma_db_summaries`.
