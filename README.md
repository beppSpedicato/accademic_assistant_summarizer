# Local RAG Summarizer (OpenAI-Compatible API)

A small CLI utility to:

- build a local **Chroma** knowledge base from all PDFs in this folder
- query an **OpenAI-compatible** chat endpoint to generate **GitHub-friendly Markdown** summaries
- follow the course ingestion format described in `CLAUDE.md` (Key Concepts / Detailed Notes / Study Questions)

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

### 1) Rebuild the RAG database (fresh ingest)

This scans `*.pdf`, extracts text with `pypdf`, and writes a Chroma DB to `./chroma_db`.

```bash
python rag_summary.py --init
```

### 2) Summarize a single PDF

```bash
python rag_summary.py --summarize "example.pdf" --url "http://localhost:8000/v1" --model "your-model"
```

Outputs a Markdown file under `./summaries/` (e.g. `summaries/01-SparseMatrices.md`).

### 3) Full run: rebuild RAG + summarize every PDF

This is the “one command” mode: it rebuilds the DB, then loops through all matching PDFs and writes one `.md` per PDF.

```bash
python rag_summary.py --run-all --url "http://localhost:4141/v1" --model "your-model"
```

### Options

- `--init`: rebuild the RAG database from all PDFs
- `--summarize <filename.pdf>`: summarize one PDF
- `--run-all`: rebuild the RAG database and summarize each PDF
- `--pdf-glob <glob>`: choose PDFs to include (default: `*.pdf`)
- `--url <url>`: base URL for the OpenAI-compatible API (default: `http://localhost:4141/v1`)
- `--model <model>`: model name (default: `gpt-5.2`)
- `--api-key <key>`: API key if your proxy requires one (default: `dummy`)

## Notes

- The summaries are written to the `summaries/` folder by default.
