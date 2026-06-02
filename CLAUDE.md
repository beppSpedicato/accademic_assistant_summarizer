# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This folder contains lecture PDFs for a university course. The script `rag_summary.py` builds a local Chroma vector store from PDFs and uses an **OpenAI-compatible** chat endpoint (via `langchain-openai`) to generate **GitHub-friendly Markdown** summaries.

Generated summaries are written to `./summaries/`.

## Common commands

### Setup

```bash
# Create/activate a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Build the local RAG database (ingest PDFs)

Scans PDFs (default `*.pdf`), extracts text with `pypdf`, and writes a Chroma DB to `./chroma_db`.

```bash
python rag_summary.py --init

# With a custom glob
python rag_summary.py --init --pdf-glob "already_summarized/*.pdf"
```

### Summarize a single PDF

```bash
python rag_summary.py --summarize "01-SparseMatrices.pdf" --url "http://localhost:4141/v1" --model "your-model"
```

### Full run (rebuild DB + summarize all PDFs)

```bash
python rag_summary.py --run-all --url "http://localhost:4141/v1" --model "your-model"

# Restrict which PDFs are processed
python rag_summary.py --run-all --pdf-glob "already_summarized/*.pdf" --url "http://localhost:4141/v1" --model "your-model"
```

### Lint / tests

No lint/test tooling is configured in this repo (no `pyproject.toml`, no `pytest` config). If you add a test suite later, document the commands here.

## High-level architecture

### Data flow

1. **PDF ingestion** (`rag_summary.py:_ensure_db`)
   - Finds PDFs using a glob.
   - Extracts text by iterating pages with `pypdf.PdfReader` (`rag_summary.py:_load_pdf_pages`).
   - Creates (or reuses) a persisted Chroma collection at `./chroma_db` with one document per PDF.
   - Stores metadata per document: `{ "source": <pdf filename> }`.

2. **Retrieval + prompt** (`rag_summary.py:create_summary`)
   - Builds a `ChatPromptTemplate` that enforces the on-disk markdown shape:
     `Key Concepts` / `Detailed Notes` / `Action Items / Study Questions`.
   - Uses `langchain_openai.ChatOpenAI` pointing at an OpenAI-compatible base URL (`--url`) and model name (`--model`).

3. **RAG query**
   - Calls `db.as_retriever(...)` with a metadata filter so retrieval is *scoped to the target PDF*:
     `filter: {"source": file_path.name}`.
   - Retrieves `k=8` chunks/documents (here effectively “top matches” from the stored text blob).
   - Concatenates retrieved `page_content` into a single `{context}` string.

4. **LLM invocation + output**
   - Calls `llm.invoke(messages)` and writes `response.content` to `summaries/<pdf_stem>.md`.

### Key directories / artifacts

- `rag_summary.py`: CLI entrypoint and all core logic.
- `chroma_db/`: persisted Chroma vector store (generated; safe to delete if you need a fresh rebuild).
- `summaries/`: generated Markdown summaries.
- `already_summarized/`: PDFs that appear to have existing summaries (used as an alternate `--pdf-glob` target).

### Operational notes (from code)

- `_ensure_db(..., rebuild=True)` deletes `./chroma_db` by walking the directory and removing files/dirs.
- The DB is created with `embedding=None` to avoid pulling in heavier embedding dependencies; Chroma will use its default embedding behavior in this configuration.

## Conventions used by the summarizer

The LLM output is expected to follow this exact section structure (see prompt in `rag_summary.py:create_summary`):

- `# <Title>`
- `## Key Concepts`
- `## Detailed Notes`
- `## Action Items / Study Questions`
