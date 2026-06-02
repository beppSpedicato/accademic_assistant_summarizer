# Custom Claude Code Skills for this Repository

This repository provides two helper scripts in `./bin/` that can be exposed as **Claude Code custom skills**. Using the skill syntax (`/<skill-name>`) lets you run these commands from any directory without remembering the relative path.

---

## Skills

| Skill | Description | Command |
|------|-------------|---------|
| `rag-chat` | Run the RAG‑based chat/re‑indexer script. | `./bin/rag-chat` |
| `rag-pdf`  | Summarize a PDF using the RAG pipeline. | `./bin/rag-pdf` |

---

## How to Install the Skills

1. **Create a `settings.json**` in your Claude Code project configuration (e.g. `~/.claude/projects/<repo‑path>/.claude/settings.json`).
2. Add the following JSON under the `customSkills` key:

```json
{
  "customSkills": {
    "rag-chat": {
      "description": "Run the rag‑chat command from the repository",
      "command": "./bin/rag-chat"
    },
    "rag-pdf": {
      "description": "Run the rag‑pdf command from the repository",
      "command": "./bin/rag-pdf"
    }
  }
}
```

3. Restart Claude Code or reload the project so the new skills appear.

---

## Usage Examples (as documented in the repository README)

### Summarize a single PDF

```bash
! /rag-pdf --summarize "01-SparseMatrices.pdf" \
    --url "http://localhost:4141/v1" \
    --model "your-model"
```

### Run the chat/re‑indexer

```bash
! /rag-chat --run-all \
    --url "http://localhost:4141/v1" \
    --model "your-model"
```

### Build the local RAG database (ingest PDFs)

```bash
! /rag-chat --init
```

You can also pass a custom glob to limit the PDFs processed:

```bash
! /rag-chat --init --pdf-glob "already_summarized/*.pdf"
```

---

## Publishing

Add this `skills.md` file to the repository and reference it in the project README. When users clone the repo they can copy the JSON snippet into their Claude Code settings and instantly have the two skills available.
