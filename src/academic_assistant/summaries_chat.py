import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import llm
from .config import Config
from .utils.chroma_utils import get_chroma
from .utils.manifest import fingerprint, load_manifest, utc_now_iso, write_manifest_atomic
from .utils.math_format import normalize_math_delimiters
from .utils.prompts import summaries_qa_prompt
from .utils.vscode_utils import clean_tmp_dir, open_markdown_in_vscode_tmp


CHROMA_PATH = "./chroma_db_summaries"
COLLECTION_NAME = "summaries"
MANIFEST_PATH = "./summaries/manifest.json"
DEFAULT_MD_GLOB = "summaries/*.md"


def _split_markdown(text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        return [cleaned]

    step = max(1, chunk_size - max(0, overlap))
    chunks = []
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def _extract_title(md_text: str) -> Optional[str]:
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("# ") and len(s) > 2:
            return s[2:].strip()
    return None


def _get_db():
    return get_chroma(persist_directory=CHROMA_PATH, collection_name=COLLECTION_NAME)


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _relative_posix(path: Path) -> str:
    return path.as_posix()


def update_index(md_glob: str = DEFAULT_MD_GLOB, chunk_size: int = 3000, overlap: int = 300) -> None:
    manifest = load_manifest(
        Path(MANIFEST_PATH), default={"manifest_version": 1, "generated_at": None, "files": {}}
    )
    files: Dict[str, Any] = manifest.setdefault("files", {})

    md_paths = sorted(Path(p) for p in glob.glob(md_glob))
    if not md_paths:
        print(f"No markdown files matched: {md_glob}")
        return

    db = _get_db()

    seen = set()
    updated = 0
    skipped = 0

    for md_path in md_paths:
        rel = _relative_posix(md_path)
        seen.add(rel)

        fp = fingerprint(md_path)
        prev = files.get(rel)
        if prev and prev.get("sha256") == fp.sha256:
            skipped += 1
            continue

        text = _read_text(md_path)
        title = _extract_title(text)
        chunks = _split_markdown(text, chunk_size=chunk_size, overlap=overlap)

        # Remove prior chunks for this file (if any) then re-add.
        db.delete(where={"source_path": rel})

        if chunks:
            metadatas = []
            for i in range(len(chunks)):
                meta = {
                    "source_path": rel,
                    "sha256": fp.sha256,
                    "chunk_index": i,
                }
                if title:
                    meta["title"] = title
                metadatas.append(meta)

            db.add_texts(texts=chunks, metadatas=metadatas)

        files[rel] = {"sha256": fp.sha256, "size": fp.size, "mtime": fp.mtime}
        updated += 1

    # Handle deletions
    deleted = 0
    for rel in list(files.keys()):
        if rel.startswith("summaries/") and rel not in seen:
            db.delete(where={"source_path": rel})
            del files[rel]
            deleted += 1

    manifest["manifest_version"] = int(manifest.get("manifest_version", 1))
    manifest["generated_at"] = utc_now_iso()
    write_manifest_atomic(Path(MANIFEST_PATH), manifest)

    print(f"Index update complete. Updated: {updated}, skipped: {skipped}, deleted: {deleted}.")
    print(f"Chroma persist dir: {CHROMA_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


def answer_question(question: str, config: Config, k: int = 8) -> str:
    db = _get_db()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    docs = retriever.invoke(question)
    context_parts = []
    for d in docs:
        sp = d.metadata.get("source_path") if hasattr(d, "metadata") else None
        header = f"\n\n---\nSOURCE: {sp}\n---\n" if sp else "\n\n---\n---\n"
        context_parts.append(header + (d.page_content or ""))

    context = "\n".join(context_parts).strip()

    messages = summaries_qa_prompt(config.subject).format_messages(context=context, question=question)

    chat_model = llm.make_chat_model(config.model, config.base_url, config.api_key)
    resp = chat_model.invoke(messages)
    return normalize_math_delimiters(resp.content)


def _try_rich_console():
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


def _render_markdown(console, md_text: str) -> None:
    if console is None:
        print(md_text)
        return

    try:
        from rich.markdown import Markdown

        console.print(Markdown(md_text))
    except Exception:
        console.print(md_text)


def chat_repl(config: Config, k: int = 8) -> None:
    console = _try_rich_console()

    banner = (
        "Summaries Q&A (RAG). Type your question and press enter.\n"
        "Commands: /reindex, /clean, /exit\n"
    )
    if console:
        console.print(banner)
    else:
        print(banner)

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q in {"/exit", "exit", "quit", ":q"}:
            break
        if q == "/reindex":
            update_index()
            continue
        if q == "/clean":
            clean_tmp_dir()
            if console:
                console.print("Cleaned .tmp/")
            else:
                print("Cleaned .tmp/")
            continue

        try:
            out = answer_question(q, config, k=k)
            _render_markdown(console, out)
            open_markdown_in_vscode_tmp(out, filename_hint="rag_chat_answer.md", unique=True)
        except Exception as e:
            msg = f"Error: {e}"
            if console:
                console.print(msg)
            else:
                print(msg)
