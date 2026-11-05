import shutil
from pathlib import Path
from typing import Optional

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_embedding_function() -> Optional[Embeddings]:
    # None lets Chroma use its built-in default (ONNX all-MiniLM-L6-v2) instead of
    # pulling in sentence-transformers/torch. Tests replace this with fake embeddings.
    return None


def remove_persist_dir(persist_directory: str) -> None:
    """Delete a persisted DB, dropping Chroma's cached client for it first.

    Otherwise a client opened earlier in the process keeps writing to the
    deleted files ("attempt to write a readonly database"). On Windows this
    still fails while the process has the index files open (WinError 32), so
    rebuilds use open_empty_chroma() and only fall back to this.
    """
    SharedSystemClient.clear_system_cache()
    shutil.rmtree(persist_directory)


def get_chroma(persist_directory: str, collection_name: str) -> Chroma:
    # Chroma caches clients by the path string, so a relative path would keep
    # pointing at the first working directory it was opened from.
    return Chroma(
        persist_directory=str(Path(persist_directory).resolve()),
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
    )


def open_empty_chroma(persist_directory: str, collection_name: str) -> Chroma:
    """Open a collection with all its records deleted, ready to be rebuilt.

    Records are deleted through Chroma instead of removing the directory, so no
    file is deleted while a client in this process still has it open.
    """
    try:
        db = get_chroma(persist_directory, collection_name)
        ids = db.get(include=[])["ids"]
        if ids:
            db.delete(ids=ids)
        return db
    except Exception:
        # An existing DB Chroma cannot open (corrupt, or from an incompatible
        # Chroma version): nothing can be holding it open, so delete it.
        remove_persist_dir(persist_directory)
        return get_chroma(persist_directory, collection_name)
