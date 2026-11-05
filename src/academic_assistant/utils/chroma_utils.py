from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_embedding_function() -> Optional[Embeddings]:
    # None lets Chroma use its built-in default (ONNX all-MiniLM-L6-v2) instead of
    # pulling in sentence-transformers/torch. Tests replace this with fake embeddings.
    return None


def get_chroma(persist_directory: str, collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
    )
