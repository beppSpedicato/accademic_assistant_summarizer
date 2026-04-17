from langchain_chroma import Chroma


def get_chroma(persist_directory: str, collection_name: str) -> Chroma:
    # Keep consistent with this repo's approach: do not force heavyweight embeddings.
    return Chroma(persist_directory=persist_directory, collection_name=collection_name)
