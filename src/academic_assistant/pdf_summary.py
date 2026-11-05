import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from pypdf import PdfReader

from . import llm
from .config import Config
from .utils.chroma_utils import get_chroma, open_empty_chroma
from .utils.math_format import normalize_math_delimiters
from .utils.prompts import pdf_summary_prompt


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "slides"


def _load_pdf_pages(pdf_path: str) -> str:
    # Avoid LangChain PDF loaders to prevent pulling in langchain_text_splitters
    # (which imports sentence-transformers/torch in your environment).
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        raw = page.extract_text()
        t = raw if isinstance(raw, str) else ""
        if t.strip():
            texts.append(t)
    return "\n\n".join(texts)


def source_key(path: str) -> str:
    """Key stored as `source` metadata; ingest and retrieval must agree on it."""
    try:
        return Path(os.path.relpath(path)).as_posix()
    except ValueError:  # Windows: path on another drive than the working dir
        return Path(path).resolve().as_posix()


def ensure_db(pdf_glob: str = "*.pdf", rebuild: bool = False) -> Tuple[Optional[Chroma], List[str]]:
    pdf_files = sorted(glob.glob(pdf_glob))
    if not pdf_files:
        print(f"No PDF files matched: {pdf_glob}")
        return None, []

    if os.path.exists(CHROMA_PATH) and not rebuild:
        print(f"Using existing Chroma DB at {CHROMA_PATH}...")
        return get_chroma(persist_directory=CHROMA_PATH, collection_name=COLLECTION_NAME), pdf_files

    print("Creating Chroma vector store from PDFs...")
    texts = []
    metadatas = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        text = _load_pdf_pages(pdf_file)
        text = text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        if text.strip():
            texts.append(text)
            metadatas.append({"source": source_key(pdf_file)})
        else:
            print(f"Warning: no text could be extracted from {pdf_file} (scanned PDF?).")

    if not texts:
        print("No content could be extracted from the PDFs.")
        return None, pdf_files

    # Chroma uses its default embedding function (see utils/chroma_utils.py).
    db = open_empty_chroma(persist_directory=CHROMA_PATH, collection_name=COLLECTION_NAME)
    db.add_texts(texts=texts, metadatas=metadatas)
    print("RAG database created successfully.")
    return db, pdf_files


def create_summary(
    filename: str,
    config: Config,
    db: Optional[Chroma] = None,
    pdf_glob: str = "*.pdf",
    output_path: str = "summaries",
) -> Optional[Path]:
    """Summarize one PDF; returns the written markdown path, or None on failure."""
    file_path = Path(filename)
    if not file_path.exists():
        print(f"Error: File {filename} not found.")
        return None

    if db is None:
        db, _ = ensure_db(pdf_glob=pdf_glob)
        if db is None:
            return None

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8, "filter": {"source": source_key(filename)}},
    )

    print("Retrieving context...")
    docs = retriever.invoke("Summarize the document")
    if not docs:
        print(
            f"Error: {filename} is not in the RAG database. "
            f"Run `academic-assistant ingest --pdf-glob ...` with a glob that includes it."
        )
        return None
    context = "\n\n".join(d.page_content for d in docs)

    print(f"Querying LLM '{config.model}' at {config.base_url}...")
    chat_model = llm.make_chat_model(config.model, config.base_url, config.api_key)
    messages = pdf_summary_prompt(config.subject).format_messages(context=context)
    try:
        response = chat_model.invoke(messages)
    except Exception as e:  # connection/auth/model errors: report and let --all continue
        print(f"Error: the LLM request for {filename} failed: {e}")
        return None

    out_dir = Path(output_path) if output_path else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_filename = out_dir / f"{file_path.stem}.md"
    output_filename.write_text(normalize_math_delimiters(response.content), encoding="utf-8")

    print(f"Summary successfully created and saved to {output_filename}")
    return output_filename


def run_all_summaries(config: Config, pdf_glob: str = "*.pdf") -> bool:
    """Rebuild the DB and summarize every PDF; returns True if all succeeded."""
    db, pdf_files = ensure_db(pdf_glob=pdf_glob, rebuild=True)
    if db is None:
        return False

    ok = True
    for pdf_file in pdf_files:
        print("\n" + "=" * 80)
        print(f"Summarizing {pdf_file}...")
        ok = create_summary(pdf_file, config, db=db) is not None and ok
    return ok
