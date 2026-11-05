import glob
import os
from pathlib import Path

from . import llm
from .utils.chroma_utils import get_chroma
from langchain_chroma import Chroma
from pypdf import PdfReader

from .utils.math_format import normalize_math_delimiters
from .utils.prompts import pdf_summary_prompt


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "slides"


def _load_pdf_pages(pdf_path: str):
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


def _ensure_db(pdf_glob: str = "*.pdf", rebuild: bool = False):
    pdf_files = sorted(glob.glob(pdf_glob))
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return None, []

    # IMPORTANT: We avoid sentence-transformers/torch here to keep installs light and
    # to prevent the import chain that breaks on some environments.
    # Chroma will use its default embedding function.

    if rebuild and os.path.exists(CHROMA_PATH):
        # Safe-delete local DB folder
        for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(CHROMA_PATH)

    if os.path.exists(CHROMA_PATH):
        print(f"Using existing Chroma DB at {CHROMA_PATH}...")
        return get_chroma(persist_directory=CHROMA_PATH, collection_name=COLLECTION_NAME), pdf_files

    print("Creating Chroma vector store from PDFs...")
    texts = []
    metadatas = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        text = _load_pdf_pages(pdf_file)
        if not isinstance(text, str):
            text = "" if text is None else str(text)
            
        text = text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        if text.strip():
            texts.append(text)
            metadatas.append({"source": pdf_file})

    if not texts:
        print("No content could be extracted from the PDFs.")
        return None, pdf_files


    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=None,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    print("RAG database created successfully.")
    return db, pdf_files


def create_summary(filename: str, model_name: str, base_url: str, api_key: str, db=None, output_path='summaries'):
    file_path = Path(filename)
    if not file_path.exists():
        print(f"Error: File {filename} not found.")
        return

    if db is None:
        db, _ = _ensure_db()
        if db is None:
            return

    prompt = pdf_summary_prompt()

    print(f"Connecting to LLM '{model_name}' at {base_url}...")
    chat_model = llm.make_chat_model(model_name, base_url, api_key)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8, "filter": {"source": file_path.name}},
    )

    print("Retrieving context...")
    docs = retriever.invoke("Summarize the document")
    context = "\n\n".join(d.page_content for d in docs)

    print("Querying the LLM...")
    messages = prompt.format_messages(context=context)
    response = chat_model.invoke(messages)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_filename = Path(output_path) / f"{file_path.stem}.md"
    else:
        output_filename = f"{file_path.stem}.md"
        
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(normalize_math_delimiters(response.content))

    print(f"Summary successfully created and saved to {output_filename}")


def run_all_summaries(model_name: str, base_url: str, api_key: str, pdf_glob: str = "*.pdf"):
    db, pdf_files = _ensure_db(pdf_glob=pdf_glob, rebuild=True)
    if db is None:
        return

    for pdf_file in pdf_files:
        print("\n" + "=" * 80)
        print(f"Summarizing {pdf_file}...")
        create_summary(pdf_file, model_name, base_url, api_key, db=db)
