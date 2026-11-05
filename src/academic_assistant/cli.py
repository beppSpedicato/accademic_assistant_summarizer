import argparse
import os
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

from . import __version__

DEFAULT_BASE_URL = "http://localhost:4141/v1"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_API_KEY = "dummy"


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=None, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--model", default=None, help="Name of the model")
    parser.add_argument("--api-key", default=None, help="API key if your endpoint requires one")


def _resolve_llm(args: argparse.Namespace):
    load_dotenv(find_dotenv(usecwd=True))
    base_url = args.url or os.getenv("BASE_URL") or DEFAULT_BASE_URL
    model = args.model or os.getenv("MODEL") or DEFAULT_MODEL
    api_key = args.api_key or os.getenv("API_KEY") or DEFAULT_API_KEY
    return model, base_url, api_key


def build_parser() -> argparse.ArgumentParser:
    # --workdir is accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workdir",
        default=argparse.SUPPRESS,
        help="Directory to run in (defaults to current directory)",
    )

    parser = argparse.ArgumentParser(
        prog="academic-assistant",
        description="Summarize lecture PDFs and chat over the summaries via an OpenAI-compatible API",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("ingest", parents=[common], help="(Re)build the PDF RAG database")
    p.add_argument("--pdf-glob", default="*.pdf", help="Glob pattern for PDFs (default: *.pdf)")

    p = sub.add_parser("summarize", parents=[common], help="Summarize one PDF, or all of them with --all")
    p.add_argument("file", nargs="?", help="PDF to summarize")
    p.add_argument("--all", action="store_true", help="Rebuild the PDF RAG database and summarize every PDF")
    p.add_argument("--pdf-glob", default="*.pdf", help="Glob pattern for PDFs (default: *.pdf)")
    _add_llm_args(p)

    p = sub.add_parser("index", parents=[common], help="Build/update the RAG index over the summaries")
    p.add_argument("--md-glob", default="summaries/*.md", help="Glob for markdown files (default: summaries/*.md)")
    p.add_argument("--chunk-size", type=int, default=3000, help="Chunk size in characters (default: 3000)")
    p.add_argument("--overlap", type=int, default=300, help="Chunk overlap in characters (default: 300)")

    p = sub.add_parser("chat", parents=[common], help="Interactive Q&A over the summaries index")
    p.add_argument("--k", type=int, default=8, help="Number of retrieved chunks per question (default: 8)")
    _add_llm_args(p)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if getattr(args, "workdir", None):
        os.chdir(args.workdir)

    # Imported lazily so `--help` stays fast (chromadb/langchain are heavy).
    if args.command == "ingest":
        from .pdf_summary import _ensure_db

        _ensure_db(pdf_glob=args.pdf_glob, rebuild=True)
    elif args.command == "summarize":
        from .pdf_summary import create_summary, run_all_summaries

        if not args.all and not args.file:
            parser.error("summarize: pass a PDF file or --all")
        model, base_url, api_key = _resolve_llm(args)
        if args.all:
            run_all_summaries(model, base_url, api_key, pdf_glob=args.pdf_glob)
        else:
            create_summary(args.file, model, base_url, api_key)
    elif args.command == "index":
        from .summaries_chat import update_index

        update_index(md_glob=args.md_glob, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command == "chat":
        from .summaries_chat import chat_repl

        model, base_url, api_key = _resolve_llm(args)
        chat_repl(model_name=model, base_url=base_url, api_key=api_key, k=args.k)

    return 0
