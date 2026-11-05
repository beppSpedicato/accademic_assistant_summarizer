import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .config import ConfigError, resolve_config, run_wizard, show_config


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("per-run overrides of the saved config")
    group.add_argument("--url", default=None, help="Base URL for the OpenAI-compatible API")
    group.add_argument("--model", default=None, help="Name of the model")
    group.add_argument("--api-key", default=None, help="API key for the endpoint")


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

    p = sub.add_parser("config", help="Set up or show the per-machine configuration")
    config_sub = p.add_subparsers(dest="config_command", metavar="<action>", required=True)
    config_sub.add_parser("init", help="Interactive setup of base URL, model, API key and course subject")
    config_sub.add_parser("show", help="Show the current configuration (API key masked)")

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

    try:
        return _dispatch(parser, args)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "config":
        if args.config_command == "init":
            run_wizard()
        else:
            print(show_config())
        return 0

    if args.command == "summarize" and not args.all and not args.file:
        parser.error("summarize: pass a PDF file or --all")

    # Fail on a missing config before changing directory or loading heavy deps.
    if args.command in ("summarize", "chat"):
        config = resolve_config(url=args.url, model=args.model, api_key=args.api_key)

    if getattr(args, "workdir", None):
        os.chdir(args.workdir)

    # Imported lazily so `--help` stays fast (chromadb/langchain are heavy).
    if args.command == "ingest":
        from .pdf_summary import ensure_db

        db, _ = ensure_db(pdf_glob=args.pdf_glob, rebuild=True)
        return 0 if db is not None else 1
    if args.command == "summarize":
        from .pdf_summary import create_summary, run_all_summaries

        if args.all:
            return 0 if run_all_summaries(config, pdf_glob=args.pdf_glob) else 1
        return 0 if create_summary(args.file, config, pdf_glob=args.pdf_glob) else 1
    if args.command == "index":
        from .summaries_chat import update_index

        update_index(md_glob=args.md_glob, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command == "chat":
        from .summaries_chat import chat_repl

        chat_repl(config, k=args.k)

    return 0
