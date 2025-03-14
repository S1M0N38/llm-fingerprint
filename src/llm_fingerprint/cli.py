"""Command-line interface for llm-fingerprint."""

import argparse
import asyncio
import sys
from argparse import Namespace
from pathlib import Path

from llm_fingerprint.generate import SamplesGenerator


def cmd_generate(args: Namespace):
    """Generate samples and save them to args.samples_path."""
    for model in args.language_model:
        generator = SamplesGenerator(
            language_model=model,
            prompts_path=args.prompts_path,
            samples_path=args.samples_path,
            samples_num=args.samples_num,
            max_tokens=args.max_tokens,
        )
        asyncio.run(generator.main())


def cmd_upload(args: Namespace):
    """Upload samples to ChromaDB."""
    ...


def cmd_query(args: Namespace):
    """Query ChromaDB for model identification."""
    ...


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Fingerprint - Identify LLMs by their response fingerprints"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        name="generate",
        help="Generate multiple LLM samples",
    )
    generate_parser.add_argument(
        "--language-model",
        type=str,
        nargs="+",
        required=True,
        help="Model(s) to use for the LLM",
    )
    generate_parser.add_argument(
        "--prompts-path",
        type=Path,
        required=True,
        help="Path to prompts JSONL file",
    )
    generate_parser.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="Path to save generated samples",
    )
    generate_parser.add_argument(
        "--samples-num",
        type=int,
        default=5,
        help="Number of samples to generate per prompt",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    generate_parser.set_defaults(func=cmd_generate)

    # Upload command
    upload_parser = subparsers.add_parser(
        name="upload",
        help="Upload samples to ChromaDB",
    )
    upload_parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Embedding model for computing embeddings",
    )
    upload_parser.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="Path to samples JSONL file",
    )
    upload_parser.set_defaults(func=cmd_upload)

    # Query command
    query_parser = subparsers.add_parser(
        name="query",
        help="Query ChromaDB for model identification",
    )
    query_parser.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="Path to samples JSONL file",
    )
    query_parser.add_argument(
        "--matches-path",
        type=Path,
        required=True,
        help="Path to save matched models",
    )
    query_parser.add_argument(
        "--matches-num",
        type=int,
        default=5,
        help="Number of model matches to return",
    )
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()

    match args.command:
        case "generate":
            cmd_generate(args)

        case "upload":
            cmd_upload(args)

        case "query":
            cmd_query(args)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
