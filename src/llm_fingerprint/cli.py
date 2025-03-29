"""Command-line interface for llm-fingerprint."""

import argparse
import asyncio
import sys
from pathlib import Path

from llm_fingerprint.commands import GenerateCommand, QueryCommand, UploadCommand


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
    generate_parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=32,
        help="Number of concurrent requests to make",
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        name="upload",
        help="Upload samples to ChromaDB",
    )
    upload_parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Model to use to compute embeddings",
    )
    upload_parser.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="Path to samples JSONL file",
    )
    upload_parser.add_argument(
        "--collection-name",
        type=str,
        default="samples",
        help="Name of the collection to upload samples to",
    )
    upload_parser.add_argument(
        "--storage",
        type=str,
        default="chroma",
        choices=["chroma", "qdrant"],
        help="Storage to upload samples to",
    )

    # Query command
    query_parser = subparsers.add_parser(
        name="query",
        help="Query ChromaDB for model identification",
    )
    query_parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Model to use to compute embeddings",
    )
    query_parser.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="Path to samples JSONL file",
    )
    query_parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to save returned results",
    )
    query_parser.add_argument(
        "--results-num",
        type=int,
        default=5,
        help="Number of results to return",
    )
    query_parser.add_argument(
        "--collection-name",
        type=str,
        default="samples",
        help="Name of the collection to query",
    )
    query_parser.add_argument(
        "--storage",
        type=str,
        default="chroma",
        choices=["chroma", "qdrant"],
        help="Storage to query for model identification",
    )

    args = parser.parse_args()

    # Command factory - maps command names to their implementations
    commands = {
        "generate": GenerateCommand,
        "upload": UploadCommand,
        "query": QueryCommand,
    }

    # Get the appropriate command class and execute it
    command_class = commands.get(args.command)
    if command_class:
        command = command_class(args)
        asyncio.run(command.execute())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
