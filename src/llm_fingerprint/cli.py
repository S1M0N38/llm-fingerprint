"""Command-line interface for llm-fingerprint."""

import argparse
import asyncio
import sys
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

from llm_fingerprint.services import GeneratorService, QuerierService, UploaderService
from llm_fingerprint.storage.implementation.chroma import ChromaStorage


async def cmd_generate(args: Namespace):
    args.samples_path.parent.mkdir(parents=True, exist_ok=True)
    for model in tqdm(args.language_model, desc="Generate samples", unit="model"):
        generator = GeneratorService(
            prompts_path=args.prompts_path,
            samples_path=args.samples_path,
            samples_num=args.samples_num,
            language_model=model,
            max_tokens=args.max_tokens,
        )
        await generator.main()


async def cmd_upload(args: Namespace, storage_factory=ChromaStorage):
    # Create a VectorStorage and initialize it
    storage = storage_factory(args.embedding_model)
    await storage.initialize(args.collection_name)

    # Upload samples
    uploader = UploaderService(args.samples_path, storage)
    await uploader.main()


async def cmd_query(args: Namespace, storage_factory=ChromaStorage):
    # Create a VectorStorage and initialize it
    storage = storage_factory(args.embedding_model)
    await storage.initialize(args.collection_name)

    # Upload samples
    querier = QuerierService(args.samples_path, args.results_path, storage)
    await querier.main()


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

    args = parser.parse_args()

    match args.command:
        case "generate":
            asyncio.run(cmd_generate(args))

        case "upload":
            asyncio.run(cmd_upload(args))

        case "query":
            asyncio.run(cmd_query(args))

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
