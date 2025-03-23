"""Command-line interface for llm-fingerprint."""

import argparse
import asyncio
import sys
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

from llm_fingerprint.generator import SamplesGenerator
from llm_fingerprint.querier import SamplesQuerier
from llm_fingerprint.uploader import SamplesUploader


def cmd_generate(args: Namespace):
    """Generate samples and save them to args.samples_path."""
    args.samples_path.parent.mkdir(parents=True, exist_ok=True)
    for model in tqdm(args.language_model, desc="Generate samples", unit="model"):
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
    uploader = SamplesUploader(
        embedding_model=args.embedding_model,
        samples_path=args.samples_path,
        collection_name=args.collection_name,
    )
    asyncio.run(uploader.main())


def cmd_query(args: Namespace):
    """Query ChromaDB for model identification."""
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    querier = SamplesQuerier(
        embedding_model=args.embedding_model,
        samples_path=args.samples_path,
        retults_path=args.results_path,
        results_num=args.results_num,
        collection_name=args.collection_name,
    )
    asyncio.run(querier.main())


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
    generate_parser.add_argument(
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
    generate_parser.add_argument(
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
