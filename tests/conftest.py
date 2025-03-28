"""Pytest configuration for LLM Fingerprint tests."""

import uuid
from pathlib import Path
from typing import AsyncGenerator

import pytest

from llm_fingerprint.io import FileIO
from llm_fingerprint.mixin import EmbeddingsMixin
from llm_fingerprint.models import Prompt, Sample
from llm_fingerprint.storage.implementation.chroma import ChromaStorage
from llm_fingerprint.storage.implementation.qdrant import QdrantStorage


@pytest.fixture
def file_io_test(tmp_path: Path) -> FileIO:
    """Create a FileIO instance with temporary paths."""
    prompts_path = tmp_path / "prompts.jsonl"
    samples_path = tmp_path / "samples.jsonl"
    results_path = tmp_path / "results.jsonl"
    return FileIO(prompts_path, samples_path, results_path)


@pytest.fixture
def prompts_test() -> list[Prompt]:
    """Create 3 simple prompts with predictable answers for testing."""
    prompts = [
        "What's the capital of France? Just responde with one word",
        "How many hours are in a day?",
        "What color is the sky on a clear day? Just responde with one word",
    ]
    return [
        Prompt(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, prompt)), prompt=prompt)
        for prompt in prompts
    ]


@pytest.fixture
def samples_test(prompts_test: list[Prompt]) -> list[Sample]:
    """Create test samples for two models with multiple completion variations
    per prompt.

    Generates 18 samples total:
    - 3 prompts × 2 models × 3 variations = 18 samples

    This fixture can be used with tests.utils.filter_samples to select a subset
    of samples according to the desired criteria.
    """
    # Define all completions by prompt index and model
    completions = {
        # Prompt 1: Capital of France
        0: {
            "test-model-1": [
                "Paris is the capital of France.",
                "The capital of France is Paris.",
                "France's capital city is Paris.",
            ],
            "test-model-2": [
                "Paris",
                "The answer is Paris, which is the capital city of France.",
                "Paris. It's a beautiful city in France and serves as the country's capital.",
            ],
        },
        # Prompt 2: Hours in a day
        1: {
            "test-model-1": [
                "There are 24 hours in a day.",
                "A day consists of 24 hours.",
                "One day has 24 hours.",
            ],
            "test-model-2": [
                "24 hours make up a day.",
                "A day is composed of 24 hours.",
                "The answer is 24 hours in a standard day.",
            ],
        },
        # Prompt 3: Color of the sky
        2: {
            "test-model-1": [
                "The sky is blue on a clear day.",
                "On a clear day, the sky appears blue.",
                "Blue is the color of the sky when it's clear.",
            ],
            "test-model-2": [
                "Blue",
                "The sky looks blue on clear days.",
                "When it's clear, the sky's color is blue.",
            ],
        },
    }

    samples = []
    for model in ["test-model-1", "test-model-2"]:
        for i, prompt in enumerate(prompts_test):
            for completion in completions[i][model]:
                sample = Sample(
                    id=str(uuid.uuid4()),
                    model=model,
                    prompt_id=prompt.id,
                    completion=completion,
                )
                samples.append(sample)

    return samples


@pytest.fixture
def samples_test_unk(samples_test: list[Sample]) -> list[Sample]:
    """The samples from unknown models are simply the sample from samples_test
    with the completion capitalized. In this way we have slightly modified text
    but with the same meaning.
    """
    samples = [
        sample.model_copy(update={"completion": sample.completion.upper()})
        for sample in samples_test
    ]
    return samples


@pytest.fixture
async def embedding_model() -> str:
    return "jinaai/jina-embeddings-v2-base-en"


@pytest.fixture
async def embedding_provider(
    embedding_model: str,
) -> AsyncGenerator[EmbeddingsMixin, None]:
    """Create an EmbeddingsMixin instance for computing embeddings."""
    provider = EmbeddingsMixin(embedding_model=embedding_model)
    try:
        yield provider
    finally:
        await provider.embedding_client.close()


@pytest.fixture
async def embeddings_test(
    embedding_provider: EmbeddingsMixin, samples_test: list[Sample]
) -> list[list[float]]:
    """Pre-computed embeddings for samples_test."""
    return await embedding_provider.embed_samples(samples_test)


@pytest.fixture
async def embeddings_test_unk(
    embedding_provider: EmbeddingsMixin, samples_test_unk: list[Sample]
) -> list[list[float]]:
    """Pre-computed embeddings for samples_test_unk."""
    return await embedding_provider.embed_samples(samples_test_unk)


@pytest.fixture
async def collection_name() -> str:
    return "test-collection"


@pytest.fixture
async def chroma_storage(
    embedding_model, collection_name
) -> AsyncGenerator[ChromaStorage, None]:
    """Create a ChromaStorage instance with a test collection.

    This fixture sets up a ChromaDB collection for testing and cleans it up
    after the test completes. Requires CHROMADB_URL, EMB_API_KEY and EMB_BASE_URL
    environment variables to be properly set.
    """
    # Setup
    storage = ChromaStorage(embedding_model=embedding_model)
    await storage.initialize(collection_name)
    print("Created ChromaDB collection")

    try:
        yield storage
    finally:
        if hasattr(storage, "client") and storage.client:
            # NOTE: deleting the elements instead of the collection is better
            # becaause there are some leftovers from dleted collections
            results = await storage.collection.get(include=[])
            await storage.collection.delete(ids=results["ids"])
            print("Deleted ChromaDB collection")


@pytest.fixture
async def qdrant_storage(
    embedding_model, collection_name
) -> AsyncGenerator[QdrantStorage, None]:
    """Create a QdrantStorage instance with a test collection."""
    # Setup
    storage = QdrantStorage(embedding_model=embedding_model)
    await storage.initialize(collection_name)
    print("Created Qdrant collection")

    try:
        yield storage
    finally:
        if hasattr(storage, "client") and storage.client:
            await storage.client.delete_collection(collection_name)
            await storage.client.close()
            print("Deleted Qdrant collection")
