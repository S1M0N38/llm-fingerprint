from pathlib import Path
from typing import AsyncGenerator

import pytest

from llm_fingerprint.io import FileIO
from llm_fingerprint.models import Sample
from llm_fingerprint.services import QuerierService, UploaderService
from llm_fingerprint.storage.implementation.chroma import ChromaStorage
from llm_fingerprint.storage.implementation.qdrant import QdrantStorage
from tests.utils import filter_samples


@pytest.fixture
async def chroma_storage_populated(
    chroma_storage: ChromaStorage,
    samples_test: list[Sample],
    tmp_path: Path,
) -> AsyncGenerator[ChromaStorage, None]:
    """Create storage with pre-populated test data."""
    file_io_test = FileIO(samples_path=tmp_path / "samples-collections.jsonl")
    await file_io_test.save_samples(samples_test)
    uploader = UploaderService(file_io=file_io_test, storage=chroma_storage)
    await uploader.main()
    print("Populated Chroma collection")
    yield chroma_storage


@pytest.fixture
async def qdrant_storage_populated(
    qdrant_storage: QdrantStorage,
    samples_test: list[Sample],
    tmp_path: Path,
) -> AsyncGenerator[QdrantStorage, None]:
    """Create storage with pre-populated test data."""
    file_io_test = FileIO(samples_path=tmp_path / "samples-collections.jsonl")
    await file_io_test.save_samples(samples_test)
    uploader = UploaderService(file_io=file_io_test, storage=qdrant_storage)
    await uploader.main()
    print("Populated Qdrant collection")
    yield qdrant_storage


@pytest.mark.parametrize(
    "language_model,prompts_num,samples_num,results_num",
    (
        (language_model, prompts_num, samples_num, results_num)
        for results_num in (1, 2)
        for samples_num in (1, 2)
        for prompts_num in (1, 3)
        for language_model in ("test-model-1", "test-model-2")
    ),
)
async def test_querier_service_chroma(
    file_io_test: FileIO,
    chroma_storage_populated: ChromaStorage,
    samples_test_unk: list[Sample],
    language_model: str,
    prompts_num: int,
    samples_num: int,
    results_num: int,
):
    """Test that the QuerierService can query the vector storage for model
    identification."""

    samples_test_unk = filter_samples(
        samples=samples_test_unk,
        prompts_num=prompts_num,
        samples_num=samples_num,
        language_model=language_model,
    )

    await file_io_test.save_samples(samples_test_unk)

    querier = QuerierService(
        file_io=file_io_test,
        storage=chroma_storage_populated,
        results_num=results_num,
    )

    await querier.main()

    results = await file_io_test.load_results()
    assert len(results) == results_num
    assert results[0].model == language_model


@pytest.mark.parametrize(
    "language_model,prompts_num,samples_num,results_num",
    (
        (language_model, prompts_num, samples_num, results_num)
        for results_num in (1, 2)
        for samples_num in (1, 2)
        for prompts_num in (1, 3)
        for language_model in ("test-model-1", "test-model-2")
    ),
)
async def test_querier_service_qdrant(
    file_io_test: FileIO,
    qdrant_storage_populated: QdrantStorage,
    samples_test_unk: list[Sample],
    language_model: str,
    prompts_num: int,
    samples_num: int,
    results_num: int,
):
    """Test that the QuerierService can query the vector storage for model
    identification."""

    samples_test_unk = filter_samples(
        samples=samples_test_unk,
        prompts_num=prompts_num,
        samples_num=samples_num,
        language_model=language_model,
    )

    await file_io_test.save_samples(samples_test_unk)

    querier = QuerierService(
        file_io=file_io_test,
        storage=qdrant_storage_populated,
        results_num=results_num,
    )

    await querier.main()

    results = await file_io_test.load_results()
    assert len(results) == results_num
    assert results[0].model == language_model


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language_model,prompts_num,samples_num,results_num",
    [
        ("test-model-1", 1, 1, 1),
        ("test-model-2", 3, 2, 2),
    ],
)
async def test_querier_service_chroma_with_embeddings(
    file_io_test: FileIO,
    chroma_storage_populated: ChromaStorage,
    samples_test_unk: list[Sample],
    embeddings_test_unk: list[list[float]],
    language_model: str,
    prompts_num: int,
    samples_num: int,
    results_num: int,
):
    """Test querying ChromaDB with pre-computed embeddings."""
    # Filter samples and embeddings for this test
    filtered_samples = filter_samples(
        samples=samples_test_unk,
        prompts_num=prompts_num,
        samples_num=samples_num,
        language_model=language_model,
    )

    # Find indices of filtered samples in the original samples_test_unk
    sample_ids = {s.id for s in filtered_samples}
    filtered_embeddings = [
        emb
        for s, emb in zip(samples_test_unk, embeddings_test_unk)
        if s.id in sample_ids
    ]

    # Query with pre-computed embeddings
    results = await chroma_storage_populated.query_samples(
        filtered_samples, results_num=results_num, embeddings=filtered_embeddings
    )

    # Save and verify results
    await file_io_test.save_results(results)
    loaded_results = await file_io_test.load_results()

    assert len(loaded_results) == results_num
    assert loaded_results[0].model == language_model


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language_model,prompts_num,samples_num,results_num",
    [
        ("test-model-1", 1, 1, 1),
        ("test-model-2", 3, 2, 2),
    ],
)
async def test_querier_service_qdrant_with_embeddings(
    file_io_test: FileIO,
    qdrant_storage_populated: QdrantStorage,
    samples_test_unk: list[Sample],
    embeddings_test_unk: list[list[float]],
    language_model: str,
    prompts_num: int,
    samples_num: int,
    results_num: int,
):
    """Test querying Qdrant with pre-computed embeddings."""
    # Filter samples and embeddings for this test
    filtered_samples = filter_samples(
        samples=samples_test_unk,
        prompts_num=prompts_num,
        samples_num=samples_num,
        language_model=language_model,
    )

    # Find indices of filtered samples in the original samples_test_unk
    sample_ids = {s.id for s in filtered_samples}
    filtered_embeddings = [
        emb
        for s, emb in zip(samples_test_unk, embeddings_test_unk)
        if s.id in sample_ids
    ]

    # Query with pre-computed embeddings
    results = await qdrant_storage_populated.query_samples(
        filtered_samples, results_num=results_num, embeddings=filtered_embeddings
    )

    # Save and verify results
    await file_io_test.save_results(results)
    loaded_results = await file_io_test.load_results()

    assert len(loaded_results) == results_num
    assert loaded_results[0].model == language_model
