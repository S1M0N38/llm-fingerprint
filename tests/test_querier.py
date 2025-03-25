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
