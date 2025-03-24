import pytest

from llm_fingerprint.io import FileIO
from llm_fingerprint.models import Sample
from llm_fingerprint.services import QuerierService
from llm_fingerprint.storage.implementation.chroma import ChromaStorage
from tests.utils import filter_samples


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language_model,prompts_num,samples_num,results_num",
    [
        (language_model, prompts_num, samples_num, results_num)
        for results_num in (1, 2)
        for samples_num in (1, 2, 3)
        for prompts_num in (1, 2, 3)
        for language_model in ("test-model-1", "test-model-2")
    ],
)
async def test_querier_service(
    file_io_test: FileIO,
    populated_chroma_storage: ChromaStorage,
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
        storage=populated_chroma_storage,
        results_num=results_num,
    )

    await querier.main()

    results = await file_io_test.load_results()
    assert len(results) == results_num
    assert results[0].model == language_model
