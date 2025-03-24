import pytest

from llm_fingerprint.io import FileIO
from llm_fingerprint.models import Sample
from llm_fingerprint.services import UploaderService
from llm_fingerprint.storage.implementation.chroma import ChromaStorage


@pytest.mark.asyncio
@pytest.mark.db
async def test_uploader_service(
    file_io_test: FileIO,
    chroma_storage: ChromaStorage,
    samples_test: list[Sample],
):
    """Test that the UploaderService can upload samples to ChromaDB.

    This test performs real API calls and requires the following environment variables:
    - CHROMADB_URL: URL of the ChromaDB server
    - EMB_API_KEY: API key for the embedding model
    - EMB_BASE_URL: Base URL for the embedding model API
    """
    await file_io_test.save_samples(samples_test)

    uploader = UploaderService(file_io=file_io_test, storage=chroma_storage)

    await uploader.main()

    results = await chroma_storage.collection.get(
        where={"centroid": False},
        include=[],
    )

    assert len(results["ids"]) == len(samples_test)
    assert set(results["ids"]) == {sample.id for sample in samples_test}

    centroids = await chroma_storage.collection.get(
        where={"centroid": True},
        include=[],
    )

    model_prompt = {(s.model, s.prompt_id) for s in samples_test}
    assert len(centroids["ids"]) == len(model_prompt)
