import pytest
from qdrant_client import models

from llm_fingerprint.io import FileIO
from llm_fingerprint.models import Sample
from llm_fingerprint.services import UploaderService
from llm_fingerprint.storage.implementation.chroma import ChromaStorage
from llm_fingerprint.storage.implementation.qdrant import QdrantStorage


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
async def test_uploader_service_chroma(
    file_io_test: FileIO,
    chroma_storage: ChromaStorage,
    samples_test: list[Sample],
):
    """Test that the UploaderService can upload samples to ChromaDB."""

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


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
async def test_uploader_service_qdrant(
    collection_name,
    file_io_test: FileIO,
    qdrant_storage: QdrantStorage,
    samples_test: list[Sample],
):
    """Test that the UploaderService can upload samples to Qdrant."""

    await file_io_test.save_samples(samples_test)

    uploader = UploaderService(file_io=file_io_test, storage=qdrant_storage)

    await uploader.main()

    samples, _ = await qdrant_storage.client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="centroid", match=models.MatchValue(value=False)
                )
            ]
        ),
        limit=1024,
    )

    assert len(samples) == len(samples_test)
    assert {sample.id for sample in samples} == {sample.id for sample in samples_test}

    centroids, _ = await qdrant_storage.client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="centroid", match=models.MatchValue(value=True)
                )
            ]
        ),
        limit=1024,
    )

    model_prompt = {(s.model, s.prompt_id) for s in samples_test}
    ids = {centroid.id for centroid in centroids}
    assert len(ids) == len(model_prompt)


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
async def test_uploader_service_chroma_with_embeddings(
    chroma_storage: ChromaStorage,
    samples_test: list[Sample],
    embeddings_test: list[list[float]],
):
    """Test that samples can be uploaded to ChromaDB with pre-computed embeddings."""
    # Upload samples with pre-computed embeddings
    await chroma_storage.upload_samples(samples_test, embeddings=embeddings_test)
    await chroma_storage.upsert_centroids()

    # Verify upload was successful
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


@pytest.mark.db
@pytest.mark.emb
@pytest.mark.asyncio
async def test_uploader_service_qdrant_with_embeddings(
    collection_name,
    qdrant_storage: QdrantStorage,
    samples_test: list[Sample],
    embeddings_test: list[list[float]],
):
    """Test that samples can be uploaded to Qdrant with pre-computed embeddings."""
    # Upload samples with pre-computed embeddings
    await qdrant_storage.upload_samples(samples_test, embeddings=embeddings_test)
    await qdrant_storage.upsert_centroids()

    # Verify upload was successful
    samples, _ = await qdrant_storage.client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="centroid", match=models.MatchValue(value=False)
                )
            ]
        ),
        limit=1024,
    )

    assert len(samples) == len(samples_test)
    assert {sample.id for sample in samples} == {sample.id for sample in samples_test}

    centroids, _ = await qdrant_storage.client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="centroid", match=models.MatchValue(value=True)
                )
            ]
        ),
        limit=1024,
    )

    model_prompt = {(s.model, s.prompt_id) for s in samples_test}
    ids = {centroid.id for centroid in centroids}
    assert len(ids) == len(model_prompt)
