"""Tests for the SamplesUploader class."""

import os
import uuid
from urllib.parse import urlparse

import pytest
from chromadb import AsyncHttpClient

from llm_fingerprint.models import Sample
from llm_fingerprint.upload import SamplesUploader

# Check for required environment variables
CHROMADB_URL = os.getenv("CHROMADB_URL", "")
CHROMADB_MODEL = os.getenv("CHROMADB_MODEL", "")

NUM_SAMPLES = 4
NUM_MODELS = 2
NUM_PROMPTS = 3


# Fixtures
@pytest.fixture
def test_samples():
    """Create test samples with known content."""
    return [
        Sample(
            id=f"id-{prompt}-{model}-{sample}",
            model=f"test-model-{model}",
            prompt_id=f"test-prompt-{prompt}",
            completion=f"This is test completion {sample} for model {model} and prompt {prompt}.",
        )
        for sample in range(NUM_SAMPLES)
        for model in range(NUM_PROMPTS)
        for prompt in range(NUM_MODELS)
    ]


@pytest.fixture
def temp_samples_file(test_samples, tmp_path):
    """Create a temporary samples file with test samples."""
    temp_file = tmp_path / "test_samples.jsonl"
    with open(temp_file, "w") as f:
        for sample in test_samples:
            f.write(sample.model_dump_json() + "\n")
    return temp_file


@pytest.fixture
def test_collection_name():
    """Create a unique collection name for testing."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def test_client(test_collection_name):
    """Setup/Teardown for database client for testing."""
    chromadb_host = urlparse(CHROMADB_URL).hostname
    chromadb_port = urlparse(CHROMADB_URL).port
    assert isinstance(chromadb_host, str)
    assert isinstance(chromadb_port, int)

    client = await AsyncHttpClient(
        host=chromadb_host,
        port=chromadb_port,
    )

    yield client
    await client.delete_collection(test_collection_name)


@pytest.fixture
async def test_collection(test_client, test_collection_name):
    """Get the collection for testing."""
    collection = await test_client.get_or_create_collection(test_collection_name)
    yield collection


################################################################################
# Unit tests for the SamplesUploader class
################################################################################


@pytest.mark.db
@pytest.mark.asyncio
async def test_uploader_init(
    temp_samples_file,
    test_collection_name,
    test_client,
):
    """Test initialization of SamplesUploader."""
    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )
    await uploader.initialize()

    # Check attributes
    assert uploader.samples_path == temp_samples_file
    assert uploader.collection_name == test_collection_name

    # Check collection was created
    collections = await test_client.list_collections()
    assert test_collection_name in collections


@pytest.mark.db
@pytest.mark.asyncio
async def test_load_samples(temp_samples_file, test_collection_name, test_samples):
    """Test loading samples from file."""
    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )

    samples = await uploader.load_samples()

    assert len(samples) == len(test_samples)
    sample_ids = {sample.id for sample in samples}
    expected_ids = {sample.id for sample in test_samples}
    assert sample_ids == expected_ids


@pytest.mark.db
@pytest.mark.asyncio
async def test_upload_samples(temp_samples_file, test_collection_name, test_collection):
    """Test uploading samples to ChromaDB."""

    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )
    await uploader.initialize()

    samples = await uploader.load_samples()
    await uploader.upload_samples(samples)

    # Check samples were uploaded
    result = await test_collection.get(
        ids=[sample.id for sample in samples],
        include=["metadatas"],  # type: ignore
    )

    assert len(result["ids"]) == len(samples)
    for sample_id in [sample.id for sample in samples]:
        assert sample_id in result["ids"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_skip_existing_samples(
    temp_samples_file, test_collection_name, test_collection
):
    """Test that existing samples are skipped during upload."""

    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )
    await uploader.initialize()

    samples = await uploader.load_samples()

    # Upload half the samples first
    first_half = samples[: len(samples) // 2]
    await uploader.upload_samples(first_half)

    # Upload all samples - should skip the first half
    await uploader.upload_samples(samples)

    # Check count matches total samples
    count = await test_collection.count()
    assert count == len(samples)


@pytest.mark.db
@pytest.mark.asyncio
async def test_upsert_centroid(
    temp_samples_file, test_collection_name, test_collection
):
    """Test creating a centroid for a model-prompt combination."""

    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )
    await uploader.initialize()

    samples = await uploader.load_samples()

    # Create centroid for first model-prompt pair
    model = samples[0].model
    prompt_id = samples[0].prompt_id

    await uploader.upload_samples(samples)
    await uploader.upsert_centroid(model, prompt_id)

    # Check centroid exists
    centroid_id = f"centroid_{model}_{prompt_id}"
    result = await test_collection.get(ids=[centroid_id], include=["metadatas"])  # type: ignore

    assert len(result["ids"]) == 1
    assert result["metadatas"] is not None
    assert result["metadatas"][0]["centroid"] is True
    assert result["metadatas"][0]["model"] == model
    assert result["metadatas"][0]["prompt_id"] == prompt_id
    assert result["metadatas"][0]["sample_count"] == NUM_SAMPLES


@pytest.mark.db
@pytest.mark.asyncio
async def test_upsert_centroids(
    temp_samples_file, test_collection_name, test_collection
):
    """Test creating centroids for all model-prompt combinations."""

    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )
    await uploader.initialize()

    samples = await uploader.load_samples()
    await uploader.upload_samples(samples)
    await uploader.upsert_centroids()

    centroids = await test_collection.get(
        where={"centroid": True},
        include=["metadatas"],  # type: ignore
    )
    assert len(centroids["ids"]) == NUM_MODELS * NUM_PROMPTS

    for sample in samples:
        centroid_id = f"centroid_{sample.model}_{sample.prompt_id}"
        result = await test_collection.get(ids=[centroid_id], include=["metadatas"])  # type: ignore
        assert len(result["ids"]) == 1


@pytest.mark.db
@pytest.mark.asyncio
async def test_main_function(temp_samples_file, test_collection_name, test_collection):
    """Test the main function for the complete upload process."""

    uploader = SamplesUploader(
        samples_path=temp_samples_file, collection_name=test_collection_name
    )

    count = await test_collection.count()
    assert count == 0

    await uploader.main()

    samples = await test_collection.get(
        where={"centroid": False},
        include=["documents"],  # type: ignore
    )
    assert len(samples["ids"]) == NUM_SAMPLES * NUM_MODELS * NUM_PROMPTS

    centroids = await test_collection.get(
        where={"centroid": True},
        include=["metadatas"],  # type: ignore
    )
    assert len(centroids["ids"]) == NUM_MODELS * NUM_PROMPTS

    count = await test_collection.count()
    assert count == NUM_SAMPLES * NUM_MODELS * NUM_PROMPTS + NUM_MODELS * NUM_PROMPTS


################################################################################
# Integration tests for the SamplesUploader class
################################################################################


@pytest.mark.db
@pytest.mark.asyncio
async def test_integration_upload_workflow(
    tmp_path, test_collection_name, test_collection
):
    """Test the complete workflow from sample creation to upload."""

    NUM_SAMPLES = 4
    NUM_MODELS = 2
    NUM_PROMPTS = 3

    TOT_NUM_SAMPLES = NUM_SAMPLES * NUM_MODELS * NUM_PROMPTS
    NUM_CENTROIDS = NUM_MODELS * NUM_PROMPTS

    samples = [
        Sample(
            id=f"id-{prompt}-{model}-{sample}",
            model=f"test-model-{model}",
            prompt_id=f"test-prompt-{prompt}",
            completion=f"This is test completion {sample} for model {model} and prompt {prompt}.",
        )
        for sample in range(NUM_SAMPLES)
        for model in range(NUM_PROMPTS)
        for prompt in range(NUM_MODELS)
    ]

    # Create samples file
    samples_path = tmp_path / "integration_samples.jsonl"
    with open(samples_path, "w") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")

    # Run uploader
    uploader = SamplesUploader(
        samples_path=samples_path, collection_name=test_collection_name
    )

    await uploader.main()

    # Check results
    count = await test_collection.count()
    assert count == NUM_CENTROIDS + TOT_NUM_SAMPLES

    # Check samples
    samples_result = await uploader.collection.get(
        where={"centroid": False},
        include=["documents"],  # type: ignore
    )
    assert len(samples_result["ids"]) == TOT_NUM_SAMPLES

    # Check centroids
    centroids_result = await uploader.collection.get(
        where={"centroid": True},
        include=["metadatas"],  # type: ignore
    )
    assert len(centroids_result["ids"]) == NUM_CENTROIDS
