"""Tests for the SamplesGenerator class."""

import json
import os
import uuid
from pathlib import Path

import pytest
from openai import NotFoundError

from llm_fingerprint.generate import SamplesGenerator
from llm_fingerprint.models import Prompt, Sample

################################################################################
# Unit tests for the SamplesGenerator class
################################################################################


@pytest.fixture
def test_prompt_texts():
    """Return 3 simple prompts with predictable answers for testing."""
    return [
        "What's the capital of France? Just responde with one word",
        "How many hours are in a day?",
        "What color is the sky on a clear day? Just responde with one word",
    ]


@pytest.fixture
def test_completion_texts():
    """Return the responses for the test prompts."""
    return [
        "paris",
        "24",
        "blue",
    ]


@pytest.fixture
def test_prompts(test_prompt_texts):
    """Create prompt objects with the same structure as the real ones."""
    return [
        Prompt(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, text)), prompt=text)
        for text in test_prompt_texts
    ]


@pytest.fixture
def temp_prompts_file(test_prompts, tmp_path):
    """Create a temporary prompts file with test prompts."""
    temp_file = tmp_path / "test_prompts.jsonl"
    with open(temp_file, "w") as f:
        for prompt in test_prompts:
            f.write(prompt.model_dump_json() + "\n")
    return temp_file


@pytest.fixture
def temp_samples_file(tmp_path):
    """Create a path for a temporary samples file."""
    return tmp_path / "test_samples.jsonl"


def test_samples_generator_init():
    """Test the initialization of SamplesGenerator with different parameters."""
    generator = SamplesGenerator(
        language_model="test-model",
        prompts_path=Path("dummy/prompts.jsonl"),
        samples_path=Path("dummy/samples.jsonl"),
        samples_num=3,
        max_tokens=100,
        concurrent_requests=5,
    )

    assert generator.language_model == "test-model"
    assert generator.prompts_path == Path("dummy/prompts.jsonl")
    assert generator.samples_path == Path("dummy/samples.jsonl")
    assert generator.samples_num == 3
    assert generator.max_tokens == 100
    assert generator.semaphore._value == 5


@pytest.mark.asyncio
async def test_load_prompts(temp_prompts_file, test_prompts):
    """Test loading prompts from a file."""
    generator = SamplesGenerator(
        language_model="test-model",
        prompts_path=temp_prompts_file,
        samples_path=Path("dummy/samples.jsonl"),
        samples_num=1,
    )

    prompts = await generator.load_prompts()

    assert len(prompts) == 3
    assert all(isinstance(prompt, Prompt) for prompt in prompts)
    prompt_ids = {prompt.id for prompt in prompts}
    expected_ids = {prompt.id for prompt in test_prompts}
    assert prompt_ids == expected_ids


@pytest.mark.llm
@pytest.mark.asyncio
async def test_generate_sample(test_prompts, test_completion_texts):
    """Test generating a sample from a prompt."""
    generator = SamplesGenerator(
        language_model="test-model",
        prompts_path=Path("dummy/path.jsonl"),
        samples_path=Path("dummy/samples.jsonl"),
        samples_num=1,
        max_tokens=100,
    )

    # Test with the first prompt ("What's the capital of France?")
    sample = await generator.generate_sample(test_prompts[0])

    assert sample.id is not None
    assert sample.model == "test-model"
    assert sample.prompt_id == test_prompts[0].id
    assert sample.completion is not None
    assert len(sample.completion) > 0

    completion_lower = sample.completion.lower()
    assert test_completion_texts[0] in completion_lower, (
        f"Expected '{test_completion_texts[0]}' in: {sample.completion}"
    )


@pytest.mark.asyncio
async def test_save_sample(temp_samples_file):
    """Test saving a sample to a file."""
    generator = SamplesGenerator(
        language_model="test-model",
        prompts_path=Path("dummy/path.jsonl"),
        samples_path=temp_samples_file,
        samples_num=1,
    )

    # Create a sample
    sample = Sample(
        id="test-id",
        model="test-model",
        prompt_id="test-prompt-id",
        completion="This is a test completion.",
    )

    await generator.save_sample(sample)

    # Check that the sample was saved correctly
    with open(temp_samples_file) as f:
        content = f.read()
        assert "test-id" in content
        assert "test-model" in content
        assert "test-prompt-id" in content
        assert "This is a test completion." in content


@pytest.mark.llm
@pytest.mark.asyncio
async def test_main(
    temp_prompts_file, temp_samples_file, test_prompts, test_completion_texts
):
    """Test the main function with real API calls."""
    generator = SamplesGenerator(
        language_model="test-model",
        prompts_path=temp_prompts_file,
        samples_path=temp_samples_file,
        samples_num=1,
        max_tokens=100,
    )
    await generator.main()

    with open(temp_samples_file) as f:
        samples = [json.loads(line) for line in f.readlines()]

    assert len(samples) == 3
    assert all(sample["model"] == "test-model" for sample in samples)
    assert {sample["prompt_id"] for sample in samples} == {
        prompt.id for prompt in test_prompts
    }

    for prompt, expected_text in zip(test_prompts, test_completion_texts):
        sample = next(s for s in samples if s["prompt_id"] == prompt.id)
        assert expected_text in sample["completion"].lower(), (
            f"Expected '{expected_text}' in response to '{prompt.prompt}'"
        )


################################################################################
# Integration tests for the SamplesGenerator class
################################################################################


@pytest.fixture
def integration_prompts():
    """Create a small set of diverse prompts for integration testing."""
    prompt_texts = [
        "Write a haiku about programming.",
        "Explain quantum computing in one sentence.",
        "List three advantages of using Python for data analysis.",
    ]
    return [
        Prompt(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, text)), prompt=text)
        for text in prompt_texts
    ]


@pytest.fixture
def integration_prompts_file(integration_prompts, tmp_path):
    """Create a temporary prompts file with integration test prompts."""
    temp_file = tmp_path / "integration_prompts.jsonl"
    with open(temp_file, "w") as f:
        for prompt in integration_prompts:
            f.write(prompt.model_dump_json() + "\n")
    return temp_file


@pytest.mark.llm
@pytest.mark.asyncio
async def test_integration_sample_generation(integration_prompts_file, tmp_path):
    """Test the complete sample generation pipeline with real API calls."""

    assert os.getenv("LLM_API_KEY"), "LLM_API_KEY environment variable is required"
    assert os.getenv("LLM_BASE_URL"), "LLM_BASE_URL environment variable is required"

    test_model = "test-model"
    samples_num = 2
    max_tokens = 256
    samples_path = tmp_path / "integration_samples.jsonl"

    generator = SamplesGenerator(
        language_model=test_model,
        prompts_path=integration_prompts_file,
        samples_path=samples_path,
        samples_num=samples_num,
        max_tokens=max_tokens,
        concurrent_requests=2,
    )

    await generator.main()
    assert samples_path.exists(), "Samples file was not created"

    with open(samples_path) as f:
        samples = [json.loads(line) for line in f]

    expected_sample_count = 3 * samples_num
    assert len(samples) == expected_sample_count, (
        f"Expected {expected_sample_count} samples, got {len(samples)}"
    )

    for sample in samples:
        # Check required fields
        assert "id" in sample, "Sample missing 'id' field"
        assert "model" in sample, "Sample missing 'model' field"
        assert "prompt_id" in sample, "Sample missing 'prompt_id' field"
        assert "completion" in sample, "Sample missing 'completion' field"

        # Validate field values
        assert sample["model"] == test_model, (
            f"Expected model '{test_model}', got '{sample['model']}'"
        )
        assert sample["completion"], "Sample completion is empty"

    with open(integration_prompts_file) as f:
        prompts = [json.loads(line) for line in f]

    prompt_ids = {prompt["id"] for prompt in prompts}
    sample_prompt_ids = {sample["prompt_id"] for sample in samples}
    assert prompt_ids == sample_prompt_ids, "Not all prompts were used"

    prompt_id_counts = {}
    for sample in samples:
        prompt_id = sample["prompt_id"]
        prompt_id_counts[prompt_id] = prompt_id_counts.get(prompt_id, 0) + 1

    for prompt_id, count in prompt_id_counts.items():
        assert count == samples_num, (
            f"Expected {samples_num} samples for prompt {prompt_id}, got {count}"
        )


@pytest.mark.llm
@pytest.mark.asyncio
async def test_integration_error_handling(integration_prompts_file, tmp_path):
    """Test error handling in real API scenarios."""
    samples_path = tmp_path / "error_samples.jsonl"

    generator = SamplesGenerator(
        language_model="invalid-model-name-that-doesnt-exist",
        prompts_path=integration_prompts_file,
        samples_path=samples_path,
        samples_num=1,
        max_tokens=100,
        concurrent_requests=1,
    )

    with pytest.raises(NotFoundError) as excinfo:
        await generator.main()

    assert "invalid-model-name-that-doesnt-exist" in str(excinfo.value)

    await generator.client.close()
