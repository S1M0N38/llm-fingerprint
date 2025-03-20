"""Tests for the SamplesGenerator class."""

import json
import uuid
from pathlib import Path

import pytest

from llm_fingerprint.generate import SamplesGenerator
from llm_fingerprint.models import Prompt, Sample


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
