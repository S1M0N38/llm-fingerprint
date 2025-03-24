"""Pytest configuration for LLM Fingerprint tests."""

import uuid

import pytest

from llm_fingerprint.io import FileIO
from llm_fingerprint.models import Prompt, Sample


@pytest.fixture
def file_io_test(tmp_path):
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
def samples_test(prompts_test) -> list[Sample]:
    completions = [
        "paris",
        "24",
        "blue",
    ]
    return [
        Sample(id="", model="test-model", prompt_id=prompt.id, completion=completion)
        for prompt, completion in zip(prompts_test, completions)
    ]
