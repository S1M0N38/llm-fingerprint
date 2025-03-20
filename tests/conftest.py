"""Pytest configuration for LLM Fingerprint tests."""

import os

import pytest


# Skip LLM-dependent tests if environment variables are not set
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "llm: mark test as requiring LLM API access")


def pytest_collection_modifyitems(config, items):
    """Skip LLM tests if environment variables are not set."""
    if not os.getenv("LLM_API_KEY") or not os.getenv("LLM_BASE_URL"):
        skip_llm = pytest.mark.skip(reason="LLM API environment variables not set")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)
