"""Tests for the command-line interface."""

from argparse import Namespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_fingerprint.cli import cmd_generate, cmd_query, cmd_upload, main


@pytest.fixture
def mock_class_with_async_main():
    """Create a mock class with AsyncMock main method."""
    mock_main = Mock()
    mock_instance = MagicMock()
    mock_instance.main = mock_main
    mock_class = MagicMock(return_value=mock_instance)
    return mock_class, mock_main


@pytest.mark.cli
@pytest.mark.asyncio
async def test_cmd_generate(mock_class_with_async_main, tmp_path):
    """Test the generate command."""
    language_models = ["test-model-1", "test-model-2"]
    mock_generator_class, mock_main = mock_class_with_async_main

    args = Namespace(
        language_model=language_models,
        prompts_path=tmp_path / "prompts.jsonl",
        samples_path=tmp_path / "samples.jsonl",
        samples_num=3,
        max_tokens=100,
    )

    with patch("llm_fingerprint.cli.SamplesGenerator", mock_generator_class):
        with patch("asyncio.run") as mock_run:
            cmd_generate(args)
            assert mock_generator_class.call_count == len(language_models)
            assert mock_main.call_count == len(language_models)
            assert mock_run.call_count == len(language_models)


@pytest.mark.cli
@pytest.mark.asyncio
async def test_cmd_upload(mock_class_with_async_main, tmp_path):
    """Test the upload command."""
    mock_uploader_class, mock_main = mock_class_with_async_main

    args = Namespace(
        samples_path=tmp_path / "samples.jsonl",
        collection_name="test_collection",
    )

    with patch("llm_fingerprint.cli.SamplesUploader", mock_uploader_class):
        with patch("asyncio.run") as mock_run:
            cmd_upload(args)
            mock_uploader_class.assert_called_once()
            mock_main.assert_called_once()
            mock_run.assert_called_once()


@pytest.mark.cli
@pytest.mark.asyncio
async def test_cmd_query(mock_class_with_async_main, tmp_path):
    """Test the query command."""
    mock_querier_class, mock_main = mock_class_with_async_main

    args = Namespace(
        samples_path=tmp_path / "samples.jsonl",
        results_path=tmp_path / "results.jsonl",
        results_num=5,
        collection_name="test_collection",
    )

    with patch("llm_fingerprint.cli.SamplesQuerier", mock_querier_class):
        with patch("asyncio.run") as mock_run:
            cmd_query(args)
            mock_querier_class.assert_called_once()
            mock_main.assert_called_once()
            mock_run.assert_called_once()


@pytest.mark.cli
@pytest.mark.parametrize(
    "args,patched_function",
    [
        (
            [
                "llm-fingerprint",
                "generate",
                "--language-model",
                "test-model",
                "--prompts-path",
                "test.jsonl",
                "--samples-path",
                "samples.jsonl",
            ],
            "llm_fingerprint.cli.cmd_generate",
        ),
        (
            [
                "llm-fingerprint",
                "upload",
                "--samples-path",
                "test.jsonl",
            ],
            "llm_fingerprint.cli.cmd_upload",
        ),
        (
            [
                "llm-fingerprint",
                "query",
                "--samples-path",
                "test.jsonl",
                "--results-path",
                "results.jsonl",
            ],
            "llm_fingerprint.cli.cmd_query",
        ),
    ],
)
def test_main(args, patched_function):
    """Test the main CLI entry point with different commands."""
    with patch("sys.argv", args):
        with patch(patched_function) as mock:
            main()
            mock.assert_called_once()
