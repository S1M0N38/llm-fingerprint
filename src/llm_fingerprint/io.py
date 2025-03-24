import uuid
from pathlib import Path

import aiofiles

from llm_fingerprint.models import Prompt, Result, Sample


class FileIO:
    """Handles all file I/O operations for the LLM Fingerprint application.

    This class centralizes file reading and writing operations, removing this
    responsibility from service classes.
    """

    def __init__(
        self,
        prompts_path: Path | None = None,
        samples_path: Path | None = None,
        results_path: Path | None = None,
    ):
        """Initialize FileIO with optional file paths.

        Args:
            prompts_path: Path to the prompts file
            samples_path: Path to the samples file
            results_path: Path to the results file
        """
        self.prompts_path = prompts_path
        self.samples_path = samples_path
        self.results_path = results_path

    async def load_prompts(self) -> list[Prompt]:
        """Load prompts from the prompts_path.

        Returns:
            List of Prompt objects

        Raises:
            ValueError: If prompts_path is not set
            FileNotFoundError: If the prompts file doesn't exist
        """
        if not self.prompts_path:
            raise ValueError("prompts_path not set")

        async with aiofiles.open(self.prompts_path, "r") as f:
            prompts: list[Prompt] = []
            async for line in f:
                prompt = Prompt.model_validate_json(line)
                assert str(uuid.uuid5(uuid.NAMESPACE_DNS, prompt.prompt)) == prompt.id
                prompts.append(prompt)
        return prompts

    async def save_prompts(self, prompts: list[str]) -> None:
        """Save prompts to the prompts_path (write).

        Args:
            prompts: List of prompt strings to save

        Raises:
            ValueError: If prompts_path is not set
        """
        if not self.prompts_path:
            raise ValueError("prompts_path not set")

        async with aiofiles.open(self.prompts_path, "w") as f:
            for prompt_str in prompts:
                prompt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, prompt_str))
                prompt = Prompt(id=prompt_id, prompt=prompt_str)
                await f.write(prompt.model_dump_json() + "\n")

    async def load_samples(self) -> list[Sample]:
        """Load samples from the samples_path.

        Returns:
            List of Sample objects

        Raises:
            ValueError: If samples_path is not set
            FileNotFoundError: If the samples file doesn't exist
        """
        if not self.samples_path:
            raise ValueError("samples_path not set")

        async with aiofiles.open(self.samples_path, "r") as f:
            samples = [Sample.model_validate_json(line) async for line in f]
        return samples

    async def save_sample(self, sample: Sample) -> None:
        """Save a single sample to the samples_path (append).

        Args:
            sample: Sample object to save

        Raises:
            ValueError: If samples_path is not set
        """
        if not self.samples_path:
            raise ValueError("samples_path not set")

        async with aiofiles.open(self.samples_path, "a") as f:
            await f.write(sample.model_dump_json() + "\n")

    async def save_samples(self, samples: list[Sample]) -> None:
        """Save multiple samples to the samples_path (append).

        Args:
            samples: List of Sample objects to save

        Raises:
            ValueError: If samples_path is not set
        """
        if not self.samples_path:
            raise ValueError("samples_path not set")

        async with aiofiles.open(self.samples_path, "a") as f:
            for sample in samples:
                await f.write(sample.model_dump_json() + "\n")

    async def save_results(self, results: list[Result]) -> None:
        """Save results to the results_path (write).

        Args:
            results: List of Result objects to save

        Raises:
            ValueError: If results_path is not set
        """
        if not self.results_path:
            raise ValueError("results_path not set")

        async with aiofiles.open(self.results_path, "w") as f:
            for result in results:
                await f.write(result.model_dump_json() + "\n")

    async def load_results(self) -> list[Result]:
        """Load results from the results_path.

        Returns:
            List of Result objects

        Raises:
            ValueError: If results_path is not set
            FileNotFoundError: If the results file doesn't exist
        """
        if not self.results_path:
            raise ValueError("results_path not set")

        async with aiofiles.open(self.results_path, "r") as f:
            results = [Result.model_validate_json(line) async for line in f]
        return results
