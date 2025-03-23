import asyncio
import uuid
from pathlib import Path

import aiofiles
from tqdm import tqdm

from llm_fingerprint.mixin import CompletionsMixin
from llm_fingerprint.models import Prompt, Result, Sample
from llm_fingerprint.storage.base import VectorStorage


class BaseService:
    @staticmethod
    async def load_prompts(prompts_path) -> list[Prompt]:
        async with aiofiles.open(prompts_path, "r") as f:
            prompts: list[Prompt] = []
            async for line in f:
                prompt = Prompt.model_validate_json(line)
                assert str(uuid.uuid5(uuid.NAMESPACE_DNS, prompt.prompt)) == prompt.id
                prompts.append(prompt)
        return prompts

    @staticmethod
    async def save_prompts(prompts_path, prompts: list[str]) -> None:
        async with aiofiles.open(prompts_path, "w") as f:
            for prompt_str in prompts:
                prompt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, prompt_str))
                prompt = Prompt(id=prompt_id, prompt=prompt_str)
                await f.write(prompt.model_dump_json() + "\n")

    @staticmethod
    async def load_samples(samples_path) -> list[Sample]:
        async with aiofiles.open(samples_path, "r") as f:
            samples = [Sample.model_validate_json(line) async for line in f]
        return samples

    @staticmethod
    async def save_sample(samples_path, sample: Sample) -> None:
        async with aiofiles.open(samples_path, "a") as f:
            await f.write(sample.model_dump_json() + "\n")

    @staticmethod
    async def save_samples(samples_path, samples: list[Sample]) -> None:
        async with aiofiles.open(samples_path, "a") as f:
            for sample in samples:
                await f.write(sample.model_dump_json() + "\n")

    @staticmethod
    async def save_results(results_path, results: list[Result]) -> None:
        async with aiofiles.open(results_path, "a") as f:
            for result in results:
                await f.write(result.model_dump_json() + "\n")


class GeneratorService(BaseService, CompletionsMixin):
    def __init__(
        self,
        prompts_path: Path,
        samples_path: Path,
        samples_num: int,
        language_model: str,
        max_tokens: int = 2048,
        concurrent_requests: int = 32,
    ):
        CompletionsMixin.__init__(self, language_model, max_tokens)
        self.prompts_path = prompts_path
        self.samples_path = samples_path
        self.samples_num = samples_num
        self.semaphore = asyncio.Semaphore(concurrent_requests)

    async def main(self):
        prompts = await self.load_prompts(self.prompts_path)

        async def generate_and_save_sample(prompt: Prompt) -> None:
            async with self.semaphore:
                sample = await self.generate_sample(prompt)
                await self.save_sample(self.samples_path, sample)

        tasks = [
            generate_and_save_sample(prompt)
            for prompt in prompts
            for _ in range(self.samples_num)
        ]

        try:
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"{self.language_model}",
                unit="sample",
                smoothing=0,
            ):
                await future
        finally:
            await self.language_client.close()


class UploaderService(BaseService):
    def __init__(self, samples_path: Path, storage: VectorStorage):
        self.samples_path = samples_path
        self.storage = storage

    async def main(self):
        samples = await self.load_samples(self.samples_path)
        await self.storage.upload_samples(samples)
        await self.storage.upsert_centroids()


class QuerierService(BaseService):
    def __init__(self, prompts_path: Path, results_path: Path, storage: VectorStorage):
        self.prompts_path = prompts_path
        self.results_path = results_path
        self.storage = storage

    async def main(self):
        samples = await self.load_samples(self.prompts_path)
        results = await self.storage.query_samples(samples)
        await self.save_results(self.results_path, results)
