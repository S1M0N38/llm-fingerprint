import asyncio

from tqdm import tqdm

from llm_fingerprint.io import FileIO
from llm_fingerprint.mixin import CompletionsMixin
from llm_fingerprint.models import Prompt
from llm_fingerprint.storage.base import VectorStorage


class GeneratorService(CompletionsMixin):
    def __init__(
        self,
        file_io: FileIO,
        samples_num: int,
        language_model: str,
        max_tokens: int = 2048,
        concurrent_requests: int = 32,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        CompletionsMixin.__init__(self, language_model, max_tokens, base_url, api_key)
        self.samples_num = samples_num
        self.file_io = file_io
        self.semaphore = asyncio.Semaphore(concurrent_requests)

    async def main(self):
        prompts = await self.file_io.load_prompts()

        async def generate_and_save_sample(prompt: Prompt) -> None:
            async with self.semaphore:
                sample = await self.generate_sample(prompt)
                await self.file_io.save_sample(sample)

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


class UploaderService:
    def __init__(self, file_io: FileIO, storage: VectorStorage):
        self.storage = storage
        self.file_io = file_io

    async def main(self):
        samples = await self.file_io.load_samples()
        await self.storage.upload_samples(samples)
        await self.storage.upsert_centroids()


class QuerierService:
    def __init__(self, file_io: FileIO, storage: VectorStorage, results_num: int = 5):
        self.file_io = file_io
        self.storage = storage
        self.results_num = results_num

    async def main(self):
        samples = await self.file_io.load_samples()
        results = await self.storage.query_samples(samples, self.results_num)
        await self.file_io.save_results(results)
