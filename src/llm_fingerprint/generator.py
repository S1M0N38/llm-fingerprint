"""Module for generating multiple responses from LLMs."""

import asyncio
import os
import uuid
from pathlib import Path

import aiofiles
import httpx
from openai import AsyncOpenAI
from tqdm import tqdm

from llm_fingerprint.models import Prompt, Sample

assert (LLM_API_KEY := os.getenv("LLM_API_KEY", ""))
assert (LLM_BASE_URL := os.getenv("LLM_BASE_URL", ""))


class SamplesGenerator:
    def __init__(
        self,
        language_model: str,
        prompts_path: Path,
        samples_path: Path,
        samples_num: int,
        max_tokens: int = 2048,
        concurrent_requests: int = 32,
    ):
        self.language_model: str = language_model
        self.prompts_path: Path = prompts_path
        self.samples_path: Path = samples_path
        self.samples_num: int = samples_num
        self.max_tokens: int = max_tokens
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.client = AsyncOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=httpx.Timeout(timeout=900, connect=5.0),
        )

    async def generate_sample(self, prompt: Prompt) -> Sample:
        response = await self.client.chat.completions.create(
            model=self.language_model,
            messages=[{"role": "user", "content": prompt.prompt}],
            max_tokens=self.max_tokens,
        )
        completion = response.choices[0].message.content
        assert completion
        return Sample(
            id=response.id,
            model=self.language_model,
            prompt_id=prompt.id,
            completion=completion,
        )

    async def save_sample(self, sample: Sample) -> None:
        async with aiofiles.open(self.samples_path, "a") as f:
            await f.write(sample.model_dump_json() + "\n")

    async def load_prompts(self) -> list[Prompt]:
        with open(self.prompts_path) as f:
            prompts = [Prompt.model_validate_json(line) for line in f]
        for prompt in prompts:
            # check for prompt integrity
            assert str(uuid.uuid5(uuid.NAMESPACE_DNS, prompt.prompt)) == prompt.id
        return prompts

    async def main(self) -> None:
        prompts = await self.load_prompts()

        async def generate_and_save_sample(prompt: Prompt) -> None:
            async with self.semaphore:
                sample = await self.generate_sample(prompt)
                await self.save_sample(sample)

        tasks = [
            generate_and_save_sample(prompt)
            for prompt in prompts
            for _ in range(self.samples_num)
        ]

        # NOTE: we are manually create multiple samples instead of using `n`
        # because llama-server does not support `n` in completions.create

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
            await self.client.close()
