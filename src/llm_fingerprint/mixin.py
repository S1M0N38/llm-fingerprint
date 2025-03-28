import os
import uuid

import httpx
from openai import AsyncOpenAI

from llm_fingerprint.models import Prompt, Sample


class CompletionsMixin:
    """Mixin providing completion generation for prompts."""

    def __init__(
        self,
        language_model: str,
        max_tokens: int = 2048,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        api_key = api_key if api_key else os.getenv("LLM_API_KEY", "")
        base_url = base_url if base_url else os.getenv("LLM_BASE_URL")
        if base_url is None:
            raise ValueError("LLM_BASE_URL environment variable not set")

        self.language_model = language_model
        self.max_tokens = max_tokens
        self.language_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout=900, connect=5.0),
        )

    async def generate_sample(self, prompt: Prompt) -> Sample:
        response = await self.language_client.chat.completions.create(
            model=self.language_model,
            messages=[{"role": "user", "content": prompt.prompt}],
            max_tokens=self.max_tokens,
        )
        completion = response.choices[0].message.content
        assert completion
        return Sample(
            id=str(uuid.uuid4()),
            model=self.language_model,
            prompt_id=prompt.id,
            completion=completion,
        )


class EmbeddingsMixin:
    """Mixin providing embedding computation for samples."""

    def __init__(
        self,
        embedding_model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        api_key = api_key if api_key else os.getenv("EMB_API_KEY", "")
        base_url = base_url if base_url else os.getenv("EMB_BASE_URL")
        if base_url is None:
            raise ValueError("EMB_BASE_URL environment variable not set")

        self.embedding_model = embedding_model
        self.embedding_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def embed_samples(self, samples: list[Sample]) -> list[list[float]]:
        if not hasattr(self, "embedding_client"):
            raise AttributeError(
                "embedding_client is not initialized. "
                "Make sure EmbeddingsMixin.__init__ was called."
            )
        response = await self.embedding_client.embeddings.create(
            input=[sample.completion for sample in samples],
            model=self.embedding_model,
        )
        return [data.embedding for data in response.data]
