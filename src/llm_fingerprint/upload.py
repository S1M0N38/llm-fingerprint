import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

from chromadb import AsyncHttpClient
from chromadb.utils import embedding_functions

from llm_fingerprint.models import Sample

assert (CHROMADB_MODEL := os.getenv("CHROMADB_MODEL", ""))
assert (CHROMADB_URL := os.getenv("CHROMADB_URL", ""))

chromadb_host = urlparse(CHROMADB_URL).hostname
chromadb_port = urlparse(CHROMADB_URL).port
assert isinstance(chromadb_host, str)
assert isinstance(chromadb_port, int)
CHROMADB_HOST = chromadb_host
CHROMADB_PORT = chromadb_port


class SamplesUploader:
    """Class for uploading samples and theri embeddings to ChromaDB."""

    def __init__(self, samples_path: Path, collection_name: str = "samples"):
        self.samples_path: Path = samples_path
        self.collection_name: str = collection_name
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore
                model_name=CHROMADB_MODEL,
                trust_remote_code=True,
                # NOTE: trust_remote_code=True is needed for some models.
                # Use at you own risk. After downloading the model, you can pin it
                # to a specific version.
            )
        )
        asyncio.run(self._init_client())

    async def _init_client(self) -> None:
        self.client = await AsyncHttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
        )
        assert await self.client.heartbeat(), "0 BPM for chromadb"
        self.collection = await self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,  #  type: ignore
        )

    async def upload_samples(self, samples: list[Sample]) -> None:
        prev_ids = await self.collection.get(
            ids=[sample.id for sample in samples], include=[]
        )
        samples = [s for s in samples if s.id not in prev_ids]
        await self.collection.add(
            ids=[sample.id for sample in samples],
            metadatas=[
                sample.model_dump(
                    include={"model", "prompt_id"},
                )
                for sample in samples
            ],
            documents=[sample.completion for sample in samples],
        )

    async def load_samples(self) -> list[Sample]:
        with open(self.samples_path) as f:
            samples = [Sample.model_validate_json(line) for line in f]
        return samples

    async def main(self):
        initial_count = await self.collection.count()
        print(f"Collection {self.collection_name} has {initial_count} samples")

        samples = await self.load_samples()
        await self.upload_samples(samples)

        final_count = await self.collection.count()
        diff_count = final_count - initial_count
        print(
            f"Collection {self.collection_name} has {diff_count} samples more "
            f"(total {final_count})"
        )
