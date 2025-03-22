import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from chromadb import AsyncHttpClient
from openai import AsyncOpenAI

from llm_fingerprint.models import Sample

assert (EMB_API_KEY := os.getenv("EMB_API_KEY", ""))
assert (EMB_BASE_URL := os.getenv("EMB_BASE_URL", ""))
assert (CHROMADB_URL := os.getenv("CHROMADB_URL", ""))

EMB_BATCH_SIZE = 32


class SamplesUploader:
    """Class for uploading samples and theri embeddings to ChromaDB."""

    def __init__(
        self,
        embedding_model: str,
        samples_path: Path,
        collection_name: str = "samples",
    ):
        self.embedding_model: str = embedding_model
        self.samples_path: Path = samples_path
        self.collection_name: str = collection_name

    async def initialize(self) -> None:
        print("Initializing ChromaDB client...", end="", flush=True)
        chromadb_host = urlparse(CHROMADB_URL).hostname
        chromadb_port = urlparse(CHROMADB_URL).port
        assert isinstance(chromadb_host, str), "Cannot parse ChromaDB URL (hostname)"
        assert isinstance(chromadb_port, int), "Cannot parse ChromaDB URL (port)"
        self.emb_client = AsyncOpenAI(
            api_key=EMB_API_KEY,
            base_url=EMB_BASE_URL,
        )
        self.db_client = await AsyncHttpClient(
            host=chromadb_host,
            port=chromadb_port,
        )
        self.collection = await self.db_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
        )
        print("done")

    async def upload_samples(self, samples: list[Sample]) -> None:
        prev_ids = await self.collection.get(
            ids=[sample.id for sample in samples], include=[]
        )

        samples = [s for s in samples if s.id not in prev_ids["ids"]]
        if samples:
            print(
                f'Uploading {len(samples)} new samples to "{self.collection_name}"...',
                end="",
                flush=True,
            )

            samples_batches = [
                samples[i : i + EMB_BATCH_SIZE]
                for i in range(0, len(samples), EMB_BATCH_SIZE)
            ]

            for samples_batch in samples_batches:
                response = await self.emb_client.embeddings.create(
                    input=[sample.completion for sample in samples_batch],
                    model=self.embedding_model,
                )
                await self.collection.add(
                    ids=[sample.id for sample in samples_batch],
                    embeddings=[emb.embedding for emb in response.data],
                    metadatas=[
                        {
                            "model": sample.model,
                            "prompt_id": sample.prompt_id,
                            "centroid": False,
                        }
                        for sample in samples_batch
                    ],
                    documents=[sample.completion for sample in samples_batch],
                )
            print("done")
        else:
            print(f'No new samples to upload to "{self.collection_name}"')

    async def upsert_centroid(self, model: str, prompt_id: str):
        samples = await self.collection.get(
            where={"$and": [{"model": model}, {"prompt_id": prompt_id}]},
            include=["embeddings"],  # type: ignore
        )
        centroid = np.array(samples["embeddings"]).mean(axis=0).tolist()
        centroid_id = f"centroid_{model}_{prompt_id}"
        await self.collection.add(
            ids=[centroid_id],
            embeddings=[centroid],
            metadatas=[
                {
                    "model": model,
                    "prompt_id": prompt_id,
                    "centroid": True,
                    "sample_count": len(samples["ids"]),
                }
            ],
        )

    async def upsert_centroids(self) -> None:
        print("Update/Inserting centroids...", end="", flush=True)

        results = await self.collection.get(include=["metadatas"])  # type: ignore
        assert results["metadatas"] is not None

        model_prompt_pairs = set()
        for metadata in results["metadatas"]:
            if metadata and "model" in metadata and "prompt_id" in metadata:
                model_prompt_pairs.add((metadata["model"], metadata["prompt_id"]))

        for model, prompt_id in model_prompt_pairs:
            await self.upsert_centroid(model, prompt_id)

        print(f"done ({len(model_prompt_pairs)})")

    async def load_samples(self) -> list[Sample]:
        with open(self.samples_path) as f:
            samples = [Sample.model_validate_json(line) for line in f]
        return samples

    async def main(self):
        await self.initialize()

        initial_count = await self.collection.count()
        print(f'Collection "{self.collection_name}" has {initial_count} samples')

        samples = await self.load_samples()
        await self.upload_samples(samples)

        await self.upsert_centroids()

        final_count = await self.collection.count()
        print(f'Collection "{self.collection_name}" has {final_count} samples')


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    samples_path = root / "data/samples/20250316T174152.jsonl"
    uploader = SamplesUploader(
        embedding_model="jinaai/jina-embeddings-v2-base-en",
        samples_path=samples_path,
        collection_name="test1",
    )
    asyncio.run(uploader.main())
