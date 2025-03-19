import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from chromadb import AsyncHttpClient
from chromadb.utils import embedding_functions

from llm_fingerprint.models import Sample

assert (CHROMADB_MODEL := os.getenv("CHROMADB_MODEL", ""))
assert (CHROMADB_URL := os.getenv("CHROMADB_URL", ""))
assert (CHROMADB_DEVICE := os.getenv("CHROMADB_DEVICE", "cpu"))

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
        asyncio.run(self._init_client())

    async def _init_client(self) -> None:
        print("Initializing ChromaDB client...", end="", flush=True)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore
                model_name=CHROMADB_MODEL,
                trust_remote_code=True,
                device=CHROMADB_DEVICE,
                # NOTE: trust_remote_code=True is needed for some models.
                # Use at you own risk. After downloading the model, you can pin it
                # to a specific version.
            )
        )
        self.client = await AsyncHttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
        )
        assert await self.client.heartbeat(), "0 BPM for chromadb"
        self.collection = await self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,  #  type: ignore
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
        samples_path=samples_path,
        collection_name="test1",
    )
    asyncio.run(uploader.main())
