import os
from urllib.parse import urlparse

import numpy as np
from chromadb import AsyncHttpClient
from chromadb.api.types import IncludeEnum

from llm_fingerprint.mixin import EmbeddingsMixin
from llm_fingerprint.models import Result, Sample
from llm_fingerprint.storage.base import VectorStorage


class ChromaStorage(VectorStorage, EmbeddingsMixin):
    def __init__(self, embedding_model: str, chroma_url: str | None = None):
        self.chormadb_url = chroma_url if chroma_url else os.getenv("CHROMADB_URL")
        if self.chormadb_url is None:
            raise ValueError("CHROMADB_URL is not set")
        super().__init__(embedding_model=embedding_model)

    async def initialize(self, collection_name: str) -> None:
        url = urlparse(self.chormadb_url)
        host, port = url.hostname, url.port
        assert isinstance(host, str), "Cannot parse ChromaDB URL (hostname)"
        assert isinstance(port, int), "Cannot parse ChromaDB URL (port)"
        self.client = await AsyncHttpClient(host=host, port=port)
        self.collection = await self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    async def upload_samples(
        self,
        samples: list[Sample],
        batch_size: int = 32,
    ) -> None:
        # Avoid to upload samples that are already in the database
        ids = {sample.id for sample in samples}
        prev_ids = await self.collection.get(ids=list(ids), include=[])
        new_ids = ids - set(prev_ids["ids"])
        new_samples = [sample for sample in samples if sample.id in new_ids]

        # Upload samples with their embeddings
        for i in range(0, len(new_samples), batch_size):
            samples = new_samples[i : i + batch_size]
            embeddings = await self.embed_samples(samples)
            await self.collection.add(
                ids=[sample.id for sample in samples],
                embeddings=[emb for emb in embeddings],
                documents=[sample.completion for sample in samples],
                metadatas=[
                    {
                        "model": sample.model,
                        "prompt_id": sample.prompt_id,
                        "centroid": False,
                    }
                    for sample in samples
                ],
            )

    async def query_sample(
        self,
        sample: Sample,
        results_num: int,
    ) -> list[Result]:
        embeddings = await self.embed_samples([sample])
        centroids = await self.collection.query(
            query_embeddings=[emb for emb in embeddings],
            n_results=results_num,
            include=[IncludeEnum.metadatas, IncludeEnum.distances],
            where={"$and": [{"centroid": True}, {"prompt_id": sample.prompt_id}]},
        )

        assert centroids["metadatas"] is not None
        assert centroids["distances"] is not None

        models = [metadata["model"] for metadata in centroids["metadatas"][0]]
        distances = centroids["distances"][0]

        assert isinstance(models, list) and all(isinstance(m, str) for m in models)
        assert isinstance(models, list)

        results = [
            Result(model=str(model), score=float(score))
            for model, score in zip(models, distances)
        ]

        return results

    async def upsert_centroid(self, model: str, prompt_id: str) -> None:
        samples = await self.collection.get(
            where={"$and": [{"model": model}, {"prompt_id": prompt_id}]},
            include=[IncludeEnum.embeddings],
        )
        await self.collection.add(
            ids=f"centroid_{model}_{prompt_id}",
            embeddings=np.array(samples["embeddings"]).mean(axis=0).tolist(),
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
        samples = await self.collection.get(include=[IncludeEnum.metadatas])
        assert samples["metadatas"] is not None

        centroids = {
            (str(metadata["model"]), str(metadata["prompt_id"]))
            for metadata in samples["metadatas"]
        }
        for model, prompt in centroids:
            await self.upsert_centroid(model, prompt)
