import os
import uuid
from urllib.parse import urlparse

import numpy as np
from chromadb import AsyncHttpClient
from chromadb.api.types import IncludeEnum

from llm_fingerprint.mixin import EmbeddingsMixin
from llm_fingerprint.models import Result, Sample
from llm_fingerprint.storage.base import VectorStorage


class ChromaStorage(VectorStorage, EmbeddingsMixin):
    def __init__(
        self,
        embedding_model: str | None = None,
        chroma_url: str | None = None,
    ):
        self.chormadb_url = chroma_url if chroma_url else os.getenv("CHROMADB_URL")
        if self.chormadb_url is None:
            raise ValueError("CHROMADB_URL is not set")

        VectorStorage.__init__(self)
        if embedding_model is not None:
            EmbeddingsMixin.__init__(self, embedding_model=embedding_model)

    async def initialize(self, collection_name: str) -> None:
        url = urlparse(self.chormadb_url)
        host, port = url.hostname, url.port
        assert isinstance(host, str), "Cannot parse ChromaDB URL (hostname)"
        assert isinstance(port, int), "Cannot parse ChromaDB URL (port)"
        self.client = await AsyncHttpClient(host=host, port=port)
        self.collection = await self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    async def upload_samples(
        self,
        samples: list[Sample],
        batch_size: int = 32,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        # Avoid to upload samples that are already in the database
        ids = {sample.id for sample in samples}
        prev_ids = await self.collection.get(ids=list(ids), include=[])
        new_ids = ids - set(prev_ids["ids"])
        new_samples = [sample for sample in samples if sample.id in new_ids]

        # Upload samples with their embeddings
        for i in range(0, len(new_samples), batch_size):
            samples_batch = new_samples[i : i + batch_size]
            if embeddings is None:
                embeddings_batch = await self.embed_samples(samples_batch)
            else:
                embeddings_batch = embeddings[i : i + batch_size]
            await self.collection.add(
                ids=[sample.id for sample in samples_batch],
                embeddings=[emb for emb in embeddings_batch],
                documents=[sample.completion for sample in samples_batch],
                metadatas=[
                    {
                        "model": sample.model,
                        "prompt_id": sample.prompt_id,
                        "centroid": False,
                    }
                    for sample in samples_batch
                ],
            )

    async def query_sample(
        self,
        sample: Sample,
        embedding: list[float] | None = None,
    ) -> list[Result]:
        if embedding is None:
            embedding = (await self.embed_samples([sample]))[0]
        centroids = await self.collection.query(
            query_embeddings=embedding,
            include=[IncludeEnum.metadatas, IncludeEnum.distances],
            where={"$and": [{"centroid": True}, {"prompt_id": sample.prompt_id}]},
        )

        assert centroids["metadatas"] is not None
        assert centroids["distances"] is not None

        language_models = [
            str(metadata["model"]) for metadata in centroids["metadatas"][0]
        ]
        distances = centroids["distances"][0]

        # NOTE:
        # ChromaDB returns the value returns cosine distance rather then cosine
        # similarity.
        # * cosine simliarity = x dot y / ||x|| ||y| -> [-1, 1]
        # * cosine distance = 1 — cosine similarity  -> [0, 2]
        # We need to convert the cosine distance to cosine similarity because
        # in the rest of the code higher score means more similar
        # => 1 - float(score)

        results = [
            Result(model=str(model), score=1 - float(score))
            for model, score in zip(language_models, distances)
        ]

        return results

    async def upsert_centroid(self, model: str, prompt_id: str) -> None:
        samples = await self.collection.get(
            where={"$and": [{"model": model}, {"prompt_id": prompt_id}]},
            include=[IncludeEnum.embeddings],
        )
        await self.collection.add(
            ids=str(uuid.uuid4()),
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
        for model, prompt_id in centroids:
            await self.upsert_centroid(model, prompt_id)
