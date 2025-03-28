import os
import uuid

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from llm_fingerprint.mixin import EmbeddingsMixin
from llm_fingerprint.models import Result, Sample
from llm_fingerprint.storage.base import VectorStorage


class QdrantStorage(VectorStorage, EmbeddingsMixin):
    def __init__(
        self,
        embedding_model: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.qdrant_url = qdrant_url if qdrant_url else os.getenv("QDRANT_URL")
        self.qdrant_api_key = (
            qdrant_api_key if qdrant_api_key else os.getenv("QDRANT_API_KEY")
        )
        if self.qdrant_url is None:
            raise ValueError("QDRANT_URL is not set")

        VectorStorage.__init__(self)
        if embedding_model is not None:
            EmbeddingsMixin.__init__(self, embedding_model=embedding_model)

    async def initialize(self, collection_name: str) -> None:
        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
        collection_exists = await self.client.collection_exists(collection_name)
        if not collection_exists:
            dummy_sample = Sample(id="", prompt_id="", model="", completion="dummy")
            embeddings = await self.embed_samples([dummy_sample])
            emb_size = len(embeddings[0])
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=emb_size,
                    distance=models.Distance.COSINE,
                ),
            )
        self.collection_name = collection_name

    async def _get_all_samples(
        self,
        with_payload: bool = False,
        with_vectors: bool = False,
    ) -> list[models.Record]:
        samples: list[models.Record] = []
        offset = None
        while True:
            samples_batch, offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="centroid", match=models.MatchValue(value=False)
                        )
                    ]
                ),
                limit=1024,
                with_payload=with_payload,
                with_vectors=with_vectors,
                offset=offset,
            )
            samples.extend(samples_batch)
            if offset is None:
                break

        return samples

    async def upload_samples(
        self,
        samples: list[Sample],
        batch_size: int = 8,
        embeddings: list[list[float]] | None = None,
    ):
        # TODO: maybe update only the new ids
        # prev_samples = await self._get_all_records(with_payload=False)
        for i in range(0, len(samples), batch_size):
            samples_batch = samples[i : i + batch_size]
            if embeddings is None:
                embeddings_batch = await self.embed_samples(samples_batch)
            else:
                embeddings_batch = embeddings[i : i + batch_size]
            points = [
                models.PointStruct(
                    id=sample.id,
                    payload={
                        "model": sample.model,
                        "prompt_id": sample.prompt_id,
                        "centroid": False,
                    },
                    vector=embedding,
                )
                for sample, embedding in zip(samples_batch, embeddings_batch)
            ]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    async def query_sample(
        self,
        sample: Sample,
        embedding: list[float] | None = None,
    ) -> list[Result]:
        if embedding is None:
            embedding = (await self.embed_samples([sample]))[0]
        centroids = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="centroid", match=models.MatchValue(value=True)
                    ),
                    models.FieldCondition(
                        key="prompt_id", match=models.MatchValue(value=sample.prompt_id)
                    ),
                ]
            ),
            limit=10,
        )

        language_models: list[str] = []
        distances: list[float] = []
        for centroid in centroids.points:
            assert isinstance(centroid.payload, dict)
            language_models.append(str(centroid.payload["model"]))
            distances.append(centroid.score)

        results = [
            Result(model=model, score=score)
            for model, score in zip(language_models, distances)
        ]

        return results

    async def upsert_centroid(self, model: str, prompt_id: str) -> None:
        samples, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="centroid", match=models.MatchValue(value=False)
                    ),
                    models.FieldCondition(
                        key="prompt_id", match=models.MatchValue(value=prompt_id)
                    ),
                    models.FieldCondition(
                        key="model", match=models.MatchValue(value=model)
                    ),
                ]
            ),
            # we assert that for each (model, prompt_id) pair there are less
            # than 1024 samples
            limit=1024,
            with_payload=False,
            with_vectors=True,
        )
        assert all(
            isinstance(sample.vector, list)
            and all(isinstance(x, float) for x in sample.vector)
            for sample in samples
        )

        embedding = np.array([s.vector for s in samples]).mean(axis=0).tolist()
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        "model": model,
                        "prompt_id": prompt_id,
                        "centroid": True,
                        "sample_count": len(samples),
                    },
                    vector=embedding,
                )
            ],
        )

    async def upsert_centroids(self) -> None:
        samples = await self._get_all_samples(with_payload=True)
        assert all(sample.payload is not None for sample in samples)

        centroids: set[tuple[str, str]] = set()
        for sample in samples:
            assert isinstance(sample.payload, dict)
            centroid = (
                str(sample.payload["model"]),
                str(sample.payload["prompt_id"]),
            )
            centroids.add(centroid)

        for model, prompt_id in centroids:
            await self.upsert_centroid(model, prompt_id)
