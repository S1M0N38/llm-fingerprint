"""Module for querying ChromaDB to identify models."""

import asyncio
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
from chromadb import AsyncHttpClient
from openai import AsyncOpenAI

from llm_fingerprint.models import Result, Sample

assert (EMB_API_KEY := os.getenv("EMB_API_KEY", ""))
assert (EMB_BASE_URL := os.getenv("EMB_BASE_URL", ""))
assert (CHROMADB_URL := os.getenv("CHROMADB_URL", ""))


class SamplesQuerier:
    """Class for querying ChromaDB to identify models."""

    def __init__(
        self,
        embedding_model: str,
        samples_path: Path,
        retults_path: Path,
        results_num: int = 5,
        collection_name: str = "samples",
    ):
        self.embedding_model: str = embedding_model
        self.samples_path: Path = samples_path
        self.results_path: Path = retults_path
        self.results_num: int = results_num
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
        self.collection = await self.db_client.get_collection(
            name=self.collection_name,
            embedding_function=None,
        )
        print("done")

    async def load_samples(self) -> list[Sample]:
        with open(self.samples_path) as f:
            samples = [Sample.model_validate_json(line) for line in f]
        return samples

    async def query_sample(self, sample: Sample) -> list[Result]:
        """Query a sample against the database and return results."""

        response = await self.emb_client.embeddings.create(
            input=[sample.completion],
            model=self.embedding_model,
        )

        assert response.data

        query_results = await self.collection.query(
            query_embeddings=response.data[0].embedding,
            n_results=self.results_num,
            include=["metadatas", "distances"],  # type: ignore
            where={"$and": [{"centroid": True}, {"prompt_id": sample.prompt_id}]},
        )

        assert query_results["metadatas"] is not None
        assert query_results["distances"] is not None

        models = [metadata["model"] for metadata in query_results["metadatas"][0]]
        distances = query_results["distances"][0]

        assert isinstance(models, list) and all(isinstance(m, str) for m in models)
        assert isinstance(models, list)

        results = [
            Result(model=model, score=score)  # type: ignore
            for model, score in zip(models, distances)
        ]

        return results

    def aggeregate_results(self, results_list: list[list[Result]]) -> list[Result]:
        """Aggregate results from multiple queries by taking the average score."""

        results_dict: defaultdict[str, list[float]] = defaultdict(list)
        for results in results_list:
            for result in results:
                results_dict[result.model].append(result.score)

        agg_results: list[Result] = []
        for model, scores in results_dict.items():
            agg_results.append(
                Result(
                    model=model,
                    score=sum(scores) / len(scores),
                )
            )

        return agg_results

    async def query_samples(self, samples: list[Sample]) -> list[Result]:
        print(f"Querying {len(samples)} samples...", end="", flush=True)
        results_list = [await self.query_sample(sample) for sample in samples]
        results = self.aggeregate_results(results_list)
        print("done")
        return results

    async def save_results(self, results: list[Result]) -> None:
        async with aiofiles.open(self.results_path, "a") as f:
            for result in results:
                await f.write(result.model_dump_json() + "\n")

    async def main(self):
        await self.initialize()

        print(f"Querying collection {self.collection_name} for results")
        samples = await self.load_samples()

        results = await self.query_samples(samples)
        await self.save_results(results)

        print(f"Query results saved to {self.results_path}")


if __name__ == "__main__":
    from llm_fingerprint.generate import SamplesGenerator

    root = Path(__file__).parent.parent.parent
    prompts_path = root / "data/prompts/prompts_single_v1.jsonl"
    samples_path = Path("/tmp/samples.jsonl")
    results_path = Path("/tmp/results.jsonl")

    samples_path.unlink(True)
    results_path.unlink(True)

    # Generate samples
    generator = SamplesGenerator(
        language_model="test-model",
        prompts_path=prompts_path,
        samples_path=samples_path,
        samples_num=1,
        max_tokens=2048,
    )
    asyncio.run(generator.main())

    # Query samples
    querier = SamplesQuerier(
        embedding_model="jinaai/jina-embeddings-v2-base-en",
        samples_path=samples_path,
        retults_path=results_path,
        results_num=5,
        collection_name="test1",
    )

    asyncio.run(querier.main())

    print(samples_path.read_text())
    print(results_path.read_text())
