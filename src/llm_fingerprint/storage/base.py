from abc import ABC, abstractmethod
from collections import defaultdict

from llm_fingerprint.models import Result, Sample


class VectorStorage(ABC):
    """Abstract interface for vector storage backends.

    This interface defines the operations needed for the LLM fingerprinting system.
    Implementations can use different vector databases (ChromaDB, Qdrant, etc.).
    """

    @abstractmethod
    async def initialize(
        self,
        collection_name: str,
    ) -> None:
        pass

    @abstractmethod
    async def upload_samples(
        self,
        samples: list[Sample],
    ) -> None:
        pass

    @abstractmethod
    async def query_sample(
        self,
        sample: Sample,
    ) -> list[Result]:
        pass

    @abstractmethod
    async def upsert_centroids(
        self,
    ) -> None:
        pass

    @staticmethod
    def aggregate_results(results_list: list[list[Result]]) -> list[Result]:
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

    async def query_samples(
        self,
        samples: list[Sample],
        results_num: int = 5,
    ) -> list[Result]:
        results_list = [await self.query_sample(sample) for sample in samples]
        results = self.aggregate_results(results_list)
        results = sorted(results, key=lambda x: x.score)[:results_num]
        return results
