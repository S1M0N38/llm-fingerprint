from abc import ABC, abstractmethod
from argparse import Namespace

from tqdm import tqdm

from llm_fingerprint.io import FileIO
from llm_fingerprint.services import GeneratorService, QuerierService, UploaderService
from llm_fingerprint.storage.implementation.chroma import ChromaStorage
from llm_fingerprint.storage.implementation.qdrant import QdrantStorage


class Command(ABC):
    """Base command interface for all CLI commands."""

    def __init__(self, args: Namespace):
        self.args = args

    @abstractmethod
    async def execute(self) -> None:
        """Execute the command with the provided arguments."""
        pass


class GenerateCommand(Command):
    """Command to generate samples from LLMs."""

    async def execute(self) -> None:
        """Execute the generate command."""
        self.args.samples_path.parent.mkdir(parents=True, exist_ok=True)

        for model in tqdm(
            self.args.language_model, desc="Generate samples", unit="model"
        ):
            # Create file IO and generator service
            file_io = FileIO(
                prompts_path=self.args.prompts_path, samples_path=self.args.samples_path
            )
            generator = GeneratorService(
                file_io=file_io,
                samples_num=self.args.samples_num,
                language_model=model,
                max_tokens=self.args.max_tokens,
                concurrent_requests=self.args.concurrent_requests,
            )

            # Execute the generate operation
            await generator.main()

            # Close the language client
            await generator.language_client.close()


class UploadCommand(Command):
    """Command to upload samples to vector storage."""

    async def execute(self) -> None:
        """Execute the upload command."""
        match self.args.storage:
            case "chroma":
                storage = ChromaStorage(self.args.embedding_model)
            case "qdrant":
                storage = QdrantStorage(self.args.embedding_model)
            case _:
                raise NotImplementedError(
                    f"Storage '{self.args.storage}' not implemented"
                )

        await storage.initialize(self.args.collection_name)

        # Create file IO and uploader service
        file_io = FileIO(samples_path=self.args.samples_path)
        uploader = UploaderService(file_io=file_io, storage=storage)

        # Execute the upload operation
        await uploader.main()


class QueryCommand(Command):
    """Command to query storage for model identification."""

    async def execute(self) -> None:
        """Execute the query command."""
        match self.args.storage:
            case "chroma":
                storage = ChromaStorage(self.args.embedding_model)
            case "qdrant":
                storage = QdrantStorage(self.args.embedding_model)
            case _:
                raise NotImplementedError(
                    f"Storage '{self.args.storage}' not implemented"
                )

        await storage.initialize(self.args.collection_name)

        # Create file IO and querier service
        file_io = FileIO(
            samples_path=self.args.samples_path, results_path=self.args.results_path
        )
        querier = QuerierService(
            file_io=file_io,
            storage=storage,
            results_num=self.args.results_num,
        )

        # Execute the query operation
        await querier.main()
