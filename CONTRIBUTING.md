# Contributing

Welcome to the LLM Fingerprinting project! This project aims to develop methods for identifying language models based on their response patterns. We appreciate your interest in contributing.

This document provides guidelines and instructions for contributing to the project. Whether you're fixing bugs, improving documentation, or proposing new features, your contributions are welcome.

## How to Contribute

1. **Report Issues**: If you find bugs or have feature requests, please create an issue on GitHub.

2. **Submit Pull Requests**: For code contributions, fork the repository, make your changes, and submit a pull request.

3. **Follow Coding Standards**: We use Ruff for linting and formatting, and pyright/basedpyright for type checking. Make sure your code passes all checks.

4. **Write Tests**: For new features or bug fixes, please include tests to validate your changes.

5. **Use Conventional Commits**: Follow the conventional commits specification for your commit messages.

## Environment Setup

This section describes how to set up the **recommended** development environment for this project [uv](https://docs.astral.sh/uv/).

1. Download the repository:

```sh
git clone https://github.com/S1M0N38/llm-fingerprint.git
cd llm-fingerprint
```

2. Create environment:

```sh
uv venv
uv sync --group dev
```

3. Set up environment variables:

```sh
cp .envrc.example .envrc
# And modify the .envrc file
```

The environment setup is now ready to use. Every time you are working on the project, you can activate the environment by running:

```sh
source .envrc
```

> You can use [direnv](https://github.com/direnv/direnv) to automatically activate the environment when you enter the project directory.

## Project Structure

### Files and Directories

The following diagram shows the main files and directories in the project. This structure helps understand how the code is organized and where to find specific functionality.

```
llm-fingerprint/
│
├── config/
│   └── llama-cpp.yaml
│
├── data/
│   ├── chroma/                  # ChromaDB storage directory
│   ├── prompts/                 # Prompt files directory
│   └── samples/                 # Generated samples directory
│
├── src/
│   └── llm_fingerprint/
│       ├── __init__.py
│       ├── cli.py               # Command-line interface
│       ├── commands.py          # Command implementations (Command pattern)
│       ├── io.py                # File I/O operations
│       ├── mixin.py             # Completion and Embedding mixins
│       ├── models.py            # Pydantic data models
│       ├── services.py          # Service layer (Generator, Uploader, Querier)
│       └── storage/
│           ├── __init__.py
│           ├── base.py          # Abstract VectorStorage class
│           └── implementation/
│               ├── __init__.py
│               ├── chroma.py    # ChromaDB implementation
│               └── qdrant.py    # Qdrant implementation
│
├── .envrc.example               # Environment variables template
├── CHANGELOG.md                 # Project changelog
├── CONTRIBUTING.md              # Contribution guidelines
├── README.md                    # Project documentation
├── justfile                     # Command runner configuration
├── pyproject.toml               # Python project configuration
└── uv.lock                      # Dependencies lock file
```

### Phases

The diagram below illustrates the three main phases of the LLM fingerprinting process and how they connect. Understanding this workflow is essential for contributing to any part of the system.

```mermaid
flowchart LR
 subgraph subGraph0["Phase 1: Generate"]
        direction TB
        GenPrompts["Load Standard Prompts"]
        GenComp["Generate Multiple Completions per Prompt for each LLM"]
        GenSave["Save Samples to File"]
        GenPrompts --> GenComp --> GenSave
  end
 subgraph subGraph1["Phase 2: Upload"]
        direction TB
        UpSamples["Load Samples"]
        GenEmbeddings["Generate Embeddings"]
        StoreVectors["Store in Vector DB"]
        ComputeCentroids["Compute Centroids for each Model-Prompt Combination"]
        UpSamples --> GenEmbeddings --> StoreVectors --> ComputeCentroids
  end
 subgraph subGraph2["Phase 3: Query"]
        direction TB
        UnkPrompts["Load Unknown Model Samples"]
        UnkEmbed["Generate Embeddings"]
        CompSim["Compare with Known Model Centroids"]
        RankMatch["Rank Matching Models"]
        UnkPrompts --> UnkEmbed --> CompSim --> RankMatch
  end
    subGraph0 --> subGraph1 --> subGraph2
```

### Data Models

The system uses three main data models to represent the data flowing through the system. These Pydantic models define the structure of prompts, samples, and query results. The `id` are UUIDs generated for each object using `uuid.uuid4()` and then converted to strings (this produces lowercase hyphenated UUIDs).

```mermaid
classDiagram
    class Prompt {
        +id: str
        +prompt: str
    }

    class Sample {
        +id: str
        +model: str
        +prompt_id: str
        +completion: str
    }

    class Result {
        +model: str
        +score: float
    }
```

### Core Classes

#### 1. Command and Service Classes

This diagram shows the command pattern implementation and the service classes that perform the actual operations.

```mermaid
classDiagram
    class Command {
        <<Abstract>>
        +args: Namespace
        +__init__(args: Namespace)
        +execute() async* None
    }

    class GenerateCommand {
        +execute() async None
    }

    class UploadCommand {
        +execute() async None
    }

    class QueryCommand {
        +execute() async None
    }

    class GeneratorService {
        +file_io: FileIO
        +samples_num: int
        +semaphore: Semaphore
        +__init__(file_io: FileIO, samples_num: int, language_model: str, max_tokens: int, concurrent_requests: int)
        +main() async None
    }

    class UploaderService {
        +file_io: FileIO
        +storage: VectorStorage
        +__init__(file_io: FileIO, storage: VectorStorage)
        +main() async None
    }

    class QuerierService {
        +file_io: FileIO
        +storage: VectorStorage
        +results_num: int
        +__init__(file_io: FileIO, storage: VectorStorage, results_num: int)
        +main() async None
    }

    Command <|-- GenerateCommand : extends
    Command <|-- UploadCommand : extends
    Command <|-- QueryCommand : extends
    GenerateCommand --> GeneratorService : creates
    UploadCommand --> UploaderService : creates
    QueryCommand --> QuerierService : creates
```

#### 2. GeneratorService

This diagram shows the GeneratorService class and its relationships with completion-related classes.

```mermaid
classDiagram
    class CompletionsMixin {
        +language_model: str
        +max_tokens: int
        +language_client: AsyncOpenAI
        +__init__(language_model: str, max_tokens: int)
        +generate_sample(prompt: Prompt) async Sample
    }

    class FileIO {
        +prompts_path: Path
        +samples_path: Path
        +__init__(prompts_path?: Path, samples_path?: Path, results_path?: Path)
        +load_prompts() async list[Prompt]
        +save_sample(sample: Sample) async None
    }

    class GeneratorService {
        +file_io: FileIO
        +samples_num: int
        +semaphore: Semaphore
        +__init__(file_io: FileIO, samples_num: int, language_model: str, max_tokens: int, concurrent_requests: int)
        +main() async None
    }

    FileIO --o GeneratorService : uses
    GeneratorService *-- CompletionsMixin : uses
```

#### 3. QuerierService and UploaderService

This diagram shows the QuerierService and UploaderService classes and their relationships with storage-related classes.

```mermaid
classDiagram
    class FileIO {
        +prompts_path: Path
        +samples_path: Path
        +results_path: Path
        +__init__(prompts_path?: Path, samples_path?: Path, results_path?: Path)
        +load_samples() async list[Sample]
        +save_results(results: list[Result]) async None
        +load_results() async list[Result]
    }

    class VectorStorage {
        <<Abstract>>
        +initialize(collection_name: str) async* None
        +upload_samples(samples: list[Sample]) async* None
        +query_sample(sample: Sample) async* list[Result]
        +upsert_centroids() async* None
        +query_samples(samples: list[Sample], results_num: int) async list[Result]
        #aggregate_results(results_list: list[list[Result]]) list[Result]
    }

    class ChromaStorage {
        -chormadb_url: str
        -client: AsyncHttpClient
        -collection: Collection
        +__init__(embedding_model: str, chroma_url: str|None)
        +initialize(collection_name: str) async None
        +upload_samples(samples: list[Sample], batch_size: int) async None
        +query_sample(sample: Sample) async list[Result]
        +upsert_centroid(model: str, prompt_id: str) async None
        +upsert_centroids() async None
    }

    class QdrantStorage {
        -qdrant_url: str
        -qdrant_api_key: str
        -client: AsyncQdrantClient
        -collection_name: str
        +__init__(embedding_model: str, qdrant_url: str|None, qdrant_api_key: str|None)
        +initialize(collection_name: str) async None
        +upload_samples(samples: list[Sample], batch_size: int) async None
        +query_sample(sample: Sample) async list[Result]
        +upsert_centroid(model: str, prompt_id: str) async None
        +upsert_centroids() async None
        -_get_all_samples(with_payload: bool, with_vectors: bool) async list[Record]
    }

    class EmbeddingsMixin {
        +embedding_model: str
        +embedding_client: AsyncOpenAI
        +__init__(embedding_model: str, base_url: str|None, api_key: str|None)
        +embed_samples(samples: list[Sample]) async list[list[float]]
    }

    class UploaderService {
        +file_io: FileIO
        +storage: VectorStorage
        +__init__(file_io: FileIO, storage: VectorStorage)
        +main() async None
    }

    class QuerierService {
        +file_io: FileIO
        +storage: VectorStorage
        +results_num: int
        +__init__(file_io: FileIO, storage: VectorStorage, results_num: int)
        +main() async None
    }

    FileIO --o UploaderService : uses
    FileIO --o QuerierService : uses
    VectorStorage <|-- ChromaStorage : implements
    VectorStorage <|-- QdrantStorage : implements
    ChromaStorage *-- EmbeddingsMixin : uses
    QdrantStorage *-- EmbeddingsMixin : uses
    UploaderService o-- VectorStorage : uses
    QuerierService o-- VectorStorage : uses
```

### Storage Schema

This diagram represents how data is organized in the vector database. The system stores both individual sample documents and computed centroids that represent the average embeddings for a specific model-prompt combination. Both documents are stored in the same collection.

```mermaid
flowchart TB

  subgraph "Centroid Documents"
      direction TB
      CentroidID["centroid_model_promptid"]
      CentroidEmbed["average_embedding_vector"]
      CentroidMeta["metadata:
          - model: string
          - prompt_id: string
          - centroid: true
          - sample_count: int"]
  end

  subgraph "Sample Documents"
      direction TB
      SampleID["sample_id"]
      SampleEmbed["embedding_vector"]
      SampleDoc["completion_text"]
      SampleMeta["metadata:
          - model: string
          - prompt_id: string
          - centroid: false"]
  end
```

### CLI Commands

This diagram shows the command-line interface structure, including the three main commands (generate, upload, query) and their respective parameters.

```mermaid
graph TD
    CLI["llm-fingerprint"]

    CLI -->|generate| Generate["
        --language-model (required)
        --prompts-path (required)
        --samples-path (required)
        --samples-num (default: 5)
        --max-tokens (default: 2048)"]

    CLI -->|upload| Upload["
        --embedding-model (required)
        --samples-path (required)
        --collection-name (default: samples)"]

    CLI -->|query| Query["
        --embedding-model (required)
        --samples-path (required)
        --results-path (required)
        --results-num (default: 5)"]
```

## Release Cycle

The project follows an automated release process using GitHub Actions:

1. **Conventional Commits**: All commits should follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

2. **Release Please PR**: The [Release Please](https://github.com/googleapis/release-please) GitHub Action automatically maintains a release PR that:

   - Updates the version in `pyproject.toml`
   - Updates the `CHANGELOG.md` based on conventional commits
   - The PR is continuously updated as new commits are added to the main branch

   **Important**: Never manually modify `uv.lock`, `CHANGELOG.md`, or version numbers in `pyproject.toml` or `__init__.py`. These are automatically maintained by the release pipeline.

3. **Version Release**: When ready for a new release, the repository owner merges the Release Please PR, which:

   - Triggers the creation of a new Git tag (e.g., `v0.4.0`)
   - Creates a GitHub Release with release notes

4. **[PyPI Publication](https://pypi.org/project/llm_fingerprint/)**: When a new version tag is pushed, the Release PyPI workflow:
   - Builds the Python package
   - Publishes it to PyPI using trusted publishing

This automated process ensures consistent versioning, comprehensive changelogs, and reliable package distribution with minimal manual intervention.
