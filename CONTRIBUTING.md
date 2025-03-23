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
│       ├── mixin.py             # Completion and Embedding mixins
│       ├── models.py            # Pydantic data models
│       ├── services.py          # Service layer (Generator, Uploader, Querier)
│       └── storage/
│           ├── __init__.py
│           ├── base.py          # Abstract VectorStorage class
│           └── implementation/
│               ├── __init__.py
│               ├── chroma.py    # ChromaDB implementation
│               └── qdrant.py    # Qdrant implementation (placeholder)
│
├── .envrc.example               # Environment variables template
├── CHANGELOG.md                 # Project changelog
├── CONTRIBUTING.md              # Contribution guidelines
├── README.md                    # Project documentation
├── justfile                     # Command runner configuration
├── pyproject.toml               # Python project configuration
└── uv.lock                      # Dependencies lock file
```

### Project Phases

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

The system uses three main data models to represent the data flowing through the system. These Pydantic models define the structure of prompts, samples, and query results.

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

This diagram shows the main classes in the system and their relationships. The architecture follows a layered approach with storage abstractions, mixins for specific capabilities, and services that orchestrate the operations.

```mermaid
classDiagram
    class VectorStorage {
        <<Abstract>>
        +initialize(collection_name: str) async* None
        +upload_samples(samples: list[Sample]) async* None
        +query_sample(sample: Sample, results_num: int) async* list[Result]
        +upsert_centroids() async* None
        +query_samples(samples: list[Sample]) async list[Result]
        -&lowbar;aggregate&lowbar;results(results_list: list[list[Result]]) list[Result]
    }

    class ChromaStorage {
        -chromadb_url: str
        -client: AsyncHttpClient
        -collection: Collection
        +initialize(collection_name: str) async None
        +upload_samples(samples: list[Sample], batch_size: int) async None
        +query_sample(sample: Sample, results_num: int) async list[Result]
        +upsert_centroids() async None
        +upsert_centroid(model: str, prompt_id: str) async None
    }

    class CompletionsMixin {
        +language_model: str
        +max_tokens: int
        +language_client: AsyncOpenAI
        +&lowbar;&lowbar;init&lowbar;&lowbar;(language_model: str, max_tokens: int)
        +generate_sample(prompt: Prompt) async Sample
    }

    class EmbeddingsMixin {
        +embedding_model: str
        +embedding_client: AsyncOpenAI
        +&lowbar;&lowbar;init&lowbar;&lowbar;(embedding_model: str)
        +embed_samples(samples: list[Sample]) async list[list[float]]
    }

    class BaseService {
        <<Static Methods>>
        +load_prompts(prompts_path: Path) async list[Prompt]
        +save_prompts(prompts_path: Path, prompts: list[str]) async None
        +load_samples(samples_path: Path) async list[Sample]
        +save_sample(samples_path: Path, sample: Sample) async None
        +save_samples(samples_path: Path, samples: list[Sample]) async None
        +save_results(results_path: Path, results: list[Result]) async None
    }

    class GeneratorService {
        +prompts_path: Path
        +samples_path: Path
        +samples_num: int
        +semaphore: Semaphore
        +&lowbar;&lowbar;init&lowbar;&lowbar;(prompts_path: Path, samples_path: Path, samples_num: int, language_model: str, max_tokens: int, concurrent_requests: int)
        +main() async None
    }

    class UploaderService {
        +samples_path: Path
        +storage: VectorStorage
        +&lowbar;&lowbar;init&lowbar;&lowbar;(samples_path: Path, storage: VectorStorage)
        +main() async None
    }

    class QuerierService {
        +prompts_path: Path
        +results_path: Path
        +storage: VectorStorage
        +&lowbar;&lowbar;init&lowbar;&lowbar;(prompts_path: Path, results_path: Path, storage: VectorStorage)
        +main() async None
    }

    VectorStorage <|-- ChromaStorage : implements
    ChromaStorage *-- EmbeddingsMixin : uses
    GeneratorService *-- CompletionsMixin : uses
    BaseService <|-- GeneratorService : extends
    BaseService <|-- UploaderService : extends
    BaseService <|-- QuerierService : extends
    UploaderService o-- VectorStorage : has-a
    QuerierService o-- VectorStorage : has-a

    note for VectorStorage "Abstract base class for vector storage backends"
    note for EmbeddingsMixin "Provides embedding computation capabilities"
    note for CompletionsMixin "Provides completion generation capabilities"
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

This diagram shows the command-line interface structure, including the three main commands (generate, upload, query) and their respective parameters. The CLI is the primary way users interact with the system.

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

### Command Flow (CLI)

This diagram illustrates how the CLI commands flow through the system, from the initial command parsing to the service execution. It shows how the different components connect and interact.

```mermaid
flowchart TD
    CLI["CLI (cli.py)"] -- generate --> GenerateCmd["cmd_generate()"]
    CLI -- upload --> UploadCmd["cmd_upload()"]
    CLI -- query --> QueryCmd["cmd_query()"]
    GenerateCmd -- creates --> Generator["GeneratorService"]
    UploadCmd -- creates --> Storage["ChromaStorage"] & Uploader["UploaderService"]
    QueryCmd -- creates --> Storage2["ChromaStorage"] & Querier["QuerierService"]
    Generator -- calls --> GeneratorMain["generator.main()"]
    Uploader -- calls --> UploaderMain["uploader.main()"]
    GeneratorMain -- uses --> CompletionMixin["CompletionsMixin"]
    Storage -- uses --> EmbeddingMixin["EmbeddingsMixin"]
    Storage2 -- uses --> EmbeddingMixin
    Querier -- calls --> QuerierMain["querier.main()"]
```
