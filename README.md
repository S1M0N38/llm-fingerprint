<div align="center">
  <h1>🕵&nbsp;LLM Fingerprint</h1>
  <p align="center">
    <a href="https://github.com/S1M0N38/llm-fingerprint">
      <img alt="Status" src="https://img.shields.io/badge/Status-WIP-yellow?style=for-the-badge"/>
    </a>
    <a href="https://github.com/S1M0N38/llm-fingerprint/tags">
      <img alt="Tag" src="https://img.shields.io/github/v/tag/S1M0N38/llm-fingerprint?style=for-the-badge&color=blue"/>
    </a>
  </p>
  <p>
    <em>Identify LLMs by their response fingerprints</em>
  </p>
  <hr>
</div>

## Research Question

Is it possible to identify which LLM generated a response by analyzing its semantic patterns across multiple standardized prompts?

## Approach

LLM Fingerprint uses semantic similarity patterns across multiple prompts to create model-specific "fingerprints":

1. **Fingerprint Creation**: Generate multiple responses from known LLMs using standardized prompts

   - Fixed sampling parameters to ensure consistent sampling behavior
   - Multiple response samples per prompt to account for sampling variance
   - Several distinct prompts to capture model characteristics

2. **Similarity Analysis**: Measure semantic similarity within and between prompt response groups

   - Within-prompt similarity reveals consistency characteristics
   - Cross-prompt similarity patterns create a unique model signature

3. **Model Identification**: Match patterns from unknown models against the fingerprint database

   - Generate responses from the unknown model using the same standardized prompts
   - Compare similarity patterns with known models
   - Identify the closest matching fingerprint

## Usage

Set required environment variables. See [`.envrc.example`](.envrc.example) for more details.

### Creating Model Fingerprints

```bash
# Generate samples for known model responses
llm-fingerprint generate \
  --language-model "model-1" "model-2" "model-3" \
  --prompts-path "./data/prompts/prompts_general_v1.jsonl" \
  --samples-path "samples.jsonl" \
  --samples-num 4

# Upload samples to ChromaDB
llm-fingerprint upload \
  --language-model "embedding-model" \
  --samples-path "samples.jsonl" \
  --collection-name "samples"
```

### Identifying Unknown Models

```bash
# Generate samples for unknown model (or use an external service)
# Let's suppose the we don't know we are using model-2
llm-fingerprint generate \
  --language-model "model-2" \
  --prompts-path "./data/prompts/prompts_single_v1.jsonl" \
  --samples-path "unk-samples.jsonl" \
  --samples-num 1

# Query ChromaDB for model identification
llm-fingerprint query \
  --language-model "embedding-model" \
  --samples-path "unk-samples.jsonl" \
  --results-path "results.jsonl" \
  --results-num 2

# matches.jsonl will contain the results
# {"model": "model-2", "score": ... }
# {"model": "model-1", "score": ... }
```

## Installation

The preferred way to install `llm-fingerprint` is using [`uv`](https://docs.astral.sh/uv/) (although you can also use `pip`).

```bash
# Clone the repository
git clone https://github.com/S1M0N38/llm-fingerprint.git
cd llm-fingerprint

# Create a virtual environment
uv venv

# Install the package
uv sync # --all-groups # for installing ml and dev groups
```

## Requirements

- Python 3.11+
- OpenAI-compatible API endpoints (`/chat/completions` and `/embeddings`)
- Access to ChromaDB (locally or hosted)

## Contributing

This toy/research project is still in its early stages, and I welcome any feedback, suggestions, and contributions! If you're interested in discussing ideas or have questions about the approach, please start a conversation in [GitHub Discussions](https://github.com/S1M0N38/llm-fingerprint/discussions).

For detailed information on setting up your development environment, understanding the project structure, and the contribution workflow, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).
