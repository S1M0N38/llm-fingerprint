# Contributing

<!--TODO: add intro-->

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
uv sync --dev
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

<!--TODO: add project structure-->

## Project Style Guide

The project style guide is enforced using [pre-commit](https://pre-commit.com/).

To install pre-commit hooks, run:

```sh
pre-commit install
```

The following hooks are used:

- [conventional-commits](https://www.conventionalcommits.org/en/v1.0.0/): for consistent commit messages.
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- [uv](https://docs.astral.sh/uv/) for managing environment and dependencies.
