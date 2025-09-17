# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

- Python-based image denoising tool using PyTorch
- Implements models from Benoit Brummer's NIND Denoise and Darktable workflow
- Supports CPU, NVIDIA GPU, Intel XPU/GPU acceleration

## Build/Lint/Test Commands

```bash
# Install dependencies (in project root)
python -m pip install --upgrade pip
pip install -e .

# Run all tests (excluding integration tests)
pytest -m "not integration"

# Run a single test file
pytest tests/test_denoise_cli.py

# Linting (using pylint configured in pyproject.toml)
pylint src/nind_denoise/
```

## Code Style Guidelines

- **Imports**: Standard Python imports, no wildcards
- **Line Length**: 100 characters (configured in pyproject.toml)
- **Type Hints**: Required for all functions and methods
- **Error Handling**: Use custom exceptions from `src/nind_denoise/exceptions.py`
- **Configuration**: YAML files for config, validated via `src/nind_denoise/config/`

## Critical Project-Specific Patterns

- **Pipeline Execution**: Always use `run_pipeline()` from `nind_denoise.pipeline` for image processing
- **Job Context**: Use `JobContext` class for managing pipeline state and resources
- **Custom Exceptions**: Import and use exceptions from `src/nind_denoise/exceptions.py`
- **Configuration**: Load configs via `Config` class in `nind_denoise.config.config`
- **Path Handling**: Use `pathlib.Path` consistently throughout the codebase

## Testing Specifics

- Tests are in `tests/` directory (no separate test folder structure)
- Integration tests marked with `@pytest.mark.integration` and excluded by default
- Requires test data from `tests/test_raw/` for some tests
- Test configurations in `conftest.py`

## Directory Structure Notes

- Source code in `src/nind_denoise/`
- Config files in `src/nind_denoise/config/`
- Models and libraries in `src/nind_denoise/libs/`
- Pipeline components in `src/nind_denoise/pipeline/`
- Training code in `src/nind_denoise/train/`

## Dependency Management

- Primary dependencies in `pyproject.toml` [project.dependencies]
- Test dependencies in `pyproject.toml` [dependency-groups.test]
- Additional requirements in `requirements.in`