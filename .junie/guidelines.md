Project development guidelines for nind-denoise

This document captures project-specific build, configuration, testing, and development practices to help future contributors work efficiently. It intentionally focuses on details specific to this repository rather than generic Python advice.

1) Build and configuration

- Python/runtime
  - Requires Python >= 3.12 (pyproject.toml) and is actively developed/tested with modern Python (3.13 works as well).
  - Preferred workflow uses uv for virtual environments and dependency installation (see README for platform-specific commands). A standard venv via Python is also fine.

- Dependencies and installation
  - Python dependencies are declared in pyproject.toml. The dev dependency group includes pytest and pylint.
  - For quick setup, you can run: uv pip install -r requirements.in --upgrade
    - requirements.in is curated to resolve platform-appropriate wheels (notably for torch and friends). If you are not using uv, you can use pip install -r requirements.in, but uv will generally resolve variants more reliably across GPU/CPU setups.
  - External binaries required by the pipeline (runtime, not for unit tests):
    - darktable-cli (Darktable) — used twice for stage 1/2 exports.
    - gmic (gmic-cli) — used for RL deblur stage.
    - exiv2 (Python bindings) — used to read/write EXIF metadata.
  - The CLI attempts to auto-locate darktable-cli and gmic, but you can override with --dt and --gmic. On Windows, ensure these tools are discoverable on PATH or provide absolute paths.

- GPU/acceleration considerations
  - The README documents GPU setup across NVIDIA (CUDA), Intel XPU/Level Zero, and AMD/ROCm status. Torch is pinned to recent versions; use uv’s wheel variant resolution where possible.
  - If you need to validate acceleration, use the Torch snippet in the README (torch.accelerator / torch.cuda / torch.xpu checks) inside the project venv.

- Source layout
  - This repo uses a src layout: Python code lives under src/. The package name is nind-denoise with module paths under src/nind_denoise/ and a top-level orchestration script at src/denoise.py.

2) Testing

- Test runner
  - We use pytest (declared under [dependency-groups].dev in pyproject.toml). Typical invocations:
    - Run everything: pytest -q
    - Run a single file: pytest -q tests\test_pipeline.py
    - Run a test by node id: pytest -q tests\test_pipeline.py::test_noop_deblur_runs
  - Warning: Running the entire suite may be slow and may require external tools, this is expected.

- Importing code from src in tests (without installing the package)
  - Mandate editable installs
  - Prefer a shared tests\conftest.py that prepends the repository’s src directory to sys.path so individual tests don’t need per-file boilerplate.
  - Run just the test files covering code with changes to keep CI/dev iterations fast, e.g.:
    pytest -q tests\test_demo_temp.py

  - We created and executed this demo during preparation of this guide and confirmed it passes. We removed the temporary file after verifying it, per the instruction to leave no extra files in the repo.

- Adding new tests
  - Prefer placing tests under tests/ with descriptive names (e.g., test_pipeline_highlevel.py).
  - If practicable, Use in-memory temporary files and directories for transient filesystem operations. Clean up any on-disk artifacts your test creates when it is not. 

3) Additional development information

- Code style and linting
    - Coding style and quality:
    - Prefer clean, concise, and readable code.
    - Prefer solutions that decrease cognitive complexity and improve readability over guards and robustness.
    - The addition of a test asserting a module is importable coupled with simple `import module` clauses are always
      preferred to import guarding.
    - Avoid using modules that are not available on all target platforms
    - Linting: CI runs pylint across all tracked *.py files. Locally, after uv tool install pylint, you can run: uv tool
      run pylint $(git ls-files '*.py') on Linux/macOS, or for PowerShell: git ls-files "*.py" | ForEach-Object { $_ } |
      ForEach-Object { uv tool run pylint $_ }
    - Formatting: black is listed in the dev group. If you use uv: uv sync --group dev, then run: uv run black src test

- CLI and orchestration
  - The primary user entry point is src/denoise.py, which exposes a Typer CLI (see cli() at the bottom). It orchestrates:
    1) Export via darktable-cli with a stage-one XMP.
    2) Denoising stage, preparing an intermediate TIFF.
    3) Optional RL deblur via gmic, implemented as a DeblurStage in src/nind_denoise/pipeline.py.
    4) EXIF cloning via exiv2.

- External tools on Windows
  - PowerShell is the default shell in this workspace. When running examples from the README, prefer backslashes and/or doubled backslashes; some Windows shells do not like single forward slashes for paths.
  - Ensure darktable-cli.exe and gmic.exe are either on PATH, have their locations recorded in a configuration file, or passed explicitly via --dt and --gmic.

- Subprocess safety
  - When adding new pipeline stages, prefer the run_cmd helper in pipeline.py to standardize logging and Path-to-str conversion.
  - Use cwd where appropriate and keep all outputs within a designated output_dir. The current pipeline passes filenames and a working directory to avoid path confusion.

- Long/slow tests and integration checks
  - Anything invoking external tools (darktable-cli, gmic) or large models will be slow and environment-dependent. Keep these under an integration marker or skip them by default; use -m integration to opt-in, or -k to exclude. Example markers can be added in pytest.ini when this is formalized.

- Configuration files
  - Operation and model configurations (YAML) live under src/config/ and src/nind_denoise/configs/. read_config in src/denoise.py merges overrides and supports a nightmode flag that moves specific operations between stages.
  - If you add new operations, ensure your YAML updates are reflected in tests that parse and validate the XMP manipulation.

- Debugging tips
  - Use --debug to preserve intermediate files from denoise.py; otherwise intermediates are cleaned up.
  - Use --verbose to surface additional logging from both CLI and pipeline stages.
  - If exiv2 raises on metadata cloning, check that input files actually have EXIF chunks (dummy artifacts won’t).

- Cross-platform notes
  - The repository is used on Windows, MacOS, and Linux. Keep path handling via pathlib whenever possible and do not hardcode separators. Subprocess args are passed as lists of strings to avoid shell quoting issues.

- Contributing checklist
  - New code paths should have unit tests that do not require external binaries.
  - Keep imports local/dynamic when crossing src layout boundaries in tests.
  - Validate that black formatting and basic pytest smoke tests pass locally before opening a PR.

End of file.
