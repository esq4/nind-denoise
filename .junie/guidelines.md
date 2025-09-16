Project development guidelines for nind-denoise

This document captures project-specific build, configuration, testing, and development practices to help future
contributors work efficiently. It intentionally focuses on details specific to this repository rather than generic
Python advice.

1) Build and configuration

- Python/runtime
    - Requires Python >= 3.12 (pyproject.toml) and is actively developed/tested with modern Python (3.13 works as well).
    - Preferred workflow uses uv for virtual environments and dependency installation (see README for platform-specific
      commands).

- Dependencies and installation
    - See the project's README.md

- Source layout
    - This repo uses a src layout: Python code lives under src/. The package name is nind-denoise with module paths
      under
      src/nind_denoise/ and a top-level orchestration script at src/denoise.py.

2) Testing

- Test runner
    - We use pytest. Typical invocations:
        - Run everything: pytest -q
        - Run a single file: pytest -q tests\test_pipeline.py
        - Run a test by node id: pytest -q tests\test_pipeline.py::test_noop_deblur_runs
        - run the long-running tests: pytest -m integration tests\

- Importing code from src in tests (without installing the package)
    - Mandate editable installs
    - Prefer a shared tests\conftest.py that prepends the repository’s src directory to sys.path so individual tests
      don’t need per-file boilerplate.

- What to do if a test fails
    - If there are multiple failures, do root cause analysis before deciding what the problems are or how to fix them.
    - A failing test should always be fully-investigated. It is OK to stop what you are doing, seek human guidance, and
      to give it your full attention.
    - Existing code should **always** be fixed/improved to bring the code into compliance
    - **Never** should existing code be circumvented, shimmed, covered with a "fallback", or modified "not to use an
      eternal tool",
      or anything other action taken to "avoid" rather than "fix" a problem causing a test failure.

- Adding new tests
    - Tests should be small, targeted to one 'thing' only, fully documented, and coverage should be extended to every
      additional piece of code as it is written.
    - Prefer placing tests under tests/ with descriptive names (e.g., test_pipeline_highlevel.py).
    - If practicable, Use in-memory temporary files and directories for transient filesystem operations.
        - Clean up any on-disk artifacts your test creates when it is not.

3) Additional development information

- Coding style and quality:
    - Use pylint as a guide for improving code, not as a guide to figure out which messages to disable in pylint.
        - Do not switch off pylint messaging in order to improve the score it gives
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
    - The primary user entry point is src/denoise.py, which exposes a Typer CLI (see cli() at the bottom). It
      orchestrates:
        1) Export via darktable-cli with a stage-one XMP.
        2) Denoising stage, preparing an intermediate TIFF.
        3) Optional RL deblur via gmic, implemented as a DeblurStage in src/nind_denoise/pipeline.py.
        4) EXIF cloning via exiv2.

- External tools on Windows
    - PowerShell is the default shell in this workspace. When running examples from the README, prefer backslashes
      and/or doubled backslashes; some Windows shells do not like single forward slashes for paths.
    - Ensure darktable-cli.exe and gmic.exe have their locations recorded in a configuration file, or passed explicitly
      via --dt and --gmic.

- Subprocess safety
    - When adding new pipeline stages, prefer the run_cmd helper in pipeline.py to standardize logging and Path-to-str
      conversion.
    - Use cwd where appropriate and keep all outputs within a designated output_dir. The current pipeline passes
      filenames and a working directory to avoid path confusion.

- Long/slow tests and integration checks
    - Anything invoking external tools (darktable-cli, gmic) or large models will be slow and environment-dependent.
        - Keep these under an integration marker or skip them by default; use -m integration to opt-in, or -k to
          exclude.
        - Example markers can be added in pytest.ini when this is formalized.

- Configuration files
    - Operation and model configurations (YAML) live under src/config/ and src/nind_denoise/configs/.

- Cross-platform notes
    - The repository is used on Windows, MacOS, and Linux.
        - Keep path handling via pathlib whenever possible and do not hardcode separators.
        - Subprocess args are passed as lists of strings to avoid shell quoting issues.

- Contributing checklist
    - Validate that black formatting, pylint, and basic pytest smoke tests pass locally before opening a PR.

End of file.
