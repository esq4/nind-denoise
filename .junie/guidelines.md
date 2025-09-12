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
  - Tests import implementation either via dynamic import from src or by adjusting PYTHONPATH. See testing section for patterns we use.

2) Testing

- Test runner
  - We use pytest (declared under [dependency-groups].dev in pyproject.toml). Typical invocations:
    - Run everything: pytest -q
    - Run a single file: pytest -q tests\test_pipeline.py
    - Run a test by node id: pytest -q tests\test_pipeline.py::test_noop_deblur_runs
  - Warning: Running the entire suite may be slow and may require external tools, this is expected.

- Importing code from src in tests (without installing the package)
  - Because we haven’t mandated editable installs (pip install -e .), tests dynamically import modules from src using importlib to avoid PYTHONPATH hassles. Example pattern:
    
    import importlib.machinery
    import importlib.util
    import pathlib as p
    import sys

    path = str(p.Path(__file__).resolve().parents[1] / 'src' / 'nind_denoise' / 'pipeline.py')
    loader = importlib.machinery.SourceFileLoader('pipeline_local_demo', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # important when modules use decorators/metaprogramming
    loader.exec_module(mod)
    Context = mod.Context
    NoOpDeblur = mod.NoOpDeblur

  - In particular, inserting the module into sys.modules before exec_module is important for modules that use @dataclass and similar, to avoid NoneType __dict__ errors during import.

- Creating and running a simple test (demonstration)
  - We verified this process by creating a temporary demo test locally and running it via pytest. The test proves the test runner is configured correctly and shows how to import from src without installing the package.
  - Example content of the demo test we used:

    import importlib.machinery
    import importlib.util
    import pathlib as p
    import sys

    # load src/nind_denoise/pipeline.py
    path = str(p.Path(__file__).resolve().parents[1] / 'src' / 'nind_denoise' / 'pipeline.py')
    loader = importlib.machinery.SourceFileLoader('pipeline_local_demo', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    loader.exec_module(mod)
    Context = mod.Context
    NoOpDeblur = mod.NoOpDeblur

    def test_noop_deblur_runs(tmp_path):
        out = tmp_path / 'out.jpg'
        s2 = tmp_path / 's2.tif'
        out.write_bytes(b'')
        s2.write_bytes(b'')
        ctx = Context(
            outpath=out,
            stage_two_output_filepath=s2,
            sigma=1,
            iteration='5',
            quality='90',
            cmd_gmic='gmic',  # not used by NoOpDeblur
            output_dir=tmp_path,
            verbose=True,
        )
        NoOpDeblur().execute(ctx)

  - Run just this test file to keep CI/dev iterations fast:
    pytest -q tests\test_demo_temp.py

  - We created and executed this demo during preparation of this guide and confirmed it passes. We removed the temporary file after verifying it, per the instruction to leave no extra files in the repo.

- Adding new tests
  - Prefer placing tests under tests/ with descriptive names (e.g., test_pipeline_highlevel.py).
  - If your test needs code under src/ and you don’t want to install the project, use the dynamic import pattern above.
  - Keep tests hermetic: avoid relying on external binaries (darktable-cli, gmic) unless the test is explicitly an integration test. For unit tests, prefer dependency injection and/or NoOp stages to avoid subprocess calls.
  - Use tmp_path for filesystem operations. Clean up any on-disk artifacts your test creates unless tmp_path is used.


3) Additional development information

- Code style and linting
  - Black is included in dependencies; the codebase generally follows Black defaults. Run black . to format locally.
  - Pylint is available under the dev group. Use it pragmatically; some modules interface with external binaries and may require disables for subprocess patterns.

- CLI and orchestration
  - The primary user entry point is src/denoise.py, which exposes a Typer CLI (see cli() at the bottom). It orchestrates:
    1) Export via darktable-cli with a stage-one XMP.
    2) Denoising stage, preparing an intermediate TIFF.
    3) Optional RL deblur via gmic, implemented as a DeblurStage in src/nind_denoise/pipeline.py.
    4) EXIF cloning via exiv2.
  - denoise.py contains defensive imports to locate pipeline.py locally if the package is not installed, which mirrors the testing import techniques.

- External tools on Windows
  - PowerShell is the default shell in this workspace. When running examples from the README, prefer backslashes and/or doubled backslashes; some Windows shells do not like single forward slashes for paths.
  - Ensure darktable-cli.exe and gmic.exe are either on PATH or passed explicitly via --dt and --gmic.

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
  - The repository is used on Windows and Linux. Keep path handling via pathlib whenever possible and do not hardcode separators. Subprocess args are passed as lists of strings to avoid shell quoting issues.

- Contributing checklist
  - New code paths should have unit tests that do not require external binaries.
  - Keep imports local/dynamic when crossing src layout boundaries in tests.
  - Validate that black formatting and basic pytest smoke tests pass locally before opening a PR.

End of file.
