# Project Coding Rules (Non-Obvious Only)

## Critical Patterns Discovered

- **Pipeline Execution**: Always use `run_pipeline()` from `nind_denoise.pipeline` for image processing
- **Job Context**: Use `JobContext` class for managing pipeline state and resources
- **Custom Exceptions**: Import and use exceptions from `src/nind_denoise/exceptions.py`
- **Configuration**: Load configs via `Config` class in `nind_denoise.config.config`
- **Path Handling**: Use `pathlib.Path` consistently throughout the codebase

## Non-Standard Directory Structures

- Source code organized in `src/nind_denoise/` with submodules for common, libs, pipeline, train
- Config files stored in `src/nind_denoise/config/` (not root config directory)

## Hidden Dependencies

- PyTorch and related libraries are primary dependencies (`torch`, `torchaudio`, `torchvision`)
- External tools like darktable-cli and gmic may be required for full functionality

## Windows-Specific Gotchas

- Path handling requires double backslashes: `r"C:\path\to\file"`
- Forward slashes work but double backslashes are preferred on Windows