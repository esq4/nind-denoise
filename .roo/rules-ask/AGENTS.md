# Project Documentation Rules (Non-Obvious Only)

## Hidden Documentation Context

- **Source Code**: `src/` contains the main project code, not web application code
- **Configuration**: Config files are in `src/nind_denoise/config/` (non-standard location)
- **Models**: Pretrained models stored in `models/brummer2019/` directory

## Misleading Structures

- **Pipeline Directory**: `src/nind_denoise/pipeline/` contains core processing logic
- **Libs Directory**: `src/nind_denoise/libs/` contains external library integrations
- **Train Directory**: `src/nind_denoise/train/` contains model training code

## Important Context Not Evident from Structure

- **External Dependencies**: darktable-cli and gmic are optional but enhance functionality
- **Acceleration Support**: Works with CPU, NVIDIA GPU, Intel XPU/GPU (and potentially AMD)
- **Windows Path Handling**: Requires special handling for paths in CLI usage

## Command Usage Notes

- **CLI Entry Point**: `python3 src/denoise.py` is the main interface
- **Path Arguments**: Windows requires double backslashes or forward slashes
- **Output Control**: Use `--output-path`, `--extension` to control results