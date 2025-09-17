# Project Architecture Rules (Non-Obvious Only)

## Hidden Architectural Patterns

- **Pipeline Orchestration**: Centralized in `nind_denoise.pipeline.orchestrator`
- **Job Context Management**: Used throughout pipeline for resource handling
- **Stage-Based Processing**: Image processing happens in distinct stages with intermediate files

## Component Coupling

- **Pipeline Components**: Tight coupling between deblur, denoise, and export stages
- **Config Dependency**: Most components rely on `nind_denoise.config.Config`
- **Path Handling**: Consistent use of `pathlib.Path` across all modules

## Performance Considerations

- **Acceleration Layers**: Multiple hardware acceleration paths (CPU, NVIDIA, Intel XPU)
- **Intermediate Files**: Used for stage separation but can be cleaned up with `--debug`

## Critical Dependencies

- **PyTorch Ecosystem**: Core processing relies on `torch`, `torchaudio`, `torchvision`
- **External Tools**: darktable-cli and gmic integrate into the pipeline
- **Model Loading**: Pretrained models from specific directory structure

## Testing Architecture

- **Isolation Strategy**: Tests mock external tool calls to avoid dependency issues
- **Data Requirements**: Some tests require specific raw image files in `tests/test_raw/`