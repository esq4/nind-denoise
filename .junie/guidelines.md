# nind-denoise Development Guidelines

## Build/Configuration Instructions

### Environment Setup

- **Python Version**: Requires Python >=3.12
- **Package Manager**: Uses `uv` for dependency management and virtual environment
- **Project Structure**: Uses `src/` layout with packages in `src/nind_denoise/`

### Dependencies Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip with requirements
pip install -r requirements.in
```

### Key Dependencies

- **PyTorch Ecosystem**: torch~=2.8.0, torchvision~=0.23.0, torchaudio~=2.8.0
- **Computer Vision**: OpenCV (opencv-python, opencv-contrib-python), Pillow, imageio
- **Image Metadata**: exiv2 for EXIF/XMP handling
- **External Tools**: Designed to work with darktable-cli and gmic (optional for integration tests)

### Configuration System

The project uses YAML-based configuration (`src/nind_denoise/config/config.yaml`):

- **Models**: Neural network model paths and defaults
- **Tools**: Platform-specific paths for external tools (Windows/POSIX)
- **Operations**: Pipeline stage definitions (first_stage, second_stage, nightmode_ops)
- **File Extensions**: Supported RAW image formats (3FR, ARW, CR2, DNG, NEF, ORF, etc.)

### External Tools (Optional)

Integration tests require external image processing tools:

- **darktable-cli**: RAW image processing
- **gmic**: Advanced image filtering
- These are platform-specific and configured in config.yaml

## Testing Information

### Test Configuration

- **Framework**: pytest with pytest-cov for coverage
- **Configuration**: `pytest.ini` defines custom markers
- **Integration Tests**: Marked with `@pytest.mark.integration` for tests requiring external tools
- **Default Behavior**: Excludes integration tests by default (`addopts = -m "not integration"`)

### Running Tests

#### Basic Test Execution

```bash
# Run all unit tests (excludes integration tests)
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_utilities.py -v

# Run integration tests (requires external tools)
python -m pytest -m integration
```

#### Test Coverage

```bash
# Run tests with coverage report
python -m pytest --cov=src/nind_denoise --cov-report=html
```

### Adding New Tests

#### Example Test Structure

```python
"""
Simple test example following project patterns.
"""
import pytest
from pathlib import Path


def test_path_operations():
    """Test path operations common in image processing."""
    # Test basic path handling
    test_path = Path("image.ORF")
    assert test_path.suffix == ".ORF"

    # Test XMP sidecar files (common pattern)
    xmp_path = test_path.with_suffix(test_path.suffix + ".xmp")
    assert str(xmp_path) == "image.ORF.xmp"


@pytest.mark.integration
def test_external_tool():
    """Example integration test requiring external tools."""
    # This test would be skipped by default
    pass
```

#### Test Organization

- **Unit Tests**: Focus on individual components without external dependencies
- **Integration Tests**: Test external tool integration (darktable, gmic)
- **Test Data**: Located in `tests/test_raw/` with sample images and XMP files
- **Fixtures**: Defined in `tests/conftest.py`

### Verified Test Example

The following test demonstrates the working testing setup:

```bash
python -m pytest test_example.py -v
# Expected output: 2 tests pass in ~0.04s
```

<<<<<<< HEAD
### Handling Test Failures After Refactors

When major refactors break existing tests, the approach should focus on fixing underlying code issues rather than
implementing backwards compatibility:

#### Core Principles

1. **Align with Architectural Intent**: Fix tests to match the new architecture, don't preserve old APIs
2. **No Backwards Compatibility Layers**: Avoid stubs, facades, or compatibility interfaces that only address test
   failures
3. **Fix Root Causes**: Address underlying problems in the code, not symptoms in tests
4. **Update Test Expectations**: Modify tests to work with the new implementation patterns

#### Preferred Solutions for Common Refactor Issues

- **API Changes**: Update test code to use new constructor signatures and method calls
- **Module Reorganization**: Update imports to reference new module locations directly
- **Method Signature Changes**: Rename methods throughout codebase consistently
- **Missing Functions**: Transition tests to use new class-based APIs instead of standalone functions

#### Implementation Priority

1. **High Priority**: Config/API changes affecting many tests
2. **Medium Priority**: Implementation bugs and missing function transitions
3. **Low Priority**: Import errors and minor method signature issues

This approach ensures the codebase evolves cleanly without accumulating technical debt from backwards compatibility
layers.

=======
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
## Development Information

### Code Style and Formatting

- **Formatter**: Black (>=25.1.0) for consistent code formatting
- **Style**: Follows Black's opinionated formatting rules
- **IDE Integration**: PyCharm/IntelliJ configuration available in `.idea/misc.xml`

### Architectural Patterns

#### Pipeline Architecture

- **Abstract Base Classes**: Operations inherit from `Operation` ABC
- **Stage Pattern**: Pipeline divided into stages (denoise, deblur, export)
- **Context Objects**: `JobContext` dataclass carries state between stages
- **Environment Pattern**: `Config` + `JobContext` pattern for operation execution

#### Key Classes

```python
# Base operation pattern
class Operation(ABC):
    @abstractmethod
    def execute_with_env(self, cfg: Config, job_ctx: JobContext) -> None: ...

    @abstractmethod
    def verify_with_env(self, cfg: Config, job_ctx: JobContext) -> None: ...


# Typed context for stage execution
@dataclass
class JobContext:
    input_path: Path
    output_path: Path
    sigma: int = 1
    iterations: int = 10
    quality: int = 90
```

#### Machine Learning Patterns

- **Dataset Classes**: Multiple torch.utils.data.Dataset implementations
- **Data Augmentation**: Built-in noise injection, compression, cropping
- **Configuration-Driven**: YAML-based training configuration
- **Validation Split**: Dedicated ValidationDataset class
- **Testing Integration**: unittest.TestCase mixed with pytest

#### File Handling Patterns

- **Path Objects**: Extensive use of `pathlib.Path` throughout
- **XMP Sidecars**: RAW images paired with `.xmp` metadata files
- **Multi-Extension**: Support for complex extensions like `.ORF.xmp`
- **Stage Files**: Intermediate processing stages with suffix patterns (`_s1`, `_s2`)

### Error Handling

- **Custom Exceptions**: `ConfigurationError`, `ExternalToolNotFound`, `StageError`
- **Defensive Programming**: Try/catch blocks with fallbacks
- **Validation**: File existence checks before processing
- **Pragma Coverage**: `# pragma: no cover` for defensive code paths

### Import Patterns

- **Future Annotations**: `from __future__ import annotations` for forward references
- **Deferred Imports**: Imports within functions to avoid circular dependencies
- **Type Hints**: Comprehensive typing throughout codebase

### Development Best Practices

1. **Type Safety**: Use dataclasses and type hints extensively
2. **Configuration**: Keep settings in YAML, validate at startup
3. **Testing**: Separate unit tests from integration tests requiring external tools
4. **Path Handling**: Use `pathlib.Path` objects, not string paths
5. **Logging**: Use module-level loggers with `logging.getLogger(__name__)`
6. **External Tools**: Graceful degradation when optional tools unavailable

### Project-Specific Notes

- **RAW Processing**: Designed for camera RAW image denoising workflow
- **Neural Networks**: PyTorch-based denoising models with downloadable weights
- **Pipeline Stages**: Two-stage processing with configurable operations
- **Platform Support**: Windows and POSIX tool configurations
- **Image Formats**: Extensive RAW format support (Canon, Sony, Nikon, Olympus, etc.)

### Debugging Tips

- **Configuration Issues**: Check `config.yaml` tool paths and model availability
- **Integration Test Failures**: Verify external tools (darktable-cli, gmic) are installed
- **Path Problems**: Use `pathlib.Path` objects and check `.exists()` before processing
- **Import Errors**: Check for circular dependencies, use deferred imports if needed