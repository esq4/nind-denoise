# Project Debug Rules (Non-Obvious Only)

## Critical Debugging Information

- **Pipeline Debugging**: Use `--debug` flag to keep intermediate files for analysis
- **Verbose Logging**: Enable with `-v` or `--verbose` for detailed operation tracking
- **External Tools**: darktable-cli and gmic paths may need manual specification if not auto-detected

## Hidden Log Locations

- Pipeline operations log to `logger` in `nind_denoise.pipeline.orchestrator`
- JobContext manages resource cleanup - check for leaked resources here

## Common Debugging Scenarios

- **Path Issues**: Windows paths require double backslashes or forward slashes
- **Dependency Errors**: External tools (darktable-cli, gmic) may not be in PATH
- **Resource Leaks**: Check JobContext finalization in pipeline operations

## Testing Gotchas

- Integration tests use `@pytest.mark.integration` and are excluded by default
- Test data required from `tests/test_raw/` directory for some tests