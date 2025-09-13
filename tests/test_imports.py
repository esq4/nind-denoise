def test_can_import_valid_extensions_from_package():
    from nind_denoise.pipeline import valid_extensions  # noqa: WPS433

    # Basic sanity: list is non-empty and normalized to lowercase with leading dot
    assert isinstance(valid_extensions, list) and valid_extensions, "valid_extensions should be a non-empty list"
    assert all(isinstance(e, str) for e in valid_extensions)
    assert all(e == e.lower() for e in valid_extensions), "Extensions should be lowercase"
    assert all(e.startswith(".") for e in valid_extensions), "Extensions should start with a dot"

    # A few common RAW extensions should be present (normalized)
    expected = {".nef", ".cr2", ".cr3", ".arw", ".dng"}
    assert expected.intersection(set(valid_extensions)), "Expected common RAW extensions to be present"


def test_can_import_denoise_image_from_package_root():
    # This should import the submodule via the package root
    from nind_denoise import denoise_image  # noqa: WPS433

    # The module should expose the entry function used by the pipeline
    assert hasattr(denoise_image, "run_from_args"), "denoise_image module should expose run_from_args()"
