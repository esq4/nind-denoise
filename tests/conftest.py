# Test configuration for the nind-denoise project
# Ensures the src/ layout is importable in all tests without per-file boilerplate.
import sys
import pathlib

_src = pathlib.Path(__file__).resolve().parents[1] / "src"
_src_str = str(_src)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)
