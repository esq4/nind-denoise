"""Configuration loading and options for nind-denoise.

Includes default config resource loading, computing valid extensions, and a
legacy read_config used by tests for operations/history manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
import logging
import io
import yaml

try:
    # Python 3.12+: importlib.resources.files
    from importlib.resources import files as PKG_FILES  # type: ignore
except Exception:  # pragma: no cover - fallback for older pythons (not expected here)  # pylint: disable=broad-exception-caught
    PKG_FILES = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------
# Public Options dataclass
# ---------------------------


@dataclass
class Options:
    """Typed options for internal pipeline execution.

    Note: CLI remains the stable public interface; this dataclass is used for
    internal orchestration and testing convenience.
    """

    output_path: Optional[Path]
    extension: str
    dt: Optional[Path]
    gmic: Optional[Path]
    quality: int
    nightmode: bool
    no_deblur: bool
    debug: bool
    sigma: int
    iterations: int
    verbose: bool


# ---------------------------
# Config loading and helpers
# ---------------------------


def _default_cli_yaml_text() -> str:
    """Load the default CLI config YAML bundled as package data.

    The file is expected at nind_denoise/configs/cli.yaml.
    """
    try:
        if PKG_FILES is None:
            raise RuntimeError("importlib.resources.files unavailable")
        cfg_path = PKG_FILES("nind_denoise").joinpath("configs/cli.yaml")
        return cfg_path.read_text(encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Falling back to in-module defaults for cli.yaml: %s", exc)
        # Provide a minimal inline default as a last resort
        return """
valid_extensions:
  - 3FR
  - ARW
  - SR2
  - SRF
  - CR2
  - CR3
  - CRW
  - DNG
  - ERF
  - FFF
  - MRW
  - NEF
  - NRW
  - ORF
  - PEF
  - RAF
  - RW2
"""


def load_cli_config(path: str | None = None) -> dict:
    """Load CLI configuration.

    If path is None, load from the packaged default. Otherwise read the given file path.
    """
    if path is None:
        text = _default_cli_yaml_text()
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return data


def _norm_ext(e: str) -> str:
    e = e.strip()
    if not e:
        return e
    e = e.lower()
    return "." + e if not e.startswith(".") else e


def compute_valid_extensions(cfg: dict) -> list[str]:
    exts: Iterable[str] = cfg.get("valid_extensions") or []
    return [_norm_ext(e) for e in exts if isinstance(e, str) and e.strip()]


# Precompute and expose valid_extensions for consumers
try:
    _cli_cfg = load_cli_config(None)
except Exception:  # pragma: no cover
    _cli_cfg = {}
valid_extensions: list[str] = compute_valid_extensions(_cli_cfg) or [
    ".3fr",
    ".arw",
    ".sr2",
    ".srf",
    ".cr2",
    ".cr3",
    ".crw",
    ".dng",
    ".erf",
    ".fff",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".rw2",
]


# ---------------------------
# Legacy operations.yaml loader (kept for tests/back-compat)
# ---------------------------

def read_config(
    config_path: str = "./src/config/operations.yaml",
    _nightmode: bool = False,
    verbose: bool = False,
) -> dict:
    """Read operations/model config from a YAML file.

    This preserves the legacy shape used by tests. It also supports a back-compat
    rename to dt_module_op in future revisions, but for now keeps existing keys.
    """
    with io.open(config_path, "r", encoding="utf-8") as instream:
        var = yaml.safe_load(instream)
    if _nightmode:
        if verbose:
            logger.info("Updating ops for nightmode ...")
        nightmode_ops = ["exposure", "toneequal"]
        var["operations"]["first_stage"].extend(nightmode_ops)
        for op in nightmode_ops:
            var["operations"]["second_stage"].remove(op)
    return var
