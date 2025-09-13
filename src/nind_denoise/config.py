"""Configuration loading and options for nind-denoise.

Includes default config resource loading, computing valid extensions, and a
legacy read_config used by tests for operations/history manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Sequence
import logging
import io
import os
import platform
import shutil
import subprocess  # re-exportable for monkeypatching
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
    """Load the default config YAML bundled as package data.

    The file is expected at nind_denoise/configs/config.yaml.
    """
    try:
        if PKG_FILES is None:
            raise RuntimeError("importlib.resources.files unavailable")
        cfg_path = PKG_FILES("nind_denoise").joinpath("configs/config.yaml")
        return cfg_path.read_text(encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Falling back to in-module defaults for config.yaml: %s", exc)
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

# Default external tool locations (used when not provided via CLI and if not on PATH)
tools:
  windows:
    darktable: "C:\\Program Files\\darktable\\bin\\darktable-cli.exe"
    gmic: "C:\\Program Files\\gmic\\gmic.exe"
  posix:
    darktable: "/usr/bin/darktable-cli"
    gmic: "/usr/bin/gmic"

# Default operations split across stages; can be customized per setup
operations:
  first_stage:
    - demosaic
    - whitebalance
    - exposure
    - denoiseprofile
    - colorin
    - flip
  second_stage:
    - colorout
    - sharpen
    - toneequal
  overrides: {}

# Models used by the pipeline
models:
  nind_generator_650.pt:
    path: "models/nind/generator_650.pt"
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


def get_tool_executable_candidates() -> dict[str, list[str]]:
    """Return executable name candidates for external tools.

    Keys:
      - "darktable_cli": names for Darktable CLI
      - "gmic": names for GMIC CLI

    Cross-platform: includes both base and .exe variants. shutil.which will
    ignore non-existent forms on the current platform.
    """
    return {
        "darktable_cli": ["darktable-cli", "darktable-cli.exe"],
        "gmic": ["gmic", "gmic.exe"],
    }


@dataclass
class Tools:
    darktable: Path
    gmic: Optional[Path]


def _which(name: str) -> Optional[Path]:
    found = shutil.which(name)
    return Path(found) if found else None


def _platform_key() -> str:
    sys_plat = platform.system().lower()
    return "windows" if "windows" in sys_plat else "posix"


def run_cmd(args: Iterable[Path | str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess command with logging and optional cwd.

    Exposed here to give tests a single place to monkeypatch process execution.
    """
    cmd = [str(a) for a in args]
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=None if cwd is None else str(cwd), check=True)


def resolve_tools(
    dt_opt: Optional[Path] | None, gmic_opt: Optional[Path] | None
) -> Tools:
    """Resolve external tool paths for darktable-cli and gmic using packaged config.

    - If explicit Path provided, use it.
    - Else, take platform-specific defaults from config.yaml (tools.windows/posix).
    - Fallback to PATH search via candidate executable names.
    - Raise ExternalToolNotFound if darktable-cli cannot be resolved.
    """
    cfg = load_cli_config(None)
    tools_cfg = (cfg.get("tools") or {}).get(_platform_key(), {})

    # From CLI args or config
    darktable: Optional[Path] = dt_opt or (
        Path(os.path.expanduser(str(tools_cfg.get("darktable"))))
        if tools_cfg.get("darktable")
        else None
    )
    gmic: Optional[Path] = gmic_opt or (
        Path(os.path.expanduser(str(tools_cfg.get("gmic"))))
        if tools_cfg.get("gmic")
        else None
    )

    # PATH fallback for darktable
    if not darktable or not Path(darktable).exists():
        for name in get_tool_executable_candidates().get("darktable_cli", []):
            p = _which(name)
            if p:
                darktable = p
                break
    if not darktable or not Path(darktable).exists():
        from .exceptions import ExternalToolNotFound
        raise ExternalToolNotFound(
            "darktable-cli not found on PATH or configured path invalid"
        )

    # PATH fallback for gmic (optional)
    if not gmic or not Path(gmic).exists():
        for name in get_tool_executable_candidates().get("gmic", []):
            p = _which(name)
            if p:
                gmic = p
                break

    return Tools(darktable=Path(darktable), gmic=Path(gmic) if gmic else None)


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
# Operations/models loader (packaged config)
# ---------------------------

def read_config(
    config_path: str | None = None,
    _nightmode: bool = False,
    verbose: bool = False,
) -> dict:
    """Read operations/model config from a YAML file or packaged config.

    If config_path is None, load the packaged nind_denoise/configs/config.yaml.
    Applies the nightmode adjustment to move selected ops between stages.
    """
    if config_path is None:
        var = load_cli_config(None)
    else:
        text = Path(config_path).read_text(encoding="utf-8")
        var = yaml.safe_load(text) or {}

    if _nightmode:
        if verbose:
            logger.info("Updating ops for nightmode ...")
        nightmode_ops = ["exposure", "toneequal"]
        var.setdefault("operations", {}).setdefault("first_stage", []).extend(nightmode_ops)
        second = var.setdefault("operations", {}).setdefault("second_stage", [])
        var["operations"]["second_stage"] = [op for op in second if op not in nightmode_ops]
    return var
