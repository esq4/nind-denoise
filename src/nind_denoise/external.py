from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess as subprocess  # re-exportable for monkeypatching
import logging
from typing import Optional, Iterable

from .exceptions import ExternalToolNotFound

logger = logging.getLogger(__name__)


@dataclass
class Tools:
    darktable: Path
    gmic: Optional[Path]


def _which(name: str) -> Optional[Path]:
    found = shutil.which(name)
    return Path(found) if found else None


def run_cmd(args: Iterable[Path | str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess command with logging and optional cwd.

    Exposed here to give tests a single place to monkeypatch process execution.
    """
    cmd = [str(a) for a in args]
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=None if cwd is None else str(cwd), check=True)


def resolve_tools(dt_opt: Optional[Path] | None, gmic_opt: Optional[Path] | None) -> Tools:
    """Resolve external tool paths for darktable-cli and gmic.

    - If explicit Path provided, use it.
    - Else, search common executable names on PATH.
    - Raise ExternalToolNotFound if darktable-cli cannot be resolved.
    """
    darktable = dt_opt or _which("darktable-cli") or _which("darktable-cli.exe")
    if not darktable or not Path(darktable).exists():
        raise ExternalToolNotFound("darktable-cli not found on PATH or provided path invalid")
    gmic = gmic_opt or _which("gmic") or _which("gmic.exe")
    return Tools(darktable=Path(darktable), gmic=Path(gmic) if gmic else None)
