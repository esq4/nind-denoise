from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

from .exceptions import ExternalToolNotFound


@dataclass
class Tools:
    darktable: Path
    gmic: Optional[Path]


def _which(name: str) -> Optional[Path]:
    found = shutil.which(name)
    return Path(found) if found else None


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
