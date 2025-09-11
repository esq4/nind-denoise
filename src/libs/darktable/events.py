from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Optional

__all__ = [
    "EVENT_NAMES",
    "EventManager",
]

# Documented Darktable Lua event names (for discoverability/testing)
EVENT_NAMES: Tuple[str, ...] = (
    "darkroom-image-history-changed",
    "darkroom-image-loaded",
    "exit",
    "global_toolbox-grouping_toggle",
    "global_toolbox-overlay_toggle",
    "image-group-information-changed",
    "inter-script-communication",
    "intermediate-export-image",
    "mouse-over-image-changed",
    "pixelpipe-processing-complete",
    "post-import-film",
    "post-import-image",
    "pre-import",
    "selection-changed",
    "shortcut",
    "view_changed",
)


class EventManager:
    """Simple event/observer registry used by Darktable stub."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., Any]]] = {}

    def register(self, name: str, callback: Callable[..., Any]) -> None:
        self._handlers.setdefault(name, []).append(callback)

    def destroy(self, name: str, callback: Optional[Callable[..., Any]] = None) -> None:
        if callback is None:
            self._handlers.pop(name, None)
            return
        lst = self._handlers.get(name)
        if not lst:
            return
        try:
            lst.remove(callback)
        except ValueError:
            pass
        if not lst:
            self._handlers.pop(name, None)

    def trigger(self, name: str, *args: Any, **kwargs: Any) -> None:
        for cb in self._handlers.get(name, []):
            cb(*args, **kwargs)
