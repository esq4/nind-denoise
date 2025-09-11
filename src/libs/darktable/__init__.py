from __future__ import annotations

# Aggregator module for the darktable package, re-exporting symbols from
# submodules darktable.py, events.py, and types.py. Includes a fallback
# loader to support direct execution of this file via importlib.SourceFileLoader
# (as done in tests), where relative imports may not work normally.

from typing import Any
import importlib
import importlib.util
import importlib.machinery
import pathlib
import sys

try:
    # Normal package-relative imports
    from .darktable import Darktable, Preferences, Configuration, GetText, get_lua_runtime
    from .events import EVENT_NAMES, EventManager
    from .types import TYPE_NAMES, TypesNamespace, normalize_type as _normalize_type
except Exception:
    # Fallback: load sibling modules by file path and register under this package
    _pkg_dir = pathlib.Path(__file__).parent
    # Mark current module as a package to allow submodule imports
    this_mod = sys.modules[__name__]
    if not hasattr(this_mod, "__path__"):
        this_mod.__path__ = [str(_pkg_dir)]  # type: ignore[attr-defined]

    def _load(name: str, filename: str):
        path = str(_pkg_dir / filename)
        loader = importlib.machinery.SourceFileLoader(f"{__name__}.{name}", path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[loader.name] = mod
        loader.exec_module(mod)
        return mod

    _events = _load("events", "events.py")
    _types = _load("types", "types.py")
    _dark = _load("darktable", "darktable.py")

    # Re-exported symbols
    Darktable = _dark.Darktable
    Preferences = _dark.Preferences
    Configuration = _dark.Configuration
    GetText = _dark.GetText
    get_lua_runtime = _dark.get_lua_runtime

    EVENT_NAMES = _events.EVENT_NAMES
    EventManager = _events.EventManager

    TYPE_NAMES = _types.TYPE_NAMES
    TypesNamespace = _types.TypesNamespace
    _normalize_type = _types.normalize_type

__all__ = [
    "Darktable",
    "Preferences",
    "Configuration",
    "GetText",
    "get_lua_runtime",
]
