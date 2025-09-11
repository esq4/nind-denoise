from __future__ import annotations

import dataclasses
from dataclasses import field
import os
import sys
import sqlite3
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .events import EVENT_NAMES, EventManager
from .types import TYPE_NAMES, TypesNamespace, normalize_type

__all__ = [
    "Darktable",
    "Preferences",
    "Configuration",
    "GetText",
    "get_lua_runtime",
]


class Preferences:
    """In-memory preferences store that mimics dt.preferences API."""

    def __init__(self):
        # key: (module, name) -> (type, value)
        self._store: Dict[tuple[str, str], tuple[str, Any]] = {}
        # registered preferences metadata (module, name, type, label, tooltip, default)
        self._registered: List[tuple[str, str, str, str, str, Any]] = []

    def read(self, module: str, name: str, type_: str) -> Any:
        t = normalize_type(type_)
        key = (module, name)
        if key not in self._store:
            return None
        stored_t, val = self._store[key]
        # attempt type coercion
        if t == stored_t:
            return val
        try:
            if t == "integer":
                return int(val)
            if t == "float":
                return float(val)
            if t == "bool":
                if isinstance(val, str):
                    return val.lower() in ("1", "true", "yes", "on")
                return bool(val)
            # string
            return str(val)
        except Exception:
            return None

    def write(self, module: str, name: str, type_: str, value: Any) -> None:
        t = normalize_type(type_)
        # basic type validation/conversion
        if t == "integer":
            value = int(value)
        elif t == "float":
            value = float(value)
        elif t == "bool":
            if isinstance(value, str):
                value = value.lower() in ("1", "true", "yes", "on")
            else:
                value = bool(value)
        else:
            value = str(value)
        self._store[(module, name)] = (t, value)

    # Non-Darktable extension: register preference metadata for UIs
    def register(
        self,
        module: str,
        name: str,
        type_: str,
        label: str,
        tooltip: str,
        default: Any,
        *_,
        **__,
    ) -> None:
        # darktable supports many parameters depending on type; we capture the basics
        self._registered.append((module, name, type_, label, tooltip, default))


class GetText:
    """Very small gettext stub used by some Lua scripts."""

    def __init__(self):
        self._domains: Dict[str, str] = {}

    def bindtextdomain(self, domain: str, path: str) -> None:
        self._domains[domain] = path

    # Basic gettext-like API
    def gettext(self, msgid: str) -> str:
        return msgid

    def dgettext(self, domain: str, msgid: str) -> str:
        return msgid

    def ngettext(self, msgid: str, msgid_plural: str, n: int) -> str:
        return msgid if n == 1 else msgid_plural


@dataclasses.dataclass
class Configuration:
    running_os: str = field(
        default_factory=lambda: (
            "windows" if os.name == "nt" else ("macos" if sys.platform == "darwin" else "linux")
        )
    )
    config_dir: str = field(default_factory=lambda: os.path.expanduser("~"))


class _Namespace:
    """Minimal attribute container used to mimic dt.* submodules.

    This is intentionally lightweight; methods/fields can be added later
    if/when needed by tests or scripts.
    """

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class Darktable:
    """A minimal Python mimic of Darktable's Lua API.

    Provided capabilities:
    - preferences (read/write, register metadata)
    - gettext binding stubs
    - configuration fields (running_os, config_dir)
    - event registration/triggering
    - print helpers (print, print_error, print_log)
    - simple widgets via new_widget
    - storage registration/destroy
    """

    def __init__(self):
        self.preferences = Preferences()
        self.gettext = GetText()
        self.configuration = Configuration()
        # Event manager and public list of known event names
        self._event_mgr = EventManager()
        self.events = EVENT_NAMES
        # Messages printed via dt.print* variants
        self._messages: List[str] = []
        # Storage and format registries
        self._storages: Dict[str, Dict[str, Any]] = {}
        self._formats: Dict[str, Dict[str, Any]] = {}
        # Registered libs (UI/library plugins)
        self._libs: Dict[str, Any] = {}
        # Simple in-memory film roll and image registries
        self._films: Dict[int, Dict[str, Any]] = {}
        self._films_by_path: Dict[str, int] = {}
        self._images: Dict[int, Dict[str, Any]] = {}
        self._images_by_path: Dict[str, int] = {}
        self._next_film_id: int = 1
        self._next_image_id: int = 1
        # Expose common submodules as lightweight namespaces
        self.collection = _Namespace()
        self.control = _Namespace()
        self.database = _Namespace()
        self.debug = _Namespace()
        self.films = _Namespace()
        self.gui = _Namespace()
        self.guides = _Namespace()
        self.password = _Namespace()
        self.styles = _Namespace()
        self.tags = _Namespace()
        self.util = _Namespace()
        # Bind minimal database helpers used by tests
        setattr(self.database, "import", self.database_import)
        setattr(self.database, "import_", self.database_import)
        self.database.find_image_by_basename = self.database_find_image_by_basename  # type: ignore[attr-defined]
        self.database.export = self.database_export  # type: ignore[attr-defined]
        self.database.get_library_stats = self.database_get_library_stats  # type: ignore[attr-defined]
        # Types namespace exposing documented type names
        self.types = TypesNamespace(TYPE_NAMES)

    # Events
    def register_event(self, name: str, callback: Callable[..., Any]) -> None:
        self._event_mgr.register(name, callback)

    def destroy_event(self, name: str, callback: Optional[Callable[..., Any]] = None) -> None:
        """Unregister callbacks for an event.

        If callback is None, all callbacks for the event are removed.
        Otherwise, only the specified callback is removed if present.
        """
        self._event_mgr.destroy(name, callback)

    def trigger_event(self, name: str, *args: Any, **kwargs: Any) -> None:
        self._event_mgr.trigger(name, *args, **kwargs)

    # Database helpers --------------------------------------------------------
    def _normalize_path(self, path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return os.path.abspath(os.path.expanduser(str(path)))

    def _image_object(self, rec: Dict[str, Any]):
        return _Namespace(**rec)

    def _get_or_create_film(self, dir_path: str) -> Dict[str, Any]:
        d = self._normalize_path(dir_path)
        film_id = self._films_by_path.get(d)
        if film_id is not None:
            return self._films[film_id]
        film_id = self._next_film_id
        self._next_film_id += 1
        rec = {
            "id": film_id,
            "path": d,
            "name": os.path.basename(d) or d,
        }
        self._films[film_id] = rec
        self._films_by_path[d] = film_id
        # Fire post-import-film event when creating a new film roll
        self.trigger_event("post-import-film", self._image_object(rec))
        return rec

    def database_import(self, path: str):
        """Import an image path into the in-memory database.

        Returns a lightweight image object with attributes: id, path, filename, dirname, film_id, ext.
        """
        if not path:
            return None
        norm = self._normalize_path(path)
        # Fire pre-import event
        self.trigger_event("pre-import", norm)
        if not os.path.isfile(norm):
            return None
        # Check for existing
        existing_id = self._images_by_path.get(norm)
        if existing_id is not None:
            return self._image_object(self._images[existing_id])
        # Ensure film (directory) exists
        film = self._get_or_create_film(os.path.dirname(norm))
        img_id = self._next_image_id
        self._next_image_id += 1
        rec = {
            "id": img_id,
            "path": norm,
            "filename": os.path.basename(norm),
            "dirname": os.path.dirname(norm),
            "film_id": film["id"],
            "ext": os.path.splitext(norm)[1].lower(),
        }
        self._images[img_id] = rec
        self._images_by_path[norm] = img_id
        obj = self._image_object(rec)
        self.trigger_event("post-import-image", obj)
        return obj

    def database_find_image_by_basename(self, name: str):
        if not name:
            return None
        name_l = str(name).lower()
        for rec in self._images.values():
            if rec.get("filename", "").lower() == name_l:
                return self._image_object(rec)
        return None

    def database_export(self, image: Any, destination: str) -> Optional[str]:
        """Export (copy) an image to the given destination.

        destination may be a directory or a full path to a file.
        Returns the full destination file path on success, otherwise None.
        """
        if image is None or not destination:
            return None
        # Resolve source path from supported input types
        src: Optional[str] = None
        if isinstance(image, str):
            src = self._normalize_path(image)
        elif isinstance(image, int):
            rec = self._images.get(image)
            if rec:
                src = rec.get("path")
        else:
            src = getattr(image, "path", None)
            if src:
                src = self._normalize_path(src)
        if not src or not os.path.isfile(src):
            return None
        dest_path = Path(destination)
        if dest_path.exists() and dest_path.is_dir():
            dest_file = dest_path / os.path.basename(src)
        else:
            # If parent directory missing, create it
            if dest_path.suffix == "":
                # Treat as directory path when no extension is provided
                dest_path.mkdir(parents=True, exist_ok=True)
                dest_file = dest_path / os.path.basename(src)
            else:
                dest_file = dest_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_file)
        return str(dest_file)

    def database_get_library_stats(self, root_dir: Optional[str] = None) -> Dict[str, Any]:
        """Inspect a Darktable library database and return basic statistics."""

        def _default_root() -> Path:
            if self.configuration.running_os == "windows":
                return Path(os.path.expanduser("~")) / "AppData" / "Local" / "darktable"
            if self.configuration.running_os == "macos":
                # Try common macOS locations
                for p in [
                    Path.home() / "Library" / "Application Support" / "darktable",
                    Path.home() / "Library" / "Preferences" / "darktable",
                ]:
                    if p.exists():
                        return p
                return Path.home() / "Library" / "Application Support" / "darktable"
            # linux and others
            return Path.home() / ".config" / "darktable"

        root = Path(self._normalize_path(root_dir)) if root_dir else _default_root()
        # Candidate DB files (order of preference)
        candidates = [
            root / "library.db",
            root / "data.db",
        ]
        db_path: Optional[Path] = None
        for p in candidates:
            if p.exists() and p.is_file():
                db_path = p
                break
        if db_path is None:
            # If not present, look for any *.db as last resort
            for p in root.glob("*.db"):
                db_path = p
                break
        if db_path is None:
            raise FileNotFoundError(f"No Darktable database found under: {root}")

        def _table_exists(con: sqlite3.Connection, name: str) -> bool:
            cur = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            )
            return cur.fetchone() is not None

        stats: Dict[str, Any] = {
            "db_path": str(db_path),
            "root_dir": str(root),
            "images_total": 0,
            "films_total": 0,
        }
        con = sqlite3.connect(str(db_path))
        try:
            if _table_exists(con, "images"):
                stats["images_total"] = int(con.execute("SELECT COUNT(*) FROM images").fetchone()[0])
            if _table_exists(con, "film_rolls"):
                stats["films_total"] = int(con.execute("SELECT COUNT(*) FROM film_rolls").fetchone()[0])
            if _table_exists(con, "tags"):
                stats["tags_total"] = int(con.execute("SELECT COUNT(*) FROM tags").fetchone()[0])
        finally:
            con.close()
        return stats

    # Printing helpers
    def print(self, *args: Any) -> None:  # noqa: A003 - mimic API name
        msg = " ".join(str(a) for a in args)
        self._messages.append(msg)
        # Also echo to stdout for visibility
        print(msg)

    def print_log(self, *args: Any) -> None:
        # Mimic dt.print_log: treat as normal print in this stub
        self.print(*args)

    def print_error(self, *args: Any) -> None:
        msg = " ".join(str(a) for a in args)
        self._messages.append(msg)
        print(msg, file=sys.stderr)

    def print_hinter(self, *args: Any) -> None:
        """Mimic dt.print_hinter (show a hint in the UI).

        In this stub it behaves like dt.print and stores the message.
        """
        self.print(*args)

    def print_toast(self, *args: Any) -> None:
        """Mimic dt.print_toast (show a transient message).

        In this stub it behaves like dt.print and stores the message.
        """
        self.print(*args)

    def get_messages(self) -> List[str]:
        return list(self._messages)

    # Widgets -----------------------------------------------------------------
    class _Widget:
        def __init__(self, kind: str, props: Dict[str, Any]):
            self.kind = kind
            for k, v in props.items():
                # sanitize key to attribute name (best-effort)
                try:
                    setattr(self, str(k), v)
                except Exception:
                    pass

        def __repr__(self) -> str:  # pragma: no cover - debug helper
            return f"<Widget {self.kind} {self.__dict__}>"

    def new_widget(self, kind: str):
        """Return a widget-constructor callable used as dt.new_widget(kind){...}."""

        def _to_dict(props: Any) -> Dict[str, Any]:
            if props is None:
                return {}
            # Lupa LuaTable exposes .items()
            if hasattr(props, "items"):
                try:
                    return dict(props.items())
                except Exception:
                    pass
            # Fallback: try direct dict()
            try:
                return dict(props)
            except Exception:
                return {}

        def ctor(props: Any):
            return Darktable._Widget(kind, _to_dict(props))

        return ctor

    # Storage/format/lib registration -----------------------------------------
    def new_format(self, name: str, **kwargs: Any) -> None:
        """Register a new export format (stub)."""
        self._formats[name] = dict(kwargs)

    def new_storage(
        self,
        name: str,
        label: str,
        store: Callable[..., Any],
        parameters: Optional[Callable[..., Any]] = None,
        supported: Optional[Callable[..., Any]] = None,
        initialize: Optional[Callable[..., Any]] = None,
        widget: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Alias to register_storage to ease porting Lua code."""
        self.register_storage(
            name,
            label,
            store,
            parameters=parameters,
            supported=supported,
            initialize=initialize,
            widget=widget,
        )

    def register_lib(self, name: str, lib: Any) -> None:
        """Register a library/lib module (stub)."""
        self._libs[name] = lib

    def register_storage(
        self,
        name: str,
        label: str,
        store: Callable[..., Any],
        parameters: Optional[Callable[..., Any]] = None,
        supported: Optional[Callable[..., Any]] = None,
        initialize: Optional[Callable[..., Any]] = None,
        widget: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._storages[name] = {
            "label": label,
            "store": store,
            "parameters": parameters,
            "supported": supported,
            "initialize": initialize,
            "widget": widget,
        }

    def destroy_storage(self, name: str) -> None:
        self._storages.pop(name, None)


# Optional Lupa integration -------------------------------------------------


def get_lua_runtime(dt: Optional[Darktable] = None):
    """Create a Lupa LuaRuntime and preload this module as `require "darktable"`."""
    import importlib

    lupa = importlib.import_module("lupa")
    LuaRuntime = getattr(lupa, "LuaRuntime")

    if dt is None:
        dt = Darktable()

    lua = LuaRuntime(unpack_returned_tuples=True)
    # Expose Python darktable instance to Lua as a global
    lua.globals().dt = dt
    # Preload module so that `require("darktable")` returns the object
    lua.execute('package.preload["darktable"] = function() return dt end')
    return lua
