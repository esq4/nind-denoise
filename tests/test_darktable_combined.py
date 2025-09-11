import os
import sys
import pathlib
import platform
import pytest


# Shared loader used across tests

def _load_darktable_module():
    import importlib.machinery
    import importlib.util
    import sys as _sys

    path = str(
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "libs"
        / "darktable"
        / "__init__.py"
    )
    loader = importlib.machinery.SourceFileLoader("darktable_local", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[loader.name] = mod
    loader.exec_module(mod)
    return mod


# Constants from Darktable API surface checks
API_MODULES = [
    # modules under darktable.* according to docs index
    "collection",
    "configuration",  # already present
    "control",
    "database",
    "debug",
    "films",
    "gettext",  # already present
    "gui",
    "guides",
    # creators
    "new_format",
    "new_storage",
    "new_widget",  # already present
    "password",
    "preferences",  # already present
    # printing helpers
    "print",  # function
    "print_error",  # function
    "print_hinter",  # function
    "print_log",  # function
    "print_toast",  # function
    # event/lib/storage registry
    "register_event",  # function
    "destroy_event",  # function
    "register_lib",  # function
    "register_storage",  # already present
    "destroy_storage",  # already present
    # other helper modules
    "styles",
    "tags",
    "util",
]

TYPE_NAMES = [
    # a subset but representative; the implementation should expose all listed
    "_pdf_mode_t",
    "_pdf_pages_t",
    "avif_color_mode_e",
    "avif_compression_type_e",
    "avif_tiling_e",
    "comp_type_t",
    "dt_collection_filter_t",
    "dt_collection_properties_t",
    "dt_collection_rating_comperator_t",
    "dt_collection_sort_order_t",
    "dt_collection_sort_t",
    "dt_imageio_exr_compression_t",
    "dt_imageio_j2k_format_t",
    "dt_imageio_j2k_preset_t",
    "dt_imageio_module_format_data_avif",
    "dt_imageio_module_format_data_copy",
    "dt_imageio_module_format_data_exr",
    "dt_imageio_module_format_data_j2k",
    "dt_imageio_module_format_data_jpeg",
    "dt_imageio_module_format_data_jpegxl",
    "dt_imageio_module_format_data_pdf",
    "dt_imageio_module_format_data_pfm",
    "dt_imageio_module_format_data_png",
    "dt_imageio_module_format_data_ppm",
    "dt_imageio_module_format_data_tiff",
    "dt_imageio_module_format_data_webp",
    "dt_imageio_module_format_data_xcf",
    "dt_imageio_module_format_t",
    "dt_imageio_module_storage_data_disk",
    "dt_imageio_module_storage_data_email",
    "dt_imageio_module_storage_data_gallery",
    "dt_imageio_module_storage_data_latex",
    "dt_imageio_module_storage_data_piwigo",
    "dt_imageio_module_storage_t",
    "dt_lib_collect_mode_t",
    "dt_lib_collect_params_rule_t",
    "dt_lighttable_layout_t",
    "dt_lua_align_t",
    "dt_lua_backgroundjob_t",
    "dt_lua_cairo_t",
    "dt_lua_ellipsize_mode_t",
    "dt_lua_film_t",
    "dt_lua_image_t",
    "dt_lua_lib_t",
    "dt_lua_orientation_t",
    "dt_lua_snapshot_t",
    "dt_lua_tag_t",
    "dt_lua_view_t",
    "dt_pdf_stream_encoder_t",
    "dt_style_item_t",
    "dt_style_t",
    "dt_ui_container_t",
    "dt_ui_panel_t",
    "hint_t",
    "lua_box",
    "lua_button",
    "lua_check_button",
    "lua_combobox",
    "lua_container",
    "lua_entry",
    "lua_file_chooser_button",
    "lua_label",
    "lua_os_type",
    "lua_pref_type",
    "lua_section_label",
    "lua_separator",
    "lua_slider",
    "lua_stack",
    "lua_text_view",
    "lua_widget",
    "snapshot_direction_t",
]

EVENT_NAMES = [
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
]


# Shared helpers for file paths and diagnostics

def _raw_sample_path():
    return (
        pathlib.Path(__file__).resolve().parent
        / "test_raw"
        / "_3317081.jpg"
    )


def _default_darktable_root():
    home = pathlib.Path.home()
    if os.name == "nt":
        return home / "AppData" / "Local" / "darktable"
    if sys_platform := platform.system().lower():
        if "darwin" in sys_platform or "mac" in sys_platform:
            # Prefer Application Support, then Preferences
            for p in [
                home / "Library" / "Application Support" / "darktable",
                home / "Library" / "Preferences" / "darktable",
            ]:
                if p.exists():
                    return p
            return home / "Library" / "Application Support" / "darktable"
    return home / ".config" / "darktable"


# ---- Tests consolidated from multiple files ----


def test_import_and_basic_objects():
    darktable = _load_darktable_module()
    Darktable = darktable.Darktable
    dt = Darktable()
    # core attributes
    assert hasattr(dt, "preferences")
    assert hasattr(dt, "gettext")
    assert hasattr(dt, "configuration")
    # config has running_os and config_dir
    assert dt.configuration.running_os in {"windows", "linux", "macos"}
    assert isinstance(dt.configuration.config_dir, str)


def test_print_and_events():
    darktable = _load_darktable_module()
    Darktable = darktable.Darktable
    dt = Darktable()
    # print, print_error, print_log should exist
    dt.print("hello", 123)
    dt.print_error("oops")
    # print_log may be implemented as alias to print
    assert hasattr(dt, "print_log")
    dt.print_log("debug", 42)

    msgs = dt.get_messages()
    assert any("hello" in m for m in msgs)
    assert any("oops" in m for m in msgs)
    assert any("debug" in m for m in msgs)

    called = {}

    def cb(x):
        called["x"] = x

    dt.register_event("test", cb)
    dt.trigger_event("test", 7)
    assert called["x"] == 7


@pytest.mark.skipif(
    "lupa" not in sys.modules and pytest.importorskip("lupa"),
    reason="lupa not installed",
)
def test_lupa_runtime_and_require_exposes_dt():
    darktable = _load_darktable_module()
    get_lua_runtime = darktable.get_lua_runtime
    Darktable = darktable.Darktable

    dt = Darktable()
    lua = get_lua_runtime(dt)
    # require the module and call a few APIs from Lua
    lua.execute(
        """
        local dt = require("darktable")
        -- preferences
        dt.preferences.write("lua", "x", "integer", 5)
        assert(dt.preferences.read("lua", "x", "integer") == 5)
        -- printing
        dt.print("hi from lua")
        dt.print_log("log from lua")
        dt.print_error("err from lua")
        -- configuration
        assert(dt.configuration.running_os == dt.configuration.running_os)
        -- event
        local hit = {v = 0}
        dt.register_event("evt", function(n) hit.v = n end)
        dt.trigger_event("evt", 9)
        assert(hit.v == 9)
        -- minimal new_widget smoke test
        local btn = dt.new_widget("button") { label = "Go" }
        assert(btn.kind == "button")
        assert(btn.label == "Go")
        -- storage registry
        local stored = 0
        dt.register_storage("exp", "Example", function() stored = stored + 1 end, nil, function() return true end)
        assert(dt._storages["exp"].label == "Example")
        dt.destroy_storage("exp")
        assert(dt._storages["exp"] == nil)
        """
    )


def test_new_widget_factory_minimal_python_side():
    darktable = _load_darktable_module()
    Darktable = darktable.Darktable

    dt = Darktable()
    # new_widget returns a constructor callable
    ctor = dt.new_widget("entry")
    w = ctor({"text": "abc"})
    assert w.kind == "entry"
    assert w.text == "abc"


def test_api_surface_presence_and_basic_behavior():
    darktable = _load_darktable_module()
    Darktable = darktable.Darktable
    dt = Darktable()

    # check presence for modules/functions
    for name in API_MODULES:
        assert hasattr(dt, name), f"missing dt.{name}"

    # check callables behave minimally
    dt.print("hello")
    dt.print_log("log")
    dt.print_error("error")
    dt.print_hinter("hint")
    dt.print_toast("toast")

    # register/destroy event should work
    hit = {"v": 0}
    dt.register_event("custom", lambda n: hit.update(v=n))
    dt.trigger_event("custom", 3)
    assert hit["v"] == 3
    dt.destroy_event("custom")
    dt.trigger_event("custom", 5)
    assert hit["v"] == 3  # unchanged after destroy

    # register_lib and storage should accept registration
    dt.register_lib("example_lib", object())
    dt.register_storage("st", "Storage", lambda: None)
    dt.destroy_storage("st")

    # new_format/new_storage should be callable
    assert callable(dt.new_format)
    assert callable(dt.new_storage)

    # submodules exist as simple namespaces/objects
    for sub in (dt.collection, dt.control, dt.database, dt.debug, dt.films, dt.gui, dt.guides, dt.password, dt.styles, dt.tags, dt.util):
        assert isinstance(sub, object)

    # types namespace must exist and contain all type names
    assert hasattr(dt, "types")
    for tname in TYPE_NAMES:
        assert hasattr(dt.types, tname), f"types missing {tname}"

    # events list/tuple or set should contain known event names
    assert hasattr(dt, "events")
    for ev in EVENT_NAMES:
        assert ev in dt.events


def test_new_format_registry():
    darktable = _load_darktable_module()
    dt = darktable.Darktable()
    dt.new_format("foo", ext=".foo", quality=90)
    assert "foo" in dt._formats
    assert dt._formats["foo"]["ext"] == ".foo"


def test_destroy_event_specific_callback():
    darktable = _load_darktable_module()
    dt = darktable.Darktable()
    a, b = {"v": 0}, {"v": 0}

    def cb_a(n):
        a["v"] += n

    def cb_b(n):
        b["v"] += n

    dt.register_event("inc", cb_a)
    dt.register_event("inc", cb_b)
    dt.trigger_event("inc", 2)
    assert a["v"] == 2 and b["v"] == 2

    # remove only one callback
    dt.destroy_event("inc", cb_a)
    dt.trigger_event("inc", 3)
    assert a["v"] == 2 and b["v"] == 5


def test_find_by_basename_case_insensitive_and_export_from_id_and_path(tmp_path):
    darktable = _load_darktable_module()
    dt = darktable.Darktable()
    src = _raw_sample_path()
    assert src.is_file(), f"Missing test asset: {src}"

    img = dt.database.import_(str(src)) if hasattr(dt.database, "import_") else getattr(dt.database, "import")(str(src))
    assert img is not None

    # Case-insensitive find
    found1 = dt.database.find_image_by_basename(src.name.upper())
    found2 = dt.database.find_image_by_basename(src.name.lower())
    assert found1 is not None and found2 is not None

    # Export using id
    out_dir1 = tmp_path / "exports1"
    exported1 = dt.database.export(getattr(img, "id", None), str(out_dir1))
    assert exported1 and pathlib.Path(exported1).exists()

    # Export using path string
    out_dir2 = tmp_path / "exports2"
    exported2 = dt.database.export(getattr(img, "path", None), str(out_dir2))
    assert exported2 and pathlib.Path(exported2).exists()


def test_print_hinter_and_toast_store_messages():
    darktable = _load_darktable_module()
    dt = darktable.Darktable()
    dt.print_hinter("hint msg")
    dt.print_toast("toast msg")
    msgs = dt.get_messages()
    assert any("hint msg" in m for m in msgs)
    assert any("toast msg" in m for m in msgs)


def test_new_storage_alias_registers():
    darktable = _load_darktable_module()
    dt = darktable.Darktable()
    stored = {"n": 0}

    def store():
        stored["n"] += 1

    dt.new_storage("alias", "Alias Storage", store)
    assert "alias" in dt._storages
    assert dt._storages["alias"]["label"] == "Alias Storage"


def test_database_imports_raw_image():
    darktable = _load_darktable_module()
    dt = darktable.Darktable()

    # Listen for post-import-image event
    hit = {"got": False, "id": None}
    dt.register_event(
        "post-import-image",
        lambda img: hit.update(got=True, id=getattr(img, "id", None)),
    )

    src = _raw_sample_path()
    assert src.is_file(), f"Missing test asset: {src}"

    img = dt.database.import_(str(src)) if hasattr(dt.database, "import_") else getattr(dt.database, "import")(str(src))

    assert img is not None, "Import did not return an image object"
    assert os.path.isabs(getattr(img, "path", "")), "Image path should be absolute"
    assert os.path.basename(getattr(img, "path", "")) == src.name
    # Registry should have this path
    assert any(rec.get("path") == str(src.resolve()) for rec in dt._images.values())
    # Event should have fired
    assert hit["got"] is True
    assert isinstance(hit["id"], int)


def test_database_exports_imported_image(tmp_path):
    darktable = _load_darktable_module()
    dt = darktable.Darktable()

    src = _raw_sample_path()
    img = getattr(dt.database, "import")(str(src))

    # Find by basename
    found = dt.database.find_image_by_basename(src.name)
    assert found is not None

    # Export to directory
    out_dir = tmp_path / "exports"
    out_path = dt.database.export(found, str(out_dir))
    assert out_path is not None
    out_file = pathlib.Path(out_path)
    assert out_file.exists() and out_file.is_file()
    assert out_file.name == src.name
    # Size should be non-zero
    assert out_file.stat().st_size > 0


def test_preferences_read_write_and_register():
    darktable = _load_darktable_module()
    Darktable = darktable.Darktable
    dt = Darktable()
    # read/write with coercion
    dt.preferences.write("mod", "a", "integer", 3)
    assert dt.preferences.read("mod", "a", "integer") == 3
    assert dt.preferences.read("mod", "a", "string") == "3"

    dt.preferences.write("mod", "b", "bool", True)
    assert dt.preferences.read("mod", "b", "bool") is True

    # register a preference (metadata only)
    assert hasattr(dt.preferences, "register")
    dt.preferences.register("mod", "tool_path", "file", "tool path", "path to tool", "")
    assert any(p[1] == "tool_path" for p in dt.preferences._registered)


def test_darktable_library_diagnostic_stats():
    darktable = _load_darktable_module()
    dt = darktable.Darktable()

    root = _default_darktable_root()
    # If there is no local darktable installation, skip this diagnostic test
    if not root.exists():
        pytest.skip(f"No Darktable directory found at {root}")

    # try to find a db to decide whether to skip
    has_db = any((root / name).exists() for name in ("library.db", "data.db")) or any(
        p.suffix == ".db" for p in root.glob("*.db")
    )
    if not has_db:
        pytest.skip(f"No Darktable database found under {root}")

    stats = dt.database.get_library_stats(str(root))
    assert isinstance(stats, dict)
    assert "db_path" in stats and pathlib.Path(stats["db_path"]).exists()
    assert "images_total" in stats and isinstance(stats["images_total"], int)
    assert "films_total" in stats and isinstance(stats["films_total"], int)

    # Print some human-friendly diagnostics for visibility in test logs
    dt.print(
        "Darktable diagnostic:",
        f"root={stats.get('root_dir')}",
        f"db={stats.get('db_path')}",
        f"images={stats.get('images_total')}",
        f"films={stats.get('films_total')}",
        f"tags={stats.get('tags_total', 'n/a')}",
    )
