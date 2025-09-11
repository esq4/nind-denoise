from __future__ import annotations

from typing import Any, Tuple

__all__ = [
    "TYPE_NAMES",
    "TypesNamespace",
    "normalize_type",
]

# A selection of documented type names (all exposed for attribute presence)
TYPE_NAMES: Tuple[str, ...] = (
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
)


def normalize_type(t: str) -> str:
    t = (t or "").strip().lower()
    if t in ("int", "integer"):
        return "integer"
    if t in ("float", "number", "double"):
        return "float"
    if t in ("bool", "boolean"):
        return "bool"
    return "string"


class TypesNamespace:
    """Container exposing documented type names as attributes.

    The attributes are just sentinel objects so that client code can
    reference them (e.g., for isinstance/identity checks in tests).
    """

    def __init__(self, names: Tuple[str, ...]):
        for n in names:
            setattr(self, n, object())
