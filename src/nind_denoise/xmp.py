from __future__ import annotations

import copy
import io
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import yaml

logger = logging.getLogger(__name__)


def parse_darktable_history_stack(
    _input_xmp: Path, config: dict, verbose: bool = False
) -> None:
    """Parse and write stage-1 and stage-2 XMP sidecars according to config.

    This function is moved from nind_denoise.pipeline for separation of concerns.
    Behavior is preserved for compatibility with existing tests.
    """
    operations = config["operations"]
    with _input_xmp.open(encoding="utf-8") as f:
        sidecar_xml = f.read()
    sidecar = BeautifulSoup(sidecar_xml, "xml")
    history = sidecar.find("darktable:history")
    history_org = copy.copy(history)
    history_ops = history.find_all("rdf:li")
    history_ops.sort(key=lambda tag: int(tag["darktable:num"]))
    for op in reversed(history_ops):
        if op["darktable:operation"] not in operations["first_stage"]:
            op.extract()
            if verbose:
                logger.debug("--removed: %s", op["darktable:operation"])
        else:
            if op["darktable:operation"] == "flip":
                op["darktable:enabled"] = "0"
                if verbose:
                    logger.debug("default: %s", op["darktable:operation"])
    s1 = _input_xmp.with_suffix(".s1.xmp")
    s1.unlink(missing_ok=True)
    s1.write_text(sidecar.prettify(), encoding="utf-8")

    history.replace_with(history_org)
    history_ops = history_org.find_all("rdf:li")
    for op in reversed(history_ops):
        if (
            op["darktable:operation"] not in operations["second_stage"]
            and op["darktable:operation"] in operations["first_stage"]
        ):
            op.extract()
            if verbose:
                logger.debug("--removed: %s", op["darktable:operation"])
        elif op["darktable:operation"] in operations.get("overrides", {}):
            for key, val in operations["overrides"][op["darktable:operation"]].items():
                op[key] = val
        if verbose:
            logger.debug(
                "default: %s %s", op["darktable:operation"], op.get("darktable:enabled")
            )
    description = sidecar.find("rdf:Description")
    description["darktable:iop_order_version"] = "5"
    if description.has_attr("darktable:iop_order_list"):
        description["darktable:iop_order_list"] = (
            description["darktable:iop_order_list"]
            .replace("colorin,0,", "")
            .replace("demosaic,0", "demosaic,0,colorin,0")
        )
    s2 = _input_xmp.with_suffix(".s2.xmp")
    s2.unlink(missing_ok=True)
    s2.write_text(sidecar.prettify(), encoding="utf-8")
