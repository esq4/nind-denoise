from __future__ import annotations

import copy
import logging
from pathlib import Path

from bs4 import BeautifulSoup

from .config import read_config

logger = logging.getLogger(__name__)


def build_xmp(xmp_text: str, stage: int, *, verbose: bool = False) -> str:
    """Return a stage-specific XMP string from an input XMP string.

    - stage=1: keep only first_stage ops; disable flip.
    - stage=2: keep only second_stage ops (and any overrides); update iop order.
    """
    cfg = read_config(verbose=verbose)
    operations = cfg["operations"]

    sidecar = BeautifulSoup(xmp_text, "xml")
    history = sidecar.find("darktable:history")
    if history is None:
        raise ValueError()

    history_org = copy.copy(history)

    def _stage1() -> None:
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

    def _stage2() -> None:
        history.replace_with(history_org)
        history_ops2 = history_org.find_all("rdf:li")
        for op in reversed(history_ops2):
            if (
                op["darktable:operation"] not in operations["second_stage"]
                and op["darktable:operation"] in operations["first_stage"]
            ):
                op.extract()
                if verbose:
                    logger.debug("--removed: %s", op["darktable:operation"])
            elif op["darktable:operation"] in operations.get("overrides", {}):
                for key, val in operations["overrides"][
                    op["darktable:operation"]
                ].items():
                    op[key] = val
            if verbose:
                logger.debug(
                    "default: %s %s",
                    op["darktable:operation"],
                    op.get("darktable:enabled"),
                )
        description = sidecar.find("rdf:Description")
        if description:
            description["darktable:iop_order_version"] = "5"
            if description.has_attr("darktable:iop_order_list"):
                description["darktable:iop_order_list"] = (
                    description["darktable:iop_order_list"]
                    .replace("colorin,0,", "")
                    .replace("demosaic,0", "demosaic,0,colorin,0")
                )

    if stage == 1:
        _stage1()
    else:
        _stage2()

    return sidecar.prettify()


def write_xmp_file(
    src_xmp_path: Path, dst_xmp_path: Path, stage: int, *, verbose: bool = False
) -> None:
    """Read XMP from src, transform for stage, and write to dst."""
    if not src_xmp_path.exists():
        raise FileNotFoundError(str(src_xmp_path))
    xmp_text = src_xmp_path.read_text(encoding="utf-8")
    out_text = build_xmp(xmp_text, stage, verbose=verbose)
    dst_xmp_path.unlink(missing_ok=True)
    dst_xmp_path.write_text(out_text, encoding="utf-8")


def parse_darktable_history_stack(
    _input_xmp: Path, config: dict, verbose: bool = False
) -> None:  # pragma: no cover
    """Removed in favor of build_xmp/write_xmp_file.

    This function used to parse a RAW sidecar and write .s1.xmp/.s2.xmp files.
    It has been removed; please call build_xmp or write_xmp_file instead, or use
    ExportOperation.write_xmp_file from the pipeline.
    """
    raise RuntimeError(
        "parse_darktable_history_stack has been removed. Use build_xmp()/write_xmp_file() or ExportOperation.write_xmp_file() instead."
    )
