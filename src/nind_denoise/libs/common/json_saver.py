# -*- coding: utf-8 -*-

from typing import Set

from nind_denoise.libs.common import utilities


class JSONSaver:
    def __init__(
        self,
        jsonfpath,
        step_type: str = "step",
        default=None,
    ):
        if default is None:
            default = {"best_val": {}}
        self.best_key_str = f"best_{step_type}"  # best step/epoch #
        self.jsonfpath = jsonfpath
        self.results = utilities.jsonfpath_load(jsonfpath, default=default)
        if self.best_key_str not in self.results:
            self.results[self.best_key_str] = {}

    def add_res(
        self,
        step: int,
        res: dict,
        minimize=True,
        write=True,
        val_type=float,
        epoch=None,
        rm_none=False,
        key_prefix="",
    ):
        """epoch is an alias for step
        Set rm_none to True to ignore zero values"""
        if step is None and epoch is not None:
            step = epoch
        if step is None:
            raise ValueError("JSONSaver.add_res: Must specify either step or epoch")
        if step not in self.results:
            self.results[step] = dict()
        if key_prefix != "":
            res_ = {}
            for akey, aval in res.items():
                res_[f"{key_prefix}{akey}"] = aval
            res = res_
        for akey, aval in res.items():
            if val_type is not None:
                aval = val_type(aval)
            self.results[step][akey] = aval
            if isinstance(aval, list):
                continue
            if rm_none and aval == 0:
                continue
            if (
                akey not in self.results["best_val"]
                and akey in self.results[self.best_key_str]
            ):  # works when best_val has been removed but best_step exists
                self.results["best_val"][akey] = self.results[
                    self.results[self.best_key_str][akey]
                ][akey]
            if (
                akey not in self.results[self.best_key_str]
                or akey not in self.results["best_val"]
                or (self.results["best_val"][akey] > aval and minimize)
                or (self.results["best_val"][akey] < aval and not minimize)
            ):
                self.results[self.best_key_str][akey] = step
                self.results["best_val"][akey] = aval
        if write:
            utilities.dict_to_json(self.results, self.jsonfpath)

    def write(self):
        utilities.dict_to_json(self.results, self.jsonfpath)

    def get_best_steps(self) -> Set[int]:
        return set(self.results[self.best_key_str].values())
