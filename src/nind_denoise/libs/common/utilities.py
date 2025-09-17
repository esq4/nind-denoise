import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess


def checksum(fpath, htype="sha1"):
    if htype == "sha1":
        h = hashlib.sha1()
    elif htype == "sha256":
        h = hashlib.sha256()
    else:
        raise NotImplementedError(htype)
    with open(fpath, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cp(inpath, outpath):
    """Copy a file, preferring reflink when available (POSIX).

    Falls back to shutil.copy2 if the 'cp' binary is not available.
    Raises on copy errors instead of silently continuing.
    """
    try:
        subprocess.run(("cp", "--reflink=auto", inpath, outpath), check=True)
    except FileNotFoundError:
        shutil.copy2(inpath, outpath)


def jsonfpath_load(fpath, default_type=dict, default=None):
    if not os.path.isfile(fpath):
        print(f"jsonfpath_load: warning: {fpath} does not exist, returning default")
        if default is None:
            return default_type()
        else:
            return default

    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {k if not k.isdigit() else int(k): v for k, v in x.items()}
        return x

    with open(fpath, "r") as f:
        return json.load(f, object_hook=jsonKeys2int)


def dict_to_json(adict, fpath):
    with open(fpath, "w") as f:
        json.dump(adict, f, indent=2)


def get_leaf(path: str) -> str:
    """Returns the leaf of a path, whether it's a file or directory followed by
    / or not."""
    return os.path.basename(os.path.relpath(path))


def get_root(fpath: str) -> str:
    """
    return root directory a file (fpath) is located in.
    """
    fpath = fpath.rstrip("/\\")
    return os.path.dirname(fpath)


def avg_listofdicts(listofdicts):
    """Compute the mean of numeric values for each key across a list of dicts.

    - If list is empty, returns an empty dict.
    - Keys are the union of keys across all dicts; missing values are ignored.
    """
    if not listofdicts:
        return {}
    res = {}
    for adict in listofdicts:
        for akey, aval in adict.items():
            res.setdefault(akey, []).append(aval)
    return {akey: statistics.mean(vals) for akey, vals in res.items()}


def list_of_tuples_to_csv(listoftuples, heading, fpath):
    with open(fpath, "w") as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(heading)
        for arow in listoftuples:
            csvwriter.writerow(arow)


def filesize(fpath):
    return os.stat(fpath).st_size
