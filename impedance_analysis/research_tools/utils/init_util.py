# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:11:18 2023.

@author: j2cle

Generate the list of functions and classes to use for init.
Latest file structure to use for target:
    defect_code:
        defect_code.equations
        defect_code.functions
        defect_code.impedance_analysis
"""

import os
import sys
import importlib
import inspect as ins

from pathlib import Path
from shutil import copytree, ignore_patterns, rmtree

def print_init(target, pth=None, ignore=None):
    if pth is None:
        pth = Path.cwd()/target

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    ignore.insert(0, "__init__")

    filesurvey = []
    for row in os.walk(Path(pth)):  # Walks through current path
        for filename in row[2]:  # row[2] is the file name
            full_path: Path = Path(row[0]) / Path(filename)  # row[0]
            if full_path.stem not in ignore and full_path.suffix == ".py":
                filesurvey.append(full_path)

    modules = []
    for f in filesurvey:
        if f.is_file() and f.stem not in ignore:
            try:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                module = importlib.util.module_from_spec(spec)
                sys.modules[target] = module
                spec.loader.exec_module(module)
                modules.append([".".join(f.relative_to(pth).parts)[:-3], module])
            except ModuleNotFoundError:
                pass

    res = {}
    for targ in modules:
        ftmp = ins.getmembers(targ[1], ins.isfunction)
        ctmp = ins.getmembers(targ[1], ins.isclass)
        tmp = ftmp + ctmp
        if len(tmp) > 0:
            res[targ[0]] = sorted([t[0] for t in tmp if targ[1].__name__ == t[1].__module__], key=str.lower)

    for k, v in res.items():
        print(f"from .{k} import (", end="\n    ")
        print(*v, sep = ",\n    ", end=",\n)\n")
        print("")
    print("")
    print(f"__all__ = [", end="\n")
    for k, v in res.items():
        print("    \"", end="")
        print(*v, sep = "\",\n    \"", end="\",\n")
    print("]\n")


def manage_research_tools(source=None, base=None, ignore="archive", **kwargs):
    """
    action:
        copy, deep_copy, remove
    """
    if source is None:
        source = Path.cwd()
    if base is None:
        base = source.parent

    source = Path(source)
    base = Path(base)

    if isinstance(ignore, str):
        ignore = [ignore]
    pths = [f for f in base.iterdir() if f.is_dir() and f.stem not in ignore]

    pths.remove(source)

    ignore.insert(0, '__pycache*')
    for p in pths:
        if kwargs.get("remove", False):
            rmtree(p/p.stem/source.stem, ignore_errors=True)
        elif kwargs.get("deep_copy", False):
            rmtree(p/p.stem/source.stem, ignore_errors=True)
            copytree(
                source/source.stem,
                p/p.stem/source.stem,
                ignore=ignore_patterns(*ignore),
                dirs_exist_ok=True,
            )
        elif kwargs.get("copy", True):
            copytree(
                source/source.stem,
                p/p.stem/source.stem,
                ignore=ignore_patterns(*ignore),
                dirs_exist_ok=True,
            )


if __name__ == "__main__":
    from research_tools.functions import p_find, f_find

    target = "functions"
    files = f_find(p_find(target, base=Path.cwd().parent), re_filter=r"[^_][.]py$")
    ignores = [
        "unit_conversion",
        "chemistry",
        "nernstplank",
        "pso",
        "statistics",
        "init_util",
        ]

    # print_init(target, p_find(target, base=Path.cwd().parent), ignore=ignores)


    manage_research_tools()
