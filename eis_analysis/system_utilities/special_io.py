# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import io
import re
import sys
import types
import inspect
import datetime
import traceback
import configparser
from typing import Any
from pathlib import Path
from collections.abc import Callable

import numpy as np
import scipy
import sympy as sp
import pandas as pd


def parse_file_info(
    paths,
    parser,
    keywords=None,
    get_all=False,
    by_date=True,
    by_time=True,
    by_count=True,
    by_sub_count=True,
):
    """
    Parse and filter files based on given criteria.
    Parameters:
    paths (list): List of file paths to parse.
    parser (function): Function to parse each file.
    keywords (str or list, optional): Keywords to filter the files. Defaults to None.
    get_all (bool, optional): If True, return all parsed data without filtering. Defaults to False.
    by_date (bool, optional): If True, filter by the latest date. Defaults to True.
    by_time (bool, optional): If True, filter by the latest time. Defaults to True.
    by_count (bool, optional): If True, filter by the highest counter. Defaults to True.
    by_sub_count (bool, optional): If True, filter by the highest sub counter. Defaults to True.
    Returns:
    pandas.DataFrame: Filtered data as a DataFrame.
    """
    res = []
    for pth in paths:
        try:
            res.append(parser(pth))
        except ValueError:
            print(f"Error parsing file: {pth}")
            continue
    res_all = pd.DataFrame(
        res, columns=["name", "date", "time", "counter", "sub counter", "path"], dtype=None
    )
    if get_all or all(not b for b in [by_date, by_time, by_count, by_sub_count]):
        return res_all
    res = []
    for _, gr in res_all.groupby(by=["name"], as_index=False, sort=False):
        if keywords is not None:
            if isinstance(keywords, str):
                keywords = [keywords]
            if not any(kw in gr["name"].to_numpy(copy=True)[0] for kw in keywords):
                continue
        if by_date:
            gr = gr.loc[gr["date"] == gr["date"].max()]
        if by_time:
            gr = gr.loc[gr["time"] == gr["time"].max()]
        if by_count:
            gr = gr.loc[gr["counter"] == gr["counter"].max()]
        if by_sub_count:
            gr = gr.loc[gr["sub counter"] == gr["sub counter"].max()]

        res.append(gr)
    res = pd.concat(res).reset_index(drop=True)

    return res


def eval_config_string(text):
    """
    Parse text into None, int, float, tuple, list, or dict via sympy parse_expr.
    Parse_expr will attempt to resolve simple math expressions but will revert
    to str if result doesn't resolve to a number.  Does not support complex
    numbers.

    text : str
        The string of text to be converted if possible

    Returns
    -------
    res : any, str
        Returns an int, float, or string (or a tuple, list, set, or dict of base
        base types). Returns text as str in case of error.
    """
    if not isinstance(text, str):
        return text
    brackets = [["(", ")"], ["[", "]"], ["{", "}"]]
    ends = re.findall(r"^.|.$", text.strip())

    if ends in brackets:
        items = re.findall(
            r"[^\,\n]*:?[\(\[\{][^\)\]\}]+[\)\]\}]|[^\,\n]+",
            text.strip()[1:-1],
        )
        if all([":" in t for t in items]) and ends == brackets[2]:
            res = {}
            for item in items:
                it = [i.strip() for i in item.split(":")]
                res[it[0]] = eval_config_string(":".join(it[1:])) if len(it) > 1 else None
        else:
            res = [eval_config_string(item.strip()) for item in items]
            if ends == brackets[0]:
                res = tuple(res)
            if ends == brackets[2]:
                try:
                    res = set(res)
                except TypeError as err:
                    print(f"Error creating set: {err}")
        return res
    if text.lower() == "none":
        return None
    elif bool(re.findall(r"\d", text)):
        try:
            res = sp.parse_expr(text, transformations=sp.parsing.sympy_parser.T[5])  # type: ignore
            return res
        except (TypeError, SyntaxError, KeyError, NameError):
            return text
    return text


def get_config(file, sections=None, **kwargs):
    """
    Get the necessary information from a configuration .ini file.

    Parameters
    ----------
    file : [str, Path]
        The path to the .ini file containing the configuration settings to be imported. If file is
        None, any kwargs are returned as the settings. This is only usefull for external functions
        which may pass None for other reasons.
    sections : list, optional
        Defines what sections of the .ini to import. If "all" is passed, function will create a
        dict of dicts, separating each section into its own dict of settings.
        If no section match, attempts to find sections which include the values provided by the
        list. If there are still no matches, the first section will be called and returned.
    kwargs : function, optional
        Pass additional items to be included in the configuration.  If the configuration
        is in the .ini file, they will be overwritten.

    Returns
    -------
    config_file : dict
        Returns a dict containing all settings imported from the .ini file
    """
    if sections is None:
        sections = ["base"]
    if file is None:
        return kwargs
    cp = configparser.ConfigParser()
    cp.optionxform = lambda option: option  # type: ignore
    # Load the configuration file
    if Path(file).is_file():
        cp.read(Path(file))
    elif (Path.cwd() / file).is_file():
        cp.read(Path.cwd() / file)
    elif (Path(file) / "config.ini").is_file():
        cp.read(Path(file) / "config.ini")
    else:
        cp.read(Path.cwd() / "config.ini")

    if isinstance(sections, str):
        if sections == "all":
            config_file = {}
            for sec in cp.sections():
                config_file[sec] = get_config(file, sections=[sec])
                config_file[sec] = {k: kwargs.get(k, v) for k, v in config_file[sec].items()}
            return config_file
        sections = [sections]

    checked_sec = [s_in for s_in in sections if s_in in cp.sections()]
    if checked_sec == []:
        checked_sec = [
            s_file
            for s_in in sections
            for s_file in cp.sections()
            if s_in.lower() in s_file.lower()
        ]
    if checked_sec == []:
        checked_sec = [cp.sections()[0]]
    config_file = {
        k: eval_config_string(v) for sc in checked_sec for k, v in dict(cp.items(sc)).items()
    }
    included = []
    if "include_sections" in config_file.keys():
        included = [
            get_config(file, [sec])
            for sec in config_file.pop("include_sections").split(", ")  # type: ignore
        ]
        for inc in included:
            config_file = {**config_file, **inc}

    return {**config_file, **kwargs}


def scrub_user_info(text: str) -> str:
    # Replace Windows user paths like C:\Users\username\
    text = re.sub(r"C:\\Users\\[^\\]+\\", r"C:\\Users\\<USER>\\", text)
    # Replace home directories (Linux/Mac)
    text = re.sub(r"/home/[^/]+/", r"/home/<USER>/", text)
    # Replace any obvious username in paths
    text = re.sub(r"([A-Za-z]:)?[\\/](Users|home)[\\/][^\\/]+", r"\1\2\\<USER>", text)
    return text


def summarize_array(
    arr: np.ndarray,
    shape: bool = False,
    dtype: bool = False,
    head: int = 0,
    tail: int = 0,
    arr_min_max: bool = False,
    col_min_max: bool = False,
    row_min_max: bool = False,
) -> str:
    """
    Summarize a numpy array in various ways.

    Parameters
    ----------
    arr : np.ndarray
        The array to summarize.
    shape : bool
        If True, include array shape and dtype.
    head : int
        If >0, show this many items/rows from the start.
    tail : int
        If >0, show this many items/rows from the end.
    arr_min_max : bool
        If True, include min and max of the whole array.
    col_min_max : bool
        If True, include min and max for each column (2D only).
    row_min_max : bool
        If True, include min and max for each row (2D only).

    Returns
    -------
    str
        Summary string.
    """
    name = type(arr).__name__
    arr = np.asarray(arr)
    details = []

    stats = []
    if shape:
        details.append(f"shape={arr.shape}")
    if dtype:
        details.append(f"dtype={arr.dtype}")
    details_str = ", ".join(details)

    # Body part
    body_str = ""

    if head or tail:
        body_parts = []
        h_arr = arr[:head]
        t_arr = arr[-tail:] if tail != 0 else np.array([])
        # If head and tail slices overlap or touch, show the full array
        if len(h_arr) + len(t_arr) >= len(arr):
            body_str = np.array2string(arr, separator=", ")
        else:
            if head:
                body_parts.append(np.array2string(h_arr, separator=", "))
            body_parts.append("...")
            if tail:
                body_parts.append(np.array2string(t_arr, separator=", "))
            body_str = ", ".join(body_parts)

    # Stats part
    stats_str = ""
    if arr.size > 0:
        stats = []
        if arr_min_max or (arr.ndim == 1 and (col_min_max or row_min_max)):
            stats.append(f"min: {np.min(arr)}, max: {np.max(arr)}")
        if arr.ndim >= 2:
            if col_min_max:
                stats.append(f"col min: {np.min(arr, axis=0)}, col max: {np.max(arr, axis=0)}")
            if row_min_max:
                stats.append(f"row min: {np.min(arr, axis=1)}, row max: {np.min(arr, axis=1)}")
        stats_str = "\n".join(stats)

    # Combine all parts
    result = ""
    if body_str:
        result = f"{name}([{body_str}])"
    if details_str:
        result = result[:-1] + f", {details_str})" if result else f"{name}({details_str})"
    if stats_str:
        result += "\n" + stats_str
    return result


def object_attributes_info(obj: Any) -> str:
    """Return public attributes and their values/types for an object."""
    lines = []
    for attr in dir(obj):
        if not attr.startswith("_"):
            try:
                val = getattr(obj, attr)
                if isinstance(val, np.ndarray):
                    lines.append(
                        f"{attr}: {summarize_array(val, shape=True, dtype=True, head=5, tail=5, arr_min_max=True)}"
                    )
                elif isinstance(val, (int, float, str, bool, type(None))):
                    lines.append(f"{attr}: {val}")
                elif callable(val):
                    lines.append(f"{attr}: {val.__name__}")
                else:
                    lines.append(f"{attr}: {type(val).__name__}")
            except Exception as exc:
                lines.append(f"{attr}: <error: {exc}>")
    return "\n".join(lines)


def item_info(items: dict[str, Any] | list[Any], *ignore_types: type) -> str:
    """
    Return a string summary of items, skipping any whose type is in ignore_types.

    Parameters
    ----------
    items : dict[str, Any] or list[Any]
        Items to summarize.
    *ignore_types : type
        Types to ignore (skip) in the output.

    Returns
    -------
    str
        Summary string for all items not ignored.
    """
    if not items:
        return ""

    if isinstance(items, dict):
        joiner = "\n"
        item_names = list(items.keys())
        items = list(items.values())
    else:
        joiner = ", "
        item_names = [""] * len(items)

    lines = []
    for name, item in zip(item_names, items):
        if ignore_types and isinstance(item, ignore_types):
            continue
        prefix = f"{name}: " if name else ""
        lines.append(f"{prefix}{item}")
    return joiner.join(lines)


def module_versions_info(modules: dict[str, Any] | list[Any]) -> str:
    """
    Return module version info as a string.

    Parameters
    ----------
    modules : dict[str, Any] or list[Any]
        Dictionary of module names to module objects, or list of module objects.

    Returns
    -------
    str
        Version info string.
    """
    if not modules:
        return ""

    if isinstance(modules, dict):
        joiner = "\n"
        module_names = list(modules.keys())
        modules = list(modules.values())
    else:
        joiner = ", "
        module_names = [""] * len(modules)

    lines = []
    for mod_name, mod in zip(module_names, modules):
        if not isinstance(mod, types.ModuleType):
            continue
        prefix = f"{mod_name} version: " if mod_name else ""
        if mod is sys:
            lines.append(f"{prefix}{mod.version_info}")
        else:
            version = getattr(mod, "__version__", None)
            if version:
                lines.append(f"{prefix}{version}")
    return joiner.join(lines)


def function_info(funcs: dict[str, Callable] | list[Callable]) -> str:
    """
    Return function signature and result if callable with no required args.
    Accepts a single function or a list of functions.
    """

    def single(f: Callable) -> str | inspect.Signature:
        sig = inspect.signature(f)
        # If no required arguments, call and print result
        if all(
            p.default != inspect.Parameter.empty
            or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in sig.parameters.values()
        ):
            buf = io.StringIO()
            original_stdout = sys.stdout
            try:
                sys.stdout = buf
                result = f()
            except Exception as exc:
                result = f"<error: {exc}>"
            finally:
                sys.stdout = original_stdout
            buf_str = buf.getvalue().strip()

            if isinstance(result, np.ndarray):
                result = summarize_array(
                    result, shape=True, dtype=True, head=3, tail=3, arr_min_max=True
                )

            if result is None and buf_str:
                result = buf_str
            elif buf_str:
                result = f"(stdout: {buf_str}; result: {result})"
        else:
            result = ""
        return result if result else sig

    results = []
    if isinstance(funcs, list):
        for func in funcs:
            if not callable(func):
                continue
            res = single(func)
            results.append(res if isinstance(res, str) else f"{func.__name__}")
    else:
        for name, func in funcs.items():
            if not callable(func):
                continue
            res = single(func)
            results.append(f"{name} -> {res}" if isinstance(res, str) else f"{name}({res})")

    return "\n".join(results)


def write_debug_report(
    output_path: Path,
    info_items: dict[str, Any] | list[Any],
    general_items: dict[str, Any] | None = None,
    exception: BaseException | None = None,
) -> None:
    """
    Write a debug report to the specified file path.

    Parameters
    ----------
    output_path : Path
        Path to the output file.
    info_items : dict[str, Any]
        Items for the VERSION/CONFIG INFO section (modules, functions, or other).
    general_items : dict[str, Any]
        Items for the ITEMS INFO section (arrays, objects, functions, etc).
    exception : BaseException, optional
        Exception to include traceback for.
    """
    buf = io.StringIO()
    # Overall header with date
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buf.write(f"===== DEBUG REPORT GENERATED {now} =====\n\n")

    # VERSION/CONFIG INFO
    buf.write("===== VERSION/CONFIG INFO =====\n")
    if info := module_versions_info(info_items):
        buf.write(info + "\n")
    if info := function_info(info_items):
        buf.write(info + "\n")
    if info := item_info(info_items, types.ModuleType, types.FunctionType):
        buf.write(info + "\n")

    # EXCEPTION TRACEBACK
    buf.write("\n===== EXCEPTION =====\n")
    if exception is not None:
        traceback.print_exception(type(exception), exception, exception.__traceback__, file=buf)
    else:
        traceback.print_exception(*sys.exc_info(), file=buf)

    # GENERAL ITEMS
    if general_items:
        buf.write("\n===== ITEMS INFO =====\n")
        for name, item in general_items.items():
            if isinstance(item, np.ndarray):
                info = summarize_array(
                    item, shape=True, dtype=True, head=5, tail=5, arr_min_max=True
                )
            elif hasattr(item, "__dict__") or hasattr(item, "__slots__"):
                info = object_attributes_info(item)
            elif callable(item):
                info = function_info({item.__name__: item})
            else:
                info = str(item)

            if info:
                buf.write(f"\n--- {name} ---\n")
                buf.write(info + "\n")

    # Scrub user info before writing to file
    text = buf.getvalue()
    text = scrub_user_info(text)
    with output_path.open("w") as f:
        f.write(text)


write_debug_report(
    output_path=Path().home() / "Documents" / "de_solver_debug2.txt",
    info_items=[scipy, np, sys, scipy.show_config],
    # general_items={
    #     "parameters_pop": parameters_pop,
    #     "solver": self,
    # }
    # exception=exc  # Pass this if you have an exception object, e.g. from except Exception as exc
)
