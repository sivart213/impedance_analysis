# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import configparser
import os
import re
from pathlib import Path

# Third-party imports
import numpy as np


# Local application imports
from ..string_operations import (
    eval_string,
)


# %% Path resolving functions
def parse_path_str(arg):
    """
    Parses a path string or a list of path strings into a normalized list of path components.

    This function takes a single argument which can be a string, a pathlib.Path object, a list of strings,
    or a numpy.ndarray of strings, representing one or more paths. It normalizes these paths by splitting them
    into their individual components (directories and file names), filtering out any empty components or redundant
    separators. The function is designed to handle various path formats and separators, making it useful for
    cross-platform path manipulation.

    Parameters:
    - arg (str, Path, list, np.ndarray): The path or paths to be parsed. This can be a single path string,
      a pathlib.Path object, a list of path strings, or a numpy.ndarray of path strings.

    Returns:
    - list: A list of the path components extracted from the input. If the input is a list or an array,
      the output will be a flattened list of components from all the paths.

    Note:
    - The function uses regular expressions to split path strings on both forward slashes (`/`) and backslashes (`\\`),
      making it suitable for parsing paths from both Unix-like and Windows systems.
    - Path components are filtered to remove any empty strings that may result from consecutive separators or leading/trailing separators.
    - The function handles string representations of paths by stripping enclosing quotes before parsing, which is particularly
      useful when dealing with paths that contain spaces or special characters.
    """
    if isinstance(arg, (str, Path)):
        return list(filter(None, re.split(r"[\\/]+", str(repr(str(arg))[1:-1]))))
    elif isinstance(arg, (list, np.ndarray, tuple)):
        if len(arg) == 1 and isinstance(arg[0], (list, np.ndarray, tuple)):
            arg = list(arg[0])
        return list(filter(None, arg))
    return arg


def my_walk(path, res_type=None, recursive=True, ignore=None, ignore_hidden=True):
    """
    Recursively yields Path objects for files and/or directories within a given directory,
    based on specified criteria.

    This function traverses the directory tree starting from `path`, yielding Path objects
    for files and directories that match the specified criteria. It allows filtering based
    on resource type (files, directories), recursion control, and the ability to ignore
    specific paths or hidden files/directories.

    Parameters:
    - path (str or Path): The root directory from which to start walking.
    - res_type (str, optional): Specifies the type of resources to yield. Can be 'file', 'dir', or None.
      If None, both files and directories are yielded. Default is None.
    - recursive (bool, optional): If True, the function will recursively walk through subdirectories.
      If False, only the immediate children of `path` are processed. Default is True.
    - ignore (list, np.ndarray, or callable, optional): A list of paths to ignore, an array of paths to ignore,
      or a callable that takes a DirEntry object and returns True if it should be ignored. Default is None.
    - ignore_hidden (bool, optional): If True, hidden files and directories (those starting with '.' or '$')
      are ignored. Default is True.

    Yields:
    - Path: Path objects for each file or directory that matches the specified criteria.

    Note:
    - The function can handle large directories by yielding results as it walks the directory tree,
      rather than building a list of results in memory.
    - The `ignore` parameter provides flexibility in filtering out unwanted paths, either through a list,
      an array, or custom logic implemented in a callable.
    - Hidden files and directories are identified by their names starting with '.' or '$'.
    - PermissionError and NotADirectoryError are silently caught, allowing the walk to continue
      in case of inaccessible or invalid directories.
    """
    try:

        if isinstance(ignore, (list, np.ndarray)) and len(ignore) > 0:
            ignore_list = ignore
            ignore = (
                lambda var: var.path in ignore_list or Path(var.path) in ignore_list
            )
        elif not callable(ignore):
            ignore = lambda var: False

        for x in os.scandir(Path(path)):
            if (
                ignore_hidden and (x.name.startswith(".") or x.name.startswith("$"))
            ) or ignore(x):
                continue
            elif x.is_dir(follow_symlinks=False):
                if not res_type or "dir" in res_type.lower():
                    yield Path(x)
                if recursive:
                    yield from my_walk(x.path, res_type, True, ignore, ignore_hidden)
            elif not res_type or "file" in res_type.lower():
                yield Path(x)
    except (PermissionError, NotADirectoryError):
        pass


def my_filter(condition, gen, yield_first_match=False):
    """
    Filters items from a generator or list based on a specified condition, optionally yielding only the first match.

    This function applies a filtering condition to each item yielded by a generator or contained in a list. It yields
    items for which the condition evaluates to True. The condition can be a boolean value or a callable that takes an
    item and returns a boolean. If `yield_first_match` is True, the function yields the first matching item and then
    terminates; otherwise, it yields all matching items.

    Parameters:
    - condition (bool or callable): The condition to apply to each item. If a boolean, it directly determines whether
      to yield items. If a callable, it should accept an item and return a boolean indicating whether the item matches.
    - gen (generator or list): The generator or list from which to filter items. If a list is provided, it is converted
      to a generator.
    - yield_first_match (bool, optional): If True, the function yields the first item that matches the condition and
      then stops. If False, it yields all items that match the condition. Default is False.

    Yields:
    - The next item from `gen` that matches the condition specified by `condition`. If `yield_first_match` is True,
      only the first matching item is yielded.

    Note:
    - This function is designed to work with both generators and lists, providing flexibility in handling different
      types of iterable inputs.
    - The ability to yield only the first match can be useful in scenarios where only one matching item is needed,
      potentially improving performance by avoiding unnecessary processing.
    """
    try:
        if isinstance(gen, list):
            gen = iter(gen)
        while True:
            g = next(gen)
            match = condition
            if callable(condition):
                match = condition(g)
            if match:
                yield g
                if yield_first_match:
                    break
    except (StopIteration, AttributeError):
        return


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
    cp.optionxform = lambda option: option
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
                config_file[sec] = {
                    k: kwargs.get(k, v) for k, v in config_file[sec].items()
                }
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
        k: eval_string(v) for sc in checked_sec for k, v in dict(cp.items(sc)).items()
    }
    included = []
    if "include_sections" in config_file.keys():
        included = [
            get_config(file, [sec])
            for sec in config_file.pop("include_sections").split(", ")
        ]
        for inc in included:
            config_file = {**config_file, **inc}

    return {**config_file, **kwargs}
