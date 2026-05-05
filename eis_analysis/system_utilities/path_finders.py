# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import os
import re
import sys
import ctypes
import importlib
import itertools
from typing import Any
from pathlib import Path

import numpy as np

# Local application imports

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_backend():
    """
    Detect which GUI backend is available.
    Returns a string identifier or None if none found.
    """
    backends = ["PyQt6", "PyQt5", "PySide6", "PySide2", "tkinter"]
    for backend in backends:
        try:
            importlib.import_module(backend if backend == "tkinter" else backend + ".QtWidgets")
            return backend
        except ImportError:
            continue
    return ""


tk: Any = None
fDialog: Any = None
INTERFACE_PKG = detect_backend()
if INTERFACE_PKG == "tkinter":
    import tkinter as tk
    from tkinter import filedialog as fDialog
elif INTERFACE_PKG == "PyQt5":
    from PyQt5.QtWidgets import QFileDialog as fDialog  # type: ignore
elif INTERFACE_PKG == "PyQt6":
    from PyQt6.QtWidgets import QFileDialog as fDialog  # type: ignore
elif INTERFACE_PKG == "PySide2":
    from PySide2.QtWidgets import QFileDialog as fDialog  # type: ignore
elif INTERFACE_PKG == "PySide6":
    from PySide6.QtWidgets import QFileDialog as fDialog  # type: ignore

# Shared dictionary of grouped file type filters
FILTER_GROUPS = {
    "text": ["*.txt", "*.md", "*.log", "*.csv"],
    "documents": ["*.doc", "*.docx", "*.odt", "*.rtf", "*.pdf"],
    "spreadsheets": ["*.xlsx", "*.xlsm", "*.xlsb", "*.xls", "*.ods", "*.csv"],
    "hdf": ["*.hdf", "*.hdf5", "*.h5", "*.hdf4", "*.h4"],
    "code": ["*.py", "*.c", "*.cpp", "*.h", "*.java", "*.js", "*.ts", "*.sh", "*.bat"],
    "config": ["*.ini", "*.cfg", "*.conf", "*.json", "*.yaml", "*.yml", "*.toml"],
    "images": ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff", "*.svg"],
    "audio": ["*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg"],
    "video": ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv"],
    "archives": ["*.zip", "*.tar", "*.gz", "*.rar", "*.7z"],
    "python": ["*.py", "*.pyw", "*.pyc"],
    "matlab": ["*.mat"],
    "all": ["*.*"] if INTERFACE_PKG == "tkinter" else ["*"],
}
FILTER_GROUPS["any"] = FILTER_GROUPS["all"]
FILTER_GROUPS |= {k: FILTER_GROUPS["documents"] for k in {"doc", "docs", "document"}}
FILTER_GROUPS |= {
    k: FILTER_GROUPS["spreadsheets"] for k in {"xls", "xlsx", "sheet", "sheets", "excel"}
}


def format_filters(filters, style="qt"):
    """
    Format filters for the active backend.
    - If INTERFACE_PKG == "tkinter": return list of (label, pattern) tuples
    - Otherwise (Qt backends): return ';;'-separated string
    filters: str or list of group names (e.g. "text" or ["text","code"])
    """
    # Normalize to list of group names
    if isinstance(filters, str):
        names = re.split(r";+" if ";" in filters else r",+", filters)
    else:
        names = filters

    # Build filter entries (n.title(), " ".join(FILTER_GROUPS.get(n.lower(), [n]))) for n in names
    entries = []
    for name in names:
        if name.lower() in FILTER_GROUPS:
            entries.append((name.title(), " ".join(FILTER_GROUPS[name.lower()])))

        elif isinstance(name, str):
            parsed = re.match(r"^(.*)\s*\((.*)\)$", name)
            if parsed:
                entries.append((parsed.group(1).strip(), parsed.group(2).strip()))
            else:
                entries.append((name.title(), "*." + name.lstrip("*.")))

    # Format based on backend
    if "qt" in style.lower() or "pyside" in style.lower():
        return ";;".join([f"{label} ({patterns})" for label, patterns in entries])
    return entries


def parse_path_str(arg: str | Path | list | np.ndarray | tuple) -> list:
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


# %% Path resolving functions
def find_path(*dir_in, base=None, as_list=False, by_re=True, by_glob=False, **kwargs):
    """
    Searches for paths matching the specified directory names within given base directories,
    optionally using glob patterns or regular expressions. If no paths are found directly,
    performs a recursive search within each base directory.
    Parameters:
    - dir_in (tuple): Directory names to be joined together and searched for. Can be a mix of strings and Path objects.
    - base (Path or str, optional): The base directory or directories to search within. If not specified, uses a default set of base directories.
    - as_list (bool, optional): If True, assumes dir_in is a list of targets.Returns a list of each result for a given dir_in value. Default is False.
    - by_re (bool, optional): If True, uses regular expressions for searching. Default is True.
    - by_glob (bool, optional): If True, uses glob patterns for searching. Overrides by_re if both are True. Default is False.
    - **kwargs: Additional keyword arguments, reserved for future use.
    Returns:
    - Path or list of Path: Depending on the value of `as_list`, either the first matching Path object or a list of all matching Path objects.
    """
    # Validate formatting
    dir_in = parse_path_str(dir_in)

    if Path(*dir_in).exists():
        return Path(*dir_in)

    if as_list:
        return [find_path(d, **kwargs) for d in dir_in]

    def overlap(path1, *path2):
        if len(path2) >= 1 and Path(*path2).parts[0] in Path(path1).parts:
            for b in path1.parents:
                if path2[0] not in b.parts:
                    path1 = b
                    break
        return path1

    drives = find_drives(exclude_nonlocal=True, exclude_hidden=True, **kwargs)
    for d in drives:
        if dir_in[0] in str(d):
            dir_in[0] = str(d)
            break

    dir_in_parents = list(Path(*dir_in).parents)
    n = 1
    while dir_in_parents[-n].exists() and n < len(dir_in) - 1:
        n += 1

    if dir_in[0] in drives and n > 1:
        # if the first directory is a drive and there are multiple directories, use the n-th parent as it's likely more accurate as a base
        base = dir_in_parents[-n]
    elif isinstance(base, str):
        # if base is a string, it is likely a keyword signaling a particular base
        if base.lower() in ["local", "caller", "argv", "sys.argv"]:
            # each of these options are assuming the base is the directory of the calling script
            base = Path(sys.argv[0]).resolve().parent
        elif "drive" in base.lower():
            # The base is a drive, check if dir in is one level down
            base_path = [p for d in drives for p in d.glob("*/" + str(Path(*dir_in)))]
            # Otherwise, check if dir in is two levels down
            if base_path == []:
                base_path = [p for d in drives for p in d.glob("*/*/" + str(Path(*dir_in)))]

            if base_path == []:
                base = None
            else:
                base_path.sort(key=lambda x: len(Path(x).parts))
                base = base_path[0]

        else:
            # it's possible that the base is an attribute of the Path class
            base = getattr(Path, base)() if hasattr(Path, base) else Path(base)
    if base is None or not isinstance(base, Path) or not base.exists():
        # if base has not been resolved, use the Documents folder of the home directory
        base = Path.home() / "Documents"

    # if there may be overlap, shrink base path until there isn't overlap
    base = overlap(base, *dir_in)

    # try just merging without glob
    if (base / Path(*dir_in)).exists():
        return base / Path(*dir_in)
    if (base.parent / Path(*dir_in)).exists():
        return base.parent / Path(*dir_in)

    # Try to find drives without exclusions
    if drives == []:
        drives = find_drives(**kwargs)

    # get list of possible bases
    bases_all = [base, Path.cwd(), Path.home()] + drives

    bases = []
    for b in bases_all:
        if b not in bases:
            bases.append(overlap(b, *dir_in))

    # define target type
    res_type = kwargs.get("res_type", "file" if Path(*dir_in).suffix else "dir")

    # Search one level down from available bases
    paths = []
    if by_glob:
        paths = [p for b in bases for p in b.glob("*/" + str(Path(*dir_in)))]
    if by_re or not by_glob:
        for b in bases:
            paths = paths + find_files(b, "path", res_type, Path(*dir_in).parts, recursive=False)

    # if paths are not found, do a recursive search
    n = 0
    while n < len(bases) and paths == []:
        if by_glob:
            paths = list(bases[n].glob("**/" + str(Path(*dir_in))))
        if by_re or not by_glob:
            paths = paths + find_files(
                bases[n], "path", res_type, Path(*dir_in).parts, ignore=bases[:n], recursive=True
            )
        n += 1

    # Sort by shortest path and return 1st option
    paths.sort(key=lambda x: len(Path(x).parts))
    if len(paths) == 1:
        return paths[0]
    elif len(paths) > 1 and Path(paths[0]).exists():
        return paths[0]

    return base / Path(*dir_in)


class RegexFilter:
    def __init__(self, patterns: str | Path | tuple[str, ...] | list[str]):
        if isinstance(patterns, Path):
            patterns = parse_path_str(patterns)
        elif isinstance(patterns, str):
            patterns = [patterns]

        self.patterns = []
        for pattern in patterns:
            try:
                self.patterns.append(re.compile(pattern))
            except re.error:
                continue

    def __call__(self, x):
        if not self.patterns:
            return True
        return all(p.search(str(x)) for p in self.patterns)


def find_files(
    path,
    attr="",
    res_type="file",
    patterns=None,
    ignore=None,
    recursive=True,
    yield_first_match=False,
    yield_shortest_match=False,
):
    """
    Searches for files or directories within a given path that match specified patterns,
    optionally filtering by resource type and controlling recursion and result quantity.
    Parameters:
    - path (str or Path): The root directory from which to start the search.
    - attr (str, optional): Additional attributes to filter the search. Default is an empty string.
    - res_type (str, optional): Specifies the type of resources to search for. Can be 'file', 'dir', or None. If None,
    both files and directories are included in the search results. Default is None.
    - patterns (str, Path, list of str, optional): Regular expression patterns to match against the names of the files
    or directories. If None, all files or directories are considered a match. Default is None.
    - ignore (str, Path, list of str, optional): Regular expression patterns for files or directories to ignore. Default is None.
    - recursive (bool, optional): If True, the function will search through all subdirectories of the given path. If False,
    only the immediate children of the path are considered. Default is False.
    - yield_first_match (bool, optional): If True, the function returns the first matching file or directory and stops the
    search. If False, all matching files or directories are returned. Default is False.
    - yield_shortest_match (bool, optional): If True, the function returns the match with the shortest path. This is useful
    when searching for the most relevant result. Default is False.
    Returns:
    - list: A list of Path objects for each file or directory that matches the specified criteria. If `yield_first_match`
      is True, the list contains at most one Path object.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Example of parameter validation
    if res_type not in ["file", "dir", None]:
        raise ValueError("res_type must be 'file', 'dir', or None")

    # Purpose: find files or dirs which match desired
    if patterns is None:
        filesurvey = list(my_walk(Path(path), res_type, recursive, ignore))
    else:
        f_filter = RegexFilter(patterns)
        yield_first_match = yield_first_match if patterns else False
        filesurvey = list(
            my_filter(
                f_filter,
                my_walk(Path(path), res_type, recursive, ignore),
                yield_first_match,
            ),
        )

    if yield_shortest_match:
        filesurvey.sort(key=lambda x: x.inode())

    if not filesurvey or attr == "":
        return filesurvey
    if hasattr(filesurvey[0], attr):
        if attr.lower() == "path":
            return [Path(str(f)) for f in filesurvey]
        return [getattr(f, attr) for f in filesurvey]
    if hasattr(filesurvey[0].stat(), attr):
        return [getattr(f.stat(), attr) for f in filesurvey]
    return filesurvey


# %% Path resolving functions
def find_drives(exclude_nonlocal=True, exclude_hidden=True):
    """
    Finds and returns a list of drive paths available on the system.
    Parameters:
    - exclude_nonlocal (bool): If True, excludes network drives from the result. Default is True.
    - exclude_hidden (bool): If True, excludes drives that are marked as hidden. Default is True.
    Returns:
    - list: A list of pathlib.Path objects representing the paths to the drives found.
    """
    if sys.platform.startswith("win"):
        drives = detect_windows_drives(exclude_nonlocal=exclude_nonlocal)
    elif sys.platform.startswith("linu"):
        drives = detect_posix_drives("m*/*", exclude_nonlocal)
    elif sys.platform.startswith("darw"):
        drives = detect_posix_drives("Vol*/*", exclude_nonlocal)
    else:
        drives = [Path.home(), Path(Path.home().parts[0])]

    if exclude_hidden:
        drives = [dr for dr in drives if not str(dr).lower().endswith(".hidden")]

    return drives


def detect_windows_drives(letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ", exclude_nonlocal=True):
    """
    Detects and returns a list of drive paths available on a Windows system, with options to exclude non-local drives.
    Parameters:
    - letters (str, optional): A string containing the uppercase alphabet letters used to check for drive presence.
      Default is "ABCDEFGHIJKLMNOPQRSTUVWXYZ".
    - exclude_nonlocal (bool, optional): If True, excludes network drives from the result. Default is True.
    Returns:
    - list of pathlib.Path: A list of Path objects representing the paths to the drives found on the system.
      Each Path object corresponds to a drive root (e.g., C:/).
    """
    import win32net

    resume = 0
    net_dr = []
    while 1:
        net_res, _, resume = win32net.NetUseEnum(None, 0, resume)  # type: ignore
        for dr in net_res:
            net_dr.append(Path(dr["local"]))
            net_dr.append(Path(dr["remote"]))
        if not resume:
            break

    drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
    drives = list(
        map(
            Path,
            map(
                "{}:/".format,
                itertools.compress(
                    letters,
                    map(lambda x: ord(x) - ord("0"), bin(drive_bitmask)[:1:-1]),
                ),
            ),
        )
    )
    if exclude_nonlocal:
        drives = [dr for dr in drives if Path(dr.drive) not in net_dr]
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    return drives


def detect_posix_drives(pattern="m*/*", exclude_nonlocal=True):
    """
    Detects and returns a list of drive paths available on POSIX-compliant systems (e.g., Linux, macOS).
    Parameters:
    - pattern (str, optional): The glob pattern used to identify potential drives at the root directory.
      Default is "m*/*", which aims to target mnt and media directories having at least one subdirectory typical of Linux structures.  Alternatively, utilize "Vol*/*" for macOS.
    - exclude_nonlocal (bool, optional): If True, excludes drives that do not have their realpath matching
      their path, which typically indicates network-mounted drives. Default is True.
    Returns:
    - list of pathlib.Path: A list of Path objects representing the mount points of the drives found on the
      system. Each Path object corresponds to a drive's mount point.
    """
    drives = [dr for dr in Path("/").glob(pattern) if dr.is_dir() and any(dr.iterdir())]
    for drn, dr in enumerate(drives):
        dr_f = [x for x in os.listdir(dr)]
        while len(dr_f) == 1:
            drives[drn] = dr / dr_f[0]
            dr_f = [x for x in drives[drn].iterdir()]
    drives.append(Path.home())
    drives.append(Path("/"))

    if exclude_nonlocal:
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    return drives


class IgnoreFilter:
    """
    Callable filter for ignoring paths during directory walks.

    Accepts:
    - None → always returns False (no ignores)
    - list/np.ndarray of paths → ignores those paths
    - callable → uses it directly
    """

    def __init__(self, ignore=None):
        self._ignore = set()
        # normalize to Path for consistent comparison
        if isinstance(ignore, (tuple, list, np.ndarray)) and len(ignore) > 0:
            self._ignore = {Path(p) for p in ignore}

    def __call__(self, dir_entry):
        if self._ignore:
            return Path(dir_entry.path) in self._ignore
        return False


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
        f_ignore = ignore if callable(ignore) else IgnoreFilter(ignore)
        # if isinstance(ignore, (list, np.ndarray)) and len(ignore) > 0:
        #     ignore_list = ignore
        #     ignore = lambda var: var.path in ignore_list or Path(var.path) in ignore_list
        # elif not callable(ignore):
        #     ignore = lambda var: False

        for x in os.scandir(Path(path)):
            if (ignore_hidden and (x.name.startswith(".") or x.name.startswith("$"))) or f_ignore(
                x
            ):
                continue
            elif x.is_dir(follow_symlinks=False):
                if not res_type or "dir" in res_type.lower():
                    yield Path(x)
                if recursive:
                    yield from my_walk(x.path, res_type, True, f_ignore, ignore_hidden)
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


# %% Interactive path input functions
def _parse_paths(raw_input: str, initial_dir: str) -> list[str]:
    # Split on semicolons, strip whitespace
    parts = [p.strip() for p in raw_input.split(";") if p.strip()]
    cd = raw_input.lower().startswith("cd")
    paths = []
    for part in parts:
        if part.lower().startswith("cd"):  # Absolute path override
            path = Path(part[2:].strip())
        elif cd and Path(part).is_absolute():  # Absolute path override
            path = Path(part)
        else:  # Relative to initial_dir
            path = Path(initial_dir) / part.strip(r"\/") if initial_dir else Path(part)
        if "*" in path.name or "?" in path.name:  # Handle wildcards
            paths.extend([str(p) for p in path.parent.glob(path.name)])
        else:
            paths.append(str(path))
    return paths


def validate_path(
    p: str | Path,
    for_write: bool = False,  # "read" or "write"
    is_file: bool = False,  # "file" or "dir"
) -> bool:
    """
    Validate path for read/write operations.

    - mode="read": path must exist and be accessible
    - mode="write": path may not exist, but some parent must exist and be writable
    - kind="file": check file semantics
    - kind="dir": check directory semantics
    """
    p = Path(p)

    if not for_write:  # check for read access
        if is_file:
            return p.is_file() and os.access(p, os.R_OK)
        return p.is_dir() and os.access(p, os.X_OK)
    else:
        # If target exists, ensure it's writable
        if p.exists() and os.access(p, os.W_OK):
            return p.is_file() if is_file else p.is_dir()
        if not is_file and p.suffix:  # if is dir but has file suffix, invalid
            return False
        # Otherwise, ensure some parent exists and is writable
        for parent in p.parents:
            if parent.exists() and os.access(parent, os.W_OK):
                return True
        return False


def input_path(
    title="Select a file",
    initial_dir="",
    is_file=False,
    for_write=False,
    multi=False,
) -> list[str]:

    initial_dir = initial_dir or str(Path.home())
    title = title.replace("Select", "Input")

    if not for_write and multi:
        raw = input(
            f'{title} path(s) (";" separated, prefix w/ "cd" for absolute):\n\t{initial_dir}> '
        ).strip()
    else:
        raw = input(f'{title} path (prefix w/ "cd" for absolute):\n\t{initial_dir}> ').strip()

    paths = _parse_paths(raw, initial_dir)

    if is_file:
        if any(not validate_path(pth, for_write, is_file) for pth in paths):
            invalid = [pth for pth in paths if not validate_path(pth, for_write, is_file)]
            raise FileNotFoundError("The following file paths are invalid:\n" + "\n".join(invalid))
        return paths
    else:
        invalid = [not validate_path(pth, for_write, is_file) for pth in paths]
        if any(invalid):
            create_dir = (
                input("One or more directories do not exist. Create them? (y/n): ").strip().lower()
                == "y"
            )
            if not create_dir:
                raise FileNotFoundError("One or more directory paths are invalid.")
        valid_paths = []
        for pth, check in zip(paths, invalid):
            if check:
                try:
                    Path(pth).mkdir(parents=True, exist_ok=True)
                    valid_paths.append(pth)
                except Exception:
                    raise IOError(f"Could not create directory: {pth}")
            else:
                valid_paths.append(pth)

        return valid_paths


def get_path_prompt(
    mode="open",
    initial_dir="",
    filters: str | list | tuple = "All files (*)",
    multi_select=False,
):
    """
    Prompt user to select file(s) or directory via GUI dialog or console input.

    Parameters:
    -----------
    - mode (str): Mode of operation. Options include:
        - "open": Select file(s) for reading.
        - "save": Select file for writing.
        - "dir": Select directory.
        - Combinations like "open dir", "save dir" are also valid.
    - initial_dir (str): Initial directory to open in the dialog.
    - filters (str or list or tuple): File type filters for file selection dialogs.
    - multi_select (bool): If True, allows selection of multiple files.

    Returns:
    --------
    - list[str]: List of selected file or directory path(s).

    Notes:
    ------
    - If no GUI backend is available, falls back to console input.
    - If no selection is made, returns a list with an empty string.
    """
    path: list[str] = []
    mode = mode.lower()
    title = ""
    as_file = "dir" not in mode
    for_write = "save" in mode or "write" in mode
    if not as_file:
        title = "Select directory"
    elif for_write:
        title = "Save as"
    else:
        title = "Select file(s)" if multi_select else "Select file"

    if fDialog is None:
        path = input_path(title, initial_dir, as_file, for_write, multi_select)
    elif INTERFACE_PKG == "tkinter":
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-alpha", 0)
        if title == "Select directory":
            path = [fDialog.askdirectory(title=title, initialdir=initial_dir)]
        else:
            filters = format_filters(filters, style="tkinter")
            if title == "Save as":
                func = fDialog.asksaveasfilename
            elif multi_select:
                func = fDialog.askopenfilenames
            else:
                func = fDialog.askopenfilename
            res = func(title=title, initialdir=initial_dir, filetypes=filters)
            path = [res] if isinstance(res, str) else list(res)

        root.destroy()

    elif (is_q5 := INTERFACE_PKG in ("PyQt5", "PySide2")) or INTERFACE_PKG in ("PyQt6", "PySide6"):
        dlg = fDialog()
        dlg.setDirectory(initial_dir)
        dlg.setWindowTitle(title)
        if title == "Select directory":
            dlg.setFileMode(fDialog.Directory if is_q5 else fDialog.FileMode.Directory)
        else:
            dlg.setNameFilter(format_filters(filters, style="qt"))
            if title == "Save as":
                dlg.setAcceptMode(fDialog.AcceptSave if is_q5 else fDialog.AcceptMode.AcceptSave)
            elif multi_select:
                dlg.setFileMode(fDialog.ExistingFiles if is_q5 else fDialog.FileMode.ExistingFiles)
            else:
                dlg.setFileMode(fDialog.ExistingFile if is_q5 else fDialog.FileMode.ExistingFile)

        if is_q5:
            if dlg.exec_():  # note: exec_() in Qt5, exec() in Qt6
                path = dlg.selectedFiles()
        else:
            if dlg.exec():
                path = dlg.selectedFiles()

    else:
        path = input_path(title, initial_dir, as_file, for_write, multi_select)

    return path or [""]


# if title == "Select file(s)":
#                 path = fDialog.askopenfilenames(
#                     title=title, initialdir=initial_dir, filetypes=filters
#                 )
#             elif title == "Select file":
#                 path = [
#                     fDialog.askopenfilename(title=title, initialdir=initial_dir, filetypes=filters)
#                 ]
#             elif title == "Save as":
#                 path = [
#                     fDialog.asksaveasfilename(
#                         title=title, initialdir=initial_dir, filetypes=filters
#                     )
#                 ]
#             elif title == "Select directory":
#                 path = [fDialog.askdirectory(title=title, initialdir=initial_dir)]
# elif INTERFACE_PKG in ("PyQt5", "PySide2"):
#     dlg = fDialog()
#     dlg.setDirectory(initial_dir)
#     dlg.setNameFilter(filters)

#     if title == "Select file(s)":
#         dlg.setWindowTitle(title)
#         dlg.setFileMode(fDialog.ExistingFiles)
#     elif title == "Select file":
#         dlg.setWindowTitle(title)
#         dlg.setFileMode(fDialog.ExistingFile)
#     elif title == "Save as":
#         dlg.setWindowTitle(title)
#         dlg.setAcceptMode(fDialog.AcceptSave)
#     elif title == "Select directory":
#         dlg.setWindowTitle(title)
#         dlg.setFileMode(fDialog.Directory)
#     if dlg.exec_():  # note: exec_() in Qt5, exec() in Qt6
#         files = dlg.selectedFiles()
#         if files:
#             return files if multi_select else files[0]

# elif INTERFACE_PKG in ("PyQt6", "PySide6"):
#     dlg = fDialog()
#     dlg.setDirectory(initial_dir)
#     dlg.setNameFilter(filters)
#     if title == "Select file(s)":
#         dlg.setWindowTitle(title)
#         dlg.setFileMode(fDialog.FileMode.ExistingFiles)
#     elif title == "Select file":
#         dlg.setWindowTitle(title)
#         dlg.setFileMode(fDialog.FileMode.ExistingFile)
#     elif title == "Save as":
#         dlg.setWindowTitle(title)
#         dlg.setAcceptMode(fDialog.AcceptMode.AcceptSave)
#     elif title == "Select directory":
#         dlg.setWindowTitle(title)
#         dlg.setFileMode(fDialog.FileMode.Directory)
#     if dlg.exec():
#         files = dlg.selectedFiles()
#         if files:
#             return files if multi_select else files[0]


# elif INTERFACE_PKG in ("PyQt5", "PySide2"):

#     if title == "Select file(s)":
#         path, _ = fDialog.getOpenFileNames(
#             None, title, initial_dir, filters, selected_filter
#         )
#     elif title == "Select file":
#         path, _ = fDialog.getOpenFileName(
#             None, title, initial_dir, filters, selected_filter
#         )
#     elif title == "Save as":
#         path, _ = fDialog.getSaveFileName(
#             None, title, initial_dir, filters, selected_filter
#         )
#     elif title == "Select directory":
#         path = fDialog.getExistingDirectory(None, title, initial_dir)
#     return path
# if not initial_dir and Path(QFileDialog().directory().path()) == Path().cwd():
#     initial_dir = str(Path().home())

# if any(not validate_path(pth, for_write, is_file) for pth in paths):
#     create_dir = False
#     if not is_file:
#         create_dir = input(
#             "One or more directories do not exist. Create them? (y/n): "
#         ).strip().lower() == "y"
#     if create_dir:
#         for pth in paths:
#             if not validate_path(pth, for_write, is_file):
#                 try:
#                     Path(pth).mkdir(parents=True, exist_ok=True)
#                 except Exception:
#                     raise IOError(f"Could not create directory: {pth}")
#     else:
#         raise FileNotFoundError("One or more paths are invalid.")
# valid_paths = []
# for pth in paths:
#     if validate_path(pth, for_write, is_file):
#         valid_paths.append(pth)
#     elif not is_file and create_dir:
#         try:
#             Path(pth).mkdir(parents=True, exist_ok=True)
#             valid_paths.append(pth)
#         except Exception:
#             raise IOError(f"Could not create directory: {pth}")
#     else:
#         raise FileNotFoundError(f"Invalid path: {pth}")

# return valid_paths


# def validate_path(p: str | Path, mode: str = "drive") -> bool:
#     """
#     mode options:
#       - "drive": ensure drive exists
#       - "dir": ensure directory exists
#       - "file": ensure file exists
#     """
#     p = Path(p)
#     if mode.lower()[:3] == "dir":
#         return p.is_dir()
#     elif mode.lower()[:4] == "file":
#         return p.is_file()
#     else:
#         if p.exists() and os.access(p, os.W_OK):
#             return True
#         for parent in p.parents:
#             if parent.exists() and os.access(parent, os.W_OK):
#                 return True
#         return False


# pth = input(
#     f'{title} path (prefix w/ "cd" to ignore default path):\n\t{initial_dir}> '
# ).strip()

# if "cd" in pth.lower():
#     pth = pth.replace("cd", "").strip()
# else:
#     pth = str(Path(initial_dir) / pth)
# return pth
