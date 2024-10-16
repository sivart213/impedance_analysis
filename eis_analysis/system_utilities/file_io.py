# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import os
import re
import sys
from datetime import datetime as dt
from pathlib import Path
import logging
import inspect

# Third-party imports
import h5py
import numpy as np
import pandas as pd



# Local application imports
from ..data_treatment import (
    dict_df,
    merge_single_key,
    dict_key_sep,
    Complex_Imp,
)

from ..string_operations import (
    slugify,
)
from .file_parsers import (
    parse_path_str,
    my_filter,
    my_walk,
)

from .system_info import (
    find_drives,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    The function first attempts to find paths directly within the specified base directories. If no matches are found,
    it recursively searches within each base directory. The search can be performed using either glob patterns or regular
    expressions, with glob patterns taking precedence if both are specified. The function sorts the found paths by their
    length, preferring shorter paths, and returns either the first match or a list of all matches based on the `as_list` parameter.
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
                base_path = [
                    p for d in drives for p in d.glob("*/*/" + str(Path(*dir_in)))
                ]

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
            paths = paths + find_files(
                b, "path", res_type, Path(*dir_in).parts, recursive=False
            )

    # if paths are not found, do a recursive search
    n = 0
    while n < len(bases) and paths == []:
        if by_glob:
            paths = list(bases[n].glob("**/" + str(Path(*dir_in))))
        if by_re or not by_glob:
            paths = paths + find_files(
                bases[n],
                "path",
                res_type,
                Path(*dir_in).parts,
                ignore=bases[:n],
                recursive=True,
            )
        n += 1

    # Sort by shortest path and return 1st option
    paths.sort(key=lambda x: len(Path(x).parts))
    if len(paths) == 1:
        return paths[0]
    elif len(paths) > 1 and Path(paths[0]).exists():
        return paths[0]

    return base / Path(*dir_in)


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

    This function is designed to find files or directories that match a set of regular expression patterns within a specified path.
    It allows for filtering the search results based on the type of resource (file or directory), and can perform either a
    non-recursive (default) or recursive search. Additionally, it can be configured to yield only the first match found or the shortest match based on the specified criteria.

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

    Note:
    - The function uses regular expressions for pattern matching, allowing for flexible and powerful search criteria.
    - If `patterns` is provided as a Path object, it is converted to a string representation before being used for matching.
    - The search is performed using a combination of `my_walk` for traversing directories and `my_filter` for applying
      the match criteria, demonstrating the use of helper functions for modular code design.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Example of parameter validation
    if res_type not in ["file", "dir", None]:
        raise ValueError("res_type must be 'file', 'dir', or None")

    # Purpose: find files or dirs which match desired
    if isinstance(patterns, Path):
        patterns = parse_path_str(patterns)
    elif isinstance(patterns, str):
        patterns = [patterns]

    if patterns:
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error:
                continue
        # compiled_patterns = [re.compile(pattern) for pattern in patterns]
        f_filter = lambda x: all(
            pattern.search(str(x)) for pattern in compiled_patterns
        )
    else:
        f_filter = lambda x: True  # No-op lambda, always returns True
        yield_first_match = False  # If no patterns, always return all matches

    if yield_first_match or callable(f_filter):
        filesurvey = list(
            my_filter(
                f_filter,
                my_walk(Path(path), res_type, recursive, ignore),
                yield_first_match,
            ),
        )
    else:
        filesurvey = list(my_walk(Path(path), res_type, recursive, ignore))

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


def save(data, path=None, name=None, file_type="xls", **kwargs):
    """Save data into excel file."""
    path = Path(path)
    if path is None:
        # TODO convert to home path
        path = find_path(
            "Data", "Analysis", "Auto", base=find_path(r"ASU Dropbox", base="drive")
        ) / dt.now().strftime("%Y%m%d")
    if name is None:
        name = "data_" + dt.now().strftime("%H_%M")
    if not os.path.exists(path):
        os.makedirs(path)

    if isinstance(data, (list, np.ndarray)):
        if isinstance(data[0], (pd.DataFrame, pd.Series)):
            data = {x: data[x] for x in range(len(data))}
        else:
            data = pd.DataFrame(data)

    if isinstance(data, (dict)):
        if not isinstance(data[list(data.keys())[0]], (pd.DataFrame, pd.Series)):
            data = pd.DataFrame(data)

    if isinstance(data, (pd.DataFrame, pd.Series)) and "xls" in file_type.lower():
        data.to_excel(
            path / f"{slugify(name)}.xlsx",
            merge_cells=kwargs.pop("merge_cells", False),
            **kwargs,
        )
    elif isinstance(data, (dict)) and "xls" in file_type.lower():
        mc_kw = kwargs.pop("merge_cells", False)
        with pd.ExcelWriter(path / f"{slugify(name)}.xlsx") as writer:
            for key, df in data.items():
                df.to_excel(
                    writer,
                    sheet_name=slugify(key, sep=" "),
                    merge_cells=mc_kw,
                    **kwargs,
                )
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        data.to_csv(
            path / f"{slugify(name)}.{file_type}",
            index=kwargs.pop("index", False),
            **kwargs,
        )
    elif isinstance(data, (dict)):
        for key, df in data.items():
            df.to_csv(
                path / f"{slugify(name)}_{key}.{file_type}",
                index=kwargs.pop("index", False),
                **kwargs,
            )

def filter_kwargs(func, kwargs):
    """
    Filters the kwargs to only include those that are valid for the given function.
    
    Parameters:
    - func (function): The function to filter kwargs for.
    - kwargs (dict): The keyword arguments to filter.
    
    Returns:
    - dict: A dictionary of filtered keyword arguments.
    """
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return valid_kwargs


def load_file(file, path=None, verbose=False, **kwargs):
    """
    Loads data from Excel or HDF5 files.
    
    Parameters:
    - file (str or Path): The file to load.
    - path (str, Path, or list): The path to the file or a list of paths to search.
    - pdkwargs (dict): Additional keyword arguments to pass to pandas read functions.
    - hdfkwargs (dict): Additional keyword arguments to pass to HDF5 read functions.
    - kwargs (dict): Additional keyword arguments.
        - flat_df (bool): Whether to flatten the DataFrame.
        - target (str): Target path for HDF5 files.
    
    Returns:
    - data (dict): Dictionary of DataFrames loaded from the file.
    - attrs (dict): Dictionary of attributes loaded from the file.
    """
    file = Path(file)
    if isinstance(path, list):
        path = find_path(path)
    if isinstance(path, str):
        path = Path(path)
    if path is not None:
        file = path / file

    data = {}
    attrs = {}

    if re.search(r"(.xls|.xls\w)$", str(file)):
        data, attrs = load_excel(file, verbose,**filter_kwargs(pd.read_excel, kwargs))
    elif re.search(r"(.h5|.hdf5)$", str(file)):
        data, attrs = load_hdf(
            file,
            None,
            kwargs.pop("target", "/"),
            kwargs.pop("key_sep", True),
            **filter_kwargs(load_excel, kwargs),
        )
        if kwargs.get("flat_df", False):
            data = dict_df(merge_single_key(data))
    elif file.exists() and file.is_dir():
        filelist = find_files(
            file,
            patterns=kwargs.get("file_filter", kwargs.get("patterns", "")),
        )
        return [load_file(f, None, verbose, **kwargs) for f in filelist]

    return data, attrs


def load_excel(file, verbose=False, **kwargs):
    """
    Loads data from an Excel file and returns it as dictionaries.
    
    Parameters:
    - file (str or Path): The Excel file to load.
    - pdkwargs (dict): Additional keyword arguments to pass to pandas read functions.
    - kwargs (dict): Additional keyword arguments.
    
    Returns:
    - data (dict): Dictionary of DataFrames loaded from the file.
    - attrs (dict): Dictionary of attributes loaded from the file.
    """
    data = {}
    attrs = None
    
    # Load the Excel file to get sheet names
    excel_file = pd.ExcelFile(file)
    names = excel_file.sheet_names
    excel_file.close()
    
    # Load the first sheet to determine headers and rows
    first_sheet_name = names[0]
    kw_header = kwargs.pop("header", 0)
    kw_index_col = kwargs.pop("index_col", 0)
    kw_nrows = kwargs.pop("nrows", 10)
    
    
    # Load the attrs sheet if it exists
    if 'attrs' in names:
        names.remove('attrs')
        attrs = pd.read_excel(file, sheet_name='attrs', header=0, index_col=0, **kwargs)
    
    
    
    test_df = pd.read_excel(file, sheet_name=first_sheet_name, header=kw_header, index_col=kw_index_col, nrows=kw_nrows, **kwargs)
    
    if test_df.isnull().all(axis=1).any():
        test_df = pd.read_excel(file, sheet_name=first_sheet_name, header=None, nrows=10)
        
        # Determine the number of header rows by looking for a row of all NaNs
        header_rows = 0
        for i, row in test_df.iterrows():
            if row.isnull().all():
                break
            header_rows += 1
        
        if header_rows < 10:
            # Multi-index DataFrame
            headers = test_df.loc[:header_rows-1, :]
            index_rows = 0
            if any(headers.isna()):
                index_rows = int(headers.columns[headers.isna().any()][0])
                index_rows = list(range(index_rows - 1)) if index_rows > 1 else [0]
            header = list(range(header_rows))
        elif np.isnan(test_df.loc[0, 0]):
            # Basic DataFrame with index
            header = 0
            index_rows = 0
        else:
            # Default import
            header = kw_header
            index_rows = kw_index_col
    else:
        # Default import
        header = kw_header
        index_rows = kw_index_col
    
    # Load all sheets at once
    all_sheets = pd.read_excel(file, sheet_name=names, header=header, index_col=index_rows, **kwargs)
    
    for sheet_name, df in all_sheets.items():
        if verbose:
            logging.info(f"Processing sheet: {sheet_name}")
        data[sheet_name] = df
        # Apply attrs to the respective DataFrame
        data[sheet_name].attrs = attrs.loc[sheet_name,:].to_dict() if isinstance(attrs, pd.DataFrame) else {}
    return data, attrs


def load_hdf(file, path=None, target="/", key_sep=False, verbose=False, **kwargs):
    """
    Loads data from an HDF5 file and returns it as dictionaries.

    This function reads an HDF5 file and extracts datasets and attributes into
    dictionaries. It allows for optional path resolution and filtering based on
    a target string.

    Parameters:
    file (str or Path): The name or path of the HDF5 file to be loaded.
    path (str, Path, or list, optional): The directory path where the file is located.
        If a list is provided, the function will search for the file in the specified paths.
        If None, the current working directory is used. Default is None.
    target (str, optional): A string to filter the datasets and attributes to be extracted.
        Only items containing the target string in their name will be included. Default is "/".
    key_sep (bool, optional): If True, the keys in the returned dictionaries will be separated into nested dictionaries
        based on the '/' character. Default is False.
    **kwargs: Additional keyword arguments to be passed to the h5py.File visititems method.

    Returns:
    tuple: A tuple containing two dictionaries:
           - ds_dict: A dictionary where keys are dataset names and values are the dataset contents.
           - attr_dict: A dictionary where keys are attribute names and values are the attribute contents.
    """

    if isinstance(path, list):
        path = find_path(path)
    if isinstance(path, Path):
        path = str(path)
    if path is not None:
        file = path / file

    def get_ds_dictionaries(name, node):
        if target in name:
            if isinstance(node, h5py.Dataset):
                if "void" in node.dtype.name:
                    ds_dict[node.name] = {
                        k: np.array(node.fields(k)).item() for k in node.dtype.names
                    }
                else:
                    ds_dict[node.name] = np.array(node[()])
                if verbose:
                    logging.info(f"Loaded dataset: {node.name}")
            if any(node.attrs):
                for key, val in node.attrs.items():
                    attr_dict[node.name + "/" + key] = val

    with h5py.File(file, "r") as hf:
        ds_dict = {}
        attr_dict = {}
        hf.visititems(get_ds_dictionaries, **kwargs)
    if key_sep:
        return dict_key_sep(ds_dict), dict_key_sep(attr_dict)
    return ds_dict, attr_dict


class DataImport(object):
    """
    A class to handle the import and processing of EIS data files.

    Attributes:
    file (str or Path): The name or path of the file to be imported.
    path (str or Path): The directory path where the file is located. Defaults to the current working directory.
    tool (str): The tool used for data acquisition. Default is "Agilent".
    read_type (str): The type of read operation to perform. Default is "full".

    Methods:
    __init__(file=None, path=None, tool="Agilent", read_type="full"):
        Initializes the DataImport object with the specified file, path, tool, and read_type.
    __getitem__(item):
        Allows indexing and slicing of the data attribute.
    """

    def __init__(self, file=None, path=None, tool="Agilent", read_type="full"):
        if isinstance(file, Path):
            path = str(file.parent)
            file = file.name
        if path is None:
            path = Path.cwd()
        self.file = file
        self.path = path
        self.tool = tool
        self.read_type = read_type

    def __getitem__(self, item):
        """Add indexing methods for Data"""
        if isinstance(item, (int, slice, np.integer)):
            return self.data[self._keys[item]]
        elif isinstance(item, str) and item in self.keys():
            return self.data[item]
        elif hasattr(self, item):
            return getattr(self, item)

    @property
    def sheets(self):
        """Pulls the actual data from the excel document."""
        if not hasattr(self, "_sheets"):
            self._sheets = load_file(self.file, self.path)[0]
            self._sheets.pop("Main", None)
            self._sheets.pop("MasterSheet", None)
        return self._sheets

    @sheets.setter
    def sheets(self, val):
        """Set input data into the sheets."""
        if not isinstance(val, dict):
            if isinstance(val, pd.DataFrame):
                self._sheets = {"Data": val}
                self._keys = ["Data"]
            elif isinstance(val, (list, np.ndarray)):
                self._sheets = {"Data": pd.DataFrame(val)}
                self._keys = ["Data"]
            else:
                raise TypeError(
                    "Data must be DataFrame, NP array, or list (or dict of said types)"
                )
        else:
            self._sheets = val
            self._keys = []
            for key, dat in val.items():
                self._keys.append(key)
                if isinstance(val, (list, np.ndarray)):
                    self._sheets[key] = pd.DataFrame(val)
                if not isinstance(val, pd.DataFrame):
                    raise TypeError("Data must be dict of DataFrames")

    @property
    def column_info(self):
        """Gets the column names from Agilent xls docs."""
        if not hasattr(self, "_column_info"):
            self._column_info = {
                key: pd.Series(val.loc[3:5, 11].to_numpy(), index=val.loc[3:5, 10])
                for key, val in self.sheets.items()
            }
        return self._column_info

    @property
    def raw_data(self):
        """Gets all data columns."""
        if not hasattr(self, "_raw_data"):
            if self.read_type.lower() == "shallow":
                return {}
            self._raw_data = {}
            for key, val in self.sheets.items():
                tmp = val.dropna(axis=1, thresh=int(val.shape[0] * 0.25)).dropna(
                    thresh=int(val.shape[1] * 0.25)
                )
                if tmp.isna().sum().sum() == 1 and np.isnan(tmp.iloc[0, 0]):
                    tmp = tmp.dropna(axis=1)
                if (
                    sum([isinstance(s, str) for s in tmp.iloc[0, :]])
                    > tmp.shape[1] * 0.75
                    and sum([isinstance(s, str) for s in tmp.iloc[0, :]])
                    > sum([isinstance(s, str) for s in tmp.iloc[1, :]]) * 2
                ):
                    self._raw_data[key] = pd.DataFrame(
                        tmp.iloc[1:, :].to_numpy(), columns=tmp.iloc[0, :], dtype=float
                    )
                else:
                    self._raw_data[key] = tmp
        return self._raw_data

    @property
    def data(self):
        """Returns the desired data in the form of freq and impedance (|Z| and phase)."""
        if not hasattr(self, "_data"):
            self.parse()
        return self._data

    def parse(self):
        """
        Parses the raw data based on the specified tool and read type.

        This method processes the raw data and converts it into a structured format
        suitable for analysis. It handles different tools and read types to ensure
        the data is parsed correctly.

        If the read type is "shallow", the method returns an empty dictionary.
        If the tool is "Agilent", the method processes the data to extract frequency
        and impedance (|Z| and phase) information.

        Returns:
        None
        """
        if self.read_type.lower() == "shallow":
            return {}
        if self.tool.lower() == "agilent":
            self._data = {
                key: pd.DataFrame(
                    {
                        "freq": val.iloc[:, 1].to_numpy(),
                        self.column_info[key]["Y1"]: Complex_Imp(val.iloc[:, [2, 3]]).Z,
                        self.column_info[key]["Y2"]: Complex_Imp(val.iloc[:, [4, 5]]).Z,
                    }
                )
                for key, val in self.raw_data.items()
            }
        if self.tool.lower() == "mfia":
            self._data = {
                key: pd.DataFrame(
                    {
                        "freq": val["frequency"].to_numpy(),
                        "real": val["realz"].to_numpy(),
                        "imag": val["imagz"].to_numpy(),
                    }
                ).sort_values("freq", ignore_index=True)
                for key, val in self.raw_data.items()
            }

        for key, val in self._data.items():
            for col in val.iloc[:, 1:].columns:
                if val[col].to_numpy().imag.sum() == 0:
                    val[col] = val[col].to_numpy().real

        return self._data

    def keys(self):
        """Return sheet names of excel file."""
        if not hasattr(self, "_keys"):
            self._keys = list(self.sheets.keys())
        return self._keys
