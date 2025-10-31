# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import re
import inspect
import logging
from io import StringIO
from typing import Any
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime as dt
from collections import Counter

import h5py
import numpy as np
import pandas as pd

from .io_tools import nest_dict, flatten_dict, path_overlap, slugify_name

logger = logging.getLogger(__name__)


def save(
    data: pd.DataFrame | pd.Series | dict | Any,
    path: str | Path | None = None,
    name: str | None = None,
    file_type: str = "xls",
    mult_to_single: bool | None = None,
    file_modifier: str = "",
    attrs=None,
    sep: str = "_",
    merge_cells: bool = False,
    glob: str = "doc*",
    mode: str = "w",
    **kwargs,
):
    """
    Save data to a file (Excel or CSV) with flexible handling for dictionaries, DataFrames, and Series.

    Parameters:
    - data: The data to save. Can be a DataFrame, Series, or (nested) dictionary of DataFrames/Series.
    - path (str or Path, optional): Directory or file path to save to. If None, uses a default autosave path.
    - name (str, optional): Name for the file or directory. If None, uses a timestamp-based name.
    - file_type (str, optional): File extension/type to save as ("xls", "xlsx", "csv", etc.).
    - mult_to_single (bool, optional): If True, saves multiple items to a single file (Excel sheets).
    - file_modifier (str, optional): String to append to the file name.
    - attrs (DataFrame, dict, or bool, optional): Attributes to save alongside data.
    - sep (str, optional): Separator for slugify_nameing names.
    - merge_cells (bool, optional): Whether to merge cells in Excel output.
    - glob (str, optional): Glob pattern for autosave path search.
    - **kwargs: Additional keyword arguments passed to pandas save functions.

    Returns:
    - None
    """
    if data is None or len(data) == 0:
        return

    if path is None:
        # If no path is provided, use a default autosave directory in the user's home
        paths = list(Path().home().glob(glob))
        path = paths[0] if path else Path().home()
        path = path / "Autosave" / dt.now().strftime("%Y%m%d")
        del paths
    path = Path(path)

    if not path.drive:  # Check if the path has a valid drive
        path = Path().home() / path  # Use home directory as base path

    full_path = _make_full_path(path, name, file_type, sep, file_modifier)
    file_type = full_path.suffix[1:]

    as_excel = "xls" in file_type.lower()
    is_pandas = isinstance(data, (pd.DataFrame, pd.Series))

    data = _sanitize_save_data(data)

    if data is None:
        logger.warning("No data to save.")
        return

    # Handle attributes if data is a DataFrame or Series
    if (
        isinstance(data, (pd.DataFrame, pd.Series))
        and as_excel
        and mult_to_single
        and (data.attrs != {} or isinstance(attrs, (pd.DataFrame, dict)))
    ):
        # If attributes are present, wrap in a dict for saving as multiple sheets
        data = {name: data}
        is_pandas = False

    # Handle saving for simple DataFrame or Series
    if is_pandas:
        if as_excel:
            # Save directly as Excel
            save_to_excel(data, full_path, merge_cells=merge_cells, **kwargs)
            return
        else:
            # Save directly as CSV
            if mode == "a" and full_path.exists():
                kwargs["mode"] = "a"  # Append mode
                kwargs["header"] = False  # Do not write header if appending
            save_to_csv(data, full_path, **kwargs)
            return

    # Handle saving for dictionaries (possibly nested)
    if isinstance(data, (dict)):
        # If saving to Excel and mult_to_single is True, check for nested dicts
        if as_excel and (mult_to_single or mult_to_single is None):
            # If all values are dicts and the first subdict contains a DataFrame or another dict
            if all(isinstance(d, dict) for d in data.values()) and (
                isinstance(next(iter(next(iter(data.values())).values())), (pd.DataFrame, dict))
            ):
                # Recursively call save, appending keys to the path (as directories)
                full_path = _make_full_path(path, name, file_type, sep, as_dir=True)
                for key, sub_data in data.items():
                    save(
                        sub_data,
                        path=full_path,
                        name=key,
                        file_type=file_type,
                        mult_to_single=True,
                        sep=sep,
                        attrs=attrs,
                        merge_cells=merge_cells,
                        file_modifier=file_modifier,
                        **kwargs,
                    )

            else:
                # Data is a dict with DataFrame items; save each as a sheet
                if "attrs" not in data and isinstance(attrs, bool) and attrs:
                    attrs = pd.DataFrame()
                elif "attrs" not in data and isinstance(attrs, dict):
                    attrs = pd.DataFrame.from_dict(attrs, orient="index")
                elif not isinstance(attrs, pd.DataFrame) or "attrs" in data:
                    attrs = None

                # Use ExcelWriter to save to sheets
                full_path = _make_full_path(path, name, file_type, sep, file_modifier)
                if mode == "a" and full_path.exists():
                    data, attrs = merge_excel_sheets(full_path, data, attrs)

                with pd.ExcelWriter(full_path) as writer:
                    for key, df in data.items():
                        safe_df = _sanitize_save_data(df, parse_dict_of_datasets=True)
                        if not isinstance(safe_df, (pd.DataFrame, pd.Series)):
                            logger.warning("No data to save for key: %s", key)
                            continue
                        skey = slugify_name(key, sep=" ", max_length=31)
                        save_to_excel(
                            safe_df,
                            writer,
                            sheet_name=skey,
                            merge_cells=merge_cells,
                            **kwargs,
                        )
                        # Update the attrs DataFrame if not None and the DataFrame has attributes
                        if attrs is not None and safe_df.attrs:
                            new_attrs = pd.DataFrame.from_dict(
                                {skey: safe_df.attrs}, orient="index"
                            )
                            attrs = pd.concat([attrs, new_attrs], axis=0)

                    # Save the attrs DataFrame to a sheet named "attrs"
                    if attrs is not None and not attrs.empty:
                        save_to_excel(
                            attrs,
                            writer,
                            sheet_name="attrs",
                            merge_cells=merge_cells,
                            **kwargs,
                        )

        else:
            # Recursively call save for each key, creating subdirectories as needed
            full_path = _make_full_path(path, name, file_type, sep, as_dir=True)
            for key, sub_data in data.items():
                save(
                    sub_data,
                    path=full_path,
                    name=key,
                    file_type=file_type,
                    mult_to_single=False,
                    sep=sep,
                    attrs=attrs,
                    merge_cells=merge_cells,
                    file_modifier=file_modifier,
                    **kwargs,
                )


def merge_excel_sheets(
    existing_file: Path, new_data: dict, attrs: pd.DataFrame | None = None
) -> tuple[dict, pd.DataFrame | None]:
    existing_data, existing_attrs = load_excel(existing_file)
    merged_data = {**existing_data, **new_data}
    if attrs is not None:
        if existing_attrs is not None:
            merged_attrs = pd.concat([existing_attrs, attrs], axis=0).drop_duplicates()
        else:
            merged_attrs = attrs
    else:
        merged_attrs = existing_attrs
    return merged_data, merged_attrs


def _make_full_path(path, name=None, file_type=None, sep="_", file_modifier="", as_dir=False):
    """
    Construct a full file or directory path for saving data.

    Parameters:
    - path (str or Path): Base directory or file path.
    - name (str, optional): Name for the file or directory.
    - file_type (str, optional): File extension/type.
    - sep (str, optional): Separator for slugify_nameing names.
    - file_modifier (str, optional): String to append to the file name.
    - as_dir (bool, optional): If True, return a directory path instead of a file path.

    Returns:
    - Path: The constructed file or directory path.

    Notes:
    - suffix precedance is as follows: file_type > path suffix > name suffix > xlsx
    """
    # Convert to Path and extract suffix
    path = Path(path)
    suffix = path.suffix
    path = path.with_suffix("")  # Remove the suffix from the path

    # Parse name if not provided
    if not isinstance(name, (str, Path)) or not Path(name).parts:
        if as_dir:
            # If path is a dir without a name provided, return the path
            if not path.exists():
                # Create the directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
            return path
        if suffix:
            name = path.stem
            path = path.parent
        else:
            name = "data_" + dt.now().strftime("%H_%M")

    # Extract the suffix from the name and clean the name
    name = Path(name)
    suffix = suffix or name.suffix
    name = name.with_suffix("")

    if as_dir:
        file_modifier, suffix = "", ""
        n_suffix = 0
    else:
        if file_modifier:
            file_modifier = slugify_name(file_modifier, sep=sep)
        if isinstance(file_type, str):
            suffix = file_type.strip(".")
        else:
            suffix = suffix.strip(".") or "xlsx"
        if suffix.lower() == "xls":
            suffix = "xlsx"
        n_suffix = len(suffix) + 1

    name = name.with_stem(f"{name.stem}{file_modifier}")

    shared = path_overlap(path, name)

    if shared.parts:
        if path.exists():  # No need to check overlap section
            name = name.relative_to(shared)  # remove overlap from name
        else:  # Safer to make sure check the full name
            path = Path(*path.parts[: len(path.parts) - len(shared.parts)])

    # Budget for the full path
    n_parts = len(name.parts)
    suffix = "" if as_dir else suffix
    space = 250 - len(str(path)) - (n_parts - 1) - n_suffix

    # Extend path incrementally
    dir_path = path
    for i, part in enumerate(name.parts):
        if not (dir_path / part).exists():
            max_len = max(1, space - (n_parts - i - 1))
            part = slugify_name(part, sep=sep, max_length=max_len)
        dir_path = dir_path / part
        space -= len(part)

    if space < 0:
        raise ValueError("The constructed path exceeds the maximum length limit.")

    if as_dir:
        # If path is a dir, return with the name now appended
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    file_path = dir_path.with_suffix(f".{suffix}")

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path

    #     # Parse file_type
    # if not isinstance(file_type, str):
    #     if path_suffix:
    #         file_type = path_suffix[1:]
    #     elif name_suffix:
    #         file_type = name_suffix[1:]
    #     else:
    #         file_type = "xlsx"
    # else:
    #     file_type = file_type.strip(".")

    # if file_type.lower() == "xls":
    #     file_type = "xlsx"

    # # Remove redundant path parts
    # if name.is_relative_to(path):
    #     name = name.relative_to(path)

    # if path.stem == str(name) or path.stem == name.parts[0]:
    #     path = path.parent

    # # slugify_name name parts to ensure valid file names
    # max_length = 250 - len(str(path))
    # name = Path(*(slugify_name(p, sep=sep, max_length=max_length) for p in name.parts))
    # dir_path = path / name


def save_to_excel(data, full_path, merge_cells=False, **kwargs):
    """
    Helper function to save a DataFrame or Series to an Excel file.

    Parameters:
    - data (DataFrame or Series): Data to save.
    - full_path (str or Path or ExcelWriter): File path or ExcelWriter object.
    - merge_cells (bool, optional): Whether to merge cells in Excel output.
    - **kwargs: Additional keyword arguments for pandas to_excel.

    Returns:
    - None
    """
    data.to_excel(
        full_path,
        merge_cells=merge_cells,
        **kwargs,
    )


def save_to_csv(data, full_path, **kwargs):
    """
    Helper function to save a DataFrame or Series to a CSV file.

    Parameters:
    - data (DataFrame or Series): Data to save.
    - full_path (str or Path): File path.
    - **kwargs: Additional keyword arguments for pandas to_csv.

    Returns:
    - None
    """
    data.to_csv(
        full_path,
        **kwargs,
    )


def _sanitize_save_data(data, parse_dict_of_datasets=False):
    """
    Sanitize data for saving to file. Converts various data types to DataFrames or dicts as needed.

    Parameters:
    - data: The data to sanitize (DataFrame, Series, list, tuple, set, dict, etc.).
    - parse_dict_of_datasets (bool, optional): If True, convert dicts to DataFrames if possible.

    Returns:
    - DataFrame, dict, or None: Sanitized data ready for saving.
    """
    if data is None:
        return None
    # If already a DataFrame or Series, return as is
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data
    if isinstance(data, np.ndarray):
        return pd.Series(data)
    if isinstance(data, set):
        return pd.Series(list(data))
    if (isinstance(data, (str, bytes)) or not hasattr(data, "__iter__")) and data:
        # Convert single values to a DataFrame
        return pd.DataFrame([data])
    # Convert valid dictionaries to DataFrames if directed
    if isinstance(data, (dict)):
        if parse_dict_of_datasets and not any(
            isinstance(d, (pd.DataFrame, pd.Series, dict)) for d in data.values()
        ):
            # Convert dictionary to DataFrame as long as it does not contain DataFrames, Series, or dicts
            return pd.DataFrame.from_dict(data, orient="index")
        return data

    if isinstance(data, (list, tuple)):
        if any(isinstance(d, pd.DataFrame) for d in data):
            data = [
                d.to_frame().T if isinstance(d, pd.Series) else d
                for d in data
                if isinstance(d, (pd.Series, pd.DataFrame))
            ]

            if all(len(d) == 1 for d in data):
                # If all DataFrames have 1 row, combine them into a single DataFrame
                return pd.concat(data, ignore_index=True)
            else:
                # If DataFrames have more than 1 row, combine them into a dict
                return {str(i): d for i, d in enumerate(data)}
        elif all(isinstance(d, dict) for d in data):
            return {str(i): d for i, d in enumerate(data)}
        else:
            return pd.DataFrame(data)
    return None


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


def load_file(file, path=None, **kwargs) -> Any:
    """
    Loads data from Excel or HDF5 files.

    Parameters:
    - file (str or Path): The file to load.
    - path (str, Path, or list): The path to the file or a list of paths to search.
    - pdkwargs (dict): Additional keyword arguments to pass to pandas read functions.
    - hdf_kwargs (dict): Additional keyword arguments to pass to HDF5 read functions.
    - kwargs (dict): Additional keyword arguments.
        - flat_df (bool): Whether to flatten the DataFrame.
        - target (str): Target path for HDF5 files.

    Returns:
    - data (dict): Dictionary of DataFrames loaded from the file.
    - attrs (dict): Dictionary of attributes loaded from the file.
    """
    file = _handle_file_path(file, path)

    data = {}
    attrs = {}

    try:
        if re.search(r"(.xls|.xls\w)$", str(file)):
            data, attrs = load_excel(
                file,
                attach_file_stats=kwargs.pop("attach_file_stats", False),
                **filter_kwargs(pd.read_excel, kwargs),
            )
        elif re.search(r"(.h5|.hdf5)$", str(file)):
            data, attrs = load_hdf(
                file,
                None,
                kwargs.pop("target", "/"),
                kwargs.pop("key_sep", True),
                kwargs.pop("attach_file_stats", False),
                **filter_kwargs(load_excel, kwargs),
            )
            if kwargs.get("flat_df", False):
                data = flatten_dict(data)
        elif file.exists() and file.is_dir():
            if kwargs.get("recursive", False):
                # filelist = [p for p in Path.cwd().rglob(kwargs.get("glob", "*")) if p.is_file()]
                filelist = [p for p in file.rglob(kwargs.get("glob", "*")) if p.is_file()]
            else:
                # filelist = [p for p in Path.cwd().glob(kwargs.get("glob", "*")) if p.is_file()]
                filelist = [p for p in file.glob(kwargs.get("glob", "*")) if p.is_file()]
            if kwargs.pop("load_to_dict", False):
                # Load each file into a dict
                return {f.stem: load_file(f, None, **kwargs) for f in filelist}
            return [load_file(f, None, **kwargs) for f in filelist]
        else:
            try:
                data, attrs = load_csv(
                    file,
                    attrs_file=kwargs.pop("attrs_file", None),
                    attach_file_stats=kwargs.pop("attach_file_stats", False),
                    **filter_kwargs(pd.read_csv, kwargs),
                )
            except (pd.errors.ParserError, TypeError, ValueError, OSError, IOError):
                data, attrs = load_str(file, sep=kwargs.pop("sep", None))
    except Exception as exc:
        logger.error(
            "%s occurred in load_file when loading %s: %s", exc.__class__.__name__, file, exc
        )
        raise IOError(
            f"{exc.__class__.__name__} occurred in load_file when loading {file}"
        ) from exc

    return data, attrs


def load_hdf(file, path=None, target="/", key_sep=False, attach_file_stats=False, **kwargs):
    """
    Loads data from an HDF5 file and returns it as dictionaries.
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
    # logger = logging.getLogger(__name__)

    file = _handle_file_path(file, path)

    # Attach file stats if requested
    base_attrs = {}
    if attach_file_stats:
        base_attrs = get_file_stats(file)

    def get_ds_dictionaries(name: str, node: h5py.Group | h5py.Dataset):
        if target in name:
            if isinstance(node, h5py.Dataset):
                if "void" in node.dtype.name:
                    ds_dict[node.name] = {
                        k: np.array(node.fields(k)).item() for k in node.dtype.names
                    }
                else:
                    ds_dict[node.name] = np.array(node[()])
                logger.debug("Loaded dataset: %s", node.name)
            if any(node.attrs):
                for key, val in node.attrs.items():
                    attr_dict[str(node.name) + "/" + str(key)] = val

    with h5py.File(file, "r") as hf:
        ds_dict = {}
        attr_dict = (
            {**base_attrs, **{k: v for k, v in hf.attrs.items()}} if any(hf.attrs) else base_attrs
        )
        # attr_dict = {}
        hf.visititems(get_ds_dictionaries, **kwargs)
    if key_sep:
        return nest_dict(ds_dict), nest_dict(attr_dict)
    return ds_dict, attr_dict


def load_excel(file, path=None, **kwargs):
    """
    Loads data from an Excel file and returns it as dictionaries.

    Parameters:
    - file (str or Path): The Excel file to load.
    - kwargs (dict): Additional keyword arguments to pass to pandas read_excel function.

    Returns:
    - data (dict): Dictionary of DataFrames loaded from the file.
    - attrs (dict): Dictionary of attributes loaded from the file.
    """

    file = _handle_file_path(file, path)

    data = {}
    attrs = None

    # Load the Excel file to get sheet names
    excel_file = pd.ExcelFile(file)
    names = excel_file.sheet_names
    excel_file.close()

    # Determine headers and index rows using the first sheet
    first_sheet_name = names[0]
    kw_header = kwargs.pop("header", 0)
    kw_index_col = kwargs.pop("index_col", 0)
    kw_nrows = kwargs.pop("nrows", 10)
    attach_file_stats = kwargs.pop("attach_file_stats", False)

    # Load the attrs sheet if it exists
    if "attrs" in names:
        names.remove("attrs")
        attrs = pd.read_excel(file, sheet_name="attrs", header=0, index_col=0, **kwargs)

    try:
        test_df = pd.read_excel(
            file,
            sheet_name=first_sheet_name,
            header=kw_header,
            index_col=kw_index_col,
            nrows=kw_nrows,
            **kwargs,
        )
    except ValueError as exc:
        test_df = pd.DataFrame()
        # logger.error("Error loading %s", file)
        logger.error(
            "%s occurred in load_excel when loading %s: %s", exc.__class__.__name__, file, exc
        )

    # Default import
    header = kw_header
    index_rows = kw_index_col
    # Re-import the file with simpler arguments if null rows are detected
    if test_df.isnull().all(axis=1).any():
        test_df = pd.read_excel(file, sheet_name=first_sheet_name, header=None, nrows=10, **kwargs)

        # Parse header and index rows
        header, index_rows = _determine_header_and_index(test_df, kw_header, kw_index_col)

    if test_df.index.name is not None and not test_df.index.is_unique:
        index_rows = None

    # Load all sheets at once
    all_sheets = pd.read_excel(
        file, sheet_name=names, header=header, index_col=index_rows, **kwargs
    )

    # Attach file stats if requested
    base_attrs = {}
    if attach_file_stats:
        base_attrs = get_file_stats(file)

    for sheet_name, df in all_sheets.items():
        try:
            logger.debug("Processing sheet: %s", sheet_name)
            data[sheet_name] = df
            # Apply attrs to the respective DataFrame
            data[sheet_name].attrs = base_attrs
            new_attrs = (
                attrs.loc[sheet_name, :].to_dict()
                if isinstance(attrs, pd.DataFrame) and sheet_name in attrs.index
                else {}
            )
            data[sheet_name].attrs.update(new_attrs)
        except (KeyError, IndexError, TypeError):
            logger.warning("Sheet %s not found in attrs DataFrame", sheet_name)
            data[sheet_name] = df
            # Apply attrs to the respective DataFrame
            data[sheet_name].attrs = base_attrs

    return data, attrs


def load_csv(file, path=None, **kwargs):
    """
    Loads data from a CSV file and returns it as dictionaries.

    Parameters:
    - file (str or Path): The CSV file to load.
    - kwargs (dict): Additional keyword arguments to pass to pandas read_csv function.

    Returns:
    - data (dict): Dictionary of DataFrames loaded from the file.
    - attrs (dict): Dictionary of attributes loaded from the file.
    """
    file = _handle_file_path(file, path)

    data = {}
    attrs = {}

    # Determine headers and index rows
    kw_header = kwargs.pop("header", 0)
    kw_index_col = kwargs.pop("index_col", 0)
    kw_nrows = kwargs.pop("nrows", 10)
    sep = kwargs.pop("sep", _find_separators(file))

    # Extract attrs
    attrs_file = kwargs.pop("attrs_file", None)
    attach_file_stats = kwargs.pop("attach_file_stats", False)

    try:
        test_df = pd.read_csv(
            file, sep=sep, header=kw_header, index_col=kw_index_col, nrows=kw_nrows, **kwargs
        )
    except ValueError as exc:
        test_df = pd.DataFrame()
        # logger.error("Error loading %s", file)
        logger.error(
            "%s occurred in load_csv when loading %s: %s", exc.__class__.__name__, file, exc
        )

    # # Re-import the file with simpler arguments if null rows are detected
    # if test_df.isnull().all(axis=1).any():
    #     test_df = pd.read_csv(file, sep=sep, header=None, nrows=10, **kwargs)

    #     # Parse header and index rows
    #     header, index_rows = _determine_header_and_index(test_df, kw_header, kw_index_col)

    # else:
    #     # Default import
    #     header = kw_header
    #     index_rows = kw_index_col

    # Default import
    header = kw_header
    index_rows = kw_index_col
    # Re-import the file with simpler arguments if null rows are detected
    if test_df.isnull().all(axis=1).any():
        test_df = pd.read_csv(file, sep=sep, header=None, nrows=10, **kwargs)

        # Parse header and index rows
        header, index_rows = _determine_header_and_index(test_df, kw_header, kw_index_col)

    if test_df.index.name is not None and not test_df.index.is_unique:
        index_rows = None

    # Load the full CSV file
    data[file.stem] = pd.read_csv(file, sep=sep, header=header, index_col=index_rows, **kwargs)
    if data[file.stem].empty:
        new_sep = _find_separators(
            file, allow_sequence=True, prefer_char=False, allow_whitespace=True, exclude="-"
        )
        if new_sep != sep:
            data[file.stem] = pd.read_csv(
                file, sep=new_sep, header=header, index_col=index_rows, **kwargs
            )
        if data[file.stem].empty:
            raise IOError(f"File {file} is empty.")

    # # Check for an 'attrs' file (if applicable, e.g., a separate metadata file)
    # attrs_file = Path(file).with_suffix(".attrs.csv")
    if isinstance(attrs_file, str) and Path(attrs_file).exists():
        attrs_file = Path(attrs_file)
    if isinstance(attrs_file, (str, Path)) and (file / attrs_file).exists():
        attrs_file = file / attrs_file
    if isinstance(attrs_file, str):
        try:
            # Search for a file in the same directory with the keyword in its name
            potential_attrs_files = list(file.parent.glob(f"*{attrs_file}*"))
            if potential_attrs_files:
                # Select the file with the longest matching substring in the stem
                attrs_file = max(
                    potential_attrs_files,
                    key=lambda f: SequenceMatcher(None, file.stem, f.stem)
                    .find_longest_match(0, len(file.stem), 0, len(f.stem))
                    .size,
                )
            else:
                attrs_file = None  # No matching file found
        except TypeError:
            attrs_file = None

    if isinstance(attrs_file, Path) and attrs_file.exists():
        attrs_data = pd.read_csv(attrs_file, sep=_find_separators(attrs_file))
        if len(attrs_data) == 1:
            attrs[file.stem] = attrs_data.iloc[0].to_dict()
        elif file.stem in attrs_data.columns:
            attrs[file.stem] = attrs_data[file.stem].to_dict()
        elif file.stem in attrs_data.index:
            attrs[file.stem] = attrs_data.loc[file.stem].to_dict()
        else:
            attrs_dict = attrs_data.T.to_dict(orient="index")
            attrs[file.stem] = {k: list(v.values()) for k, v in attrs_dict.items()}
            # if attrs_data.shape[0] < attrs_data.shape[1]*0.25:
            # attrs_data = attrs_data.T
            # attrs[file.stem] = {str(i):v for i,v in enumerate(attrs_data.to_dict(orient="records"))}

    # Attach file stats if requested
    if attach_file_stats:
        attrs[file.stem] = {**get_file_stats(file), **attrs.get(file.stem, {})}

    # Apply attributes to the DataFrame
    if attrs:
        data[file.stem].attrs = attrs.get(file.stem, {})

    return data, attrs


def load_str(file, path=None, sep=None, attach_file_stats=True, **_):
    """
    Load a file and divide it into datablocks and comment blocks.

    Parameters:
    - file (str or Path): Path to the file to load.
    - sep (str, optional): Separator to use for splitting columns. If None, it will be inferred.

    Returns:
    - dict: A dictionary with keys "datablocks" and "comment_blocks", where:
        - "datablocks" is a list of datablock dictionaries.
        - "comment_blocks" is a list of comment block dictionaries.
    """
    file = _handle_file_path(file, path)

    # Read and clean up lines
    lines = _get_file_lines(file, start=0, n_lines=None)

    # Infer the separator if not provided
    if sep is None:
        sep = _find_separators(
            lines, allow_sequence=True, prefer_char=False, allow_whitespace=True, exclude="-"
        )

    # Group lines into blocks based on separator counts
    blocks = []
    current_block = []
    current_sep_count = None
    attrs = {}
    for line in lines:
        line_sep_count = line.count(sep)
        if current_sep_count is None or line_sep_count == current_sep_count:
            current_block.append(line)
        else:
            blocks.append({"count": current_sep_count, "lines": current_block})
            current_block = [line]
        current_sep_count = line_sep_count

    if current_block:
        blocks.append({"count": current_sep_count, "lines": current_block})

    # Separate datablocks and comment blocks
    datablocks, comment_blocks = [], []
    merged_comments = []
    average_length = sum(len(block["lines"]) for block in blocks) / len(blocks)

    if len(blocks) == 1:
        # If only one block, treat it as a datablock
        data = [_convert_block_to_dataframe(blocks[0]["lines"], sep)]
        keys = [file.stem]

    else:
        for block in blocks:
            if block["count"] > 1 and len(block["lines"]) > average_length:
                if merged_comments:
                    comment_blocks.append(merged_comments)
                    merged_comments = []
                datablocks.append(block["lines"])
            else:
                merged_comments.extend(block["lines"])

        if merged_comments:
            comment_blocks.append(merged_comments)

        # # Generate unique keys for the blocks
        # keys = _generate_keys(file, comment_blocks, len(datablocks))

        # Process datablocks and comments into DataFrames and attributes
        # data = {}
        # attrs = {}
        data = []
        attrs = []
        comments_sep = None

        # for key, data_lines, comments in zip(keys, datablocks, comment_blocks):
        for data_lines, comments in zip(datablocks, comment_blocks):
            # Determine header line
            first_line = data_lines[0]
            first_line_df = pd.read_csv(
                StringIO(first_line), sep=sep, header=None, engine="python"
            ).convert_dtypes()

            if all(dtype.name == "string" for dtype in first_line_df.dtypes):
                header_line = data_lines.pop(0)
            else:
                header_line = next(
                    (
                        comments.pop(j)
                        for j, comment in enumerate(comments)
                        if comment.count(sep) == first_line.count(sep)
                    ),
                    None,
                )
                if header_line is None:
                    num_columns = first_line.count(sep) + 1
                    header_line = sep.join([f"Column_{k+1}" for k in range(num_columns)])

            data_lines.insert(0, header_line)

            # Convert comments to dictionary
            comments_dict = _convert_comments_to_dict(comments, sep=comments_sep)
            if comments_sep is None:
                comments_sep = comments_dict.pop("comment_sep", None)

            # Attach file stats if requested
            if attach_file_stats:
                comments_dict.update(get_file_stats(file))

            # Convert data to DataFrame and attach attributes
            data_df = _convert_block_to_dataframe(data_lines, sep)
            data_df.attrs.update(comments_dict)

            # # Add to final output
            # data[key] = data_df
            # attrs[key] = comments_dict

            # Add to final output
            data.append(data_df)
            attrs.append(comments_dict)

        keys = _generate_keys(attrs, len(data), exclusions=file.stem)

    data = {key: data_df for key, data_df in zip(keys, data)}
    if attrs:
        attrs = {key: comments for key, comments in zip(keys, attrs)}

    return data, attrs


def _handle_file_path(file, path=None):
    """
    Standardizes file path handling by combining the file and path, if provided.

    Parameters:
    - file (str or Path): The file to handle.
    - path (str or Path, optional): The directory path to combine with the file.

    Returns:
    - Path: The resolved file path.
    """
    file = Path(file)

    if path is not None:
        path = Path(path)
        file = path / file

    if not file.exists():
        raise IOError(f"File {file} does not exist.")

    return file


def _get_file_lines(file, start=0, n_lines=None):
    """
    Helper function to read lines from a file with optional line limit and starting point.

    Parameters:
    - file (str, Path, or list of str): The file to read or the lines to check.
    - n_lines (int, optional): Maximum number of lines to read. If None, read all lines.
    - start (int, optional): Line number to start reading from. Default is 0.

    Returns:
    - list: A list of lines from the file.
    """
    try:
        if isinstance(file, str) and Path(file).exists():
            file = Path(file)

        if not isinstance(start, int):
            start = 0

        if isinstance(file, Path):
            if "xls" in file.suffix.lower():
                # Convert Excel sheet to CSV-like lines
                df = pd.read_excel(file, header=None, skiprows=start, nrows=n_lines)
                csv_data = df.to_csv(index=False, header=False)  # Export as CSV-like string
                csv_data = re.sub(r"[,\s]+$", "", csv_data, flags=re.MULTILINE)
                raw_lines = csv_data.splitlines()
            else:
                with open(file, "r", encoding="utf-8") as f:
                    # Skip lines up to the start point
                    for _ in range(start):
                        f.readline()
                    if n_lines is None:
                        # Read the whole file from the current position
                        raw_lines = f.readlines()
                    else:
                        # Read the specified number of lines
                        raw_lines = [f.readline() for _ in range(n_lines) if f.readline()]
        elif isinstance(file, str):
            raw_lines = file.splitlines()
        else:
            raw_lines = list(file)

        if start and start < len(raw_lines):
            raw_lines = raw_lines[start:]
        if isinstance(n_lines, int) and start + n_lines < len(raw_lines):
            raw_lines = raw_lines[: start + n_lines]
        # Return stripped lines
        return [line.strip() for line in raw_lines if line.strip()]
    except Exception as exc:
        logger.error("Error reading lines from %s: %s", file, exc)
        raise ValueError(f"Error reading lines from {file}") from exc


def _find_separators(
    file_or_lines,
    limit_occurrence=0,
    allow_sequence=False,
    prefer_char=True,
    allow_whitespace=True,
    exclude="",
):
    r"""
    Extract the most common separator from a file or list of lines.

    Parameters:
    - file_or_lines (str, Path, or list of str): The file or lines to extract separators from.
    - limit_occurrence (int): If > 0, limits the count of occurrences of a separator per line.
                              Default is 0 (no limit).
    - allow_sequence (bool): Whether to allow sequences of separators (e.g., `--` or `::`).
    - prefer_char (bool): If True, includes whitespace (`\s`) in the regex and prioritizes it in a secondary check.
    - allow_whitespace (bool): If False, excludes whitespace (`\s`) from the regex entirely.
    - exclude (str): Additional characters to exclude from being considered as separators.

    Returns:
    - str: The most common separator found, or `,` if no separator is found.
    """
    # Read lines from the file or use the provided list of lines
    lines = _get_file_lines(file_or_lines)

    if isinstance(limit_occurrence, (bool, float)):
        limit_occurrence = int(limit_occurrence)
    if not isinstance(limit_occurrence, int):
        limit_occurrence = 0

    # Build the regex pattern
    base_pattern = r"[^\w"  # Start with non-word characters
    if not allow_whitespace or prefer_char:
        base_pattern += r"\s"  # Exclude whitespace if not allowed or not preferred
    base_pattern += re.escape(exclude) + r"]"  # Add excluded characters
    if allow_sequence:
        base_pattern += r"+"  # Allow sequences of separators

    # Initialize a Counter for potential separators
    potential_seps = Counter()

    for line in lines:
        # Find all matches based on the regex
        matches = re.findall(base_pattern, line)
        if not matches and prefer_char and allow_whitespace:
            # If no matches found and prefer_char is True, check for whitespace sequences
            matches = re.findall(r"\s+", line)
        # If limit_occurrence > 0, only count separators that occur exactly `limit_occurrence` times
        if limit_occurrence > 0:
            for match in matches:
                if line.count(match) == limit_occurrence:
                    potential_seps[match] += 1
        else:
            # Otherwise, count all matches
            for match in matches:
                potential_seps[match] += 1

    # Select the most common separator
    if potential_seps:
        return potential_seps.most_common(1)[0][0]

    return ","


def _convert_comments_to_dict(comments, sep=None):
    """
    Convert a list of comment lines into a dictionary.

    Parameters:
    - comments (list of str): The comment lines to convert.
    - sep (str, optional): Separator to use for splitting lines. If None, it will be inferred.

    Returns:
    - dict: A dictionary with keys and values derived from the comment lines.
    """
    comm_sep = sep
    if sep is None:
        comm_sep = _find_separators(
            comments, limit_occurrence=1, allow_sequence=False, prefer_char=True
        )

    comment_dict = {}
    for i, line in enumerate(comments):
        if isinstance(comm_sep, str) and comm_sep in line:
            # Split the line at the first occurrence of the separator
            key, value = line.split(comm_sep, 1)
            comment_dict[key.strip()] = value.strip()
        else:
            # Use the index as the key if no separator is found in the line
            comment_dict[str(i)] = line.strip()

    # Add the inferred separator to the dictionary if it was determined
    if sep is None:
        comment_dict["comment_sep"] = comm_sep

    return comment_dict


def _convert_block_to_dataframe(block, sep=None):
    """
    Helper function to convert a block of lines into a DataFrame.

    Parameters:
    - block (list of str): Lines to convert.
    - sep (str): Separator to use for splitting columns.

    Returns:
    - pd.DataFrame: The resulting DataFrame.
    """
    # Join the block into a single string
    block_data = "\n".join(block)

    # Use StringIO to treat the string as a file
    if sep is not None:
        return pd.read_csv(StringIO(block_data), sep=sep)
    return pd.read_csv(StringIO(block_data), sep=sep, engine="python")


def get_file_stats(file):
    """
    Helper function to retrieve file stats safely, skipping unavailable attributes.

    Parameters:
    - file (Path): The file for which stats are to be retrieved.

    Returns:
    - dict: A dictionary containing file stats.
    """

    def _convert_to_datetime(file_stat, time_stat):
        """
        Converts a nanoseconds timestamp into a datetime object.

        Parameters:
        - ns_time (int or None): The nanoseconds timestamp to convert.

        Returns:
        - datetime or None: The converted datetime object, or None if input is None.
        """
        ns_time = getattr(file_stat, time_stat, None)
        if ns_time is not None:
            return dt.fromtimestamp(ns_time / 1e9)
        return None

    file = _handle_file_path(file)

    stats = {}

    try:
        f_stat = file.stat()
        stats["st_name"] = file.stem
        stats["st_path"] = str(file.resolve())
        stats["st_stype"] = file.suffix
        stats["st_parent"] = str(file.parent.resolve())
        stats["st_size"] = f_stat.st_size
        stats["st_uid"] = getattr(f_stat, "st_uid", None)
        stats["st_atime"] = _convert_to_datetime(f_stat, "st_atime_ns")
        stats["st_mtime"] = _convert_to_datetime(f_stat, "st_mtime_ns")
        stats["st_ctime"] = _convert_to_datetime(f_stat, "st_ctime_ns")
        stats["st_birthtime"] = _convert_to_datetime(f_stat, "st_birthtime_ns")

    except Exception as e:
        logger.warning("Could not retrieve file stats for %s: %s", file, e)
    return stats


def _determine_header_and_index(test_df, default_header, default_index_col):
    """
    Helper function to determine the header and index rows for a DataFrame.

    Parameters:
    - test_df (pd.DataFrame): A sample DataFrame loaded with limited rows.
    - default_header (int): Default header row.
    - default_index_col (int): Default index column.

    Returns:
    - header (int or list): Header row(s) to use.
    - index_rows (int or list): Index column(s) to use.
    """
    header_rows = 0
    for _, row in test_df.iterrows():
        if row.isnull().all():
            break
        header_rows += 1

    if header_rows < 10:
        # Multi-index DataFrame
        headers = test_df.loc[: header_rows - 1, :]
        index_rows = 0
        if any(headers.isna()):
            index_rows = int(headers.columns[headers.isna().any()][0])
            index_rows = list(range(index_rows - 1)) if index_rows > 1 else [0]
        header = list(range(header_rows))
    elif isinstance(test_df.loc[0, 0], (float, int)) and np.isnan(test_df.loc[0, 0]):
        # Basic DataFrame with index
        header = 0
        index_rows = 0
    else:
        # Default import
        header = default_header
        index_rows = default_index_col

    return header, index_rows


def _generate_keys(comment_blocks, num_blocks, possibilities=None, exclusions=None):
    """
    Generate unique keys for datablocks based on comment attributes.

    Parameters:
    - comment_blocks (list): List of comment blocks.
    - num_blocks (int): The total number of datablocks.
    - possibilities (list, optional): List of possible key names to consider (e.g., ["name", "title", "label"]).
                                       Defaults to ["name", "title", "label"] if None.
    - exclusions (str, list, optional): List of values to exclude from consideration. Defaults to None.

    Returns:
    - list: A list of unique keys for the datablocks.
    """
    if possibilities is None:
        possibilities = ["name", "title", "label"]
    if exclusions is None:
        exclusions = []
    elif isinstance(exclusions, str):
        exclusions = [exclusions]

    exclusions = [exclusion.lower() for exclusion in exclusions]
    # Collect possible keys based on the provided possibilities
    possible_keys = []
    keyless_keys = []
    for comment_block in comment_blocks:
        for key, value in comment_block.items():
            if key in possible_keys or key in keyless_keys:
                continue

            if isinstance(value, str) and value.lower() not in exclusions:
                if any(possibility in key.lower() for possibility in possibilities):
                    possible_keys.append(key)
                elif key.isnumeric() and num_blocks > 1:
                    keyless_keys.append(key)

    possible_keys.extend(keyless_keys)

    # Filter keys to find unique ones
    unique_keys = []
    for key in possible_keys:
        values = [comment_block.get(key) for comment_block in comment_blocks]
        if len(set(values)) == len(values):  # Ensure values are unique
            unique_keys.append(key)

    # Determine the best key to use if unique_keys are available
    best_key = None
    if unique_keys:
        # Find the key that most closely matches the first possibility
        best_key = min(
            unique_keys,
            key=lambda k: next(
                (possibility for possibility in possibilities if possibility in k.lower()),
                float("inf"),
            ),
        )

    # Generate keys for each block
    keys = []
    for i, comment_block in enumerate(comment_blocks):
        if best_key:
            keys.append(comment_block.get(best_key, str(i)))
        else:
            keys.append(str(i))

    return keys
