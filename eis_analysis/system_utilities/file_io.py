# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import os
import re
# import sys
from datetime import datetime as dt
from pathlib import Path
import logging
import inspect
# import ctypes
# import itertools

# Third-party imports
import h5py
import numpy as np
import pandas as pd

# Local application imports
from ..dict_ops import dict_df, merge_single_key, dict_key_sep
from ..string_ops import slugify #, parse_path_str

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save(data, path=None, name=None, file_type="xls", **kwargs):
    """Save data into excel file."""
    path = Path(path)
    if path is None:
        path = list(Path().home().glob(kwargs.pop("glob", "doc*")))
        path = path[0] if path else Path().home()
        path = path / "Autosave" / dt.now().strftime("%Y%m%d")
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
    
    if isinstance(path, str):
        path = Path(path)
    if path is not None and isinstance(path, Path):
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
        if kwargs.get("recursive", False):
            filelist = [p for p in Path.cwd().rglob(kwargs.get("glob", "*")) if p.is_file()]
        else:
            filelist = [p for p in Path.cwd().glob(kwargs.get("glob", "*")) if p.is_file()]

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
    
    
    try:
        test_df = pd.read_excel(file, sheet_name=first_sheet_name, header=kw_header, index_col=kw_index_col, nrows=kw_nrows, **kwargs)
    except ValueError:
        test_df = pd.DataFrame()
        print(f"Error loading {file}")
        
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
    # if isinstance(path, Path):
    #     path = str(path)
    # if path is not None:
    #     file = path / file
    file = Path(file)
    if isinstance(path, str):
        path = Path(path)
    if path is not None and isinstance(path, Path):
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

