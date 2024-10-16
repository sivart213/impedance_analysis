# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import datetime
import numpy as np
# import pandas as pd

# from .string_operations import (
#     common_substring,
#     slugify,
#     str_in_list,
# )

# from dataclasses import dataclass, InitVar
def sig_figs_ceil(number, digits=3):
    """Round based on desired number of digits."""
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    root = 10 ** (int(power) - digits)
    return np.ceil(number / root) * root

def sanitize_types(data, decode_bytes=True):
    """
    Recursively sanitizes the types of elements in the input data.

    This function processes the input data, which can be a dictionary, list, tuple,
    numpy array, or basic data type, and converts its elements to standard Python types.
    Optionally, it can decode byte strings using various encodings.

    Parameters:
    data (any): The input data to sanitize. Can be a dictionary, list, tuple, numpy array,
                or basic data type (int, float, bool, complex, str, bytes).
    decode_bytes (bool, optional): If True, attempts to decode byte strings using various
                                   encodings. Default is True.

    Returns:
    any: The sanitized data with elements converted to standard Python types.
    """
    res = data
    # if iterables
    if isinstance(
        data, dict
    ):  # and all([isinstance(d, pd.DataFrame) for d in data.values()])
        res = {k: sanitize_types(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, np.ndarray)):
        res = [sanitize_types(d) for d in data]
        if len(res) == 1:
            res = res[0]

    # if a type that can be sanitized
    elif isinstance(data, (int, np.integer)):
        res = int(data)
    elif isinstance(data, (float, np.number)):
        res = float(data)
    elif isinstance(data, bool):
        res = bool(data)
    elif isinstance(data, complex):
        res = complex(data)
    elif isinstance(data, str):
        res = str(data)
    elif isinstance(data, bytes):
        res = bytes(data)
        if decode_bytes:
            encodes = [
                "utf_8",
                "utf_16",
                "utf_32",
                "utf_7",
                "cp037",
                "cp437",
                "utf_8_sig",
            ]
            res = data.decode("ascii")
            n = 0
            while n < len(encodes) and res not in str(data):
                try:
                    res = data.decode(encodes[n])
                    n += 1
                except UnicodeDecodeError:
                    n += 1

    return res

def convert_from_unix_time(arg, t_0=0):
    """
    Converts Unix time to datetime objects.

    This function converts Unix timestamps to Python `datetime` objects. It can handle
    single values, tuples, lists, and numpy arrays. An optional starting time `t_0` can
    be provided to offset the conversion.

    Parameters:
    arg (int, float, tuple, list, np.ndarray): The Unix time(s) to convert.
    t_0 (int, float, datetime.datetime, optional): The starting time to offset the conversion.
                                                   Default is 0.

    Returns:
    datetime.datetime or tuple or list or np.ndarray: The converted datetime object(s).
    """
    if isinstance(t_0, datetime.datetime):
        t_0 = t_0.timestamp()
    try:
        if isinstance(arg, (int, float)):
            return datetime.datetime.fromtimestamp(t_0 + arg)
        elif isinstance(arg, (tuple)):
            return tuple(datetime.datetime.fromtimestamp(t_0 + t) for t in arg)
        elif isinstance(arg, (list)):
            return [datetime.datetime.fromtimestamp(t_0 + t) for t in arg]
        elif isinstance(arg, (np.ndarray)):
            return np.array(
                [
                    (
                        datetime.datetime.fromtimestamp(t_0 + int(t))
                        if not np.isnan(t)
                        else np.nan
                    )
                    for t in arg
                ]
            )
    except OSError:
        return arg
    return arg


def find_nearest(array, target, index=True):
    """Get the nearest value in array or its index to target"""
    if not isinstance(target, (int, float, np.number)):
        if target is None or isinstance(target, str):
            return None
        return [find_nearest(array, targ, index) for targ in target]

    array = np.asarray(array)
    if np.all(array[:-1] <= array[1:]) and target <= max(array):
        idx = np.argmax(array >= target)
    else:
        idx = (np.abs(array - target)).argmin()

    if index:
        return idx
    else:
        return array[idx]

