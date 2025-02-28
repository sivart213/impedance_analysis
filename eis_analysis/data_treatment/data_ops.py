# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import datetime
import numpy as np
import pandas as pd
import sympy as sp
import sympy.physics.units as su


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

def convert_val(val, iunit=None, funit=None, expon=1):
    """
    Converts values from one unit to another by sanitizing the input and using sympy.convert_to()

    Parameters
    ----------
    val : [int, float, list/tuple/array/series of int/float]
        input value(s) to be converted
    iunit : [str, sympy.unit]
        current unit of val
    funit : [list, str]
        desired units
    expon : int
        exponential value of unit
        default : 1
    
    Returns
    -------
    val : [float, array/series of floats]
        returns float version of value using sympy.convert_to
    """
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        res = [convert_val(v, iunit, funit, expon) for v in val]
        if isinstance(val, np.ndarray):
            return np.array(res)
        if isinstance(val, pd.Series):
            return pd.Series(res)
        else:
            return res
    if val == 0:
        return val
    
    if isinstance(val, str):
        pattern = r'([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+)\^?([+-]?\d+)?'
        results = re.findall(pattern, val)[0]
        val = eval(results[0])
        iunit = results[1] if iunit is None or not iunit else iunit
        expon = results[2] if iunit is None or not iunit else expon
    
    if isinstance(iunit, str):
        iunit = getattr(su, iunit)
    # if funit is None:
    #     for unit in su.si.SI._base_units:
    #         if unit.dimension == iunit.dimension:
    #             funit = [unit]
    #             break
    
    if not isinstance(funit, list):
        funit = [funit]
    funit = [getattr(su, un) if isinstance(un, str) else un for un in funit]
    return float(su.convert_to(val * iunit**expon, funit).n().args[0])


# def sig_figs_ceil(number, digits=3):
#     """Round based on desired number of digits."""
#     digits = digits - 1
#     power = "{:e}".format(number).split("e")[1]
#     root = 10 ** (int(power) - digits)
#     return np.ceil(number / root) * root

# def find_nearest(array, target, index=True):
#     """Get the nearest value in array or its index to target"""
#     if not isinstance(target, (int, float, np.number)):
#         if target is None or isinstance(target, str):
#             return None
#         return [find_nearest(array, targ, index) for targ in target]

#     array = np.asarray(array)
#     if np.all(array[:-1] <= array[1:]) and target <= max(array):
#         idx = np.argmax(array >= target)
#     else:
#         idx = (np.abs(array - target)).argmin()

#     if index:
#         return idx
#     else:
#         return array[idx]

def get_const(name, symbolic=False, unit=None):
    """
    Call sympy to get the requested constant.

    Uses sympy syntax to get the necessary constant. Can return either the float value or sympy with
    units for unit evaluation.

    Typcial constants and their units:
        elementary_charge : C
        e0 : "farad", "cm"
        boltzmann : "eV", "K"


    Parameters
    ----------
    name : str
        The name of the constant as provided by sympy
    unit : list, optional
        Takes a list of sympy units to pass to convert_to which then returns the desired unit
    symbolic : bool
        Expressly stipulates whether units should be returned with the value

    Returns
    -------
    const : [float, sympy.unit.Quantity]
        requested value
    """
    name = name.strip()
    try:
        const = getattr(su, name)
    except AttributeError:
        if hasattr(sp, name):
            const = getattr(sp, name)
            if not symbolic:
                return float(const)
            else:
                return const
        else:
            raise AttributeError(f"Sympy has no attribute '{name}'")
    if unit is not None:
        # if isinstance(unit, (su.Quantity, list)):
        if not isinstance(unit, list):
            unit = [unit]
        try:
            unit = [
                getattr(su, un) if isinstance(un, str) else un for un in unit
            ]
            const = su.convert_to(const, unit).n()
        except (AttributeError, ValueError, TypeError):
            pass
        else:
            if not isinstance(const, su.Quantity):
                if symbolic:
                    return const
                else:
                    return float(const.args[0])
    if not symbolic:
        return float(const.scale_factor)
    return const