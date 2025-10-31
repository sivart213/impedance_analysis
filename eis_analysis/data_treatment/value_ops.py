# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import datetime as dt
from typing import Any, Literal, TypeAlias, overload
from difflib import get_close_matches

import numpy as np
import sympy as sp
import sympy.physics.units as su
from numpy.typing import ArrayLike

IsFalse: TypeAlias = Literal[False]
IsTrue: TypeAlias = Literal[True]
ValueTypes: TypeAlias = int | float | complex | str | bytes

encodes = [
    "ascii",
    "utf_8",
    "utf_16",
    "utf_32",
    "utf_7",
    "cp037",
    "cp437",
    "utf_8_sig",
]


def most_frequent(arg: list[ValueTypes] | tuple[ValueTypes, ...]) -> int:
    """
    Finds the most frequent element in an array.

    This function takes an array-like input and returns the most frequently occurring
    element. If there are multiple elements with the same highest frequency, it returns
    the first one encountered.

    Parameters:
    arg (array-like): The input array to analyze.

    Returns:
    int: The most frequent element in the array.
    """
    unique, counts = np.unique(arg, return_counts=True)
    index = np.argmax(counts)
    return int(unique[index])


@overload
def find_nearest(array: ArrayLike, target: Any, index: IsTrue = True, **kwargs) -> int: ...
@overload
def find_nearest(array: ArrayLike, target: float, index: IsFalse, **kwargs) -> float: ...
@overload
def find_nearest(array: ArrayLike, target: str, index: IsFalse, **kwargs) -> str: ...


def find_nearest(
    array: ArrayLike,
    target: Any,
    index: bool = True,
    **kwargs,
) -> Any:
    """
    Get the nearest value in array or its index to target.
    If index is True, returns the index (int).
    If index is False, returns the value (float or str).
    """
    # if target is None:
    #     return None

    # Allow recursion for array like targets
    if not isinstance(target, (int, float, np.number, str)):
        raise TypeError(
            f"Unsupported target type: {type(target)}. Expected int, float, np.number, or str."
        )
        # res = [find_nearest(array, targ, index) for targ in target]
        # return np.array(res) if isinstance(target, np.ndarray) else res

    array = np.asarray(array)

    # String operations
    if isinstance(target, str):
        # if any(not isinstance(t, str) for t in array):
        #     return None
        str_array = array.astype(str)
        if kwargs.get("ignore_case", False):
            target = target.lower()
            str_array = np.char.lower(str_array)
        closest_matches = get_close_matches(
            target, str_array, n=1, cutoff=kwargs.get("cutoff", 0.2)
        )
        if not closest_matches:
            raise ValueError(
                f"No close matches found for target '{target}' in the provided array."
            )
        idx = np.where(str_array == closest_matches[0])[0][0]

    # Numeric operations
    else:
        # if monotonically increasing and target is less than max(array), use argmax
        if np.all(array[:-1] <= array[1:]) and target <= max(array):
            idx = np.argmax(array >= target)
        # else subtract the target from the array and get the index of the minimum value
        else:
            idx = (np.abs(array - target)).argmin()

    try:
        return idx.item() if index else array[idx].item()
    except AttributeError:
        return idx if index else array[idx]


def sanitize_types(
    data: Any,
    decode_bytes: bool = True,
) -> Any:
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

    def sanitize_bytes(val):
        """
        Helper function to decode bytes if decode_bytes is True.
        """
        if decode_bytes:
            for enc in encodes:
                try:
                    return val.decode(enc)
                except UnicodeDecodeError:
                    continue
        return bytes(val)

    res = data
    # if iterables
    if isinstance(data, dict):  # and all([isinstance(d, pd.DataFrame) for d in data.values()])
        return {k: sanitize_types(v) for k, v in data.items()}

    if isinstance(data, (list, tuple, np.ndarray)):
        res = [sanitize_types(d) for d in data]
        return res[0] if len(res) == 1 else res
        # res = np.asarray(data).flatten().tolist()
        # if any(isinstance(d, bytes) for d in res):
        #     res = [sanitize_types(d, decode_bytes=decode_bytes) for d in res]
        # if len(res) == 1:
        #     res = res[0]
        # return res

    if isinstance(data, np.generic):
        return data.item()

    if isinstance(data, bytes):
        return sanitize_bytes(data)

    return res


def convert_unix_time_value(
    arg: str | dt.datetime | ArrayLike,
    t_0: int | float | dt.datetime = 0,
    use_first: bool = True,
) -> dt.datetime:
    """
    Converts a single Unix time value (or the first/last from an array-like) to a datetime object.

    Parameters:
    arg: int, float, datetime, tuple, list, or np.ndarray
        The Unix time(s) to convert.
    t_0: int, float, or datetime, optional
        The starting time to offset the conversion. Default is 0.
    use_first: bool, optional
        If arg is array-like, use the first value (True) or last value (False).

    Returns:
    datetime.datetime: The converted datetime object.
    """
    if isinstance(t_0, dt.datetime):
        t_0 = t_0.timestamp()
    # If already a datetime, return as is
    try:
        if isinstance(arg, dt.datetime):
            return dt.datetime.fromtimestamp(t_0 + arg.timestamp())
        # If scalar
        if isinstance(arg, (int, float)):
            if np.isnan(arg) or arg < 0:
                return dt.datetime.fromtimestamp(t_0)
            return dt.datetime.fromtimestamp(t_0 + arg)
        if isinstance(arg, str):
            val = dt.datetime.fromisoformat(arg).timestamp()
            return dt.datetime.fromtimestamp(t_0 + val)

        # If array-like
        # if isinstance(arg, (tuple, list, np.ndarray)):
        arr = np.asarray(a=arg).flatten()
        if arr[~np.isnan(arr)].size == 0:
            return dt.datetime.fromtimestamp(t_0)
        val = arr[~np.isnan(arr)][0 if use_first else -1]
        return dt.datetime.fromtimestamp(t_0 + float(val))
        # raise ValueError(f"Invalid Unix time value: {arg}")
    except (OSError, ValueError, OverflowError, TypeError) as exc:
        # Handle cases where the timestamp is out of range or invalid
        raise ValueError(f"Invalid Unix time value: {arg}") from exc


def convert_unix_time_array(
    arg: ArrayLike | str | dt.datetime,
    t_0: int | float | dt.datetime = 0,
) -> np.ndarray | list:
    """
    Converts Unix time values to datetime objects, returning an array or list.

    Parameters:
    arg: int, float, datetime, tuple, list, or np.ndarray
        The Unix time(s) to convert.
    t_0: int, float, or datetime, optional
        The starting time to offset the conversion. Default is 0.

    Returns:
    np.ndarray or list: The converted datetime objects.
    """
    if isinstance(t_0, dt.datetime):
        t_0 = t_0.timestamp()
    arr = np.asarray(arg).flatten()
    # Use np.vectorize to apply the scalar function
    vec_func = np.vectorize(lambda t: convert_unix_time_value(t, t_0), otypes=[object])
    result = vec_func(arr)
    # Return as np.ndarray if input was np.ndarray, else as list
    return result if isinstance(arg, np.ndarray) else result.tolist()


def convert_val(
    val: int | float | complex | str,
    i_unit: str | Any | None = None,
    f_unit: str | Any | None = None,
    exponent: int = 1,
) -> int | float:
    """
    Converts values from one unit to another by sanitizing the input and using sympy.convert_to()
    """
    e_val = get_unit_expr(val)

    if e_val.is_zero:
        return 0

    if i_unit is None:
        i_unit = 1
    i_unit = get_unit_expr(i_unit)

    if f_unit is None:
        f_unit = [su.meter, su.kilogram, su.second, su.ampere, su.mole, su.candela, su.kelvin]
    if not isinstance(f_unit, (list, tuple)):
        f_unit = [f_unit]
    f_unit = [getattr(su, un) if isinstance(un, str) else un for un in f_unit]
    return float(su.convert_to(e_val * i_unit**exponent, f_unit).n().args[0])


def get_unit_expr(value: ValueTypes, get_units: bool = False) -> sp.Expr:
    """
    Parses a string containing a numeric value and its units into a sympy expression.

    This function extracts the numeric value (if present) and sanitizes the unit string to create a
    sympy-compatible expression using sympy's parse_expr. If no numeric value is provided, it
    defaults to 1. The function also supports unit strings bracketed with [] or ().

    Parameters
    ----------
    value : str
        The input string containing a numeric value and units.

    Returns
    -------
    sympy.Expr
        A sympy expression representing the units.
    """
    try:
        value = float(value) if not isinstance(value, complex) else value
        return sp.sympify(value)
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str):
        raise ValueError(f"Unable to parse value and units from input: {value}")
    # Updated regex to allow units to be bracketed with [] or ()
    pattern = r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)?\s*[\[\(]?([a-zA-Z0-9/*^\s.-]+)[\]\)]?$"
    match = re.match(pattern, value)
    if not match:
        raise ValueError(f"Unable to parse value and units from input: {value}")

    numeric_value = (
        float(match.group(1)) if match.group(1) else 1
    )  # Default to 1 if no numeric value
    units_str = match.group(2).strip()  # Remove any spaces in the units

    # Sanitize the unit string
    units_str = units_str.replace("^", "**").replace(".", "*").replace(" ", "*")
    units_str = re.sub(r"([a-zA-Z]+)([+-]?\d+)", r"\1**\2", units_str)  # Add '**' for exponents

    # Parse the sanitized unit string into a sympy expression
    try:
        unit_expr = sp.parse_expr(
            units_str, local_dict={u: getattr(su, u) for u in dir(su) if not u.startswith("_")}
        )
    except Exception as e:
        raise ValueError(
            f"Error parsing units: {units_str}. Ensure units are valid sympy units."
        ) from e
    if get_units:
        return unit_expr
    return numeric_value * unit_expr


def get_const(name: str, symbolic: bool = False, unit: list | None = None) -> float | su.Quantity:
    """
    Call sympy to get the requested constant.

    Uses sympy syntax to get the necessary constant. Can return either the float value or sympy
    with units for unit evaluation.

    Typical constants and their units:
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
    except AttributeError as exc:
        if hasattr(sp, name):
            const = getattr(sp, name)
            if not symbolic:
                return float(const)
            else:
                return const
        else:
            raise AttributeError(
                f"AttributeError in get_const. Sympy has no attribute '{name}'"
            ) from exc
    if unit is not None:
        # if isinstance(unit, (su.Quantity, list)):
        if not isinstance(unit, list):
            unit = [unit]
        try:
            unit = [getattr(su, un) if isinstance(un, str) else un for un in unit]
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


# ARCHIVE
