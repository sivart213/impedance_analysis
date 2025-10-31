# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import unicodedata
from pathlib import Path

import numpy as np

# fmt: off
eng_dict = {
    -15: "f", -12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T",
}
eval_vars = {
    "np": np, "inf": np.inf, "nan": np.nan, "pi": np.pi, "e": np.e, "ln": np.log,
    "log": np.log, "log10": np.log10, "true": True, "false": False, "none": None,
}
# fmt: on


def safe_eval(
    value: str,
    assume_list_from_commas: bool = True,
    return_func_if_callable: bool = False,
    **kwargs,
) -> float | int | object:
    """Safely evaluate a string expression and return the result."""
    if not isinstance(value, str) or value in ["np", "ln", "log", "log10"]:
        return value

    # returns single type
    try:
        val = eval(value, kwargs, eval_vars)
        if callable(val):
            if return_func_if_callable:
                return val
            return value
        if isinstance(val, bool) or val is None:
            return val
        if isinstance(val, (float, int)):
            if not np.isnan(val) and abs(val) != np.inf and int(val) == val:
                return int(val)
            return float(val)
        return val
    except (SyntaxError, TypeError, ValueError, NameError):
        # Check if the value is a list, tuple, or dict
        if re.match(r"^[\[\(\{].*[\]\)\}]$", value):
            # Remove the outer brackets, parentheses, or braces and split the parts
            parts = re.findall(r"[^,\s]+", value[1:-1])
            # Determine the container type and return the evaluated parts
            if value.startswith("["):
                return [safe_eval(part, **kwargs) for part in parts]
            elif value.startswith("("):
                return tuple(safe_eval(part, **kwargs) for part in parts)
            elif value.startswith("{"):
                if ":" in parts[0]:
                    return {
                        safe_eval(k, **kwargs): safe_eval(v, **kwargs)
                        for k, v in (part.split(":") for part in parts)
                    }
                else:
                    return set(safe_eval(part, **kwargs) for part in parts)
        elif value.strip() != value:
            return safe_eval(value.strip(), **kwargs)
        elif assume_list_from_commas and len(parts := value.split(",")) > 1:
            return [safe_eval(part.strip(), **kwargs) for part in parts]
        else:
            # If none of the above, return the original value
            return value


def sci_note(num: float | int, prec: int = 2) -> str:
    """Return formatted text version of value."""
    if not isinstance(num, (float, int)) or np.isnan(num) or num == np.inf:
        return str(num)
    fmt = "{:.%dE}" % int(prec)
    return fmt.format(num)


def re_not(word: str, ign_case: bool = True) -> str:
    """
    Finds the closest matching string in a list of strings.

    This function searches for the closest matching string in a list of strings (`keys`)
    based on a given `key`. If an exact match is not found, it looks for partial matches
    with a minimum length specified by `min_len`. The function returns the closest match
    or the original key if no match is found.

    Parameters:
    key (str): The string to search for in the list.
    keys (list of str): The list of strings to search within.
    min_len (int, optional): The minimum length of the substring to consider for partial matches.
                             Default is 3.

    Returns:
    str: The closest matching string from the list, or the original key if no match is found.
    """
    if ign_case:
        return r"^((?!(?i:" + word + r")).)*$"
    return r"^((?!" + word + r").)*$"


def slugify(value: str, allow_unicode: bool = False, sep: str = "-") -> str:
    """
    Originally taken from https://github.com/django/django/blob/master/django/utils/text.py.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores. Replace whitespace with desired
    separator such as '-', '_', or ' '.
    """
    value = str(value)
    if allow_unicode:
        new_value = unicodedata.normalize("NFKC", value)
    else:
        new_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    # Remove non-alphanumeric characters
    new_value = re.sub(r"[^\w\s-]", "", new_value.lower())
    return re.sub(rf"[{re.escape(sep)}\s-]+", sep, new_value).strip("-_")


def eng_not(num: float | int, precision: int = 2, kind: str = "eng", space: str = "") -> str:
    """
    Converts a numeric value to a string in engineering notation.

    This function takes a numeric value and converts it to a string representation
    in engineering notation, with a specified precision and format kind. Engineering
    notation uses powers of 10 that are multiples of three (e.g., 1.23k for 1230).

    Parameters:
    val (float or int): The numeric value to convert.
    precision (int, optional): The number of decimal places to include in the output. Default is 2.
    kind (str, optional): The format kind for the output. Can be "eng" for engineering notation
                          or "exp" for exponential notation. Default is "eng".
    space (str, optional): The space to include between the number and the unit prefix. Default is " ".

    Returns:
    str: The string representation of the numeric value in engineering notation.
    """
    if not isinstance(num, (float, int)) or num <= 0 or np.isnan(num) or num == np.inf:
        return str(num)

    if "exp" in kind.lower():
        return str(int(np.floor(np.log10(num))))

    if "sci" in kind.lower():
        fmt = "{:.%de}" % int(precision)
        return fmt.format(num)

    if isinstance(space, bool) and space:
        space = " "

    exponent = int(np.floor(np.log10(num) / 3) * 3)
    fmt = "{:.%df}" % int(precision)
    if "eng" in kind.lower():
        return fmt.format(10 ** (np.log10(num) - exponent)) + str(space) + eng_dict[exponent]
    else:
        if len(kind) == 1 and kind.lower() == "e":
            if -3 < int(np.floor(np.log10(num))) < 3:
                return fmt.format(num)
            return fmt.format(10 ** (np.log10(num) - exponent)) + kind + str(exponent)
        return fmt.format(10 ** (np.log10(num) - exponent)) + f"e{exponent}"


def format_number(
    num: float | int | object,
    precision: int = 2,
    upper_exponent: float | int = 3,
    lower_exponent: float | int | None = None,
) -> str:
    """Format a number using general formatting with adjustable limits for scientific notation."""
    if (
        not isinstance(num, (float, int))
        or np.isnan(num)
        or num == np.inf
        or num == -np.inf
        or isinstance(num, bool)
    ):
        return str(num)
    if lower_exponent is None:
        lower_exponent = -1 * upper_exponent
    exponent = int(np.floor(np.log10(abs(num)))) if num != 0 else 0

    if lower_exponent < exponent < upper_exponent:
        if np.float64(num).is_integer():
            return f"{int(num):d}"
        return f"{num:.{max(0, precision-exponent)}f}".rstrip("0").rstrip(".")
    else:
        return f"{num/10**exponent:.{precision}f}".rstrip("0").rstrip(".") + f"e{exponent}"


def compile_search_patterns(search_terms: list, reject_terms: list | None = None) -> list:
    """
    Compiles search patterns from a list of search terms and optional reject terms.

    Parameters:
    - search_terms (list): A list of search terms (strings or tuples of strings).
    - reject_terms (list): A list of common reject terms (strings).

    Returns:
    - list: A list containing the desired groupings and residuals.
    """

    search_terms = [
        (re_not(term.replace("not ", "").strip()) if term.startswith("not ") else term.strip())
        for term in search_terms
    ]

    if isinstance(reject_terms, (tuple, list)):
        reject_terms = [
            (re_not(term.replace("not ", "").strip()) if term.startswith("not ") else term.strip())
            for term in reject_terms
        ]
        search_terms = [
            ([*t, *reject_terms] if isinstance(t, (tuple, list)) else [t, *reject_terms])
            for t in search_terms
        ]

    # Compile regex patterns for search terms
    patterns = [re.compile(combine_search_patterns(term), re.IGNORECASE) for term in search_terms]

    return patterns


def combine_search_patterns(term: str | tuple | list) -> str:
    """
    Compiles search patterns into a single regex string.

    Parameters:
    - term (str, tuple, or list): The search term(s) to compile. Can be a string,
      a tuple of (pattern, bool), or a list of such tuples.

    Returns:
    - str: A single regex string that combines the search patterns.
    """
    if isinstance(term, str):
        term = (term, True)  # Assume regex if no boolean provided
    if isinstance(term, (tuple, list)):
        if len(term) == 2 and isinstance(term[1], bool):
            term = [term]
        compiled_patterns = []
        term_sets = []
        for t in term:
            if isinstance(t, (tuple, list)):
                if len(t) == 2 and isinstance(t[1], bool):
                    term_sets.append(t)
                else:
                    raise ValueError("Invalid format for search term tuple: {}".format(str(t)))
            else:
                term_sets.append((t, True))  # Assume regex if no boolean provided
        for pattern, is_regex in term_sets:
            if is_regex:
                if "|" in pattern:
                    compiled_patterns.append(
                        "("
                        + "|".join(combine_search_patterns(part) for part in pattern.split("|"))
                        + ")"
                    )
                elif "?=" not in pattern and "?!" not in pattern:
                    compiled_patterns.append(
                        f"(?={pattern})" if pattern[:2] == ".*" else f"(?=.*{pattern})"
                    )
                else:
                    compiled_patterns.append(
                        pattern if pattern[0] == "(" else f"({pattern})"
                    )  # or pattern[:2] == "^("
            else:
                compiled_patterns.append(f"(?=.*{re.escape(pattern)})")
        return "".join(compiled_patterns)
    else:
        return term


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


if __name__ == "__main__":
    # %% Safe Eval test
    var_list = "[1,2.0,True,v1,log(5),log, pi]"
    var_list_open = "10,20.0,False,v2,log10(5),log10, e"
    var_math = "1.2e3 + 5e2"

    res_1 = safe_eval(var_list, return_func_if_callable=True)
    res_2 = safe_eval(var_list_open)
