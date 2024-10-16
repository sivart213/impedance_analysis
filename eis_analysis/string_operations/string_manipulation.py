# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import unicodedata
import numpy as np
# import sympy as sp

def sci_note(num, prec=2):
    """Return formatted text version of value."""
    fmt = "{:.%dE}" % int(prec)
    return fmt.format(num)

def re_not(word, ign_case=True):
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


def slugify(value, allow_unicode=False, sep="-"):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores. Replace whitespace with desired
    separator such as '-', '_', or ' '.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    # return re.sub(r"[-\s]+", "-", value).strip("-_")
    return re.sub(r"[-\s]+", sep, value).strip("-_")


def eng_not(num, precision=2, kind="eng", space=""):
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
    if num <= 0 or np.isnan(num) or num == np.inf or not isinstance(num, (float, int)):
        return str(num)

    if len(kind) > 2 and "exp" in kind.lower():
        return str(int(np.floor(np.log10(num))))

    eng_dict = {
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
    }

    if isinstance(space, bool) and space:
        space = " "

    exponent = int(np.floor(np.log10(num) / 3) * 3)
    fmt = "{:.%df}" % int(precision)
    if len(kind) > 2 and "eng" in kind.lower():
        return fmt.format(10 ** (np.log10(num) - exponent)) + str(space) + eng_dict[exponent]
    else:
        if len(kind) == 1 and kind.lower() == "e":
            return fmt.format(10 ** (np.log10(num) - exponent)) + kind + str(exponent)
        return fmt.format(10 ** (np.log10(num) - exponent)) + f"e{exponent}"
