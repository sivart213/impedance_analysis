# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
# import unicodedata
# import numpy as np
import sympy as sp

def eval_string(text):
    """
    Parse text into None, int, float, tuple, list, or dict via sympy parse_expr.
    Parse_expr will attempt to resolve simple math expressions but will revert
    to str if result doesn't resolve to a number.  Does not support complex
    numbers.

    text : str
        The string of text to be converted if possible

    Returns
    -------
    res : any, str
        Returns an int, float, or string (or a tuple, list, set, or dict of base
        base types). Returns text as str in case of error.
    """
    if not isinstance(text, str):
        return text
    brackets = [["(", ")"], ["[", "]"], ["{", "}"]]
    ends = re.findall(r"^.|.$", text.strip())

    if ends in brackets:
        items = re.findall(
            r"[^\,\n]*:?[\(\[\{][^\)\]\}]+[\)\]\}]|[^\,\n]+",
            text.strip()[1:-1],
        )
        if all([":" in t for t in items]) and ends == brackets[2]:
            res = {}
            for item in items:
                it = [i.strip() for i in item.split(":")]
                res[it[0]] = eval_string(":".join(it[1:])) if len(it) > 1 else None
        else:
            res = [eval_string(item.strip()) for item in items]
            if ends == brackets[0]:
                res = tuple(res)
            if ends == brackets[2]:
                try:
                    res = set(res)
                except TypeError as err:
                    print(f"Error creating set: {err}")
        return res
    if text.lower() == "none":
        return None
    elif bool(re.findall(r"\d", text)):
        try:
            res = sp.parse_expr(text, transformations=sp.parsing.sympy_parser.T[5])
            return res
        except (TypeError, SyntaxError, KeyError, NameError):
            return text
    return text


def common_substring(strings, sep=""):
    """
    Returns the longest common substring from a list of strings.

    Parameters:
    strings (list of str): The list of strings to find the common substring from.

    Returns:
    str: The longest common substring found in all strings.
    """
    if not strings:
        return ""
    strings = [[s for s in re.split(sep, st) if s != ""] for st in strings]
    # Find the shortest string in the list
    shortest_str = min(strings, key=len)

    # Check all substrings of the shortest string
    for length in range(len(shortest_str), 0, -1):
        for start in range(len(shortest_str) - length + 1):
            substring = shortest_str[start : start + length]
            if all(sep.join(substring) in sep.join(s) for s in strings):
                substring = sep.join(substring)
                residuals = [
                    (
                        re.sub(substring + sep, "", sep.join(s))
                        if substring != sep.join(s)
                        else ""
                    )
                    for s in strings
                ]
                return [substring, residuals]

    return sep.join(shortest_str), [
        (
            re.sub(sep.join(shortest_str) + sep, "", sep.join(s))
            if sep.join(shortest_str) != sep.join(s)
            else ""
        )
        for s in strings
    ]


def str_in_list(key, keys, min_len=3):
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
    if key in keys:
        # return if perfect match
        return key
    else:
        if abs(min_len) >= len(key):
            min_len = 0
        n_keys = [k for k in keys if k in key and len(k) >= abs(min_len)] # check if a string in the list is wi
        n = None
        while len(n_keys) == 0 and abs(min_len) <= len(key[:n]):
            n_keys = [k for k in keys if key[:n] in k]
            if n is None:
                n = -1
            else:
                n -= 1

        if len(n_keys) == 0:
            return key
        elif isinstance(key, str):
            n_keys.sort(key=lambda x: len(x) - len(key))
        else:
            n_keys.sort()
        return n_keys[0]

    return key


    # elif any(key in k for k in keys):
    #     n_keys = [k for k in keys if key in k]
        
    #     if len(n_keys) == 0:
    #         return key
    #     elif isinstance(key, str):
    #         n_keys.sort(key=lambda x: len(x) - len(key))
    #     else:
    #         n_keys.sort()
    #     return n_keys[0]