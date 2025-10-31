# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


def nest_dict(data: dict, sep: str = "/") -> dict:
    """
    Splits concatenated keys in a dictionary into nested dictionaries using the specified separator.
    """
    if not isinstance(data, dict):
        return data

    def _insert(d, keys, value):
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    # Convert defaultdicts to dicts recursively
    def _to_dict(d):
        if isinstance(d, (defaultdict, dict)):
            return {k: _to_dict(v) for k, v in d.items()}
        return d

    result = nested_defaultdict()
    for key, val in data.items():
        if isinstance(key, tuple) or not isinstance(sep, str):
            keys = [k for k in key if k]
        else:
            keys = [k for k in key.split(sep) if k]
        _insert(result, keys, val)

    return _to_dict(result)


def flatten_dict(
    arg: dict,
    parent_key: str | tuple = "",
    sep: str | object = "/",
) -> dict:
    """
    Flattens a nested dictionary into a single dictionary, combining keys with a specified separator
    or as a tuple if sep is not a string.

    Parameters:
    - arg (dict): The nested dictionary to flatten.
    - parent_key (str or tuple, optional): The base key for the current level. Default is an empty string.
    - sep (str or any, optional): The separator to use for combining keys. If not a string, keys are returned as tuples.

    Returns:
    - dict: The flattened dictionary.
    """
    if not isinstance(arg, dict):
        return arg

    def combine_keys(*args):
        """
        Combines multiple keys into a tuple, flattening nested tuples/lists and filtering out invalid entries.
        """
        flat_args = []
        for arg in args:
            if isinstance(arg, (tuple, list)):
                flat_args.extend(tuple(arg))  # Flatten nested tuples/lists
            else:
                flat_args.append(arg)  # Add single keys directly
        # Filter out invalid entries (e.g., None or non-string types)
        return tuple([f_arg for f_arg in flat_args if isinstance(f_arg, str) and f_arg])

    items = []
    for k, v in arg.items():
        if isinstance(sep, str):
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
        else:
            new_key = combine_keys(parent_key, k)

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def slugify_name(
    value: str, allow_unicode: bool = False, sep: str = "-", max_length: int | float = 0
) -> str:
    """
    Inspired by https://github.com/django/django/blob/master/django/utils/text.py.

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
    # Remove non-alphanumeric characters except for the separator
    new_value = re.sub(r"[^\w\s-]", "", new_value.lower())
    slug = re.sub(rf"[{re.escape(sep)}\s-]+", sep, new_value).strip("-_")

    # If the slug is already within the limit, return it
    if not max_length or len(slug) <= max_length:
        return slug

    # Helper function to check and return if within limit
    def within_limit(s):
        return s if max_length * 0.75 <= len(s) <= max_length else ""

    # 1. Try removing separators
    if result := within_limit(slug.replace(sep, "")):
        return result

    # 2. Try removing a suffix if present
    if "." in value:
        sub_value = slugify_name(value.rsplit(".", 1)[0], allow_unicode, sep, max_length)
        if result := within_limit(sub_value):
            return result

    trim_target = len(slug) - max_length
    segments = slug.split(sep)
    if len(segments) > 1:
        len_candidate = slug
        num_candidate = slug
        for seg in segments:
            if trim_target <= len(seg) <= len(len_candidate):
                len_candidate = seg
            if seg.isnumeric() and trim_target <= len(seg) <= len(num_candidate):
                num_candidate = seg
        if num_candidate != slug:
            if result := within_limit(sep.join([s for s in segments if s != num_candidate])):
                return result
        if len_candidate != slug:
            if result := within_limit(sep.join([s for s in segments if s != len_candidate])):
                return result

    return slug[:max_length]


def trim_path_overlap(
    path: Path | str, name: Path | str, *, trim_name: bool = True, ensure_part: bool = True
) -> tuple[Path, Path]:
    """
    Remove overlap between `path` and `name`, retaining on only one side.

    Parameters
    ----------
    path : Path
        Base path.
    name : Path
        Candidate name/path to append.
    trim_name : bool, default True
        If True, trim overlap from `name`. If False, trim from `path`.
    ensure_part : bool, default True
        Ensure at least one part remains in both `path` and `name`.

    Returns
    -------
    (Path, Path)
        Adjusted (path, name) with overlap removed.
    """
    name, path = Path(name), Path(path)
    n_parts, p_parts = len(name.parts), len(path.parts)
    # If one is empty, nothing to trim
    if not p_parts or not n_parts:
        if ensure_part:
            if p_parts >= 2:
                return path.parent, Path(path.name)
            if n_parts >= 2:
                return name.parent, Path(name.name)
        return path, name

    if ensure_part and n_parts + p_parts == 2:
        return path, name  # Cannot trim both to empty -> may return overlap

    first = name.parts[0]
    if first not in path.parts:
        return path, name

    # last occurrence of first part in path
    idx = len(path.parts) - 1 - path.parts[::-1].index(first)
    subpath = Path(*path.parts[idx:])

    # Fast path: if anchors match, use is_relative_to
    try:
        if name.is_relative_to(subpath):
            rel = name.relative_to(subpath)
        else:
            return path, name
    except ValueError:
        # Fallback: compare parts directly
        overlap_len = len(path.parts) - idx
        if name.parts[:overlap_len] == path.parts[idx:]:
            rel = Path(*name.parts[overlap_len:])
        else:
            return path, name

    # Apply trimming
    if trim_name:
        # Trim from name
        new_name, new_path = rel, path
        if ensure_part and not new_name.parts:
            new_path, new_name = path.parent, Path(path.name)
    else:
        # Trim from path
        new_path, new_name = Path(*path.parts[:idx]), name
        if ensure_part and not new_path.parts:
            new_path, new_name = Path(new_name.parts[0]), Path(*new_name.parts[1:])

    return new_path, new_name


def path_overlap(
    path: Path | str,
    name: Path | str,
) -> Path:
    """
    Find overlap between `path` and `name`.

    Parameters
    ----------
    path : Path
        Base path.
    name : Path
        Candidate name/path to append.

    Returns
    -------
    (Path)
        Overlapping portion of (path, name).

    """
    name, path = Path(name), Path(path)

    p_parts = len(path.parts)
    if not len(name.parts) or not p_parts:
        return Path()

    first = name.parts[0]
    if first not in path.parts:
        return Path()

    # last occurrence of first part in path
    idx = p_parts - 1 - path.parts[::-1].index(first)
    if name.parts[: p_parts - idx] == path.parts[idx:]:
        return Path(*path.parts[idx:])

    return Path()
