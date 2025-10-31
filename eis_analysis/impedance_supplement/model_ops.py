import re

import numpy as np
import pandas as pd

from .elements import ELEMENT_MAP
from .model_eval import validate_model, extract_ckt_elements, extract_first_number


def _renumberer(offset: int = 0):
    """Return a function that renumbers circuit elements by a given offset."""

    def wrapper(e_match):
        return f"{e_match.group(1)}{offset + int(e_match.group(2))}"

    return wrapper


def shift_elem_num(
    text: str,
    offset: int | str = 1,
    source: str = "",
) -> str:
    """
    Shift the numbering of circuit elements in a text string.

    Parameters:
    -----------
    text : str
        The text containing circuit elements to renumber.
    offset : int
        The initial offset to try. Default is 1.
    source : str
        Source model text to avoid conflicts with.
        If provided, will find the smallest offset that doesn't result in repeated elements.

    Returns:
        str: The text with renumbered circuit elements.
    """
    if isinstance(offset, str):
        num_range = extract_ckt_elements(offset, lambda x: int(x[2]))
        offset = min(num_range) if num_range else 0
    else:
        offset = int(offset)
    if source:
        # Extract all elements and their numbers from text and source
        text_elements = extract_ckt_elements(text, lambda x: (x[0] + x[1], int(x[2])))
        source_elements = set(extract_ckt_elements(source, lambda x: (x[0] + x[1], int(x[2]))))

        # Find the smallest offset with no conflicts
        while True:
            # Check if any element in text would conflict with source elements when using this offset
            conflicts = False
            for elem_type, num in text_elements:
                if (elem_type, num + offset) in source_elements:
                    conflicts = True
                    break

            if not conflicts:
                break

            offset += 1

        # Apply the found offset to the text
        func = _renumberer(offset)
        return re.sub(r"([a-zA-Z]+_?)(\d+)", func, text)
    else:

        func = _renumberer(offset)
        return re.sub(r"([a-zA-Z]+_?)(\d+)", func, text)


# %% Model Cleaning
def clean_model_basic(model_string: str) -> str:
    """
    Basic cleaning of a model string by removing extra spaces and duplicate separators.
    Also ensures "p(" is lowercase and removes leading/trailing separators.

    Parameters:
    -----------
    model_string : str
        The model string to clean.

    Returns:
    -----------
    str: The cleaned model string.
    """
    if model_string.lower().strip() == "linkk":
        return model_string.lower().strip()

    current = model_string.replace(" ", "").replace("P(", "p(")
    return re.sub(r"([-,])\1+", r"\1", current).strip("-,")


def clean_model_general(model_string: str) -> str:
    """
    General cleaning of a model string by removing redundant elements and empty groups.

    Parameters:
    -----------
    model_string : str
        The model string to clean.

    Returns:
    -----------
    str: The cleaned model string.

    Notes:
        - Applies basic cleaning first.
        - Removes duplicate separators and empty parallel groups.
        - Cleans up commas around parentheses.
        - Flattens single-element parallel groups.
        - Repeats until no further changes occur.
    """
    current = clean_model_basic(model_string)

    if not current:
        return current

    while True:
        # Apply all cleaning operations
        init = current
        current = re.sub(r"([-,])\1+", r"\1", current)
        current = re.sub(r"-*,-*", ",", current)

        current = re.sub(r"[,-]*p\([,-]*\)[,-]*", "", current)  # Remove empty p(...) groups
        current = re.sub(r"[,-]+\)", ")", current)  # Clean up trailing commas before )
        current = re.sub(r"p\([,-]+", "p(", current)  # Clean up leading commas after p(
        current = re.sub(r"p\(([^,]+)\)", r"\1", current)

        if current == init:
            break
    return current


def clean_parallel_model(model_string):
    """
    Clean a circuit model string by handling nested parallel groups.

    Parameters:
    -----------
    model_string : str
        The circuit model string to clean.

    Returns:
    -----------
    tuple: A tuple containing the cleaned circuit model string and a mapping of parallel placeholders to their content.

    Notes:
        - Applies basic cleaning first.
        - Recursively processes innermost parallel groups outward.
        - Replaces parallel groups with placeholders and stores them in a mapping.
    """
    # First apply basic cleaning
    cleaned = clean_model_general(model_string)
    p_map = {}
    # Early return if the string doesn't contain any parallel groups
    if "p(" not in cleaned:
        return cleaned, p_map

    # Process the string from innermost parallel groups outward
    while True:
        # Find the innermost parallel groups (those with no nested p())
        inner_p = re.search(r"p\([^()]*\)", cleaned)
        if not inner_p:
            break

        # Store the cleaned content
        new_str = clean_model_general(inner_p.group(0))
        while "parallel" in new_str:
            init = new_str
            for p_holder in sorted(p_map.keys(), reverse=True, key=extract_first_number):
                new_str = new_str.replace(p_holder, p_map[p_holder])

            if new_str == init:
                break
        n_p = len(p_map)
        p_map[f"parallel{n_p}"] = clean_model_general(new_str)
        cleaned = cleaned.replace(inner_p.group(0), f"parallel{n_p}")

    return cleaned, p_map


def clean_model_full(model_string):
    """
    Fully clean a circuit model string by handling nested parallel groups.

    Parameters:
    -----------
    model_string : str
        The circuit model string to clean.

    Returns:
    -----------
    str: The fully cleaned circuit model string.

    Notes:
    --------
        - Applies basic cleaning first.
        - Recursively processes innermost parallel groups outward.
        - Replaces parallel groups with placeholders and restores them after cleaning.
    """
    # First apply basic cleaning
    cleaned = clean_model_general(model_string)
    # Early return if the string doesn't contain any parallel groups
    if "p(" not in cleaned:
        return cleaned

    cleaned, p_map = clean_parallel_model(cleaned)

    for placeholder in sorted(p_map.keys(), reverse=True, key=extract_first_number):
        cleaned = cleaned.replace(placeholder, p_map[placeholder])

    # Final cleaning pass to handle any new patterns
    cleaned = clean_model_general(cleaned)

    return cleaned


def clean_model_elements(text: str) -> tuple[str, list]:
    """
    Clean a model string and standardize element names to proper casing.

    Parameters:
    -----------
    text : str
        The model string to clean.

    Returns:
    -----------
    tuple: A tuple containing the cleaned model string and a list of standardized elements.
    """
    if not text:
        return "", []

    cleaned = clean_model_basic(text)
    elements = extract_ckt_elements(cleaned, lambda x: (x[0], f"{x[1]}{x[2]}"))
    try:
        new_elems = []
        for elem in elements:
            new = f"{ELEMENT_MAP[elem[0].lower()]}{elem[1]}"
            cleaned = cleaned.replace(f"{elem[0]}{elem[1]}", new)
            new_elems.append(new)
    except KeyError as ke:
        raise ValueError(f"Invalid element '{ke.args[0]}' found in the input.") from ke

    return cleaned, new_elems


# %% Model Group Parsing
def parse_model_groups(model, method: str = "series"):
    """
    Separate the model into top-level series groups or subgroups based on the first number of elements.

    Top-level series groups are defined as series-connected groups not within a parallel block.
    If `use_numbers` is True, subgroups are further split based on the first number of elements.

    Parameters:
    -----------
    model : str
        The circuit model string.
    use_numbers : bool
        Whether to split subgroups by the first number of elements.

    Returns:
        list: A list of sub-model strings.
    """
    model = clean_model_full(model)
    method = method.lower()

    p_model, p_map = clean_parallel_model(model)
    groups = p_model.split("-")

    if method not in ["series", "numbers"]:
        method = "series" if len(groups) > 1 else "numbers"

    # --- Split by top-level series groups first ---
    if method == "series":
        # Split the placeholder model by "-"
        groups = p_model.split("-")

        # Restore placeholders in each group
        for i, group in enumerate(groups):
            current = group
            # Restore all placeholders
            for placeholder, content in p_map.items():
                current = current.replace(placeholder, content)
            groups[i] = clean_model_general(current)  # Apply final cleaning

        # Sort groups as before
        groups.sort(key=lambda x: (extract_first_number(x), len(x)))
        return groups

    # --- Split by first number of elements ---
    subgroups = []
    elements = extract_ckt_elements(model)
    numbers = [int(re.sub(r"\D", "", e)[0]) for e in elements]

    # Create a DataFrame with elements and their group numbers
    df = pd.DataFrame(np.array([elements, numbers]).T, columns=["name", "grp_n"])

    # Iterate through unique group numbers
    for grp_n, group_df in df.groupby("grp_n"):
        gr_model = model

        # Remove invalid elements and their nearby syntax
        for invalid in df[df["grp_n"] != grp_n]["name"]:
            # Remove the invalid element and its surrounding syntax
            gr_model = re.sub(rf"(?<![a-zA-Z0-9]){re.escape(invalid)}", "", gr_model)

        gr_model = clean_model_full(gr_model)
        # Add the cleaned group to subgroups
        subgroups.append(gr_model)

    # Sort subgroups by the first number found and then by string length
    subgroups.sort(key=lambda x: extract_first_number(x))
    return subgroups


def get_valid_model(input_str):
    if not input_str:
        return "", []
    try:
        if validate_model(input_str):
            model, elems = clean_model_elements(input_str)
            return model, elems
        return "", []
    except Exception:
        return "", []


def get_valid_sub_model(main, sub_model):
    if not sub_model or not main:
        return "", []
    try:
        if validate_model(sub_model):
            sub_model, s_elems = clean_model_elements(sub_model)
            if sub_model not in main:
                return "", []
            main, m_elems = clean_model_elements(main)
            if not all(e in m_elems for e in s_elems):
                return "", []
            return sub_model, s_elems
        return "", []
    except Exception:
        return "", []
