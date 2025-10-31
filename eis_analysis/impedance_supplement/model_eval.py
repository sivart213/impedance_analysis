import re
from collections.abc import Callable

import numpy as np
from impedance.models.circuits.fitting import calculateCircuitLength

from .elements import ELEMENTS

MODEL_COMPARE_MODES = {
    "exact": lambda q, t: q == t,
    "substring": lambda q, t: q in t,
}


# %% Element/Param from model parsers
def extract_ckt_elements(circuit: str, convert: Callable = lambda x: "".join(x)) -> list:
    """
    Extract circuit elements from a circuit string as defined by `impedance.py`.

    Parameters:
    -----------
    circuit : str
        The circuit model string.
    convert : Callable
        A function to convert the regex match groups into a desired format (see notes).

    Returns:
    -----------
    list: A list of extracted circuit elements.

    Notes:
        - The regex pattern `r"([A-Za-z]+)(_?)(\\d+)"` splits the circuit string into:
            - Element type (see `ELEMENTS`)
            - Optional separator (underscore)
            - Parameter index (digits)
        - Utilyze the convert function to customize use of the 3 match groups.
        - The default convert function joins the regex match groups into a single string.
    """
    return [convert(r) for r in re.findall(r"([A-Za-z]+)(_?)(\d+)", circuit)]


def parse_parameters(model: str = "") -> list[str]:
    """Get the parameters of the model."""
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string.")
    if model.lower() == "linkk":
        return ["M", "mu"]
    # try:
    params = extract_ckt_elements(model)
    if len(params) != calculateCircuitLength(model):
        all_params = []
        for param in params:
            length = calculateCircuitLength(param)
            if length >= 2:
                all_params.append(f"{param}_0")
                for i in range(1, length):
                    all_params.append(f"{param}_{i}")
            else:
                all_params.append(param)
        params = all_params
    return params


# Unused
def num_parameters(model: str = "") -> int:
    """Get the number of parameters in the model."""
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string.")
    if model.lower() == "linkk":
        return 2
    return calculateCircuitLength(model)


def extract_first_number(text: str, prefix: str = "") -> int | float:
    """Extract the first number from a string, returning infinity if none found."""
    match = re.search(rf"{prefix}(\d+)", text)
    return int(match.group(1)) if match else float("inf")


# %% Model Validation (relies on extract_ckt_elements)
def validate_model_base(text):
    """
    Internal handler to validate the text when editing is finished.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Invalid input: must be a non-empty string.")

    if text.lower() == "linkk":
        return

    # Second validation: Check for unmatched parentheses
    if text.count("(") != text.count(")"):
        raise ValueError("Unmatched parentheses in the input.")

    if not re.match(r"^[A-Za-z].*[)\d]$", text):
        raise ValueError("Input must start and end with a valid element or parallel grouping.")

    if invalid := re.search(r"[,\-\(]\)", text):
        raise ValueError(f"Invalid sequence '{invalid.group()}' found in the input.")


def validate_model_structure(text: str):
    """
    Internal handler to validate the text when editing is finished.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Invalid input: must be a non-empty string.")

    if text.lower() == "linkk":
        return

    bases, elements = zip(*extract_ckt_elements(text, lambda x: (x[0], "".join(x))))

    if any(e not in ELEMENTS for e in bases):
        raise ValueError("Invalid element found in the input.")

    if any(e not in elements for e in extract_ckt_elements(text)):
        raise ValueError("Invalid element found in the input.")

    # Remove empty tokens caused by consecutive separators
    element_pattern = "|".join(re.escape(elem) for elem in elements)

    # Check for elements not at the start and not properly preceded
    improper_elements = re.findall(rf"(?<!^)(?<!p\()(?<!,)(?<!-)({element_pattern})", text)

    if improper_elements:
        raise ValueError(
            f"Element '{improper_elements[0]}' must be preceded by 'p(', ',', or '-', "
            "or be at the start of the model."
        )

    # Check that "p(" is also properly positioned (start, after ",", after "-")
    improper_p = re.findall(r"(?<!^)(?<!p\()(?<!,)(?<!-)p\(", text)
    if improper_p:
        raise ValueError("'p(' must be preceded by ',', '-', or be at the start of the model.")

    return text


def validate_model(text, verbose=False, raise_err=False):
    """
    Internal handler to validate the text when editing is finished.
    """
    try:
        validate_model_base(text)
    except ValueError as ve:
        if raise_err:
            raise ValueError(f"Invalid base aspect: {ve}") from ve
        elif verbose:
            print(f"Invalid base aspect: {ve}")
        return False
    try:
        validate_model_structure(text)
    except ValueError as ve:
        if raise_err:
            raise ValueError(f"Invalid structure aspect: {ve}") from ve
        elif verbose:
            print(f"Invalid structure aspect: {ve}")
        return False
    return True


# %% Model Evaluation
def simplify_model(circuit: str, idx_mode=0) -> str:
    """
    Simplify a circuit model string by replacing elements with their standard forms.

    Parameters:
    -----------
    circuit : str
        The circuit model string to simplify.
    idx_mode : int
        The simplification mode:
        0 - Replace all elements with "<element>".
        1 - Replace elements with "<element_n>" (n => element count).
        2 - Replace elements with "<element_n>" (n => component count).
        3 - Replace elements with "<element_n-m>" (n => element count, m => component index).

    Returns:
    -----------
    str: The simplified circuit model string.

    Notes:
        - mode 1 simplifies to the basic element types (e.g., R, C, L), ignoring numbering.
        - mode 2 simplifies to unique components, prioritizing numbering (e.g., R1, C2).
        - mode 3 is similar to mode 2 but retains the original component index for clarity.
            This is different than mode 2 whos numbering is based on order of appearance.
    """
    # Extract as (component, element, component index)
    elements = extract_ckt_elements(circuit, lambda x: ("".join(x), x[0], int(x[2])))
    simplified = circuit
    elem_map = {}
    for elem in elements:
        name = "<element>"
        if idx_mode == 1:  # Number by element (count)
            elem_map.setdefault(elem[1], len(elem_map))
            name = f"<element_{elem_map[elem[1]]}>"
        elif idx_mode == 2:  # Number by Component (count)
            elem_map.setdefault(elem[0], len(elem_map))
            name = f"<element_{elem_map[elem[0]]}>"
        elif idx_mode == 3:  # Number by Element (count) and Component Index
            elem_map.setdefault(elem[1], len(elem_map))
            name = f"<element_{elem_map[elem[1]]}-{elem[2]}>"
        simplified = simplified.replace(elem[0], name)
    return simplified


def model_compare(
    query: str, model: str, precision=0, mode: str | Callable[[str, str], bool] = "exact"
) -> bool:
    """
    Check if one circuit model is a sub-model of another.

    Parameters:
    -----------
    query : str
        The alternate or sub-model circuit string to check.
    model : str
        The main circuit model reference.
    precision : int
        The simplification mode (0, 1, or 2). See `simplify_model` for details.
    mode : str or Callable
        The comparison mode:
        - If a string, must be one of:
            - "exact": Check if the simplified sub-model exactly matches a part of the simplified main model.
            - "substring": Check if the simplified sub-model is a substring of the simplified main model.
        - If a callable, it should be a function that takes two strings (sub-model, main model) and returns a boolean.

    Returns:
    -----------
    bool: True or False depending on the mode.  Default is "exact" thus True if an exact match.
    """
    precision = np.clip(int(precision), 0, 2)
    simple_model = simplify_model(model, precision)
    simple_query = simplify_model(query, precision)
    if isinstance(mode, str):
        try:
            compare = MODEL_COMPARE_MODES[mode.lower()]
        except KeyError as ke:
            raise ValueError(f"Invalid mode '{mode}' specified.") from ke
    elif callable(mode):
        compare = mode
    else:
        raise ValueError("Mode must be a string or callable function.")
    return compare(simple_query, simple_model)


# def _renumberer(offset: int = 0):
#     """Return a function that renumbers circuit elements by a given offset."""

#     def wrapper(e_match):
#         return f"{e_match.group(1)}{offset + int(e_match.group(2))}"

#     return wrapper


# def shift_elem_num(
#     text: str,
#     offset: int | str = 1,
#     source: str = "",
# ) -> str:
#     """
#     Shift the numbering of circuit elements in a text string.

#     Parameters:
#     -----------
#     text : str
#         The text containing circuit elements to renumber.
#     offset : int
#         The initial offset to try. Default is 1.
#     source : str
#         Source model text to avoid conflicts with.
#         If provided, will find the smallest offset that doesn't result in repeated elements.

#     Returns:
#         str: The text with renumbered circuit elements.
#     """
#     if isinstance(offset, str):
#         num_range = extract_ckt_elements(offset, lambda x: int(x[2]))
#         offset = min(num_range) if num_range else 0
#     else:
#         offset = int(offset)
#     if source:
#         # Extract all elements and their numbers from text and source
#         text_elements = extract_ckt_elements(text, lambda x: (x[0] + x[1], int(x[2])))
#         source_elements = set(extract_ckt_elements(source, lambda x: (x[0] + x[1], int(x[2]))))

#         # Find the smallest offset with no conflicts
#         while True:
#             # Check if any element in text would conflict with source elements when using this offset
#             conflicts = False
#             for elem_type, num in text_elements:
#                 if (elem_type, num + offset) in source_elements:
#                     conflicts = True
#                     break

#             if not conflicts:
#                 break

#             offset += 1

#         # Apply the found offset to the text
#         func = _renumberer(offset)
#         return re.sub(r"([a-zA-Z]+_?)(\d+)", func, text)
#     else:

#         func = _renumberer(offset)
#         return re.sub(r"([a-zA-Z]+_?)(\d+)", func, text)
