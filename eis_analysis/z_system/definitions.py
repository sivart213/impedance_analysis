# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re

# fmt: off
BASE_COMPLEX_FORMS = {
    "impedance", "admittance", "modulus", "capacitance", "resistivity", "conductivity",
    "permittivity", "relative_permittivity", "relative_permittivity_corrected", "susceptibility",
}

CONST_ALIASES = {
    "e_0": "permittivity_constant",
    "ε_0": "permittivity_constant",
    "ε₀": "permittivity_constant",
    "a_d": "area_over_thickness",
    "c_0": "characteristic_capacitance",
    "c₀": "characteristic_capacitance",
    "vacuum_capacitance": "characteristic_capacitance",
}
ARR_ALIASES = {
    "f": "frequency",
    "freq": "frequency",
    "w": "angular_frequency",
    "ω": "angular_frequency",
    "omega": "angular_frequency",
    "sigma_dc": "dc_conductivity",
    "σ_dc": "dc_conductivity",
}

COMP_ALIAS_SETS = {
    "impedance": {"z", "imp"},
    "admittance": {"y", "adm"},
    "capacitance": {"c", "cap"},
    "modulus": {"m", "mod"},
    "conductivity": {"σ", "cond", "sigma"},
    "resistivity": {"ρ", "rho", "resis"},
    "permittivity": {"ε", "e", "perm", "epsilon"},
    "relative_permittivity": {"εᵣ", "ε_r", "e_r", "epsilon_r", "perm_r", "permittivity_r"},
    "relative_permittivity_corrected": {"εᵣ_dc", "ε_r_dc", "e_rdc", "e_r_dc", "epsilon_r_dc", "perm_r_dc", "permittivity_r_dc"},
    "susceptibility": {"χ", "χₑ", "x_e", "chi", "chi_e"},
}

COMP_ALIASES = (
    {s: s for s in BASE_COMPLEX_FORMS}
    | {alias: key for key, aliases in COMP_ALIAS_SETS.items() for alias in aliases}
)

ALIASES = CONST_ALIASES | ARR_ALIASES | COMP_ALIASES

NEG_IMAG_FORMS = {"capacitance", "permittivity", "relative_permittivity", "relative_permittivity_corrected", "susceptibility"}

MOD_GRPS = {
    "cartesian": ["real", "imag"],
    "polar": ["mag", "phase"],
}

INDEP_SYMBOL_MAP = {
    "f": "frequency",
    "ω": "angular_frequency",
}

ARR_SYMBOL_MAP = INDEP_SYMBOL_MAP | {"σ_dc": "dc_conductivity"}

COMPLEX_SYMBOL_MAP = {
    "Z": "impedance",
    "Y": "admittance",
    "M": "modulus",
    "εᵣ": "relative_permittivity",
    "σ": "conductivity",
    "ρ": "resistivity",
    "C": "capacitance",
    "ε": "permittivity",
    "εᵣ_dc": "relative_permittivity_corrected",
    "χ": "susceptibility",
}

SYMBOL_MAP = INDEP_SYMBOL_MAP | COMPLEX_SYMBOL_MAP | {"σ_dc": "dc_conductivity"}

FUNC_MAP = {
    "ln": "ln ",
    "log": "log10 ",
    "ƒₛₚₗ": "interpolated ",
    "ƒₛₘ": "smoothed ",
    "∂": "derivative ",
}

COMPONENT_MAP = {
    "||": "mod",
    "'": "real",
    '"': "imag",
    "θ": "phase",
    "tan(δ)": "slope",
}


mode_parts = [
    (re.compile(r"(?P<var>\w+)\s?''"), "{var}.imag"),
    (re.compile(r"(?P<var>\w+)\s?\""), "{var}.imag"),
    (re.compile(r"(?P<var>\w+)\s?'"), "{var}.real"),
    (re.compile(r"(?P<var>\w+)\s?θ"), "{var}.phase"),
    (re.compile(r"(?P<var>\w+)\s?tan\(δ\)"), "{var}.slope"),
    (re.compile(r"\|\s?(?P<var>\w+)\s?\|"), "{var}.mag"),
]


# fmt: on+-
# SORTED_ALIASES = sorted(ALIASES.keys(), key=len, reverse=True)
SORTED_ALIASES = sorted(COMPLEX_SYMBOL_MAP.keys(), key=len, reverse=True)
# SORTED_COMPS = sorted(COMP_ALIASES.keys(), key=len, reverse=True) SYMBOL_MAP
# sorted_mode_keys = sorted(self.mode_mapping.keys(), key=len, reverse=True)

def parse_modes(text: str) -> str:
    # Parse modes
    for regex, template in mode_parts:
        text = regex.sub(lambda m: template.format(var=m.group("var")), text)
    return text

def parse_abs(text: str) -> str:
    return re.sub(r"\|\s*([^|]+)\s*\|", r"abs(\1)", text)

def parse_system_key(text: str) -> str:
    # Replace component aliases with full names
    text = parse_modes(text)
    text = parse_abs(text)
    for key in SORTED_ALIASES: # Contains all aliases, longest first
        text = text.replace(key, COMPLEX_SYMBOL_MAP[key])
    for key, repl in FUNC_MAP.items():
        text = text.replace(key, repl)
    text = text.replace(" ", "")
    # text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    # text = re.sub(r"\s*\(\s*", "(", text)  # Replace ' (' with '('
    # text = re.sub(r"\s*\)\s*", ")", text)  # Replace ' )' with ')'
    return text.strip()
