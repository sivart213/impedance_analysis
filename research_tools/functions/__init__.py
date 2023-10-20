# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:54:59 2022

@author: j2cle
"""
from .data_treatment import (
    all_symbols,
    BaseClass,
    Complexer,
    convert_prefix,
    convert_temp,
    convert_val,
    cost_base10,
    cost_basic,
    cost_log,
    cost_sqr,
    create_function,
    curve_fit_wrap,
    dict_df,
    dict_flat,
    dict_key_sep,
    dict_search,
    DictMixin,
    extract_arguments,
    extract_variable,
    find_nearest,
    format_time_str,
    function_to_expr,
    gen_bnds,
    gen_mask,
    get_const,
    has_arrays,
    has_symbols,
    has_units,
    insert_attr_row,
    myround,
    nprint,
    ode_bounds,
    parse_constant,
    parse_unit,
    pick_math_module,
    precise_round,
    print_to_txt,
    sample_array,
    sci_note,
    sig_figs_ceil,
    sig_figs_round,
    solve_for_variable,
)

from .plotters import (
    bode,
    bode2,
    lineplot_slider,
    map_plt,
    nyquist,
    nyquist2,
    scatter,
)

from .system_utilities import (
    f_find,
    get_config,
    load,
    load_hdf,
    p_find,
    pathify,
    pathlib_mk,
    PickleJar,
    save,
    slugify,
)


__all__ = [
    "all_symbols",
    "BaseClass",
    "Complexer",
    "convert_prefix",
    "convert_temp",
    "convert_val",
    "cost_base10",
    "cost_basic",
    "cost_log",
    "cost_sqr",
    "create_function",
    "curve_fit_wrap",
    "dict_df",
    "dict_flat",
    "dict_key_sep",
    "dict_search",
    "DictMixin",
    "extract_arguments",
    "extract_variable",
    "find_nearest",
    "format_time_str",
    "function_to_expr",
    "gen_bnds",
    "gen_mask",
    "get_const",
    "has_arrays",
    "has_symbols",
    "has_units",
    "insert_attr_row",
    "myround",
    "nprint",
    "ode_bounds",
    "parse_constant",
    "parse_unit",
    "pick_math_module",
    "precise_round",
    "print_to_txt",
    "sample_array",
    "sci_note",
    "sig_figs_ceil",
    "sig_figs_round",
    "solve_for_variable",

    "bode",
    "bode2",
    "lineplot_slider",
    "map_plt",
    "nyquist",
    "nyquist2",
    "scatter",

    "f_find",
    "get_config",
    "load",
    "load_hdf",
    "p_find",
    "pathify",
    "pathlib_mk",
    "PickleJar",
    "save",
    "slugify",
]


import sympy as sp
import sympy.physics.units as su
from sympy.physics.units.systems import SI

# adjust quantity latex/printing
su.e0._latex_repr = su.electric_constant._latex_repr = su.vacuum_permittivity._latex_repr = r'\epsilon_{\text{0}}'
su.avogadro._latex_repr = su.avogadro_constant._latex_repr = r'N_{A}'
su.boltzmann._latex_repr = su.boltzmann_constant._latex_repr = r'k_{B}'

# adjust entire quantity
su.q = su.elementary_charge = su.quantities.PhysicalConstant("elementary_charge", abbrev="q", latex_repr = r'q')
SI.set_quantity_dimension(su.elementary_charge, su.charge)
SI.set_quantity_scale_factor(su.elementary_charge, 1.602176634e-19*su.coulomb)

# create new unit
su.angstrom = su.angstroms = su.Quantity("angstrom", latex_repr=r'\dot{A}')
su.angstrom.set_global_relative_scale_factor(sp.Rational(1, 10**10), su.meter)
