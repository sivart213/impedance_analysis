# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import numpy as np
import pandas as pd
import sympy as sp
import sympy.physics.units as su

from inspect import getfullargspec
from dataclasses import dataclass, InitVar
from scipy.optimize import curve_fit


# %% Local Eqns
temperature_eqns = dict(
    FtoC=lambda T: (T - 32) * 5.0 / 9.0,
    FtoK=lambda T: (T - 32) * 5.0 / 9.0 + 273.15,
    FtoR=lambda T: T - 32 + (273.15 * 9.0 / 5.0),
    CtoF=lambda T: T * 9.0 / 5.0 + 32,
    CtoK=lambda T: T + 273.15,
    CtoR=lambda T: (T + 273.15) * 9.0 / 5.0,
    KtoF=lambda T: (T - 273.15) * 9.0 / 5.0 + 32,
    KtoC=lambda T: T - 273.15,
    KtoR=lambda T: T * 9.0 / 5.0,
    RtoF=lambda T: T + 32 - (273.15 * 9.0 / 5.0),
    RtoC=lambda T: (T * 5.0 / 9.0) - 273.15,
    RtoK=lambda T: T * 5.0 / 9.0,
)


# %% Equation Manipulators
def has_units(arg):
    return any(
        [
            len(v.atoms(su.Quantity)) != 0
            for v in arg.values()
            if isinstance(v, sp.Basic)
        ]
    )


def has_arrays(arg, func):
    symb_var = {
        k: sp.symbols(str(k), real=True)
        for k, v in arg.items()
        if isinstance(v, (np.ndarray, sp.Array))
    }
    if symb_var == {}:  # or any([isinstance(v, sp.Symbol) for v in arg.values()]):
        return False, None
    else:
        expr = func(**{**arg, **symb_var})
        try:
            return True, sp.lambdify(symb_var.keys(), expr)(
                **{k: arg[k] for k in symb_var.keys() if k in arg.keys()}
            )
        except (TypeError, ValueError):
            return True, expr


def has_symbols(arg):
    return any([isinstance(v, sp.Basic) for v in arg.values()])


def all_symbols(arg):
    return all([isinstance(v, sp.Basic) for v in arg.values()])


def pick_math_module(arg):
    module = np
    if has_symbols(arg):
        module = sp
    return module


def function_to_expr(func, **kwargs):
    """
    Call sympy to solve for the target variable.

    Uses sympy to rework the function, solving for a new variable. Requires the input function to
    use sympy compatable functions. i.e. use sp.sqrt vice np.sqrt.  Any non-sympy functions must
    not contain a variable.

    Parameters
    ----------
    target : str, sympy.Symbol
        The name of the target variable
    func : function
        callable function to be converted
    dep_var : str, sympy.Symbol
        name/symbol of the dependant variable/solution
    res_form : str
        Name of requested ouptut.
        Options :
            expr : result of function with sympy variables
            eqn : sympy.Eq of dep_var and expr
            res_set : sympy.solveset of eqn solved for target
            res : expression of res_set
    kwargs :
        Dictionary of any additional function arguments

    Returns
    -------
    res : varied
        requested value
    """
    argspec = getfullargspec(func)

    # Pull *args out of kwargs if named
    varargs = kwargs.pop(argspec[1], [])
    kargs = kwargs.pop("args", {})

    # Pull any dictionarys to top level
    kkeys = list(kwargs.keys())
    [kwargs.update(kwargs.pop(k)) for k in kkeys if isinstance(kwargs[k], dict)]

    # get function arguments as symbols
    if isinstance(kargs, dict):
        args = [
            kargs.pop(a, kwargs.pop(a, sp.symbols(a, real=True))) for a in argspec[0]
        ]
        kwonlyargs = {
            a: kargs.pop(a, kwargs.pop(a, sp.symbols(a, real=True))) for a in argspec[4]
        }
    else:
        args = [kwargs.pop(a, sp.symbols(a, real=True)) for a in argspec[0]]
        kwonlyargs = {a: kwargs.pop(a, sp.symbols(a, real=True)) for a in argspec[4]}
        if isinstance(kargs, int):
            varargs = kwargs.pop(argspec[1], kargs)
            kargs = [kargs]
        else:
            kargs = list(kargs)
            args[len(args) - len(kargs[: len(args)]) :] = [
                sp.symbols(str(a), real=True) for a in kargs[: len(args)]
            ]

    varargs = kwargs.pop(argspec[1], kwargs.pop("args", varargs))
    if argspec[1] is None:
        varargs = []
    elif len(kargs) > len(args):
        varargs = kargs[len(args) :]
    elif varargs == [] and "args" in kkeys:
        varargs = list(kwargs.values())

    varkw = {}
    if argspec[2] is not None:
        varkw = kwargs

    if varargs != []:
        # If args is not empty, try to parse
        if isinstance(varargs, (list, tuple, np.ndarray)) and len(varargs) == 1:
            varargs = varargs[0]
        if isinstance(varargs, int):
            sign = int(-varargs / abs(varargs))
            varargs = [
                sp.symbols(f"C_{a+sign}", real=True) for a in range(varargs, 0, sign)
            ]
        elif isinstance(varargs, str):
            varargs = [sp.symbols(a, real=True) for a in varargs.split(" ")]
        elif isinstance(varargs, (list, tuple, np.ndarray)):
            varargs = [sp.symbols(str(a), real=True) for a in varargs]
        elif isinstance(varargs, (dict)):
            varargs = [sp.symbols(str(a), real=True) for a in varargs.values()]

    expr = func(*args, *varargs, **kwonlyargs, **varkw)
    if not hasattr(expr, "free_symbols"):
        return expr
    res_dict = {
        a: eval(str(a))
        for a in expr.free_symbols
        if not bool(
            re.findall("[a-df-zA-DF-Z\\\\@!&^]|^[/eE]|[/eE]$|^\\..+\\.$", str(a))
        )
    }
    res = expr.subs(res_dict)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def solve_for_variable(func, target=None, dep_var="res", res_form="res", **kwargs):
    """
    Call sympy to solve for the target variable.

    Uses sympy to rework the function, solving for a new variable. Requires the input function to
    use sympy compatable functions. i.e. use sp.sqrt vice np.sqrt.  Any non-sympy functions must
    not contain a variable.

    Parameters
    ----------
    target : str, sympy.Symbol
        The name of the target variable
    func : function
        callable function to be converted
    dep_var : str, sympy.Symbol
        name/symbol of the dependant variable/solution
    res_form : str
        Name of requested ouptut.
        Options :
            expr : result of function with sympy variables
            eqn : sympy.Eq of dep_var and expr
            res_set : sympy.solveset of eqn solved for target
            res : expression of res_set
    kwargs :
        Dictionary of any additional function arguments

    Returns
    -------
    res : varied
        requested value
    """
    if callable(func):
        expr = function_to_expr(func, **kwargs)

    elif isinstance(func, sp.Basic):
        expr = func

    if res_form.lower() == "expr" or not isinstance(expr, sp.Basic):
        return expr

    if isinstance(dep_var, str):
        dep_var = sp.symbols(dep_var, real=True)

    eqn = sp.Eq(dep_var, expr)
    if res_form.lower() == "eqn" or target is None:
        return eqn

    if isinstance(target, str):
        target = sp.symbols(target, real=True)

    res_set = sp.solveset(eqn, target)
    res = res_set.args
    if len(res) > 0:
        res = res[0]
    try:
        return vars()[res_form]
    except KeyError:
        return res


def create_function(func, targ, var=None, cost=None, **kwargs):
    """
    Call sympy to convert function for use in fitting.

    Uses sympy to rework the function, converting the inputs. Can wrap the function inside a cost
    function for solving. Target value is placed first in new function. Constants may be inserted
    into function however arrays are ignored. Symbols can also be replaced.

    Parameters
    ----------

    func : function
        callable function to be converted
    targ : str, sympy.Symbol
        The name of the target variable
    var : str, sympy.Symbol
        name/symbol of the dependant variable/solution
    cost : function
        callable function to wrap result for minimization

    kwargs : dict
        Dictionary of any additional function arguments to be treated as constants

    Returns
    -------
    res : varied
        requested value
    """
    if isinstance(targ, str):
        targ = sp.symbols(targ, real=True)
    if var is None:
        var = targ
    if isinstance(var, str):
        var = sp.symbols(var, real=True)

    if callable(func):
        expr = function_to_expr(func)

    elif isinstance(func, sp.Basic):
        expr = func

    args = {}

    if hasattr(expr, "free_symbols"):
        syms = extract_arguments(expr, method="union", args=[var])
        args = {s: dict_search(kwargs, str(s), s) for s in syms}
    else:
        return func

    if callable(cost):
        expr = cost(expr, var)

    expr = expr.subs(args)

    syms = list(expr.free_symbols)
    if targ in syms:
        syms.insert(0, syms.pop(syms.index(targ)))

    res = sp.lambdify(syms, expr)
    return res


def extract_arguments(func, method="str", **kwargs):
    """
    Call sympy to convert function for use in fitting.

    Uses sympy to rework the function, converting the inputs. Can wrap the function inside a cost
    function for solving. Target value is placed first in new function. Constants may be inserted
    into function however arrays are ignored. Symbols can also be replaced.

    Parameters
    ----------

    func : function
        callable function to be converted
    targ : str, sympy.Symbol
        The name of the target variable
    var : str, sympy.Symbol
        name/symbol of the dependant variable/solution
    cost : function
        callable function to wrap result for minimization

    kwargs : dict
        Dictionary of any additional function arguments to be treated as constants

    Returns
    -------
    res : varied
        requested value
    """
    if callable(func):
        expr = function_to_expr(func)
    elif isinstance(func, sp.Basic):
        expr = func

    ignore = kwargs.pop("ignore", ["kwargs", "args"])
    if not isinstance(ignore, (list, tuple)):
        ignore = []

    args = []
    syms = []
    if hasattr(expr, "free_symbols"):
        syms = [a for a in list(expr.free_symbols) if str(a) not in ignore]
    for k, v in kwargs.items():
        args.append(sp.symbols(str(k), real=True))
        if isinstance(v, dict):
            args = args + extract_arguments(func, method="dict", **v)
        elif isinstance(v, (list, tuple)):
            args = args + [
                sp.symbols(str(a), real=True)
                for a in v
                if isinstance(a, (str, sp.Symbol))
            ]
    args = [a for a in args if str(a) not in ignore]

    res = syms
    if "dict" in method.lower() or "args" in method.lower():
        res = args

    if "union" in method.lower():
        res = list(set(syms + args))
    elif "inter" in method.lower():
        res = [r for r in syms if r in args]
    elif "comp" in method.lower():
        res = [r for r in list(set(syms + args)) if r not in res]

    if "str" in method.lower():
        return [str(r) for r in res]
    return res


def curve_fit_wrap(fcn, pnts, **params):
    """Calculate. generic discription."""
    pnts = np.array(pnts)
    params = {
        **{"method": "trf", "x_scale": "jac", "xtol": 1e-12, "jac": "3-point"},
        **params,
    }
    fit = curve_fit(fcn, pnts[:, 0], pnts[:, 1], **params)
    return [fit[0], np.sqrt(np.diag(fit[1]))]


# %% Converters
def convert_val(val, iunit, funit, expon=1):
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
    if isinstance(iunit, str):
        iunit = getattr(su, iunit)
    if not isinstance(funit, list):
        funit = [funit]
    funit = [getattr(su, un) if isinstance(un, str) else un for un in funit]
    return float(su.convert_to(val * iunit**expon, funit).n().args[0])


def convert_prefix(val, i_pre, f_pre="", expon=1):
    pre_list = su.prefixes.PREFIXES
    if len(i_pre) <= 1:
        p_val = pre_list.get(i_pre, None)
    else:
        p_val = pre_list.get(i_pre, pre_list.get(i_pre[:-1], None))
        if p_val is None and hasattr(su, i_pre):
            p_val = getattr(su, i_pre)
        elif p_val is None and hasattr(su, i_pre[:-1]):
            p_val = getattr(su, i_pre[:-1])
    if p_val is None:
        return val
    res = float(val * p_val.scale_factor)
    if f_pre != "":
        res = res / convert_prefix(1, f_pre)

    return precise_round(res)


def convert_temp(val, iunit, funit, expon=1):
    """Convert input value to kelvin then to desired temperature unit."""
    if len(iunit) > 1:
        val = convert_prefix(val, iunit[:-1])
        iunit = iunit[-1]

    p_val = 1
    if len(funit) > 1:
        p_val = convert_prefix(1, funit[:-1])
        funit = funit[-1]

    res = (
        temperature_eqns[f"{iunit.upper()}to{funit.upper()}"](val ** (1 / expon))
        / p_val
    )

    res = res**expon

    return precise_round(res)


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


def parse_constant(const, unit_system="SI"):
    if not isinstance(const, su.quantities.PhysicalConstant):
        return const
    dims = [getattr(su, str(d)) for d in const.dimension.atoms(sp.Symbol)]
    if isinstance(unit_system, str):
        unit_system = getattr(su.systems, "SI")
    return su.convert_to(
        const, [unit_system.derived_units[d] for d in dims], unit_system
    )


def parse_unit(expr, unit_system="SI"):
    if isinstance(expr, sp.Number) or not isinstance(expr, sp.Basic):
        return float(expr)
    if isinstance(unit_system, str):
        unit_system = getattr(su.systems, "SI")
    consts = list(expr.atoms(su.quantities.PhysicalConstant))
    expr = expr.subs({c: parse_constant(c, unit_system) for c in consts}).simplify()

    units = list(expr.atoms(su.Quantity))
    pows = list(expr.atoms(sp.Pow))
    unitless = expr.subs({k: 1 for k in units})
    symbs = unitless.subs({x: 1 for x in unitless.atoms(sp.Number)})
    pow_units = [x for x in pows if isinstance(x.args[0], su.Quantity)]
    for po_un in pow_units:
        for n in range(len(units)):
            if list(po_un.atoms(su.Quantity))[0] == units[n]:
                units[n] = po_un
    var = expr.subs(np.prod(units), 1)
    if var.subs({x: 1 for x in var.atoms(sp.Number)}) == symbs:
        units = np.prod(units)
    for n in range(len(units.atoms(su.Quantity))):
        if len(units.atoms(su.Quantity)) == 1:
            break
        munits = [su.convert_to(units, u) for u in unit_system.get_units_non_prefixed()]
        for t in munits:
            if len(t.atoms(su.Quantity)) < len(units.atoms(su.Quantity)):
                units = t
    return unitless, units


def sci_note(num, prec=2):
    """Return formatted text version of value."""
    fmt = "{:.%dE}" % int(prec)
    return fmt.format(num)


def precise_round(val, max_precision=15):
    prec = np.log10(abs(val))
    if abs(prec) == np.inf:
        return val
    return round(val, int(max_precision - np.ceil(prec)))


def sig_figs_round(number, digits=3):
    """Round based on desired number of digits."""
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    return round(number, -(int(power) - digits))


def sig_figs_ceil(number, digits=3):
    """Round based on desired number of digits."""
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    root = 10 ** (int(power) - digits)
    return np.ceil(number / root) * root


def myround(x, base=5):
    """Calculate. generic discription."""
    return base * round(float(x) / base)


def closest(K, lst):
    """Calculate. generic discription."""
    # lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]


def myprint(filename, *info):
    """Calculate. generic discription."""
    print_file = open(filename, "a+")
    args = ""
    for arg in info:
        args = args + str(arg)
    print(args)
    print(args, file=print_file)
    print_file.close()
    return


def find_nearest(array, target, index=True):
    """Get the nearest value in array or its index to target"""
    if not isinstance(target, (int, float, np.number)):
        if target is None or isinstance(target, str):
            return None
        return [find_nearest(array, targ, index) for targ in target]

    array = np.asarray(array)
    if np.all(array[:-1] <= array[1:]) and target <= max(array):
        idx = np.argmax(array >= target)
    else:
        idx = (np.abs(array - target)).argmin()

    if index:
        return idx
    else:
        return array[idx]


def sample_array(array, get_index=False, **kwargs):
    """Parse large array based on given requirements."""
    if not isinstance(array, (tuple, list, np.ndarray)) or len(array) <= 1:
        return array

    arr_n_max = find_nearest(array, kwargs.get("arr_max", max(array)))
    if arr_n_max is None or arr_n_max == 0:
        arr_n_max = len(array)
    arr_max = array[arr_n_max]
    array = array[: arr_n_max + 1]

    arr_step = kwargs.get("arr_step", None)
    if arr_step is None:
        # Get max value of array and ensure max(array) less than desired maximum value
        arr_size = min(int(kwargs.get("arr_size", len(array))), len(array))
        arr_step = arr_max / (arr_size - 1)
    else:
        if arr_max % arr_step != 0:
            arr_steps = [x for x in range(int(2 * arr_step), 0, -1) if arr_max % x == 0]
            if len(arr_steps) != 0:
                arr_step = arr_steps[0]
        arr_size = min(int(kwargs.get("arr_size", len(array))), len(array))
        arr_size = max(arr_size, int(arr_max / arr_step + 1))

    # get index value (argmax) using ratios (val/val_max = n/n_max)
    if arr_step >= 1:
        arr_ind_vals = np.arange(arr_max + arr_step, step=arr_step)
    else:
        arr_ind_vals = np.linspace(min(array), arr_max, int(arr_max / arr_step + 1))

    arr_ind = find_nearest(array, arr_ind_vals)

    # Get array by index where the index is less than the desired length
    res = array[arr_ind[:arr_size]]

    if get_index:
        return res, arr_ind
    return res


# %% Dict operations
def dict_key_sep(data, sep="/"):
    if not isinstance(data, dict):
        return data
    blank = {}
    for key, val in data.items():
        pre = ""
        if key[0] == sep:
            pre = sep
        keys = key.split(sep)
        subkey = pre + sep.join(keys[2:])
        if len(subkey) > 1:
            if keys[1] not in list(blank.keys()):
                blank[keys[1]] = {}
            blank[keys[1]][subkey] = val
        else:
            try:
                blank[keys[1]] = val
            except IndexError:
                blank[key] = val
    return {k: dict_key_sep(v) for k, v in blank.items()}


def dict_flat(data):
    if isinstance(data, dict):
        if len(data) <= 1:
            return dict_flat(list(data.values())[0])
        else:
            return {k: dict_flat(v) for k, v in data.items()}
    else:
        return data


def dict_search(data, key, default=None):
    if isinstance(data, dict):
        if key in data.keys() or not any([isinstance(v, dict) for v in data.values()]):
            return data.get(key, default)
        for k, v in data.items():
            res = dict_search(v, key)
            if res is not None:
                return res
    return default


def dict_df(data, single=True):
    if not isinstance(data, dict):
        return data
    try:
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            if all([isinstance(v, dict) for v in val.values()]):
                data[key] = dict_df(val, single)
            else:
                vals = {
                    k: v
                    for k, v in val.items()
                    if not isinstance(v, dict)
                    and isinstance(v, (np.ndarray, list, tuple))
                }
                vals_items = {
                    k: v
                    for k, v in val.items()
                    if not isinstance(v, (np.ndarray, list, tuple, dict))
                }
                vals_dicts = {
                    k: dict_df(v, single) for k, v in val.items() if isinstance(v, dict)
                }
                if single and len(vals) > 0:
                    vlen = max([len(v) for v in vals.values()])
                    data[key] = pd.DataFrame(
                        {str(kk): vv for kk, vv in vals.items() if len(vv) == vlen}
                    )
                else:
                    tmp_new = {
                        str(len(v)): pd.DataFrame(
                            {kk: vv for kk, vv in vals.items() if len(vv) == len(v)}
                        )
                        for k, v in vals.items()
                    }
                    if len(vals_items) > 0:
                        attrs = pd.Series(vals_items)
                    if len(tmp_new) + len(vals_dicts) == 0:
                        data[key] = attrs
                    else:
                        data[key] = {**tmp_new, **vals_dicts, **{"attrs": attrs}}
    except (AttributeError, ValueError) as e:
        print(e)
        return data

    try:
        dlens = [v.size for v in data.values()]
        if all([d == max(dlens) for d in dlens]):
            return pd.DataFrame(data)
        else:
            return data
    except (AttributeError, ValueError) as e:
        print(e)
        return data


# %% Fitting functions
def gen_mask(self, mask, arr):
    if isinstance(mask, (list, np.ndarray)) and len(mask) != arr:
        mask = None
    if mask is None:
        mask = np.full_like(arr, True, dtype=bool)
    elif isinstance(mask, (float, np.float, int, np.integer)):
        mask = arr < mask
    elif callable(mask):
        mask = mask(arr)
    return mask


def gen_bnds(arr, dev=0.1, dev_type=" ", abs_bnd=[(0, np.inf)], max_bnd=False):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, float)
    arr = arr[~np.isnan(arr)]

    if isinstance(dev, (float, int, np.integer, np.float)) or len(dev) == 1:
        dev = np.array([dev] * len(arr), float)
    if not isinstance(dev, np.ndarray):
        dev = np.array(dev)
    if not isinstance(dev_type, list):
        dev_type = list(dev_type)
    if len(dev_type) != len(dev):
        dev_type = [dev_type[0]] * len(dev)
    if len(abs_bnd) != len(dev):
        abs_bnd = [abs_bnd[0]] * len(dev)
    dev = abs(dev)
    amin = arr * 0.9
    amax = arr * 1.1
    if len(dev) != len(arr):
        print("Error: bad bound input")
        return (amin, amax)
    for n in range(len(arr)):
        if (int(dev[n]) == dev[n] and dev_type[n] == " ") or "exp" in dev_type[
            n
        ].lower():
            # intergers are assumed to be log variation
            amin[n] = 10 ** (np.log10(arr[n]) - dev[n])
            amax[n] = 10 ** (np.log10(arr[n]) + dev[n])
        elif (
            (abs(np.log10(arr[n]) - np.log10(dev[n])) <= 2 and dev_type[n] == " ")
            or "std" in dev_type[n].lower()
            or "dev" in dev_type[n].lower()
        ):
            # assumes the value is a std dev
            amin[n] = arr[n] - dev[n]
            amax[n] = arr[n] + dev[n]
        else:
            # floats are assumed to be percentages
            amin[n] = arr[n] * (1 - dev[n])
            amax[n] = arr[n] * (1 + dev[n])
    amin = [
        amin[m] if amin[m] >= abs_bnd[m][0] else abs_bnd[m][0] for m in range(len(amin))
    ]
    amax = [
        amax[m] if amax[m] < abs_bnd[m][1] else abs_bnd[m][1] for m in range(len(amax))
    ]

    if max_bnd:
        amin = np.array([0 if np.log10(an) >= 0 else an for an in amin])
        amax = np.array([1 if np.log10(ax) < 0 else ax for ax in amax])
    return (amin, amax)


def cost_basic(res, targ=0, func=None, **kwargs):
    if callable(func):
        if targ is None:
            targ = func(**kwargs)
        else:
            res = func(res, **kwargs)
    return res - targ


def cost_sqr(res, targ=0, func=None, **kwargs):
    if callable(func):
        if targ is None:
            targ = func(**kwargs)
        else:
            res = func(res, **kwargs)
    return (res - targ) ** 2


def cost_log(res, targ=1, func=None, **kwargs):
    if callable(func):
        if targ is None:
            targ = func(**kwargs)
        else:
            res = func(res, **kwargs)
    return (np.log10(res) - np.log10(targ)) ** 2


def cost_base10(res, targ=0, func=None, **kwargs):
    res = 10**res
    if callable(func):
        if targ is None:
            targ = func(**kwargs)
        else:
            res = func(res, **kwargs)
    return (res - targ) ** 2


@dataclass
class Complexer(object):
    """Calculate. generic discription."""

    data: InitVar[np.ndarray] = np.ndarray(0)
    name: str = "Z"

    def __post_init__(self, data):
        """Calculate. generic discription."""
        self.array = data

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, item.upper()):
            return getattr(self, item.upper())
        elif hasattr(self, item.lower()):
            return getattr(self, item.lower())

    @property
    def array(self):
        """Calculate. generic discription."""
        return self._array  # .reshape((-1, 1))

    @array.setter
    def array(self, arr):
        if isinstance(arr, np.ndarray):
            self._array = arr
        else:
            self._array = np.array(arr).squeeze()

        if not self._array.dtype == "complex128":
            if len(self._array.shape) == 2 and self._array.shape[1] >= 2:
                if "pol" in self.name.lower():
                    if (abs(self._array[:, 1]) > np.pi / 2).any():
                        self._array[:, 1] = np.deg2rad(self._array[:, 1])

                    self._array = self._array[:, 0] * (
                        np.cos(self._array[:, 1]) + 1j * np.sin(self._array[:, 1])
                    )
                else:
                    self._array = self._array[:, 0] + 1j * self._array[:, 1]
            else:
                self._array = self._array + 1j * 0
        elif len(self._array.shape) == 2 and self._array.shape[1] >= 2:
            self._array = self._array[:, 0]

    @property
    def real(self):
        """Calculate. generic discription."""
        return self.array.real

    @real.setter
    def real(self, _):
        pass

    @property
    def imag(self):
        """Calculate. generic discription."""
        return self.array.imag

    @imag.setter
    def imag(self, _):
        pass

    @property
    def mag(self):
        """Calculate. generic discription."""
        return np.abs(self.array)

    @mag.setter
    def mag(self, _):
        pass

    @property
    def phase(self):
        """Calculate. generic discription."""
        return np.angle(self.array, deg=True)

    @phase.setter
    def phase(self, _):
        pass

    @property
    def df(self):
        """Calculate. generic discription."""
        vals = [
            self.real,
            self.imag,
            -1 * self.imag,
            self.mag,
            self.phase,
            -1 * self.phase,
        ]
        columns = ["real", "imag", "inv_imag", "mag", "phase", "inv_phase"]
        # self._data = pd.DataFrame(dict(zip(columns, vals)))
        return pd.DataFrame(dict(zip(columns, vals)))

    @df.setter
    def df(self, _):
        pass


# %% Testing
if __name__ == "__main__":
    # examples
    import research_tools.equations as eqs
    from scipy import optimize

    xx = np.linspace(1, 100)

    yy = eqs.line(xx, 2, 20)

    yy_var = yy + np.random.random(len(xx)) - 0.5

    extr1 = extract_arguments(
        eqs.line,
        method="str, intersection",
        kwargs=dict(targ=None, z=dict(x=xx), y=yy_var),
        new="y",
    )

    exp1 = solve_for_variable(eqs.line, "b", "y")
    func1 = create_function(exp1, "b", kwargs=dict(x=xx, m=2, y=yy_var))
    test1 = optimize.least_squares(
        cost_basic, 15, kwargs=dict(targ=None, func=func1, x=xx, y=yy_var)
    )

    func2 = create_function(exp1, "b", kwargs=dict(m=2))
    test2 = optimize.least_squares(
        cost_basic, 15, kwargs=dict(targ=None, func=func2, x=xx, y=yy_var)
    )

    func3 = create_function(eqs.line, "b", "y", cost=cost_basic, kwargs=dict(m=2))
    test3 = optimize.least_squares(func3, 15, kwargs=dict(x=xx, y=yy_var))

    func4 = create_function(eqs.line, "b", "y", kwargs=dict(m=2))
    test4 = optimize.least_squares(
        cost_basic, 15, kwargs=dict(targ=yy_var, func=func4, x=xx)
    )

    func5 = create_function(eqs.nernst_planck_analytic_sol, "C", "C")
    test5 = optimize.least_squares(
        cost_basic,
        15,
        kwargs=dict(targ=None, func=func5, x=1, t=1, z=1, E=0, L=1, conc0=1, D=1, T=1),
    )

    func6 = create_function(eqs.nernst_planck_analytic_sol, "t", "t")
    test6 = optimize.least_squares(
        cost_basic,
        2,
        kwargs=dict(targ=test5["x"], func=func6, x=1, z=1, E=0, L=1, conc0=1, D=1, T=1),
    )
