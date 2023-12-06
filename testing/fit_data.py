import numpy as np
import pandas as pd
import configparser
from pathlib import Path
import re
import warnings
import sympy as sp

convert_sp = {
    sp.Float: float,
    sp.Integer: int,
    sp.Symbol: str,
    sp.Mul: str,
    sp.Add: str,
}


def get_config(file, sections=["base"], **kwargs):
    """
    Get the necessary information from a configuration .ini file.

    Parameters
    ----------
    file : [str, Path]
        The path to the .ini file containing the configuration settings to be imported. If file is
        None, any kwargs are returned as the settings. This is only usefull for external functions
        which may pass None for other reasons.
    sections : list, optional
        Defines what sections of the .ini to import. If "all" is passed, function will create a
        dict of dicts, separating each section into its own dict of settings.
        If no section match, attempts to find sections which include the values provided by the
        list. If there are still no matches, the first section will be called and returned.
    kwargs : function, optional
        Pass additional items to be included in the configuration.  If the configuration
        is in the .ini file, they will be overwritten.

    Returns
    -------
    config_file : dict
        Returns a dict containing all settings imported from the .ini file
    """
    if file is None:
        return kwargs
    cp = configparser.ConfigParser()
    cp.optionxform = lambda option: option
    # Load the configuration file
    if Path(file).is_file():
        cp.read(Path(file))
    elif (Path.cwd() / file).is_file():
        cp.read(Path.cwd() / file)
    elif (Path(file) / "config.ini").is_file():
        cp.read(Path(file) / "config.ini")
    else:
        cp.read(Path.cwd() / "config.ini")

    if isinstance(sections, str):
        if sections == "all":
            config_file = {}
            for sec in cp.sections():
                config_file[sec] = get_config(file, sections=[sec])
                config_file[sec] = {
                    k: kwargs.get(k, v) for k, v in config_file[sec].items()
                }
            return config_file
        sections = [sections]

    checked_sec = [s_in for s_in in sections if s_in in cp.sections()]
    if checked_sec == []:
        checked_sec = [
            s_file
            for s_in in sections
            for s_file in cp.sections()
            if s_in.lower() in s_file.lower()
        ]
    if checked_sec == []:
        checked_sec = [cp.sections()[0]]
    config_file = {
        k: eval_string(v)
        for sc in checked_sec
        for k, v in dict(cp.items(sc)).items()
    }
    included = []
    if "include_sections" in config_file.keys():
        included = [
            get_config(file, [sec])
            for sec in config_file.pop("include_sections").split(", ")
        ]
        for inc in included:
            config_file = {**config_file, **inc}

    return {**config_file, **kwargs}


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
    bkts = [["(", ")"], ["[", "]"], ["{", "}"]]
    ends = re.findall(r"^.|.$", text.strip())

    if ends in bkts:
        items = re.findall(
            r"[^\,\n]*:?[\(\[\{][^\)\]\}]+[\)\]\}]|[^\,\n]+",
            text.strip()[1:-1],
        )
        if all([":" in t for t in items]) and ends == bkts[2]:
            res = {}
            for item in items:
                it = [i.strip() for i in item.split(":")]
                res[it[0]] = (
                    eval_string(":".join(it[1:])) if len(it) > 1 else None
                )
        else:
            res = [eval_string(item.strip()) for item in items]
            if ends == bkts[0]:
                res = tuple(res)
            if ends == bkts[2]:
                try:
                    res = set(res)
                except TypeError as err:
                    print(f"Error creating set: {err}")
        return res
    if text.lower() == "none":
        return None
    elif bool(re.findall(r"\d", text)):
        try:
            res = sp.parse_expr(text)
            return convert_sp[type(res)](res)
        except (TypeError, SyntaxError, KeyError):
            return text
    return text


if __name__ == "__main__":
    from research_tools.functions import f_find, p_find
    from impedance_analysis.data_analysis import (
        DataImport,
        Complex_Imp,
        IS_Ckt,
    )
    from research_tools.functions import (
        gen_bnds,
        gen_mask,
        nyquist,
        bode,
        Complexer,
    )

    # test1 = "(5, None, 1e-12, test)"
    # res1 = eval_string(test1)

    # test2 = "[5, None, 1e-12, test]"
    # res2 = eval_string(test2)

    # test3 = "{5, None, 1e-12, test}"
    # res3 = eval_string(test3)

    # test4 = "{A:5, B:None, C:1e-12, D:test}"
    # res4 = eval_string(test4)

    # test5 = "{A:5, B:None, C:1e-12, D:{A:5, B:None, C:1e-12, D:test}, E:[3,4,2], F:{}, G:()}"
    # res5 = eval_string(test5)

    # test6 = "({}, (), {5,4})"
    # res6 = eval_string(test6)

    my_folder_path = p_find("impedance_analysis", "testing", base="cwd")
    files = f_find(my_folder_path, re_filter="circuits")[0]
    config_data = get_config(files, "randals_3")
    # data_in = DataImport(files[0], tool="MFIA", read_type="full")

    # # Import data using by first getting the appropriate filename.  f_find and p_find
    # # search for the desired files given a list of folder names. DataImport handles the actual
    # # importing of data
    # my_folder_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home")

    # files = f_find(my_folder_path)
    # file = files[0]
    # data_in = DataImport(file, tool="MFIA", read_type="full")

    # # The impedance class wraps the complex class with terms common to impedance.  Used internally
    # # by several of the eis modules/classes.
    # imp_data = Complex_Imp(data_in[data_in.keys()[0]])

    # # Begin fitting of impedance data by first declaring the initial conditions needed by
    # # impedance.py
    # model = "R_0-p(R_1,C_1)"
    # guess = [1e4, 1e8, 1e-12]
    # constants = {}

    # # Establish the ckt object. Data is retained in the object for fitting and refitting as well as
    # # exporting.
    # ckt = IS_Ckt(data_in[data_in.keys()[0]], guess, constants, model)

    # # Call base fit (which uses impedance.py fit, which in turn uses least squares) on the data
    # # contained within the object.
    # ckt.base_fit()

    # # from impedance_analysis import *

    # # test = ia.Complex_Imp()
    conv_dict = {
        sp.Float: float,
        sp.Integer: int,
        sp.Symbol: str,
        sp.Mul: str,
        sp.Add: str,
    }
