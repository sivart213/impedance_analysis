# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import re
import configparser
from pathlib import Path

# Third-party imports
import pandas as pd
import sympy as sp

def parse_files(
    paths,
    parser,
    keywords=None,
    get_all=False,
    by_date=True,
    by_time=True,
    by_count=True,
    by_sub_count=True,
):
    """
    Parse and filter files based on given criteria.
    Parameters:
    paths (list): List of file paths to parse.
    parser (function): Function to parse each file.
    keywords (str or list, optional): Keywords to filter the files. Defaults to None.
    get_all (bool, optional): If True, return all parsed data without filtering. Defaults to False.
    by_date (bool, optional): If True, filter by the latest date. Defaults to True.
    by_time (bool, optional): If True, filter by the latest time. Defaults to True.
    by_count (bool, optional): If True, filter by the highest counter. Defaults to True.
    by_sub_count (bool, optional): If True, filter by the highest sub counter. Defaults to True.
    Returns:
    pandas.DataFrame: Filtered data as a DataFrame.
    """
    res = []
    for pth in paths:

        res.append(parser(pth))

    res_all = pd.DataFrame(
        res, columns=["name", "date", "time", "counter", "sub counter", "id"]
    )
    if get_all or all(
        not b for b in [by_date, by_time, by_count, by_sub_count]
    ):
        return res_all
    res = []
    for _, gr in res_all.groupby(by=["name"], as_index=False, sort=False):
        if keywords is not None:
            if isinstance(keywords, str):
                keywords = [keywords]
            if not any(kw in gr["name"].to_numpy()[0] for kw in keywords):
                continue
        if by_date:
            gr = gr.loc[gr["date"] == gr["date"].max()]
        if by_time:
            gr = gr.loc[gr["time"] == gr["time"].max()]
        if by_count:
            gr = gr.loc[gr["counter"] == gr["counter"].max()]
        if by_sub_count:
            gr = gr.loc[gr["sub counter"] == gr["sub counter"].max()]

        res.append(gr)
    res = pd.concat(res).reset_index(drop=True)

    return res

def eval_config_string(text):
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
                res[it[0]] = eval_config_string(":".join(it[1:])) if len(it) > 1 else None
        else:
            res = [eval_config_string(item.strip()) for item in items]
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

def get_config(file, sections=None, **kwargs):
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
    if sections is None:
        sections = ["base"]
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
        k: eval_config_string(v) for sc in checked_sec for k, v in dict(cp.items(sc)).items()
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
