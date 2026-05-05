# -*- coding: utf-8 -*-
"""
Insert module description/summary.

Provide any or all of the following:
1. extended summary
2. routine listings/functions/classes
3. see also
4. notes
5. references
6. examples

@author: j2cle
Created on Mon Nov 24 18:05:11 2025
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from eis_analysis.z_system.system import ComplexSystem
from eis_analysis.z_system.convert import complex_shared_sign_mask
from eis_analysis.z_system.definitions import COMP_ALIASES
from eis_analysis.system_utilities.file_io import save, load_file  # noqa: F401

# %% Code
BASE_PATH = Path(
    r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Analysis\IS\EVA\Fit_Results\2025"
)


def extract_name_parts(
    string: str,
) -> tuple[str | None, str | None, int | None, int, str | None, str | None]:
    """
    Extract sample name, condition, temperature, and run from a formatted string.

    Parameters
    ----------
    string : str
        Formatted string containing sample name, condition, temperature, and run.

    Returns
    -------
    tuple[str, str, float, str]
        A tuple containing:
        - sample_name: Name of the sample.
        - condition: Condition of the sample.
        - temp: Temperature in degrees Celsius.
        - run: Run identifier.

    Raises
    ------
    ValueError
        If the string does not match the expected format.
    """
    pattern = re.compile(
        r"(9100|406)?_?(?P<sample_name>.*\d)_?(?P<condition>[a-zA-Z]+)_?(?P<temp>\d{2}c)(?P<run>_r\d+)?[^f]*(?P<fit>fit\d+)?$"
    )
    res = pattern.match(string)
    if res:
        name = res.group("sample_name").strip() if res.group("sample_name") else ""
        contam = int(np.round(int(name), -1)) if name.isdigit() else 0
        return (
            name if name else None,  # 0
            res.group("condition") if res.group("condition") else "",  # 1
            # int(res.group("temp")[:2]) if res.group("temp") else 25,  # 2
            int(res.group("temp")[:2]) if res.group("temp") else 25,  # 2
            contam,  # 3
            res.group("run").strip("_") if res.group("run") else "r0",  # 4
            res.group("fit").strip("_") if res.group("fit") else "fit0",  # 5
        )
        # return [p for p in parts if p]
    else:
        raise ValueError(f"Key '{string}' does not match the expected format.")


def filter_df(df):
    data = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
    mask = complex_shared_sign_mask(data)
    valid_df = df[mask].copy()
    return valid_df.reset_index(drop=True), df[~mask].copy()


def get_z_df(df, rename: bool = True):
    res_df = df[["freq", "real", "imag", "pr_freq", "pr_real", "pr_imag"]].copy()
    if rename:
        res_df.columns = [f"Z_{col}".replace("Z_pr_", "fit_Z_") for col in res_df.columns]
    res_df.attrs = df.attrs.copy()
    return res_df


NAME_POS = {
    name: idx
    for idx, name in enumerate(["sample", "cond", "temp", "sodium", "run", "fit", "grp", "index"])
}


def split_df_dict(df_dict, group_levels=("sample", "fit", "grp"), sep="_"):
    """
    Organize a dict of DataFrames into a dict-of-dicts grouped by given levels.

    Parameters
    ----------
    df_dict : dict[str, pd.DataFrame]
        Input dictionary of DataFrames keyed by underscore-separated strings.
    group_levels : list[str]
        List of grouping level names. Must be subset of:
        ["sample", "cond", "temp", "sodium", "run", "fit", "grp", "index"]

    Returns
    -------
    grouped_dict : dict
        Dict keyed by grouping values (tuple if >1 level).
        Each value is a dict of {original_key: original_df}.
    """
    gr_dict = defaultdict(dict)
    gr_idxs = [NAME_POS[lvl] for lvl in group_levels if lvl in NAME_POS]
    if not gr_idxs:
        gr_idxs = [0]

    for key, df in df_dict.items():
        parts = str(key).split(sep)
        gr_key = sep.join(parts[idx] for idx in gr_idxs if idx < len(parts))
        gr_dict[gr_key][key] = df

    return dict(gr_dict)


def establish_forms(
    df,
    forms: tuple[str, ...] = ("e_r", "e_r_dc"),
    smooth: bool | str | tuple[str, ...] = "",
    **kwargs,
):
    kwargs.pop("frequency", None)
    kwargs.setdefault("thickness", df.attrs.get("thickness", 0.045))
    kwargs.setdefault("area", df.attrs.get("area", 25))

    if "Z_freq" in df.columns:
        res_df = df.copy()
        raw_data = df["Z_real"].to_numpy() + 1j * df["Z_imag"].to_numpy()
        raw_fit = df["fit_Z_real"].to_numpy() + 1j * df["fit_Z_imag"].to_numpy()
        freq_data = df["Z_freq"].to_numpy()
        freq_fit = df["fit_Z_freq"].to_numpy()
    else:
        res_df = get_z_df(df, rename=True)
        raw_data = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
        raw_fit = df["pr_real"].to_numpy() + 1j * df["pr_imag"].to_numpy()
        freq_data = df["freq"].to_numpy()
        freq_fit = df["pr_freq"].to_numpy()

    data_sys = ComplexSystem(raw_data, frequency=freq_data, **kwargs)
    fit_sys = ComplexSystem(raw_fit, frequency=freq_fit, **kwargs)

    if isinstance(smooth, bool):
        smooth = forms if smooth else ()

    for form in forms:
        if ";" in form:
            parts = [f.strip() for f in form.split(";")]
            for pt in parts:
                name, fm = [s.strip() for s in pt.split(",")][:2]
                if name.endswith("freq"):
                    res_df[name] = data_sys["freq"]
                elif fm in smooth or form in smooth:
                    res_df[name] = data_sys[f"smooth({fm})"]
                else:
                    res_df[name] = data_sys[fm]
            for pt in parts:
                name, fm = [s.strip() for s in pt.split(",")][:2]
                if name.endswith("freq"):
                    res_df[f"fit_{name}"] = fit_sys["freq"]
                elif fm in smooth or form in smooth:
                    res_df[f"fit_{name}"] = fit_sys[f"smooth({fm})"]
                else:
                    res_df[f"fit_{name}"] = fit_sys[fm]

        elif form.lower() in COMP_ALIASES:
            res_df[f"{form}_freq"] = freq_data
            if form in smooth:
                res_df[f"{form}_real"] = data_sys[f"smooth({form}.real)"]
                res_df[f"{form}_imag"] = data_sys[f"smooth({form}.imag)"]
            else:
                res_df[f"{form}_real"] = data_sys[f"{form}.real"]
                res_df[f"{form}_imag"] = data_sys[f"{form}.imag"]

            res_df[f"fit_{form}_freq"] = freq_fit
            res_df[f"fit_{form}_real"] = fit_sys[f"{form}.real"]
            res_df[f"fit_{form}_imag"] = fit_sys[f"{form}.imag"]
        elif form.endswith("freq"):
            res_df[form] = data_sys["freq"]
            res_df[f"fit_{form}"] = fit_sys["freq"]
        else:
            res_df[form] = data_sys[f"smooth({form})"] if form in smooth else data_sys[form]
            res_df[f"fit_{form}"] = fit_sys[form]

    res_df.attrs = df.attrs.copy()
    return res_df


def make_polar_df(
    df: pd.DataFrame,
    forms: tuple[str, ...] = ("Z",),
    deg: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Build a 'polar version' of the dataframe: magnitude/phase instead of real/imag.
    Ensures required columns exist by calling establish_forms once if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, ideally already processed by establish_forms.
    forms : tuple[str, ...], default ("Z",)
        Forms to convert. Each form expects columns like
        '{form}_freq', '{form}_real', '{form}_imag' (and fit equivalents).
    deg : bool, default True
        If True, return phase in degrees. If False, return radians.
    kwargs : dict
        Passed through to establish_forms if needed.

    Returns
    -------
    pd.DataFrame
        New dataframe containing only freq, mag, phase columns for each form.
    """
    # --- Precheck: ensure all required cols exist ---
    missing = False
    for form in forms:
        required = [
            f"{form}_freq",
            f"{form}_real",
            f"{form}_imag",
            f"fit_{form}_freq",
            f"fit_{form}_real",
            f"fit_{form}_imag",
        ]
        for col in required:
            if col not in df.columns:
                missing = True
                break
        if missing:
            break

    if missing:
        df = establish_forms(df, forms=forms, smooth=False, **kwargs)

    # data_sys = ComplexSystem(raw_data, frequency=freq_data, **kwargs)
    # fit_sys = ComplexSystem(raw_fit, frequency=freq_fit, **kwargs)
    if forms[0] != "Z":
        forms = ("Z",) + forms

    # --- Build new polar dataframe ---
    res_df = pd.DataFrame()
    for form in forms:
        # Data mag/phase
        comp = df[f"{form}_real"].to_numpy() + 1j * df[f"{form}_imag"].to_numpy()
        res_df[f"{form}_freq"] = df[f"{form}_freq"].copy()
        res_df[f"{form}_mag"] = np.abs(comp)
        res_df[f"{form}_phase"] = np.angle(comp, deg=deg)

        # Fit mag/phase
        comp_fit = df[f"fit_{form}_real"].to_numpy() + 1j * df[f"fit_{form}_imag"].to_numpy()
        res_df[f"fit_{form}_freq"] = df[f"fit_{form}_freq"].copy()
        res_df[f"fit_{form}_mag"] = np.abs(comp_fit)
        res_df[f"fit_{form}_phase"] = np.angle(comp_fit, deg=deg)

    res_df.attrs = df.attrs.copy()
    return res_df


def mod_norm_form(data, add_levels={"cond"}):
    g_levels = {"sample", "fit", "grp"} | add_levels
    grouped = split_df_dict(data, group_levels=g_levels)
    # grp_concats = {k: pd.concat(v) for k, v in grouped.items()}
    # grp_maxes = {k: pd.concat(v)["Z_real"].max() for k, v in grouped.items()}
    for gr_dfs in grouped.values():
        df_all = pd.concat(gr_dfs)
        gmax = df_all.get("Z_norm_real", df_all["Z_real"]).max()
        # gmax = grp_maxes[gr_key]
        for key, df in gr_dfs.items():
            # fmt: off
            data[key]["Z_norm_freq"] = df.get("Z_norm_freq", df["Z_freq"])
            data[key]["Z_norm_real"] = df.get("Z_norm_real", df["Z_real"]) / gmax
            data[key]["Z_norm_imag"] = df.get("Z_norm_imag", df["Z_imag"]) / gmax
            data[key]["fit_Z_norm_freq"] = df.get("fit_Z_norm_freq", df["fit_Z_freq"])
            data[key]["fit_Z_norm_real"] = df.get("fit_Z_norm_real", df["fit_Z_real"]) / gmax
            data[key]["fit_Z_norm_imag"] = df.get("fit_Z_norm_imag", df["fit_Z_imag"]) / gmax
            # fmt: on
    return data


def decades(f_arr, first_only=True):
    base = np.array([float(10 ** (np.floor(np.log10(abs(a))))) for a in f_arr])
    ubase = np.unique(base)

    res1 = np.array([np.nan] * len(f_arr))

    # n0 = 0
    for b in ubase:
        n = np.argmin(abs(f_arr - b))
        if first_only:
            res1[n] = b
        else:
            res1[n:] = b

    return res1


def insert_data_decades(df, col="freq"):
    col = col if col in df.columns else df.columns[0]
    res_df = pd.DataFrame()
    res_df["decades"] = np.log10(decades(df[col].to_numpy(), first_only=False))
    res_df["decade_labels"] = np.log10(decades(df[col].to_numpy(), first_only=True))

    res_df[df.columns] = df.copy()
    res_df.attrs = df.attrs.copy()
    return res_df


def main(
    in_path: Path | str = BASE_PATH,
    glob: str = "*_fit_r*.xlsx",
    out_path: Path | str = "",
    save_polar: bool = True,
):
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path / "datasets"

    raw = load_file(in_path, load_to_dict=True, glob=glob)
    # Excract and organize data
    data = {k: v[0] for k, v in raw.items()}
    flat = {}
    trimmed = {}
    for key, val in data.items():
        grp = re.sub(r".*_fit_?", "", key)
        grp = grp if grp == "rc" else f"v{grp[1:]}"
        for skey, sval in val.items():
            if skey == "fit results":
                continue
            sval.attrs["source_file"] = key
            sval.attrs["fit_group"] = grp
            parts = extract_name_parts(skey)
            sval.attrs["fit"] = int(parts[-1].replace("fit", "")) if parts[-1] is not None else 0
            nkey = "_".join([str(p) for p in parts if p is not None] + [grp])
            # flat[nkey] = sval
            valid, invalid = filter_df(sval)
            # valid = insert_data_decades(valid)
            if len(invalid) > 0:
                trimmed[nkey] = invalid
            flat[nkey] = valid

    simple = {k: get_z_df(v, rename=False) for k, v in flat.items()}
    forms = (
        "e_r",
        # "Z_norm_freq, freq; Z_norm_real, Z.real/max(Z.real); Z_norm_imag, Z.imag/max(Z.real)",
        "Z_norm_freq, freq; Z_norm_real, Z.real; Z_norm_imag, Z.imag",
        "chi",
        "M",
    )
    revised = {k: establish_forms(v, forms=forms, smooth=True) for k, v in simple.items()}
    revised = mod_norm_form(revised, add_levels={"cond"})

    w_dec = {k: insert_data_decades(v, col="Z_freq") for k, v in revised.items()}
    save_form = split_df_dict(w_dec, group_levels=["sample", "grp", "fit"])

    save(save_form, out_path, file_type="xlsx", attrs=True)

    if save_polar:
        polar = {k: make_polar_df(v, ("Z", "e_r")) for k, v in revised.items()}

        w_dec_pol = {k: insert_data_decades(v, col="Z_freq") for k, v in polar.items()}
        save_form_pol = split_df_dict(w_dec_pol, group_levels=["sample", "grp", "fit"])
        save_form_pol = {f"{k}_pol": v for k, v in save_form_pol.items()}

        save(save_form_pol, out_path / "polar", file_type="xlsx", attrs=True)


# from eis_analysis.system_utilities.json_io import JSONSettings, DefaultJSONSettings
# class fake0(DefaultJSONSettings):
#     def __init__(self):
#         pth = Path("C:/Users/j2cle/Documents/Python/impedance_analysis/eis_analysis/z_fit")
#         super().__init__(pth, pth)
# %% Operations
if __name__ == "__main__":
    # main()
    from eis_analysis.system_utilities.json_io import JSONSettings

    # from eis_analysis.widgets.settings_handlers import manage_z_fit_settings_files

    # path = Path(__file__)
    # alt = DefaultJSONSettings()
    path = Path(
        "C:/Users/j2cle/Documents/Python/impedance_analysis/local/settings/curve_org_settings.json"
    )
    settings = JSONSettings(path)
    # print(JSONSettings._JSONSettings__location)
    # test0 = JSONSettings(path, path)

    # class fake1(DefaultJSONSettings):
    #     def __init__(self):
    #         pth = Path("C:/Users/j2cle/Documents/Python/impedance_analysis/eis_analysis/z_fit")
    #         super().__init__(pth, pth)

    # # alt = DefaultJSONSettings(path, path)
    # test1 = fake0()
    # test2 = fake1()

    # test2 = manage_z_fit_settings_files()
    # raw = load_file(BASE_PATH, load_to_dict=True, glob="*_fit_r*.xlsx")
    # # %% Excract and organize data
    # data_attrs = {k: v[1] for k, v in raw.items()}
    # data = {k: v[0] for k, v in raw.items()}
    # flat = {}
    # trimmed = {}
    # for key, val in data.items():
    #     grp = re.sub(r".*_fit_?", "", key)
    #     grp = grp if grp == "rc" else f"v{grp[1:]}"
    #     for skey, sval in val.items():
    #         if skey == "fit results":
    #             continue
    #         sval.attrs["source_file"] = key
    #         sval.attrs["fit_group"] = grp
    #         parts = extract_name_parts(skey)
    #         sval.attrs["fit"] = int(parts[-1].replace("fit", "")) if parts[-1] is not None else 0
    #         nkey = "_".join([str(p) for p in parts if p is not None] + [grp])
    #         # flat[nkey] = sval
    #         valid, invalid = filter_df(sval)
    #         # valid = insert_data_decades(valid)
    #         if len(invalid) > 0:
    #             trimmed[nkey] = invalid
    #         flat[nkey] = valid
    # # %% Cleanup data
    # simple = {k: get_z_df(v, rename=False) for k, v in flat.items()}
    # forms = (
    #     "e_r",
    #     # "Z_norm_freq, freq; Z_norm_real, Z.real/max(Z.real); Z_norm_imag, Z.imag/max(Z.real)",
    #     "Z_norm_freq, freq; Z_norm_real, Z.real; Z_norm_imag, Z.imag",
    #     "chi",
    #     "M",
    # )
    # revised = {k: establish_forms(v, forms=forms, smooth=True) for k, v in simple.items()}
    # revised = mod_norm_form(revised, add_levels={"cond"})
    # # %% Save cleaned data
    # w_dec = {k: insert_data_decades(v, col="Z_freq") for k, v in revised.items()}
    # save_form = split_df_dict(w_dec, group_levels=["sample", "grp", "fit"])

    # save_path = BASE_PATH / "datasets"
    # save(save_form, save_path, file_type="xlsx", attrs=True)

    # # %% Make/Save polar variation
    # polar = {k: make_polar_df(v, ("Z", "e_r")) for k, v in revised.items()}

    # w_dec_pol = {k: insert_data_decades(v, col="Z_freq") for k, v in polar.items()}
    # save_form_pol = split_df_dict(w_dec_pol, group_levels=["sample", "grp", "fit"])
    # save_form_pol = {f"{k}_pol": v for k, v in save_form_pol.items()}

    # save_path_pol = save_path / "polar"
    # save(save_form_pol, save_path_pol, file_type="xlsx", attrs=True)

# def extract_fit(full_name: str, *knowns: Any, digit_only: bool = False) -> str:
#     """
#     Extract the run number from a sample name based on known identifiers.

#     Parameters
#     ----------
#     full_name : str
#         The sample name string to extract the run number from.
#     *knowns : Any
#         Known identifiers that may precede the run number in the sample name.

#     Returns
#     -------
#     int | str
#         The extracted run number as an integer, or the original sample name if no run number is found.
#     """
#     name = str(full_name)
#     if knowns:
#         for k in knowns:
#             name = name.replace(str(k), "").strip().strip("_")

#     if not name:
#         name = str(full_name)
#         patterns = [
#             r"[_-]+(f[_-]?\d+)$",
#             r"[_-]+(fit[_-]?\d+)$",
#             r"[_-]+(\d+)$",
#             r"(\d+)$",
#         ]

#         for pattern in patterns:
#             if mt := re.search(pattern, name, re.I):
#                 name = mt.group(1)
#                 break

#     if digit_only and name and not name.isdigit():
#         if match := re.match(r"(\d+)$", name, re.I):
#             name = match.group(1)
#         elif match := re.search(r"(\d+)", name, re.I):
#             name = match.group(1)

#     return name
