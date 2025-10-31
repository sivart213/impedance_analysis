import re
from typing import Any
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from eis_analysis.string_ops import find_common_str
from eis_analysis.data_treatment import range_maker
from eis_analysis.system_utilities import save  # noqa: F401
from eis_analysis.dc_fit.dc_data_post import fit_arrhenius_for_points  # noqa: F401
from eis_analysis.dc_fit.extract_tools import (
    BASE_KEYS,
    group_points,
    form_std_df_index,
    partial_selection,
)
from eis_analysis.dc_fit.fit_functions import data_group_trend_eval
from eis_analysis.impedance_supplement import get_impedance, parse_parameters
from eis_analysis.data_treatment.z_array_ops import (
    arc_quality,
    f_peak_stats,
    find_peak_vals,
    find_f_peak_idx,
    f_r_c_conversion,
)
from eis_analysis.impedance_supplement.model_ops import parse_model_groups

DEFAULT_KEYS = BASE_KEYS + ("fit",)
REVISED_COND = ["pre", "dh", "dry", "pre-dh", "dh-dry", "pre-dry"]


def load_fit_results(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load fit results and attributes from an Excel file.

    Parameters
    ----------
    file_path : Path
        Path to the Excel file containing fit results

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing (fit_results_df, attrs_df)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the required sheets are not found in the Excel file
    """
    file_path = Path(file_path)
    if not file_path.exists() or file_path.suffix not in [".xlsx", ".xls"]:
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load the two sheets
        fit_results_df = pd.read_excel(file_path, sheet_name="fit results", index_col=1)
        # Remove the first column (which is now at position 0 after setting the index)
        fit_results_df = fit_results_df.iloc[:, 1:]

        # Load the attrs sheet normally
        attrs_df = pd.read_excel(file_path, sheet_name="attrs", index_col=0)

        attrs_df = parse_attrs_df(attrs_df, fit_results_df)

        attrs_df.index.name = fit_results_df.index.name

        return fit_results_df, attrs_df

    except ValueError as e:
        # This will catch if the sheet names don't exist
        raise ValueError(f"Required sheets not found in {file_path}: {e}")


def extract_run(full_name: str, *knowns: Any) -> int:
    """
    Extract the run number from a sample name based on known identifiers.

    Parameters
    ----------
    full_name : str
        The sample name string to extract the run number from.
    *knowns : Any
        Known identifiers that may precede the run number in the sample name.

    Returns
    -------
    int | str
        The extracted run number as an integer, or the original sample name if no run number is found.
    """
    name = str(full_name)
    if knowns:
        for k in knowns:
            name = name.replace(str(k), "").strip().strip("_")

    patterns = [
        r"[_-]+r[_-]?(\d+)",
        r"[_-]+run[_-]?(\d+)",
        r"r[_-]?(\d+)",
        r"run[_-]?(\d+)",
        r"[_-]+(\d+)",
        r"(\d+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def extract_fit(full_name: str, *knowns: Any, digit_only: bool = False) -> str:
    """
    Extract the run number from a sample name based on known identifiers.

    Parameters
    ----------
    full_name : str
        The sample name string to extract the run number from.
    *knowns : Any
        Known identifiers that may precede the run number in the sample name.

    Returns
    -------
    int | str
        The extracted run number as an integer, or the original sample name if no run number is found.
    """
    name = str(full_name)
    if knowns:
        for k in knowns:
            name = name.replace(str(k), "").strip().strip("_")

    if not name:
        name = str(full_name)
        patterns = [
            r"[_-]+(f[_-]?\d+)$",
            r"[_-]+(fit[_-]?\d+)$",
            r"[_-]+(\d+)$",
            r"(\d+)$",
        ]

        for pattern in patterns:
            if match := re.search(pattern, name, re.I):
                name = match.group(1)
                break

    if digit_only and name and not name.isdigit():
        if match := re.match(r"(\d+)$", name, re.I):
            name = match.group(1)
        elif match := re.search(r"(\d+)", name, re.I):
            name = match.group(1)

    return name


def parse_attrs_df(attrs_df: pd.DataFrame, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:

    # Make copies to avoid modifying original DataFrames

    attrs_df = attrs_df.copy()
    priority_attrs = ["sample_name", "condition", "temp", "sodium"]
    priority_attrs = [pri for pri in priority_attrs if pri in attrs_df.columns]

    name_list = pd.Series(attrs_df.index, index=attrs_df.index)
    if "fit" not in attrs_df.columns:
        if fit_df is not None:
            name_list = fit_df.get("Datasets", name_list)
        attrs_df["fit"] = pd.NA
        for idx in attrs_df.index:
            attrs_df.at[idx, "fit"] = extract_fit(str(idx), name_list[idx])

    if "run" not in attrs_df.columns:
        attrs_df["run"] = pd.NA
        for idx in attrs_df.index:
            knowns = ["9100", "406"]
            knowns += attrs_df.loc[idx, priority_attrs[:-1] + ["fit"]].to_list()  # type: ignore
            # Extract run number using the dataset and knowns
            attrs_df.at[idx, "run"] = extract_run(str(name_list[idx]), *knowns)

    # Convert sample_name column to string type if it exists
    if "sample_name" in attrs_df.columns:
        attrs_df["sample_name"] = attrs_df["sample_name"].astype(str)

    try:
        attrs_df["run"] = attrs_df["run"].astype(int)
    except ValueError:
        pass

    if attrs_df["fit"].dtype == float:
        attrs_df["fit"] = attrs_df["fit"].astype(int)
    elif not pd.api.types.is_numeric_dtype(attrs_df["fit"]):
        if attrs_df["fit"].nunique() > 1:
            _, ids = find_common_str(*attrs_df["fit"])
            try:
                fit_num = np.asarray(ids, dtype=int)
            except ValueError:
                unique_fits = attrs_df["fit"].unique()
                found_n = []
                fit_map = {}
                for val in unique_fits:
                    num_str = extract_fit(val, digit_only=True)
                    if num_str.isdigit():
                        fit_map[val] = int(num_str)
                        found_n.append(int(num_str))
                    else:
                        fit_map[val] = np.nan

                if len(found_n) != len(fit_map):
                    found_n.sort()
                    for key, value in fit_map.items():
                        if np.isnan(value):
                            next_n = 1
                            for n in found_n:
                                if next_n < n:
                                    break
                                next_n = n + 1
                            fit_map[key] = next_n
                            found_n.append(next_n)
                            found_n.sort()

                fit_num = attrs_df["fit"].map(fit_map)
        else:
            fit_str = extract_fit(attrs_df["fit"].iloc[0], digit_only=True)
            try:
                fit_num = int(fit_str)
            except ValueError:
                fit_num = 1
        attrs_df["fit"] = fit_num

    return attrs_df


def combine_fit_data(
    fit_df: pd.DataFrame, attrs_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine fitting results with attribute data into a single DataFrame with specific column ordering.

    Parameters
    ----------
    fit_df : pd.DataFrame
        DataFrame containing fitting results with R and C parameters
    attrs_df : pd.DataFrame
        DataFrame containing sample attributes and additional data

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns ordered as:
        'Model', 'sample_name', 'condition', 'temp', 'sodium',
        remaining fit_df columns, remaining attrs_df columns
    """
    # Make copies to avoid modifying original DataFrames
    fit_df = fit_df.copy()
    attrs_df = attrs_df.copy()
    priority_attrs = ["sample_name", "condition", "temp", "sodium"]

    # Drop unwanted columns from fit_df
    columns_to_drop = ["Dataset", "Comments"]
    fit_df = fit_df.drop(columns=[col for col in columns_to_drop if col in fit_df.columns])

    priority_attrs += ["run", "fit"]

    # Create list of remaining columns from fit_df (excluding 'Model' which is handled separately)
    fit_remaining_cols = [col for col in fit_df.columns if col != "Model"]

    # Create list of remaining columns from attrs_df (excluding the priority columns)

    attrs_remaining_cols = [col for col in attrs_df.columns if col not in priority_attrs]

    # Combine the DataFrames (they have matching indexes)
    combined_df = pd.concat([fit_df, attrs_df], axis=1)

    # Remove any duplicate columns that might have occurred during the concatenation
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    return (
        make_param_df(combined_df[priority_attrs + fit_remaining_cols]),
        combined_df[["Model"] + attrs_remaining_cols],
    )


def err_prop_mult_div(
    result: float,
    *values: tuple[float],
) -> tuple[float, float]:
    """
    Calculate the error propagation for multiplication and division.
    It is assumed that values is a flat list where each value is followed by its error.

    """
    # Unpack the values and errors
    arr = np.array(values).reshape(-1, 2)
    arr = np.array([float(val) for val in arr.flatten()])

    # Separate the values and errors
    vals = arr[::2]
    errs = arr[1::2]

    # Calculate the relative error
    rel_err = np.sqrt(np.sum((errs / vals) ** 2))

    # Calculate the absolute error
    abs_err = result * rel_err

    return abs_err


def make_param_df(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame with only the model parameters and sample information.
    """
    # Make a copy to avoid modifying the original DataFrame
    all_cols = base_cols = ["sample_name", "condition", "temp", "sodium", "run", "fit"]

    param_cols = []
    for col in combined_df.columns:
        if col[-1].isdigit() and f"{col}_std" in combined_df.columns:
            param_cols.append(col)
            all_cols.append(col)
            all_cols.append(f"{col}_std")

    df = combined_df.copy()[all_cols]

    df.attrs["base"] = base_cols
    df.attrs["params"] = param_cols

    return df


def add_frequency_parameters(
    param_df: pd.DataFrame, info_df: pd.DataFrame, freq: np.ndarray, err_by_mult: bool = False
) -> pd.DataFrame:
    """
    Add peak frequency (f) and time constant (tau) columns for each submodel.

    Parameters
    ----------
    info_df : pd.DataFrame
        DataFrame with model parameters and sample information
    freq : np.ndarray
        Array of frequencies to use for impedance calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with added frequency and time constant columns
    """

    result_df = pd.DataFrame(index=param_df.index)

    # First pass to determine the maximum number of submodels
    models = pd.Series()
    for idx in result_df.index:
        models[idx] = parse_model_groups(info_df.loc[idx, "Model"], "numbers")

    max_submodels = models.apply(len).max()
    # Create empty columns for f and tau parameters
    for i in range(max_submodels):
        result_df[f"f{i}"] = 0.0
        result_df[f"f{i}_std"] = 0.0

    for i in range(max_submodels):
        result_df[f"tau{i}"] = 0.0
        result_df[f"tau{i}_std"] = 0.0

    # Second pass to calculate and fill values
    for idx in result_df.index:
        for i, model in enumerate(models[idx]):
            params = parse_parameters(model)
            values = param_df.loc[idx, params].to_list()  # type: ignore
            stdevs = param_df.loc[idx, [f"{p}_std" for p in params]].to_list()  # type: ignore
            # fpeak = find_f_peak(values, model, freq)
            fpeak, tau = find_peak_vals(
                f=freq, values=["fpeak", "tau"], params=values, model=model
            )
            # tau = f_r_c_conversion(fpeak, default=0)

            result_df.at[idx, f"f{i}"] = fpeak
            result_df.at[idx, f"tau{i}"] = tau
            if fpeak != 0:
                if err_by_mult:
                    # Approach 1: Error propagation for multiplication
                    # For fpeak = 1/(2π*RC), we use error propagation for division
                    # We multiply all parameters to get their product, then divide 1/(2π) by that product
                    product = np.prod(values)
                    if product > 0:
                        # Create pairs of values and their stdevs for err_prop_mult_div
                        val_std_pairs = []
                        for val, std in zip(values, stdevs):
                            val_std_pairs.extend([val, std])

                        # Calculate the error in the product
                        prod_error = err_prop_mult_div(product, *val_std_pairs)

                        # Error for fpeak (division error propagation)
                        fpeak_std = fpeak * (prod_error / product)
                        result_df.at[idx, f"f{i}_std"] = fpeak_std

                        # Error for tau (same as product error since tau = RC)
                        result_df.at[idx, f"tau{i}_std"] = tau * (prod_error / product)
                else:
                    # Approach 2: Min/max parameter variation
                    # Create combinations of parameters varying by ±1 std
                    # Approach 2: Test all combinations of parameter variations
                    min_fpeak, max_fpeak, min_tau, max_tau = f_peak_stats(
                        values, stdevs, model, freq
                    )

                    # Calculate standard deviations based on the range
                    fpeak_range = max_fpeak - min_fpeak
                    tau_range = max_tau - min_tau

                    # Standard deviation approximation from range (divide by 2 for +/- 1 std dev)
                    result_df.at[idx, f"f{i}_std"] = fpeak_range / 2
                    result_df.at[idx, f"tau{i}_std"] = tau_range / 2

    return result_df


def add_material_parameters(
    param_df: pd.DataFrame,
    info_df: pd.DataFrame,
    area: int | float = 0.0,
    thickness: int | float = 0.0,
    suffix: str = "std",
) -> pd.DataFrame:
    """
    Convert resistance (R) to resistivity (rho) and capacitance (C) to relative
    permittivity (epsilon) using sample dimensions.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame with frequency parameters to be modified
    info_df : pd.DataFrame
        Source DataFrame containing area and thickness information

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with added resistivity and permittivity columns
    """
    # Constants
    EPSILON_0 = 8.8541878128e-14  # Vacuum permittivity in F/cm

    # Make a copy to avoid modifying the original DataFrame
    params_cols: list[str] = param_df.attrs["params"]

    result_df = pd.DataFrame(index=param_df.index)

    if area:
        a_arr = np.array([area] * len(result_df))
    elif "area" in info_df.columns:
        a_arr = info_df["area"].to_numpy(dtype=float)
    else:
        raise ValueError("Area must be provided and greater than zero.")
    if thickness:
        th_arr = np.array([thickness] * len(result_df))
    elif "thickness" in info_df.columns:
        th_arr = info_df["thickness"].to_numpy(dtype=float)
    else:
        raise ValueError("Thickness must be provided and greater than zero.")

    for col in params_cols:
        elem = re.match(r"^(ICPE|CPE|R|C)(.+)", col)
        if not elem or ("CPE" in elem.group(1) and col[-1] != "0"):
            continue
        if elem.group(1) == "R" or elem.group(1) == "ICPE":
            r_value = param_df[col].to_numpy(dtype=float)
            r_std = param_df[f"{col}_{suffix}"].to_numpy(dtype=float)
            result_df[f"rho{elem.group(2)}"] = r_value * a_arr / th_arr
            result_df[f"rho{elem.group(2)}_{suffix}"] = r_std * a_arr / th_arr
        elif elem.group(1) == "C" or elem.group(1) == "CPE":
            c_value = param_df[col].to_numpy(dtype=float)
            c_std = param_df[f"{col}_{suffix}"].to_numpy(dtype=float)
            result_df[f"epsilon{elem.group(2)}"] = c_value * th_arr / (EPSILON_0 * a_arr)
            result_df[f"epsilon{elem.group(2)}_{suffix}"] = c_std * th_arr / (EPSILON_0 * a_arr)

    return result_df


def add_peak_results(
    param_df: pd.DataFrame,
    info_df: pd.DataFrame,
    freq: np.ndarray | None = None,
    area: float = 0.0,
    thickness: float = 0.0,
) -> pd.DataFrame:
    """
    Add peak frequency, time constant, impedance maximum, and capacitance maximum columns.

    Parameters
    ----------
    info_df : pd.DataFrame
        DataFrame with model parameters and sample information
    freq : np.ndarray, optional
        Array of frequencies to use for impedance calculation if RC_freq not available
    area : float, optional
        Sample area in cm² if not in info_df
    thickness : float, optional
        Sample thickness in cm if not in info_df

    Returns
    -------
    pd.DataFrame
        DataFrame with added peak results columns
    """
    # Make a copy of base data
    result_df = pd.DataFrame(index=param_df.index)

    # Get area and thickness values
    if area:
        a_arr = pd.Series([area] * len(result_df), index=result_df.index)
    elif "area" in info_df.columns:
        a_arr = info_df["area"]
    else:
        a_arr = pd.Series(np.ones(len(result_df)), index=result_df.index)

    if thickness:
        th_arr = pd.Series([thickness] * len(result_df), index=result_df.index)
    elif "thickness" in info_df.columns:
        th_arr = info_df["thickness"]
    else:
        th_arr = pd.Series(np.ones(len(result_df)), index=result_df.index)

    if freq is None:
        freq = np.logspace(-3, 6, 1000)

    # Process each row
    for idx in result_df.index:
        # Get or calculate RC_freq
        rc_freq = rc_tau = z_max = c_max = 0.0
        Q_f = Q_tau = z_ref = c_ref = np.nan
        Z = np.array([])
        peak_idx = 0
        if "Model" in info_df.columns:
            model = str(info_df.loc[idx, "Model"])
            params = parse_parameters(model)
            if all(p in param_df.columns for p in params):
                values = param_df.loc[idx, params].to_list()  # type: ignore
                Z = get_impedance(freq, values, model=model)
                peak_idx = find_f_peak_idx(Z)
                # rc_freq, Q_f, rc_tau, Q_tau, Z = find_peak_stats(values, model, freq)
                rc_freq, Q_f, rc_tau, Q_tau = arc_quality(freq, Z, peak_idx)

        if "RC_freq" in info_df.columns:
            rc_freq = float(info_df.loc[idx, "RC_freq"])  # type: ignore
            rc_tau = f_r_c_conversion(rc_freq, default=0.0)

        # Get or calculate Z_max (maximum impedance)
        if Z.size:
            z_max = float(2 * (max(np.real(Z)) - np.real(Z[peak_idx])))
            z_ref = float(abs(z_max - 2 * (np.real(Z[-1]) - np.real(Z[peak_idx]))))
        elif "Z_max" in info_df.columns:
            z_max = float(info_df.loc[idx, "Z_max"])  # type: ignore
        elif "dc_Z" in info_df.columns:
            z_max = float(info_df.loc[idx, "dc_Z"])  # type: ignore
        elif "dc_cond" in info_df.columns:
            # Convert conductivity to resistance using area and thickness
            dc_cond = info_df.loc[idx, "dc_cond"]
            z_max = float((1.0 / dc_cond) * th_arr[idx] / a_arr[idx])  # type: ignore

        # Calculate C_max from Z_max and RC_tau
        if z_max > 0 and rc_tau > 0:
            c_max = rc_tau / z_max
            c_ref = c_max - rc_tau / z_ref if z_ref else c_ref
        # z_ref = z_max - z_ref
        # else:
        #     c_max = 0.0

        # Store results
        result_df.at[idx, "f_RC"] = rc_freq
        result_df.at[idx, "f_RC_qual"] = 0 if np.isnan(Q_f) else Q_f
        result_df.at[idx, "tau_RC"] = rc_tau
        result_df.at[idx, "tau_RC_qual"] = 0 if np.isnan(Q_tau) else Q_tau
        result_df.at[idx, "R_RC"] = z_max
        result_df.at[idx, "R_RC_dev"] = 0 if np.isnan(z_ref) else z_ref
        result_df.at[idx, "C_RC"] = c_max
        result_df.at[idx, "C_RC_dev"] = 0 if np.isnan(c_ref) else c_ref

    result_df.attrs["params"] = ["f_RC", "tau_RC", "R_RC", "C_RC"]

    return pd.concat(
        [result_df, add_material_parameters(result_df, info_df, area, thickness, "dev")], axis=1
    )


def append_condition_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate deltas between different conditions and add them to the dataframe.

    For each set of matching rows (same sample_name, temp, sodium, run, fit but different condition),
    calculate the difference between conditions and create new rows with delta conditions:
    - pre-dh: difference between 'pre' and 'dh' conditions
    - dh-dry: difference between 'dh' and 'dry' conditions
    - pre-dry: difference between 'pre' and 'dry' conditions

    The index for each new delta row is created by replacing the condition in the original index
    with the new delta condition name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing impedance data with columns including
        'sample_name', 'condition', 'temp', 'sodium', 'run', 'fit'

    Returns
    -------
    pd.DataFrame
        DataFrame with added delta conditions
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Filter to only use id_cols that exist in the dataframe
    id_cols = [
        col for col in DEFAULT_KEYS if col in df.columns and col not in ["condition", "run"]
    ]

    # Check if condition column exists
    if "condition" not in df.columns:
        raise ValueError("Required column 'condition' not found in DataFrame")

    # Get all columns that are not identification columns - these are data columns
    data_cols = [col for col in df.columns if col not in DEFAULT_KEYS]

    temps = sorted(df.get("temp", pd.Series()).unique())
    range_df = pd.DataFrame()
    # Group by identification columns to find sets of rows that differ only by condition
    df_groups = df.groupby(id_cols)
    for g_key, group in df_groups:
        # For each pair of conditions, calculate the delta if both conditions exist
        cond0_data = pd.Series(index=data_cols, dtype=float)
        if group["condition"].str.contains("pre").any():
            cond0_data = group[group["condition"] == "pre"][data_cols].iloc[0]
        elif g_key[1] in temps:
            if g_key not in range_df.attrs.get("keys", []):
                keys = [(g_key[0], t) + g_key[2:] for t in temps]
                range_df = pd.DataFrame(index=temps, columns=data_cols, dtype=float)
                for key in keys:
                    if key in df_groups.groups:
                        grp = df_groups.get_group(key)
                        grp = grp[grp["condition"] == "pre"]
                        if not grp.empty:
                            range_df.loc[key[1]] = grp[data_cols].iloc[0]
                range_df.attrs["keys"] = keys
                range_df = range_df.interpolate(method="pchip")
            cond0_data = range_df.loc[g_key[1]]

        # base = 1 if base.empty else base[data_cols].iloc[0]

        for cond1, cond2 in combinations(group["condition"].unique(), 2):
            # Find rows with these conditions
            cond1_rows = group[group["condition"] == cond1][data_cols]
            cond2_rows = group[group["condition"] == cond2][data_cols]

            idx1 = cond1_rows.index.to_list()
            idx2 = cond2_rows.index.to_list()
            idx0 = idx1.copy()
            if len(idx1) > len(idx2):
                idx2 += idx2 * ((len(idx1) // len(idx2)) + 1)
                idx0 = idx1.copy()
            elif len(idx2) > len(idx1):
                idx1 += idx1 * ((len(idx2) // len(idx1)) + 1)
                idx0 = idx2.copy()

            for id0, id1, id2 in zip(idx0, idx1, idx2):
                delta_name = f"{cond1}-{cond2}"

                delta_row = result_df.loc[id0].copy()

                # Set the delta condition
                delta_row["condition"] = delta_name

                delta_row[data_cols] = (cond1_rows.loc[id1] - cond2_rows.loc[id2]) / cond0_data

                # Create new index by replacing the condition
                # Pattern to match whole words 'pre', 'dh', or 'dry'
                new_idx = str(id0)
                pattern = r"pre|dh|dry"

                if re.search(pattern, new_idx, re.I):
                    # Replace the first occurrence
                    new_idx = re.sub(pattern, delta_name, new_idx, flags=re.I)
                else:
                    # If no match found, modify index by appending delta name
                    new_idx = f"{new_idx}_{delta_name}"

                result_df.loc[new_idx] = delta_row

    return result_df


def merge_common_cols(df0, col_re):
    """
    For columns like C1 and C1_0 (or R2 and R2_0):
    - Verify they are mutually exclusive (never both non-NaN in the same row).
    - Merge _0 into the base column.
    - Drop the _0 column.
    """
    df = df0.copy()
    for col in list(df.columns):
        m = col_re.match(col)
        if not m:
            continue
        param, sep, idx, special, suffix = m.groups(default="")
        if special == "_0" and (base := f"{param}{sep}{idx}{suffix}") in df.columns:
            # Check exclusivity
            overlap = df[base].notna() & df[col].notna()
            if overlap.any():
                continue  # Skip if not mutually exclusive

            # Merge
            df[base] = df[base].combine_first(df[col])
            df.drop(columns=[col], inplace=True)
    df.attrs = df0.attrs
    return df


def preprocess_columns(df, col_re):
    """
    Pre-parse DataFrame columns:
    - Map ICPE -> R
    - Map CPE  -> C
    Only rename if the mapped version does not already exist.
    """
    fixed_head = ["R", "C"]
    fixed_tail = ["f", "tau", "rho", "epsilon"]
    new_cols = []
    params = set()
    for col in df.columns:
        m = col_re.match(col)
        if not m:
            new_cols.append(col)
            continue

        param, sep, idx, special, suffix = m.groups(default="")

        # Map ICPE -> R, CPE -> C
        candidate = col
        if param == "ICPE":
            candidate = f"R{sep}{idx}{special}{suffix}"
            param = "R"
        elif param == "CPE":
            candidate = f"C{sep}{idx}{special}{suffix}"
            param = "C"

        new_cols.append(candidate)
        params.add(param)

    df = df.copy()
    df.columns = new_cols

    unknowns = sorted(p for p in params if p not in fixed_head + fixed_tail)
    df.attrs["sorting"] = {p: i for i, p in enumerate(fixed_head + unknowns + fixed_tail)}
    return merge_common_cols(df, col_re)


def parse_col(col, param_priority, col_re):
    m = col_re.match(col)
    if not m:
        return (len(param_priority) + 1, float("inf"), 0, 0, 1, col)
    param, _, idx, special, suffix = m.groups(default="")
    p_ord = param_priority.get(param, len(param_priority))
    # idx order
    idx_ord = float("inf") if idx == "RC" else int(idx)
    # specials come after all primaries
    s_ord = int(special[1:]) if special else 0
    # stat cols always follow their primary
    stat_flag = 0 if suffix == "" else 1

    return (p_ord, s_ord, idx_ord, stat_flag, suffix)


def sort_columns(df):
    # Get groups (element, opt sep, index (or RC), second index, any additional)
    col_re = re.compile(r"^([A-Za-z]+)(_?)(\d+|RC)(_\d)?(.*)")
    df = preprocess_columns(df.copy(), col_re)
    sorted_cols = sorted(df.columns, key=lambda c: parse_col(c, df.attrs["sorting"], col_re))
    return df[sorted_cols]


def fit_param_sort(columns):
    """
    Sort column names based on a predefined suffix order.

    Parameters
    ----------
    columns : list-like
        Column names to sort.
    Returns
    -------
    list
        Sorted list of column names.
    """
    # Map suffix → rank
    order_map = {s: i for i, s in enumerate(["A", "Ea", "alpha", "beta", "b", "m", "R2"])}

    def sort_key(col):
        # Find matching suffix
        for suf, rank in order_map.items():
            if col.endswith(suf):
                # prefix = col[: -len(suf)]
                return (col.replace(suf, ""), rank)
        # Fallback: unknown suffix → put at end, then alphabetical
        return (col, len(order_map))

    return sorted(columns, key=sort_key)


def ensure_temp_rows(df):
    """
    Ensure that each (sample_name, condition, fit) group has at least one row
    for each required temperature. Missing temps are added with NaNs for
    parameter columns. Rows are added directly via .loc to a copy of df.
    """
    id_cols = ["sample_name", "condition", "sodium", "fit"]
    param_cols = {c: np.nan for c in df.columns if c not in id_cols + ["temp", "run"]}
    required_temps = [85.0, 80.0, 75.0, 70.0, 65.0, 60.0]
    df_out = df.copy()

    man_idx = 0
    for (samp, cond, sodium, fit), sub in df.groupby(id_cols, dropna=False):
        if sub["temp"].dropna().nunique() < 2:
            continue
        existing_temps = set(sub["temp"].tolist())
        for t in required_temps:
            if t not in existing_temps:
                idx = f"man{man_idx}_{samp}{cond}{int(t)}c_r1_fit{fit}"
                man_idx += 1
                # Build the row dict
                row = {
                    "sample_name": samp,
                    "condition": cond,
                    "temp": t,
                    "sodium": sodium,
                    "run": 1,
                    "fit": fit,
                }

                row |= param_cols
                df_out.loc[idx] = row

    return df_out


def convert_rc_to_multi_index(
    df: pd.DataFrame, keep_orig_index: bool = True, dtypes: dict | None = None
) -> pd.DataFrame:
    """
    Convert impedance data DataFrame to use a standard MultiIndex for Arrhenius fitting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from impedance analysis with columns like 'sample_name', 'condition', etc.
    keep_orig_index : bool, optional
        Whether to preserve the original index as the first level, by default True
    orig_idx_name : str, optional
        Name to give the original index if it doesn't have one, by default "measurement"
    std_keys : tuple[str, ...], optional
        Standard columns to include in the MultiIndex, by default
        ("sample_name", "condition", "temp", "sodium", "run")
    dtypes : dict, optional
        Data types for index levels, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame with a standardized MultiIndex
    """

    # Make a copy to avoid modifying original
    result = ensure_temp_rows(df.copy())

    # Check which standard columns are present
    present_cols = [col for col in DEFAULT_KEYS if col in result.columns]

    if not present_cols:
        raise ValueError(
            f"None of the required columns {DEFAULT_KEYS} were found in the DataFrame"
        )
    result = result.reset_index(drop=False)

    if keep_orig_index:
        orig_idx_name = result.columns[0]
        all_names = tuple(present_cols) + (orig_idx_name,)
    else:
        all_names = tuple(present_cols)

    result = form_std_df_index(result, names=all_names, dtypes=dtypes)

    return result


def save_fit_data(
    data_dict: dict,
    save_path: Path,
    reorganize_index: bool = False,
    save_depths: int | tuple[int, ...] = 0,
    simplify_names: bool = True,
    **kwargs,
) -> None:
    """
    Save fit data with optional index reorganization to make "Name" the first level.

    Parameters
    ----------
    data_dict : dict
        Dictionary of DataFrames to save
    save_path : Path
        Path where the data should be saved
    reorganize_index : bool, optional
        Whether to reorganize the index to make "Name" the first level, by default False

    Returns
    -------
    None
    """
    # Make a copy to avoid modifying the original dictionary
    processed_data = {}

    # Process each DataFrame in the dictionary
    for key, df in data_dict.items():
        if (
            reorganize_index
            and isinstance(df, pd.DataFrame)
            and isinstance(df.index, pd.MultiIndex)
            and "Name" in df.index.names
        ):

            name_pos = df.index.names.index("Name")

            if name_pos > 0:
                new_order = ["Name"] + [name for name in df.index.names if name != "Name"]
                df = df.reorder_levels(new_order)

        processed_data[re.sub(r"(\d+)\.0\s", r"\1 ", key)] = df

    if isinstance(save_depths, int):
        save_depths = (save_depths,)

    prior_keys = []
    prior_keysets = []

    for depth in sorted(save_depths, reverse=True):
        depth = int(depth)
        if depth <= 0:
            continue
        for key in processed_data:
            parts = re.split(" ", key)
            if len(parts) <= depth or parts[:depth] in prior_keys:
                continue
            prior_keys.append(parts[:depth])

    for parts in prior_keys:
        tmp_path = save_path.with_stem(f"{save_path.stem}_{'_'.join(parts)}")
        nm = " ".join(parts)
        subset = partial_selection(processed_data, nm)
        subset["fit_results"] = partial_selection(processed_data["fit_results"], nm)
        if simplify_names:
            subset = {
                k.replace(nm, "").strip() if k != nm else "all": v for k, v in subset.items()
            }
        prior_keysets.append(list(subset.keys()))
        save(subset, tmp_path, file_type="xls", **kwargs)
        print(f"Data saved to {tmp_path}")

    if 0 in save_depths and list(processed_data.keys()) not in prior_keysets:
        save(processed_data, save_path, file_type="xls", **kwargs)
        print(f"Data saved to {save_path}")


# %%
if __name__ == "__main__":
    from copy import deepcopy

    base_path = Path(
        r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Analysis\IS\EVA\Fit_Results\2025"
    )
    f_bases = ["9100_cln2_fit", "9100_100_fit", "9100_200_fit", "9100_301_fit"]
    fits = ["r1", "r2", "r3", "r4"]
    area = 25.0  # cm^2
    thickness = 0.04  # cm (400 microns)
    freq = np.logspace(*range_maker(-5, 7, 50, fmt="np", is_exp=True))  # type: ignore

    data = {}
    for fit in fits:
        dfs = []
        for base in f_bases:
            file = f"{base}_{fit}.xlsx"
            if not (base_path / file).exists():
                print(f"File not found: {base_path / file}")
                continue
            fit_df, attrs_df = load_fit_results(base_path / file)

            param_df, info_df = combine_fit_data(fit_df, attrs_df)

            res = [param_df]
            # # Base organization
            # res.append(make_param_df(combined_data))
            # Add frequency and tau parameters
            res.append(add_frequency_parameters(param_df, info_df, freq))
            # Converts R to resistivity and C to permittivity
            res.append(add_material_parameters(param_df, info_df, area=area, thickness=thickness))
            # Adds f_RC, tau_RC, R_RC, C_RC
            res.append(add_peak_results(param_df, info_df, freq, area=area, thickness=thickness))
            # Appends condition deltas
            result_df = append_condition_deltas(pd.concat(res, axis=1))
            # Finalizes construction and saves
            dfs.append(
                convert_rc_to_multi_index(
                    result_df, True, dtypes={"Name": str, "condition": REVISED_COND}
                )
            )
            # changes: Organize (R and C)

        data[fit] = sort_columns(pd.concat(dfs))
        # data[(base_path / file).stem] = convert_rc_to_multi_index(result_df, True, dtypes={"Name": str})

    data_init = deepcopy(data)

    # %%
    data = group_points(
        data_init, r".*", False, False, gr_levels=("fit", ("fit", 0, 1), ("fit", 2, 1))
    )

    fit_data, fit_results = data_group_trend_eval(
        data,
        x_data=["temp", "sodium"],
        # fit_func="general",
        sign_default=0,
        fit_resid=True,
        col_selector=lambda s: bool(re.match(r".+(_RC$|\d$)", s)),
        mode="exp",
        cond_kwargs={"mode": "lin"},
        pass_kwargs_eval=lambda k, c: any(s in k for s in REVISED_COND[3:]),
    )
    fit_results_df = pd.DataFrame(fit_results, dtype=float).T
    fit_data["fit_results"] = fit_results_df[fit_param_sort(fit_results_df.columns)]

    save_fit_data(
        fit_data,
        base_path / "summaries" / "fit_summary.xlsx",
        reorganize_index=True,
        save_depths=(0, 1, 2),
    )


# def f_peak_stats(
#     values: list[float], stdevs: list[float], model: str, freq: np.ndarray
# ) -> tuple[float, float, float, float]:
#     """
#     Calculate the bounds of frequency and time constant by testing all combinations
#     of parameter variations.

#     Parameters
#     ----------
#     values : list[float]
#         List of parameter values
#     stdevs : list[float]
#         List of parameter standard deviations
#     model : str
#         Circuit model string
#     freq : np.ndarray
#         Frequency array for impedance calculation

#     Returns
#     -------
#     tuple[float, float, float, float]
#         Minimum fpeak, maximum fpeak, minimum tau, maximum tau
#     """
#     # Create variations of each parameter (min and max)
#     param_variations = []
#     for val, std in zip(values, stdevs):
#         param_variations.append([val - std, val + std])

#     # Generate all combinations of parameter variations
#     combinations = list(itertools.product(*param_variations))

#     # Calculate fpeak for each combination
#     fpeaks = []
#     for params in combinations:
#         fpeak = find_f_peak(params, model, freq)
#         if fpeak != 0:
#             fpeaks.append(fpeak)

#     # Return min and max values
#     if not fpeaks:
#         return 0.0, 0.0, 0.0, 0.0

#     # Convert fpeaks to tau values
#     taus = [f_r_c_conversion(f, default=0) for f in fpeaks]

#     return min(fpeaks), max(fpeaks), min(taus), max(taus)


# def find_f_peak(
#     params: tuple | list, model: str = "", f: np.ndarray | None = None, use_max=True
# ) -> float:
#     """
#     Find the frequency at which the imaginary part of the impedance is maximum.

#     Parameters
#     ----------
#     f : np.ndarray
#         Array of frequencies.
#     Z : np.ndarray
#         Array of complex impedance values.

#     Returns
#     -------
#     float
#         Frequency at which the imaginary part of the impedance is maximum.
#     """
#     if len(params) <= 1:
#         return 0.0

#     if not use_max or not model or f is None:
#         return f_r_c_conversion(*params, default=0)

#     circuit_func = wrapCircuit(model, {})
#     Z = np.array(np.hsplit(circuit_func(f, *params), 2)).T
#     Z = Z[:, 0] + 1j * Z[:, 1]

#     # Find the index of the maximum imaginary part of Z
#     max_index = np.argmax(abs(np.imag(Z)))

#     if max_index == 0 or max_index == len(Z) - 1:
#         # If the maximum is at the boundary, use the maximum of the imaginary part of the admittance
#         Z = 1 / Z
#         max_index = np.argmax(abs(np.imag(Z)))
#         if max_index == 0 or max_index == len(Z) - 1:
#             return 0.0

#     # Return the corresponding frequency
#     return f[max_index]


# def find_peak_stats(
#     params: tuple | list, model: str = "", f: np.ndarray | None = None, use_max=True
# ) -> tuple[float, float, float, float, np.ndarray]:
#     """
#     Find the frequency at which the imaginary part of the impedance is maximum.

#     Parameters
#     ----------
#     f : np.ndarray
#         Array of frequencies.
#     Z : np.ndarray
#         Array of complex impedance values.

#     Returns
#     -------
#     float
#         Frequency at which the imaginary part of the impedance is maximum.
#     """
#     if len(params) <= 1:
#         return 0.0, 0.0, 0.0, 0.0, np.array([])

#     if not use_max or not model or f is None:
#         f_peak = f_r_c_conversion(*params, default=0)
#         return f_peak, 0.0, f_r_c_conversion(f_peak, default=0), 0.0, np.array([])

#     circuit_func = wrapCircuit(model, {})
#     Z = np.array(np.hsplit(circuit_func(f, *params), 2)).T
#     Z = Z[:, 0] + 1j * Z[:, 1]

#     # Find the index of the maximum imaginary part of Z
#     max_index = np.argmax(abs(np.imag(Z)))

#     if max_index == 0 or max_index == len(Z) - 1:
#         # If the maximum is at the boundary, use the maximum of the imaginary part of the admittance
#         Z = 1 / Z
#         max_index = np.argmax(abs(np.imag(Z)))
#         if max_index == 0 or max_index == len(Z) - 1:
#             return 0.0, 0.0, 0.0, 0.0, np.array([])

#     return *peak_and_width_logf(f, Z, max_index), Z


# def peak_and_width_logf(freq, Z, i_peak) -> tuple[float, float, float, float]:
#     # Work in log10(f) for stable interpolation
#     logf = np.log10(freq)
#     y = abs(np.imag(Z))  # assuming Z_im is complex imag part; otherwise pass the array directly

#     # Peak index
#     # i_peak = np.nanargmax(y)
#     y_max = y[i_peak]
#     h = 0.5 * y_max

#     # Find crossings on each side via linear interpolation in logf
#     def crossing(idx_range):
#         i0, i1 = idx_range
#         x = logf[i0 : i1 + 1]
#         z = y[i0 : i1 + 1]
#         # indices where z crosses h
#         sign = np.sign(z - h)
#         cs = np.where(sign[:-1] * sign[1:] < 0)[0]
#         if len(cs) == 0:
#             return np.nan
#         k = cs[0]
#         # linear interpolation in x
#         x0, x1 = x[k], x[k + 1]
#         z0, z1 = z[k], z[k + 1]
#         t = (h - z0) / (z1 - z0)
#         res = x0 + t * (x1 - x0)
#         return res if np.isfinite(res) else np.nan

#     f_p = freq[i_peak]
#     f_lo = 10 ** crossing((0, i_peak))
#     f_hi = 10 ** crossing((i_peak, len(freq) - 1))

#     Q_f = f_p / (f_hi - f_lo)

#     tau_p = 1.0 / (2 * np.pi * f_p)

#     tau_lo = 1.0 / (2 * np.pi * f_hi)
#     tau_hi = 1.0 / (2 * np.pi * f_lo)
#     # dlogtau = -dlogf if np.isfinite(dlogf) else np.nan
#     Q_tau = tau_p / (tau_hi - tau_lo)

#     return f_p, Q_f, tau_p, Q_tau


# def f_r_c_conversion(*vals: float, default: float = 0.0) -> float:
#     """
#     Convert frequency to RC time constant.

#     Parameters
#     ----------
#     f : float
#         Frequency in Hz.

#     Returns
#     -------
#     float
#         RC time constant in seconds.
#     """
#     if any(v == 0 for v in vals) or not vals:
#         return default
#     return float(1 / (2 * np.pi * np.prod(vals)))
