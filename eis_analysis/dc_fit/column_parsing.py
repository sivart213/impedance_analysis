from typing import overload

import numpy as np
import pandas as pd

from eis_analysis.dc_fit.extract_tools import (
    BASE_KEYS,
    DEFAULT_KEYS,
    form_std_df_index,
)
from eis_analysis.equipment.temperature_devices.thermocouple import Thermocouple

np.seterr(invalid="raise")

tc = Thermocouple(thermo_type="K", unit="C")
PRE_DRY_SET = {"pre", "dry"}


# %% Standardization and Column Sorting
# def clear_nonraw_columns(df: pd.DataFrame, base: pd.Series | int = 1) -> None:


def clear_nonraw_columns(df: pd.DataFrame, temp: int | float = 0.0) -> None:
    """
    Clear non-raw columns in a DataFrame with array values for parsing.
    Sets all current and resistance columns to arrays of np.nan,
    sets Voltage and Voltage abs to arrays of 2, and Voltage rms to arrays of 2 * sqrt(2).
    Operates in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clear.
    temp : float
        Temperature value to use for temperature columns.
    """
    for col in df.columns[:8]:
        if col == "Voltage rms":
            val = 2.0 * np.sqrt(2)
        elif "Volt" in col:
            val = 2.0
        elif "Temp" in col:
            if not temp and "temp" in df.index.names:
                val = df.index.get_level_values("temp")
            else:
                val = temp
        else:
            val = np.nan
        df[col] = val


def convert_raw_data(data_df: pd.DataFrame) -> None:
    """
    Fill standard (non-raw) columns in data_df from raw columns and Ohm's law.

    This function updates data_df in-place. For each  column, it first tries to fill
    from the corresponding raw column. If not available, it attempts to calculate the value
    using Ohm's law (V = I * R) where possible.

    Expected columns:
        - Voltage, Current, Resistance, Temperature, Voltage rms, Current rms, Voltage abs, Current abs,
        raw_Voltage, raw_Current, raw_Resistance, raw_Temperature, raw_Voltage rms, raw_Current rms,
        raw_Voltage abs, raw_Current abs, multiplier, t_multiplier

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with  and raw columns. Modified in-place.
    """

    def apply_c_multiplier(key: str) -> None:
        """Apply the multiplier to the raw column if it exists."""
        std_key = data_df[key]
        mask = (data_df["multiplier"] != 1) & std_key.notna() & (std_key.abs() > 1e-8)
        data_df.loc[mask, key] = data_df.loc[mask, key] * data_df.loc[mask, "multiplier"]

    data_df["Resistance"] = data_df["raw_Resistance"]

    # Precompute sign
    raw_sign = data_df["raw_Voltage"].to_numpy(copy=True)
    mask = np.isnan(raw_sign)
    if np.any(mask):
        raw_sign[mask] = data_df["raw_Voltage rms"].to_numpy(copy=True)[mask]
        mask = np.isnan(raw_sign)
    if np.any(mask):
        raw_sign[mask] = data_df["raw_Current"].to_numpy(copy=True)[mask]
        mask = np.isnan(raw_sign)
    if np.any(mask):
        raw_sign[mask] = data_df["raw_Current rms"].to_numpy(copy=True)[mask]
        mask = np.isnan(raw_sign)
    raw_sign[mask] = 1.0
    sign = np.sign(raw_sign)

    # Voltage
    voltage = data_df["raw_Voltage"].to_numpy(copy=True)
    mask = np.isnan(voltage)
    if np.any(mask):
        voltage[mask] = data_df["raw_Voltage abs"].to_numpy(copy=True)[mask] * sign[mask]
        mask = np.isnan(voltage)
    if np.any(mask):
        voltage[mask] = data_df["raw_Voltage rms"].to_numpy(copy=True)[mask] / np.sqrt(2)
        mask = np.isnan(voltage)
    if np.any(mask):
        voltage[mask] = data_df["Voltage"].to_numpy(copy=True)[mask] * sign[mask]
    data_df["Voltage"] = voltage

    # Current
    current = data_df["raw_Current"].to_numpy(copy=True)
    mask = np.isnan(current)
    if np.any(mask):
        current[mask] = data_df["raw_Current abs"].to_numpy(copy=True)[mask] * sign[mask]
        mask = np.isnan(current)
    if np.any(mask):
        current[mask] = data_df["raw_Current rms"].to_numpy(copy=True)[mask] / np.sqrt(2)
        mask = np.isnan(current)
    if np.any(mask):
        current[mask] = (
            data_df["Voltage"].to_numpy(copy=True)[mask]
            / data_df["Resistance"].to_numpy(copy=True)[mask]
        )
    data_df["Current"] = current

    apply_c_multiplier("Current")

    # Resistance
    resistance = data_df["Resistance"].to_numpy(copy=True)
    mask = np.isnan(resistance)
    if np.any(mask):
        resistance[mask] = (
            data_df["Voltage"].to_numpy(copy=True)[mask]
            / data_df["Current"].to_numpy(copy=True)[mask]
        )
    data_df["Resistance"] = resistance

    # Temperature
    temp = (data_df["raw_Temperature"] * data_df["t_multiplier"]).abs().to_numpy(copy=True)
    mask = np.isnan(temp)
    if np.any(mask):
        temp[mask] = data_df["Temperature"].to_numpy(copy=True)[mask]
    # Thermocouple conversion
    temp = np.array(
        [tc.convert(float(t_val), ambient=30.0) if t_val < 10 else t_val for t_val in temp]
    )
    data_df["Temperature"] = temp

    # Voltage rms
    voltage_rms = data_df["raw_Voltage rms"].to_numpy(copy=True)
    mask = np.isnan(voltage_rms)
    if np.any(mask):
        voltage_rms[mask] = data_df["Voltage"].to_numpy(copy=True)[mask] * np.sqrt(2)
    data_df["Voltage rms"] = voltage_rms

    # Current rms
    current_rms = data_df["raw_Current rms"].to_numpy(copy=True)
    mask = np.isnan(current_rms)
    if np.any(mask):
        current_rms[mask] = data_df["Current"].to_numpy(copy=True)[mask] * np.sqrt(2)
    data_df["Current rms"] = current_rms

    # Voltage abs
    voltage_abs = data_df["raw_Voltage abs"].to_numpy(copy=True)
    mask = np.isnan(voltage_abs)
    if np.any(mask):
        voltage_abs[mask] = np.abs(data_df["Voltage"].to_numpy(copy=True)[mask])
    data_df["Voltage abs"] = voltage_abs

    # Current abs
    current_abs = data_df["raw_Current abs"].to_numpy(copy=True)
    mask = np.isnan(current_abs)
    if np.any(mask):
        current_abs[mask] = np.abs(data_df["Current"].to_numpy(copy=True)[mask])
    data_df["Current abs"] = current_abs

    apply_c_multiplier("Current rms")
    apply_c_multiplier("Current abs")


def evaluate_multipliers(df: pd.DataFrame, strict: bool = True) -> None:
    """
    Evaluate and update the 'multiplier' column in df for rows where:
    - multiplier is 1
    - raw_Current is not NaN and abs(raw_Current) > 1e-8

    The function tries to find the best multiplier for raw_Current by comparing to:
    1. The existing 'Current' value in the same row.
    2. If not available, mean 'Current' with the same sample, condition, and temp.
    3. If still not available, mean 'Current' with the same condition and temp.
    4. If still not available, use all temps for the same sample and condition, and select the multiplier that best fits the expected trend (abs(Current) increases with temp).

    Updates df['multiplier'] in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'multiplier', 'raw_Current', 'Current', 'sample_name', 'condition', 'temp'.
    strict : bool, optional
        If True, only consider rows where 'multiplier' is 1. If False, also consider rows with NaN 'Current'.
        Default is True.
    """

    def compare_multipliers(val, reference, reference2: float | None = None) -> float:
        best_multiplier = 1
        best_diff = float("inf")

        for m in [1, 1e-8, 1e-9, 1e-10]:
            # Avoid log(0) by clipping to a small positive value
            # diff = abs(reference - val * m)
            diff = abs(np.log10(max(abs(reference), 1e-20)) - np.log10(max(abs(val * m), 1e-20)))
            if reference2 is not None:
                if max(abs(val * m), 1e-20) > reference2:  # allow for floating point tolerance
                    continue
            if diff < best_diff:
                best_diff = diff
                best_multiplier = m
        return best_multiplier

    def temps_compare_multipliers(
        val: float, temps: pd.Series, t_idx, reference2: float | None = None
    ) -> float:
        multipliers = [1, 1e-8, 1e-9, 1e-10]
        temp_pos = np.searchsorted(temps.index, t_idx)
        if temp_pos >= len(temps) / 2:
            multipliers = sorted(multipliers)
        best_multiplier = 1
        best_score = float("inf")
        for m in multipliers:
            if reference2 is not None:
                if abs(val * m) > reference2:
                    continue
            test_df = temps.copy()
            test_df.loc[t_idx] = abs(val * m)
            vals = test_df.sort_index().values.astype(float)
            log_vals = np.log10(np.clip(vals, 1e-20, None))
            log_deltas = np.diff(log_vals)
            score = np.std(log_deltas)
            if np.all(log_deltas >= 0) and score < best_score:
                best_score = score
                best_multiplier = m
        return best_multiplier

    # Step 1: Filter for candidates
    mask = (df["multiplier"] == 1) & df["raw_Current"].notna() & (df["raw_Current"].abs() > 1e-8)
    candidates = df[mask].copy()

    # Precompute group means for efficiency
    group_sct = (
        df.abs().groupby(["sample_name", "condition", "temp"], observed=True)["Current"].mean()
    )
    group_sct = group_sct.where(group_sct <= 1e-8, np.nan)

    group_ct = group_sct.groupby(["condition", "temp"], observed=True).mean()
    # group_t = group_sct.groupby(["temp"]).mean()

    for idx, row in candidates.iterrows():
        idx = tuple(idx)  # type: ignore
        # raw_current = abs(row["raw_Current"])
        raw_current = df.abs()["raw_Current"][idx[:3]].mean()
        multipliers_found = []

        # 1. If Current is available, use it to find best multiplier
        if not pd.isna(row["Current"]) and row["Current"] <= 1e-8:
            m = compare_multipliers(raw_current, abs(row["Current"]))
            if m != 1:
                multipliers_found.append(m)

        # 3. Try with same sample, condition, temp (mean)
        targ_val = group_sct.loc[idx[:3]].mean()
        if not pd.isna(targ_val):
            m = compare_multipliers(raw_current, targ_val)
            if m != 1:
                multipliers_found.append(m)

        # --- Custom: Compare to mean of both conditions for this sample/temp, or to other condition as reference2 ---
        try:
            if idx[1] in PRE_DRY_SET:
                # Use mean of both conditions for this sample and temp
                both_cond_mean = group_sct.loc[(idx[0], list(PRE_DRY_SET), idx[2])].mean()
                if not pd.isna(both_cond_mean):
                    m = compare_multipliers(raw_current, both_cond_mean)
                    if m != 1:
                        multipliers_found.append(m)
            else:
                targ_val = group_sct.loc[idx[:3]].mean()
                both_cond_mean = group_sct.loc[(idx[0], list(PRE_DRY_SET), idx[2])].mean()
                if not pd.isna(both_cond_mean):
                    m = compare_multipliers(raw_current, both_cond_mean, reference2=both_cond_mean)
                    if m != 1:
                        multipliers_found.append(m)
        except KeyError:
            # If the index is not found, skip this step
            pass

        # 4. Try with all temps for same sample and condition, and select the multiplier that best fits the expected trend (abs(Current) increases with temp)
        temp_df = group_sct.loc[idx[:2]]
        temp_df = temp_df.sort_index().dropna()
        if len(temp_df) >= 2 or (len(temp_df) == 1 and temp_df.get(idx[2], np.nan) is np.nan):
            m = temps_compare_multipliers(raw_current, temp_df, idx[2])
            if m != 1:
                multipliers_found.append(m)

        # 6. Try with all temps for same sample and condition group, and select the multiplier that best fits the expected trend (abs(Current) increases with temp)
        try:
            m = 1
            if idx[1] in PRE_DRY_SET:
                temp_df = group_sct.loc[(idx[0], list(PRE_DRY_SET))].groupby(level=2).mean()
                temp_df = temp_df.sort_index().dropna()
                if len(temp_df) >= 2 or (
                    len(temp_df) == 1 and temp_df.get(idx[2], np.nan) is np.nan
                ):  # type: ignore
                    m = temps_compare_multipliers(raw_current, temp_df, idx[2])
            else:
                temp_df = group_sct.loc[idx[:2]]
                temp_df = temp_df.sort_index().dropna()
                if len(temp_df) >= 2 or (
                    len(temp_df) == 1 and temp_df.get(idx[2], np.nan) is np.nan
                ):  # type: ignore
                    m = temps_compare_multipliers(raw_current, temp_df, idx[2])
            if m != 1:
                multipliers_found.append(m)
        except KeyError:
            # If the index is not found, skip this step
            pass

        # Choose the most common non-1 multiplier, or 1 if none found
        if multipliers_found:
            # Count occurrences, prefer most common non-1
            df.loc[idx, "multiplier"] = pd.Series(multipliers_found).value_counts().idxmax()

        # else:
        if not strict:
            # 2. Try with same condition, temp (mean)
            targ_val = group_ct.loc[idx[1:3]].mean()
            if not pd.isna(targ_val):
                m = compare_multipliers(raw_current, targ_val)
                if m != 1:
                    multipliers_found.append(m)

            # 5. Try with all temps for same condition, and select the multiplier that best fits the expected trend (abs(Current) increases with temp)
            temp_df = group_ct.loc[idx[1]]
            temp_df = temp_df.sort_index().dropna()
            if len(temp_df) >= 2 or (len(temp_df) == 1 and temp_df.get(idx[2], np.nan) is np.nan):
                m = temps_compare_multipliers(raw_current, temp_df, idx[2])
                if m != 1:
                    multipliers_found.append(m)

            try:
                # 6. Try with all temps for same condition group, and select the multiplier that best fits the expected trend (abs(Current) increases with temp)
                if idx[1] in PRE_DRY_SET:
                    temp_df = (
                        group_ct.loc[list(PRE_DRY_SET)].groupby(level=1, observed=True).mean()
                    )
                else:
                    temp_df = group_ct.loc[idx[1]]
                temp_df = temp_df.sort_index().dropna()
                if len(temp_df) >= 2 or (
                    len(temp_df) == 1 and temp_df.get(idx[2], np.nan) is np.nan
                ):  # type: ignore
                    m = temps_compare_multipliers(raw_current, temp_df, idx[2])
                    if m != 1:
                        multipliers_found.append(m)
            except KeyError:
                # If the index is not found, skip this step
                pass

            # Choose the most common non-1 multiplier, or 1 if none found
            if multipliers_found:
                # Count occurrences, prefer most common non-1
                df.loc[idx, "multiplier"] = pd.Series(multipliers_found).value_counts().idxmax()


def has_a_current(df: pd.DataFrame, idx: pd.Index) -> bool:
    """
    Check if the DataFrame has a valid current column for the given index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for current values.
    idx : pd.Index
        Index to check within the DataFrame.

    Returns
    -------
    bool
        True if any of the current columns have non-null values for the given index, False otherwise.
    """
    return bool(
        df.loc[list(idx), "raw_Current"].notna().all()
        or df.loc[list(idx), "raw_Current abs"].notna().all()
        or df.loc[list(idx), "raw_Current rms"].notna().all()
    )


def has_values(df: pd.DataFrame, idx: pd.Index, col: str) -> bool:
    """
    Check if a specific column in the DataFrame has non-null values for the given index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    idx : pd.Index
        Index to check within the DataFrame.
    col : str
        Column name to check for non-null values.

    Returns
    -------
    bool
        True if all values in the specified column for the given index are non-null, False otherwise.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    return bool(df.loc[list(idx), col].notna().all())


def valid_temp(df, key) -> bool:
    limit = 0.03 if "auxin" in key else 0.003
    return (df[key].dropna().abs() < limit).all()


@overload
def has_cycling(data: pd.Series, pass_single: bool = ...) -> bool: ...
@overload
def has_cycling(data: pd.DataFrame, pass_single: bool = ...) -> pd.Series: ...


def has_cycling(data: pd.Series | pd.DataFrame, pass_single: bool = False) -> bool | pd.Series:
    """
    Check if each value has an inverse sign to the previous.

    For a Series, returns a bool.
    For a DataFrame, returns a Series of bools for each column.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data to check for clear cycling.

    pass_single : bool, optional
        If True, a single value is considered clear cycling.
        Default is False, meaning a single value will return False.
        Invert desired behavior if checking `not clear_cycling(...)`.

    Returns
    -------
    bool or pd.Series of bool
        For a Series input, returns True if every consecutive value has an opposite sign.
        For a DataFrame input, returns a Series of bools indicating this for each column.
    """

    def _series_clear_cycling(series: pd.Series) -> bool:
        values = series.dropna().to_numpy(copy=True)
        if len(values) < 2:
            return pass_single
        return (np.sign(values[1:]) == -np.sign(values[:-1])).all()

    if isinstance(data, pd.DataFrame):
        return data.apply(_series_clear_cycling)
    else:
        return _series_clear_cycling(data)


def parse_columns(cols: set[str], pre_fix: str = "raw_Current") -> pd.Series:
    """
    Return a Series with keys as suffixes and values as column names (or np.nan if not found).

    Parameters
    ----------
    cols : set[str]
        Set of column names.
    pre_fix : str, optional
        Prefix to add to the keys (e.g., 'raw_'), by default 'raw_'.

    Returns
    -------
    pd.Series
        Series with index ["", " abs", " rms"] (with prefix) and values as column names or pd.NA.
    """
    result = pd.Series({pre_fix + k: pd.NA for k in ["", " abs", " rms"]}, dtype=str)
    for col in cols:
        if col.endswith(".r"):
            result[pre_fix + " abs"] = col
        elif col.endswith(".y"):
            result[pre_fix + " rms"] = col
        else:
            result[pre_fix + ""] = col
    return result


def review_remaining(
    df: pd.DataFrame,
    map_df: pd.DataFrame,
    group_df: pd.DataFrame,
    group_idx: pd.Index,
    unpicked_cols: list,
    mode: str = "temp",
    tol: float = 0.05,
) -> list:
    """
    Assign the remaining column as temperature or current if only one column is left and criteria are met.

    Parameters
    ----------
    df : pd.DataFrame
        The standardized DataFrame to update.
    map_df : pd.DataFrame
        The mapping DataFrame to update.
    group_df : pd.DataFrame
        The group DataFrame containing the candidate columns.
    group_idx : pd.Index
        The index for the group.
    unpicked_cols : list
        List of columns not yet assigned.
    mode : str
        "temp" to assign as temperature, "current" to assign as current.
    tol : float, optional
        Tolerance for numerical comparisons, default is 0.05.

    Returns
    -------
    unpicked_cols : list
        Updated list of unpicked columns.
    """
    if not unpicked_cols:
        return []

    if mode == "current" and 1 <= len(unpicked_cols) <= 3:
        # All columns must share the same base
        def get_base(col: str) -> str:
            return ".".join(col.split(".")[:-1]) if "." in col else col

        # Get standardized column mapping for unpicked and already assigned columns
        # Use parse_columns to get mapping from standardized names to actual column names
        col_series = parse_columns(set(unpicked_cols.copy()), pre_fix="raw_Current")

        # Add already assigned raw current columns for verification
        overlap = set()
        for std_col in col_series.index:
            if has_values(df, group_idx, col=std_col):
                # if the current col has values -> include for comparison
                if (root_col := map_df.loc[group_idx[0], std_col]) in unpicked_cols:
                    # No cols in unpicked should have values, retain the violations for removal later
                    overlap.add(root_col)
                col_series[std_col] = root_col

        col_series = col_series.dropna()

        # If there is overlap, we can't set all passed unpicked, so return original
        if overlap:
            return [col for col in unpicked_cols if col not in overlap]

        if len({get_base(c) for c in set(col_series.values)}) != 1:
            return unpicked_cols

        vals = {
            str(k): group_df[v].dropna() for k, v in col_series.items() if v in group_df.columns
        }

        # Only proceed if all present columns are non-empty and not in exponential form
        if not vals:
            return unpicked_cols
        for key, col in col_series.items():
            key = str(key)
            if pd.isna(col):
                continue
            # Make sure source cols are not empty or converted from nA/pA to A
            if col in unpicked_cols and (vals[key].empty or (vals[key].abs() < 1e-5).all()):
                return unpicked_cols
            # Try to revert any added cols that are in A or drop.
            elif col not in unpicked_cols and (vals[key].abs() < 1e-5).all():
                # Try dividing by multiplier to see if it brings values in spec
                candidate = vals[key] / df.loc[group_idx, "multiplier"]
                if not (candidate.abs() < 1e-5).all():
                    vals[key] = candidate
                else:
                    del vals[key]

        # Verification logic
        checks = []
        # Use suffixes as keys: ["raw_Current", "raw_Current abs", "raw_Current rms"]
        if "raw_Current" in vals:
            checks.append(has_cycling(vals["raw_Current"], pass_single=True))
            if "raw_Current abs" in vals:
                checks.append(
                    np.allclose(
                        vals["raw_Current abs"],
                        vals["raw_Current"].abs(),
                        rtol=tol,
                        atol=1e-12,
                    )
                )
                checks.append(not has_cycling(vals["raw_Current abs"]))
            if "raw_Current rms" in vals:
                checks.append(
                    np.allclose(
                        vals["raw_Current rms"],
                        vals["raw_Current"] * np.sqrt(2),
                        rtol=tol,
                        atol=1e-12,
                    )
                )
                checks.append(has_cycling(vals["raw_Current rms"], pass_single=True))
        elif "raw_Current abs" in vals and "raw_Current rms" in vals:
            checks.append(
                np.allclose(
                    vals["raw_Current rms"].abs(),
                    vals["raw_Current abs"] * np.sqrt(2),
                    rtol=tol,
                    atol=1e-12,
                )
            )
            checks.append(not has_cycling(vals["raw_Current abs"]))
            checks.append(has_cycling(vals["raw_Current rms"], pass_single=True))
        if not checks:
            # Only allow unchecked setting if there is only one unpicked column provided
            if len(unpicked_cols) == 1:
                checks.append(True)
            else:
                return unpicked_cols

        if all(checks):
            # Assign all present columns
            for std_col, val in vals.items():
                if not has_values(df, group_idx, col=std_col):
                    df.loc[group_idx, std_col] = val
                    map_df.loc[group_idx, std_col] = col_series[std_col]
            return []
        return unpicked_cols

    elif (
        mode == "temp"
        and len(unpicked_cols) == 1
        and not has_values(df, group_idx, "raw_Temperature")
        and valid_temp(group_df, unpicked_cols[0])
        and has_a_current(df, group_idx)
        and not has_cycling(group_df[unpicked_cols[0]])
    ):
        df.loc[group_idx, "raw_Temperature"] = group_df[unpicked_cols[0]]
        map_df.loc[group_idx, "raw_Temperature"] = unpicked_cols[0]
        if "auxin" in unpicked_cols[0]:
            df.loc[group_idx, "t_multiplier"] = 1e2
        return []
    elif (
        mode == "unknown"
        and len(unpicked_cols) == 1
        and has_a_current(df, group_idx)
        # and not has_values(df, group_df.index, "Unknown")
    ):
        # Only try to assign as unknown if raw_Temperature is not filled
        unknowns = unpicked_cols.copy()
        if not has_values(df, group_idx, "raw_Temperature"):
            # Try to assign as temp first
            unknowns = review_remaining(
                df, map_df, group_df, group_idx, unpicked_cols, mode="temp"
            )
        if unknowns:
            # Try to assign to current
            unknowns = review_remaining(
                df, map_df, group_df, group_idx, unpicked_cols, mode="current"
            )
        if unknowns:
            # If still not assigned, assign as Unknown
            # df.loc[group_idx, "Unknown"] = group_df[unpicked_cols[0]]
            df.loc[group_idx, unpicked_cols[0]] = group_df[unpicked_cols[0]]
            # map_df does not have Unknown, but could be added if needed
        return []

    return unpicked_cols


def resistance_trend_check(df: pd.DataFrame) -> None:
    """
    Enforce that for each (sample_name, condition):
    - For each temperature group, the max Resistance at lower temp is greater than the min Resistance at higher temp.
    - For each temperature, 'dh' resistance is lower than 'pre' and 'dry' at the same temp.
    Only modifies rows where multiplier != 1.
    Modifies df in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index levels including 'sample_name', 'condition', 'temp', and columns
         including 'Resistance', 'multiplier', 'Voltage', 'Current', etc.
    """
    mask_unconfirmed = df["multiplier"] != 1

    # Precompute unique sample_names and conditions
    sample_names = df.index.get_level_values("sample_name")
    conditions = df.index.get_level_values("condition")
    temps = df.index.get_level_values("temp")

    # Loop over all unique (sample_name, condition) pairs
    for sample_name in sample_names.unique():
        for condition in conditions.unique():
            group_mask = (
                (sample_names == sample_name) & (conditions == condition) & mask_unconfirmed
            )
            group_df = df[group_mask]
            if group_df.empty or group_df["Resistance"].isna().all():
                continue

            # Split by voltage polarity
            polarity = np.sign(group_df["Voltage"].fillna(0))
            for pol in [-1, 1]:
                pol_mask = polarity == pol
                if not pol_mask.any():
                    continue
                pol_group_df = group_df[pol_mask]
                temp_groups = pol_group_df.groupby(pol_group_df.index.get_level_values("temp"))
                temp_keys = sorted(temp_groups.groups.keys())
                if temp_keys:
                    prev = 10 * temp_groups.get_group(temp_keys[0])["Resistance"].copy()
                    for temp in temp_keys:
                        temp_group = temp_groups.get_group(temp).copy()
                        # temp_min = temp_group["Resistance"].min()
                        temp_max = temp_group["Resistance"].max()

                        if temp_max >= prev.min():
                            violators = temp_group[temp_group["Resistance"] >= prev.mean()]
                            for idx in violators.index:
                                current_res = df.loc[idx, "Resistance"]
                                required_shift = np.ceil(
                                    np.log10(current_res / (prev.mean() * 0.99))
                                )
                                df.loc[idx, "multiplier"] = df.loc[idx, "multiplier"] * (
                                    10**required_shift
                                )
                                temp_group.loc[idx, "multiplier"] = df.loc[idx, "multiplier"]
                            clear_nonraw_columns(temp_group)
                            convert_raw_data(temp_group)
                        # prev_min = temp_group["Resistance"].min()
                        prev = temp_group["Resistance"].copy()

                # After all adjustments, normalize multipliers to allowed set
                group_mask = (
                    (sample_names == sample_name) & (conditions == condition) & mask_unconfirmed
                )
                group_multipliers = np.log10(df[group_mask]["multiplier"].to_numpy(copy=True))
                # Find the exponent range of multipliers
                if group_multipliers.min() < -10:
                    df.loc[group_mask, "multiplier"] *= 10 ** (-10 - group_multipliers.min())
                elif group_multipliers.max() > -8:
                    df.loc[group_mask, "multiplier"] /= 10 ** (group_multipliers.max() + 8)

    clear_nonraw_columns(df)
    convert_raw_data(df)

    # Now enforce dh < pre and dh < dry for each sample_name
    for sample_name in df.index.get_level_values("sample_name").unique():
        # Get all dh rows for this sample
        dh_mask = (sample_names == sample_name) & (conditions == "dh") & mask_unconfirmed
        dh_block = df[dh_mask]
        if dh_block.empty:
            continue

        # For each temp, compare mean dh resistance to mean pre/dry resistance
        # required_shifts = []
        required_shift = 0
        for temp in dh_block.index.get_level_values("temp").unique():
            # Get mean dh resistance at this temp
            dh_temp_mask = dh_block.index.get_level_values("temp") == temp
            dh_mean = dh_block[dh_temp_mask]["Resistance"].mean()

            # Get mean pre/dry resistance at this temp
            # for cond in ["pre", "dry"]:
            cond_mask = (
                (sample_names == sample_name)
                & ((conditions == "pre") | (conditions == "dry"))
                & (temps == temp)
            )
            cond_block = df[cond_mask]
            if cond_block.empty:
                continue
            cond_mean = cond_block["Resistance"].mean()
            if dh_mean >= cond_mean:
                # Calculate required shift for the whole dh block
                required_shift = max(
                    required_shift, np.ceil(np.log10(dh_mean / (cond_mean * 0.99)))
                )
                # required_shifts.append(required_shift)

        if required_shift:
            # Use the maximum required shift to ensure all temps are below pre/dry
            # total_shift = max(required_shifts)
            df.loc[dh_mask, "multiplier"] = df.loc[dh_mask, "multiplier"] * (10**required_shift)

            group_mask = (sample_names == sample_name) & mask_unconfirmed
            group_multipliers = np.log10(df[group_mask]["multiplier"].to_numpy(copy=True))
            # Find the exponent range of multipliers
            if group_multipliers.min() < -10:
                df.loc[group_mask, "multiplier"] *= 10 ** (-10 - group_multipliers.min())
            elif group_multipliers.max() > -8:
                df.loc[group_mask, "multiplier"] /= 10 ** (group_multipliers.max() + 8)

    # Recalculate all standard columns after multiplier changes
    clear_nonraw_columns(df)
    convert_raw_data(df)


def evaluate_segments(
    unsorted_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse and standardize the segments of the results DataFrame.

    Parameters
    ----------
    unsorted_df : pd.DataFrame
        DataFrame containing the results to be standardized.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - The standardized DataFrame with columns for Voltage, Current, Resistance, Temperature, etc.
        - A mapping DataFrame that maps raw columns to standardized columns.
    """
    unsorted_df = unsorted_df.copy()

    base_cols = [
        "Voltage",
        "Current",
        "Resistance",
        "Temperature",
        "Voltage rms",
        "Current rms",
        "Voltage abs",
        "Current abs",
    ]

    raw_cols = [f"raw_{col}" for col in base_cols]

    std_cols = base_cols + raw_cols + ["multiplier", "t_multiplier"]

    map_df = form_std_df_index(pd.DataFrame(index=unsorted_df.index, columns=raw_cols, dtype=str))
    sorted_df = form_std_df_index(
        pd.DataFrame(
            index=unsorted_df.index,
            columns=std_cols + list(unsorted_df.columns),
            dtype=float,
        )
    )
    del base_cols
    # del raw_cols
    # del std_columns

    sorted_df["multiplier"] = 1.0
    sorted_df["t_multiplier"] = 1e3

    for col in list(unsorted_df.columns):
        if "fft" in col:
            sorted_df[col] = unsorted_df[col]
            unsorted_df.drop(columns=col, inplace=True)
        elif "imps" in col:
            sorted_df["raw_Resistance"] = sorted_df["raw_Resistance"].combine_first(
                unsorted_df[col]
            )
            map_df.loc[sorted_df["raw_Resistance"].notna(), "raw_Resistance"] = col
            unsorted_df.drop(columns=col, inplace=True)

    # sorted_df["Unknown"] = np.nan

    # --- Pass 1: Identify voltage, voltage_rms, resistance, abs/rms current, and possible current col ---
    history = {}
    for group_keys, group_df in unsorted_df.iloc[:, 1:].groupby(
        level=list(BASE_KEYS), observed=True
    ):
        group_df = group_df.copy().dropna(axis=1, how="all")
        # tool_cols = [c for c in group_df.columns if c not in ("date",)]
        unpicked_cols = []

        # Build a single assignment series for voltage and current
        col_map = pd.Series(index=raw_cols, dtype=str)

        for col in group_df.columns:
            vals = group_df[col].dropna().to_numpy(copy=True)
            if len(vals) == 0:
                continue
            minv, maxv = np.nanmin(np.abs(vals)), np.nanmax(np.abs(vals))

            # Voltage (DC or RMS)
            if (1.9 < minv < 2.1 and 1.9 < maxv < 2.1) or (2.8 < minv < 3.0 and 2.8 < maxv < 3.0):
                # Update col_map with voltage columns
                col_map.update(parse_columns({col}, pre_fix="raw_Voltage").dropna())
                # pd.concat([col_map, parse_columns({col}, pre_fix="raw_Voltage").dropna()])
            # Current (abs or rms, reliable catch)
            elif maxv < 1e-8:
                # Update col_map with current columns
                col_map.update(parse_columns({col}, pre_fix="raw_Current").dropna())
                # pd.concat([col_map, parse_columns({col}, pre_fix="raw_Current")])
            # Unknown (possible current)
            elif (
                minv != 0
                and maxv != 0
                and -5 > np.log10(abs(minv)) > -8
                and -5 > np.log10(abs(maxv)) > -8
            ):
                # sorted_df.loc[group_df.index, "Unknown"] = group_df[col]
                sorted_df.loc[group_df.index, col] = group_df[col]
            else:
                unpicked_cols.append(col)

        # Assign all found voltage/current columns
        for std_col, raw_col in col_map.dropna().items():
            sorted_df.loc[group_df.index, str(std_col)] = group_df[raw_col]
            map_df.loc[group_df.index, str(std_col)] = raw_col

        # only effective if there is only 1 column left
        unpicked_cols = review_remaining(
            sorted_df, map_df, group_df, group_df.index, unpicked_cols, mode="temp"
        )

        history[tuple(group_keys)] = unpicked_cols
        if not unpicked_cols:
            map_df.loc[group_df.index, :] = map_df.loc[group_df.index, :].fillna("N/A")

    # Fill standard columns from raw columns and Ohm's law
    clear_nonraw_columns(sorted_df)
    convert_raw_data(sorted_df)

    # --- Pass 2: Check for cyclical values to attempt to fill value
    # At this stage unknowns should be > 1 with temp likely one of them
    for group_keys, group_df in unsorted_df.iloc[:, 1:].groupby(
        level=list(BASE_KEYS), observed=True
    ):
        unpicked_cols: list = history.get(tuple(group_keys), [])
        if not unpicked_cols:
            continue
        group_df = group_df.copy().dropna(axis=1, how="all")

        cycling_cols = has_cycling(group_df[unpicked_cols])
        if not any(cycling_cols):
            del cycling_cols
            continue

        # Sort cycling_cols so cycling columns (True) come first, then iterate index
        cycling_cols_sorted = cycling_cols.sort_values(ascending=False)
        still_unpicked_cols = []
        for col in cycling_cols_sorted.index:
            still_unpicked_cols.extend(
                review_remaining(
                    sorted_df, map_df, group_df, group_df.index, [col], mode="current"
                )
            )

        unpicked_cols = review_remaining(
            sorted_df, map_df, group_df, group_df.index, still_unpicked_cols, mode="temp"
        )
        del still_unpicked_cols
        del cycling_cols
        del cycling_cols_sorted

        if (
            unpicked_cols
            and has_values(sorted_df, group_df.index, "raw_Temperature")
            and has_a_current(sorted_df, group_df.index)
        ):
            # If there are still unpicked columns and we have a temp, try to assign them
            unpicked_cols = review_remaining(
                sorted_df, map_df, group_df, group_df.index, unpicked_cols, mode="unknown"
            )

        history[tuple(group_keys)] = unpicked_cols
        if not unpicked_cols:
            map_df.loc[group_df.index, :] = map_df.loc[group_df.index, :].fillna("N/A")

    # Fill standard columns from raw columns and Ohm's law
    evaluate_multipliers(sorted_df)
    clear_nonraw_columns(sorted_df)
    convert_raw_data(sorted_df)

    # --- Pass 3: Assign temperature columns by matching to index temp ---
    for group_keys, group_df in unsorted_df.iloc[:, 1:].groupby(
        level=list(BASE_KEYS), observed=True
    ):
        unpicked_cols = history.get(tuple(group_keys), [])
        if not unpicked_cols:
            continue
        group_df = group_df.copy().dropna(axis=1, how="all")

        if has_values(sorted_df, group_df.index, "raw_Temperature"):
            # If temperature is already assigned, skip this group
            continue

        best_score = float("inf")
        best_candidate = None

        # Try each unpicked column as a possible temperature
        for col in unpicked_cols:
            limit = 15 if col in set(map_df["raw_Temperature"]) else 4
            candidate_vals = abs(group_df[col].dropna() * (1e2 if "auxin" in col else 1e3))
            if candidate_vals.empty or any(candidate_vals > 3):
                continue
            converted = tc.convert(candidate_vals.to_numpy(copy=True), ambient=30.0)
            idx_temp = float(group_keys[2])
            score = np.nanmean(np.abs(converted - idx_temp))
            if np.any(np.abs(converted - idx_temp) <= limit) and score < best_score:
                best_score = score
                best_candidate = col

        # If a temperature column is found, assign it
        if best_candidate is not None:
            # review_remaining(
            #     sorted_df, map_df, group_df, group_df.index, [best_candidate], mode="temp"
            # )
            sorted_df.loc[group_df.index, "raw_Temperature"] = group_df[best_candidate]
            map_df.loc[group_df.index, "raw_Temperature"] = best_candidate
            if "auxin" in best_candidate:
                sorted_df.loc[group_df.index, "t_multiplier"] = 1e2
            # unpicked_cols = [c for c in unpicked_cols if c != best_candidate]
            unpicked_cols.remove(best_candidate)
            unpicked_cols = review_remaining(
                sorted_df, map_df, group_df, group_df.index, unpicked_cols, mode="current"
            )
            unpicked_cols = review_remaining(
                sorted_df, map_df, group_df, group_df.index, unpicked_cols, mode="unknown"
            )
            history[tuple(group_keys)] = unpicked_cols
            if not unpicked_cols:
                map_df.loc[group_df.index, :] = map_df.loc[group_df.index, :].fillna("N/A")

    # Recalculate after temperature assignment
    evaluate_multipliers(sorted_df)
    clear_nonraw_columns(sorted_df)
    convert_raw_data(sorted_df)

    # --- Pass 4: Assign remaining columns by review_remaining only ---
    # This pass should be placed after convert_raw_data(sorted_df)
    for group_keys, group_df in unsorted_df.iloc[:, 1:].groupby(
        level=list(BASE_KEYS), observed=True
    ):
        unpicked_cols = history.get(tuple(group_keys), [])
        if not unpicked_cols:
            # Clean up empty lists from history
            history.pop(tuple(group_keys), None)
            continue
        group_df = group_df.copy().dropna(axis=1, how="all")

        # Try to assign each remaining column using review_remaining (unknown mode)
        still_unpicked_cols = []
        for col in unpicked_cols:
            still_unpicked_cols.extend(
                review_remaining(
                    sorted_df, map_df, group_df, group_df.index, [col], mode="unknown"
                )
            )

        # Update history, removing empty lists
        if still_unpicked_cols:
            history[tuple(group_keys)] = still_unpicked_cols
        else:
            history.pop(tuple(group_keys), None)
            map_df.loc[group_df.index, :] = map_df.loc[group_df.index, :].fillna("N/A")

        del still_unpicked_cols

    # Recalculate after temperature assignment
    evaluate_multipliers(sorted_df)
    clear_nonraw_columns(sorted_df)
    convert_raw_data(sorted_df)

    # --- Pass 5: Assign remaining columns by trend fit ---
    # For each group with remaining unpicked columns, test assignment using a larger dataset (same sample and condition)
    for group_keys, unpicked_cols in list(history.items()):
        test_std_df = pd.DataFrame(sorted_df.xs(group_keys[:2], drop_level=False).copy(deep=True))
        # Should be able to recreate test group this way
        group_df = pd.DataFrame(
            unsorted_df.xs(group_keys, drop_level=False).copy(deep=True)
        ).dropna(axis=1, how="all")

        # For each unpicked column, test if assigning it as raw_Current produces a monotonic trend with temp
        best_score = -np.inf
        best_assignment = None

        test_std_df = test_std_df.sort_values(by="Temperature", ascending=True)
        multipliers = test_std_df.abs().groupby("temp").mean()["multiplier"].mode().item()
        for col in unpicked_cols:
            # Assign candidate to raw_Current for the test group
            test_std_df.loc[group_df.index, "Current"] = group_df[col] * multipliers

            # Fill standard columns and update multiplier
            current_vals = test_std_df.abs().groupby("temp").mean()["Current"].to_numpy(copy=True)

            is_monotonic = np.sum(np.diff(current_vals) >= 0)
            score = is_monotonic / test_std_df["Current"].notna().sum()

            if score > best_score:
                best_score = score
                best_assignment = col

        # Apply the best assignment if found
        if best_assignment is not None:
            sorted_df.loc[group_df.index, "raw_Current"] = group_df[best_assignment]
            map_df.loc[group_df.index, "raw_Current"] = best_assignment

        # Try to assign any remaining as temp or unknown
        unpicked_cols = review_remaining(
            sorted_df, map_df, group_df, group_df.index, unpicked_cols, mode="temp"
        )
        unpicked_cols = review_remaining(
            sorted_df, map_df, group_df, group_df.index, unpicked_cols, mode="unknown"
        )

        # Update history, removing empty lists
        if unpicked_cols:
            history[group_keys] = unpicked_cols
        else:
            history.pop(group_keys, None)
            map_df.loc[group_df.index, :] = map_df.loc[group_df.index, :].fillna("N/A")

    # sorted_df = sorted_df.dropna(axis=1, how="all", subset = unsorted_df.columns)
    sorted_df = sorted_df[
        [col for col in sorted_df.columns if col in std_cols or not sorted_df[col].isna().all()]
    ]

    evaluate_multipliers(sorted_df, False)
    clear_nonraw_columns(sorted_df)
    convert_raw_data(sorted_df)

    resistance_trend_check(sorted_df)
    # sorted_df.columns.get_loc("t_multiplier")
    sorted_df.insert(sorted_df.columns.get_loc("t_multiplier") + 1, "date", unsorted_df["date"].copy())  # type: ignore[call-overload]
    # sorted_df["date"] = unsorted_df["date"].copy()
    # columns_to_drop = [col for col in unsorted_df.columns if sorted_df[col].isna().all()]
    # sorted_df = sorted_df.drop(columns=columns_to_drop)
    # std_cols
    return sorted_df, map_df


def apply_col_mapping(
    arrays_dict: dict[tuple, pd.DataFrame],
    parent_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> dict[tuple, pd.DataFrame]:
    """
    Build a dict of standardized segment DataFrames from arrays_dict, applying mapping and multipliers,
    and running convert_raw_data on each segment. Optionally trims each segment to the stable voltage region.

    Parameters
    ----------
    arrays_dict : dict[tuple, pd.DataFrame]
        Dict of segment DataFrames, as returned by process_segments_mean.
    parent_df : pd.DataFrame
        DataFrame with multipliers and columns for each segment (index must match arrays_dict keys).
    mapping_df : pd.DataFrame
        DataFrame mapping standardized columns to raw columns.

    Returns
    -------
    standardized_dict : dict[tuple, pd.DataFrame]
        Dict of standardized segment DataFrames, ready for fitting.
    """
    # Prepare all std_dfs and collect for batch processing
    std_dfs = {}
    segment_mod = False
    if len(next(iter(arrays_dict.keys()))) != 6:
        segment_mod = True

    for idx, seg_df in arrays_dict.items():
        if segment_mod:
            idx = idx + (0,)
        std_df = pd.DataFrame(index=seg_df.index, columns=parent_df.columns, dtype=float)
        for col in ["multiplier", "t_multiplier"]:
            if col in parent_df.columns:
                std_df[col] = parent_df.loc[idx, col]
        for col in mapping_df.columns:
            mapping_val = mapping_df.loc[idx, col]
            if mapping_val == "N/A" or mapping_val not in seg_df.columns:
                std_df[col] = np.nan
            else:
                std_df[col] = seg_df[mapping_val]
        clear_nonraw_columns(std_df, float(idx[2]))

        std_df.insert(0, "time", seg_df["time"].copy())
        std_dfs[idx] = std_df

    # Concatenate all segments for batch processing
    all_df = form_std_df_index(pd.concat(std_dfs, axis=0), names=DEFAULT_KEYS + ("meas_time",))

    # Apply convert_raw_data in bulk
    convert_raw_data(all_df)
    idx_len = len(next(iter(std_dfs))) - (1 if segment_mod else 0)
    standardized_dict = {}
    for idx in std_dfs:
        standardized_dict[idx[:idx_len]] = all_df.loc[idx, :"raw_Current abs"]
        standardized_dict[idx[:idx_len]].attrs |= arrays_dict[idx[:idx_len]].attrs.copy()
    # if segment_mod:
    #     # all_df = all_df.droplevel("segment", axis=0)
    #     standardized_dict = {
    #         idx[:-1]: all_df.loc[idx, :"raw_Current abs"] for idx in std_dfs.keys()
    #     }
    # else:
    #     standardized_dict = {idx: all_df.loc[idx, :"raw_Current abs"] for idx in std_dfs.keys()}

    return standardized_dict

    # all_df = pd.concat(
    #     std_dfs, axis=0, names=["sample_name", "condition", "temp", "run", "segment", "meas_time"]
    # ).sort_index()


# def clear_nonraw_columns(df: pd.DataFrame) -> None:
#     """
#     Clear non-raw columns in a DataFrame with array values.
#     Sets all current and resistance columns to arrays of np.nan,
#     sets Voltage and Voltage abs to arrays of 2, and Voltage rms to arrays of 2 * sqrt(2).
#     Operates in-place.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame to clear.
#     lengths : pd.Series or list[int]
#         Series or list of array lengths for each row (must align with df.index).
#     """
#     for col in df.columns[:8]:
#         if col == "Voltage rms":
#             val = 2.0 * np.sqrt(2)
#         elif "Volt" in col:
#             val = 2.0
#         elif "Temp" in col:
#             val = df.index.get_level_values("temp")
#         else:
#             val = np.nan
#         df[col] = val


# def build_standard_df_old(
#     arrays_dict: dict[tuple, pd.DataFrame],
#     parent_df: pd.DataFrame,
#     mapping_df: pd.DataFrame,
# ) -> dict[tuple, pd.DataFrame]:
#     """
#     Build a dict of standardized segment DataFrames from arrays_dict, applying mapping and multipliers,
#     and running convert_raw_data on each segment. Optionally trims each segment to the stable voltage region.

#     Parameters
#     ----------
#     arrays_dict : dict[tuple, pd.DataFrame]
#         Dict of segment DataFrames, as returned by process_segments_mean.
#     parent_df : pd.DataFrame
#         DataFrame with multipliers and columns for each segment (index must match arrays_dict keys).
#     mapping_df : pd.DataFrame
#         DataFrame mapping standardized columns to raw columns.
#     trim_stable_voltage : bool, optional
#         If True, trims each segment to the stable voltage region based on threshold.
#     threshold : float, optional
#         Threshold for deviation from mean voltage to trim (default 0.5).
#     reset_time_zero : bool, optional
#         If True and trimming occurs, resets "time" to start at 0.

#     Returns
#     -------
#     standardized_dict : dict[tuple, pd.DataFrame]
#         Dict of standardized segment DataFrames, ready for fitting.
#     """
#     standardized_dict = {}

#     for idx, seg_df in arrays_dict.items():
#         # Create a new DataFrame with the same columns as parent_df, filled with np.nan
#         std_df = pd.DataFrame(index=seg_df.index, columns=parent_df.columns, dtype=float)
#         # Copy multipliers and t_multipliers from parent_df
#         for col in ["multiplier", "t_multiplier"]:
#             if col in parent_df.columns:
#                 std_df[col] = parent_df.loc[idx, col]
#         # Fill raw columns using mapping_df and seg_df
#         for col in mapping_df.columns:
#             mapping_val = mapping_df.loc[idx, col]
#             if mapping_val == "N/A" or mapping_val not in seg_df.columns:
#                 std_df[col] = np.nan
#             else:
#                 std_df[col] = seg_df[mapping_val]
#         # Fill calculated columns using your existing logic
#         clear_nonraw_columns(std_df, float(idx[2]))
#         convert_raw_data(std_df)

#         std_df.insert(0, "time", seg_df["time"].copy())

#         standardized_dict[idx] = std_df.iloc[:, :17]

#     return standardized_dict


# def apply_col_mapping(
#     arrays_dict: dict[tuple, pd.DataFrame],
#     parent_df: pd.DataFrame,
#     mapping_df: pd.DataFrame,
#     trim_stable_voltage: bool = False,
#     threshold: float = 0.5,
#     shift_start: int = 0,
#     reset_time_zero: bool = True,
# ) -> dict[tuple, pd.DataFrame]:
#     """
#     Build a dict of standardized segment DataFrames from arrays_dict, applying mapping and multipliers,
#     and running convert_raw_data on each segment. Optionally trims each segment to the stable voltage region.

#     Parameters
#     ----------
#     arrays_dict : dict[tuple, pd.DataFrame]
#         Dict of segment DataFrames, as returned by process_segments_mean.
#     parent_df : pd.DataFrame
#         DataFrame with multipliers and columns for each segment (index must match arrays_dict keys).
#     mapping_df : pd.DataFrame
#         DataFrame mapping standardized columns to raw columns.
#     trim_stable_voltage : bool, optional
#         If True, trims each segment to the stable voltage region based on threshold.
#     threshold : float, optional
#         Threshold for deviation from mean voltage to trim (default 0.5).
#     reset_time_zero : bool, optional
#         If True and trimming occurs, resets "time" to start at 0.

#     Returns
#     -------
#     standardized_dict : dict[tuple, pd.DataFrame]
#         Dict of standardized segment DataFrames, ready for fitting.
#     """
#     standardized_dict = {}

#     for idx, seg_df in arrays_dict.items():
#         # Create a new DataFrame with the same columns as parent_df, filled with np.nan
#         std_df = pd.DataFrame(index=seg_df.index, columns=parent_df.columns, dtype=float)
#         # Copy multipliers and t_multipliers from parent_df
#         for col in ["multiplier", "t_multiplier"]:
#             if col in parent_df.columns:
#                 std_df[col] = parent_df.loc[idx, col]
#         # Fill raw columns using mapping_df and seg_df
#         for col in mapping_df.columns:
#             mapping_val = mapping_df.loc[idx, col]
#             if mapping_val == "N/A" or mapping_val not in seg_df.columns:
#                 std_df[col] = np.nan
#             else:
#                 std_df[col] = seg_df[mapping_val]
#         # Fill calculated columns using your existing logic
#         clear_nonraw_columns(std_df, float(idx[2]))
#         convert_raw_data(std_df)

#         std_df.insert(0, "time", seg_df["time"].copy())

#         # --- Optional: Trim to stable voltage region (after mapping) ---
#         if trim_stable_voltage:
#             if len(std_df) < 2:
#                 continue

#             voltage = std_df["Voltage"].to_numpy(copy=True)
#             v_mean = np.nanmean(voltage)
#             start = 0
#             end = len(voltage)
#             while start < end and abs(voltage[start] - v_mean) > threshold:
#                 start += 1

#             start += shift_start
#             start = min(max(start, 0), len(voltage) - 1)

#             while end > start and abs(voltage[end - 1] - v_mean) > threshold:
#                 end -= 1

#             std_df = std_df.iloc[start:end].copy()
#             if reset_time_zero and "time" in std_df.columns and not std_df.empty:
#                 std_df["time"] = std_df["time"] - std_df["time"].iloc[0]

#         standardized_dict[idx] = std_df.iloc[:, :17]

#     return standardized_dict


# def build_segmented_std_df_from_parsed(
#     raw_df: pd.DataFrame,
#     parent_df: pd.DataFrame,
#     mapping_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Build a DataFrame with the same index and columns as parent_df, using parent_df's columns and multipliers,
#     and mapping_df to direct the segments to the right raw columns, then fill calculated columns using convert_raw_data.

#     Parameters
#     ----------
#     raw_df : pd.DataFrame
#         DataFrame with segmented arrays for each raw column (same shape as results_df).
#     parent_df : pd.DataFrame
#         DataFrame with multipliers and columns for each segment.
#     mapping_df : pd.DataFrame
#         DataFrame mapping standardized columns to raw columns.

#     Returns
#     -------
#     new_df : pd.DataFrame
#         DataFrame with the same columns as parent_df, filled with segment arrays and calculated columns.
#     """
#     new_df = pd.DataFrame(index=parent_df.index, columns=parent_df.columns, dtype=object)

#     # Copy multipliers and t_multipliers
#     for col in ["multiplier", "t_multiplier"]:
#         if col in parent_df.columns:
#             new_df[col] = parent_df[col]

#     # Fill raw columns using mapping_df and raw_df, preserving grouping/index
#     for col in mapping_df.columns:
#         # Group by mapping value to avoid dropping group structure
#         for mapping_val, block_idx in mapping_df.groupby(col).groups.items():
#             if mapping_val == "N/A":  # or mapping_val not in raw_df.columns:
#                 new_df.loc[block_idx, col] = np.nan
#             else:
#                 new_df.loc[block_idx, col] = raw_df.loc[block_idx, mapping_val]  # type: ignore

#     # Fill calculated columns using your existing logic
#     clear_nonraw_columns(new_df, raw_df["time"].map(len).map(np.ones))
#     convert_raw_data(new_df, True)

#     return new_df.iloc[:, :16]

# def convert_raw_data(data_df: pd.DataFrame, arrays: bool = False) -> None:
#     """
#     Fill standard (non-raw) columns in data_df from raw columns and Ohm's law.

#     This function updates data_df in-place. For each  column, it first tries to fill
#     from the corresponding raw column. If not available, it attempts to calculate the value
#     using Ohm's law (V = I * R) where possible.

#     Parameters
#     ----------
#     data_df : pd.DataFrame
#         DataFrame with  and raw columns. Modified in-place.
#     arrays : bool, optional
#         If True, handle array-valued cells. Default is False.
#     """

#     def apply_c_multiplier(key: str) -> None:
#         """
#         Apply the multiplier to the raw column if it exists.
#         """
#         std_key = data_df[key].apply(np.mean) if arrays else data_df[key]
#         mask = (data_df["multiplier"] != 1) & std_key.notna() & (std_key.abs() > 1e-8)
#         data_df.loc[mask, key] = data_df.loc[mask, key] * data_df.loc[mask, "multiplier"]

#     data_df["Resistance"] = data_df["raw_Resistance"]

#     # Get sign from available raw columns (should be the same for all)
#     raw_sign = (
#         data_df["raw_Voltage"]
#         .combine_first(data_df["raw_Voltage rms"])
#         .combine_first(data_df["raw_Current"])
#         .combine_first(data_df["raw_Current rms"])
#         .fillna(1.0)  # Default to 1.0 if all are NaN
#     )
#     if arrays:
#         sign = pd.Series(raw_sign.apply(np.mean).apply(np.sign))
#     else:
#         sign = pd.Series(np.sign(raw_sign))
#     # for col in ["Voltage", "Current"]:
#     data_df["Voltage"] = (
#         data_df["raw_Voltage"]
#         .combine_first(data_df["raw_Voltage abs"] * sign)
#         .combine_first(data_df["raw_Voltage rms"] / np.sqrt(2))
#         .combine_first(data_df["Voltage"] * sign)
#         # .combine_first(2 * sign)
#     )

#     data_df["Current"] = (
#         data_df["raw_Current"]
#         .combine_first(data_df["raw_Current abs"] * sign)
#         .combine_first(data_df["raw_Current rms"] / np.sqrt(2))
#         .combine_first(data_df["Voltage"] / data_df["Resistance"])
#     )

#     apply_c_multiplier("Current")

#     data_df["Resistance"] = data_df["Resistance"].combine_first(
#         data_df["Voltage"] / data_df["Current"]
#     )
#     data_df["Temperature"] = (
#         (data_df["raw_Temperature"] * data_df["t_multiplier"]).abs()
#     ).combine_first(data_df["Temperature"])

#     if arrays:
#         data_df["Temperature"] = [
#             tc.convert(t_val, ambient=30.0) if t_val[0] < 10 else t_val
#             for t_val in data_df["Temperature"]
#         ]
#     else:
#         data_df["Temperature"] = [
#             tc.convert(float(t_val), ambient=30.0) if t_val < 10 else t_val
#             for t_val in data_df["Temperature"]
#         ]

#     data_df["Voltage rms"] = data_df["raw_Voltage rms"].combine_first(
#         data_df["Voltage"] * np.sqrt(2)
#     )
#     data_df["Current rms"] = data_df["raw_Current rms"].combine_first(
#         data_df["Current"] * np.sqrt(2)
#     )
#     data_df["Voltage abs"] = data_df["raw_Voltage abs"].combine_first(data_df["Voltage"].abs())
#     data_df["Current abs"] = data_df["raw_Current abs"].combine_first(data_df["Current"].abs())

#     apply_c_multiplier("Current rms")
#     apply_c_multiplier("Current abs")

# def convert_raw_data_old(data_df: pd.DataFrame) -> None:
#     """
#     Fill standard (non-raw) columns in data_df from raw columns and Ohm's law.

#     This function updates data_df in-place. For each  column, it first tries to fill
#     from the corresponding raw column. If not available, it attempts to calculate the value
#     using Ohm's law (V = I * R) where possible.

#     Expected columns:
#         - Voltage, Current, Resistance, Temperature, Voltage rms, Current rms, Voltage abs, Current abs,
#         raw_Voltage, raw_Current, raw_Resistance, raw_Temperature, raw_Voltage rms, raw_Current rms,
#         raw_Voltage abs, raw_Current abs, multiplier, t_multiplier


#     Parameters
#     ----------
#     data_df : pd.DataFrame
#         DataFrame with  and raw columns. Modified in-place.
#     """

#     def apply_c_multiplier(key: str) -> None:
#         """Apply the multiplier to the raw column if it exists."""
#         std_key = data_df[key]
#         mask = (data_df["multiplier"] != 1) & std_key.notna() & (std_key.abs() > 1e-8)
#         data_df.loc[mask, key] = data_df.loc[mask, key] * data_df.loc[mask, "multiplier"]

#     data_df["Resistance"] = data_df["raw_Resistance"]

#     # Get sign from available raw columns (should be the same for all)
#     raw_sign = (
#         data_df["raw_Voltage"]
#         .combine_first(data_df["raw_Voltage rms"])
#         .combine_first(data_df["raw_Current"])
#         .combine_first(data_df["raw_Current rms"])
#         .fillna(1.0)  # Default to 1.0 if all are NaN
#     )
#     sign = pd.Series(np.sign(raw_sign))

#     data_df["Voltage"] = (
#         data_df["raw_Voltage"]
#         .combine_first(data_df["raw_Voltage abs"] * sign)
#         .combine_first(data_df["raw_Voltage rms"] / np.sqrt(2))
#         .combine_first(data_df["Voltage"] * sign)
#     )

#     data_df["Current"] = (
#         data_df["raw_Current"]
#         .combine_first(data_df["raw_Current abs"] * sign)
#         .combine_first(data_df["raw_Current rms"] / np.sqrt(2))
#         .combine_first(data_df["Voltage"] / data_df["Resistance"])
#     )

#     apply_c_multiplier("Current")

#     data_df["Resistance"] = data_df["Resistance"].combine_first(
#         data_df["Voltage"] / data_df["Current"]
#     )
#     data_df["Temperature"] = (
#         (data_df["raw_Temperature"] * data_df["t_multiplier"]).abs()
#     ).combine_first(data_df["Temperature"])

#     data_df["Temperature"] = [
#         tc.convert(float(t_val), ambient=30.0) if t_val < 10 else t_val
#         for t_val in data_df["Temperature"]
#     ]

#     data_df["Voltage rms"] = data_df["raw_Voltage rms"].combine_first(
#         data_df["Voltage"] * np.sqrt(2)
#     )
#     data_df["Current rms"] = data_df["raw_Current rms"].combine_first(
#         data_df["Current"] * np.sqrt(2)
#     )
#     data_df["Voltage abs"] = data_df["raw_Voltage abs"].combine_first(data_df["Voltage"].abs())
#     data_df["Current abs"] = data_df["raw_Current abs"].combine_first(data_df["Current"].abs())

#     apply_c_multiplier("Current rms")
#     apply_c_multiplier("Current abs")

# def convert_raw_data(data_df: pd.DataFrame) -> None:
#     """
#     Fill standard (non-raw) columns in data_df from raw columns and Ohm's law.

#     This function updates data_df in-place. For each  column, it first tries to fill
#     from the corresponding raw column. If not available, it attempts to calculate the value
#     using Ohm's law (V = I * R) where possible.

#     Expected columns:
#         - Voltage, Current, Resistance, Temperature, Voltage rms, Current rms, Voltage abs, Current abs,
#         raw_Voltage, raw_Current, raw_Resistance, raw_Temperature, raw_Voltage rms, raw_Current rms,
#         raw_Voltage abs, raw_Current abs, multiplier, t_multiplier

#     Parameters
#     ----------
#     data_df : pd.DataFrame
#         DataFrame with  and raw columns. Modified in-place.
#     """

#     # Pre-extract all relevant columns as arrays to minimize repeated indexing
#     raw_voltage = data_df["raw_Voltage"].to_numpy(copy=True)
#     raw_voltage_rms = data_df["raw_Voltage rms"].to_numpy(copy=True)
#     raw_voltage_abs = data_df["raw_Voltage abs"].to_numpy(copy=True)
#     raw_current = data_df["raw_Current"].to_numpy(copy=True)
#     raw_current_rms = data_df["raw_Current rms"].to_numpy(copy=True)
#     raw_current_abs = data_df["raw_Current abs"].to_numpy(copy=True)
#     raw_resistance = data_df["raw_Resistance"].to_numpy(copy=True)
#     raw_temperature = data_df["raw_Temperature"].to_numpy(copy=True)
#     multiplier = data_df["multiplier"].to_numpy(copy=True)
#     t_multiplier = data_df["t_multiplier"].to_numpy(copy=True)
#     voltage_col = data_df["Voltage"].to_numpy(copy=True)
#     resistance_col = data_df["Resistance"].to_numpy(copy=True)
#     # current_col = data_df["Current"].to_numpy(copy=True)
#     temperature_col = data_df["Temperature"].to_numpy(copy=True)

#     # Precompute sign
#     raw_sign = raw_voltage.copy()
#     mask = np.isnan(raw_sign)
#     if np.any(mask):
#         raw_sign[mask] = raw_voltage_rms[mask]
#         mask = np.isnan(raw_sign)
#     if np.any(mask):
#         raw_sign[mask] = raw_current[mask]
#         mask = np.isnan(raw_sign)
#     if np.any(mask):
#         raw_sign[mask] = raw_current_rms[mask]
#         mask = np.isnan(raw_sign)
#     raw_sign[mask] = 1.0
#     sign = np.sign(raw_sign)

#     # Voltage
#     voltage = raw_voltage.copy()
#     mask = np.isnan(voltage)
#     if np.any(mask):
#         voltage[mask] = raw_voltage_abs[mask] * sign[mask]
#         mask = np.isnan(voltage)
#     if np.any(mask):
#         voltage[mask] = raw_voltage_rms[mask] / np.sqrt(2)
#         mask = np.isnan(voltage)
#     if np.any(mask):
#         voltage[mask] = voltage_col[mask] * sign[mask]
#     data_df["Voltage"] = voltage

#     # Current
#     current = raw_current.copy()
#     mask = np.isnan(current)
#     if np.any(mask):
#         current[mask] = raw_current_abs[mask] * sign[mask]
#         mask = np.isnan(current)
#     if np.any(mask):
#         current[mask] = raw_current_rms[mask] / np.sqrt(2)
#         mask = np.isnan(current)
#     if np.any(mask):
#         # Use updated voltage and resistance for calculation
#         current[mask] = voltage[mask] / resistance_col[mask]
#     data_df["Current"] = current

#     def apply_c_multiplier(key: str, arr: np.ndarray) -> None:
#         """Apply the multiplier to the raw column if it exists."""
#         mask = (multiplier != 1) & ~np.isnan(arr) & (np.abs(arr) > 1e-8)
#         arr[mask] = arr[mask] * multiplier[mask]
#         data_df[key] = arr

#     apply_c_multiplier("Current", current)

#     # Resistance
#     resistance = raw_resistance.copy()
#     mask = np.isnan(resistance)
#     if np.any(mask):
#         resistance[mask] = voltage[mask] / current[mask]
#     data_df["Resistance"] = resistance

#     # Temperature
#     temp = np.abs(raw_temperature * t_multiplier)
#     mask = np.isnan(temp)
#     if np.any(mask):
#         temp[mask] = temperature_col[mask]
#     temp = np.array(
#         [tc.convert(float(t_val), ambient=30.0) if t_val < 10 else t_val for t_val in temp]
#     )
#     data_df["Temperature"] = temp

#     # Voltage rms
#     voltage_rms = raw_voltage_rms.copy()
#     mask = np.isnan(voltage_rms)
#     if np.any(mask):
#         voltage_rms[mask] = voltage[mask] * np.sqrt(2)
#     data_df["Voltage rms"] = voltage_rms

#     # Current rms
#     current_rms = raw_current_rms.copy()
#     mask = np.isnan(current_rms)
#     if np.any(mask):
#         current_rms[mask] = current[mask] * np.sqrt(2)
#     data_df["Current rms"] = current_rms

#     # Voltage abs
#     voltage_abs = raw_voltage_abs.copy()
#     mask = np.isnan(voltage_abs)
#     if np.any(mask):
#         voltage_abs[mask] = np.abs(voltage[mask])
#     data_df["Voltage abs"] = voltage_abs

#     # Current abs
#     current_abs = raw_current_abs.copy()
#     mask = np.isnan(current_abs)
#     if np.any(mask):
#         current_abs[mask] = np.abs(current[mask])
#     data_df["Current abs"] = current_abs

#     apply_c_multiplier("Current rms", current_rms)
#     apply_c_multiplier("Current abs", current_abs)
