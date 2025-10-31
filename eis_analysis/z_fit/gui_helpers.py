import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

from eis_analysis.data_treatment import calculate_rc_freq


def get_param_df(*args, include_locks=True, include_bounds=True):
    """
    Construct a DataFrame from parameters, stds, and bounds.

    Parameters:
    self: The self object containing parameters, stds, and bounds.

    Returns:
    pd.DataFrame: DataFrame containing the parameter values, stds, and bounds.
    """
    # Automated part using the main class
    if len(args) == 1 and hasattr(args[0], "parameters"):
        parameters = args[0].parameters
        stds = args[0].parameters_std
        bounds = args[0].bounds
    else:
        parameters, stds, bounds = (args + (None,) * 3)[:3]
    if parameters is None:
        return pd.DataFrame()

    # Primary section independent of the main class
    # Initialize lists to store the data
    param_names = []
    values = []
    std_values = []
    locked = []
    bnd_lows = []
    bnd_highs = []
    bnd_low_locked = []
    bnd_high_locked = []

    # Iterate over the parameters to populate the lists
    for param in parameters:
        param_names.append(param.name)
        values.append(float(param.values[0]))
        if stds is not None:
            std_values.append(float(stds[param.name].values[0]))
        locked.append(bool(param.is_checked[0]))
        if bounds is not None:
            bnd_lows.append(float(bounds[param.name].values[0]))
            bnd_highs.append(float(bounds[param.name].values[1]))
            bnd_low_locked.append(bool(bounds[param.name].is_checked[0]))
            bnd_high_locked.append(bool(bounds[param.name].is_checked[1]))

    # Create the default Parameter DataFrame
    param_df = pd.DataFrame(
        {
            "value": values,
            "std": std_values,
            "lock": locked,
            "bnd_low": bnd_lows,
            "bnd_high": bnd_highs,
            "bnd_low_lock": bnd_low_locked,
            "bnd_high_lock": bnd_high_locked,
        },
        index=param_names,
    )

    if not include_locks and not include_bounds:
        param_df = param_df.drop(
            columns=["lock", "bnd_low", "bnd_high", "bnd_low_lock", "bnd_high_lock"]
        )
    elif not include_locks:
        param_df = param_df.drop(columns=["lock", "bnd_low_lock", "bnd_high_lock"])
    elif not include_bounds:
        param_df = param_df.drop(columns=["bnd_low", "bnd_high", "bnd_low_lock", "bnd_high_lock"])

    return param_df


# def construct_param_df(
#     self, df, ref_df=None, bound_range=None, include_locks=True, include_bounds=True
# ):
def construct_param_df(df, ref_df, bound_range=None, include_locks=True, include_bounds=True):
    """
    Construct a DataFrame similar to the one created by get_param_df.

    Parameters:
    self: The self object containing the initial DataFrame structure.
    df (pd.DataFrame): DataFrame containing the parameter values.

    Returns:
    pd.DataFrame: DataFrame containing the parameter values, stds, and bounds.
    """
    # # Automated part using the main class
    # if ref_df is None:
    #     init_df = get_param_df(self)
    # else:
    #     init_df = ref_df.copy()
    # if bound_range is None:
    #     bound_range = np.array((self.quick_bound_vals["low"], self.quick_bound_vals["high"]))
    df = df.copy()
    ref_df = ref_df.copy()

    if bound_range is None:
        bound_range = np.array([0.1, 10])

    # Primary section independent of the main class
    # Importantly only adds to given list if col in ref_df
    base_cols = ["value", "std"]
    lock_cols = []
    bound_lock_cols = []
    bound_val_cols = []
    low_cols = []
    high_cols = []
    for col in ref_df.columns:
        if "lock" in col:
            lock_cols.append(col)
        elif "bnd" in col or "bound" in col:
            # bound_cols.append(col)
            if "lock" in col:
                bound_lock_cols.append(col)
            else:
                bound_val_cols.append(col)
                if "low" in col:
                    low_cols.append(col)
                elif "high" in col:
                    high_cols.append(col)

    with pd.option_context("future.no_silent_downcasting", True):
        # Ensure the DataFrame has the required columns
        df = df.reindex(columns=ref_df.columns, fill_value=pd.NA)

        # Update df with values from init_df where indexes match
        if any(ind in ref_df.index for ind in df.index):
            df.update(ref_df, overwrite=False)

        # Fill missing lock columns with False
        if lock_cols:
            df[lock_cols] = df[lock_cols].fillna(False)
            df[lock_cols] = df[lock_cols].astype(bool)

        df[base_cols] = df[base_cols].astype(float)
        if bound_val_cols:
            df[bound_val_cols] = df[base_cols + bound_val_cols].astype(float)

        if low_cols and high_cols:
            df[low_cols] = df[low_cols].fillna((bound_range[0] * df["value"]).astype(float))
            df[high_cols] = df[high_cols].fillna((bound_range[1] * df["value"]).astype(float))

    cols_to_drop = []
    if not include_locks and not include_bounds:
        cols_to_drop = lock_cols + bound_val_cols
    elif not include_locks:
        cols_to_drop = lock_cols
    elif not include_bounds:
        cols_to_drop = bound_val_cols + bound_lock_cols

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def pinned_to_param(df):
    """
    Convert pinned data to a dictionary of parameter DataFrames.

    Parameters:
    self: The self object containing the initial DataFrame structure.
    df (pd.DataFrame): DataFrame containing the pinned data.

    Returns:
    dict: Dictionary of DataFrames with parameter values and stds, keyed by the 'Name' column.
    """
    # Automated part using the main class
    if not isinstance(df, pd.DataFrame):
        if hasattr(df, "pinned"):
            df = df.pinned.df.copy()
        else:
            return {}

    # Primary section independent of the main class
    param_dfs = {}
    for _, row in df.iterrows():
        param_df = pin_to_param(row)
        param_dfs[row["Name"]] = param_df

    return param_dfs


def pin_to_param(row):
    """
    Convert a single pinned row to a parameter DataFrame.

    Parameters:
    row (pd.Series): A single row from the pinned DataFrame.

    Returns:
    pd.DataFrame: DataFrame with parameter values and stds.
    """
    param_names = [col[:-7] for col in row.index if col.endswith("_values")]
    param_values = [row[f"{name}_values"] for name in param_names]
    param_stds = [row[f"{name}_std"] for name in param_names]

    param_df = pd.DataFrame({"value": param_values, "std": param_stds}, index=param_names)

    for col in ["Name", "Dataset", "Model", "Comments"]:
        param_df.attrs[col] = row[col]

    return param_df


def param_to_pin(self, param_dfs=None, dataset="", model="", comments=""):
    """
    Convert parameter default DataFrames to pinned data format.

    Parameters:
    self: The self object containing the initial DataFrame structure.
    param_dfs (dict or pd.DataFrame): Dictionary of DataFrames or a single DataFrame with parameter values and stds.

    Returns:
    pd.DataFrame: DataFrame containing the pinned data.
    """
    # Automated part using the main class
    if param_dfs is None:
        param_dfs = get_param_df(self)
    dataset = dataset or self.data.var.currentText()
    model = model or self.settings.model
    # if param_dfs is None:
    #     param_dfs = get_param_df(self)

    # Primary section independent of the main class
    if isinstance(param_dfs, pd.DataFrame):
        name = param_dfs.attrs.get("Name", self.data.var.currentText() + "_pin")
        n = 0
        while name in self.pinned.df["Name"].values:
            n += 1
        name = f"{name}_{n}"
        param_dfs = {name: param_dfs}

    pin_data = []
    for name, param_df in param_dfs.items():
        result_row = {
            "Name": name,
            "Dataset": param_df.attrs.get("Dataset", dataset),
            "Model": param_df.attrs.get("Model", model),
            "Show": "",
            "Comments": param_df.attrs.get("Comments", comments),
            **{f"{param}_values": param_df.loc[param, "value"] for param in param_df.index},
            **{f"{param}_std": param_df.loc[param, "std"] for param in param_df.index},
        }
        pin_data.append(result_row)

    pin_df = pd.DataFrame(pin_data)

    return pin_df


def space(value, mult=1e3):
    """Return the space of a floating-point number."""
    return np.spacing(value) * mult


def validate_vals_and_band(
    df,
    bound_range: np.ndarray | None = None,
    rebound_unlocked: bool = False,
    shift_band: bool = False,
    shift_value: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Ensure that the bands encompass a given value or adjust the value to fit within the bands.

    Parameters:
    df (pd.DataFrame): DataFrame containing the parameter values and bounds.
    bound_range (np.ndarray): Array with two elements specifying the relative bounds (e.g., [0.1, 10]).
    rebound_unlocked (bool): If True, adjust unlocked bounds based on the value and bound_range.
    shift_band (bool): If True, adjust the locked bounds to fit the value.
    shift_value (bool): If True, adjust the value to fit within the locked bounds.

    Returns:
    pd.DataFrame: DataFrame with adjusted bounds or values.

    Raises:
    ValueError: If neither `shift_band` nor `shift_value` is True and values are out of bounds.
    """
    # if bound_range is None:
    #     bound_range = np.array((self.quick_bound_vals["low"], self.quick_bound_vals["high"]))
    if bound_range is None:
        bound_range = np.array([0.1, 10])

    df = df.copy()

    max_exp = np.floor(np.log10(np.max(np.finfo(np.float64).max))) / 4

    df["bnd_low"] = (
        df["bnd_low"]
        .fillna((bound_range[0] * df["value"]).astype(float))
        .fillna(kwargs.get("default_low", 10**-max_exp))
    )
    df["bnd_high"] = (
        df["bnd_high"]
        .fillna((bound_range[1] * df["value"]).astype(float))
        .fillna(kwargs.get("default_high", 10**max_exp))
    )

    if rebound_unlocked:
        # Rebound unlocked bounds (not bnd_low_lock/bnd_high_lock) using the bound_range and value
        df.loc[~df["bnd_low_lock"], "bnd_low"] = (bound_range[0] * df["value"]).astype(float)
        df.loc[~df["bnd_high_lock"], "bnd_high"] = (bound_range[1] * df["value"]).astype(float)

    # Replace problematic values in bounds
    # Replace inf with a large number and 0 with a small positive number
    df["bnd_high"] = df["bnd_high"].replace(np.inf, 10**max_exp)
    df["bnd_low"] = df["bnd_low"].replace(-np.inf, -(10**max_exp))
    df["bnd_low"] = df["bnd_low"].replace(0, 10**-max_exp)

    # Track rows where values are out of bounds
    out_of_bounds = df[(df["value"] <= df["bnd_low"]) | (df["value"] >= df["bnd_high"])]

    if not out_of_bounds.empty:
        if shift_band:
            # Adjust bounds to fit the values, with a small perturbation
            df["bnd_low"] = df.apply(
                lambda row: float(
                    min(row["bnd_low"], row["value"]) - space(min(row["bnd_low"], row["value"]))
                ),
                axis=1,
            )
            df["bnd_high"] = df.apply(
                lambda row: float(
                    max(row["bnd_high"], row["value"]) + space(max(row["bnd_high"], row["value"]))
                ),
                axis=1,
            )
        elif shift_value:
            # Adjust values to fit within the bounds
            df["value"] = df.apply(
                lambda row: float(
                    np.clip(
                        row["value"],
                        row["bnd_low"] + space(row["bnd_low"]),
                        row["bnd_high"] - space(row["bnd_high"]),
                    )
                ),
                axis=1,
            )
        else:
            # Raise an error with details of out-of-bounds values
            error_message = "The following values are out of bounds:\n" + out_of_bounds.to_string(
                columns=["value", "bnd_low", "bnd_high"], index=False
            )
            raise ValueError(error_message)

    df["value"] = df["value"].fillna(df[["bnd_low", "bnd_high"]].mean(axis=1))
    df["std"] = df["std"].fillna(df["value"] * 0.1)

    return df


def calculate_rc_freq_sets(self, param_df=None, return_type="dict"):
    """
    Calculate RC pairs from a parameter DataFrame.

    Parameters:
    self: The self object containing the initial DataFrame structure.
    param_df (pd.DataFrame): DataFrame containing the parameter values.

    Returns:
    dict or list: Dictionary or list of RC pairs.
    """
    # Automated part using the main class
    if param_df is None:
        param_df = get_param_df(self)

    # Primary section independent of the main class
    param_names = param_df.index.tolist()
    values = param_df["value"].tolist()

    res = {}
    for i in range(len(param_names) - 1):
        if param_names[i].startswith("R") and param_names[i + 1].startswith("C"):
            rc_name = f"{param_names[i]}_{param_names[i + 1]}"
            res[rc_name] = calculate_rc_freq(values[i], values[i + 1])

    if return_type == "str":
        return [f"{key}: {val:.3e}" for key, val in res.items()]

    return res


# def convert_circuit(self, subset_idxs=(0, 1)):
#     data = get_param_df(self)
#     sub_models = self.model_entry.sub_models


def quick_rc(self, df=None, result_type="dict"):
    """
    Quickly calculate the RC values for a dataset.

    Parameters:
    self: The self object containing the initial DataFrame structure.
    dataset (pd.DataFrame): The dataset to calculate the RC values for.

    Returns:
    list: A list of dictionaries or strings containing the RC values for each row in the dataset.
    """
    # Get dict of Param Dataframes
    param_dfs = pinned_to_param(df)

    reslist = []
    for name, param_df in param_dfs.items():
        try:
            res = calculate_rc_freq_sets(self, param_df, result_type)
            reslist.append(res)
        except (IndexError, ValueError) as e:
            print(f"Error processing {name}: {e}")
            continue

    return reslist


def sort_parameters(param_df, constraints=None, pairings=None, value_column=None, columns=None):
    """
    Sort parameters based on constraints and pairings/groupings.

    Parameters:
    param_df (pd.DataFrame): DataFrame containing the parameter values.
    constraints (list): List of constraints in the format ["R2 > R1", "C2 < C1"].
    pairings (list): List of pairings/groupings in the format [["R2", "C2"], ["R1", "C1"]].
    columns (list): List of columns to include in the sorting of the DataFrame.

    Returns:
    pd.DataFrame: Sorted DataFrame with parameters.
    """

    def swap_rows(df, idx1, idx2):
        df.loc[[idx1, idx2]] = df.loc[[idx2, idx1]].values

    def get_constraints(param_df):
        constraints = []

        # Generate constraints for R parameters
        r_params = sorted(
            [param for param in param_df.index if re.match(r"R_?\d+", param)],
            key=lambda x: int(re.match(r"R_?(\d+)", x).group(1)),  # type: ignore
        )
        for i in range(len(r_params) - 1):
            constraints.append(f"{r_params[i + 1]} > {r_params[i]}")

        return constraints

    def get_pairings(param_df):
        pairings = []

        # Extract parameter names and group them by their numeric suffix
        param_groups = {}
        for param in param_df.index:
            match = re.match(r"([a-zA-Z]+_?)(\d+)", param)
            if match:
                _, num = match.groups()
                num = int(num)
                if num not in param_groups:
                    param_groups[num] = []
                param_groups[num].append(param)

        # Generate pairings for parameters with the same numeric suffix
        for group in param_groups.values():
            pairings.append(group)

        return pairings

    param_df = param_df.copy()

    # def_constraints = None
    # def_pairings = None
    if constraints is None:
        constraints = get_constraints(param_df)
    if pairings is None:
        pairings = get_pairings(param_df)

    init_df = param_df.copy()
    if columns is not None:
        param_df = param_df[columns]

    if value_column is None:
        value_column = param_df.columns[0]
    for constraint in constraints:
        # Create a dictionary of the DataFrame values for eval
        values_dict = param_df[value_column].to_dict()

        # Evaluate the constraint
        if not eval(constraint, {}, values_dict):
            params = [param for param in param_df.index if param in constraint]
            if len(params) < 2:
                continue
            # Find the pairings for the two parameters
            pairing1 = next((pair for pair in pairings if params[0] in pair), None)
            pairing2 = next((pair for pair in pairings if params[1] in pair), None)

            if pairing1 and pairing2:
                # Swap the values for each parameter in the pairings
                for param1 in pairing1:
                    # Find the counterpart in the other pairing with the same non-numerical part
                    if base_match := re.match(r"([a-zA-Z]+_?)", param1):
                        base_name = base_match.group(1)
                        param2 = next((p for p in pairing2 if p.startswith(base_name)), None)

                        if param2:
                            swap_rows(param_df, param1, param2)

    if columns is not None:
        init_df.update(param_df, overwrite=True)
        return init_df

    return param_df


def find_arc_minima(data: Any):
    """
    Find the minima of the imaginary part of the impedance data.

    self: The self object containing the initial DataFrame structure.
    data (pd.DataFrame): DataFrame containing the impedance data.

    Returns:
    tuple: tuple containing the real part of the first and last minima.

    """
    # Automated part using the main class
    # data = data.sort_values(by=data.columns[0])
    # if data is None:
    #     data = self.data.raw[self.data.primary()].get_df("Z")
    if not isinstance(data, pd.DataFrame):
        if hasattr(data, "data") and hasattr(data.data, "raw"):
            data = data.data.raw[data.data.primary()].get_df("Z")
        elif hasattr(data, "raw"):
            data = data.raw[data.primary()].get_df("Z")
        else:
            return [np.nan, np.nan]

    # Primary section independent of the main class
    imag_vals = -1 * data.iloc[:, 1].to_numpy(copy=True)

    # Split the data into two halves
    mid_index = np.argmax(imag_vals)
    first_half = imag_vals[:mid_index]
    second_half = imag_vals[mid_index:]

    # Find the minimum of each half
    first_minima = int(np.argmin(first_half))
    last_minima = int(np.argmin(second_half))

    res = (data.iloc[first_minima, 0].item(), data.iloc[last_minima + mid_index, 0].item())  # type: ignore
    return sorted(res)  # type: ignore


def find_x_intercepts(data):
    """
    Find the x-intercepts of the Nyquist plot using PchipInterpolator.

    Parameters:
    data (pd.DataFrame): DataFrame with real values in the first column and imaginary values in the second column.

    Returns:
    list: List of x-intercepts [first_intercept, second_intercept].
    """

    # if data is None:
    #     data = self.data.raw[self.data.primary()].get_df("Z")
    if not isinstance(data, pd.DataFrame):
        if hasattr(data, "data") and hasattr(data.data, "raw"):
            data = data.data.raw[data.primary()].get_df("Z")
        elif hasattr(data, "raw"):
            data = data.raw[data.primary()].get_df("Z")
        else:
            return [np.nan, np.nan]

    # Step 1: Find the approximate x-intercepts using find_arc_minima
    first_intercept, second_approx = find_arc_minima(data)

    # Extract real and imaginary values from the data
    real_vals = data.iloc[:, 0].to_numpy(copy=True)
    imag_vals = -data.iloc[:, 1].to_numpy(
        copy=True
    )  # Invert imaginary values as they are negative

    # Apply Savitzky-Golay filter to smooth the imaginary values
    imag_vals = savgol_filter(imag_vals, len(real_vals) // 10, 3)

    # Step 2: First interpolation using PchipInterpolator
    new_real_vals = np.linspace(real_vals.min(), real_vals.max(), len(real_vals) // 2)
    pchip_interpolator = PchipInterpolator(real_vals, imag_vals, extrapolate=True)
    first_interpolated_imag_vals = pchip_interpolator(new_real_vals)

    # Step 3: Second interpolation on the interpolated dataset
    second_interpolator = PchipInterpolator(
        new_real_vals, first_interpolated_imag_vals, extrapolate=True
    )

    # Step 4: Find the roots of the second interpolator
    roots = second_interpolator.roots()

    # Step 5: Select the root closest to the second approximate x-intercept
    second_intercept = roots[np.argmin(np.abs(roots - second_approx))]

    # if second_intercept not within 10 percent of second_approx, use second_approx
    if abs(second_intercept - second_approx) > 0.1 * second_approx:
        second_intercept = second_approx

    # Return the first intercept (from find_arc_minima) and the refined second intercept
    return [first_intercept, second_intercept]


tools = {
    "calculate_rc_freq": calculate_rc_freq,
    "get_param_df": get_param_df,
    "construct_param_df": construct_param_df,
    "validate_vals_and_band": validate_vals_and_band,
    "pinned_to_param": pinned_to_param,
    "pin_to_param": pin_to_param,
    "param_to_pin": param_to_pin,
    "calculate_rc_freq_sets": calculate_rc_freq_sets,
    "quick_rc": quick_rc,
    "sort_parameters": sort_parameters,
    "find_arc_minima": find_arc_minima,
    "find_x_intercepts": find_x_intercepts,
}

# def find_x_intercepts(data):
#     """
#     Find the x-intercepts of the Nyquist plot using cubic spline interpolation.

#     Parameters:
#     data (pd.DataFrame): DataFrame with real values in the first column and imaginary values in the second column.

#     Returns:
#     list: List of x-intercepts.
#     """
#     # Sort the data by the real values
#     data = data.sort_values(by=data.columns[0])

#     real_vals = data.iloc[:, 0].values
#     imag_vals = data.iloc[:, 1].values

#     window_length = len(real_vals) // 10

#     # Apply Savitzky-Golay filter to smooth the imaginary values
#     smoothed_imag_vals = savgol_filter(imag_vals, window_length, 3)

#     # Use CubicSpline to interpolate the imaginary values
#     cs = CubicSpline(real_vals, smoothed_imag_vals)

#     # Find the roots of the cubic spline (where the imaginary part is zero)
#     roots = cs.roots()

#     return roots.tolist()

# def find_x_intercepts(data, window_percent=0.1):
#     """
#     Find the x-intercepts of the Nyquist plot using linear fits within a sliding window.

#     Parameters:
#     data (pd.DataFrame): DataFrame with real values in the first column and imaginary values in the second column.
#     window_percent (float): The percentage of the data range to use for the window size.

#     Returns:
#     list: List of x-intercepts.
#     """
#     # Sort the data by the first column
#     data = data.sort_values(by=data.columns[0])

#     real_vals = data.iloc[:, 0].values
#     imag_vals = data.iloc[:, 1].values

#     # Invert the second column
#     inverted_imag_vals = -imag_vals

#     # Calculate the window size based on the data range
#     data_range = real_vals.max() - real_vals.min()
#     window_size = data_range * window_percent

#     # Initialize lists to store the x-intercepts
#     intercepts = []
#     x_intercepts = []

#     prior_slope = None
#     i = 0

#     while i < len(real_vals):
#         # Define the window range
#         window_start = inverted_imag_vals[i]
#         window_end = window_start + window_size

#         window_mask = np.array([False] * len(real_vals))
#         n = i
#         # convert mask to true while the value is within the window
#         while n < len(real_vals) and inverted_imag_vals[n] <= window_end and inverted_imag_vals[n] >= window_start:
#             window_mask[n] = True
#             n += 1

#         # # Get the data points within the window
#         # window_mask = (real_vals >= window_start) & (real_vals <= window_end)
#         # if sum(window_mask) < 2:
#         #     if i < len(real_vals) - 2:
#         #         # create array of false values make i true and the next value true
#         #         window_mask = np.array([False] * len(real_vals))
#         #         window_mask[i:i+2] = True
#         #     else:
#         #         window_mask = np.array([False] * len(real_vals))
#         #         window_mask[i-2:i] = True
#         window_real_vals = real_vals[window_mask]
#         window_imag_vals = inverted_imag_vals[window_mask]

#         # Perform a linear fit
#         slope, intercept = np.polyfit(window_real_vals, window_imag_vals, 1)

#         # Calculate the x-intercept
#         x_intercept = -intercept / slope

#         if prior_slope is None:
#             prior_slope = slope
#             intercepts.append(x_intercept)
#         elif np.sign(prior_slope) == np.sign(slope):
#             prior_slope = slope
#             intercepts.append(x_intercept)
#         elif np.sign(prior_slope) != np.sign(slope):
#             if np.sign(prior_slope) == 1:
#                 x_intercepts.append(max(intercepts))
#             else:
#                 x_intercepts.append(min(intercepts))
#             prior_slope = slope
#             intercepts = []

#         # Calculate the midpoint of the current window
#         midpoint = (window_start + window_end) / 2

#         # Break if the midpoint is out of range
#         if midpoint > real_vals.max():
#             break

#         # Move to the next window starting from the midpoint
#         i = np.searchsorted(real_vals, midpoint)

#     # Return the x-intercepts
#     return x_intercepts
# def check_bands(df, shift_band=True, shift_value=False):
#     """
#     Ensure that the bands encompass a given value.

#     Parameters:
#     self: The self object containing the bound range.
#     df (pd.DataFrame): DataFrame containing the parameter values and bounds.
#     value (float): The value to ensure is within the bounds.

#     Returns:
#     pd.DataFrame: DataFrame with adjusted bounds.
#     """
#     max_exp = np.floor(np.log10(np.max(np.finfo(np.float64).max))) / 4

#     df['bnd_low'] = df['bnd_low'].replace(np.inf, 10**max_exp)  # Replace inf with a large positive number
#     df['bnd_high'] = df['bnd_high'].replace(-np.inf, -10**max_exp)  # Replace -inf with a large negative number
#     df['bnd_low'] = df['bnd_low'].replace(0, 10**-max_exp)  # Replace 0 with a small positive number

#     # Ensure that the bounds are adjusted so that the value is within bounds
#     df['bnd_low'] = df.apply(lambda row: float(min(row['bnd_low'], row['value'])-10**-max_exp), axis=1)
#     df['bnd_high'] = df.apply(lambda row: float(max(row['bnd_high'], row['value'])+10**-max_exp), axis=1)

#     return df

# def pin_to_param(self, df=None):
#     """
#     Convert pinned data to parameter DataFrames.

#     Parameters:
#     self: The self object containing the initial DataFrame structure.
#     df (pd.DataFrame): DataFrame containing the pinned data.

#     Returns:
#     dict: Dictionary of DataFrames with parameter values and stds, keyed by the 'Name' column.
#     """
#     # Automated part using the main class
#     df = df or self.pinned.df.copy()

#     # Primary section independent of the main class
#     param_dfs = {}
#     for _, row in df.iterrows():
#         param_names = [col[:-7] for col in df.columns if col.endswith('_values')]
#         param_values = [row[f"{name}_values"] for name in param_names]
#         param_stds = [row[f"{name}_std"] for name in param_names]

#         param_df = pd.DataFrame({
#             'value': param_values,
#             'std': param_stds
#         }, index=param_names)

#         param_df.attrs['Name'] = row['Name']
#         param_df.attrs['Dataset'] = row['Dataset']
#         param_df.attrs['Model'] = row['Model']
#         param_df.attrs['Comments'] = row['Comments']

#         param_dfs[row['Name']] = param_df

#     return param_dfs
