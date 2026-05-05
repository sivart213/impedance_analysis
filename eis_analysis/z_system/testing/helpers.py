import time

import numpy as np
import pandas as pd

from testing.rc_ckt_sim import RCCircuit
from eis_analysis.z_system.complexer import parse_z_array


def baseline_parse_timings(n_runs=3, verbose=False, case_group_mode=0):
    """
    Run parse_z_array on three archetypal inputs and return their timings.
    Averaged over n_runs for stability.
    """
    ckt = RCCircuit(freq=(-4, 7, 100), true_values=[24, 1e9, 1e-11], noise=0.01)
    timings = {key: [] for key in ("init", "direct", "searched", "full")}

    for _ in range(n_runs):
        # primary mode: Clear and/or common cases
        if case_group_mode != 2:
            # Case 1: immediate return
            t0 = time.perf_counter()
            parse_z_array(ckt.Z)
            timings["init"].append(time.perf_counter() - t0)

            # Case 2: after eval of complex cols
            arr_complex = np.column_stack([ckt.Z_noisy.real, 1j * ckt.Z_noisy.imag])
            t0 = time.perf_counter()
            parse_z_array(arr_complex)
            timings["direct"].append(time.perf_counter() - t0)

            # Case 3: after simple search eval
            arr_complex = np.column_stack([ckt.Z.real, ckt.Z.imag])
            t0 = time.perf_counter()
            parse_z_array(arr_complex)
            timings["searched"].append(time.perf_counter() - t0)

            # Case 4: full eval
            arr_full = np.column_stack(
                [ckt.freq, (1 / ckt.Z).real, (1 / ckt.Z).imag, abs((1 / ckt.Z))]
            )
            t0 = time.perf_counter()
            parse_z_array(arr_full)
            timings["full"].append(time.perf_counter() - t0)

        # secondary mode: less common cases
        if case_group_mode != 1:
            if verbose:
                print("-", end=" ")

            # Case 1: immediate return
            t0 = time.perf_counter()
            parse_z_array([])
            timings["init"].append(time.perf_counter() - t0)

            # Case 2: after eval of complex cols
            arr_complex = np.column_stack([ckt.freq, ckt.Z_noisy])
            t0 = time.perf_counter()
            parse_z_array(arr_complex)
            timings["direct"].append(time.perf_counter() - t0)

            # Case 3: after simple search eval
            arr_complex = np.column_stack([ckt.freq, ckt.Z.real, ckt.Z.imag])
            t0 = time.perf_counter()
            parse_z_array(arr_complex)
            timings["searched"].append(time.perf_counter() - t0)

            # Case 4: full eval
            arr_full = np.column_stack([(1 / ckt.Z).real, (1 / ckt.Z).imag, abs((1 / ckt.Z))])
            t0 = time.perf_counter()
            parse_z_array(arr_full)
            timings["full"].append(time.perf_counter() - t0)

        if verbose:
            print()

    # average
    return {k: np.array(v) for k, v in timings.items()}


def make_timing_bins(n_runs=20, q_low=0.1, q_high=0.9, margin=1.05):
    """
    Build bin-check callables using shared boundaries informed by both distributions.

    - q_low/q_high: quantiles used to form boundaries (robust to tails/outliers).
    - margin: small expansion factor to absorb jitter.
    """
    s = baseline_parse_timings(n_runs)

    # robust endpoints for each distribution
    init_hi = np.quantile(s["init"], q_high)
    complex_lo = np.quantile(s["direct"], q_low)
    complex_hi = np.quantile(s["direct"], q_high)
    searched_lo = np.quantile(s["searched"], q_low)
    searched_hi = np.quantile(s["searched"], q_high)
    full_lo = np.quantile(s["full"], q_low)

    # shared boundaries (midpoints of robust endpoints), with margin
    lower = ((init_hi + complex_lo) / 2.0) * margin  # init vs complex
    mid = ((complex_hi + searched_lo) / 2.0) * margin  # complex vs searched
    upper = ((searched_hi + full_lo) / 2.0) / margin  # searched vs full
    # upper = s["full"].min()

    # bin predicates on elapsed time (you’ll pass elapsed in your tests)
    bins = {
        "init": lambda t: t < lower,
        "direct": lambda t: lower <= t < mid,
        "searched": lambda t: mid <= t < upper,
        "full": lambda t: t >= upper,
    }

    # optional: a classifier that maps elapsed → label
    def classify(t):
        if t < lower:
            return "init"
        if t < mid:
            return "direct"
        if t < upper:
            return "searched"
        return "full"

    # expose boundaries and samples for inspection
    meta = {
        "lower": lower,
        "mid": mid,
        "upper": upper,
        "samples": s,
        "q_low": q_low,
        "q_high": q_high,
        "margin": margin,
    }

    return bins, classify, meta


def percent_mismatch(arr1, arr2):
    # relative mismatch: mean absolute difference / mean magnitude
    if np.iscomplexobj(arr1) or np.iscomplexobj(arr2):
        real_perc = percent_mismatch(arr1.real, arr2.real)
        imag_perc = percent_mismatch(arr1.imag, arr2.imag)
        return (real_perc + imag_perc) / 2.0
    mismatch = np.isclose(arr1, arr2).mean()
    return np.round(mismatch * 100.0, 2)


def explain_parsed_mismatch(df, result, prefix="res ->", perc="", print_arr=False):
    """
    Try to guess which df columns contributed to parsed.
    Returns a human-readable message.
    """

    def search(arr1, arr2=None):
        cols = {}
        min_len = 2 if arr2 is not None else 1
        for col in df.columns:
            vals = df[col].to_numpy()
            if np.allclose(abs(arr1), abs(vals)):
                cols["col1"] = col
            if arr2 is not None and np.allclose(abs(arr2), abs(vals)):
                cols["col2"] = col
            if len(cols) >= min_len:
                break
        return cols

    if np.iscomplexobj(result):
        found = search(result.real, result.imag)
        find_type = "rectangular"
        if len(found) < 2:
            alt_found = search(np.abs(result), np.angle(result, deg=True))
            # if len(polar_found) < 2 use longer result
            if len(alt_found) == 2 or len(alt_found) > len(found):
                found = alt_found
                find_type = "polar"
        if len(found) == 2:
            msg = " and ".join(found.values())
            msg += f" (as {find_type})"
        elif len(found) == 1:
            msg = list(found.values())[0] + " and another column"
            msg += f" (as {find_type})"
        else:
            msg = "unknown columns"

    else:
        found = search(result)
        if len(found) == 1:
            msg = list(found.values())[0]
        else:
            msg = "unknown"
    msg = f"{prefix} {msg}"
    msg += f" ({perc} off)" if perc else "."
    if print_arr:
        msg += f"\nParsed array: {np.array2string(result, threshold=10, edgeitems=3)}\n"

    return msg


def get_data_dict(data, freq, sign):
    return {
        "freq": freq,
        "complex": data,
        "real": data.real,
        "imag": sign * data.imag,
        "1j*imag": sign * 1j * data.imag,
        "mag": np.abs(data),
        "phase": sign * np.angle(data, deg=True),
        "r-phase": sign * np.angle(data, deg=False),
    }


def get_data_form(rc_data, noise_opt, data_form="Z"):

    data = rc_data.Z_noisy if noise_opt else rc_data.Z
    if data_form == "Y":
        data = 1 / data
    elif data_form == "M":
        data = 1j * 2 * np.pi * rc_data.freq * 8.854e-14 * (25 / 450e-4) * data
    elif data_form == "e_r":
        data = 1 / (1j * 2 * np.pi * rc_data.freq * 8.854e-14 * (25 / 450e-4) * data)
    return data


def get_data_df(
    data, freq, form, freq_opt, extra_opt, sign, etype, alt_data=None, seqential=False
):

    ETYPE_MAP = {"a": 0, "b": 1, "c": 2}
    x_group = "rect"
    x_group_map = {
        "rect": ["real", "imag", "1j*imag"],
        "polar": ["mag", "phase", "1j*imag"],
    }
    base = get_data_dict(data, freq, sign)
    x_base = base if alt_data is None else get_data_dict(alt_data, freq, sign)
    # Determine main target columns by form
    target_cols = []
    if freq_opt:
        target_cols.append("freq")

    if form == "complex":
        target_cols.append("complex")
    elif form == "1j*rect":
        target_cols.extend(["real", "1j*imag"])
        x_group = "polar"
    elif form == "rect":
        target_cols.extend(["real", "imag"])
        x_group = "polar"
    elif form == "polar":
        target_cols.extend(["mag", "phase"])
    elif form == "r-polar":
        target_cols.extend(["mag", "r-phase"])
    else:
        raise ValueError("Unknown form")

    df = pd.DataFrame({col: base[col] for col in target_cols})

    if extra_opt == 1:
        esign = 1
        etype = etype or "a"
        if etype[0] == "-":
            esign = -1
            etype = etype[1:]

        col = x_group_map[x_group][ETYPE_MAP.get(etype, 1)]
        df[f"x-{col}"] = esign * x_base[col]
    elif extra_opt == 2:
        for i in range(extra_opt):
            col = x_group_map[x_group][i]
            df[f"x-{col}"] = x_base[col]

    if seqential:
        # return pd.concat([df, df[::-1]], ignore_index=True).sort_values("arr", ignore_index=True)
        return pd.DataFrame(np.repeat(df.values, 2, axis=0), columns=df.columns)

    return df
