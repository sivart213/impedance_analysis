import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.vq import kmeans2

from eis_analysis.dc_fit.extract_tools import (
    form_std_df_index,
)

np.seterr(invalid="raise")


def extract_name_parts(string: str) -> tuple[str, str, float, int, str]:
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
        r"(?P<sample_name>.*\d)(?P<condition>[a-zA-Z]+)(?P<temp>\d{2}c)(?P<run>_r\d+)"
    )
    match = pattern.match(string)
    if match:
        name = match.group("sample_name").strip()
        contam = int(np.round(int(name), -1)) if name.isdigit() else 0
        return (
            name,
            match.group("condition"),
            float(match.group("temp")[:2]),
            contam,
            match.group("run").strip("_"),
        )
    else:
        raise ValueError(f"Key '{string}' does not match the expected format.")


# %% Periodicity Detection
def detect_period(column: pd.Series) -> tuple[int | None, float, float, float]:
    """
    Detect the periodicity of a given column using Fast Fourier Transform (FFT).

    Parameters
    ----------
    column : pd.Series
        Series of measured values to analyze.

    Returns
    -------
    tuple[int | None, float, float, float]
        A tuple containing:
        - period: Detected period in samples, or None if no periodicity is found.
        - strength: Strength of the periodic signal.
        - snr: Signal-to-noise ratio.
        - sharpness: Sharpness of the peak in the frequency domain.
    """
    # Remove the mean to focus on oscillations
    detrended_column = column - float(np.mean(column))

    # Normalize BEFORE FFT to ensure scale invariance
    std = np.std(detrended_column)
    if std == 0:
        std = 1  # Prevent division by zero for constant signals
    normalized_column = detrended_column / std  # type: ignore

    # Calculate the sampling period
    T = 1
    N = len(normalized_column)

    # Compute the FFT
    yf = fft(normalized_column)  # Perform the Fast Fourier Transform
    xf = fftfreq(N, T)[: N // 2]  # Compute the frequency bins

    fft_mag = np.abs(yf[: N // 2])  # type: ignore

    # Ignore DC component
    fft_mag_no_dc = fft_mag[1:]
    xf_no_dc = xf[1:]

    # Find peak frequency and its magnitude
    idx_peak = np.argmax(fft_mag_no_dc)
    peak_mag = fft_mag_no_dc[idx_peak]

    # # Estimate noise floor as median of all other bins
    # noise_floor = np.median(np.delete(fft_mag_no_dc, idx_peak))

    # Improved noise floor: use 90th percentile (robust to outliers)
    noise_floor = np.percentile(np.delete(fft_mag_no_dc, idx_peak), 90)

    # Peak sharpness: compare to immediate neighbors
    if 0 < idx_peak < len(fft_mag_no_dc) - 1:
        neighbor_mean = (fft_mag_no_dc[idx_peak - 1] + fft_mag_no_dc[idx_peak + 1]) / 2
    elif idx_peak == 0 and len(fft_mag_no_dc) > 1:
        neighbor_mean = fft_mag_no_dc[1]
    elif idx_peak == len(fft_mag_no_dc) - 1 and len(fft_mag_no_dc) > 1:
        neighbor_mean = fft_mag_no_dc[-2]
    else:
        neighbor_mean = 1e-12  # fallback for very short signals

    # Sharpness ratio: how much higher is the peak than its neighbors
    sharpness = peak_mag / (neighbor_mean + 1e-12)

    # Signal-to-noise ratio
    if noise_floor == 0:
        snr = 0
    else:
        snr = peak_mag / noise_floor

    # Composite metric: geometric mean (or other function) of snr and sharpness
    strength = sharpness / (snr if snr > 0 else 1e-12)  # Avoid division by zero

    freq = xf_no_dc[idx_peak]
    if freq == 0:
        return None, 0, snr, sharpness

    period = int(1 / freq / 2)
    return period, strength, snr, sharpness


def refine_breakpoints(column: pd.Series, period: int, times: np.ndarray) -> list[int]:
    """
    Refine breakpoints in a periodic column using peak detection and smoothing.

    Parameters
    ----------
    column : pd.Series
        Series of measured values to analyze.
    period : int
        Expected period of the signal in samples.
    times : np.ndarray | None, optional
        Array of time values corresponding to the column, used for smoothing. If None, uses index.

    Returns
    -------
    list[int]
        List of refined breakpoints in the column.
    """

    def find_peaks_using_heights(array, q=0.85):
        """
        Find peaks in the data based on the specified height threshold.
        """
        peaks, heights = find_peaks(abs(array), height=np.quantile((array), q))
        heights = heights["peak_heights"]
        digitized = np.digitize(heights, np.histogram_bin_edges(heights, bins=4), right=True)
        max_heights = np.mean(heights[digitized >= 2])
        if np.mean(abs(array)) / max_heights > 0.2:
            return np.array([0, len(array)])
        return peaks[digitized >= 2]

    # Smooth the data
    s_values = savgol_filter(column, 3, 1, deriv=0, delta=float(np.mean(np.diff(times))))
    slopes = savgol_filter(s_values, 3, 2, deriv=1, delta=float(np.mean(np.diff(times))))
    d_slopes = savgol_filter(s_values, 3, 2, deriv=2, delta=float(np.mean(np.diff(times))))

    # Find peaks in the data
    slope_peaks = find_peaks_using_heights(slopes)
    d_slope_peaks = find_peaks_using_heights(d_slopes)

    # Combine all peaks
    all_peaks = np.union1d(slope_peaks, d_slope_peaks)

    # Number of segments (not breakpoints)
    n_segments = max(1, int(round(len(column) / period)))

    # Filter out peaks too close to the start or end
    min_idx = period // 2
    max_idx = len(column) - period // 2
    all_peaks = all_peaks[(all_peaks >= min_idx) & (all_peaks <= max_idx)]

    if all_peaks.size == 0:
        # If no peaks found, return default breakpoints
        return [0, len(column)]

    centers = np.linspace(period, len(column) - period, n_segments - 1)

    if len(column) % period != 0 and len(all_peaks) > n_segments:
        centroids, _ = kmeans2(
            all_peaks.reshape(-1, 1).astype(float), centers.reshape(-1, 1), minit="matrix"
        )

        all_peaks = np.array([round(c) for c in centroids.flatten()])

    closest_peaks = all_peaks[np.argmin(np.abs(all_peaks - centers.reshape(-1, 1)), axis=1)]

    refined_peaks = set()
    for center, peak in zip(centers, closest_peaks):
        if abs(peak - center) <= period / 2:
            refined_peaks.add(int(peak))

    return sorted(refined_peaks | {0} | {len(column)})


def find_periodicity(
    dataframe: pd.DataFrame,
    time: np.ndarray,
    strength_threshold: float = 1.0,
    sparse_periodicity: bool = True,
) -> tuple[list[int], list[str]]:
    """
    Detect periodic columns and segment breakpoints in a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame with columns to analyze.
    time : np.ndarray
        Time array corresponding to the DataFrame rows.
    strength_threshold : float, optional
        Minimum strength required to consider a column periodic.
    sparse_periodicity : bool, optional
        If True, enforces the assumption that the number of periods should be less than their length.

    Returns
    -------
    breakpoints : list[int]
        List of segment breakpoints.
    periodic_columns : list[str]
        List of columns identified as periodic.
    """
    results = []

    # Detect period, strength, snr, and sharpness for each column
    for column in dataframe.columns:
        period, strength, snr, sharpness = detect_period(dataframe[column])
        # Store all metrics for later analysis
        if (
            period is not None
            and strength >= strength_threshold
            # Only allow periods where period length > number of periods (expect few periods)
            and (not sparse_periodicity or len(dataframe) // period < period)  # type: ignore
        ):
            results.append(
                {
                    "column": column,
                    "period": period,
                    "strength": strength,
                    "snr": snr,
                    "sharpness": sharpness,
                }
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return [0, len(dataframe)], []

    # strongest_period: float = results_df.loc[results_df["strength"].idxmax(), "period"]  # type: ignore
    most_common_period = results_df["period"].value_counts().idxmax()
    average_period = results_df.loc[
        (results_df["period"] - results_df["period"].mean()).abs().idxmin(), "period"
    ]

    # Add deviation columns to results_df (lower is better)
    results_df["dev_from_common"] = (results_df["period"] - int(most_common_period)).abs()
    results_df["dev_from_avg"] = (results_df["period"] - average_period).abs()  # type: ignore

    # --- Period ranking: rank periods by best metric and closeness to common/average ---
    period_metrics = (
        results_df.groupby("period")[
            ["strength", "snr", "sharpness", "dev_from_common", "dev_from_avg"]
        ]
        .max()
        .reset_index()
    )

    # Rank each metric (higher is better for strength, snr, sharpness; lower is better for deviation)
    period_metrics["strength_rank"] = period_metrics["strength"].rank(
        ascending=False, method="min"
    )
    period_metrics["snr_rank"] = period_metrics["snr"].rank(ascending=False, method="min")
    period_metrics["sharpness_rank"] = period_metrics["sharpness"].rank(
        ascending=False, method="min"
    )
    period_metrics["common_rank"] = period_metrics["dev_from_common"].rank(
        ascending=True, method="min"
    )
    period_metrics["avg_rank"] = period_metrics["dev_from_avg"].rank(ascending=True, method="min")

    # Sum the ranks (you can adjust weights if desired)
    period_metrics["total_rank"] = (
        period_metrics["strength_rank"]
        + period_metrics["snr_rank"]
        + period_metrics["sharpness_rank"]
        + period_metrics["common_rank"]
        + period_metrics["avg_rank"]
    )

    # Select the period with the lowest total rank
    best_period_row = period_metrics.loc[period_metrics["total_rank"].idxmin()]
    most_likely_period = int(best_period_row["period"])  # type: ignore

    # --- Ranked evaluation for best column representing the period ---
    period_df = results_df[results_df["period"] == most_likely_period].copy()
    if len(period_df) == 1:
        idx_best = period_df.index[0]
    else:
        # Rank each metric (lower rank is better)
        period_df["strength_rank"] = period_df["strength"].rank(ascending=False, method="min")
        period_df["snr_rank"] = period_df["snr"].rank(ascending=False, method="min")
        period_df["sharpness_rank"] = period_df["sharpness"].rank(ascending=False, method="min")
        # Sum the ranks
        period_df["total_rank"] = (
            period_df["strength_rank"] + period_df["snr_rank"] + period_df["sharpness_rank"]
        )
        # Select the index with the lowest total rank
        idx_best = period_df["total_rank"].idxmin()
    most_periodic_column = period_df.loc[idx_best, "column"]

    # Use the most periodic column to refine breakpoints
    refined_breakpoints = refine_breakpoints(
        dataframe[most_periodic_column], most_likely_period, time
    )

    return [int(b) for b in refined_breakpoints], results_df["column"].to_list()


# %% Data Preparation and initial guesses
def prepare_data_for_processing(df: pd.DataFrame) -> dict:
    """
    Prepare time series, data columns, time array, and periodicity info for steady-state analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing datetime and measurement columns.

    Returns
    -------
    prepared_data : dict
        Dictionary with keys: data_df, time_seconds, segments, periodic_columns, data_date.
    """
    prepared_data = {
        "data_date": pd.Timestamp.now(),  # datetime.now().date(),
        "data_time": datetime.now().time(),
        "data_df": pd.DataFrame(),
        "time": np.array([]),
        "segments": [],
        "periodic_columns": [],
    }

    time_df: pd.DataFrame = df.select_dtypes(include=["datetime"])
    data_df = df.select_dtypes(exclude=["datetime"])

    if not time_df.empty:
        time_series: pd.Series = time_df.iloc[:, 0]
        times = (time_series - time_series.iloc[0]).apply(lambda x: x.total_seconds())
        slope, t_min, r2, *_ = tuple(linregress(times.index, times))
        prepared_data["time"] = slope * times.index.to_numpy(copy=True)
        if r2 >= 0.9:
            # Use linear fit
            prepared_data["data_date"] = time_series.iloc[0] + pd.to_timedelta(t_min, "s")
            prepared_data["data_time"] = prepared_data["data_date"].time()
        else:
            # Fallback to interpolation
            times[times.duplicated(keep="first")] = np.nan
            prepared_data["time"] = times.interpolate(
                method="from_derivatives", extrapolate=True
            ).to_numpy(copy=True)
            prepared_data["data_date"] = time_series.iloc[0]
            prepared_data["data_time"] = time_series.iloc[0].time()

    if not data_df.empty:
        # Invariant: At least one column will always contain '.value'
        # Data-specific: Expect multiple columns to end with '.value' (based on the data source).
        data_df.columns = [re.sub(r"\.value$", "", col) for col in data_df.columns]
        prepared_data["data_df"] = data_df

    if not data_df.empty and not time_df.empty:
        segments, periodic_columns = find_periodicity(data_df, prepared_data["time"])
        prepared_data["segments"] = segments
        prepared_data["periodic_columns"] = periodic_columns

    return prepared_data


def process_segments_mean(
    raw_data: dict,
    *keys: str,
    return_parsed_info: bool = False,
    inclusive: bool = True,
) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame], dict[tuple, pd.DataFrame]]:
    """
    Process all selected raw_data entries, segmenting and calculating means for each segment.

    Parameters
    ----------
    raw_data : dict
        Dictionary of loaded data, each value should be a DataFrame.
    *keys : str
        Optional keys to filter raw_data (substring match).
    return_parsed_info : bool, optional
        If True, unsegmented_dict will be a dict of dicts containing the parsed info and un_seg_df.
    inclusive : bool, optional
        If True, only include entries whose key contains any of *keys.
        If False, exclude entries whose key contains any of *keys.

    Returns
    -------
    seg_means_df : pd.DataFrame
        Concatenated DataFrame of segment means for all entries, MultiIndex with sample info and segment.
        Index: (sample_name, condition, temp, run, segment)
        Columns: includes 'date' and all data columns.
    arrays_dict : dict[tuple, pd.DataFrame]
        Dict mapping MultiIndex tuples (sample_name, condition, temp, run, segment) to DataFrames for each segment.
        Each DataFrame has absolute time as the index (named "meas_time") and as the first column ("time").
    unsegmented_dict : dict[tuple, pd.DataFrame] or dict[tuple, dict]
        Dict mapping MultiIndex tuples (sample_name, condition, temp, run) to unsegmented DataFrames,
        or to dicts containing parsed info and un_seg_df if return_parsed_info is True.
    """
    means_frames = {}
    dates = []
    arrays_dict = {}
    unsegmented_dict = {}

    for key, data in raw_data.items():
        if keys:
            match = any(k in key for k in keys)
            if (inclusive and not match) or (not inclusive and match):
                continue

        prepared = prepare_data_for_processing(data.copy().dropna())
        segments = prepared["segments"]

        if prepared["data_df"].empty or len(prepared["time"]) == 0 or len(segments) < 2:
            continue

        sample_info = extract_name_parts(key)

        un_seg_df = prepared["data_df"].copy()
        un_seg_df.insert(0, "time", prepared["time"])
        un_seg_df.index = pd.Index(prepared["time"], name="meas_time")
        un_seg_df.attrs["date"] = prepared["data_date"]
        if return_parsed_info:
            prepared_with_df = dict(prepared)
            prepared_with_df["combined_df"] = un_seg_df
            unsegmented_dict[sample_info] = prepared_with_df
        else:
            unsegmented_dict[sample_info] = un_seg_df

        for seg_idx in range(len(segments) - 1):
            start, end = segments[seg_idx], segments[seg_idx + 1]
            seg_len = end - start
            if seg_len == 0:
                continue

            # Slice from un_seg_df for this segment
            seg_df = un_seg_df.iloc[start:end].copy()
            seg_df.attrs["date"] = prepared["data_date"].date()
            seg_df.attrs["timestamp"] = prepared["data_date"] + pd.to_timedelta(
                seg_df["time"].iloc[0], "s"
            )
            seg_df["time"] = seg_df["time"] - seg_df["time"].iloc[0]
            arrays_dict[sample_info + (seg_idx,)] = seg_df

            data_cols = seg_df.columns.difference(["time"])
            if seg_len < 4:
                means = seg_df[data_cols].mean(axis=0)
            else:
                start_idx = int(seg_len * 0.6)
                clip_end = seg_len - max(1, int(0.05 * seg_len))
                if start_idx >= clip_end:
                    means = seg_df[data_cols].mean(axis=0)
                else:
                    means = seg_df.iloc[start_idx:clip_end][data_cols].mean(axis=0)

            means_frames[sample_info + (seg_idx,)] = means
            # dates.append(prepared["data_date"])
            dates.append(seg_df.attrs["timestamp"])

    if not means_frames:
        return pd.DataFrame(), {}, {}

    seg_means_df = form_std_df_index(pd.DataFrame(means_frames, dtype=float).T)
    seg_means_df.insert(0, "date", dates)

    # --- Reorder arrays_dict and unsegmented_dict to match sorted index ---
    arrays_dict = {k: arrays_dict[k] for k in list(seg_means_df.index)}
    unsegmented_dict = {k[:-1]: unsegmented_dict[k[:-1]] for k in list(seg_means_df.index)}

    return seg_means_df, arrays_dict, unsegmented_dict


# def process_segments_mean_old(
#     data_date: datetime,
#     data_df: pd.DataFrame,
#     time_seconds: np.ndarray,
#     segments: list,
#     **_,
# ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
#     """
#     Calculate mean values for each segment in the data.

#     Parameters
#     ----------
#     data_date : datetime
#         Date of the data, used for indexing.
#     data_df : pd.DataFrame
#         DataFrame containing measurement data.
#     time_seconds : np.ndarray
#         Array of time values in seconds corresponding to the data.
#     segments : list
#         List of segment indices, where each segment is defined by its start and end indices.

#     Returns
#     -------
#     tuple[pd.DataFrame | None, pd.DataFrame | None]
#         Tuple containing two DataFrames:
#         - segment_means_df: DataFrame with mean values for each segment.
#         - segment_arrays_df: DataFrame with arrays of values for each segment.
#         If no data is available, returns (None, None).
#     """
#     if data_df.empty or len(time_seconds) == 0 or len(segments) < 2:
#         return None, None

#     # Add time_seconds as the first column in segment_arrays_df, corrected to start at 0 for each segment
#     time_segments = []
#     for i in range(len(segments) - 1):
#         start, end = segments[i], segments[i + 1]
#         t_seg = time_seconds[start:end]
#         if len(t_seg) > 0:
#             t_seg = t_seg - t_seg[0]
#         time_segments.append(t_seg)
#     segment_arrays = {"time": time_segments}

#     segment_means = {}
#     for col in data_df.columns:
#         val_arr = data_df[col].to_numpy(copy=True)
#         seg_means = []
#         seg_arrs = []
#         for i in range(len(segments) - 1):
#             start, end = segments[i], segments[i + 1]
#             seg_arrs.append(val_arr[start:end])
#             seg_len = end - start
#             if seg_len < 4:
#                 seg_means.append(np.mean(val_arr[start:end]))
#                 continue
#             # Calculate indices for last 60%, excluding last 5%
#             start_idx = start + int(seg_len * 0.6)
#             clip_end = end - max(1, int(0.05 * seg_len))
#             if start_idx >= clip_end:
#                 seg_means.append(np.mean(val_arr[start:end]))
#                 continue
#             seg_means.append(np.mean(val_arr[start_idx:clip_end]))
#         segment_means[col] = seg_means
#         segment_arrays[col] = seg_arrs

#     # Build DataFrame: rows=segments, cols=columns
#     segment_means_df = pd.DataFrame(segment_means)
#     segment_means_df.insert(0, "date", data_date)
#     segment_means_df.index.name = "segment"

#     segment_arrays_df = pd.DataFrame(segment_arrays)
#     segment_arrays_df.index.name = "segment"

#     return segment_means_df, segment_arrays_df
# For each centroid, pick the actual peak closest to it
# refined_peaks = []
# for center in sorted(centroids.flatten()):
#     closest_peak = int(all_peaks[np.argmin(np.abs(all_peaks - center))])
#     refined_peaks.append(closest_peak)

# # Always include 0 and len(column) as breakpoints, and sort
# breakpoints = sorted(set([0] + refined_peaks + [len(column)]))

# If not enough peaks, fallback to old method
# if len(all_peaks) < n_segments:
# breakpoints = [0]
# for i in range(1, n_segments):
#     expected_peak = i * period
#     closest_peak = int(min(all_peaks, key=lambda x, x0=expected_peak: abs(x - x0)))
#     if (
#         closest_peak not in breakpoints
#         and closest_peak != len(column)
#         and abs(closest_peak - center) <= period / 2
#     ):
#         breakpoints.append(closest_peak)

# # Use scipy kmeans2 to cluster peaks
# # Initial centers are spaced at expected segment locations
# # Use scipy kmeans2 to cluster peaks
# # Initial centers are spaced at expected segment locations
# centers = np.linspace(min_idx, max_idx, n_segments).reshape(-1, 1)
# centroids, labels = kmeans2(all_peaks.reshape(-1, 1).astype(float), centers, minit='matrix')

# # For each centroid, pick the actual peak closest to it
# refined_peaks = []
# for center in sorted(centroids.flatten()):
#     closest_peak = int(all_peaks[np.argmin(np.abs(all_peaks - center))])
#     refined_peaks.append(closest_peak)

# # Always include 0 and len(column) as breakpoints, and sort
# breakpoints = sorted(set([0] + refined_peaks + [len(column)]))

# return breakpoints
