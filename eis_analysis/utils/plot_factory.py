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
Created on Thu Oct  3 15:05:23 2024
"""
import re
import logging
from abc import ABC, abstractmethod
from typing import Any
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import Cursor
from matplotlib.collections import PathCollection, PolyCollection

logger = logging.getLogger(__name__)

np.seterr(all="raise")


def log_exceptions(func):
    """Decorator to log exceptions in a function."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (FloatingPointError, RuntimeWarning) as e:
            logger.error("Error in %s: %s", func.__name__, e)

    return wrapper


def rounded_floats(arr, as_exp: bool = False) -> np.ndarray:
    """
    Given an array of positive values, return the rounded decade exponents.
    Rule: mantissa < 5 → round down, mantissa >= 5 → round up.
    """
    arr = np.asarray(arr, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("All values must be positive")

    exp = np.floor(np.log10(arr)).astype(int)
    mant = arr / (10.0**exp)

    # Apply rule: if mant >= 5, bump exponent
    rounded_exp = exp + (mant >= 5)

    # print(rounded_exp)
    if as_exp:
        return rounded_exp
    return 10.0**rounded_exp


def sig_figs_ceil(number: int | float, digits: int = 3) -> float:
    """Round based on desired number of digits."""
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    root = 10 ** (int(power) - digits)
    return float(np.ceil(number / root) * root)


def annotation_positioner(
    data: Any, annotations: list, min_distance: float, mag_off: float, base_angle: float, **kwargs
) -> tuple:
    """
    Add annotations to a plot.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to add annotations to.
    - data (pd.DataFrame or np.ndarray): A 2-column DataFrame or NumPy array with x and y data.
    - annotations (list): Indices of the data points to annotation.
    - min_distance (float): Minimum distance between annotations.
    - mag_off (float): Magnitude offset for annotation positioning.
    - base_angle (float): Base angle for annotation positioning.
    - **kwargs: Additional keyword arguments for annotation.

    Returns:
    - annotation_positions (list): List of annotation positions.
    """
    annotation_positions = kwargs.pop("annotation_positions", [])

    xy_positions = []
    new_annotation_positions = []
    index = []

    # Ensure data is a NumPy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Check that rows > columns
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Get window from kwargs or derive from data
    window = kwargs.pop("window", None)
    if window is None:
        window = (
            (data[:, 0].min(), data[:, 0].max()),
            (data[:, 1].min(), data[:, 1].max()),
        )
    new_position = (0.0, 0.0)
    for anno in annotations:
        x, y = data[anno, :2]

        # Drop anno if the data position has effectively already been labeled
        if any(
            np.linalg.norm(np.array([x, y]) - np.array(pos)) <= min_distance
            for pos in xy_positions
        ):
            continue

        angle_mod = 5
        angle = base_angle
        iteration_count = 0
        max_iterations = 200  # Set a reasonable iteration limit
        mult = 1
        while iteration_count < max_iterations:
            dx = mag_off * np.cos(np.radians(angle))
            dy = mag_off * np.sin(np.radians(angle))
            new_position = (x + dx, y + dy)

            if all(
                np.linalg.norm(np.array(new_position) - np.array(pos)) > min_distance
                for pos in annotation_positions
            ):
                # Ensure the anno position is inside the window
                if (window[0][0] <= new_position[0] <= window[0][1]) and (
                    window[1][0] <= new_position[1] <= window[1][1]
                ):
                    # Perturb the anno away from the data points
                    if not any(
                        np.linalg.norm(np.array(new_position) - np.array([dx, dy])) < min_distance
                        for dx, dy in data[:, :2]
                    ):
                        break

            angle += 15 * mult
            if angle >= 360:
                angle = angle - 360 + angle_mod
                mag_off *= 1.1 * mult

            iteration_count += 1

        if iteration_count >= max_iterations / 2:
            mult = 2

        xy_positions.append((x, y))
        annotation_positions.append(new_position)
        new_annotation_positions.append(new_position)
        index.append(anno)

    return xy_positions, new_annotation_positions, index


def get_plot_data(
    ax: Axes,
    axis: Any = None,
    label: str = "",
    res_format: str = "arr",
    plt_type: str = "line, scatter, fill",
) -> Any:
    """Retrieve all data from the Axes object."""
    all_data = []
    all_labels = []

    if "any" in plt_type.lower() or "all" in plt_type.lower():
        plt_type = "line, scatter, fill"

    # Get data from lines
    if "line" in plt_type.lower():
        for line in ax.lines:
            if not label or line.get_label() == label:
                xy_data = line.get_xydata()
                all_data.append(xy_data)
                all_labels.append(line.get_label())

    # Get data from scatter plots and fill-between plots
    for collection in ax.collections:
        if "scatter" in plt_type.lower() and isinstance(collection, PathCollection):
            if not label or collection.get_label() == label:
                offsets = collection.get_offsets().data  # type: ignore
                all_data.append(offsets)
                all_labels.append(collection.get_label())
        elif "fill" in plt_type.lower() and isinstance(collection, PolyCollection):
            if not label or collection.get_label() == label:
                paths = collection.get_paths()
                for path in paths:
                    vertices = path.vertices
                    all_data.append(vertices)
                    all_labels.append(collection.get_label())

    # Return based on res_format
    # if not res_format or "arr" in res_format.lower():
    if "labels" in res_format.lower():
        return all_labels
    elif "data" in res_format.lower() or "list" in res_format.lower():
        return all_data
    elif "dict" in res_format.lower():
        return {label: data for label, data in zip(all_labels, all_data)}
    else:
        # Combine all data into a single array
        all_data = np.vstack(all_data)
        if axis is None:
            return all_data
        elif axis == "x" or axis == 0:
            return all_data[:, 0]
        elif axis == "y":
            return all_data[:, 1:]
        elif isinstance(axis, int) and axis < all_data.shape[1]:
            return all_data[:, axis]
        else:
            raise ValueError("Invalid axis argument")
        # raise ValueError(f"Invalid res_format: {res_format}")


class Annotations:
    def __init__(self, ax, base_angle=10, mag_off=0.1, min_distance=0.05, **kwargs):
        self.ax = ax
        self.base_angle = base_angle
        self.mag_off = mag_off
        self.min_distance = min_distance
        self.xy = []
        self.xyann = []
        self.text = []
        self._prev_kwargs = {}
        self.prev_kwargs(**kwargs)

    @property
    def all_xy(self):
        """Get all xy coordinates from the annotations in the ax."""
        return [annotation.xy for annotation in self.ax.texts]

    @property
    def all_xyann(self):
        """Get all xyann coordinates from the annotations in the ax."""
        return [annotation.xyann for annotation in self.ax.texts]

    @property
    def window(self):
        """Get the window from the associated ax."""
        return [self.ax.get_xlim(), self.ax.get_ylim()]

    def annotate(self, ax=None):
        """Annotate the associated ax."""
        if ax is not None:
            self.ax = ax
        for xy, xyann, text in zip(self.xy, self.xyann, self.text):
            self.ax.annotate(text, xy, xyann, **self._prev_kwargs)

    def prev_kwargs(self, **kwargs):
        """Update and return the previous kwargs."""
        self._prev_kwargs.update(kwargs)
        self._prev_kwargs.pop("mag_off", None)
        self._prev_kwargs.pop("base_angle", None)
        self._prev_kwargs.pop("window", None)
        return self._prev_kwargs

    def clear(self):
        """Clear annotations for the associated ax."""
        for annotation in self.ax.texts:
            if annotation.xyann in self.xyann:
                annotation.remove()
        self.xy.clear()
        self.xyann.clear()
        self.text.clear()

    def prepare_update(self):
        """Clear annotations and return xyann, base_angle, and mag_off."""
        self.clear()
        return self.xyann, self.base_angle, self.mag_off

    def shift_location(self):
        """Shift the location of the annotation."""
        base_angle = self.base_angle + 30
        mag_off = self.mag_off
        if base_angle > 180:
            base_angle = 10
            mag_off *= 1.25
        self.base_angle = base_angle
        self.mag_off = mag_off


# %% Abstract Classes


class AbstractFormatter(ABC):
    """Abstract class for formatting plots."""

    def __init__(self):

        self.font_props = {
            # "fontname": "DejaVu Sans",  # "Arial",
            "fontfamily": ["Arial", "DejaVu Sans"],  # "Arial",
            "fontsize": 18,
            "fontweight": "bold",
        }

        self.tick_props = {
            "axis": "both",
            "which": "major",
            "labelsize": 16,
            "labelfontfamily": ["Arial", "DejaVu Sans"],  # "Arial",
        }

    def apply_base_formatting(self, ax, **kwargs):
        """Apply base formatting to the Axes object."""
        font_props = {**self.font_props, **kwargs.get("font_props", {})}
        tick_props = {**self.tick_props, **kwargs.get("tick_props", {})}
        x_power_lim = kwargs.pop("x_power_lim", None)
        y_power_lim = kwargs.pop("y_power_lim", None)

        font_family = font_props.pop("fontfamily", ["Arial", "DejaVu Sans"])
        plt.rcParams["font.family"] = font_family

        if kwargs.pop("set_xlabel", False) and isinstance(kwargs.get("xlabel"), str):
            label = kwargs.get("xlabel", "").replace("$", "")
            ax.set_xlabel(label, **font_props)
        if kwargs.pop("set_ylabel", False) and isinstance(kwargs.get("ylabel"), str):
            label = kwargs.get("ylabel", "").replace("$", "")
            ax.set_ylabel(label, **font_props)
        if kwargs.pop("set_title", False) and isinstance(kwargs.get("title"), str):
            ax.set_title(kwargs.get("title", ""), **font_props)

        ax.tick_params(**tick_props)
        ax.grid(True)

        if x_power_lim is not None:
            x_formatter = ScalarFormatter()
            if isinstance(x_power_lim, int):
                x_formatter.set_powerlimits((-x_power_lim, x_power_lim))
            elif isinstance(x_power_lim, (list, tuple)) and len(x_power_lim) == 2:
                x_formatter.set_powerlimits(tuple(x_power_lim))
            ax.xaxis.set_major_formatter(x_formatter)

        if y_power_lim is not None:
            yformatter = ScalarFormatter()
            if isinstance(y_power_lim, int):
                yformatter.set_powerlimits((-y_power_lim, y_power_lim))
            elif isinstance(y_power_lim, (list, tuple)) and len(y_power_lim) == 2:
                yformatter.set_powerlimits(tuple(y_power_lim))
            ax.yaxis.set_major_formatter(yformatter)

        # plt.tight_layout()
        return ax

    @abstractmethod
    def apply_formatting(self, *ax, **kwargs) -> Any:
        """Abstract method to apply formatting to the Axes."""
        pass


class AbstractScaler(ABC):
    def __init__(self, axis):
        self.axis = axis

    @abstractmethod
    def calc_lims(self, arr, **kwargs) -> tuple[float, float]:
        """Apply scaling to the specified axis of the given Axes object."""
        pass

    @abstractmethod
    def scale(self, ax, arr, **kwargs) -> Any:
        """Apply scaling to the specified axis of the given Axes object."""
        pass

    @staticmethod
    def get_scale_functions(ax, axis):
        """Get the scale functions for the specified axis."""
        if "x" in axis:
            return ax.set_xscale, ax.xaxis.set_ticks, ax.set_xlim
        else:
            return ax.set_yscale, ax.yaxis.set_ticks, ax.set_ylim

    @staticmethod
    def filter_outliers(arr, quantile=5):
        """Filter outliers from the array based on the specified quantile."""
        if quantile <= 0:
            return arr
        if quantile > 1:
            quantile /= 100
        if quantile > 1:
            return arr

        Q1, Q3 = np.quantile(arr, [quantile, 1 - quantile])
        IQR = Q3 - Q1
        filtered_arr = arr[(arr >= Q1 - 1.5 * IQR) & (arr <= Q3 + 1.5 * IQR)]
        return filtered_arr


class AbstractAnnotator(ABC):
    """Abstract class for annotating plots."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.window = kwargs.pop("window", None)
        self.arrowprops = {
            **dict(
                facecolor="none",
                edgecolor="black",
                arrowstyle="->",
                linewidth=2.5,
            ),
            **kwargs.pop("arrowprops", {}),
        }
        self.bbox = {
            **dict(
                boxstyle="circle,pad=0.2",
                edgecolor="none",
                facecolor="white",
                alpha=0.7,
            ),
            **kwargs.pop("bbox", {}),
        }

        self.kwargs = kwargs

    def find_annotation_positions(
        self,
        data,
        annotations,
        min_distance=0.05,
        mag_off=0.1,
        base_angle=10,
        scale_by_window=True,
        window=None,
        **kwargs,
    ):
        """Find annotation positions for the given data and annotations."""

        if scale_by_window and window and min_distance < 1 and mag_off < 1:
            max_window_length = max(window[0][1] - window[0][0], window[1][1] - window[1][0])
            min_distance *= max_window_length
            mag_off *= max_window_length

        if "annotation_positions" in kwargs:
            annotation_positions = kwargs["annotation_positions"]
            if annotation_positions and isinstance(annotation_positions[0], list):
                kwargs["annotation_positions"] = [
                    item for sublist in annotation_positions for item in sublist
                ]

        return annotation_positioner(
            data,
            annotations,
            min_distance,
            mag_off,
            base_angle,
            window=window,
            **{**self.kwargs, **kwargs},
        )

    @abstractmethod
    def annotate(self, ax, data, **kwargs) -> Any:
        """Abstract method to annotate the plot."""
        pass


def parse_fmt_string(fmt):
    """
    Simplified version of Matplotlib's `_process_plot_format` function.

    Adapted from:
        - Function: `_process_plot_format`
        - Location: `matplotlib.axes._base`
        - Version: Matplotlib v3.10.1
        - Date: April 12, 2025

    Converts a MATLAB-style color/line style format string to a
    (*linestyle*, *marker*, *color*) tuple.

    Example format strings include:
    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines
    * 'C2--': the third color in the color cycle, dashed lines

    The format is absolute in the sense that if a linestyle or marker is not
    defined in *fmt*, there is no line or marker. This is expressed by
    returning 'None' for the respective quantity.

    Parameters:
        fmt (str): The format string to process.

    Returns:
        tuple: A tuple of (linestyle, marker, color).
    """

    linestyle = None
    marker = None
    color = None

    # First check whether fmt is just a colorspec
    if fmt not in ["0", "1"]:
        try:
            color = mcolors.to_rgba(fmt)
            return {"color": fmt}
        except ValueError:
            pass

    cn_color = None
    i = 0
    re_fmt = ""
    while i < len(fmt):
        c = fmt[i]
        if fmt[i : i + 2] in mlines.lineStyles and linestyle is None:  # Two-character linestyles
            linestyle = fmt[i : i + 2]
            re_fmt += linestyle
            i += 2
        elif c in mlines.lineStyles and linestyle is None:  # Single-character linestyles
            linestyle = c
            re_fmt += linestyle
            i += 1
        elif c in mlines.lineMarkers and marker is None:  # Markers
            marker = c
            re_fmt += marker
            i += 1
        elif c in mcolors.get_named_colors_mapping() and color is None:  # Named colors
            color = c
            re_fmt += color
            i += 1
        elif c == "C" and color is None:  # Color cycle (e.g., 'C0', 'C1', etc.)
            cn_color = re.match(r"C\d+", fmt[i:])
            if cn_color:
                color = mcolors.to_rgba(cn_color[0])
                re_fmt += cn_color[0]
                i += len(cn_color[0])
        else:
            break

    result = {}
    if re_fmt != fmt:
        return {"color": fmt}
    if linestyle is not None:
        result["linestyle"] = linestyle
    if marker is not None:
        result["marker"] = marker
    if color is not None:
        result["color"] = color
    return result


class AbstractPlot(ABC):
    def __init__(self):
        """
        Initialize the AbstractPlot with ignore_kwargs and short_kwargs.
        """
        # List of kwargs to always ignore (e.g., always defined in concrete classes)
        self.ignore_kwargs = ["data", "x", "y", "y1", "y2"]

        # Mapping of short kwargs to their full versions
        self.short_kwargs = {
            "c": "color",
            "ls": "linestyle",
            "lw": "linewidth",
            "m": "marker",
            "ms": "markersize",
            "mec": "markeredgecolor",
            "mew": "markeredgewidth",
            "mfc": "markerfacecolor",
            "alpha": "alpha",
        }

    @property
    def primary_kwargs(self):
        """
        Generic primary_kwargs property to be overridden by concrete classes.
        Returns an empty list by default.
        """
        return []

    @abstractmethod
    def plot(self, ax, data, keys, **kwargs):
        pass

    def sanitize_kwargs(self, **kwargs):
        """
        Sanitize the kwargs for plotting, including processing the 'fmt' argument.

        Args:
            data (pd.DataFrame or np.ndarray): The data being plotted.
            **kwargs: Keyword arguments passed to the plot function.

        Returns:
            dict: Sanitized kwargs with 'fmt' processed and removed.
        """
        fmt = kwargs.pop("fmt", None)

        if isinstance(fmt, str):
            # Process the fmt string using parse_fmt_string
            kwargs = {**parse_fmt_string(fmt), **kwargs}

        # Remove ignored kwargs
        kwargs = {key: value for key, value in kwargs.items() if key not in self.ignore_kwargs}

        # Resolve short kwargs to their full versions
        resolved_kwargs = {}
        for key, value in kwargs.items():
            full_key = self.short_kwargs.get(key, key)  # Map short key to full key if it exists
            if full_key in resolved_kwargs:
                # Prevent duplicates by skipping if the full key is already set
                continue
            resolved_kwargs[full_key] = value

        return resolved_kwargs


# %% Classes
class ScatterPlot(AbstractPlot):
    @property
    def primary_kwargs(self):
        return [
            "x",
            "y",
            "c",
            "data",
            "edgecolor",
            "label",
            "s",
            "norm",
            "marker",
            "alpha",
        ]

    def plot(self, ax, data, keys, **kwargs):
        kwargs = self.sanitize_kwargs(**kwargs)
        ax.scatter(
            x=keys[0],
            y=keys[1],
            c=kwargs.pop("color", "b"),
            data=data,
            edgecolor=kwargs.pop("edgecolor", "none"),
            label=kwargs.pop("label", "_none"),
            **kwargs,
        )
        # plt.tight_layout()
        return ax


class LinePlot(AbstractPlot):
    @property
    def primary_kwargs(self):
        return [
            "*[[x], y, [fmt]]",
            "data",
            "label",
            "linewidth",
            "linestyle",
            "marker",
            "markersize",
            "alpha",
        ]

    def plot(self, ax, data, keys, **kwargs):
        kwargs = self.sanitize_kwargs(**kwargs)
        ax.plot(
            keys[0],
            keys[1],
            data=data,
            label=kwargs.pop("label", "_none"),
            color=kwargs.pop("color", "k"),
            **kwargs,
        )
        # plt.tight_layout()
        return ax


class BandPlot(AbstractPlot):
    @property
    def primary_kwargs(self):
        return [
            "x",
            "y1",
            "y2",
            "data",
            "color",
            "alpha",
            "where",
            "interpolate",
            "step",
        ]

    def plot(self, ax, data, keys, **kwargs):
        kwargs = self.sanitize_kwargs(**kwargs)
        ax.fill_between(
            x=keys[0],
            y1=keys[1],
            y2=keys[2],
            data=data,
            color=kwargs.pop("color", "grey"),
            alpha=kwargs.pop("alpha", 0.25),
            **kwargs,
        )
        # plt.tight_layout()
        return ax


class DefaultFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs) -> Any:
        ax = ax[0]
        kwargs["set_xlabel"] = True
        kwargs["set_ylabel"] = True
        kwargs["set_title"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class BotAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs) -> Any:
        ax = ax[0]
        kwargs["set_xlabel"] = True
        kwargs["set_ylabel"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class MidAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs) -> Any:
        ax = ax[0]

        kwargs["x_power_lim"] = None

        kwargs["set_ylabel"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class TopAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs) -> Any:
        ax = ax[0]

        kwargs["x_power_lim"] = None

        kwargs["set_ylabel"] = True
        kwargs["set_title"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class StackFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs) -> Any:
        try:
            ylabels = kwargs.pop("ylabel")
            if isinstance(ylabels, str) or ylabels is None:
                ylabels = [ylabels] * len(ax)
            elif len(ylabels) != len(ax):
                raise ValueError("Length of ylabels must match the number of axes")
        except TypeError as exc:
            raise TypeError("ylabel(s) must be a string, an indexable sequence, or None") from exc

        try:
            if kwargs["title"] and not isinstance(kwargs["title"], str):
                kwargs["title"] = kwargs["title"][0]
        except (TypeError, IndexError) as exc:
            raise TypeError("title must be a string or an indexable sequence") from exc

        try:
            if kwargs["xlabel"] and not isinstance(kwargs["xlabel"], str):
                kwargs["xlabel"] = kwargs["xlabel"][-1]
        except (TypeError, IndexError) as exc:
            raise TypeError("xlabel must be a string or an indexable sequence") from exc

        ax = list(ax)

        if len(ax) == 1:
            default_formatter = DefaultFormatter()
            ax[0] = default_formatter.apply_formatting(ax[0], ylabel=ylabels[0], **kwargs)
        else:
            top_formatter = TopAxisFormatter()
            ax[0] = top_formatter.apply_formatting(ax[0], ylabel=ylabels[0], **kwargs)

            if len(ax) > 2:
                mid_formatter = MidAxisFormatter()
                for i in range(1, len(ax) - 1):
                    ax[i] = mid_formatter.apply_formatting(ax[i], ylabel=ylabels[i], **kwargs)

            bot_formatter = BotAxisFormatter()
            ax[-1] = bot_formatter.apply_formatting(ax[-1], ylabel=ylabels[-1], **kwargs)

        return ax


class TwoAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs) -> Any:
        ax1, ax2 = ax

        kwargs["x_power_lim"] = None

        kwargs["set_ylabel"] = True
        kwargs["set_title"] = True
        ax1 = self.apply_base_formatting(ax1, **kwargs)

        kwargs["set_xlabel"] = True
        kwargs["set_ylabel"] = True
        ax2 = self.apply_base_formatting(ax2, **kwargs)

        return ax1, ax2


@log_exceptions
class LogScaler(AbstractScaler):
    @log_exceptions
    def calc_lims(self, arr, **kwargs) -> tuple[float, float]:
        """Applies log scaling to the specified axis of the given Axes object."""
        pad = abs(kwargs.get("pad", 0.2))
        invert_threshold = kwargs.get("invert_threshold", 0.95)  # Threshold for inversion
        invert_threshold = invert_threshold if invert_threshold <= 1 else invert_threshold / 100

        if isinstance(arr, Axes):
            arr = get_plot_data(arr, self.axis)
        else:
            arr = np.asarray(arr)

        inv = 1
        # Check if more than the threshold proportion of values are negative
        if (arr < 0).mean() > invert_threshold:
            inv = -1
        arr = self.filter_outliers(inv * arr, kwargs.get("quantile", 5))
        lims = (
            inv * 10 ** np.floor(np.log10(arr[arr > 0].min()) - pad),
            inv * 10 ** np.ceil(np.log10(arr[arr > 0].max()) + pad),
        )
        return min(lims), max(lims)

    @log_exceptions
    def scale(self, ax, arr, **kwargs) -> Any:
        """Applies log scaling to the specified axis of the given Axes object."""
        scale, _, lim = self.get_scale_functions(ax, self.axis)

        lims = self.calc_lims(ax if arr is None else arr, **kwargs)

        scale_str = "log"
        if any(limit < 0 for limit in lims):
            if not kwargs.get("allow_invert", False):
                # Bypass scaling if negative count above threshold
                return ax
            scale_str = "symlog"

        scale(scale_str)
        lim(lims)
        return ax


class LinFrom0Scaler(AbstractScaler):
    @log_exceptions
    def calc_lims(self, arr, **kwargs) -> tuple[float, float]:
        """Applies log scaling to the specified axis of the given Axes object."""
        pad = kwargs.get("pad", 0.2)
        digits = kwargs.get("digits", 2)
        invert_threshold = kwargs.get("invert_threshold", 0.5)  # Threshold for inversion
        invert_threshold = invert_threshold if invert_threshold <= 1 else invert_threshold / 100

        if isinstance(arr, Axes):
            arr = get_plot_data(arr, self.axis)
        else:
            arr = np.asarray(arr)

        arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

        inv = 1
        if (arr < 0).mean() > invert_threshold:
            inv = -1

        lims = [
            0.0,
            inv * sig_figs_ceil((inv * arr).max() * (1 + pad), digits),
        ]
        return min(lims), max(lims)

    @log_exceptions
    def scale(self, ax, arr, **kwargs) -> Any:
        """Applies linear scaling to the specified axis of the given Axes object."""
        scale, _, lim = self.get_scale_functions(ax, self.axis)

        scale("linear")
        lim(self.calc_lims(ax if arr is None else arr, **kwargs))
        return ax


class LinScaler(AbstractScaler):
    @log_exceptions
    def calc_lims(self, arr, **kwargs) -> tuple[float, float]:
        """Applies log scaling to the specified axis of the given Axes object."""
        pad = kwargs.get("pad", 0.2)
        digits = kwargs.get("digits", 2)

        if isinstance(arr, Axes):
            arr = get_plot_data(arr, self.axis)
        else:
            arr = np.asarray(arr)

        arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

        lims = [
            sig_figs_ceil((arr).max() * (1 + pad), digits),
            sig_figs_ceil((arr).max() * (1 - pad), digits),
            -1 * sig_figs_ceil((-1 * arr).max() * (1 + pad), digits),
            -1 * sig_figs_ceil((-1 * arr).max() * (1 - pad), digits),
        ]

        return min(lims), max(lims)

    @log_exceptions
    def scale(self, ax, arr, **kwargs) -> Any:
        """Applies linear scaling to the specified axis of the given Axes object."""
        scale, _, lim = self.get_scale_functions(ax, self.axis)

        scale("linear")
        lim(self.calc_lims(ax if arr is None else arr, **kwargs))
        return ax


class DegScaler(AbstractScaler):
    @log_exceptions
    def calc_lims(self, arr, **kwargs) -> tuple[float, float]:
        """Applies log scaling to the specified axis of the given Axes object."""
        return -100, 100

    @log_exceptions
    def scale(self, ax, arr, **kwargs) -> Any:
        """Applies linear scaling to the specified axis of the given Axes object."""
        base = kwargs.get("base", 30)

        _, ticks, lim = self.get_scale_functions(ax, self.axis)

        ticks(np.arange(-90 - base, 90 + base, base))
        lim(self.calc_lims(ax if arr is None else arr, **kwargs))
        return ax


class DegFocusedScaler(AbstractScaler):
    @log_exceptions
    def calc_lims(self, arr, **kwargs) -> tuple[float, float]:
        """Applies log scaling to the specified axis of the given Axes object."""
        pad = kwargs.get("pad", 0.2)
        base = kwargs.get("base", 30)

        if isinstance(arr, Axes):
            arr = get_plot_data(arr, self.axis)
        elif isinstance(arr, (tuple, list)):
            arr = np.asarray(arr)

        if isinstance(arr, np.ndarray):
            arr = self.filter_outliers(arr, kwargs.get("quantile", 5))
            tmin = float(np.floor(arr.min() / base) * base)
            tmax = float(np.ceil(arr.max() / base) * base)
        else:
            tmin, tmax = -120, 120

        # ticks(np.arange(tmin - base, tmax + base, base))
        return tmin - base * pad, tmax + base * pad

    @log_exceptions
    def scale(self, ax, arr, **kwargs) -> Any:
        """Applies linear scaling to the specified axis of the given Axes object."""
        pad = kwargs.get("pad", 0.2)
        base = kwargs.get("base", 30)

        _, ticks, lim = self.get_scale_functions(ax, self.axis)

        lims = self.calc_lims(ax if arr is None else arr, **kwargs)

        # Tick range set to (tmin - base, tmax + base). Math extracts tmin/tmax from lims.
        ticks(np.arange(lims[0] + base * (pad - 1), lims[1] + base * (1 - pad), base))
        lim(lims)
        return ax


# class LogScaler(AbstractScaler):
#     @log_exceptions
#     def scale(self, ax, arr, **kwargs) -> Any:
#         """Applies log scaling to the specified axis of the given Axes object."""
#         pad = abs(kwargs.get("pad", 0.2))
#         # digits = kwargs.get("digits", 2)
#         allow_invert = kwargs.get("allow_invert", False)  # Option to invert data
#         invert_threshold = kwargs.get("invert_threshold", 0.95)  # Threshold for inversion
#         invert_threshold = invert_threshold if invert_threshold <= 1 else invert_threshold / 100

#         scale, _, lim = self.get_scale_functions(ax, self.axis)

#         if isinstance(arr, (tuple, list)):
#             arr = np.asarray(arr)
#         elif arr is None:
#             arr = get_plot_data(ax, self.axis)

#         if allow_invert:
#             scale_str = "log"
#             inv = 1
#             # Check if more than the threshold proportion of values are negative
#             if (arr < 0).mean() > invert_threshold:
#                 inv = -1
#                 scale_str = "symlog"
#             arr = self.filter_outliers(inv * arr, kwargs.get("quantile", 5))

#             scale(scale_str)
#             lim(
#                 [
#                     inv * 10 ** np.floor(np.log10(arr[arr > 0].min()) - pad),
#                     inv * 10 ** np.ceil(np.log10(arr[arr > 0].max()) + pad),
#                 ]
#             )
#         else:
#             if (arr < 0).mean() > invert_threshold:
#                 # Bypass log scaling if all values are negative
#                 return ax

#             arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

#             scale("log")
#             lim(
#                 [
#                     10 ** np.floor(np.log10(arr[arr > 0].min()) - pad),
#                     10 ** np.ceil(np.log10(arr[arr > 0].max()) + pad),
#                 ]
#             )
#         return ax


# class LinFrom0Scaler(AbstractScaler):
#     @log_exceptions
#     def scale(self, ax, arr, **kwargs) -> Any:
#         """Applies linear scaling to the specified axis of the given Axes object."""
#         pad = kwargs.get("pad", 0.2)
#         digits = kwargs.get("digits", 2)

#         scale, _, lim = self.get_scale_functions(ax, self.axis)

#         if isinstance(arr, (tuple, list)):
#             arr = np.array(arr)
#         elif arr is None:
#             arr = get_plot_data(ax, self.axis)

#         arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

#         inv = 1
#         if (arr < 0).mean() > 0.5:
#             inv = -1

#         lims = [
#             0,
#             inv * sig_figs_ceil((inv * arr).max() * (1 + pad), digits),
#         ]
#         scale("linear")
#         lim([min(lims), max(lims)])
#         return ax


# class LinScaler(AbstractScaler):
#     @log_exceptions
#     def scale(self, ax, arr, **kwargs) -> Any:
#         """Applies linear scaling to the specified axis of the given Axes object."""
#         pad = kwargs.get("pad", 0.2)
#         digits = kwargs.get("digits", 2)

#         scale, _, lim = self.get_scale_functions(ax, self.axis)

#         if isinstance(arr, (tuple, list)):
#             arr = np.array(arr)
#         elif arr is None:
#             arr = get_plot_data(ax, self.axis)

#         arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

#         lims = [
#             sig_figs_ceil((arr).max() * (1 + pad), digits),
#             sig_figs_ceil((arr).max() * (1 - pad), digits),
#             -1 * sig_figs_ceil((-1 * arr).max() * (1 + pad), digits),
#             -1 * sig_figs_ceil((-1 * arr).max() * (1 - pad), digits),
#         ]
#         scale("linear")
#         lim([min(lims), max(lims)])
#         return ax


# class DegScaler(AbstractScaler):
#     @log_exceptions
#     def scale(self, ax, arr, **kwargs) -> Any:
#         """Applies linear scaling to the specified axis of the given Axes object."""
#         pad = kwargs.get("pad", 0.2)
#         base = kwargs.get("base", 30)

#         _, ticks, lim = self.get_scale_functions(ax, self.axis)

#         ticks(np.arange(-90 - base, 90 + base, base))
#         lim(-100, 100)
#         return ax


# class DegFocusedScaler(AbstractScaler):
#     @log_exceptions
#     def scale(self, ax, arr, **kwargs) -> Any:
#         """Applies linear scaling to the specified axis of the given Axes object."""
#         pad = kwargs.get("pad", 0.2)
#         base = kwargs.get("base", 30)

#         _, ticks, lim = self.get_scale_functions(ax, self.axis)

#         if isinstance(arr, (tuple, list)):
#             arr = np.array(arr)
#         elif arr is None:
#             arr = get_plot_data(ax, self.axis)

#         arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

#         tmin = -120 if arr is None else np.floor(arr.min() / base) * base
#         tmax = 120 if arr is None else np.ceil(arr.max() / base) * base

#         ticks(np.arange(tmin - base, tmax + base, base))
#         lim(tmin - base * pad, tmax + base * pad)
#         return ax


class DecadeAnnotator(AbstractAnnotator):
    def annotate(self, ax, data, **kwargs):
        """
        Annotate the plot with decade markers.
        """
        # Ensure data is a NumPy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check that rows > columns
        if data.shape[0] < data.shape[1]:
            data = data.T

        annotations = kwargs.pop("annotations", None)
        if annotations is None:
            annotations = Annotations(
                ax,
                base_angle=kwargs.get("base_angle", 10),
                mag_off=kwargs.get("mag_off", 0.1),
                min_distance=kwargs.pop("min_distance", 0.05),
            )

        # Find unique decades based on the third column (e.g., frequency)
        exponents = rounded_floats(abs(data[:, 2]), True)
        base = np.unique(10.0**exponents, return_index=False)
        annotations_indices = [np.argmin(abs(data[:, 2] - b)) for b in base]
        # annotations_indices = np.unique(np.floor(np.log10(data[:, 2])), return_index=True)[1]
        # Sort annotations by frequency
        annotations_indices = sorted(annotations_indices, key=lambda i: data[i, 2], reverse=True)

        scale_by_window = kwargs.pop("scale_by_window", True)
        window = annotations.window

        annotations.prev_kwargs(
            arrowprops=kwargs.pop("arrowprops", self.arrowprops),
            bbox=kwargs.pop("bbox", self.bbox),
            textcoords="data",
            ha="center",
            **kwargs,
        )

        annotations.clear()

        res = self.find_annotation_positions(
            data,
            annotations_indices,
            min_distance=annotations.min_distance,
            mag_off=annotations.mag_off,
            base_angle=annotations.base_angle,
            scale_by_window=scale_by_window,
            window=window,
            annotation_positions=annotations.all_xyann,
            **kwargs,
        )

        for pos, lpos, annotation in zip(*res):
            annotations.xy.append(pos)
            annotations.xyann.append(lpos)
            annotations.text.append(int(exponents[annotation]))

        annotations.annotate(ax)

        return ax, annotations


class TopNAnnotator(AbstractAnnotator):
    def annotate(self, ax, data, **kwargs):
        """
        Annotate the plot with the top N peaks in the y-values.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes to annotate.
        - data (np.ndarray): A 2D NumPy array where the second column represents y-values.
        - **kwargs: Additional keyword arguments for annotations.

        Returns:
        - ax (matplotlib.axes.Axes): The annotated axes.
        - annotations (Annotations): The annotations object.
        """
        # Ensure data is a NumPy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check that rows > columns
        if data.shape[0] < data.shape[1]:
            data = data.T

        annotations = kwargs.pop("annotations", None)
        if annotations is None:
            annotations = Annotations(
                ax,
                base_angle=kwargs.get("base_angle", 10),
                mag_off=kwargs.get("mag_off", 0.1),
                min_distance=kwargs.pop("min_distance", 0.05),
            )

        # Extract kwargs for peak finding
        peak_distance_percent = kwargs.pop("peak_distance_percent", 0.1)
        target_num_peaks = kwargs.pop("target_num_peaks", None)

        # Calculate the minimum distance between peaks in data units
        peak_distance = peak_distance_percent * (data[:, 0].max() - data[:, 0].min())

        # Find peaks in the y-values (second column)
        peaks, _ = find_peaks(data[:, 1], distance=peak_distance)

        # If a target number of peaks is specified, keep the largest peaks
        if target_num_peaks is not None and len(peaks) > target_num_peaks:
            peak_heights = data[peaks, 1]
            largest_peaks_indices = np.argsort(peak_heights)[-target_num_peaks:]
            peaks = peaks[np.sort(largest_peaks_indices)]

        annotations_indices = peaks

        scale_by_window = kwargs.pop("scale_by_window", True)
        window = annotations.window

        annotations.prev_kwargs(
            arrowprops=kwargs.pop("arrowprops", self.arrowprops),
            bbox=kwargs.pop("bbox", self.bbox),
            textcoords="data",
            ha="center",
            **kwargs,
        )

        annotations.clear()

        res = self.find_annotation_positions(
            data,
            annotations_indices,
            min_distance=annotations.min_distance,
            mag_off=annotations.mag_off,
            base_angle=annotations.base_angle,
            scale_by_window=scale_by_window,
            window=window,
            annotation_positions=annotations.all_xyann,
            **kwargs,
        )

        for pos, lpos, annotation in zip(*res):
            annotations.xy.append(pos)
            annotations.xyann.append(lpos)
            annotations.text.append("{:.1E}".format(np.log10(data[annotation, 2])))
        # fmt = "{:.%dE}" % int(prec)
        #     return fmt.format(num)
        annotations.annotate(ax)

        return ax, annotations


class CustomAnnotator(AbstractAnnotator):
    def annotate(self, ax, data, **kwargs):
        """
        Annotate the plot using a custom function.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes to annotate.
        - data (np.ndarray): A 2D NumPy array to be processed by the custom function.
        - **kwargs: Additional keyword arguments for annotations.

        Returns:
        - ax (matplotlib.axes.Axes): The annotated axes.
        - annotations (Annotations): The annotations object.
        """
        # Ensure data is a NumPy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check that rows > columns
        if data.shape[0] < data.shape[1]:
            data = data.T

        annotations = kwargs.pop("annotations", None)
        if annotations is None:
            annotations = Annotations(
                ax,
                base_angle=kwargs.get("base_angle", 10),
                mag_off=kwargs.get("mag_off", 0.1),
                min_distance=kwargs.pop("min_distance", 0.05),
            )

        # Check for a custom function in kwargs
        custom_func = kwargs.pop("custom_func", None)
        if custom_func is None:
            # Fall back to the function provided in init_args
            if not self.init_args or not callable(self.init_args[0]):
                raise ValueError(
                    "CustomAnnotator requires a callable function in init_args or kwargs['custom_func']"
                )
            custom_func = self.init_args[0]

        # Apply the custom function to determine annotations
        annotations_indices = custom_func(data)

        scale_by_window = kwargs.pop("scale_by_window", True)
        window = annotations.window

        annotations.prev_kwargs(
            arrowprops=kwargs.pop("arrowprops", self.arrowprops),
            bbox=kwargs.pop("bbox", self.bbox),
            textcoords="data",
            ha="center",
            **kwargs,
        )

        annotations.clear()

        res = self.find_annotation_positions(
            data,
            annotations_indices,
            min_distance=annotations.min_distance,
            mag_off=annotations.mag_off,
            base_angle=annotations.base_angle,
            scale_by_window=scale_by_window,
            window=window,
            annotation_positions=annotations.all_xyann,
            **kwargs,
        )

        for pos, lpos, annotation in zip(*res):
            annotations.xy.append(pos)
            annotations.xyann.append(lpos)
            annotations.text.append(f"Custom {annotation}")

        annotations.annotate(ax)

        return ax, annotations


# %% Factory
class PlotFactory:
    """Factory class to create plot objects."""

    @staticmethod
    def get_plot(plot_type):
        """Factory method to get the appropriate plot class based on the plot type."""
        if plot_type.lower() == "scatter":
            return ScatterPlot()
        elif plot_type.lower() == "line":
            return LinePlot()
        elif plot_type.lower() == "band":
            return BandPlot()
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    @staticmethod
    def get_scaler(*keys, axis="y"):
        """Factory method to get the appropriate scaler class based on the keys."""

        def create_scaler(key, axis_type):
            if "log" in key.lower():
                return LogScaler(axis_type)
            elif "lin" in key.lower() and "from" in key.lower():
                return LinFrom0Scaler(axis_type)
            elif "deg" in key.lower() and "focus" in key.lower():
                return DegFocusedScaler(axis_type)
            elif "deg" in key.lower():
                return DegScaler(axis_type)
            else:
                return LinScaler(axis_type)

        try:
            if not isinstance(axis, str):
                return [create_scaler(key, axis[num]) for num, key in enumerate(keys)]
            return [create_scaler(key, axis) for key in keys]
        except (TypeError, IndexError) as exc:
            raise TypeError(
                "axis must be a string or an indexable sequence of 'x' or 'y'"
            ) from exc

    @staticmethod
    def get_formatter(formatter_type="default"):
        """Factory method to get the appropriate formatter class based on the formatter type."""
        if formatter_type.lower() == "two_axis":
            return TwoAxisFormatter()
        elif formatter_type.lower() == "stack":
            return StackFormatter()
        elif formatter_type.lower() == "default":
            return DefaultFormatter()
        else:
            raise ValueError(f"Unknown formatter type: {formatter_type}")

    @staticmethod
    def get_annotator(labeler_type, *args, **kwargs):
        """Factory method to get the appropriate annotator class based on the labeler type."""
        if labeler_type == "top_n" or labeler_type.lower().startswith("topn"):
            return TopNAnnotator(*args, **kwargs)
        elif labeler_type == "decade":
            return DecadeAnnotator(*args, **kwargs)
        elif labeler_type == "custom":
            return CustomAnnotator(*args, **kwargs)
        else:
            raise ValueError(f"Unknown annotator type: {labeler_type}")


CURSOR_KWARGS = {
    "useblit": True,  # Use blitting for better performance
    "color": "gray",  # Gray color for the lines
    "linewidth": 0.8,  # Thin lines
    "alpha": 0.6,  # Semi-transparent
    "horizOn": True,  # Show horizontal line
    "vertOn": True,  # Show vertical line
}


class StylizedPlot:
    """Class to generate plots with various formatting and scaling options."""

    def __init__(
        self,
        ax: Any,
        formatter="stack",
        scales=None,
        annotator="decade",
        labels=None,
        init_formats=None,
        **kwargs,
    ):
        self.ax = list(ax) if isinstance(ax, (list, tuple, np.ndarray)) else [ax]
        # self.data = {}  # Attribute to store data
        self.directory = OrderedDict()
        self._cursors = []
        # self._cross_lines = []
        self._prev_scales = []

        self.count = 0
        self.init_formats = init_formats if init_formats is not None else ["formatting", "scale"]

        # self.all_annotations = {}

        self.f_kwargs = kwargs.get("f_kwargs", {})
        self.lkwargs = kwargs.get("lkwargs", {})
        self.skwargs = kwargs.get("skwargs", {})
        self.ckwargs = CURSOR_KWARGS | kwargs.get("ckwargs", {})
        # self.a_kwargs = kwargs.get("a_kwargs", {})

        self.labels = labels
        self.scales = scales
        self.kwargs = kwargs

        self.formatter = formatter
        self.annotator = annotator

        # Initialize properties from kwargs
        self.x_scale = kwargs.get("x_scale", None)
        self.y_scales = kwargs.get("y_scales", None)

        self.title = kwargs.get("title", self.f_kwargs.get("title", None))
        self.xlabel = kwargs.get("xlabel", self.f_kwargs.get("xlabel", None))
        self.ylabels = kwargs.get("ylabels", self.f_kwargs.get("ylabels", []))

    @property
    def scales(self):
        """Returns the scales for the x and y axes."""
        return [self.x_scale] + self.y_scales

    @scales.setter
    def scales(self, scales):
        if isinstance(scales, str):
            scales = [scales] * (1 + len(self.ax))
        if isinstance(scales, (tuple, list)):
            self.x_scale = scales[0]
            self.y_scales = scales[1:]

    @property
    def x_scale(self):
        """Returns the scale for the x-axis."""
        if not hasattr(self, "_xscale"):
            self.x_scale = "lin"
        return self._xscale

    @x_scale.setter
    def x_scale(self, scale):
        if isinstance(scale, (tuple, list)):
            scale = scale[0]
        if isinstance(scale, str):
            self._xscale = PlotFactory.get_scaler(scale, axis="x")[0]
        else:
            if not hasattr(self, "_xscale"):
                self._xscale = PlotFactory.get_scaler("lin", axis="x")[0]

    @property
    def y_scales(self):
        """Returns the scales for the y-axes."""
        if not hasattr(self, "_yscales"):
            self.y_scales = "lin"
        return self._yscales

    @y_scales.setter
    def y_scales(self, scales):
        if isinstance(scales, str):
            scales = [scales] * len(self.ax)
        if isinstance(scales, (tuple, list)) and all(isinstance(s, str) for s in scales):
            if len(scales) == len(self.ax):
                self._yscales = PlotFactory.get_scaler(*scales)
            elif len(scales) > len(self.ax):
                self._yscales = PlotFactory.get_scaler(*scales[1 : len(self.ax) + 1])
                self.x_scale = scales[0]
            else:
                scales = [scales[0]] * len(self.ax)
                self._yscales = PlotFactory.get_scaler(*scales)
        else:
            if not hasattr(self, "_yscales"):
                scales = ["lin"] * len(self.ax)
                self._yscales = PlotFactory.get_scaler(*scales)

    @property
    def f_kwargs(self):
        """Returns the formatting kwargs."""
        return self._f_kwargs

    @f_kwargs.setter
    def f_kwargs(self, kwargs):
        if not hasattr(self, "_f_kwargs"):
            self._f_kwargs = {}
        if kwargs is None:
            self._f_kwargs = {}
        elif isinstance(kwargs, dict):
            self.xlabel = (kwargs.pop("xlabel", self.xlabel),)
            self.ylabels = (kwargs.pop("ylabels", self.ylabels),)
            self.title = (kwargs.pop("title", self.title),)

            self._f_kwargs = {**self._f_kwargs, **kwargs.copy()}

            if self._f_kwargs.get("power_lim") is not None:
                self._f_kwargs["x_power_lim"] = (
                    self._f_kwargs["power_lim"]
                    if self._f_kwargs.get("x_power_lim") is None
                    else self._f_kwargs["x_power_lim"]
                )
                self._f_kwargs["y_power_lim"] = (
                    self._f_kwargs["power_lim"]
                    if self._f_kwargs.get("y_power_lim") is None
                    else self._f_kwargs["y_power_lim"]
                )

    @property
    def title(self):
        """Returns the title of the plot."""
        if not hasattr(self, "_title"):
            self._title = None
        return self._title

    @title.setter
    def title(self, label):
        if isinstance(label, str):
            self._title = label

    @property
    def labels(self):
        """Returns the labels for the x and y axes."""
        if self.ylabels is None:
            return None
        return [self.xlabel] + self.ylabels

    @labels.setter
    def labels(self, labels):
        if isinstance(labels, str):
            labels = [labels] * (1 + len(self.ax))
        if isinstance(labels, (tuple, list)):
            self.xlabel = labels[0]
            self.ylabels = labels[1:]

    @property
    def xlabel(self):
        """Returns the label for the x-axis."""
        if not hasattr(self, "_xlabel"):
            self._xlabel = None
        return self._xlabel

    @xlabel.setter
    def xlabel(self, label):
        if isinstance(label, (tuple, list)):
            self._xlabel = label[0]
        elif isinstance(label, str):
            self._xlabel = label

    @property
    def ylabels(self) -> list[str] | None:
        """Returns the labels for the y-axes."""
        if not hasattr(self, "_ylabels"):
            self._ylabels = None
        return self._ylabels

    @ylabels.setter
    def ylabels(self, labels):
        if isinstance(labels, (tuple, list)) and all(isinstance(lb, str) for lb in labels):
            if len(labels) == len(self.ax):
                self._ylabels = list(labels)
            elif len(labels) > len(self.ax):
                self._ylabels = list(labels[1 : len(self.ax) + 1])
                self.xlabel = labels[0]
            elif labels == []:
                self._ylabels = None
            else:
                self._ylabels = [labels[0]] * len(self.ax)
        elif isinstance(labels, str):
            self._ylabels = [labels] * len(self.ax)

    @property
    def formatter(self):
        """Returns the formatter for the plot."""
        if not hasattr(self, "_formatter"):
            self.formatter = "stack"
        return self._formatter

    @formatter.setter
    def formatter(self, formatter):
        if isinstance(formatter, str):
            self._formatter = PlotFactory.get_formatter(formatter)

    @property
    def _formating_methods(self):
        """Returns the list of formatting methods."""
        return ["scale", "formatting", "annotate", "square"]

    def formatting(self, **kwargs):
        """Apply formatting to the axes."""
        self.f_kwargs = kwargs

        form_res = self.formatter.apply_formatting(
            *self.ax,
            xlabel=self.xlabel,
            ylabel=self.ylabels,
            title=self.title,
            **self.f_kwargs,
        )
        self.ax = list(form_res) if isinstance(form_res, (list, tuple)) else [form_res]

    def get_plot_data(self, axis=None, label="", as_dict=False):
        """
        Retrieve data from the plot based on the specified axis and label.

        Args:
            ax (matplotlib.axes.Axes): The axes object to retrieve data from.
            axis (str or int): The axis to retrieve data from ('x', 'y', or None for both).
            label (str): The label to filter data from the plot.

        Returns:
            list: The data retrieved from the plot.
        """
        res_format = "dict" if as_dict else ""
        return [get_plot_data(ax, axis, label, res_format) for ax in self.ax]

    def get_plot_data_labels(self, index=None):
        """
        Retrieve data labels from the plot or a specific label by index.

        Args:
            index (int, optional): The index of the label to retrieve. If None, all labels are returned.

        Returns:
            list or str: A list of all labels if index is None, or a single label if index is provided.
        """
        all_labels = []
        for ax in self.ax:
            labels = get_plot_data(ax, None, "", "labels")
            if labels is not None:
                all_labels.extend(labels)

        if index is not None:
            try:
                return all_labels[index]  # Return the label at the specified index
            except IndexError as exc:
                raise ValueError(f"Index {index} is out of range for available labels.") from exc
        return all_labels  # Return all labels if no index is specified

    def scale(
        self,
        data: ArrayLike | list[Any] | tuple[Any, ...] | str | None = None,
        axis: str | int | None = None,
        label: str = "",
        use_prior: bool = False,
        **kwargs,
    ):
        """
        Scale the axes based on the provided data or retrieve data from the plot.

        Args:
            data (list): A list of np.ndarrays, lists, or min/max pairs for scaling.
            axis (str or int): The axis to scale ('x', 'y', or None for both).
            label (str): The label to filter data from the plot.
            **kwargs: Additional keyword arguments for scaling.

        Returns:
            None
        """
        if use_prior and self._prev_scales and len(self._prev_scales) == len(self.ax):
            for n, ax in enumerate(self.ax):
                prev = self._prev_scales[n]
                ax.set_xscale(prev["xscale"])
                ax.set_yscale(prev["yscale"])
                ax.set_xlim(prev["xlim"])
                ax.set_ylim(prev["ylim"])
            return

        if isinstance(data, str):
            if data in self.directory:
                data = self.directory[data]["data"]

        # Validate data input
        if isinstance(data, (tuple, list)) and len(data) == 1 + len(self.ax):
            processed_data = []
            for i, d in enumerate(data):
                # Convert lists to arrays and handle min/max pairs
                if isinstance(d, (tuple, list)):
                    d = np.array(d)
                if len(d) == 2:
                    d = np.linspace(d[0], d[1], num=50)
                if d.ndim == 2 and 1 in d.shape:
                    d = d.flatten()
                if d.ndim != 1:
                    processed_data = None
                    break
                if i > 0:
                    processed_data.append(np.column_stack((data[0], d)))

            data = processed_data

        if (
            not isinstance(data, (tuple, list))
            or len(data) != len(self.ax)
            or not all(isinstance(d, (np.ndarray, pd.DataFrame)) for d in data)
        ):
            data = None
        # Scale each axis
        for i, ax in enumerate(self.ax):
            if data is None:
                ax_data = get_plot_data(ax, axis, label)
            else:
                ax_data = data[i]
            if isinstance(ax_data, pd.DataFrame):
                ax_data = ax_data.to_numpy(copy=True)
            ax_data = np.asarray(ax_data)
            if axis is None or axis == "x" or (isinstance(axis, int) and axis == 0):
                self.ax[i] = self.x_scale.scale(ax, ax_data[:, 0], **{**self.skwargs, **kwargs})
            if axis is None or axis == "y" or (isinstance(axis, int) and axis > 0):
                self.ax[i] = self.y_scales[i].scale(
                    ax, ax_data[:, 1], **{**self.skwargs, **kwargs}
                )

    def square(self):
        """Set the aspect ratio to be equal and match x-ticks to y-ticks for all axes."""
        for n, ax in enumerate(self.ax):
            if ax.get_xlim()[1] < ax.get_ylim()[1]:
                self.ax[n].set_xlim(ax.get_ylim())
            else:
                self.ax[n].set_ylim(ax.get_xlim())

            self.ax[n].set_aspect(
                "equal",
                adjustable="box",
                anchor="C",
                share=True,
            )

    def clear(self):
        """Clear all plots from the axes."""
        self.count = 0
        self.directory = {}

        for ax in self.ax:
            if ax.get_xscale() == "log":
                ax.set_xscale("linear")
            if ax.get_yscale() == "log":
                ax.set_yscale("linear")
            ax.clear()

    def scale_history(self, save=True):
        """Clear all plots from the axes."""
        self._prev_scales = []
        if save:
            for ax in self.ax:
                ax_settings = {
                    "xscale": ax.get_xscale(),
                    "xlim": ax.get_xlim(),
                    "yscale": ax.get_yscale(),
                    "ylim": ax.get_ylim(),
                }
                self._prev_scales.append(ax_settings)

    def update_annotation(self, key=None, **kwargs):
        """
        Update annotations by clearing old ones and adding new ones.

        Args:
            data (pd.DataFrame or list): The data to annotate.
            key (int or str): The key for the annotations. If int, it is treated as an index for the plot label.
            cols (list): The columns to use for annotations.
            **kwargs: Additional keyword arguments for annotations.
        """
        # Handle backward compatibility for key as an integer
        if isinstance(key, int):
            # key = self.get_plot_data_labels(index=key)
            key = list(self.directory.keys())[key]

        if key is None:
            # Iterate through all keys in self.all_annotations
            all_annotations = [k for k, v in self.directory.items() if v.get("annotated", False)]
            for c_key in all_annotations:
                self.update_annotation(key=c_key, **kwargs)
            return

        if key in self.directory and self.directory[key].get("annotated", False):
            anno_list = self.directory[key]["annotations"]
            for anno in anno_list:
                _, kwargs["base_angle"], kwargs["mag_off"] = anno.prepare_update()

        self.annotate(key, **kwargs)

    def annotate(self, key=None, **kwargs):
        """
        Apply labeling to the axes.

        Args:
            data (pd.DataFrame or list): The data to annotate.
            key (int or str): The key for the annotations. If int, it is treated as an index for the plot label.
            cols (list): The columns to use for annotations.
            **kwargs: Additional keyword arguments for annotations.
        """
        if key is None:
            raise ValueError("Key must be specified for annotation.")

        if key not in self.directory:
            raise ValueError(f"Key '{key}' not found in the directory.")

        entry = self.directory[key]

        # If the annotator is a string, replace it with the corresponding annotator object
        if isinstance(entry["annotator"], str):
            entry["annotator"] = PlotFactory.get_annotator(entry["annotator"])

        annotations_list = entry.get("annotations", [])
        kwargs = {**entry.get("a_kwargs", {}), **kwargs}

        angles = []
        offs = []
        if not annotations_list:
            # for existing_key, annos in self.all_annotations.items():
            for existing_key, annos in {
                k: v.get("annotations", [])
                for k, v in self.directory.items()
                if v.get("annotated", False)
            }.items():
                if not annos:
                    continue
                angles.append(annos[0].base_angle)
                offs.append(annos[0].mag_off)
                if not annos[0].xyann:
                    annotations_list = annos
                    key = existing_key
                    break
            else:
                base_angle = angles[-1] if angles else kwargs.pop("base_angle", 10)
                mag_off = offs[-1] if offs else kwargs.pop("mag_off", 0.1)

                annotations_list = [
                    Annotations(ax, base_angle=base_angle, mag_off=mag_off) for ax in self.ax
                ]

        base_angle = kwargs.pop("base_angle", annotations_list[0].base_angle)
        mag_off = kwargs.pop("mag_off", annotations_list[0].mag_off)
        min_distance = kwargs.pop("min_distance", annotations_list[0].min_distance)

        for annos in annotations_list:
            annos.min_distance = min_distance
            if key is None:
                annos.base_angle = base_angle
                annos.mag_off = mag_off
                if base_angle in angles and mag_off in offs:
                    annos.shift_location()

        new_annotations_list = []
        for ax, d, anno in zip(self.ax, entry["data"], annotations_list):
            if isinstance(d, pd.DataFrame):
                d = d.to_numpy(copy=True)
            if isinstance(d, np.ndarray) and d.shape[0] < d.shape[1]:
                d = d.T

            # Annotate the axis
            ax, new_annotations = entry["annotator"].annotate(
                ax, d[:, :3].copy(), annotations=anno, **kwargs.copy()
            )
            new_annotations_list.append(new_annotations)

        # Store the new annotations in the directory
        entry["annotations"] = new_annotations_list
        entry["a_kwargs"] = kwargs

        self.directory[key] = entry

    def prepare_data(self, data, y_int=1):
        """
        Prepare a list of reordered DataFrames for plotting.

        Args:
            data (pd.DataFrame or list): The input data to prepare.
            y_int (int): The interval for y-columns (default is 1).

        Returns:
            list: A list of reordered DataFrames, one for each axis.
        """
        if isinstance(data, list):
            # If data is already a list, return it as is
            return data

        keys = data.columns.to_list()
        reordered_data = []

        for n in range(len(self.ax)):
            reordered_keys = (
                [keys[0]]
                + keys[1:][(n * y_int) % (len(keys) - 1) :]
                + keys[1:][: (n * y_int) % (len(keys) - 1)]
            )
            reordered_data.append(data[reordered_keys].copy())

        return reordered_data

    def update_data(self, key, data):
        """
        Update the plot and directory with new data.

        Args:
            key (str): The key identifying the plot in the directory.
            data (pd.DataFrame or list): The new data to update the plot with.
        """
        if isinstance(key, (tuple, list)):
            if not isinstance(data, (tuple, list)) or isinstance(data[0], pd.DataFrame):
                data = [data] * len(key)
            for k, d in zip(key, data):
                self.update_data(k, d)
        # Parse the data using prepare_data
        parsed_data = self.prepare_data(data, self.directory[key]["y_dims"])

        # Ensure the key exists in the directory
        if key not in self.directory:
            raise KeyError(f"Key '{key}' not found in the directory.")

        for p_df, df in zip(parsed_data, self.directory[key]["data"]):
            if not isinstance(p_df, pd.DataFrame):
                raise ValueError("Each parsed data item must be a DataFrame.")
            if p_df.shape != df.shape:
                raise ValueError(
                    f"Parsed data shape {p_df.shape} does not match existing data shape {df.shape}."
                )
        # Update the directory with the new data
        self.directory[key]["data"] = parsed_data

        # # Update the plot using set_xdata and set_ydata
        # for ax, df in zip(self.ax, parsed_data):
        #     if not ax.lines:
        #         raise ValueError(f"No lines found in the axes for key '{key}'.")
        #     line = ax.lines[0]  # Assuming the first line corresponds to the plot

        #     line.set_xdata(df.iloc[:, 0])  # Set x-data using the first column
        #     line.set_ydata(df.iloc[:, 1:self.directory[key]["y_dims"] + 1])  # Set y-data using the second column

        # # Redraw the updated plots
        # for ax in self.directory[key]["ax"]:
        #     ax.figure.canvas.draw_idle()

    def plot(self, plot_type, data, add_to_annotations=False, exclude_from_all=False, **kwargs):
        """Add a plot to the existing axes."""
        # keys=None, cols=None, rescale=True,
        if not isinstance(plot_type, str):
            plot_type = "scatter"

        plot = PlotFactory.get_plot(plot_type)

        if isinstance(data, list) and len(data) >= len(self.ax):
            data = [d if isinstance(d, pd.DataFrame) else pd.DataFrame(d) for d in data]
        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        y_int = 1
        if any(m_col in plot.primary_kwargs for m_col in ["x2", "y2"]):
            y_int = 2

        # ordered, keys = self.filter_data(data, keys, cols, y_int, kwargs.get("styling", "b"))
        key = kwargs.get("label", f"plot_{len(self.directory) + 1}")
        a_kwargs = kwargs.pop("a_kwargs", {})

        kwargs["label"] = key
        name = kwargs.pop("name", key)
        # Store metadata in the directory
        self.directory[name] = {
            "data": [],  # the data of the figure
            "plot_type": plot_type,  # the type of plot
            # "ax": [], # the axes of the figure
            "y_dims": y_int,  # the number of y dimensions
            "kwargs": kwargs,
            "annotated": add_to_annotations,
            "annotator": kwargs.pop("annotator", a_kwargs.pop("annotator", "decade")),
            "annotations": [],
            "a_kwargs": a_kwargs,
            "attrs": kwargs.pop("attrs", {}),
        }

        # if "All" not in self.directory:
        #     self.directory["All"] = {
        #         "data": [],
        #         "plot_type": [plot_type],
        #     }
        # else:
        #     self.directory["All"]["plot_type"].append(plot_type)

        data = self.prepare_data(data, y_int=y_int)

        for n, (ax, df) in enumerate(zip(self.ax, data)):
            self.ax[n] = plot.plot(
                ax,
                df,
                df.columns.to_list(),
                **kwargs,
            )
            # self.data[key].append(df.copy())
            self.directory[name]["data"].append(df.copy())
            # self.directory[name]["ax"].append(ax)

            if "All" in self.directory and not exclude_from_all:
                self.directory["All"]["data"][n] = pd.concat(
                    [self.directory["All"]["data"][n], df.copy()],
                    ignore_index=True,
                )

        if "All" not in self.directory and not exclude_from_all:
            self.directory["All"] = {
                "data": [df.copy() for df in data],
                "plot_type": [plot_type],
            }
        elif not exclude_from_all:
            self.directory["All"]["plot_type"].append(plot_type)

        if self.count == 0:
            # self.apply_formats(self.init_formats)
            if "format" in self.init_formats or "formatting" in self.init_formats:
                self.formatting()
            if "scale" in self.init_formats:
                self.scale()
            if "square" in self.init_formats:
                self.square()

        self.count += 1

    def apply_formats(self, formats=None):
        """Apply formatting to the axes."""
        if formats is None:
            formats = self.init_formats

        for method in formats:
            if "format" in method:
                self.formatting()
            elif "scale" in method:
                self.scale()
            elif "square" in method:
                self.square()

    def toggle_crosshair(self, visible=False):
        """
        Toggle crosshair cursor visibility on all axes.

        Args:
            visible (bool, optional): If provided, explicitly set crosshair visibility.
                                    If None, toggle the current visibility state.

        Returns:
            bool: The new visibility state.
        """
        # if "line" in cross_type.lower():
        #     if not self._cross_lines or len(self._cross_lines) != len(self.ax):
        #         self._cross_lines = []
        #         for ax in self.ax:
        #             vline = ax.axvline(color="gray", lw=0.8, alpha=0.6, visible=False)
        #             hline = ax.axhline(color="gray", lw=0.8, alpha=0.6, visible=False)
        #             self._cross_lines.append((vline, hline))

        #     for vline, hline in self._cross_lines:
        #         vline.set_visible(visible)
        #         hline.set_visible(visible)
        #         # Make sure to draw the figure containing the axis
        #         # vline.figure.canvas.draw_idle()
        # else:
        if not self._cursors or len(self._cursors) != len(self.ax):
            self._cursors = [Cursor(ax, **self.ckwargs) for ax in self.ax]

        # Create or update cursors for each axis
        for i, ax in enumerate(self.ax):
            # Set visibility
            self._cursors[i].visible = visible
            # Make sure to draw the figure containing the axis
            # ax.figure.canvas.draw_idle()
        # return visible  # Return the new visibility state

    # def update_crosshair(self, *values):
    #     """
    #     Update crosshair cursor properties on all axes.

    #     Args:
    #         **kwargs: Properties to update for the crosshair cursors.
    #     """
    #     if self._cross_lines:
    #         for vline, hline in self._cross_lines:
    #             for vals in values:
    #                 vline.set_xdata(vals[0])
    #                 hline.set_ydata(vals[1])

    @staticmethod
    def subplots(*args, **kwargs):
        """Create subplots with the specified arguments."""
        return plt.subplots(*args, **kwargs)

    @staticmethod
    def get_cmap(name=None, lut=None):
        """Get a colormap by name or LUT."""
        return plt.get_cmap(name=name, lut=lut)

    @staticmethod
    def get_line(ax):
        """Get all line objects from the axes."""
        return [line for line in ax.lines]

    @staticmethod
    def get_scatter(ax):
        """Get all scatter objects from the axes."""
        return [scatter for scatter in ax.collections if isinstance(scatter, PathCollection)]

    @staticmethod
    def get_fill(ax):
        """Get all fill objects from the axes."""
        return [
            collection for collection in ax.collections if isinstance(collection, PolyCollection)
        ]

    @staticmethod
    def LogNorm(**kwargs):
        """Generate a logarithmic normalization."""
        return plt.matplotlib.colors.LogNorm(**kwargs)  # type: ignore

    @staticmethod
    def BoundaryNorm(boundaries, ncolors, clip=False, extend="neither"):
        """Generate a BoundaryNorm for discretizing a colormap."""
        return plt.matplotlib.colors.BoundaryNorm(  # type: ignore
            boundaries=boundaries, ncolors=ncolors, clip=clip, extend=extend
        )

    @staticmethod
    def LogCmapNorm(cmap="coolwarm", **kwargs):
        """Generate a colormap and normalization for logarithmic data."""
        return {"cmap": cmap, "norm": plt.matplotlib.colors.LogNorm(**kwargs)}  # type: ignore

    @staticmethod
    def DecadeCmapNorm(data, cmap="coolwarm", clip=True, extend="neither"):
        """ "Generate a colormap and normalization for decade-based data."""
        # Calculate the number of decades
        try:
            decade_min = np.floor(np.log10(data.min()))
            decade_max = np.ceil(np.log10(data.max()))
            num_decades = int(decade_max - decade_min)

            # Define the boundaries for each decade
            decades = np.logspace(decade_min, decade_max, num=num_decades + 1)
        except FloatingPointError:
            decade_min = np.floor(data.min())
            decade_max = np.ceil(data.max())
            num_decades = int(decade_max - decade_min)

            # Define the boundaries for each decade
            decades = np.linspace(decade_min, decade_max, num=num_decades + 1)

        # Create a colormap with a specified number of colors
        mod_cmap = plt.get_cmap(cmap, num_decades)

        # Create a BoundaryNorm to discretize the colormap
        norm = plt.matplotlib.colors.BoundaryNorm(  # type: ignore
            boundaries=decades, ncolors=num_decades, clip=clip, extend=extend
        )

        return {"cmap": mod_cmap, "norm": norm}

    @staticmethod
    def mod_cmap(cmap, mod_factor=1.0, reverse=False):
        """
        Modify a colormap by darkening or limiting its brightness range and optionally reversing it.

        Args:
            cmap (str or Colormap): The name of the colormap or a Colormap instance.
            mod_factor (float or tuple): If float, darken the colormap by scaling RGB values (0 < mod_factor <= 1).
                                        If tuple, limit brightness as (min_brightness, max_brightness) (0.0 to 1.0).
            reverse (bool): If True, reverse the colormap.

        Returns:
            ListedColormap: A modified colormap.
        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if isinstance(mod_factor, float):
            # Darken the colormap by scaling RGB values
            colors = cmap(np.linspace(0, 1, cmap.N))
            colors[:, :3] *= mod_factor  # Scale RGB values
        elif isinstance(mod_factor, tuple) and len(mod_factor) == 2:
            # Limit brightness range
            min_brightness, max_brightness = mod_factor
            colors = cmap(np.linspace(min_brightness, max_brightness, cmap.N))
        else:
            raise ValueError("mod_factor must be a float or a tuple of length 2.")

        # Reverse the colormap if requested
        if reverse:
            colors = colors[::-1]

        return mcolors.ListedColormap(colors)

    @staticmethod
    def DynamicColor(index=None, total=None, cmap="viridis", return_list=False):
        """
        Generate a dynamic color or a list of colors based on the index and total number of items.

        Args:
            index (int): The current index (0-based). Ignored if `return_list` is True.
            total (int): The total number of items.
            cmap (str): The name of the colormap to use (default: "viridis").
            return_list (bool): If True, return a list of colors for all indices.

        Returns:
            list or tuple: A list of colors if `return_list` is True, otherwise a single color.
        """

        if total is None or total <= 0:
            raise ValueError("Total must be a positive integer.")

        # Normalize the index range
        norm = Normalize(vmin=0, vmax=max(1, total - 1))
        colormap = plt.get_cmap(cmap)

        if return_list:
            # Generate a list of colors for all indices
            return [colormap(norm(i)) for i in range(total)]
        elif index is not None:
            # Return the color for the specified index
            return colormap(norm(index))
        else:
            raise ValueError("Index must be specified if return_list is False.")

    @staticmethod
    def marker_list():
        """
        Return a list of predefined markers.

        Returns:
            list: A list of marker styles.
        """
        return ["o", "v", "^", "<", ">", "8", "s", "p", "*", ".", "h", "H", "D", "d", "P", "X"]

    @staticmethod
    def uniform_cmap_list(ignore=None, reverse=False):
        """
        Return a list of uniform sequential colormaps.
        """
        sequential_cmaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
        if reverse:
            sequential_cmaps = [cmap + "_r" for cmap in sequential_cmaps]
        if isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(ignore, (tuple, list)):
            sequential_cmaps = [cmap for cmap in sequential_cmaps if cmap not in ignore]

        return sequential_cmaps

    @staticmethod
    def diverging_named_cmap_list(ignore=None, reverse=False):
        """
        Return a list of diverging colormaps.
        """
        diverging_cmaps = ["Spectral", "coolwarm", "managua", "seismic", "vanimo", "berlin"]
        if reverse:
            diverging_cmaps = [cmap + "_r" for cmap in diverging_cmaps]
        if isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(ignore, (tuple, list)):
            diverging_cmaps = [cmap for cmap in diverging_cmaps if cmap not in ignore]

        return diverging_cmaps

    @staticmethod
    def diverging_clr_cmap_list(ignore=None, reverse=False):
        """
        Return a list of diverging colormaps.
        """
        diverging_cmaps = ["PiYG", "RdYlGn", "PRGn", "RdYlBu", "BrBG", "PuOr", "RdBu", "RdGy"]
        if reverse:
            diverging_cmaps = [cmap + "_r" for cmap in diverging_cmaps]
        if isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(ignore, (tuple, list)):
            diverging_cmaps = [cmap for cmap in diverging_cmaps if cmap not in ignore]

        return diverging_cmaps

    @staticmethod
    def sequential_cmap_list(ignore=None, reverse=False):
        """
        Return a list of sequential colormaps.
        """
        sequential_cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]
        if reverse:
            sequential_cmaps = [cmap + "_r" for cmap in sequential_cmaps]
        if isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(ignore, (tuple, list)):
            sequential_cmaps = [cmap for cmap in sequential_cmaps if cmap not in ignore]

        return sequential_cmaps

    @staticmethod
    def qualitative_cmap_list(ignore=None, reverse=False):
        """
        Return a list of qualitative colormaps.
        """
        qualitative_cmaps = ["Set1", "Set2", "Set3", "Pastel1", "Pastel2"]
        if reverse:
            qualitative_cmaps = [cmap + "_r" for cmap in qualitative_cmaps]
        if isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(ignore, (tuple, list)):
            qualitative_cmaps = [cmap for cmap in qualitative_cmaps if cmap not in ignore]

        return qualitative_cmaps

    @staticmethod
    def get_cmaps(
        axes: Any, cmap_groups: str | tuple | list = "diverging_named_cmap", reverse: bool = False
    ):
        """
        Get a list of colormaps including current colormaps for each axis.
        """
        cycler_map = {
            "uniform_cmap": StylizedPlot.uniform_cmap_list,
            "diverging_named_cmap": StylizedPlot.diverging_named_cmap_list,
            "diverging_clr_cmap": StylizedPlot.diverging_clr_cmap_list,
            "sequential_cmap": StylizedPlot.sequential_cmap_list,
            "qualitative_cmap": StylizedPlot.qualitative_cmap_list,
        }
        if isinstance(cmap_groups, str):
            cmap_groups = [cmap_groups]
        if isinstance(axes, (tuple, list)):
            # save time by updating
            cln_grps = []
            for grp in cmap_groups:
                if grp in cycler_map:
                    cln_grps.append(grp)
                else:
                    for key, method in cycler_map.items():
                        if grp in key:
                            cln_grps.append(key)
                            break
            return [StylizedPlot.get_cmaps(ax, cln_grps, reverse) for ax in axes]

        cmaps = [scatter.get_cmap().name for scatter in StylizedPlot.get_scatter(axes)]

        combined_cmaps = cmaps.copy()
        for grp in cmap_groups:
            if grp in cycler_map:
                combined_cmaps += cycler_map[grp](cmaps, reverse)
            else:
                for key, method in cycler_map.items():
                    if grp in key:
                        combined_cmaps += method(cmaps, reverse)
                        break

        return combined_cmaps

    @staticmethod
    def set_style(style=None, **kwargs):
        """
        Set the style of the plot using matplotlib's style context.

        Args:
            style (str or list): The style(s) to apply. If None, use the default style.
            **kwargs: Additional keyword arguments for the style context.

        Returns:
            None
        """
        # if style is None:
        #     style = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if style is None:
            style = "ggplot"
        elif isinstance(style, str):
            plt.style.use(style)
        elif isinstance(style, (tuple, list)):
            plt.style.use(style)

        plt.rcParams.update(kwargs)


# %% Operations
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython import get_ipython  # type: ignore

    # from eis_analysis.data_treatment import ComplexSystem, ImpedanceConfidence
    from testing.rc_ckt_sim import RCCircuit
    from eis_analysis.z_system.system import ComplexSystem
    from eis_analysis.z_system.impedance_band import ImpedanceConfidence

    get_ipython().run_line_magic("matplotlib", "Qt5")  # inline # type: ignore

    data_sim = RCCircuit((-2, 6.5, 200))

    system_data = ComplexSystem(data_sim.Z_noisy, data_sim.freq, 450e-4, 25)
    system_fit = ComplexSystem(data_sim.Z, data_sim.freq, 450e-4, 25)

    # Generate confidence interval data_sim
    ci_analysis = ImpedanceConfidence(10, data_sim.true_values, std=0.20)
    ci_df = ci_analysis.gen_conf_band(
        data_sim.freq,
        num_x_points=100,
        func=data_sim.circuit_func,
        target_form=["freq", "impedance.mag", "sigma.mag"],
        thickness=450e-4,
        area=25,
    )

    # Defaults for axis labels and scales
    defaults = {
        "freq/label": "Frequency [Hz]",
        "freq/scale": "log",
        "real/label": r"Z' [$\Omega$]",
        "real/scale": "lin",
        "imag/label": r"Z'' [$\Omega$]",
        "imag/scale": "lin",
        "inv_imag/label": r"-Z'' [$\Omega$]",
        "inv_imag/scale": "lin",
        "mag/label": r"|Z| [$\Omega$]",
        "mag/scale": "log",
        "phase/label": r"$\varphi$ [deg]",
        "phase/scale": "deg",
        "inv_phase/label": r"-$\varphi$  [deg]",
        "inv_phase/scale": "deg",
    }

    # Create a figure with two subplots sharing the x-axis for Bode plot
    fig, ax1 = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    # Initialize the StylizedPlot class for Bode plot
    bode_plot = StylizedPlot(
        ax1,
        labels=pd.Series(defaults)[["freq/label", "mag/label", "phase/label"]].to_list(),
        title="Bode Plot",
        scales=["log", "log", "log"],
        init_formats=["scale", "format"],
    )

    # # Create a BoundaryNorm to discretize the colormap
    # norm = StylizedPlot.BoundaryNorm(boundaries=decades, ncolors=num_decades, clip=True)  # CHANGE: Create BoundaryNorm
    perc_cols = [col for col in list(ci_df.values())[0].columns if "%" in col]
    min_col = perc_cols[0] if perc_cols else "min"
    max_col = perc_cols[-1] if perc_cols else "max"

    # Create a Bode plot
    bode_plot.plot(
        "scatter",
        # data_df[["freq", "mag", "phase"]],
        system_data.get_df("freq", "impedance.mag", "sigma.mag"),
        color="freq",
        # **StylizedPlot.DecadeCmapNorm(data_df["freq"], "coolwarm"), #RdYlGn
        **StylizedPlot.DecadeCmapNorm(system_data["freq"], "Spectral_r"),  # RdYlGn Spectral
        # cmap=plt.get_cmap('viridis', int(np.ceil(np.log10(data_df["freq"])).m ax() - np.floor(np.log10(data_df["freq"])).min()))
    )
