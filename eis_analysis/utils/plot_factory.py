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
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PathCollection, PolyCollection

from impedance.models.circuits.fitting import wrapCircuit

class RCCircuit:
    """Class to create a test dataset for RC circuit fitting."""

    def __init__(
        self,
        freq=(-3, 6, 100),
        true_values=None,
        noise=0.01,
        guess_range=0.9,
        bounds_range=2,
    ):
        self._Z = None
        self._true_values = (
            true_values
            if true_values is not None
            else [101.56e3, 10.210e4, 142.453e-7]
        )
        self.freq = freq
        self._noise = noise  # Default noise multiplier
        self._guess_range = guess_range  # Default initial guess multiplier
        self._bounds_range = bounds_range  # Default bounds multiplier

    @property
    def model(self):
        return "R1-p(R2,C2)"

    @property
    def true_values(self):
        return self._true_values

    @true_values.setter
    def true_values(self, value):
        if len(value) != 3:
            raise ValueError("true_values must be of length 3")
        self._true_values = value
        self._Z = None  # Invalidate Z to regenerate it
        # self.Z

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            self._freq = np.logspace(value[0], value[1], num=int(value[2]))
        else:
            self._freq = value
        self._Z = None  # Invalidate Z to regenerate it
        # self.Z

    @property
    def Z(self):
        if self._Z is None:
            Z = np.array(
                np.hsplit(self.circuit_func(self._freq, *self._true_values), 2)
            ).T
            self._Z = Z[:, 0] + 1j * Z[:, 1]
        return self._Z

    @property
    def Z_noisy(self):
        if self._Z is not None:
            np.random.seed(0)
            noise_real = np.random.normal(
                0, self._noise * abs(self.Z), size=self.Z.real.shape
            )
            noise_imag = np.random.normal(
                0, self._noise * abs(self.Z), size=self.Z.imag.shape
            )
            return self.Z + noise_real + 1j * noise_imag
        return None

    @Z_noisy.setter
    def Z_noisy(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Z_noisy multiplier must be an int or float")
        self._noise = value

    @property
    def initial_guess(self):
        return [v * self._guess_range for v in self._true_values]

    @initial_guess.setter
    def initial_guess(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError(
                "Initial guess multiplier must be an int or float"
            )
        self._guess_range = value

    @property
    def bounds(self):
        lower_bounds = [v / self._bounds_range for v in self._true_values]
        upper_bounds = [v * self._bounds_range for v in self._true_values]
        return (lower_bounds, upper_bounds)

    @bounds.setter
    def bounds(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Bounds multiplier must be an int or float")
        self._bounds_range = value

    @property
    def circuit_func(self):
        try:
            return wrapCircuit(self.model, {})
        except ImportError:
            raise ImportError(
                "Please install the impedance package and import wrapCircuit using: "
                "from impedance.models.circuits.fitting import wrapCircuit"
            )

    @property
    def Z_hstack(self):
        if self._Z is None:
            return None
        return np.hstack([self.Z.real, self.Z.imag])

    @property
    def Z_noisy_hstack(self):
        if self._Z is None:
            return None
        return np.hstack([self.Z_noisy.real, self.Z_noisy.imag])

    @property
    def lsq_kwargs(self):
        return {
            # "absolute_sigma": False,
            # "check_finite": None,
            "method": "trf",
            "jac": "3-point",
            "x_scale": "jac",
            "ftol": 1e-14,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "loss": "cauchy",
            "diff_step": None,
            "tr_solver": None,
            "tr_options": {},
            "jac_sparsity": None,
            "verbose": 1,
            "max_nfev": 1e6,
        }

    @property
    def objective(self):
        circuit_func = self.circuit_func

        def minimizer(params, f, Z_data):
            Z0 = np.array(np.hsplit(circuit_func(f, *params), 2)).T
            Z_fit = np.hstack([Z0[:, 0], Z0[:, 1]])
            if len(Z_data) == len(Z_fit) / 2:
                Z_data = np.hstack([Z_data.real, Z_data.imag])
            return Z_data - Z_fit

        return minimizer

    @property
    def objective_complex(self):
        circuit_func = self.circuit_func

        def minimizer(params, freq, Z_data):
            Z0 = np.array(np.hsplit(circuit_func(freq, *params), 2)).T
            Z_fit = Z0[:, 0] + 1j * Z0[:, 1]
            Z2 = np.array(np.hsplit(Z_data, 2)).T
            Z_noisy = Z2[:, 0] + 1j * Z2[:, 1]
            return Z_noisy - Z_fit

        return minimizer

    @property
    def objective_sq(self):
        circuit_func = self.circuit_func

        def minimizer(params, f, Z_data):
            Z0 = np.array(np.hsplit(circuit_func(f, *params), 2)).T
            Z_fit = np.hstack([Z0[:, 0], Z0[:, 1]])
            if len(Z_data) == len(Z_fit) / 2:
                Z_data = np.hstack([Z_data.real, Z_data.imag])
            return (Z_data - Z_fit) ** 2

        return minimizer


def sig_figs_ceil(number, digits=3):
    """Round based on desired number of digits."""
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    root = 10 ** (int(power) - digits)
    return np.ceil(number / root) * root


def annotation_positioner(
    data, annotations, min_distance, mag_off, base_angle, **kwargs
):
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

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Get window from kwargs or derive from data
    window = kwargs.pop("window", None)
    if window is None:
        window = (
            (data.iloc[:, 0].min(), data.iloc[:, 0].max()),
            (data.iloc[:, 1].min(), data.iloc[:, 1].max()),
        )

    for anno in annotations:
        x, y = data.iloc[anno, :2]

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
                np.linalg.norm(np.array(new_position) - np.array(pos))
                > min_distance
                for pos in annotation_positions
            ):
                # Ensure the anno position is inside the window
                if (window[0][0] <= new_position[0] <= window[0][1]) and (
                    window[1][0] <= new_position[1] <= window[1][1]
                ):
                    # Perturb the anno away from the data points
                    if not any(
                        np.linalg.norm(
                            np.array(new_position) - np.array([dx, dy])
                        )
                        < min_distance
                        for dx, dy in data.iloc[:, :2].values
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


def get_plot_data(ax, axis=None, label=None):
    """Retrieve all data from the Axes object."""
    all_data = []

    # Get data from lines
    for line in ax.lines:
        if label is None or line.get_label() == label:
            xy_data = line.get_xydata()
            all_data.append(xy_data)

    # Get data from scatter plots and fill-between plots
    for collection in ax.collections:
        if isinstance(collection, PathCollection):  # Scatter plot
            if label is None or collection.get_label() == label:
                offsets = collection.get_offsets().data
                all_data.append(offsets)
        elif isinstance(collection, PolyCollection):  # Fill between
            if label is None or collection.get_label() == label:
                paths = collection.get_paths()
                for path in paths:
                    vertices = path.vertices
                    all_data.append(vertices)

    all_data = np.vstack(all_data)

    if axis is None:
        return all_data
    elif axis == "x" or axis == 0:
        return all_data[:, 0]
    elif axis == "y":
        return all_data[:, 1:]
    elif isinstance(axis, int) and axis < all_data.shape[1]:
        return all_data[:, axis]

    raise ValueError("Invalid data_type argument")


class Annotations:
    def __init__(
        self, ax, base_angle=10, mag_off=0.1, min_distance=0.05, **kwargs
    ):
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
class AbstractPlot(ABC):
    @abstractmethod
    def plot(self, ax, data, keys, **kwargs):
        pass


class AbstractFormatter(ABC):
    def __init__(self):

        self.font_props = {
            "fontname": "Arial",
            "fontsize": 18,
            "fontweight": "bold",
        }

        self.tick_props = {
            "axis": "both",
            "which": "major",
            "labelsize": 16,
            "labelfontfamily": "Arial",
        }

    def apply_base_formatting(self, ax, **kwargs):
        font_props = {**self.font_props, **kwargs.get("font_props", {})}
        tick_props = {**self.tick_props, **kwargs.get("tick_props", {})}
        x_power_lim = kwargs.pop("x_power_lim", None)
        y_power_lim = kwargs.pop("y_power_lim", None)

        if kwargs.pop("set_xlabel", False) and isinstance(
            kwargs.get("xlabel"), str
        ):
            ax.set_xlabel(kwargs.get("xlabel", ""), **font_props)
        if kwargs.pop("set_ylabel", False) and isinstance(
            kwargs.get("ylabel"), str
        ):
            ax.set_ylabel(kwargs.get("ylabel", ""), **font_props)
        if kwargs.pop("set_title", False) and isinstance(
            kwargs.get("title"), str
        ):
            ax.set_title(kwargs.get("title", ""), **font_props)

        ax.tick_params(**tick_props)
        ax.grid(True)

        if x_power_lim is not None:
            xformatter = ScalarFormatter()
            if isinstance(x_power_lim, int):
                xformatter.set_powerlimits((-x_power_lim, x_power_lim))
            elif (
                isinstance(x_power_lim, (list, tuple))
                and len(x_power_lim) == 2
            ):
                xformatter.set_powerlimits(x_power_lim)
            ax.xaxis.set_major_formatter(xformatter)

        if y_power_lim is not None:
            yformatter = ScalarFormatter()
            if isinstance(y_power_lim, int):
                yformatter.set_powerlimits((-y_power_lim, y_power_lim))
            elif (
                isinstance(y_power_lim, (list, tuple))
                and len(y_power_lim) == 2
            ):
                yformatter.set_powerlimits(y_power_lim)
            ax.yaxis.set_major_formatter(yformatter)

        plt.tight_layout()
        return ax

    @abstractmethod
    def apply_formatting(self, *ax, **kwargs):
        pass


class AbstractScaler(ABC):
    def __init__(self, axis):
        self.axis = axis

    @abstractmethod
    def scale(self, ax, arr, **kwargs):
        pass

    @staticmethod
    def get_scale_functions(ax, axis):
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

        if scale_by_window and window and min_distance < 1 and mag_off < 1:
            max_window_length = max(
                window[0][1] - window[0][0], window[1][1] - window[1][0]
            )
            min_distance *= max_window_length
            mag_off *= max_window_length

        if "annotation_positions" in kwargs:
            annotation_positions = kwargs["annotation_positions"]
            if annotation_positions and isinstance(
                annotation_positions[0], list
            ):
                kwargs["annotation_positions"] = [
                    item
                    for sublist in annotation_positions
                    for item in sublist
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
    def annotate(self, ax, data, min_distance, mag_off, base_angle):
        pass


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
        ax.scatter(
            x=keys[0],
            y=keys[1],
            c=kwargs.pop("styling", "b"),
            data=data,
            edgecolor=kwargs.pop("edgecolor", "none"),
            label=kwargs.pop("label", "_none"),
            **kwargs,
        )
        plt.tight_layout()
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
        ax.plot(
            keys[0],
            keys[1],
            kwargs.pop("styling", "k"),
            data=data,
            label=kwargs.pop("label", "_none"),
            **kwargs,
        )
        plt.tight_layout()
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
        ax.fill_between(
            x=keys[0],
            y1=keys[1],
            y2=keys[2],
            data=data,
            color=kwargs.pop("styling", "grey"),
            alpha=kwargs.pop("alpha", 0.25),
            **kwargs,
        )
        plt.tight_layout()
        return ax


class DefaultFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs):
        ax = ax[0]
        kwargs["set_xlabel"] = True
        kwargs["set_ylabel"] = True
        kwargs["set_title"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class BotAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs):
        ax = ax[0]
        kwargs["set_xlabel"] = True
        kwargs["set_ylabel"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class MidAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs):
        ax = ax[0]

        kwargs["x_power_lim"] = None

        kwargs["set_ylabel"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class TopAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs):
        ax = ax[0]

        kwargs["x_power_lim"] = None

        kwargs["set_ylabel"] = True
        kwargs["set_title"] = True
        ax = self.apply_base_formatting(ax, **kwargs)
        return ax


class StackFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs):
        try:
            ylabels = kwargs.pop("ylabel")
            if isinstance(ylabels, str) or ylabels is None:
                ylabels = [ylabels] * len(ax)
            elif len(ylabels) != len(ax):
                raise ValueError(
                    "Length of ylabels must match the number of axes"
                )
        except TypeError as exc:
            raise TypeError(
                "ylabel(s) must be a string, an indexable sequence, or None"
            ) from exc

        try:
            if kwargs["title"] and not isinstance(kwargs["title"], str):
                kwargs["title"] = kwargs["title"][0]
        except (TypeError, IndexError) as exc:
            raise TypeError(
                "title must be a string or an indexable sequence"
            ) from exc

        try:
            if kwargs["xlabel"] and not isinstance(kwargs["xlabel"], str):
                kwargs["xlabel"] = kwargs["xlabel"][-1]
        except (TypeError, IndexError) as exc:
            raise TypeError(
                "xlabel must be a string or an indexable sequence"
            ) from exc

        ax = list(ax)

        if len(ax) == 1:
            default_formatter = DefaultFormatter()
            ax[0] = default_formatter.apply_formatting(
                ax[0], ylabel=ylabels[0], **kwargs
            )
        else:
            top_formatter = TopAxisFormatter()
            ax[0] = top_formatter.apply_formatting(
                ax[0], ylabel=ylabels[0], **kwargs
            )

            if len(ax) > 2:
                mid_formatter = MidAxisFormatter()
                for i in range(1, len(ax) - 1):
                    ax[i] = mid_formatter.apply_formatting(
                        ax[i], ylabel=ylabels[i], **kwargs
                    )

            bot_formatter = BotAxisFormatter()
            ax[-1] = bot_formatter.apply_formatting(
                ax[-1], ylabel=ylabels[-1], **kwargs
            )

        return ax


class TwoAxisFormatter(AbstractFormatter):
    def apply_formatting(self, *ax, **kwargs):
        ax1, ax2 = ax

        kwargs["x_power_lim"] = None

        kwargs["set_ylabel"] = True
        kwargs["set_title"] = True
        ax1 = self.apply_base_formatting(ax1, **kwargs)

        kwargs["set_xlabel"] = True
        kwargs["set_ylabel"] = True
        ax2 = self.apply_base_formatting(ax2, **kwargs)

        return ax1, ax2


class LogScaler(AbstractScaler):
    def scale(self, ax, arr, **kwargs):
        pad = kwargs.get("pad", 0.2)
        digits = kwargs.get("digits", 2)

        scale, _, lim = self.get_scale_functions(ax, self.axis)

        if isinstance(arr, (tuple, list)):
            arr = np.array(arr)
        elif arr is None:
            arr = get_plot_data(ax, self.axis)

        arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

        scale("log")
        lim(
            [
                10 ** np.floor(np.log10(arr[arr > 0].min()) - abs(pad)),
                10 ** np.ceil(np.log10(arr[arr > 0].max()) + abs(pad)),
            ]
        )
        return ax


class LinFrom0Scaler(AbstractScaler):
    def scale(self, ax, arr, **kwargs):
        pad = kwargs.get("pad", 0.2)
        digits = kwargs.get("digits", 2)

        scale, _, lim = self.get_scale_functions(ax, self.axis)

        if isinstance(arr, (tuple, list)):
            arr = np.array(arr)
        elif arr is None:
            arr = get_plot_data(ax, self.axis)

        arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

        inv = 1
        if (arr < 0).mean() > 0.5:
            inv = -1

        scale("linear")
        lims = [
            0,
            inv * sig_figs_ceil((inv * arr).max() * (1 + pad), digits),
        ]
        scale("linear")
        lim([min(lims), max(lims)])
        return ax


class LinScaler(AbstractScaler):
    def scale(self, ax, arr, **kwargs):
        pad = kwargs.get("pad", 0.2)
        digits = kwargs.get("digits", 2)

        scale, _, lim = self.get_scale_functions(ax, self.axis)

        if isinstance(arr, (tuple, list)):
            arr = np.array(arr)
        elif arr is None:
            arr = get_plot_data(ax, self.axis)

        arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

        lims = [
            sig_figs_ceil((arr).max() * (1 + pad), digits),
            sig_figs_ceil((arr).max() * (1 - pad), digits),
            -1 * sig_figs_ceil((-1 * arr).max() * (1 + pad), digits),
            -1 * sig_figs_ceil((-1 * arr).max() * (1 - pad), digits),
        ]
        scale("linear")
        lim([min(lims), max(lims)])
        return ax


class DegScaler(AbstractScaler):
    def scale(self, ax, arr, **kwargs):
        pad = kwargs.get("pad", 0.2)
        base = kwargs.get("base", 30)

        _, ticks, lim = self.get_scale_functions(ax, self.axis)

        ticks(np.arange(-90 - base, 90 + base, base))
        lim(-100, 100)
        return ax


class DegFocusedScaler(AbstractScaler):
    def scale(self, ax, arr, **kwargs):
        pad = kwargs.get("pad", 0.2)
        base = kwargs.get("base", 30)

        _, ticks, lim = self.get_scale_functions(ax, self.axis)

        if isinstance(arr, (tuple, list)):
            arr = np.array(arr)
        elif arr is None:
            arr = get_plot_data(ax, self.axis)

        arr = self.filter_outliers(arr, kwargs.get("quantile", 5))

        tmin = -120 if arr is None else np.floor(arr.min() / base) * base
        tmax = 120 if arr is None else np.ceil(arr.max() / base) * base

        ticks(np.arange(tmin - base, tmax + base, base))
        lim(tmin - base * pad, tmax + base * pad)
        return ax


class TopNAnnotator(AbstractAnnotator):
    def annotate(self, ax, data, **kwargs):
        arrowprops = kwargs.pop("arrowprops", self.arrowprops)
        bbox = kwargs.pop("bbox", self.bbox)

        # Example logic: label the top N points based on the y-value
        annotations = data.nlargest(
            self.init_args[0], data.columns[1]
        ).index.tolist()
        old_annotation = kwargs.pop("annotation_positions", [])
        scale_by_window = kwargs.pop("scale_by_window", True)
        window = kwargs.pop("window", self.window)
        if scale_by_window and window is None:
            window = [ax.get_xlim(), ax.get_ylim()]

        res = self.find_annotation_positions(
            data,
            annotations,
            min_distance=kwargs.pop("min_distance", 0.05),
            mag_off=kwargs.pop("mag_off", 0.1),
            base_angle=kwargs.pop("base_angle", 10),
            scale_by_window=scale_by_window,
            window=window,
            annotation_positions=old_annotation,
            **kwargs,
        )
        for pos, lpos, annotation in zip(*res):
            ax.annotate(
                f"{annotation}",
                xy=pos,
                xytext=lpos,
                textcoords="data",
                ha="center",
                arrowprops=arrowprops,
                bbox=bbox,
                **kwargs,
            )
        return ax, old_annotation + res[1]


class DecadeAnnotator(AbstractAnnotator):
    def annotate(self, ax, data, **kwargs):
        annotations = kwargs.pop("annotations", None)
        if annotations is None:
            annotations = Annotations(
                ax,
                base_angle=kwargs.get("base_angle", 10),
                mag_off=kwargs.get("mag_off", 0.1),
                min_distance=kwargs.pop("min_distance", 0.05),
            )
        annotations_indices = np.unique(
            np.floor(np.log10(data.iloc[:, 2])), return_index=True
        )[1]
        annotations_indices = sorted(
            annotations_indices, key=lambda i: data.iloc[i, 2], reverse=True
        )  # Sort annotations by frequency

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
            annotations.text.append(
                int(np.floor(np.log10(data.iloc[annotation, 2])))
            )

        annotations.annotate(ax)

        # return ax, res[1], annotations if annotations else ax, res[1]
        return ax, annotations


class CustomAnnotator(AbstractAnnotator):
    def annotate(self, ax, data, **kwargs):
        arrowprops = kwargs.pop("arrowprops", self.arrowprops)
        bbox = kwargs.pop("bbox", self.bbox)

        # Apply custom logic to determine annotations
        annotations = self.init_args[0](data)
        old_annotation = kwargs.pop("annotation_positions", [])
        scale_by_window = kwargs.pop("scale_by_window", True)
        window = kwargs.pop("window", self.window)
        if scale_by_window and window is None:
            window = [ax.get_xlim(), ax.get_ylim()]
        res = self.find_annotation_positions(
            data,
            annotations,
            min_distance=kwargs.pop("min_distance", 0.05),
            mag_off=kwargs.pop("mag_off", 0.1),
            base_angle=kwargs.pop("base_angle", 10),
            scale_by_window=scale_by_window,
            window=window,
            annotation_positions=old_annotation,
            **kwargs,
        )
        for pos, lpos, annotation in zip(*res):
            ax.annotate(
                f"{annotation}",
                xy=pos,
                xytext=lpos,
                textcoords="data",
                ha="center",
                arrowprops=arrowprops,
                bbox=bbox,
                **kwargs,
            )
        return ax, old_annotation + res[1]


# %% Factory
class PlotFactory:
    @staticmethod
    def get_plot(plot_type):
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
                return [
                    create_scaler(key, axis[num])
                    for num, key in enumerate(keys)
                ]
            return [create_scaler(key, axis) for key in keys]
        except (TypeError, IndexError) as exc:
            raise TypeError(
                "axis must be a string or an indexable sequence of 'x' or 'y'"
            ) from exc

    @staticmethod
    def get_formatter(formatter_type="default"):
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
        if labeler_type == "top_n":
            return TopNAnnotator(*args, **kwargs)
        elif labeler_type == "decade":
            return DecadeAnnotator(*args, **kwargs)
        elif labeler_type == "custom":
            return CustomAnnotator(*args, **kwargs)
        else:
            raise ValueError(f"Unknown annotator type: {labeler_type}")


class GeneratePlot:
    def __init__(
        self,
        ax,
        formatter="stack",
        scales=None,
        annotator="decade",
        labels=None,
        init_formats=None,
        **kwargs,
    ):
        self.ax = ax if isinstance(ax, (list, tuple, np.ndarray)) else [ax]
        self.data = None  # Attribute to store data

        self.count = 0
        self.init_formats = (
            init_formats
            if init_formats is not None
            else ["formatting", "scale"]
        )
        self.fkwargs = kwargs.get("fkwargs", {})
        self.lkwargs = kwargs.get("lkwargs", {})
        self.annotations_arr = []

        self.labels = labels
        self.scales = scales
        self.kwargs = kwargs

        self.formatter = formatter
        self.annotator = annotator

        # Initialize properties from kwargs
        self.xscale = kwargs.get("xscale", None)
        self.yscales = kwargs.get("yscales", None)

        self.title = kwargs.get("title", self.fkwargs.get("title", None))
        self.xlabel = kwargs.get("xlabel", self.fkwargs.get("xlabel", None))
        self.ylabels = kwargs.get("ylabels", self.fkwargs.get("ylabels", []))

    @property
    def scales(self):
        return [self.xscale] + self.yscales

    @scales.setter
    def scales(self, scales):
        if isinstance(scales, str):
            scales = [scales] * (1 + len(self.ax))
        if isinstance(scales, (tuple, list)):
            self.xscale = scales[0]
            self.yscales = scales[1:]

    @property
    def xscale(self):
        if not hasattr(self, "_xscale"):
            self.xscale = "lin"
        return self._xscale

    @xscale.setter
    def xscale(self, scale):
        if isinstance(scale, (tuple, list)):
            scale = scale[0]
        if isinstance(scale, str):
            self._xscale = PlotFactory.get_scaler(scale, axis="x")[0]
        else:
            if not hasattr(self, "_xscale"):
                self._xscale = PlotFactory.get_scaler("lin", axis="x")[0]

    @property
    def yscales(self):
        if not hasattr(self, "_yscales"):
            self.yscales = "lin"
        return self._yscales

    @yscales.setter
    def yscales(self, scales):
        if isinstance(scales, str):
            scales = [scales] * len(self.ax)
        if isinstance(scales, (tuple, list)) and all(
            isinstance(s, str) for s in scales
        ):
            if len(scales) == len(self.ax):
                self._yscales = PlotFactory.get_scaler(*scales)
            elif len(scales) > len(self.ax):
                self._yscales = PlotFactory.get_scaler(
                    *scales[1 : len(self.ax) + 1]
                )
                self.xscale = scales[0]
            else:
                scales = [scales[0]] * len(self.ax)
                self._yscales = PlotFactory.get_scaler(*scales)
        else:
            if not hasattr(self, "_yscales"):
                scales = ["lin"] * len(self.ax)
                self._yscales = PlotFactory.get_scaler(*scales)

    @property
    def fkwargs(self):
        return self._fkwargs

    @fkwargs.setter
    def fkwargs(self, kwargs):
        if not hasattr(self, "_fkwargs"):
            self._fkwargs = {}
        if kwargs is None:
            self._fkwargs = {}
        elif isinstance(kwargs, dict):
            self.xlabel = (kwargs.pop("xlabel", self.xlabel),)
            self.ylabels = (kwargs.pop("ylabels", self.ylabels),)
            self.title = (kwargs.pop("title", self.title),)

            self._fkwargs = {**self._fkwargs, **kwargs.copy()}

            if self._fkwargs.get("power_lim") is not None:
                self._fkwargs["x_power_lim"] = (
                    self._fkwargs["power_lim"]
                    if self._fkwargs.get("x_power_lim") is None
                    else self._fkwargs["x_power_lim"]
                )
                self._fkwargs["y_power_lim"] = (
                    self._fkwargs["power_lim"]
                    if self._fkwargs.get("y_power_lim") is None
                    else self._fkwargs["y_power_lim"]
                )

    @property
    def title(self):
        if not hasattr(self, "_title"):
            self._title = None
        return self._title

    @title.setter
    def title(self, label):
        if isinstance(label, str):
            self._title = label

    @property
    def labels(self):
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
    def ylabels(self):
        if not hasattr(self, "_ylabels"):
            self._ylabels = None
        return self._ylabels

    @ylabels.setter
    def ylabels(self, labels):
        if isinstance(labels, (tuple, list)) and all(
            isinstance(lb, str) for lb in labels
        ):
            if len(labels) == len(self.ax):
                self._ylabels = labels
            elif len(labels) > len(self.ax):
                self._ylabels = labels[1 : len(self.ax) + 1]
                self.xlabel = labels[0]
            elif labels == []:
                self._ylabels = None
            else:
                self._ylabels = [labels[0]] * len(self.ax)
        elif isinstance(labels, str):
            self._ylabels = [labels] * len(self.ax)

    @property
    def formatter(self):
        if not hasattr(self, "_formatter"):
            self.formatter = "stack"
        return self._formatter

    @formatter.setter
    def formatter(self, formatter):
        if isinstance(formatter, str):
            self._formatter = PlotFactory.get_formatter(formatter)

    @property
    def annotator(self):
        if not hasattr(self, "_annotator"):
            self.annotator = "decade"
        return self._annotator

    @annotator.setter
    def annotator(self, annotator):
        if isinstance(annotator, str):
            self._annotator = PlotFactory.get_annotator(annotator)

    @property
    def _formating_methods(self):
        return ["scale", "formatting", "annotate", "square"]

    def formatting(self, **kwargs):
        """Apply formatting to the axes."""
        self.fkwargs = kwargs

        self.ax = self.formatter.apply_formatting(
            *self.ax,
            xlabel=self.xlabel,
            ylabel=self.ylabels,
            title=self.title,
            **self.fkwargs,
        )

    def scale(self, data=None, axis=None, label=None, **kwargs):
        if (
            not isinstance(data, list)
            or len(data) != len(self.ax)
            or not all(isinstance(d, np.ndarray) for d in data)
        ):
            data = None
        for i, ax in enumerate(self.ax):
            if data is None:
                ax_data = get_plot_data(ax, axis, label)
            else:
                ax_data = data[i]

            if axis is None or axis == "x" or axis == 0:
                self.ax[i] = self.xscale.scale(ax, ax_data[:, 0])
            if axis is None or axis == "y" or axis > 0:
                self.ax[i] = self.yscales[i].scale(ax, ax_data[:, 1])
        return

    def clear(self):
        """Clear all plots from the axes."""
        self.count = 0
        self.annotations_arr = []
        for ax in self.ax:
            if ax.get_xscale() == "log":
                ax.set_xscale("linear")
            if ax.get_yscale() == "log":
                ax.set_yscale("linear")
            ax.clear()

    def clear_annotations(self):
        """Clear annotations for the specified group of positions."""
        self.annotations_arr = []

    def update_annotation(self, data=None, index=None, cols=None, **kwargs):
        """Update annotations by clearing old ones and adding new ones."""
        if index is not None and index < len(self.annotations_arr):
            anno_list = self.annotations_arr[index]
            for anno in anno_list:
                _, kwargs["base_angle"], kwargs["mag_off"] = (
                    anno.prepare_update()
                )

        self.annotate(data, index, cols, **kwargs)

    def annotate(self, data=None, index=None, cols=None, **kwargs):
        """Apply labeling to the axes.  lkwargs=dict(min_distance=0.05, mag_off=0.1, base_angle=10)"""
        if data is None:
            data = self.data

        if isinstance(data, pd.DataFrame):
            data = [data] * len(self.ax)
        elif isinstance(data, list):
            if len(data) != len(self.ax):
                raise ValueError(
                    "Length of data list must match the number of axes."
                )
        else:
            raise TypeError(
                "Data must be a DataFrame or a list of DataFrames."
            )

        # annotations_arr = self.lkwargs.get("positions", [])
        index = (
            int(index)
            if index is not None and index < len(self.annotations_arr)
            else None
        )

        annotations_list = (
            self.annotations_arr[index] if index is not None else []
        )

        angles = []
        offs = []
        if annotations_list == []:
            for i, annos in enumerate(self.annotations_arr):
                angles.append(annos[0].base_angle)
                offs.append(annos[0].mag_off)
                if annos[0].xyann == []:
                    annotations_list = annos
                    index = i
                    break
            else:
                base_angle = (
                    angles[-1]
                    if angles != []
                    else kwargs.pop("base_angle", 10)
                )
                mag_off = (
                    offs[-1] if offs != [] else kwargs.pop("base_angle", 0.1)
                )

                annotations_list = [
                    Annotations(
                        ax,
                        base_angle=base_angle,
                        mag_off=mag_off,
                    )
                    for ax in self.ax
                ]

        # allow for passing kwargs but
        base_angle = kwargs.pop("base_angle", annotations_list[0].base_angle)
        mag_off = kwargs.pop("mag_off", annotations_list[0].mag_off)
        min_distance = kwargs.pop(
            "min_distance", annotations_list[0].min_distance
        )

        for annos in annotations_list:
            annos.min_distance = min_distance
            if (
                index is None
            ):  # if index is not None then break called, so no need to shift location
                annos.base_angle = base_angle
                annos.mag_off = mag_off
                if base_angle in angles and mag_off in offs:
                    annos.shift_location()

        new_annotations_list = []
        for ax, df, anno in zip(self.ax, data, annotations_list):
            if cols is None:
                cols = list(df.columns)[:3]

            # Ensure axis is up to date
            anno.ax = ax

            ax, new_annotations = self.annotator.annotate(
                ax,
                df[cols].copy(),
                annotations=anno,
                **kwargs,
            )

            new_annotations_list.append(new_annotations)

        if index is None:
            self.annotations_arr.append(new_annotations_list)
        else:
            self.annotations_arr[index] = new_annotations_list

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

    def plot(
        self, plot_type, data, keys=None, cols=None, rescale=True, **kwargs
    ):
        """Add a plot to the existing axes."""
        if not isinstance(plot_type, str):
            plot_type = "scatter"

        plot = PlotFactory.get_plot(plot_type)

        y_int = 1
        if any(m_col in plot.primary_kwargs for m_col in ["x2", "y2"]):
            y_int = 2

        ordered, keys = self.filter_data(data, keys, cols, y_int)

        # Data is now a dictionary of dataframes with columns in the correct plot order
        for n, ax in enumerate(self.ax):
            try:
                self.ax[n] = plot.plot(
                    ax,
                    ordered[keys[n]],
                    list(ordered[keys[n]].columns),
                    **kwargs,
                )

            except IndexError as exc:
                raise IndexError(
                    "Length of y cols must match the number of axes"
                ) from exc
        self.data = ordered
        if self.count == 0:
            # self.apply_formats(self.init_formats)
            if (
                "format" in self.init_formats
                or "formatting" in self.init_formats
            ):
                self.formatting()
            if "scale" in self.init_formats:
                self.scale()
            if "square" in self.init_formats:
                self.square()

        self.count += 1

    def filter_data(self, data, keys=None, cols=None, y_int=1):
        """Filter data to only include items in keys and organize by cols."""
        ordered = {}
        if isinstance(data, dict):  # Ensure each dataframe is sorted by cols
            if keys is None:
                keys = list(data.keys())
            if cols is None:
                cols = list(data[keys[0]].columns)[
                    : 1 + y_int
                ]  # assumes all df are the same
            for key in keys:
                df = data[key].copy()
                ordered[key] = df[
                    cols + [col for col in df.columns if col not in cols]
                ]
        elif isinstance(
            data, pd.DataFrame
        ):  # Duplicate data, reorganizing columns for each key
            if (
                cols is None
            ):  # Should not be None but maybe cols was passed through keys
                cols = list(data.columns) if keys is None else keys
            keys = []
            df = data.copy()
            for i in range(1, len(cols), y_int):
                key = cols[i : i + y_int]
                cols_reordered = (
                    [cols[0]]
                    + key
                    + [col for col in df.columns if col not in [cols[0]] + key]
                )
                ordered[", ".join(key)] = df[cols_reordered]
                keys.append(", ".join(key))

        final_data = {k: ordered[k] for k in keys}
        return final_data, keys

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

    @staticmethod
    def subplots(*args, **kwargs):
        return plt.subplots(*args, **kwargs)

    @staticmethod
    def get_cmap(name=None, lut=None):
        return plt.get_cmap(name=name, lut=lut)

    @staticmethod
    def get_line(ax):
        return [line for line in ax.lines]

    @staticmethod
    def get_scatter(ax):
        return [
            scatter
            for scatter in ax.collections
            if isinstance(scatter, PathCollection)
        ]

    @staticmethod
    def get_fill(ax):
        return [
            collection
            for collection in ax.collections
            if isinstance(collection, PolyCollection)
        ]

    @staticmethod
    def LogNorm(**kwargs):
        return plt.matplotlib.colors.LogNorm(**kwargs)

    @staticmethod
    def BoundaryNorm(boundaries, ncolors, clip=False, extend="neither"):
        return plt.matplotlib.colors.BoundaryNorm(
            boundaries=boundaries, ncolors=ncolors, clip=clip, extend=extend
        )

    @staticmethod
    def LogCmapNorm(cmap="coolwarm", **kwargs):
        return {"cmap": cmap, "norm": plt.matplotlib.colors.LogNorm(**kwargs)}

    @staticmethod
    def DecadeCmapNorm(data, cmap="coolwarm", clip=True, extend="neither"):
        # Calculate the number of decades
        decade_min = np.floor(np.log10(data.min()))
        decade_max = np.ceil(np.log10(data.max()))
        num_decades = int(decade_max - decade_min)

        # Define the boundaries for each decade
        decades = np.logspace(decade_min, decade_max, num=num_decades + 1)

        # Create a colormap with a specified number of colors
        mod_cmap = plt.get_cmap(cmap, num_decades)

        # Create a BoundaryNorm to discretize the colormap
        norm = plt.matplotlib.colors.BoundaryNorm(
            boundaries=decades, ncolors=num_decades, clip=clip, extend=extend
        )

        return {"cmap": mod_cmap, "norm": norm}


# %% Operations
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython import get_ipython
    from eis_analysis.data_treatment import ComplexSystem, ConfidenceAnalysis

    get_ipython().run_line_magic("matplotlib", "Qt5")  # inline

    data_sim = RCCircuit((-2, 6.5, 200))

    system_data = ComplexSystem(data_sim.Z_noisy, data_sim.freq, 450e-4, 25)
    system_fit = ComplexSystem(data_sim.Z, data_sim.freq, 450e-4, 25)

    data_df = system_data.df
    data_df.insert(0, "freq", data_sim.freq)

    fit_df = system_fit.df
    fit_df.insert(0, "freq", data_sim.freq)

    # Generate confidence interval data_sim
    ci_analysis = ConfidenceAnalysis(
        10, data_sim.true_values, std=0.20, func=data_sim.circuit_func
    )
    ci_df = ci_analysis.gen_conf_band(
        data_sim.freq,
        num_freq_points=100,
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
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    # Initialize the GeneratePlot class for Bode plot
    bode_plot = GeneratePlot(
        ax,
        labels=pd.Series(defaults)[
            ["freq/label", "mag/label", "phase/label"]
        ].to_list(),
        title="Bode Plot",
        scales=["log", "log", "log"],
        init_formats=["scale", "format"],
    )

    # # Create a BoundaryNorm to discretize the colormap
    # norm = GeneratePlot.BoundaryNorm(boundaries=decades, ncolors=num_decades, clip=True)  # CHANGE: Create BoundaryNorm
    perc_cols = [col for col in list(ci_df.values())[0].columns if "%" in col]
    min_col = perc_cols[0] if perc_cols else "min"
    max_col = perc_cols[-1] if perc_cols else "max"

    # Create a Bode plot
    bode_plot.plot(
        "scatter",
        # data_df[["freq", "mag", "phase"]],
        system_data.get_custom_df("freq", "impedance.mag", "sigma.mag"),
        styling="freq",
        # **GeneratePlot.DecadeCmapNorm(data_df["freq"], "coolwarm"), #RdYlGn
        **GeneratePlot.DecadeCmapNorm(
            system_data["freq"], "Spectral_r"
        ),  # RdYlGn Spectral
        # cmap=plt.get_cmap('viridis', int(np.ceil(np.log10(data_df["freq"])).m ax() - np.floor(np.log10(data_df["freq"])).min()))
    )
    # bode_plot.plot(
    #     "line",
    #     # fit_df[["freq", "mag", "phase"]],
    #     system_fit.get_custom_df("freq", "impedance.mag", "sigma.mag"),
    #     styling="r",
    #     )
    # bode_plot.plot(
    #     "band",
    #     ci_df,
    #     ["impedance.mag", "sigma.mag"],
    #     ["freq", min_col, max_col],
    #     styling="grey",
    # )

    # # # Create a figure for Nyquist plot
    # # fig2, ax2 = plt.subplots(figsize=(6, 6))

    # # # Initialize the GeneratePlot class for Nyquist plot
    # # nyquist_plot = GeneratePlot(
    # #     ax2,
    # #     labels=[defaults["real/label"], defaults["inv_imag/label"]],
    # #     title="Nyquist Plot",
    # #     scales="LinFrom0Scaler",
    # #     init_formats=["scale", "format", "square"],
    # #     fkwargs=dict(power_lim=2),
    # #     power_lim=2,
    # # )

    # # # Create a Nyquist plot
    # # nyquist_plot.plot(
    # #     "scatter",
    # #     data_df[["real", "inv_imag", "freq"]],
    # #     styling="freq",
    # #     label="_data",
    # #     **GeneratePlot.DecadeCmapNorm(data_df["freq"], "coolwarm"),
    # # )
    # # nyquist_plot.annotate(data_df[["real", "inv_imag", "freq"]])
    # # nyquist_plot.plot(
    # #     "line", fit_df[["real", "inv_imag", "freq"]], styling="r", label="_line",
    # # )
    # # nyquist_plot.annotate(fit_df[["real", "inv_imag", "freq"]])
    # # # nyquist_plot.clear_annotation()
    # # nyquist_plot.update_annotation(fit_df[["real", "inv_imag", "freq"]], index=0)
    # # # nyquist_plot.plot(
    # # #     "band",
    # # #     ci_df,
    # # #     ["nyquist"],
    # # #     ["real", "min", "max"],
    # # #     styling="grey",
    # # #     step="mid",
    # # #     label="_band",
    # # # )

    # # # line = ax2.lines[0]

    # # # line.set_xdata(line.get_xdata()/2)

    # # Show the plots
    # plt.show()

    # def scale(self, data=None, keys=None, **kwargs):
    #     """Apply scaling to the data."""
    #     if data is None:
    #         data = self.data
    #     if data is not None:
    #         if not isinstance(data, dict) or len(data) != len(self.ax):
    #             data, _ = self.filter_data(data, keys, kwargs.pop("cols", None), kwargs.pop("y_int", 1))

    #         keys_was_none = False
    #         if isinstance(keys, str):
    #             keys = [keys]
    #         elif keys is None:
    #             keys = list(data.keys())
    #             keys_was_none = True
    #         for i, ((key, df), scaler) in enumerate(
    #             zip(data.items(), self.yscales)
    #         ):
    #             cols = list(df.columns)
    #             if keys_was_none:
    #                 keys = [cols[0]] + keys if i == 0 else [cols[0]] + keys[1:]
    #             try:
    #                 if keys[0] == "x" or keys[0] == cols[0]:
    #                     self.ax[i] = self.xscale.scale(
    #                         self.ax[i], df.iloc[:, 0]
    #                     )
    #                 if keys[0] == "y" or (
    #                     key in keys and key.split(", ")[0] == cols[1]
    #                 ):
    #                     self.ax[i] = scaler.scale(self.ax[i], df.iloc[:, 1])
    #             except (TypeError, IndexError) as exc:
    #                 raise TypeError(
    #                     "keys must be a string or a list of strings"
    #                 ) from exc

    # def rescale_axes(self, data):
    #     """Rescale the axes when the scale is changed."""
    #     self.scale(data)
    #     self.formatting()

    # def update_annotation(self, data=None, cols=None, **kwargs):
    #     """Update annotations by clearing old ones and adding new ones."""
    #     index = kwargs.pop("index", None)
    #     if index is not None:
    #         self.clear_annotation(index)
    #         # CHANGE: Reuse base_angle and mag_off
    #         angle_mag_off = self.lkwargs.get("angle_mag_off", {})
    #         for key, angles in angle_mag_off.items():
    #             if index < len(angles):
    #                 kwargs["base_angle"], kwargs["mag_off"] = angles[index]
    #     self.annotate(data, cols, **kwargs)

    # def annotate(self, data=None, cols=None, **kwargs):
    #     """Apply labeling to the axes.  lkwargs=dict(min_distance=0.05, mag_off=0.1, base_angle=10)"""
    #     if data is None:
    #         data = self.data
    #     if data is not None:
    #         if not isinstance(data, dict) or len(data) != len(self.ax):
    #             data, _ = self.filter_data(data, kwargs.get("keys", None), cols, kwargs.pop("y_int", 1))

    #         offset = {}
    #         if (
    #             "base_angle" not in kwargs.keys()
    #             and "mag_off" not in kwargs.keys()
    #         ):
    #             bump = 30 if self.count != 0 else 0
    #             offset["base_angle"] = (
    #                 self.lkwargs.get("base_angle", 10) + bump
    #             )
    #             offset["mag_off"] = self.lkwargs.get("mag_off", 0.1)
    #             if offset["base_angle"] > 180:
    #                 offset["base_angle"] = 10
    #                 offset["mag_off"] *= 1.25

    #         if kwargs == {}:
    #             kwargs = self.lkwargs.copy()
    #             kwargs.pop("positions", {})
    #             kwargs.pop("angle_mag_off", {})
    #         kwargs = {**kwargs, **offset}

    #         keys = kwargs.pop("keys", list(data.keys()))
    #         # positions = {**kwargs.pop("positions", {}), **self.lkwargs.get("positions", {k:[] for k in list(data.keys())})}
    #         positions = self.lkwargs.get(
    #             "positions", {k: [] for k in list(data.keys())}
    #         ).copy()

    #         angle_mag_off = self.lkwargs.pop(
    #             "angle_mag_off", {k: [] for k in list(data.keys())}
    #         ).copy()

    #         for ax, (key, df) in zip(self.ax, data.items()):
    #             if key not in keys:
    #                 continue

    #             if cols is None:
    #                 cols = list(df.columns)[:3]

    #             _, annotation_positions = self.annotator.annotate(
    #                 ax,
    #                 df[cols].copy(),
    #                 annotation_positions=positions[key],
    #                 **kwargs,
    #             )

    #             # positions[key] = annotation_positions
    #             positions[key].append(annotation_positions)
    #             angle_mag_off[key].append((kwargs["base_angle"], kwargs["mag_off"]))

    #     self.lkwargs = {
    #         **self.lkwargs,
    #         **kwargs.copy(),
    #         "positions": positions,
    #         "angle_mag_off": angle_mag_off,  # CHANGE: Store angle_mag_off
    #     }
