# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import re
from typing import Any
from pathlib import Path
from threading import Event
from collections.abc import Callable

# from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (
    QMessageBox,
    QProgressDialog,
)

from .gui_helpers import (
    get_param_df,
    find_arc_minima,
    sort_parameters,
    validate_vals_and_band,
)
from .data_handlers import DataGenerator
from ..data_treatment import CachedColumnSelector
from ..z_system.system import ComplexSystem
from ..impedance_supplement import ImpedanceFunc, parse_parameters
from ..system_utilities.file_io import save, load_file
from ..data_treatment.data_analysis import Statistics, FittingMethods

# from ..equipment.mfia_ops import convert_mfia_df_for_fit
CommonExceptions = (
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    AttributeError,
)
# Path(__file__).parent / "settings.json"


class WorkerError(Exception):
    """Custom exception class for handling unexpected errors."""


class WorkerFunctions:
    """Mix-in class for GUI classes to handle worker functions."""

    worker: Any = None
    thread: QThread | None = None
    progress_dialog: Any = None
    # kill_operation: bool = False
    # suppress_window: bool = False
    thread_finished: bool = True
    _is_debugging: bool = False
    # worker_lock: bool = False
    cancel_event: Event = Event()

    def create_progress_dialog(
        self,
        parent,
        title="Progress",
        label_text="Processing...",
        cancel=None,
        minimum=0,
        maximum=100,
        cancel_func=None,
        **kwargs,
    ):
        """Create and return a QProgressDialog."""
        # self.kill_operation = False
        self.cancel_event.clear()
        self.progress_dialog = QProgressDialog(label_text, cancel, minimum, maximum, parent)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        if cancel:
            cancel_func = self.cancel_operation if cancel_func is None else cancel_func
            self.progress_dialog.canceled.connect(cancel_func)

        kwargs.setdefault("setMinimumWidth", 300)
        self.progress_dialog.setMinimumWidth(kwargs["setMinimumWidth"])

        self.progress_dialog.show()

    def run_in_main(
        self,
        finished_slot=None,
        error_slot=None,
        progress_slot=None,
        progress_dialog=None,
    ):
        """Helper function to run a worker in the main thread."""
        # self.kill_operation = False
        self.cancel_event.clear()
        if finished_slot is None:
            finished_slot = self.finished_default
        if error_slot is None:
            error_slot = self.on_worker_error

        self.worker.finished.connect(finished_slot)
        self.worker.error.connect(error_slot)
        self.worker.finished.connect(self.worker.deleteLater)

        # Connect progress signal if provided
        if progress_slot:
            self.worker.progress.connect(progress_slot)

        self.worker.run()

    def run_in_thread(
        self,
        finished_slot=None,
        error_slot=None,
        progress_slot=None,
        progress_dialog=None,
    ):
        """Helper function to run a worker in a separate thread with optional progress dialog."""
        # self.kill_operation = False
        self.cancel_event.clear()
        if finished_slot is None:
            finished_slot = self.finished_default
        if error_slot is None:
            error_slot = self.on_worker_error

        if self._is_debugging:
            return self.run_in_main(finished_slot, error_slot)

        thread = QThread()  # Use a local variable for the thread
        self.worker.moveToThread(thread)

        thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished_slot)
        self.worker.error.connect(error_slot)
        self.worker.finished.connect(thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Connect progress signal if provided
        if progress_slot:
            self.worker.progress.connect(progress_slot)

        # Start the thread
        thread.start()

        # Keep a reference to the thread to prevent garbage collection
        if not hasattr(self, "_threads"):
            self._threads = []
        self._threads.append(thread)

        # Clean up finished threads
        thread.finished.connect(lambda: self._threads.remove(thread))

    def update_progress(self, value, sub_value=None):
        """Update the progress bar."""

        if sub_value is not None:
            sub_val = (value / 100) * sub_value
            main_progress = int(sub_val)  # /sub_value*100
            sub_progress = (sub_val - int(sub_val)) * 100
            self.progress_dialog.setLabelText(
                f"Step {main_progress} of {sub_value}; Step Progress: {sub_progress:.2f}%\n\nTotal Progress..."
            )

        self.progress_dialog.setValue(int(value))

    def on_worker_error(self, error_message):
        """Handle errors from worker functions."""
        try:
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
        except RuntimeError:
            pass
        self.thread_finished = True

        # if not self.suppress_window:
        # self.kill_operation = False
        self.cancel_event.clear()
        # self.worker_lock = False
        QMessageBox.critical(self, "Error", f"Operation failed: {error_message}")

    def cancel_operation(self):
        """Cancel the bootstrap fit."""
        # self.kill_operation = True  # Set cancellation flag
        self.cancel_event.set()

    def finished_default(self, *_, **__):
        """Handle the completion of data I/O operations."""
        try:
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
        except RuntimeError:
            pass
        self.thread_finished = True
        # self.kill_operation = False
        self.cancel_event.clear()


def make_data_filter(f_min, f_max):
    """Returns a function to filter data based on frequency range."""

    def wrapper(data: pd.DataFrame) -> pd.DataFrame:
        return data[(data["freq"] >= f_min) & (data["freq"] <= f_max)]

    return wrapper


class ModelFuncSys:
    def __init__(
        self, model: str, constants: dict | None, thickness: float, area: float, y_forms: list[str]
    ):
        self.func = ImpedanceFunc(model, constants)
        self.system = ComplexSystem(thickness=thickness, area=area)
        # self.thickness = thickness
        # self.area = area
        self.y_forms = y_forms

    def __call__(self, x_data, *params):
        self.system.cirith_ungol(x_data, self.func(x_data, *params))
        return np.hstack([self.system[c] for c in self.y_forms])

        # sim_data = ComplexSystem(
        #     self.func(x_data, *params),
        #     x_data,
        #     thickness=self.thickness,
        #     area=self.area,
        # )
        # return np.hstack([sim_data[c] for c in self.y_forms])

    # # Optional: explicit pickling contract
    # def __getstate__(self):
    #     # Only store the minimal state needed to rebuild
    #     return {
    #         "model": self.func.model,
    #         "constants": self.func.constants,
    #         "thickness": self.thickness,
    #         "area": self.area,
    #         "y_forms": self.y_forms,
    #     }

    # def __setstate__(self, state):
    #     # Rebuild func from stored model/constants
    #     self.func = ImpedanceFunc(state["model"], state["constants"])
    #     self.thickness = state["thickness"]
    #     self.area = state["area"]
    #     self.y_forms = state["y_forms"]


def model_func_sys_wrap(
    model: str, constants: dict, thickness: int | float, area: int | float, y_forms: list[str]
):
    """Used primarily for circuitfit"""
    func = ImpedanceFunc(model, constants)

    def wrapped(x_data, *params) -> np.ndarray:
        """Wrap the circuit function."""
        sim_data = ComplexSystem(
            func(x_data, *params),
            x_data,
            thickness=thickness,
            area=area,
        )
        return np.hstack([sim_data[c] for c in y_forms])

    return wrapped


# def model_func_sys_wrap_old(
#     eq_func: Callable, thickness: int | float, area: int | float, y_forms: list[str]
# ):
#     """Used primarily for circuitfit"""

#     def wrapped(x_data, *params) -> np.ndarray:
#         """Wrap the circuit function."""
#         sim_data = eq_func(x_data, *params)
#         sim_data = ComplexSystem(
#             np.array(np.hsplit(sim_data, 2)).T,  # numpy
#             x_data,
#             thickness=thickness,
#             area=area,
#         )
#         return np.hstack([sim_data[c] for c in y_forms])

#     return wrapped


def data_parse(data, y_forms) -> np.ndarray:
    """Convert data for residual calculation."""
    y_forms = y_forms if isinstance(y_forms, (list, tuple)) else [y_forms]
    return np.hstack([data[c] for c in y_forms])


class ParamContext:
    """
    Class to encapsulate fitting parameters and data for a specific fitting operation.

    This class holds all necessary information for a fitting operation, including
    model, dataset, parameters, and options, providing clear isolation of state
    between different fitting operations.
    """

    def __init__(
        self,
        model: str,
        dataset_var: str,
        bound_range: np.ndarray,
        ref_df: pd.DataFrame,
        df: pd.DataFrame | None = None,
        params: pd.Series | None = None,
        sort_params: bool = False,
        **kwargs,
    ):
        """
        Initialize a fitting context with the necessary parameters.

        Parameters
        ----------
        model : str
            Circuit model string
        dataset_var : str
            Name/identifier of the dataset
        bound_range : np.ndarray
            Array containing lower and upper bound multipliers [low, high]
        ref_df : pd.DataFrame
            Reference DataFrame with default parameter values
        df : pd.DataFrame, optional
            DataFrame containing parameter values and constraints
        **kwargs
            Additional arguments for validate_vals_and_band function
        """
        self.model = model
        self.dataset_var = dataset_var
        self.bound_range = bound_range
        self.sort_params = sort_params
        self.ref_df = self._set_df_structure(ref_df, model=model, set_vals=True, **kwargs)

        # self.df = pd.DataFrame(index=parse_parameters(self.model))
        # self.df: pd.DataFrame = pd.DataFrame(columns=self._df_cols)
        self.df = self.ref_df.copy() if df is None or df.empty else df.copy()
        if isinstance(params, pd.Series):
            self.make_base_df(
                params,
                model=self.model,
                set_params=True,
                ref_df=self.df,
                **kwargs,
            )
        else:
            self.set_parameters(self.df, **kwargs)

    @property
    def initial_guess(self):
        """Return the initial guess values for unlocked parameters."""
        return self.df[~self.df["lock"]]["value"].to_numpy(copy=True)

    @property
    def param_names(self):
        """Return the parameter names for unlocked parameters."""
        return self.df[~self.df["lock"]].index.to_list()

    @property
    def bounds(self):
        """Return the bounds for unlocked parameters as a tuple (lower_bounds, upper_bounds)."""
        unlocked_df = self.df[~self.df["lock"]]
        return (
            unlocked_df["bnd_low"].to_numpy(copy=True),
            unlocked_df["bnd_high"].to_numpy(copy=True),
        )

    @property
    def constants(self):
        """Return the constants dictionary for locked parameters."""
        if self.df.empty or not self.df["lock"].any():
            return {}
        return self.df[self.df["lock"]]["value"].to_dict()

    # @property
    # def eq_func(self):
    #     """Return the equation function for the current model."""
    #     return wrapCircuit(self.model, self.constants)

    @property
    def _df_cols(self):
        """Return a list of all expected DataFrame columns."""
        return [
            "value",
            "std",
            "lock",
            "bnd_low",
            "bnd_high",
            "bnd_low_lock",
            "bnd_high_lock",
        ]

    def _set_df_structure(
        self, df: pd.DataFrame, model: str = "", set_vals: bool = False, **kwargs
    ):
        """
        Ensure the DataFrame has the correct shape and columns.
        """
        new_df = df.copy().astype(float)
        if model and set(params := parse_parameters(model)) != set(df.index):
            new_df = new_df.reindex(columns=self._df_cols, index=params)
        else:
            new_df = new_df.reindex(columns=self._df_cols)

        if set_vals:
            new_df = self._set_df_dtypes(new_df)
            return validate_vals_and_band(new_df, self.bound_range, **kwargs)
        return new_df

    def _set_df_dtypes(self, df: pd.DataFrame):
        """
        Ensure cols have correct dtypes and fill missing lock columns with False.
        """
        lock_cols = ["lock", "bnd_low_lock", "bnd_high_lock"]
        new_df = df.copy().astype(float)

        try:
            # Fill missing lock columns with False and convert to bool
            new_df[lock_cols] = new_df[lock_cols].fillna(0.0).astype(bool)
            return new_df
        except KeyError as ke:
            raise KeyError(f"DataFrame is missing required lock columns: {ke}") from ke

    def set_parameters(
        self, df: pd.DataFrame | None = None, overwrite_locked: bool = False, **kwargs
    ):
        """
        Set the parameter DataFrame after processing it with validate_vals_and_band.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing parameter values and constraints
        init_df : pd.DataFrame, optional
            Reference DataFrame with default values for missing columns
        **kwargs
            Additional arguments for validate_vals_and_band function

        Returns
        -------
        self : FitContext
            Returns self for method chaining
        """
        if df is None or df.empty:
            # If no DataFrame is provided, assume kwarg (band setting) update
            self.df = validate_vals_and_band(self.df, self.bound_range, **kwargs)
            return

        df = self._set_df_structure(df, **kwargs)

        df = df.fillna(self.ref_df.astype(float))

        df = self._set_df_dtypes(df)

        if overwrite_locked:
            # Overwrite locked parameter values from ref_df (before band check)
            locked_names = self.ref_df.index[
                self.ref_df["lock"] & self.ref_df.index.isin(df.index)
            ]
            df.loc[locked_names, ["value", "std"]] = self.ref_df.loc[
                locked_names, ["value", "std"]
            ].values

        # Apply validate_vals_and_band to get the final DataFrame
        self.df = validate_vals_and_band(df, self.bound_range, **kwargs)

    def make_base_df(
        self,
        params: pd.Series,
        model: str = "",
        set_params: bool = False,
        ref_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Create a parameter DataFrame from the model and params."""
        param_names = parse_parameters(model or self.model)

        # param_values = [params[f"{name}_values"] for name in param_names]
        # param_stds = [params[f"{name}_std"] for name in param_names]
        param_values = [params.get(f"{name}_values", np.nan) for name in param_names]
        param_stds = [params.get(f"{name}_std", np.nan) for name in param_names]

        # Create a DataFrame for the parameters to pass to set_parameters
        df = pd.DataFrame(
            {"value": param_values, "std": param_stds}, index=param_names, dtype=float
        )  # .dropna(how="all")

        if isinstance(ref_df, pd.DataFrame):
            df = df.fillna(ref_df.astype(float))

        if kwargs.pop("sort_params", self.sort_params):
            df = sort_parameters(df.dropna(subset=["value"]))
            df = df.reindex(index=param_names)  # restore dropped rows

        if set_params:
            self.set_parameters(df, **kwargs)
            return self.df
        return df

    def clone(
        self,
        model: str = "",
        dataset_var: str = "",
        bound_range: np.ndarray | None = None,
        ref_df: pd.DataFrame | None = None,
        df: pd.DataFrame | None = None,
        sort_params: bool | None = None,
        **kwargs,
    ) -> "ParamContext":
        """Create a copy of this FitContext."""
        new_context = ParamContext(
            model=model or self.model,
            dataset_var=dataset_var or self.dataset_var,
            bound_range=bound_range if bound_range is not None else self.bound_range,
            ref_df=ref_df if ref_df is not None else self.ref_df.copy(),
            df=df if df is not None else self.df.copy(),
            sort_params=sort_params if sort_params is not None else self.sort_params,
            **kwargs,
        )
        return new_context


class DataContext:
    """
    Class to encapsulate dataset-related information for fitting operations.

    This class manages dataset access, filtering, and model function generation,
    providing clear isolation of data handling concerns.
    """

    def __init__(
        self,
        datasets: dict,
        thickness: float,
        area: float,
        forms: list[str],
        data_filter: Callable,
        weight_by_mode: str = "",
        rs_param: str = "",
    ):
        """
        Initialize a data context with the necessary parameters.

        Parameters
        ----------
        datasets : dict
            Dictionary of ComplexSystem datasets
        dataset_var : str
            Name/identifier of the current dataset
        thickness : float
            Thickness value for the system
        area : float
            Area value for the system
        forms : list
            List of fitting forms (e.g., ['Z.real', 'Z.imag'])
        f_min : float
            Minimum frequency for filtering
        f_max : float
            Maximum frequency for filtering
        weight_by_mode : str, optional
            Mode to use for weighting data points
        error_func : callable, optional
            Error function for calculating goodness of fit
        error_name : str, optional
            Name of the error function
        loss_func : callable, optional
            Loss function for optimization
        """
        self.datasets = datasets
        self.thickness = thickness
        self.area = area
        self.forms = forms
        self.weight_by_mode = weight_by_mode
        self.data_filter = data_filter
        self.rs_param = rs_param

    def get_df(self, data_name, keys=None, filter=True) -> pd.DataFrame:
        """
        Get the filtered data based on the selected dataset.

        Parameters
        ----------
        data_name : str
            Name of the dataset to retrieve
        keys : list, optional
            List of column keys to retrieve from the dataset
        filter : bool, optional
            Whether to apply frequency filtering

        Returns
        -------
        pd.DataFrame
            DataFrame containing the requested data
        """
        if keys is None:
            keys = ["freq", "real", "imag", "e_r"]
        data = self.datasets[data_name].get_df(*keys)
        if filter:
            return self.data_filter(data)
        return data

    def get_system(
        self, data: str | pd.DataFrame | ComplexSystem, filter: bool = True
    ) -> ComplexSystem:
        """
        Get the filtered system based on the selected dataset.

        Parameters
        ----------
        data : pd.DataFrame or ComplexSystem, optional
            Data to use instead of the cached dataset
        weights : np.ndarray, optional
            Weights to use for the fit

        Returns
        -------
        tuple[ComplexSystem, np.ndarray]
            Tuple containing the complex system and weights
        """
        if isinstance(data, str):
            data = self.get_df(data, filter=filter)
        if isinstance(data, pd.DataFrame):
            data = ComplexSystem(
                data[["freq", "real", "imag"]],
                thickness=self.thickness,
                area=self.area,
            )

        return data

    def get_weights(
        self, data: ComplexSystem, weights: np.ndarray | None = None
    ) -> np.ndarray | None:
        """
        Get the filtered system based on the selected dataset.

        Parameters
        ----------
        data : pd.DataFrame or ComplexSystem, optional
            Data to use instead of the cached dataset
        weights : np.ndarray, optional
            Weights to use for the fit

        Returns
        -------
        tuple[ComplexSystem, np.ndarray]
            Tuple containing the complex system and weights
        """
        # Calculate weights based on weight_by_mode if not provided
        if self.weight_by_mode and weights is None:
            if "." in self.weight_by_mode:
                w_list = [abs(data[self.weight_by_mode])] * len(self.forms)
            else:
                w_list = [
                    abs(data[col.split(".")[0]][self.weight_by_mode])  # type: ignore
                    for col in self.forms
                ]
            weights = np.hstack([w / max(w) for w in w_list])  # type: ignore

        return weights

    def get_model_func(self, forms, model: str, constants: dict) -> Callable:
        """
        Return the model function for fitting.

        Parameters
        ----------
        forms : list[str]
            List of fitting forms (e.g., ['Z.real', 'Z.imag'])
        model : str
            Circuit model string
        constants : dict
            Dictionary of constant parameters and their values

        Returns
        -------
        callable
            Function that generates model data for given parameters
        """
        model_func = ModelFuncSys(
            model,
            constants=constants,
            thickness=self.thickness,
            area=self.area,
            y_forms=forms,
        )
        return model_func

    def rs_to_minima(self, df: pd.DataFrame, dataset_var: str):
        """Update the Rs parameter based on the arc minima if applicable."""
        if self.rs_param in df.index:
            # Call find_arc_minima and set the value of the parameter
            arc_minima = find_arc_minima(self.datasets[dataset_var].get_df("Z"))
            df.loc[self.rs_param, "value"] = arc_minima[0]
            df.loc[self.rs_param, "std"] = arc_minima[0] * 0.1
            # Lock the parameter
            df.loc[self.rs_param, "lock"] = True
        return df


class FittingWorker(QObject):
    """Worker class to handle fitting operations."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(float, object)

    def __init__(self, main: Any, iterations: int = 1, run_type: str = "fit"):
        """Initialize the worker with the main window and fitting parameters."""
        super().__init__()
        # self.main = main
        self.iterations = iterations
        self.run_type = run_type
        # self.cancelled = False
        self.auto_fit_params = main.auto_fit_params
        self._progress_count = 0
        self._progress_sub_count = None
        self.cancel_event = main.cancel_event
        self.cancel_event.clear()

        self._sequential_fit = main.options["fit"].get("sequential_fit", False)
        self._sequential_fit_update = main.options["fit"].get("sequential_fit_update", False)

        prime_df = get_param_df(main.parameters, main.parameters_std, main.bounds)
        rs_param = main.options["fit"].get("Rs", "")
        if rs_param in prime_df.index and prime_df.loc[rs_param, "lock"]:
            rs_param = ""

        modes = (
            main.options["fit"]["modes"]
            if isinstance(main.options["fit"]["modes"], (list, tuple))
            else [main.options["fit"]["modes"]]
        )
        raw_forms = (
            main.options["fit"]["type"]
            if isinstance(main.options["fit"]["type"], (list, tuple))
            else [main.options["fit"]["type"]]
        )
        if not raw_forms:
            raw_forms = ["Z"]

        forms = []
        for rform in raw_forms:
            if "." in rform:
                forms.append(rform)
            else:
                forms.extend([rform + "." + mode for mode in modes])

        # if isinstance(main.options["fit"]["type"], (list, tuple)):
        #     forms: list[str] = [
        #         mode if "." in mode else fit_type + "." + mode
        #         for fit_type in main.options["fit"]["type"]
        #         for mode in modes
        #     ]
        # else:
        #     forms: list[str] = [
        #         mode if "." in mode else main.options["fit"]["type"] + "." + mode for mode in modes
        #     ]

        self.data = DataContext(
            datasets=main.data.raw.copy(),
            thickness=main.options["simulation"]["thickness"],
            area=main.options["simulation"]["area"],
            forms=forms,
            data_filter=make_data_filter(
                main.options["fit"]["f_min"], main.options["fit"]["f_max"]
            ),
            weight_by_mode=main.options["fit"]["weight_by_mode"],
            rs_param=rs_param,
        )

        # Check if filtered_data is empty self.data.get_df
        # if self.get_data_df(main.data.var.currentText()).empty:
        if self.data.get_df(main.data.var.currentText()).empty:
            QMessageBox.warning(
                main.root,
                "Warning",
                "No data points within the specified frequency range.",
            )
            return

        prime_df = self.data.rs_to_minima(prime_df, main.data.var.currentText())
        self.prime = ParamContext(
            model=main.settings.model,
            dataset_var=main.data.var.currentText(),
            bound_range=np.array((main.quick_bound_vals["low"], main.quick_bound_vals["high"])),
            ref_df=prime_df,
            df=prime_df,
            sort_params=main.options["fit"]["resort_by_RC"],
            shift_value=main.options["fit"]["prioritize_bounds"],
            shift_band=not main.options["fit"]["prioritize_bounds"],
            rebound_unlocked=main.options["fit"]["rebound_before_fit"],
        )

        self.active = self.prime.clone()
        options = main.options.as_dict()
        self.options = {}
        for key in ("fit", "curve_fit", "diff_evolution", "basinhopping", "least_sq"):
            self.options[key] = options[key].copy()

        self.loss_func = None
        # error_method, self.error_name = main.get_error_name()
        error_method = main.settings.error_methods[main.error_var.currentText()]
        self.error_name = main.settings.error_methods_abbr[main.error_var.currentText()]

        self.error_func: Callable = Statistics()[error_method]
        if "sq" in self.options["fit"]["function"]:
            self.loss_func = Statistics().as_array(error_method)
        else:
            self.loss_func = Statistics()[error_method]

    def get_error_res(self, fit_result):
        dataset = self.data.get_system(self.active.dataset_var, filter=False)
        # model_func = self.data.get_model_func(["Z.real", "Z.imag"], self.active.eq_func)
        model_func = self.data.get_model_func(
            ["Z.real", "Z.imag"], self.active.model, self.active.constants
        )
        model_data = model_func(dataset["freq"], *fit_result.loc[self.active.param_names, "value"])
        error = self.error_func(model_data, data_parse(dataset, ["Z.real", "Z.imag"]))

        if abs(np.log10(abs(error))) > 2:
            error_printout = f"{self.error_name}: {error:.4e}"
        else:
            error_printout = f"{self.error_name}: {error:.4f}"
        return error_printout

    @pyqtSlot()
    def run(self):
        """Run the fitting operation."""
        try:
            if "boot" in self.run_type:
                self.perform_bootstrap_fit()
            elif "iter" in self.run_type:
                self.perform_iterative_fit()
            elif "auto" in self.run_type:
                self.perform_auto_fit()
            else:
                # convert data to system
                fit_results = self.perform_fit()
                if not self.cancel_event.is_set():  # Check if cancelled
                    self.finished.emit(fit_results)
        except WorkerError as exc:
            self.error.emit(str(exc))
            self.finished.emit(None)

    def perform_fit(self, data=None, weights=None) -> pd.DataFrame:
        """Perform a single round of fitting."""
        try:
            if not isinstance(data, (str, pd.DataFrame, ComplexSystem)):
                data = self.active.dataset_var
            data = self.data.get_system(data)
            weights = self.data.get_weights(data, weights)
            # Get x and y data
            x_data = data["freq"]
            y_data = np.hstack([data[c] for c in self.data.forms])

            # Define the inputs dictionary
            inputs = dict(
                x_data=x_data,
                y_data=y_data,
                initial_guess=self.active.initial_guess,
                # model_func=self.data.get_model_func(self.data.forms, self.active.eq_func),
                model_func=self.data.get_model_func(
                    self.data.forms, self.active.model, self.active.constants
                ),
                bounds=self.active.bounds,
                weights=weights,
                scale=self.options["fit"]["scale"],
                loss_func=self.loss_func,
                # kill_operation=lambda: self.main.kill_operation,  # Pass kill_operation
                kill_operation=self.cancel_event.is_set,  # Pass cancel_event
            )
        except CommonExceptions as exc:
            raise WorkerError(
                f"{exc.__class__.__name__} occurred in FittingWorker.perform_fit when compiling the inputs.\n{str(exc)}"
            ) from exc

        # Check if sequential fitting is enabled
        if self._sequential_fit:
            fit_results, std_results = self._perform_sequential_fit(inputs)
        else:
            fit_results, std_results = self._perform_fit(inputs)

        try:
            # Process the results
            if fit_results is None:
                return pd.DataFrame()

            fit_df = pd.DataFrame(
                np.array([fit_results, std_results]).T,
                index=self.active.param_names,
                columns=["value", "std"],
                dtype=float,
            )

            res_df = pd.concat(
                [fit_df, self.active.df.loc[list(self.active.constants.keys()), ["value", "std"]]]
            ).loc[self.active.df.index]

            if self.active.sort_params:
                res_df = sort_parameters(res_df)

            res_df.attrs["model"] = self.active.model
            res_df.attrs["dataset_var"] = self.active.dataset_var
            res_df.attrs["area"] = self.data.area
            res_df.attrs["thickness"] = self.data.thickness
            res_df.attrs["error"] = self.get_error_res(res_df)

            return res_df
        except CommonExceptions as exc:
            raise WorkerError(
                f"{exc.__class__.__name__} occurred in FittingWorker.perform_fit.\n{str(exc)}"
            ) from exc

    def _perform_fit(self, inputs):
        """Core fitting logic."""
        try:
            # Perform the fitting based on the selected method
            if "circuit" in self.options["fit"]["function"]:
                inputs.pop("loss_func")
                fit_results, std_results = FittingMethods().circuit_fit(
                    **{**inputs, **self.options["curve_fit"]},
                )
            elif "diff" in self.options["fit"]["function"]:
                fit_results, std_results = FittingMethods().de_fit(
                    **{**inputs, **self.options["diff_evolution"]},
                )
            elif "basin" in self.options["fit"]["function"]:
                fit_results, std_results = FittingMethods().basin_fit(
                    **{**inputs, **self.options["basinhopping"]},
                )
            else:
                fit_results, std_results = FittingMethods().ls_fit(
                    **{**inputs, **self.options["least_sq"]},
                )
            return fit_results, std_results
        except CommonExceptions as exc:
            raise WorkerError(
                f"{exc.__class__.__name__} occurred in FittingWorker._perform_fit while fitting the data.\n{str(exc)}"
            ) from exc

    def _perform_sequential_fit(self, inputs):
        """Perform sequential fitting by iterating over parameters."""
        try:
            # Initialize arrays for fit results and standard deviations copying self.initial_guess
            fit_results = self.active.initial_guess
            std_results = fit_results * 0.1

            # Copy the inputs dictionary for this parameter
            seq_inputs = inputs.copy()

            constants = {
                **self.active.constants,
                **{k: float(v) for k, v in zip(self.active.param_names, fit_results)},
            }

            # Iterate over each parameter
            for i, param in enumerate(self.active.param_names):
                # Update initial_guess and bounds for the current parameter
                seq_inputs["initial_guess"] = np.array([fit_results[i]])
                seq_inputs["bounds"] = (
                    np.array([self.active.bounds[0][i]]),  # Lower bound for the current parameter
                    np.array([self.active.bounds[1][i]]),  # Upper bound for the current parameter
                )

                # Update the model function with other parameters as constants
                old_val = constants.pop(param)
                seq_inputs["model_func"] = self.data.get_model_func(
                    self.data.forms,
                    self.active.model,
                    constants,
                    #   wrapCircuit(self.active.model, constants)
                )

                # Perform the fit for the current parameter
                param_fit_results, param_std_results = self._perform_fit(seq_inputs)

                if param_fit_results is None:
                    param_fit_results = [fit_results[i]]
                    param_std_results = [std_results[i]]
                if param_std_results is None:
                    param_std_results = [std_results[i]]

                # Store the results
                fit_results[i] = param_fit_results[0]
                std_results[i] = param_std_results[0]

                # Update all_vals with the new result if desired
                if self._sequential_fit_update:
                    constants[param] = float(param_fit_results[0])
                else:
                    constants[param] = old_val

            return fit_results, std_results
        except CommonExceptions as exc:
            raise WorkerError(
                f"{exc.__class__.__name__} occurred during sequential fitting.\n{str(exc)}"
            ) from exc

    def _perform_iterative_fit(self, iterations=None) -> pd.DataFrame:
        """Perform an iterative fit."""
        try:
            data = self.data.get_system(self.active.dataset_var)
            weights = self.data.get_weights(data)
        except CommonExceptions as exc:
            raise WorkerError(
                f"{exc.__class__.__name__} occurred while preparing the data for iterative fitting.\n{str(exc)}"
            ) from exc

        if iterations is None:
            iterations = self.iterations

        # local_df = pd.DataFrame(self.prime_df.copy()["value"])
        local_df = pd.DataFrame(self.active.df.copy()["value"])
        fit_results = pd.DataFrame()
        for i in range(iterations):
            new_results = self.perform_fit(data, weights)
            try:
                if new_results.empty:
                    break
                else:
                    fit_results = new_results

                local_df.update(fit_results)

                # Update parameters for the next iteration.  Allows bounds and values to shift.
                self.active.set_parameters(
                    local_df.copy(), rebound_unlocked=True, shift_value=True
                )

                self.report_progress()
                # if self.main.kill_operation:  # Check if cancelled
                if self.cancel_event.is_set():  # Check if cancelled
                    break
            except CommonExceptions as exc:
                raise WorkerError(
                    f"{exc.__class__.__name__} occurred during iterative fit on level {i}.\n{str(exc)}"
                ) from exc

        return fit_results

    def perform_iterative_fit(self):
        """Perform an iterative fit."""
        self._progress_count = 0
        fit_results = self._perform_iterative_fit()
        self.finished.emit(fit_results)

    def _perform_bootstrap_fit(self, iterations=None) -> pd.DataFrame:
        """Perform a bootstrap fit."""

        if iterations is None:
            iterations = self.iterations

        fit_results = pd.DataFrame()
        bootstrap_results = []
        df = self.data.get_df(self.active.dataset_var)
        for i in range(iterations):
            try:
                data_system = self.data.get_system(df.copy().sample(frac=1, replace=True))
            except CommonExceptions as exc:
                raise WorkerError(
                    f"{exc.__class__.__name__} occurred while resampling the data.\n{str(exc)}"
                ) from exc
                # convert resampled_data to system
            fit_results = self.perform_fit(data_system)
            try:
                if fit_results.empty:
                    break
                # Collect results
                bootstrap_results.append(fit_results["value"])

                self.report_progress()
                # if self.main.kill_operation:  # Check if cancelled
                if self.cancel_event.is_set():  # Check if cancelled
                    break
            except CommonExceptions as exc:
                raise WorkerError(
                    f"{exc.__class__.__name__} occurred during bootstrap fit on level {i}.\n{str(exc)}"
                ) from exc
        try:
            bootstrap_df = pd.DataFrame(bootstrap_results, columns=self.active.param_names)
            if (
                isinstance(self.options["fit"]["bootstrap_percent"], (float, int))
                and 0 < self.options["fit"]["bootstrap_percent"] < 100
            ):
                # Use describe to calculate summary statistics
                # Retrieve the saved percent
                bootstrap_percent = self.options["fit"]["bootstrap_percent"]
                bootstrap_percent = (
                    bootstrap_percent * 100 if bootstrap_percent < 1 else bootstrap_percent
                )
                lower_percentile = (100 - bootstrap_percent) / 200
                upper_percentile = 1 - lower_percentile

                bootstrap_summary = bootstrap_df.describe(
                    percentiles=[lower_percentile, 0.5, upper_percentile]
                )

                # Calculate "std" as half the confidence interval range
                pseudo_std = (bootstrap_summary.iloc[6] - bootstrap_summary.iloc[4]) / 2

                # Create a new DataFrame with "value" and "std"
                result_df = pd.DataFrame(
                    {"value": bootstrap_summary.loc["mean"], "std": pseudo_std}
                )
            else:
                result_df = bootstrap_df.describe().loc[["mean", "std"]].T
                # rename index
                result_df.columns = ["value", "std"]

            final_df = fit_results.copy()
            final_df.update(result_df)
            # Add metadata to the result DataFrame
            final_df.attrs["model"] = self.active.model
            final_df.attrs["dataset_var"] = self.active.dataset_var
            final_df.attrs["area"] = self.data.area  # self.options["simulation"]["area"]
            final_df.attrs["thickness"] = self.data.thickness
            final_df.attrs["error"] = self.get_error_res(final_df)

            return final_df
        except CommonExceptions as exc:
            # if self.main.kill_operation:
            if self.cancel_event.is_set():
                self.finished.emit(None)
                return pd.DataFrame()
            else:
                raise WorkerError(
                    f"{exc.__class__.__name__} occurred while parsing the bootstrap results.\n{str(exc)}"
                ) from exc

    def perform_bootstrap_fit(self):
        """Perform a bootstrap fit."""
        self._progress_count = 0
        bootstrap_results = self._perform_bootstrap_fit()
        self.finished.emit(bootstrap_results)

    def perform_auto_fit(self):
        """Perform an automatic sequence of fits."""
        fit_results = []

        # Retrieve the necessary parameters from the dictionary
        df = self.auto_fit_params["df"]  # .dropna(axis=1, how="all")
        fit_type = self.auto_fit_params["fit_type"]
        iterations = self.auto_fit_params["iterations"]
        temp_save_path = self.auto_fit_params["temp_save_path"]
        lock_bound = self.auto_fit_params["lock_bound"]
        name_suffix = self.auto_fit_params["suffix"]
        use_pin_model = self.auto_fit_params["use_pin_model"]
        use_pin_const = self.auto_fit_params["use_pin_const"]
        run_order = self.auto_fit_params["run_order"]
        self._sequential_fit = self.auto_fit_params["sequential_fit"]

        kwargs: dict[str, Any] = {
            "rebound_unlocked": True,
            "overwrite_locked": not bool(use_pin_const),
        }
        if lock_bound:
            # Shifts values to respect locked bounds
            kwargs |= {"shift_value": True, "shift_band": False}
        else:
            # Shifts locked bounds to respect initial values
            kwargs |= {"shift_value": False, "shift_band": True}

        self._progress_sub_count = len(df)
        # if use_pin_model:
        #     run_order = list(range(len(df)))
        # for index, row in df.iterrows():
        for index in run_order:
            try:
                row = df.loc[index]
                # Set the model and dataset for the current row
                # Check if filtered_data is empty
                if self.data.get_df(row["Dataset"]).empty:
                    self.error.emit(
                        f"No data points within the specified frequency range for row {index}."
                    )
                    continue

                self.active = self.prime.clone(
                    model=row["Model"] if use_pin_model else "",
                    dataset_var=row["Dataset"],
                    params=row,
                    **kwargs,
                )
                # if use_pin_model:
                #     self.active = self.prime.clone(
                #         model=row["Model"],
                #         dataset_var=row["Dataset"],
                #         params=row,
                #         **kwargs,
                #     )
                # else:
                #     self.active = self.prime.clone(
                #         dataset_var=row["Dataset"],
                #         **kwargs.copy(),
                #     )
                #     self.active.update_base_df(
                #         row,
                #         set_params=True,
                #         **kwargs,
                #     )

                self.active.df = self.data.rs_to_minima(self.active.df, row["Dataset"])

                if "iter" in fit_type:
                    fit_result = self._perform_iterative_fit(iterations)
                elif "boot" in fit_type:
                    fit_result = self._perform_bootstrap_fit(iterations)
                else:
                    fit_result = self.perform_fit()

                # if self.main.kill_operation:  # Check if cancelled
                if self.cancel_event.is_set():  # Check if cancelled
                    break

                if not fit_result.empty:

                    res_df = fit_result

                    param_df = self.active.df.copy()
                    param_df.update(res_df)

                    result_row = {
                        "Name": row["Dataset"] + "_" + name_suffix,
                        "Dataset": row["Dataset"],
                        "Model": self.active.model,
                        "Show": "",
                        "Comments": self.get_error_res(res_df),
                        **{
                            f"{name}_values": param_df.loc[name, "value"]
                            for name in param_df.index
                        },
                        **{f"{name}_std": param_df.loc[name, "std"] for name in param_df.index},
                    }
                    fit_results.append(result_row)

                    # Save the result row to the temporary file
                    pd.DataFrame(fit_results).to_csv(temp_save_path, index=False)
                if "iter" in fit_type or "boot" in fit_type:
                    self.report_progress(False)
                else:
                    self.report_progress()
                # if self.main.kill_operation:  # Check if cancelled
                if self.cancel_event.is_set():  # Check if cancelled
                    break
            except CommonExceptions as exc:
                # if self.main.kill_operation:
                if self.cancel_event.is_set():  # Check if cancelled
                    self.finished.emit(None)
                else:
                    raise WorkerError(
                        f"{exc.__class__.__name__} occurred during auto fit for row {index}: {exc}"
                    ) from exc

        # Create a DataFrame from the fit results
        if not fit_results:
            fit_results_df = pd.DataFrame([])
        else:
            fit_results_df = pd.DataFrame(
                [fit_results[i] for i in np.argsort(run_order[: len(fit_results)])]
            )
        # fit_results_df = (
        #     pd.DataFrame(fit_results, index=run_order).sort_index().reset_index(drop=True)
        # )

        try:
            Path(temp_save_path).unlink()
        except (*CommonExceptions, FileNotFoundError):
            pass

        # Emit the combined fit results
        self.finished.emit(fit_results_df)

    def report_progress(self, increase=True) -> None:
        """Report the progress of the fitting operation."""
        if increase:
            self._progress_count += 1

        self.progress.emit(self._progress_count / self.iterations * 100, self._progress_sub_count)


class LoadDataWorker(QObject):
    """Worker class to handle data loading operations."""

    finished = pyqtSignal(dict, object)  # Signal to emit the results
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path: str | Path, options):
        """Initialize the worker with the file path and options."""
        super().__init__()
        self.file_path = Path(file_path)
        self.options = options

    @pyqtSlot()
    def run(self):
        """Run the data loading operation."""
        try:
            try:
                valid_sheets = {}
                alt_sheets = dict.fromkeys(["params", "attrs", "fit profile", "fit results"])

                data_in, alt_sheets["attrs"] = load_file(self.file_path)

            except CommonExceptions as e:
                raise WorkerError("Error occurred while loading the file.") from e
            cached_keys = CachedColumnSelector(["freq", "real", "imag"])
            if isinstance(data_in, dict):

                # col_names = [["freq", "real", "imag"]]
                for sheet_name, df in data_in.items():
                    try:
                        # Skip specific sheet names
                        if sheet_name in alt_sheets:
                            alt_sheets[sheet_name] = df.copy()

                        elif isinstance(df, pd.DataFrame):
                            df = cached_keys.get_valid_columns(df, ["imps"])
                            if not df.empty:
                                valid_sheets[sheet_name] = ComplexSystem(
                                    df.sort_values("freq", ignore_index=True),  # df
                                    thickness=self.options["simulation"]["thickness"],
                                    area=self.options["simulation"]["area"],
                                )

                    except CommonExceptions as e:
                        raise WorkerError(
                            f"Error occurred while parsing the data for {sheet_name}"
                        ) from e

                name_map = {}
                if isinstance(alt_sheets["fit results"], pd.DataFrame):
                    try:
                        name_map = dict(
                            zip(
                                alt_sheets["fit results"]["Name"].str.lower(),
                                alt_sheets["fit results"]["Dataset"],
                            )
                        )
                        valid_sheets = {
                            name_map.get(k.lower(), k): v for k, v in valid_sheets.items()
                        }
                    except CommonExceptions as e:
                        raise WorkerError("Error occurred while translating the names.") from e
                if alt_sheets["attrs"] is not None:
                    try:
                        rev_name_map = {v: k for k, v in name_map.items()}
                        df = alt_sheets["attrs"].copy()
                        for key, values in df.iterrows():
                            if key in valid_sheets:
                                valid_sheets[key].attrs |= values.to_dict()
                            elif name_map.get(key.lower(), "") in valid_sheets:
                                # If the dataset is in valid_sheets, update its attributes
                                valid_sheets[name_map[key.lower()]].attrs |= values.to_dict()
                            elif rev_name_map.get(key.lower(), "") in valid_sheets:
                                # If the dataset is in valid_sheets, update its attributes
                                valid_sheets[rev_name_map[key.lower()]].attrs |= values.to_dict()
                    except CommonExceptions as e:
                        raise WorkerError("Error occurred while translating the names.") from e
            elif isinstance(data_in, pd.DataFrame):
                try:
                    df = cached_keys.get_valid_columns(data_in, ["imps"])
                    if not df.empty:
                        valid_sheets[self.file_path.stem] = ComplexSystem(
                            df.sort_values("freq", ignore_index=True),
                            thickness=self.options["simulation"]["thickness"],
                            area=self.options["simulation"]["area"],
                        )

                except CommonExceptions as e:
                    raise WorkerError("Error occurred while parsing the data.") from e

            # all_attrs = {}
            for key in valid_sheets:
                attr_ser = pd.Series(valid_sheets[key].attrs)
                null_mask = attr_ser.isna() | (attr_ser == "")
                valid_sheets[key].attrs = attr_ser[~null_mask].to_dict()
                valid_sheets[key].attrs |= self.parse_dataset_name(key)

            self.finished.emit(valid_sheets, alt_sheets["fit results"])
        except WorkerError as e:
            self.error.emit(str(e))

    @staticmethod
    def parse_dataset_name(dataset_name: str) -> dict:
        """
        Parse a dataset name to extract components in any order, removing matches
        as they're found to improve subsequent pattern matching.

        Parameters
        ----------
        dataset_name : str
            String containing the dataset name.

        Returns
        -------
        dict
            Dictionary containing the extracted components.
        """
        result = {}
        working_text = dataset_name

        # Define patterns in order of specificity
        patterns = [
            # Pattern, key, transformation function
            (r"^(9100|406)", "prefix", lambda _: None),
            (r"([5-9][05])c", "temp", lambda x: int(x[:-1])),
            (r"_r\d+", "run", lambda _: None),
            (r"([1-3]\d[01]|cln\d)", "sample_name", lambda x: x.lower()),
            (r"[a-zA-Z]+", "condition", lambda x: x.lower()),
        ]

        # Keep extracting patterns until no more matches are found
        found_match = True
        while found_match:
            found_match = False

            for pattern, key, transform_func in patterns:
                # Skip if we've already found this component
                if key in result:
                    continue

                if match := re.search(pattern, working_text):
                    try:
                        value = transform_func(match.group(0))
                    except Exception:
                        continue

                    found_match = True
                    if value is not None:
                        result[key] = value

                    # Remove the matched text to avoid interfering with future matches
                    working_text = (
                        working_text[: match.start()] + " " + working_text[match.end() :]
                    )
                    break  # Start over with the updated text

        if "sample_name" in result:
            if result["sample_name"].startswith("cln"):
                result["sodium"] = 0.0
            else:
                try:
                    result["sodium"] = float(np.round(float(result["sample_name"]), -1))
                except ValueError:
                    # If conversion fails, sample_name was invalid and is removed
                    result.pop("sample_name", None)

        if "condition" in result:
            if result["condition"][:4] == "init":
                result["condition"] = "pre"
            elif result["condition"][:4] == "post":
                result["condition"] = "dry"
            if result["condition"] not in ["pre", "dh", "dry", "dryout"]:
                result.pop("condition", None)

        return result


class SaveResultsWorker(QObject):
    """
    Worker class to handle saving results.

    Saved sheets:
    'fit results' -> a copy of pinned df (if available/requested)
    'fit profile' -> the active profile from the current parameters
    'params' -> the current parameters
    datasheets will be saved with their current names or that of a fit named in the pinned df
    'attrs' -> the compiled attrs of the datasets.

    """

    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path, main, export=False):
        """Initialize the worker with the file path and data."""
        super().__init__()
        self.file_path = file_path
        # self.data = main.data
        self.raw = main.data.raw
        # self.options = main.options
        self.generator = DataGenerator(
            options=main.options.get_view(["simulation"], flatten=True).as_dict(),
            model=main.settings.model,
        )
        # self.pinned = main.pinned
        self.pinned_df: pd.DataFrame = main.pinned.df.copy()
        self.pinned_cols = {
            "base": main.pinned.df_base_cols.copy(),
            "sort": main.pinned.df_sort_cols.copy(),
        }

        self.options = {
            "save_pinned": False,
            "save_params": False,
            "save_current_fit": False,
            "save_only_pins": False,
        }
        if not export:
            # true if params_df and not export save
            self.options["save_pinned"] = main.options["general"]["save_pinned_results"]
            # based on settings and not export save
            self.options["save_params"] = main.options["general"]["save_current_profile"]
            # based on settings and not export save
            self.options["save_current_fit"] = main.options["general"]["save_current_parameters"]
            # self.save_all_datasets = True # always true, if pinned and save only, then gets turned to false, otherwise only
            # base on settings include_unfit_datasets
            self.options["save_only_pins"] = not main.options["general"]["include_unfit_datasets"]

        self.active_data_df = pd.DataFrame()

        self.params_df = pd.DataFrame()
        self._parse_params(main)

        # save forms should be
        self.save_forms = ["freq", "real", "imag"]
        if form_mod := main.options["general"]["additional_save_forms"]:
            if isinstance(form_mod, str):
                form_mod = [form_mod]
            for f_mod in form_mod:
                if f_mod not in self.save_forms and ComplexSystem.is_valid_key(f_mod):
                    self.save_forms.append(f_mod)

        if self.raw != {}:
            self.active_data_df: pd.DataFrame = self.raw[main.data.var.currentText()].get_df(
                *self.save_forms
            )

    @pyqtSlot()
    def run(self):
        """Run the saving operation."""
        try:
            res = {}
            if self.options["save_params"] and not self.params_df.empty:
                res["params"] = self.params_df
            if self.options["save_current_fit"]:
                self._parse_simulated(res)
            if self.options["save_pinned"] and not self.pinned_df.empty:  # self.pinned.df.empty:
                self._parse_pinned(res)

            try:
                fit_res_df = res.get("fit results")
                # all_attrs = []  # Use a list of dictionaries for easier DataFrame creation
                seen_datasets = set()
                save_datasets = True
                # First loop: Iterate over fit_res_df to handle datasets already added under another name
                if (
                    fit_res_df is not None
                    and "Dataset" in fit_res_df.columns
                    and "Name" in fit_res_df.columns
                ):
                    if self.options["save_only_pins"]:
                        save_datasets = False
                    for _, row in fit_res_df.iterrows():
                        dataset_key = row["Dataset"]
                        dataset_name = row["Name"]

                        # Ensure the dataset exists in self.raw
                        if dataset_key in self.raw:
                            value = self.raw[dataset_key]

                            # Mark the dataset as seen
                            seen_datasets.add(dataset_key)

                # Second loop: Iterate over self.raw, skipping datasets already seen
                for key, value in self.raw.items():
                    dataset_name = key  # Default to the raw dataset key
                    if key in seen_datasets:
                        continue  # Skip datasets already handled in the first loop

                    if save_datasets and dataset_name not in res:
                        # Add the dataset to res
                        res[dataset_name] = value.get_df(*self.save_forms)

            except CommonExceptions as e:
                raise WorkerError(
                    f"{e.__class__.__name__} occurred while parsing the raw data."
                ) from e
            try:
                save(
                    res,
                    Path(self.file_path).parent,
                    name=Path(self.file_path).stem,
                    file_type=Path(self.file_path).suffix,
                    mult_to_single=True,
                    attrs=True,
                )
                self.finished.emit()
            except PermissionError as e:
                # breakpoint()
                raise WorkerError(
                    f"Permission error: {str(e)}. Please check the file is closed or not in use."
                ) from e
            except (TypeError, ValueError, IndexError, KeyError, AttributeError) as e:
                raise WorkerError(
                    f"{e.__class__.__name__} occurred while saving the results."
                ) from e
        except WorkerError as e:
            self.error.emit(str(e))

    def _parse_pinned(self, res):
        """Parse the pinned DataFrame."""
        # Check if the pinned data is available, will result in saving the fits
        try:
            # filtered_cols = self.pinned.df_base_cols.copy()
            # for col in self.pinned.df_sort_cols:
            #     filtered_cols += [c for c in self.pinned.df.columns if col in c]
            filtered_cols = self.pinned_cols["base"].copy()
            sort_cols = self.pinned_cols["sort"].copy()
            for col in sort_cols:
                filtered_cols += [c for c in self.pinned_df.columns if col in c]

            # res_df = self.pinned.df[filtered_cols].copy()
            # res_df.columns = [
            #     (
            #         col.replace(f"_{self.pinned.df_sort_cols[0].lower()}", "")
            #         if self.pinned.df_sort_cols[0].lower() in col
            #         else col
            #     )
            #     for col in res_df.columns
            # ]
            res_df = self.pinned_df[filtered_cols].copy()
            res_df.columns = [
                (
                    col.replace(f"_{sort_cols[0].lower()}", "")
                    if sort_cols[0].lower() in col
                    else col
                )
                for col in res_df.columns
            ]
            res["fit results"] = res_df
        except KeyError as e:
            raise WorkerError(
                "Error occurred while parsing the pinned data into a dataframe."
            ) from e

        for _, row in res_df.iterrows():
            try:
                # params_values = [
                #     row[name] for name in self.data.parse_parameters(model=row["Model"])
                # ]
                params_values = [
                    row[name] for name in self.generator.parse_parameters(model=row["Model"])
                ]
                if self.raw != {} and row["Dataset"] in self.raw.keys():
                    # get the data from the raw data
                    local_data = self.raw[row["Dataset"]].get_df(*self.save_forms)
                else:
                    # use simulation settings to generate the data
                    # local_data = self.data.generate(
                    #     params_values, **self.options["simulation"]
                    # ).get_df(*self.save_forms)
                    local_data = self.generator.get(params_values).get_df(*self.save_forms)

                # generated_data = self.data.generate(
                #     params_values=params_values,
                #     model=row["Model"],
                #     freq=local_data["freq"].to_numpy(copy=True),
                #     **{**self.options["simulation"], **{"interp": False}},
                # ).get_df(*self.save_forms)
                generated_data = self.generator.get(
                    params_values=params_values,
                    model=row["Model"],
                    freq=local_data["freq"].to_numpy(copy=True),
                    interp=False,
                ).get_df(*self.save_forms)

                for col in generated_data.columns:
                    local_data[f"pr_{col}"] = generated_data[col]

                res[row["Name"]] = local_data
            except CommonExceptions as e:
                raise WorkerError(
                    f"Error occurred while parsing the data for {row['Name']}"
                ) from e

    def _parse_params(self, root):
        """Construct a DataFrame of the current parameters. Called from __init__ only."""
        try:
            params_names = [entry.name for entry in root.parameters]
            params_values = [entry.values[0] for entry in root.parameters]
            params_stds = [entry.values[0] for entry in root.parameters_std]

            values = (
                [root.settings.model, root.data.var.currentText()] + params_values + params_stds
            )
            names = ["Model", "Dataset"] + params_names + [f"{name}_std" for name in params_names]

            self.params_df = pd.DataFrame([values], columns=names)

        except CommonExceptions as e:
            raise WorkerError("Error occurred while parsing the params (in _parse_params).") from e

    def _parse_simulated(self, res):
        """"""
        try:
            # valid names are those without std and are not model or dataset
            params_names = [
                name
                for name in self.params_df.columns
                if name not in ["Model", "Dataset"] and "_std" not in name
            ]
            # get values from self.params_df using params_names
            params_values = list(self.params_df[params_names].to_numpy(copy=True)[0])

            # adds simulation to current dataset
            if not self.active_data_df.empty:

                res["fit profile"] = self.active_data_df.copy()

                # generated_data = self.data.generate(
                #     params_values,
                #     freq=res["fit profile"]["freq"].to_numpy(copy=True),
                #     **{**self.options["simulation"], **{"interp": False}},
                # ).get_df(*self.save_forms)
                generated_data = self.generator.get(
                    params_values,
                    freq=res["fit profile"]["freq"].to_numpy(copy=True),
                    interp=False,
                ).get_df(*self.save_forms)
                for col in generated_data.columns:
                    res["fit profile"][f"pr_{col}"] = generated_data[col]
            # or generates a singular profile
            else:
                # generated_data = self.data.generate(
                #     params_values, **self.options["simulation"]
                # ).get_df(*self.save_forms)
                generated_data = self.generator.get(params_values).get_df(*self.save_forms)

                res["fit profile"] = generated_data

            # res["fit results"] = pd.DataFrame(
            #     [params_values], columns=params_names
            # )
        except CommonExceptions as e:
            raise WorkerError(
                "Error occurred while parsing the data (in _parse_simulated)."
            ) from e


class SaveFiguresWorker(QObject):
    """Worker class to handle saving figures."""

    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path, fig1, fig2):
        """Initialize the worker with the file path and figures."""
        super().__init__()
        self.file_path = file_path
        self.fig1 = fig1
        self.fig2 = fig2

    @pyqtSlot()
    def run(self):
        """Run the saving operation."""
        try:
            base_path = Path(self.file_path)
            nyquist_file_path = base_path.with_name(base_path.stem + "_nyquist").with_suffix(
                base_path.suffix
            )
            bode_file_path = base_path.with_name(base_path.stem + "_bode").with_suffix(
                base_path.suffix
            )

            self.fig1.savefig(nyquist_file_path)
            self.fig2.savefig(bode_file_path)

            self.finished.emit()
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, PermissionError) as e:
            self.error.emit(str(e))

    # def update_base_df(
    #     self,
    #     params: pd.Series,
    #     set_params: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     """
    #     Update values in the existing parameter DataFrame from params.

    #     Parameters
    #     ----------
    #     params : pd.Series
    #         Series containing parameter values and standard deviations
    #     set_params : bool, optional
    #         Whether to call set_parameters after updating, by default False
    #     **kwargs
    #         Additional arguments to pass to set_parameters

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Updated parameter DataFrame
    #     """
    #     # Get parameter names from the existing self.df
    #     param_names = self.df.index.tolist()

    #     # Directly update the values in self.df where they exist
    #     for name in param_names:
    #         if f"{name}_values" in params:
    #             self.df.at[name, "value"] = params[f"{name}_values"]
    #         if f"{name}_std" in params:
    #             self.df.at[name, "std"] = params[f"{name}_std"]

    #     if kwargs.pop("sort_params", self.sort_params):
    #         self.df = sort_parameters(self.df)

    #     if set_params:
    #         self.set_parameters(self.df, **kwargs)

    #     return self.df

    # def set_parameters(self, df=None, overwrite_locked:bool=False, **kwargs):
    #     """
    #     Set the parameter DataFrame after processing it with validate_vals_and_band.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame, optional
    #         DataFrame containing parameter values and constraints
    #     init_df : pd.DataFrame, optional
    #         Reference DataFrame with default values for missing columns
    #     **kwargs
    #         Additional arguments for validate_vals_and_band function

    #     Returns
    #     -------
    #     self : FitContext
    #         Returns self for method chaining
    #     """
    #     if df is None or df.empty:
    #         # If no DataFrame is provided, assume kwarg (band setting) update
    #         self.df = validate_vals_and_band(self.df, self.bound_range, **kwargs)
    #         return

    #     # Process the incoming DataFrame
    #     with pd.option_context("future.no_silent_downcasting", True):
    #         # Ensure the DataFrame has the required columns
    #         df = df.reindex(columns=self._df_cols, fill_value=pd.NA)

    #         # Update df with values from init_df where indexes match
    #         if any(ind in self.ref_df.index for ind in df.index):
    #             df.update(self.ref_df, overwrite=False)

    #         # Fill missing lock columns with False
    #         df[self._locked_cols] = df[self._locked_cols].fillna(False)

    #         # Explicitly set the data types
    #         df[self._locked_cols] = df[self._locked_cols].astype(bool)
    #         df[self._value_cols] = df[self._value_cols].astype(float)

    #         # Fill missing bound columns with calculated values
    #         df["bnd_low"] = df["bnd_low"].fillna(float(self.bound_range[0]) * df["value"])
    #         df["bnd_high"] = df["bnd_high"].fillna(float(self.bound_range[0]) * df["value"])

    #         # df["bnd_low"] = df["bnd_low"].fillna((self.bound_range[0] * df["value"]).astype(float))
    #         # df["bnd_high"] = df["bnd_high"].fillna((self.bound_range[1] * df["value"]).astype(float))

    #     if overwrite_locked:
    #         locked_names = self.ref_df.index[self.ref_df["lock"] & self.ref_df.index.isin(df.index)]
    #         # Update 'value' and 'std' in df for these parameters
    #         df.loc[locked_names, ["value", "std"]] = self.ref_df.loc[locked_names, ["value", "std"]].values

    #     # Apply validate_vals_and_band to get the final DataFrame
    #     self.df = validate_vals_and_band(df, self.bound_range, **kwargs)
