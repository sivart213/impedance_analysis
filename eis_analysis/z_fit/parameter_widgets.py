from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd
from PyQt5 import sip  # type: ignore
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QDialog,
    QWidget,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from ..string_ops import MathEvaluator
from ..impedance_supplement import parse_parameters
from ..widgets.widget_helpers import create_separator
from ..data_treatment.data_ops import TypeList
from ..widgets.generic_widgets import IncLineEdit, fLabel, tLabel
from ..data_treatment.z_array_ops import find_peak_vals
from ..impedance_supplement.model_ops import parse_model_groups


class ParameterStatPanel(QWidget):
    """
    A container widget that arranges multiple LabeledFloat widgets
    in a grid layout (labels aligned in one column, values in another).
    Supports dynamic reconstruction of rows.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self._model = ""
        self._values = {}

        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(8, 8, 8, 8)

        # Create left and right grid layouts
        self._left_layout = QGridLayout()
        self._left_layout.setSpacing(4)
        self._right_layout = QGridLayout()
        self._right_layout.setSpacing(4)

        # Add grid layouts to the main layout
        left_layout = QVBoxLayout()
        left_layout.addLayout(self._left_layout)
        left_layout.addStretch()
        right_layout = QVBoxLayout()
        right_layout.addLayout(self._right_layout)
        right_layout.addStretch()

        self._layout.addLayout(left_layout)
        self._layout.addLayout(right_layout)

        self.setLayout(self._layout)

        self.parameters = kwargs.get("parameters", None)
        self.model_entry = kwargs.get("model_entry", None)
        self.model = kwargs.get("model", "")
        self.data = kwargs.get("data", pd.DataFrame(columns=["freq", "real", "imag"]))

        self.precision = kwargs.get("precision", 3)
        self.upper_exponent = kwargs.get("upper_exponent", 1)
        self.lower_exponent = kwargs.get("lower_exponent", -2)

    # --- Row Management ---

    @property
    def model(self):
        if self.model_entry is not None:
            return self.model_entry.text()
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    @property
    def values(self):
        if self.parameters is not None:
            return self.parameters.value_dict
        return self._values

    @values.setter
    def values(self, value: dict):
        self._values = value.copy()

    def clear_layout(self, layout):
        """Clear all widgets from a given layout."""
        while layout.count():
            item = layout.takeAt(0)
            widg = item.widget()
            if widg is not None:
                widg.setParent(None)
        return layout

    def constructRows(
        self,
        layout,
        items: dict[str, str | float | int | list],
        headers: list[str] | None = None,
        **kwargs,
    ):
        """
        Clear existing rows and rebuild from a dict of {title: value}.

        Args:
            items: Dictionary of items to display
            headers: Optional header labels for columns
            layout: Target layout to populate (defaults to left layout)
            **kwargs: Additional formatting parameters
        """
        if not items:
            return

        kwargs.setdefault("precision", self.precision)
        kwargs.setdefault("upper_exponent", self.upper_exponent)
        kwargs.setdefault("lower_exponent", self.lower_exponent)

        # --- Clear layout ---
        layout = self.clear_layout(layout)

        # --- Add headers if provided ---
        vshift = 0
        if headers and len(headers) >= 2:
            vshift = 2
            for n, head in enumerate(headers):
                string = str(head).strip(" :") + ":"
                layout.addWidget(tLabel(string), 0, n)
            layout.addWidget(create_separator(), 1, 0, 1, len(headers))

        for row, (title, value) in enumerate(items.items()):
            hsift = 0
            if title:
                layout.addWidget(tLabel(title), row + vshift, 0)
                hsift = 1
            if not isinstance(value, list):
                value = [value]
            for n, val in enumerate(value):
                widget = fLabel(val, **kwargs)
                layout.addWidget(widget, row + vshift, n + hsift)

    def _get_stats(
        self,
        values: dict,
        models: tuple[str, ...] | list[str],
        frequency: np.ndarray | None = None,
        method: str | list[str] = "fpeak",
    ) -> dict[str, Any]:
        """
        Sort a list of models based on the first number found in each model or by a corresponding value.

        Parameters:
        -----------
        models : list of str
            The list of circuit model strings to sort.
        values : dict
            The dict of values corresponding to each model for sorting.
        method : str

        descending : bool
            Whether to sort in descending order. Default is False (ascending).

        """
        if not isinstance(frequency, np.ndarray):
            frequency = np.logspace(-4, 6, 1000)

        g_map = {}
        for model in models:
            if not model or model.lower() == "linkk":
                continue
            parameters = parse_parameters(model)
            sub_vals = [values.get(param, 1) for param in parameters]

            g_map[model] = find_peak_vals(f=frequency, values=method, params=sub_vals, model=model)

        return g_map

    def group_stats(
        self,
        values: dict,
        models: tuple[str, ...] | list[str] = (),
        frequency: np.ndarray | None = None,
        method: str | list[str] = "fpeak",
    ):
        """
        Sort a list of models based on the first number found in each model or by a corresponding value.

        Parameters:
        -----------
        models : list of str
            The list of circuit model strings to sort.
        values : dict
            The dict of values corresponding to each model for sorting.
        method : str

        descending : bool
            Whether to sort in descending order. Default is False (ascending).

        """
        if not values:
            return
        if isinstance(method, str):
            method = [method]
        if not models:
            if self.model.lower() == "linkk":
                return
            models = (
                parse_model_groups(self.model, "infer")
                if self.model_entry is None
                else self.model_entry.sub_models
            )
        g_map = self._get_stats(values, models, frequency, method)
        self.constructRows(self._right_layout, g_map, ["Model"] + method)

    def model_stats(
        self,
        values: dict,
        model: str = "",
        frequency: np.ndarray | None = None,
        data: pd.DataFrame | None = None,
    ):
        """
        Sort a list of models based on the first number found in each model or by a corresponding value.

        Parameters:
        -----------
        model : str
            The list of circuit model strings to sort.
        values : dict
            The dict of values corresponding to each model for sorting.
        method : str

        descending : bool
            Whether to sort in descending order. Default is False (ascending).

        """
        methods = ["f_peak", "RC", "R_max", "C_peak"]

        # --- Data stats ---
        data_stats = []
        data = data if data is not None and not data.empty else self.data
        if data is not None and not data.empty:
            Z = data["real"].to_numpy() + 1j * data["imag"].to_numpy()
            data_stats = find_peak_vals(f=data["freq"].to_numpy(), values=methods, Z=Z)

        # --- Model stats ---
        model_stats = []
        model = model or self.model
        if model.lower() == "linkk" or not model:
            if self._model and self._model.lower() != "linkk":
                model = self._model
                values = self._values
            else:
                model = ""
        if model and values:
            model_stats = self._get_stats(values, [model], frequency, methods).get(model, [])

        # --- Update Layout Display ---
        if model_stats and data_stats:
            self.constructRows(
                self._left_layout,
                {m: [v, d] for m, v, d in zip(methods, model_stats, data_stats)},
                ["Stat", "Model", "Data"],
            )
        elif data_stats:
            self.constructRows(
                self._left_layout,
                {m: v for m, v in zip(methods, data_stats)},
                ["Data Stat", "Value"],
            )
        elif model_stats:
            self.constructRows(
                self._left_layout,
                {m: v for m, v in zip(methods, model_stats)},
                ["Model Stat", "Value"],
            )

    def all_stats(
        self,
        values: dict,
        model: str = "",
        models: tuple[str, ...] | list[str] = (),
        frequency: np.ndarray | None = None,
        data: pd.DataFrame | None = None,
        method: str | list[str] = ["f_peak", "RC"],
    ):
        """
        Sort a list of models based on the first number found in each model or by a corresponding value.

        Parameters:
        -----------
        model : str
            The list of circuit model strings to sort.
        models : list of str
            The list of circuit model strings to sort.
        values : dict
            The dict of values corresponding to each model for sorting.
        method : str

        descending : bool
            Whether to sort in descending order. Default is False (ascending).

        """
        self.model_stats(values, model, frequency, data)
        self.group_stats(values, models, frequency, method)

    def refresh(self, *, frequency: np.ndarray | None = None):
        """Refresh the display based on current model and parameters."""
        # if self.parameters is None:
        #     return
        self.all_stats(self.values, self.model, frequency=frequency, data=self.data)

    def update(self, **kwargs):
        """
        Update internal state and refresh display.

        Args:
            **kwargs: Optional parameters to update:
                - model (str): New model string
        """
        if "data" in kwargs:
            self.data = kwargs.pop("data")
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        if "parameters" in kwargs:
            self.parameters = kwargs.pop("parameters")
        if "model_entry" in kwargs:
            self.model_entry = kwargs.pop("model_entry")

    def window(self, title="LabeledFloatStack"):
        """
        Wrap this stack in a QDialog for standalone display.
        """
        dlg = QDialog(self.parent())
        dlg.setWindowTitle(title)
        layout = QVBoxLayout(dlg)
        layout.addWidget(self)

        # Add a close button for convenience
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.close)
        layout.addWidget(btn)

        btn = QPushButton("Refresh")
        btn.clicked.connect(self.refresh)
        layout.addWidget(btn)

        self.show()
        dlg.show()


class MultiEntryWidget:
    """Class to create a custom entry widget."""

    def __init__(
        self,
        name: str,
        frame: QFrame | None,
        init_values: list | tuple | object | None = None,
        has_value: bool = True,
        has_check: bool = True,
        set_settings: bool = True,
        num_entries: int = 1,
        default_values: list | None = None,
        label_width: int | float | str = "auto",
        layout: type = QHBoxLayout,
        interval: int | float = 1,
        fine_interval: int | float = 0,
        debounce_interval: int = 2000,
    ):
        self.name = name
        self.has_value = has_value
        self.has_check = has_check
        self.num_entries = num_entries
        self.frame = frame
        self.main_layout = layout
        self.frame_dead = False
        self.label_width = label_width
        self.accents: dict[str, Any] = {}
        self.widgets = TypeList([])
        self.interval = interval
        self.fine_interval = fine_interval or interval / 10
        self.debounce_interval = debounce_interval

        if init_values is None:
            init_values = [1] if default_values is None else default_values

        init_values = self._list_check(init_values)

        self.default_values = (
            self._list_check(default_values) if default_values is not None else init_values
        )

        self._values = init_values
        self._checked = [False] * num_entries
        self._bounds = [(-np.inf, np.inf)] * num_entries
        self._eval = MathEvaluator()

        if self.frame is not None and set_settings:
            self.create_widgets(frame, init_values, num_entries, has_value, has_check)
        else:
            self.frame_dead = True

    def create_widgets(
        self,
        frame,
        values,
        num_entries=1,
        has_value=True,
        has_check=True,
        layout="default",
    ):
        """Create the widgets for the entry."""
        self.widgets = TypeList([])

        label = QLabel(self.name)
        text_width = 40
        if self.label_width == "auto":
            label.setFixedWidth(QFontMetrics(label.font()).width(self.name))
            font_metrics = QFontMetrics(label.font())
            text_width = font_metrics.width(label.text())
            text_width += text_width * 0.2
        elif isinstance(self.label_width, (int, float)):
            text_width = self.label_width
        label.setFixedWidth(int(text_width))
        self.widgets.append(label)

        values = self._list_check(values)

        for i in range(num_entries):
            entry = IncLineEdit()
            if has_value:
                # entry.setTextValue(safe_eval(str(values[i])))
                entry.setTextValue(self._eval.parse(values[i]))
                entry.interval = self.interval
                entry.fine_interval = self.fine_interval
                entry.debounce_interval = self.debounce_interval
                entry.bounds = (-np.inf, np.inf) if len(self._bounds) < i + 1 else self._bounds[i]
                self.widgets.append(entry)

            if has_check:
                check_var = QCheckBox()
                # Connect the checkbox to toggle the enable state of the entry
                check_var.stateChanged.connect(
                    lambda state, widget=entry: self.toggle_enable(widget, state)
                )
                self.widgets.append(check_var)

        self.apply_layout(frame, layout)

        if self.accents:
            self.highlight(**self.accents)

    def toggle_enable(self, widget, state):
        """
        Toggle the enable state of a widget based on the checkbox state.

        Args:
            widget (QWidget): The widget to enable or disable.
            state (int): The state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        widget.setEnabled(state != Qt.Checked)

    def apply_layout(self, frame: Any, layout: str | Callable | None = "default"):
        """Add the entry widgets to a layout."""
        if self.frame is None or self.frame_dead:
            self.frame = frame if frame is not None else QFrame()
            self.frame_dead = False

        # Remove existing layout if it exists
        if self.frame.layout() is not None:
            old_layout = self.frame.layout()
            QWidget().setLayout(old_layout)

        if layout is None:
            return

        if (
            "Q" not in str(type(layout))
            or "Layout" not in str(type(layout))
            or layout == "default"
        ):
            layout = QHBoxLayout
        if not callable(layout):
            return
        self.main_layout: Any = layout()
        self.frame.setLayout(self.main_layout)

        for widget in self.widgets:
            self.main_layout.addWidget(widget)

    def save_values(self):
        """Save the current values and checked states."""
        self._values = self.values
        self._checked = self.is_checked
        self._bounds = self.bounds

    def reset_widget(self, frame, default=False, layout="default"):
        """Reset the frame with the saved values and checked states."""
        if not self.frame_dead:
            self.destroy()
        self.frame = frame

        values_to_use = self.default_values if default else self._values

        self.create_widgets(
            frame,
            values_to_use,
            self.num_entries,
            self.has_value,
            self.has_check,
            layout,
        )

        self.is_checked = self._checked
        self.frame_dead = False

    def uncheck(self):
        """Save the current values and checked states."""
        self._checked = [False] * self.num_entries

    def destroy(self, error_call=False):
        """Destroy the entry widgets and the parent frame."""
        if not error_call:
            self.save_values()
        if hasattr(self, "entry_widgets"):
            # for widget in self.entry_widgets + self.check_vars:
            for widget in self.widgets.of_type(IncLineEdit, QCheckBox):
                if widget is not None and not sip.isdeleted(widget):
                    widget.deleteLater()
        if self.frame is not None and not sip.isdeleted(self.frame):
            self.frame.deleteLater()

        self.widgets = TypeList([])
        self.frame_dead = True

    def _list_check(self, values: Any) -> list:
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values] * self.num_entries
        if len(values) != self.num_entries:
            values = [values[0]] * self.num_entries
        if isinstance(values, np.ndarray):
            values = values.tolist()
        return list(values)

    @property
    def values(self):
        """Get the current values of the entry widgets."""
        if not self.frame_dead:
            try:
                if self.has_value:
                    self._values = [
                        self._eval.parse(entry.text().lower())
                        for entry in self.widgets.of_type(IncLineEdit)
                    ]
            except (ValueError, SyntaxError):
                pass  # Ignore invalid entries and keep the current values
            except AttributeError:
                self.destroy(True)
        return self._values

    @values.setter
    def values(self, new_values):
        """Set new values in the entry widgets."""
        new_values = self._list_check(new_values)

        if not self.frame_dead:
            try:
                if self.has_value:
                    for entry, new_value in zip(self.widgets.of_type(IncLineEdit), new_values):
                        entry.setTextValue(self._eval.parse(new_value))
                    self._values = new_values
                    return
            except (ValueError, SyntaxError):
                pass  # Ignore invalid entries and keep the current values
            except AttributeError:
                self.destroy(True)
        self._values = new_values

    @property
    def is_checked(self):
        """Check if the entries are selected (checked)."""
        if not self.frame_dead:
            try:
                if self.has_check:
                    self._checked = [
                        check_var.isChecked() for check_var in self.widgets.of_type(QCheckBox)
                    ]
            except (ValueError, SyntaxError):
                pass  # Ignore invalid entries and keep the current values
            except AttributeError:
                self.destroy(True)
        return self._checked

    @is_checked.setter
    def is_checked(self, checked):
        """Set the check state of the entries."""
        checked = self._list_check(checked)

        if not self.frame_dead:
            try:
                if self.has_check:
                    for check_var, state in zip(self.widgets.of_type(QCheckBox), checked):
                        check_var.setChecked(state)
                    self._checked = checked
                    return
            except (ValueError, SyntaxError):
                pass  # Ignore invalid entries and keep the current values
            except AttributeError:
                self.destroy(True)
        self._checked = checked

    @property
    def bounds(self):
        """Get the current values of the entry widgets."""
        if not self.frame_dead:
            try:
                if self.has_value:
                    self._bounds = [entry.bounds for entry in self.widgets.of_type(IncLineEdit)]
                    return self._bounds
            except (ValueError, SyntaxError, AttributeError):
                pass  # Ignore invalid entries and keep the current values
            # except AttributeError:
            #     self.destroy(True)
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds):
        """Set new values in the entry widgets."""
        # new_values = self._list_check(new_values)
        if not isinstance(new_bounds, (list, tuple, np.ndarray)):
            return
        if len(new_bounds) == 2 and all(isinstance(b, (int, float)) for b in new_bounds):
            new_bounds = [tuple(new_bounds)] * self.num_entries
        bounds = []
        for bound in new_bounds:
            if not isinstance(bound, (tuple, list, np.ndarray)) or min(bound) == max(bound):
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append((min(bound), max(bound)))

        if not self.frame_dead:
            try:
                if self.has_value:
                    for entry, bound in zip(self.widgets.of_type(IncLineEdit), bounds):
                        entry.bounds = (min(bound), max(bound))
            except (ValueError, SyntaxError, AttributeError):
                pass  # Ignore invalid entries and keep the current values
        self._bounds = bounds

    def highlight(
        self,
        *,
        bold=True,
        italic=True,
        text_size=None,
        color=None,
        border=None,
        tooltip=None,
        border_radius=None,
    ):
        """
        Highlight the widget with optional styling.

        Parameters:
        - bold (bool): Set to True to make the label text bold.
        - italic (bool): Set to True to italicize the label text.
        - text_size (int): Set the text size of the label.
        - color (str): Set the background color of the frame. Example: "lightblue", "yellow".
        - border (str): Set the border style of the frame. Example: "2px solid red", "1px dashed black".
        - tooltip (str): Set the tooltip text for the label.
        - border_radius (str): Set the border radius of the frame. Example: "10px".

        Example usage:
        entry.highlight(bold=True, italic=True, text_size=14, color="lightblue", border="2px solid red", tooltip="This is a tooltip", border_radius="10px")
        """
        self.accents = dict(
            bold=bold,
            italic=italic,
            text_size=text_size,
            color=color,
            border=border,
            tooltip=tooltip,
            border_radius=border_radius,
        )

        if self.frame is None or self.frame_dead:
            return

        # label = self.main_layout.itemAt(0).widget()
        label = self.widgets[0]
        font = label.font()

        if bold:
            font.setBold(True)
        if italic:
            font.setItalic(True)
        if text_size:
            font.setPointSize(text_size)
        if tooltip:
            label.setToolTip(tooltip)

        label.setFont(font)

        style = ""
        if color:
            style += f"background-color: {color};"
        if border:
            style += f"border: {border};"
        if border_radius:
            style += f"border-radius: {border_radius};"

        self.frame.setStyleSheet(style)


class MultiEntryManager:
    """Class to create a list of custom entry widgets."""

    def __init__(
        self,
        frame: QFrame,
        entries: list | None = None,
        num_entries: int = 1,
        has_value: bool = True,
        has_check: bool = True,
        label_width: str = "uniform",
        callback: Callable | None = None,
        interval: int | float = 1,
        fine_interval: int | float = 0,
        debounce_interval: int = 2000,
    ):
        self.frame = frame
        self._entries = []
        self._archive = {}
        self._connected_funcs = None
        self._highlighted = []
        self._history = []
        self.has_value = has_value
        self.has_check = has_check
        self.num_entries = num_entries
        self.label_width = label_width
        self.connected_funcs = callback
        self.interval = interval
        self.fine_interval = fine_interval or interval / 10
        self.debounce_interval = debounce_interval

        if entries is not None:
            for entry in entries:
                if isinstance(entry, MultiEntryWidget):
                    self.append(entry)
                elif isinstance(entry, dict):
                    self.append(**entry)
                elif isinstance(entry, (tuple, list)):
                    self.append(*entry)
                else:
                    self.append(str(entry))

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._entries[key]
        for entry in self._entries:
            if entry.name == key:
                return entry
        raise KeyError(f"'MultiEntryManager' object has no entry row named '{key}'")

    def __setitem__(self, key, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            value = [value] * self.num_entries
        for entry in self._entries:
            if entry.name == key:
                if all(isinstance(v, bool) for v in value):
                    entry.is_checked = value
                else:
                    self.update_history()
                    entry.values = value
                return
        self.append(key, value)

    def __bool__(self):
        """Evaluate the instance as True if there are any entries."""
        return bool(self._entries)

    def __contains__(self, item):
        return item in self.names

    @property
    def shape(self):
        """Get the shape of the entries."""
        if self._entries:
            return (len(self._entries), self.num_entries)
        return (0, self.num_entries)

    @property
    def entries(self):
        """Get the entries."""
        return self._entries

    @property
    def highlighted(self):
        """Get the entries."""
        return [entry.name for entry in self._highlighted]

    @property
    def _df(self):
        """Get all values of the entries as a DataFrame with entry names as index."""
        return pd.DataFrame([entry.values for entry in self._entries], dtype="object")

    @property
    def connected_funcs(self):
        """Get the connected functions."""
        return self._connected_funcs

    @connected_funcs.setter
    def connected_funcs(self, funcs):
        """Set the connected functions."""
        if funcs is None:
            self._connected_funcs = None
        elif callable(funcs):
            self._connected_funcs = [funcs] * self.num_entries
        elif isinstance(funcs, (list, tuple)) and all(callable(func) for func in funcs):
            if len(funcs) == self.num_entries:
                self._connected_funcs = funcs
            else:
                self._connected_funcs = [funcs[0]] * self.num_entries
        else:
            raise ValueError("Connected functions must be callable or a list of callables.")
        self.apply_connected_funcs()

    @property
    def values(self):
        """Get all values of the entries."""
        return self._df.to_numpy(copy=True).tolist()

    @values.setter
    def values(self, new_values):
        """Set all values of the entries."""
        if not hasattr(new_values, "__len__") or len(new_values) != len(self._entries):
            raise ValueError("Passed number of values does not match the number of entries.")
        self.update_history()
        for entry, values in zip(self._entries, new_values):
            entry.values = values

    @property
    def checked_values(self):
        """Get the values of the checked entries."""
        checks = np.array(self.checks)
        return self._df[checks].to_numpy(copy=True).tolist()

    @checked_values.setter
    def checked_values(self, new_values):
        """Set the values of the checked entries."""
        checks = np.array(self.checks)
        df = self._df.copy()
        if np.array(new_values).shape != df.shape and len(new_values) != checks.sum():
            raise ValueError(
                "Passed number of values does not match the number of checked entries."
            )
        self.update_history()
        df[checks] = new_values
        for entry, values in zip(self._entries, df.to_numpy(copy=True).tolist()):
            entry.values = values

    @property
    def unchecked_values(self):
        """Get the values of the unchecked entries."""
        checks = np.array(self.checks)
        return self._df[~checks].to_numpy(copy=True).tolist()

    @unchecked_values.setter
    def unchecked_values(self, new_values):
        """Set the values of the unchecked entries."""
        checks = np.array(self.checks)
        df = self._df.copy()
        if np.array(new_values).shape != df.shape and len(new_values) != (~checks).sum():
            raise ValueError(
                "Passed number of values does not match the number of unchecked entries."
            )
        self.update_history()
        df[~checks] = new_values
        for entry, values in zip(self._entries, df.to_numpy(copy=True).tolist()):
            entry.values = values

    @property
    def names(self):
        """Get all names of the entries."""
        return [entry.name for entry in self._entries]

    @names.setter
    def names(self, new_names):
        """Set all names of the entries."""
        if len(new_names) != len(self._entries):
            raise ValueError("Passed number of names does not match the number of entries.")
        for entry, name in zip(self._entries, new_names):
            entry.name = name

    @property
    def checked_names(self):
        """Get the names of the checked entries."""
        checks = np.array(self.checks)
        return [entry.name for entry, check in zip(self._entries, checks) if check]

    @checked_names.setter
    def checked_names(self, new_names):
        """Set the names of the checked entries."""
        checks = np.array(self.checks)
        if len(new_names) != checks.sum():
            raise ValueError(
                "Passed number of names does not match the number of checked entries."
            )
        for entry, name in zip(
            [entry for entry, check in zip(self._entries, checks) if check],
            new_names,
        ):
            entry.name = name

    @property
    def unchecked_names(self):
        """Get the names of the unchecked entries."""
        checks = np.array(self.checks)
        return [entry.name for entry, check in zip(self._entries, checks) if not check]

    @unchecked_names.setter
    def unchecked_names(self, new_names):
        """Set the names of the unchecked entries."""
        checks = np.array(self.checks)
        if len(new_names) != (~checks).sum():
            raise ValueError(
                "Passed number of names does not match the number of unchecked entries."
            )
        for entry, name in zip(
            [entry for entry, check in zip(self._entries, checks) if not check],
            new_names,
        ):
            entry.name = name

    @property
    def checks(self):
        """Get the checked state of the entries."""
        return [entry.is_checked for entry in self._entries]

    @checks.setter
    def checks(self, new_checks):
        """Set the checked state of the entries."""
        if isinstance(new_checks, bool):
            new_checks = [new_checks] * len(self._entries)
        if len(new_checks) != len(self._entries):
            raise ValueError("Passed number of checks does not match the number of entries.")
        new_checks = np.array(new_checks)
        if new_checks.ndim == 1:
            new_checks = new_checks[:, None]
        if new_checks.shape != self.shape:
            new_checks = np.tile(new_checks[:, :1], (1, self.shape[1]))
        for entry, checked in zip(self._entries, new_checks):
            entry.is_checked = checked

    @property
    def history(self):
        # Return the last appended item from the list, removing it from the list
        if self._history:
            return self._history.pop()
        return self.values

    @history.setter
    def history(self, values):
        # Ensure the value is a list of length num_entries
        if (
            isinstance(values, list)
            and len(values) == len(self.values)
            and values not in self._history
        ):
            self._history.append(values)
            # Ensure the length does not exceed 10 entries
            if len(self._history) > 10:
                self._history.pop(0)

    @property
    def has_history(self):
        if self._history:
            return True
        return False

    @property
    def value_dict(self) -> dict:
        """Return a dictionary of the current values with the entry names as keys."""
        if self.num_entries == 1:
            return dict(zip(self.names, [val[0] for val in self.values]))
        return dict(zip(self.names, self.values))

    @property
    def checked_dict(self) -> dict:
        """Return a dictionary of the checked values with the entry names as keys."""
        if self.num_entries == 1:
            return dict(zip(self.checked_names, [val[0] for val in self.checked_values]))
        return dict(zip(self.checked_names, self.checked_values))

    @property
    def unchecked_dict(self) -> dict:
        """Return a dictionary of the unchecked values with the entry names as keys."""
        if self.num_entries == 1:
            return dict(zip(self.unchecked_names, [val[0] for val in self.unchecked_values]))
        return dict(zip(self.unchecked_names, self.unchecked_values))

    def undo_recent(self):
        old_values = self.values.copy()
        while self.has_history and self.values == old_values:
            for entry, val in zip(self.entries, self.history):
                entry.values = val.copy()
        return

    def update_history(self):
        self.history = self.values.copy()

    def apply_connected_funcs(self, entry=None):
        """Apply the connected functions to the entry widgets."""
        if self._connected_funcs is not None:
            if entry is None:
                for entry in self._entries:
                    for widget, func in zip(
                        entry.widgets.of_type(IncLineEdit),
                        self._connected_funcs,
                    ):
                        try:
                            widget.textDelayedChanged.disconnect()  # Disconnect the debounced signal
                        except TypeError:
                            pass  # Ignore if no connection exists
                        widget.textDelayedChanged.connect(func)  # Connect the debounced signal
            else:
                for widget, func in zip(entry.widgets.of_type(IncLineEdit), self._connected_funcs):
                    try:
                        # widget.editingFinished.disconnect()  # Previous form
                        widget.textDelayedChanged.disconnect()  # Disconnect the debounced signal
                    except TypeError:
                        pass  # Ignore if no connection exists
                    widget.textDelayedChanged.connect(func)  # Connect the debounced signal

    def save(self):
        """Save the current values and checked states."""
        for entry in self._entries:
            entry.save_values()

    def destroy(self):
        """Destroy the entries ensuring they are saved."""
        for entry in self._entries:
            entry.destroy()

    def parse_label_width(self, names=None):
        if names is None:
            names = self.names
        if names == []:
            return 40
        label_width = self.label_width
        if isinstance(self.label_width, str):
            if "auto" in self.label_width.lower():
                label_width = "auto"
            elif self.frame is not None and not sip.isdeleted(self.frame):
                font_metrics = QFontMetrics(self.frame.font())
                label_width = max([font_metrics.width(name) for name in names])
                label_width *= 1.25
        elif isinstance(self.label_width, (int, float)):
            label_width = self.label_width
        else:
            label_width = 40
        return label_width

    def append(self, entry, init_values=None, without_frame=False):
        """Add a row to the list."""
        if isinstance(entry, str):
            param_frame = None
            make_widgets = False
            if not without_frame and self.frame is not None:
                param_frame = QFrame(self.frame)
                make_widgets = True

            entry = MultiEntryWidget(
                entry,
                param_frame,
                init_values,
                self.has_value,
                self.has_check,
                make_widgets,
                self.num_entries,
                None,
                self.parse_label_width(self.names + [entry]),
                interval=self.interval,
                fine_interval=self.fine_interval,
            )
        if isinstance(entry, MultiEntryWidget):
            if entry.num_entries == self.shape[1] or self.shape == (0, 0):
                self._history = []
                self._entries.append(entry)
                self.apply_connected_funcs(entry)
            else:
                raise ValueError(
                    "Entry must be a 'MultiEntryWidget' object of the same size as the current entries in order to append."
                )

    def remove(self, index_or_name):
        """Remove a row from the list by index or name."""
        self._history = []
        if isinstance(index_or_name, (list, tuple)):
            # Remove by multiple names
            for name in index_or_name:
                self.remove(name)
        if isinstance(index_or_name, int):
            # Remove by index
            if 0 <= index_or_name < len(self._entries):
                entry = self._entries.pop(index_or_name)
                self._archive[entry.name] = entry
                entry.destroy()  # Call the destroy method
            else:
                raise IndexError("Index out of range.")
        elif isinstance(index_or_name, str):
            # Remove by name using get_names to find the index
            names = self.names
            if index_or_name in names:
                index = names.index(index_or_name)
                entry = self._entries.pop(index)
                self._archive[entry.name] = entry
                entry.destroy()  # Call the destroy method
            else:
                raise ValueError(f"No entry found with name '{index_or_name}'.")
        else:
            raise TypeError("Argument must be an integer index or a string name.")

    def update_entries(self, names, inits=None, frame=None, bounds=None):
        """Update the entries based on the provided list of names."""
        if isinstance(names, dict) and inits is None:
            inits = list(names.values())
            names = list(names.keys())

        if not isinstance(names, (set, tuple, list, np.ndarray)):
            raise ValueError("Names must be a list or tuple of strings.")
        if not isinstance(inits, (set, tuple, list, np.ndarray)):
            inits = [[1] * self.num_entries] * len(names)
        if len(inits) != len(names):
            raise ValueError("inits must be a list or tuple of the same length as the names.")

        current_names = self.names
        if names == self.names:
            # self.update_history()
            self.values = inits
        else:
            self._history = []
            ordered_entries = []

            # Add rows for names not in current entries or restore from archive
            for name, values in zip(names, inits):
                if name not in current_names:
                    if name in self._archive:  # Restore from archive
                        entry = self._archive.pop(name)
                    else:  # Add new row
                        self.append(name, values, without_frame=True)
                        entry = self._entries[-1]
                else:
                    entry = self[name]
                ordered_entries.append(entry)

            # ensures excluded entries are archived properly
            for old_name in current_names:
                if old_name not in names:
                    self.remove(old_name)

            self._entries = ordered_entries

        if bounds is not None:
            for entry, bound in zip(self._entries, bounds):
                entry.bounds = bound

        # self.apply_connected_funcs(entry)
        if frame is not None:
            # Ensure all entries are in an active frame
            self.reset_frame(frame)
        else:
            self.apply_connected_funcs()

    def reset_frame(self, frame, default=False):
        """Reset the frame with the saved values and checked states."""
        if frame is not None:
            self.frame = frame
        elif not (isinstance(self.frame, QFrame) and self.frame.isVisible()):
            raise ValueError("No valid frame provided available.")

        width = self.parse_label_width()
        form_layout = QGridLayout()
        for n, entry in enumerate(self._entries):
            entry.label_width = width
            entry.reset_widget(QFrame(), default, None)
            for i, widget in enumerate(entry.widgets):
                form_layout.addWidget(widget, n, i)
                widget.interval = self.interval
                widget.fine_interval = self.fine_interval

        # Clear old layout if present
        if self.frame.layout() is not None:
            old_layout = self.frame.layout()
            QWidget().setLayout(old_layout)

        # --- Outer layout for the provided frame ---
        outer_layout = QVBoxLayout(self.frame)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # --- Inner content frame that holds the actual parameters ---
        content = QFrame()
        content.setLayout(form_layout)

        # --- Scroll area inside the frame ---
        scroll = QScrollArea(self.frame)
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMaximumHeight(np.clip(self.frame.maximumHeight(), 50, 16777215))
        scroll.setWidget(content)

        outer_layout.addWidget(scroll)

        self.apply_connected_funcs()

    def update_entry_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            for entry in self._entries:
                for widget in entry.widgets:
                    if hasattr(widget, key):
                        setattr(widget, key, value)

    def highlight(
        self,
        key=None,
        reset=True,
        check=None,
        *,
        bold=True,
        italic=False,
        text_size=None,
        color="lightblue",
        border=None,
        tooltip="Measured Data",
        border_radius="4px",
    ):
        """Highlight an entry in the manager with optional styling."""
        if key is None or reset:
            for entry in self._highlighted:
                entry.highlight(
                    bold=False,
                    italic=False,
                    text_size=None,
                    color=None,
                    border=None,
                    tooltip=None,
                    border_radius=None,
                )
                if check is not None:
                    entry.is_checked = [not check] * self.num_entries

        for entry in self._entries:
            if entry.name == key:
                entry.highlight(
                    bold=bold,
                    italic=italic,
                    text_size=text_size,
                    color=color,
                    border=border,
                    tooltip=tooltip,
                    border_radius=border_radius,
                )
                if check is not None:
                    entry.is_checked = [check] * self.num_entries
                self._highlighted.append(entry)
                break
            if entry.name in key:
                entry.highlight(
                    bold=bold,
                    italic=italic,
                    text_size=text_size,
                    color=color,
                    border=border,
                    tooltip=tooltip,
                    border_radius=border_radius,
                )
                if check is not None:
                    entry.is_checked = [check] * self.num_entries
                self._highlighted.append(entry)

    def set_tab_order(self, previous_widget=None, following_widget=None):
        """
        Set the tab order for all widgets in the entries.

        Args:
            previous_widget (QWidget): The widget to set as the previous in the tab order.
            following_widget (QWidget): The widget to set as the next in the tab order.

        Returns:
            QWidget: The last widget in the tab order sequence.
        """
        current_widget = previous_widget
        for entry in self._entries:
            for widget in entry.widgets.of_type(QLineEdit):  # Iterate through QLineEdit widgets
                # for widget, check in zip(entry.widgets.of_type(QLineEdit), entry.widgets.of_type(QCheckBox)):  # Iterate through QLineEdit widgets
                #     if not check.isChecked():
                #         continue
                if current_widget:
                    current_widget.setTabOrder(current_widget, widget)
                current_widget = widget

        if current_widget and following_widget:
            current_widget.setTabOrder(current_widget, following_widget)

        return current_widget  # Return the last widget in the sequence
