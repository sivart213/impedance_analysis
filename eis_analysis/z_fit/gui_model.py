

# import re
# # import sys

# # from copy import deepcopy

# import numpy as np
# import pandas as pd

# import sip
# from PyQt5.QtCore import Qt, QObject, pyqtSignal
# from PyQt5.QtWidgets import (
#     # QApplication,
#     QDialog,
#     QVBoxLayout,
#     QHBoxLayout,
#     QCheckBox,
#     QLabel,
#     QWidget,
#     # QTabWidget,
#     # QTableWidget,
#     QLineEdit,
#     QPushButton,
#     QFrame,
#     QMessageBox,
#     QGridLayout,
#     # QFormLayout,
#     # QDialogButtonBox,
#     # QTreeWidget,
#     # QTreeWidgetItem,
#     # QAbstractItemView,
#     QMainWindow,
#     # QTableWidgetItem,
#     # QTextEdit,
#     # QProgressDialog,
#     # QListWidget,
#     # QListWidgetItem,
#     # QInputDialog,
# )
# from PyQt5.QtGui import QFontMetrics

# from ..string_ops import format_number

# from impedance.models.circuits.fitting import (
#     # wrapCircuit,
#     extract_circuit_elements,
#     calculateCircuitLength,
# )


# def update_model_frame(self, update_plots=True):
#     """Update the model frame with the new model."""
#     self.data.model = self.model_entry.text().strip()
#     self.parameters.interval = self.options["simulation"]["interval"]

#     if self.data.model.lower() == "linkk":
#         param_values = self.data.generate_linkk(
#             self.data.base_df(self.data.primary(), None, "frequency", self.options["simulation"]["area (cm^2)"], self.options["simulation"]["thickness (cm)"]),# dx=self.options["simulation"]["dx"]),
#             area=self.options["simulation"]["area (cm^2)"],
#             thickness=self.options["simulation"]["thickness (cm)"],
#             # dx=self.options["simulation"]["dx"],
#             **self.options["linkk"],
#         )

#         self.parameters.update_entries(
#             ["M", "mu"], param_values, self.param_frame
#         )
#         self.parameters_std.update_entries(["M", "mu"], [0.1, 0.1], None)
#         self.bounds.update_entries(["M", "mu"], [[0, 200], [0, 1]], None)

#     else:
#         param_names = self.data.parse_parameters()

#         self.parameters.update_entries(
#             param_names,
#             [
#                 self.data.parse_default(name, self.options["element"])
#                 for name in param_names
#             ],
#             self.param_frame,
#         )

#         self.parameters_std.update_entries(
#             param_names,
#             [
#                 self.data.parse_default(name, self.options["element"])
#                 * 0.1
#                 for name in param_names
#             ],
#             None,
#         )

#         self.bounds.update_entries(
#             param_names,
#             [
#                 self.data.parse_default(
#                     name, self.options["element_range"]
#                 )
#                 for name in param_names
#             ],
#             None,
#         )

#     self.ci_inputs = None
#     # self.print_window.write("Model updated")
#     if update_plots:
#         self.update_graphs()

# class DataParser:
#     """
#     Class to handle data parsing and labeling.

#     Responsibilities:
#     - Parse default values for parameters.
#     - Generate labels based on selected options.
#     - Extract and parse model parameters.

#     Integration:
#     - Use this class to parse and label data for visualization and analysis.
#     - Instantiate with model, var_units, and var_val, and call methods like `parse_default`, `parse_label`, and `parse_parameters`.
#     """

#     def __init__(self, model, var_units, var_val):
#         self.model = model
#         self.var_units = var_units
#         self.var_val = var_val

#     def parse_default(self, name, defaults, override=None):
#         """Parse the default value for the parameter."""
#         if override is not None and name in override.keys():
#             return override[name]
#         if self.model.lower() == "linkk":
#             return 1
#         if name in defaults.keys():
#             return defaults[name]
#         split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
#         if split_name[0] not in defaults.keys():
#             return 1
#         if "_" in split_name[1]:
#             index = int(eval(split_name[1].split("_")[-1], {}, {"inf": np.inf}))
#             return defaults[split_name[0]][index]
#         return defaults[split_name[0]]

#     def parse_label(self, main, mode):
#         """Parse the label based on the selected option."""
#         units = self.var_units[main]
#         if mode.lower() == "real":
#             return f"{main}' {units}"
#         elif mode.lower() == "imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "+imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "mag":
#             return f"|{main}| {units}"
#         elif mode == "phase" or self.var_val[mode] == "phase":
#             return "θ [deg]"
#         elif mode == "tan" or self.var_val[mode] == "tan":
#             return "tan(δ) [1]"

#     def parse_parameters(self, model=None):
#         """Get the parameters of the model."""
#         if model is None:
#             model = self.model
#         if not isinstance(model, str):
#             QMessageBox.warning(None, "Warning", f"Model must be a string. Current value is {model} of type {type(model)}")
#             model = self.model
#         if model.lower() == "linkk":
#             return ["M", "mu"]
#         params = extract_circuit_elements(model)
#         if len(params) != calculateCircuitLength(model):
#             all_params = []
#             for param in params:
#                 length = calculateCircuitLength(param)
#                 if length >= 2:
#                     all_params.append(f"{param}_0")
#                     for i in range(1, length):
#                         all_params.append(f"{param}_{i}")
#                 else:
#                     all_params.append(param)
#             params = all_params
#         return params


# class ModelManager(QObject):
#     """
#     Class to manage the model and its parameters.

#     Responsibilities:
#     - Manage the model equation and its parameters.
#     - Retrieve constants/parameters from the model.
#     - Use default values for parameters.
#     - Manage parameters through MultiEntryManager.

#     Integration:
#     - Use this class to manage model-related operations in the application.
#     - Connect to the `model_changed` signal to handle model updates.
#     """

#     model_changed = pyqtSignal()  # Signal to notify model changes

#     def __init__(self, model, options, param_frame=None):
#         super().__init__()
#         self.model = model
#         self.options = options
#         self.param_frame = param_frame

#         # Initialize DataParser
#         self.data_parser = DataParser(model, options["var_units"], options["var_val"])

#         # Initialize MultiEntryManager for parameters
#         self.parameters = MultiEntryManager(
#             param_frame,
#             callback=self.update_parameters,
#             interval=options["simulation"]["interval"],
#         )

#     def update_model(self, new_model):
#         """Update the model and notify changes."""
#         self.model = new_model
#         self.data_parser.model = new_model
#         self.update_parameters()
#         self.model_changed.emit()  # Emit signal when model changes

#     def update_parameters(self):
#         """Update the parameters based on the current model."""
#         if self.model.lower() == "linkk":
#             param_values = self.generate_linkk_params()
#             self.parameters.update_entries(["M", "mu"], param_values, self.param_frame)
#         else:
#             param_names = self.data_parser.parse_parameters()
#             param_values = [
#                 self.data_parser.parse_default(name, self.options["element"])
#                 for name in param_names
#             ]
#             self.parameters.update_entries(param_names, param_values, self.param_frame)

#     def generate_linkk_params(self):
#         """Generate parameters for the Lin-KK model."""
#         base_df = self.data_parser.base_df(
#             self.data_parser.primary(), None, "frequency",
#             self.options["simulation"]["area (cm^2)"],
#             self.options["simulation"]["thickness (cm)"]
#         )
#         param_values = self.data_parser.generate_linkk(
#             base_df,
#             area=self.options["simulation"]["area (cm^2)"],
#             thickness=self.options["simulation"]["thickness (cm)"],
#             **self.options["linkk"]
#         )
#         return param_values

#     def get_parameters(self):
#         """Get the current parameters."""
#         return self.parameters.values

#     def set_parameters(self, param_values):
#         """Set new parameter values."""
#         self.parameters.values = param_values
#         self.model_changed.emit()  # Emit signal when parameters change

#     def get_default_parameters(self):
#         """Get the default parameter values."""
#         param_names = self.data_parser.parse_parameters()
#         return [
#             self.data_parser.parse_default(name, self.options["element"])
#             for name in param_names
#         ]

#     def get_parameter_bounds(self):
#         """Get the parameter bounds."""
#         param_names = self.data_parser.parse_parameters()
#         return [
#             self.data_parser.parse_default(name, self.options["element_range"])
#             for name in param_names
#         ]

# class MultiEntryWidget:
#     """Class to create a custom entry widget."""

#     def __init__(
#         self,
#         name,
#         frame,
#         init_values=None,
#         has_value=True,
#         has_check=True,
#         set_settings=True,
#         num_entries=1,
#         default_values=None,
#         label_width="auto",
#         layout=QHBoxLayout,
#         interval=1,
#     ):
#         self.name = name
#         self.has_value = has_value
#         self.has_check = has_check
#         self.num_entries = num_entries
#         self.frame = frame
#         self.main_layout = layout
#         self.frame_dead = False
#         self.label_width = label_width
#         self.accents = None
#         self.widgets = TypeList([])
#         self.interval = interval

#         if init_values is None:
#             init_values = [1] if default_values is None else default_values

#         init_values = self._list_check(init_values)

#         self.default_values = (
#             self._list_check(default_values)
#             if default_values is not None
#             else init_values
#         )

#         self._values = init_values
#         self._checked = [False] * num_entries

#         if self.frame is not None and set_settings:
#             self.create_widgets(
#                 frame, init_values, num_entries, has_value, has_check
#             )
#         else:
#             self.frame_dead = True

#     def create_widgets(
#         self,
#         frame,
#         values,
#         num_entries=1,
#         has_value=True,
#         has_check=True,
#         layout="default",
#     ):
#         """Create the widgets for the entry."""
#         self.widgets = TypeList([])

#         label = QLabel(self.name)
#         text_width = 40
#         if self.label_width == "auto":
#             label.setFixedWidth(QFontMetrics(label.font()).width(self.name))
#             font_metrics = QFontMetrics(label.font())
#             text_width = font_metrics.width(label.text())
#             text_width += text_width * 0.2
#         elif isinstance(self.label_width, (int, float)):
#             text_width = self.label_width
#         label.setFixedWidth(int(text_width))
#         self.widgets.append(label)

#         values = self._list_check(values)

#         for i in range(num_entries):
#             if has_value:
#                 entry = IncLineEdit()
#                 entry.setText(
#                     format_number(
#                         eval(str(values[i]), {}, {"inf": np.inf}),
#                         precision=5,
#                     )
#                 )
#                 entry.interval = self.interval
#                 self.widgets.append(entry)

#             if has_check:
#                 check_var = QCheckBox()
#                 self.widgets.append(check_var)

#         self.apply_layout(frame, layout)

#         if self.accents is not None:
#             self.highlight(**self.accents)

#     def apply_layout(self, frame, layout="default"):
#         """Add the entry widgets to a layout."""
#         if self.frame is None or self.frame_dead:
#             self.frame = frame if frame is not None else QFrame()
#             self.frame_dead = False

#         # Remove existing layout if it exists
#         if self.frame.layout() is not None:
#             old_layout = self.frame.layout()
#             QWidget().setLayout(old_layout)

#         if layout is None:
#             return

#         if (
#             "Q" not in str(type(layout))
#             or "Layout" not in str(type(layout))
#             or layout == "default"
#         ):
#             layout = QHBoxLayout

#         self.main_layout = layout()
#         self.frame.setLayout(self.main_layout)

#         for widget in self.widgets:
#             self.main_layout.addWidget(widget)

#     def save_values(self):
#         """Save the current values and checked states."""
#         self._values = self.values
#         self._checked = self.is_checked

#     def reset_widget(self, frame, default=False, layout="default"):
#         """Reset the frame with the saved values and checked states."""
#         if not self.frame_dead:
#             self.destroy()
#         self.frame = frame

#         values_to_use = self.default_values if default else self._values

#         self.create_widgets(
#             frame,
#             values_to_use,
#             self.num_entries,
#             self.has_value,
#             self.has_check,
#             layout,
#         )

#         self.is_checked = self._checked
#         self.frame_dead = False

#     def uncheck(self):
#         """Save the current values and checked states."""
#         self._values = [False] * self.num_entries

#     def destroy(self, error_call=False):
#         """Destroy the entry widgets and the parent frame."""
#         if not error_call:
#             self.save_values()
#         if hasattr(self, "entry_widgets"):
#             # for widget in self.entry_widgets + self.check_vars:
#             for widget in self.widgets.of_type(IncLineEdit, QCheckBox):
#                 if widget is not None and not sip.isdeleted(widget):
#                     widget.deleteLater()
#         if self.frame is not None and not sip.isdeleted(self.frame):
#             self.frame.deleteLater()

#         self.widgets = TypeList([])
#         self.frame_dead = True

#     def _list_check(self, values):
#         if not isinstance(values, (list, tuple, np.ndarray)):
#             values = [values] * self.num_entries
#         if len(values) != self.num_entries:
#             values = [values[0]] * self.num_entries
#         return values

#     @property
#     def values(self):
#         """Get the current values of the entry widgets."""
#         if not self.frame_dead:
#             try:
#                 if self.has_value:
#                     self._values = [
#                         eval(entry.text().lower(), {}, {"inf": np.inf})
#                         for entry in self.widgets.of_type(IncLineEdit)
#                     ]
#             except (ValueError, SyntaxError):
#                 pass  # Ignore invalid entries and keep the current values
#             except AttributeError:
#                 self.destroy(True)
#         return self._values

#     @values.setter
#     def values(self, new_values):
#         """Set new values in the entry widgets."""
#         new_values = self._list_check(new_values)

#         if not self.frame_dead:
#             try:
#                 if self.has_value:
#                     for entry, new_value in zip(
#                         self.widgets.of_type(IncLineEdit), new_values
#                     ):
#                         entry.setText(
#                             format_number(
#                                 eval(str(new_value), {}, {"inf": np.inf}),
#                                 precision=5,
#                             )
#                         )
#                     self._values = new_values
#                     return
#             except (ValueError, SyntaxError):
#                 pass  # Ignore invalid entries and keep the current values
#             except AttributeError:
#                 self.destroy(True)
#         self._values = new_values

#     @property
#     def is_checked(self):
#         """Check if the entries are selected (checked)."""
#         if not self.frame_dead:
#             try:
#                 if self.has_check:
#                     self._checked = [
#                         check_var.isChecked()
#                         for check_var in self.widgets.of_type(QCheckBox)
#                     ]
#             except (ValueError, SyntaxError):
#                 pass  # Ignore invalid entries and keep the current values
#             except AttributeError:
#                 self.destroy(True)
#         return self._checked

#     @is_checked.setter
#     def is_checked(self, checked):
#         """Set the check state of the entries."""
#         checked = self._list_check(checked)

#         if not self.frame_dead:
#             try:
#                 if self.has_check:
#                     for check_var, state in zip(
#                         self.widgets.of_type(QCheckBox), checked
#                     ):
#                         check_var.setChecked(state)
#                     self._checked = checked
#                     return
#             except (ValueError, SyntaxError):
#                 pass  # Ignore invalid entries and keep the current values
#             except AttributeError:
#                 self.destroy(True)
#         self._checked = checked

#     def highlight(
#         self,
#         *,
#         bold=True,
#         italic=True,
#         text_size=None,
#         color=None,
#         border=None,
#         tooltip=None,
#         border_radius=None,
#     ):
#         """
#         Highlight the widget with optional styling.

#         Parameters:
#         - bold (bool): Set to True to make the label text bold.
#         - italic (bool): Set to True to italicize the label text.
#         - text_size (int): Set the text size of the label.
#         - color (str): Set the background color of the frame. Example: "lightblue", "yellow".
#         - border (str): Set the border style of the frame. Example: "2px solid red", "1px dashed black".
#         - tooltip (str): Set the tooltip text for the label.
#         - border_radius (str): Set the border radius of the frame. Example: "10px".

#         Example usage:
#         entry.highlight(bold=True, italic=True, text_size=14, color="lightblue", border="2px solid red", tooltip="This is a tooltip", border_radius="10px")
#         """
#         self.accents = dict(
#             bold=bold,
#             italic=italic,
#             text_size=text_size,
#             color=color,
#             border=border,
#             tooltip=tooltip,
#             border_radius=border_radius,
#         )

#         if self.frame is None or self.frame_dead:
#             return

#         # label = self.main_layout.itemAt(0).widget()
#         label = self.widgets[0]
#         font = label.font()

#         if bold:
#             font.setBold(True)
#         if italic:
#             font.setItalic(True)
#         if text_size:
#             font.setPointSize(text_size)
#         if tooltip:
#             label.setToolTip(tooltip)

#         label.setFont(font)

#         style = ""
#         if color:
#             style += f"background-color: {color};"
#         if border:
#             style += f"border: {border};"
#         if border_radius:
#             style += f"border-radius: {border_radius};"

#         self.frame.setStyleSheet(style)


# class MultiEntryManager:
#     """Class to create a list of custom entry widgets."""

#     def __init__(
#         self,
#         frame,
#         entries=None,
#         num_entries=1,
#         has_value=True,
#         has_check=True,
#         label_width="uniform",
#         callback=None,
#         interval=1,
#     ):
#         self.frame = frame
#         self._entries = []
#         self._archive = {}
#         self.has_value = has_value
#         self.has_check = has_check
#         self.num_entries = num_entries
#         self.label_width = label_width
#         self._highlighted = []
#         self.connected_funcs = callback
#         self._history = []
#         self.interval = interval

#         if entries is not None:
#             for entry in entries:
#                 if isinstance(entry, MultiEntryWidget):
#                     self.append(entry)
#                 elif isinstance(entry, dict):
#                     self.append(**entry)
#                 elif isinstance(entry, (tuple, list)):
#                     self.append(*entry)
#                 else:
#                     self.append(str(entry))

#     def __iter__(self):
#         return iter(self._entries)

#     def __getitem__(self, key):
#         if isinstance(key, int):
#             return self._entries[key]
#         for entry in self._entries:
#             if entry.name == key:
#                 return entry
#         raise KeyError(
#             f"'MultiEntryManager' object has no entry row named '{key}'"
#         )

#     def __setitem__(self, key, value):
#         if not isinstance(value, (list, tuple, np.ndarray)):
#             value = [value] * self.num_entries
#         for entry in self._entries:
#             if entry.name == key:
#                 if all(isinstance(v, bool) for v in value):
#                     entry.is_checked = value
#                 else:
#                     self.update_history()
#                     entry.values = value
#                 return
#         self.append(key, value)

#     def __bool__(self):
#         """Evaluate the instance as True if there are any entries."""
#         return bool(self._entries)

#     def __contains__(self, item):
#         return item in self.names

#     @property
#     def shape(self):
#         """Get the shape of the entries."""
#         if self._entries:
#             return (len(self._entries), self.num_entries)
#         return (0, self.num_entries)

#     @property
#     def entries(self):
#         """Get the entries."""
#         return self._entries

#     @property
#     def highlighted(self):
#         """Get the entries."""
#         return [entry.name for entry in self._highlighted]

#     @property
#     def _df(self):
#         """Get all values of the entries as a DataFrame with entry names as index."""
#         return pd.DataFrame(
#             [entry.values for entry in self._entries], dtype="object"
#         )

#     @property
#     def connected_funcs(self):
#         """Get the connected functions."""
#         return self._connected_funcs

#     @connected_funcs.setter
#     def connected_funcs(self, funcs):
#         """Set the connected functions."""
#         if funcs is None:
#             self._connected_funcs = None
#         elif callable(funcs):
#             self._connected_funcs = [funcs] * self.num_entries
#         elif isinstance(funcs, (list, tuple)) and all(
#             callable(func) for func in funcs
#         ):
#             if len(funcs) == self.num_entries:
#                 self._connected_funcs = funcs
#             else:
#                 self._connected_funcs = [funcs[0]] * self.num_entries
#         else:
#             raise ValueError(
#                 "Connected functions must be callable or a list of callables."
#             )
#         self.apply_connected_funcs()

#     @property
#     def values(self):
#         """Get all values of the entries."""
#         return self._df.to_numpy().tolist()

#     @values.setter
#     def values(self, new_values):
#         """Set all values of the entries."""
#         if not hasattr(new_values, "__len__") or len(new_values) != len(
#             self._entries
#         ):
#             raise ValueError(
#                 "Passed number of values does not match the number of entries."
#             )
#         self.update_history()
#         for entry, values in zip(self._entries, new_values):
#             entry.values = values

#     @property
#     def checked_values(self):
#         """Get the values of the checked entries."""
#         checks = np.array(self.checks)
#         return self._df[checks].to_numpy().tolist()

#     @checked_values.setter
#     def checked_values(self, new_values):
#         """Set the values of the checked entries."""
#         checks = np.array(self.checks)
#         df = self._df.copy()
#         if (
#             np.array(new_values).shape != df.shape
#             and len(new_values) != checks.sum()
#         ):
#             raise ValueError(
#                 "Passed number of values does not match the number of checked entries."
#             )
#         self.update_history()
#         df[checks] = new_values
#         for entry, values in zip(self._entries, df.to_numpy().tolist()):
#             entry.values = values

#     @property
#     def unchecked_values(self):
#         """Get the values of the unchecked entries."""
#         checks = np.array(self.checks)
#         return self._df[~checks].to_numpy().tolist()

#     @unchecked_values.setter
#     def unchecked_values(self, new_values):
#         """Set the values of the unchecked entries."""
#         checks = np.array(self.checks)
#         df = self._df.copy()
#         if (
#             np.array(new_values).shape != df.shape
#             and len(new_values) != (~checks).sum()
#         ):
#             raise ValueError(
#                 "Passed number of values does not match the number of unchecked entries."
#             )
#         self.update_history()
#         df[~checks] = new_values
#         for entry, values in zip(self._entries, df.to_numpy().tolist()):
#             entry.values = values

#     @property
#     def names(self):
#         """Get all names of the entries."""
#         return [entry.name for entry in self._entries]

#     @names.setter
#     def names(self, new_names):
#         """Set all names of the entries."""
#         if len(new_names) != len(self._entries):
#             raise ValueError(
#                 "Passed number of names does not match the number of entries."
#             )
#         for entry, name in zip(self._entries, new_names):
#             entry.name = name

#     @property
#     def checked_names(self):
#         """Get the names of the checked entries."""
#         checks = np.array(self.checks)
#         return [
#             entry.name for entry, check in zip(self._entries, checks) if check
#         ]

#     @checked_names.setter
#     def checked_names(self, new_names):
#         """Set the names of the checked entries."""
#         checks = np.array(self.checks)
#         if len(new_names) != checks.sum():
#             raise ValueError(
#                 "Passed number of names does not match the number of checked entries."
#             )
#         for entry, name in zip(
#             [entry for entry, check in zip(self._entries, checks) if check],
#             new_names,
#         ):
#             entry.name = name

#     @property
#     def unchecked_names(self):
#         """Get the names of the unchecked entries."""
#         checks = np.array(self.checks)
#         return [
#             entry.name
#             for entry, check in zip(self._entries, checks)
#             if not check
#         ]

#     @unchecked_names.setter
#     def unchecked_names(self, new_names):
#         """Set the names of the unchecked entries."""
#         checks = np.array(self.checks)
#         if len(new_names) != (~checks).sum():
#             raise ValueError(
#                 "Passed number of names does not match the number of unchecked entries."
#             )
#         for entry, name in zip(
#             [
#                 entry
#                 for entry, check in zip(self._entries, checks)
#                 if not check
#             ],
#             new_names,
#         ):
#             entry.name = name

#     @property
#     def checks(self):
#         """Get the checked state of the entries."""
#         return [entry.is_checked for entry in self._entries]

#     @checks.setter
#     def checks(self, new_checks):
#         """Set the checked state of the entries."""
#         if isinstance(new_checks, bool):
#             new_checks = [new_checks] * len(self._entries)
#         if len(new_checks) != len(self._entries):
#             raise ValueError(
#                 "Passed number of checks does not match the number of entries."
#             )
#         new_checks = np.array(new_checks)
#         if new_checks.ndim == 1:
#             new_checks = new_checks[:, None]
#         if new_checks.shape != self.shape:
#             new_checks = np.tile(new_checks[:, :1], (1, self.shape[1]))
#         for entry, checked in zip(self._entries, new_checks):
#             entry.is_checked = checked

#     @property
#     def history(self):
#         # Return the last appended item from the list, removing it from the list
#         if self._history:
#             return self._history.pop()
#         return self.values

#     @history.setter
#     def history(self, values):
#         # Ensure the value is a list of length num_entries
#         if (
#             isinstance(values, list)
#             and len(values) == len(self.values)
#             and values not in self._history
#         ):
#             self._history.append(values)
#             # Ensure the length does not exceed 10 entries
#             if len(self._history) > 10:
#                 self._history.pop(0)

#     @property
#     def has_history(self):
#         if self._history:
#             return True
#         return False
    
#     @property
#     def value_dict(self):
#         """Return a dictionary of the current values with the entry names as keys."""
#         if self.num_entries == 1:
#             return dict(zip(self.names, [val[0] for val in self.values]))
#         return dict(zip(self.names, self.values))
    
#     @property
#     def checked_dict(self):
#         """Return a dictionary of the checked values with the entry names as keys."""
#         if self.num_entries == 1:
#             return dict(zip(self.checked_names, [val[0] for val in self.checked_values]))
#         return dict(zip(self.checked_names, self.checked_values))
    
#     @property
#     def unchecked_dict(self):
#         """Return a dictionary of the unchecked values with the entry names as keys."""
#         if self.num_entries == 1:
#             return dict(zip(self.unchecked_names, [val[0] for val in self.unchecked_values]))
#         return dict(zip(self.unchecked_names, self.unchecked_values))

#     def undo_recent(self):
#         old_values = self.values.copy()
#         while self.has_history and self.values == old_values:
#             for entry, val in zip(self.entries, self.history):
#                 entry.values = val.copy()
#         return

#     def update_history(self):
#         self.history = self.values.copy()

#     def apply_connected_funcs(self, entry=None):
#         """Apply the connected functions to the entry widgets."""
#         if self._connected_funcs is not None:
#             if entry is None:
#                 for entry in self._entries:
#                     for widget, func in zip(
#                         entry.widgets.of_type(IncLineEdit),
#                         self._connected_funcs,
#                     ):
#                         try:
#                             widget.editingFinished.disconnect()
#                         except TypeError:
#                             pass  # Ignore if no connection exists
#                         widget.editingFinished.connect(func)
#             else:
#                 for widget, func in zip(
#                     entry.widgets.of_type(IncLineEdit), self._connected_funcs
#                 ):
#                     try:
#                         widget.editingFinished.disconnect()
#                     except TypeError:
#                         pass  # Ignore if no connection exists
#                     widget.editingFinished.connect(
#                         func
#                     )  # Connect the new function

#     def save(self):
#         """Save the current values and checked states."""
#         for entry in self._entries:
#             entry.save_values()

#     def destroy(self):
#         """Destroy the entries ensuring they are saved."""
#         for entry in self._entries:
#             entry.destroy()

#     def parse_label_width(self, names=None):
#         if names is None:
#             names = self.names
#         if names == []:
#             return 40
#         label_width = self.label_width
#         if isinstance(self.label_width, str):
#             if "auto" in self.label_width.lower():
#                 label_width = "auto"
#             elif self.frame is not None and not sip.isdeleted(self.frame):
#                 font_metrics = QFontMetrics(self.frame.font())
#                 label_width = max([font_metrics.width(name) for name in names])
#                 label_width *= 1.25
#         elif isinstance(self.label_width, (int, float)):
#             label_width = self.label_width
#         else:
#             label_width = 40
#         return label_width

#     def append(self, entry, init_values=None, without_frame=False):
#         """Add a row to the list."""
#         if isinstance(entry, str):
#             param_frame = None
#             make_widgets = False
#             if not without_frame and self.frame is not None:
#                 param_frame = QFrame(self.frame)
#                 make_widgets = True

#             entry = MultiEntryWidget(
#                 entry,
#                 param_frame,
#                 init_values,
#                 self.has_value,
#                 self.has_check,
#                 make_widgets,
#                 self.num_entries,
#                 None,
#                 self.parse_label_width(self.names + [entry]),
#                 interval=self.interval,
#             )
#         if isinstance(entry, MultiEntryWidget):
#             if entry.num_entries == self.shape[1] or self.shape == (0, 0):
#                 self._history = []
#                 self._entries.append(entry)
#                 self.apply_connected_funcs(entry)
#             else:
#                 raise ValueError(
#                     "Entry must be a 'MultiEntryWidget' object of the same size as the current entries in order to append."
#                 )

#     def remove(self, index_or_name):
#         """Remove a row from the list by index or name."""
#         self._history = []
#         if isinstance(index_or_name, (list, tuple)):
#             # Remove by multiple names
#             for name in index_or_name:
#                 self.remove(name)
#         if isinstance(index_or_name, int):
#             # Remove by index
#             if 0 <= index_or_name < len(self._entries):
#                 entry = self._entries.pop(index_or_name)
#                 self._archive[entry.name] = entry
#                 entry.destroy()  # Call the destroy method
#             else:
#                 raise IndexError("Index out of range.")
#         elif isinstance(index_or_name, str):
#             # Remove by name using get_names to find the index
#             names = self.names
#             if index_or_name in names:
#                 index = names.index(index_or_name)
#                 entry = self._entries.pop(index)
#                 self._archive[entry.name] = entry
#                 entry.destroy()  # Call the destroy method
#             else:
#                 raise ValueError(
#                     f"No entry found with name '{index_or_name}'."
#                 )
#         else:
#             raise TypeError(
#                 "Argument must be an integer index or a string name."
#             )

#     def update_entries(self, names, inits=None, frame=None):
#         """Update the entries based on the provided list of names."""
#         if isinstance(names, dict) and inits is None:
#             inits = list(names.values())
#             names = list(names.keys())

#         if not isinstance(names, (set, tuple, list, np.ndarray)):
#             raise ValueError("Names must be a list or tuple of strings.")
#         if not isinstance(inits, (set, tuple, list, np.ndarray)):
#             inits = [[1] * self.num_entries] * len(names)
#         if len(inits) != len(names):
#             raise ValueError(
#                 "inits must be a list or tuple of the same length as the names."
#             )

#         current_names = self.names
#         if names == self.names:
#             self.update_history()
#             self.values = inits
#         else:
#             self._history = []
#             ordered_entries = []

#             # Add rows for names not in current entries or restore from archive
#             for name, values in zip(names, inits):
#                 if name not in current_names:
#                     if name in self._archive:  # Restore from archive
#                         entry = self._archive.pop(name)
#                     else:  # Add new row
#                         self.append(name, values, without_frame=True)
#                         entry = self._entries[-1]
#                 else:
#                     entry = self[name]
#                 ordered_entries.append(entry)

#             # ensures excluded entries are archived properly
#             for old_name in current_names:
#                 if old_name not in names:
#                     self.remove(old_name)

#             self._entries = ordered_entries

#         # self.apply_connected_funcs(entry)
#         if frame is not None:
#             # Ensure all entries are in an active frame
#             self.reset_frame(frame)
#         else:
#             self.apply_connected_funcs()

#     def reset_frame(self, frame, default=False):
#         """Reset the frame with the saved values and checked states."""
#         if frame is not None:
#             self.frame = frame
#         elif not (isinstance(self.frame, QFrame) and self.frame.isVisible()):
#             raise ValueError("No valid frame provided available.")

#         width = self.parse_label_width()
#         form_layout = QGridLayout()
#         for n, entry in enumerate(self._entries):
#             entry.label_width = width
#             entry.reset_widget(QFrame(), default, None)
#             for i, widget in enumerate(entry.widgets):
#                 form_layout.addWidget(widget, n, i)
#                 widget.interval = self.interval

#         self.apply_connected_funcs()

#         if self.frame.layout() is not None:
#             old_layout = self.frame.layout()
#             QWidget().setLayout(old_layout)
#         self.frame.setLayout(form_layout)

#     def update_interval(self, interval):
#         self.interval = interval
#         for entry in self._entries:
#             for widget in entry.widgets:
#                 widget.interval = self.interval

#     def highlight(
#         self,
#         key=None,
#         reset=True,
#         check=None,
#         *,
#         bold=True,
#         italic=False,
#         text_size=None,
#         color="lightblue",
#         border=None,
#         tooltip="Measured Data",
#         border_radius="4px",
#     ):
#         """Highlight an entry in the manager with optional styling."""
#         if key is None or reset:
#             for entry in self._highlighted:
#                 entry.highlight(
#                     bold=False,
#                     italic=False,
#                     text_size=None,
#                     color=None,
#                     border=None,
#                     tooltip=None,
#                     border_radius=None,
#                 )
#                 if check is not None:
#                     entry.is_checked = [not check] * self.num_entries

#         for entry in self._entries:
#             if entry.name == key:
#                 entry.highlight(
#                     bold=bold,
#                     italic=italic,
#                     text_size=text_size,
#                     color=color,
#                     border=border,
#                     tooltip=tooltip,
#                     border_radius=border_radius,
#                 )
#                 if check is not None:
#                     entry.is_checked = [check] * self.num_entries
#                 self._highlighted.append(entry)
#                 break
#             if entry.name in key:
#                 entry.highlight(
#                     bold=bold,
#                     italic=italic,
#                     text_size=text_size,
#                     color=color,
#                     border=border,
#                     tooltip=tooltip,
#                     border_radius=border_radius,
#                 )
#                 if check is not None:
#                     entry.is_checked = [check] * self.num_entries
#                 self._highlighted.append(entry)


# class MultiEntryWindow(MultiEntryManager):
#     def __init__(
#         self,
#         root,
#         entries=None,
#         num_entries=1,
#         has_value=True,
#         has_check=True,
#         callbacks=None,
#     ):
#         super().__init__(None, entries, num_entries, has_value, has_check)
#         self.root = root
#         self.window = None
#         self.callbacks = callbacks if isinstance(callbacks, dict) else {}

#     def show(self):
#         """Create and show the multi-entry window."""
#         self.window = QDialog(self.root)
#         self.window.setWindowTitle("Add Entries")
#         self.window.finished.connect(self.save)
#         self.window.setMinimumSize(200, 100)

#         main_layout = QVBoxLayout(self.window)

#         self.frame = QFrame(self.window)

#         self.reset_frame(self.frame)

#         # Create a frame for the buttons
#         button_frame = QFrame(self.window)
#         button_layout = QVBoxLayout(button_frame)

#         # Create frames for grouping buttons
#         base_frame = QFrame(button_frame)
#         base_layout = QHBoxLayout(base_frame)
#         button_layout.addWidget(base_frame)

#         # Create buttons
#         save_button = QPushButton("Save", base_frame)
#         save_button.clicked.connect(self.save)
#         base_layout.addWidget(save_button)

#         clear_button = QPushButton("Clear", base_frame)
#         clear_button.clicked.connect(self.clear)
#         base_layout.addWidget(clear_button)

#         if self.callbacks:
#             callback_frame = QFrame(button_frame)
#             callback_layout = QVBoxLayout(callback_frame)
#             button_layout.addWidget(callback_frame)

#             for key in self.callbacks.keys():
#                 if key.startswith("button_"):
#                     name = key.replace("button_", "").replace("_", " ").title()
#                     button = QPushButton(name, callback_frame)
#                     button.clicked.connect(self.callbacks[key])
#                     callback_layout.addWidget(button)

#         main_layout.addWidget(self.frame)
#         main_layout.addWidget(button_frame)

#         self.window.setLayout(main_layout)
#         self.window.exec_()

#     def save(self):
#         """Save the entries to the parameters and verify their validity."""

#         if self.callbacks.get("save"):
#             res = self.callbacks["save"](np.array(self.values))
#             if res is not None:
#                 self.values = res

#         self.destroy()
#         # Destroy entry window if it exists
#         if self.window:
#             self.window.close()
#             self.window = None
#         if self.frame:
#             self.frame.close()
#             self.frame = None

#     def clear(self):
#         """Clear the entries from the parameters."""
#         clear_callback = self.callbacks.get("clear", None)
#         checked = self.callbacks.get("clear_flag", "unchecked")
#         defaults = None
#         if callable(clear_callback):
#             defaults = clear_callback(self.values)

#         if defaults is None:
#             defaults = np.array(
#                 [entry.default_values for entry in self.entries]
#             )
#         if checked is True or checked == "checked":
#             self.checked_values = defaults
#         elif checked is False or checked == "unchecked":
#             self.unchecked_values = defaults
#         else:
#             self.values = defaults

# class TypeList(list):
#     """Class to create a list with 'type' information."""

#     def __init__(self, values):
#         super().__init__(values)

#     def of_type(self, *item_type):
#         """Return a list of items of the given type."""
#         if len(item_type) == 0:
#             return self
#         if len(item_type) == 1:
#             item_type = item_type[0]
#         if isinstance(item_type, str):
#             return [
#                 item
#                 for item in self
#                 if item_type.lower() in str(type(item)).lower()
#             ]
#         elif isinstance(item_type, type) or (
#             isinstance(item_type, tuple)
#             and all(isinstance(i, type) for i in item_type)
#         ):
#             return [item for item in self if isinstance(item, item_type)]
#         return []


# class IncLineEdit(QLineEdit):
#     """Class to create a custom line edit widget."""

#     def __init__(self, *args, **kwargs):
#         self.interval = kwargs.pop("interval", 1)
#         self.precision = kwargs.pop("precision", 5)
#         self.upper_exponent = kwargs.pop("upper_exponent", 3)
#         self.lower_exponent = kwargs.pop("lower_exponent", None)
#         super().__init__(*args, **kwargs)
#         self.setText("0")

#     def keyPressEvent(self, event):
#         """Handle key press events."""
#         if event.key() == Qt.Key_Up:
#             self.increment_value()
#         elif event.key() == Qt.Key_Down:
#             self.decrement_value()
#         else:
#             super().keyPressEvent(event)

#     def increment_value(self):
#         """Increment the value of the line edit."""
#         try:
#             texts = self.text().split("e")
#             texts[0] = str(float(texts[0]) + self.interval)
#             self.setText(
#                 format_number(
#                     float("e".join(texts)),
#                     self.precision,
#                     self.upper_exponent,
#                     self.lower_exponent,
#                 )
#             )
#         except ValueError:
#             pass

#     def decrement_value(self):
#         """Decrement the value of the line edit."""
#         try:
#             texts = self.text().split("e")
#             if len(texts) == 2 and float(texts[0]) == self.interval:
#                 texts[0] = str(float(texts[0]) - self.interval / 10)
#             else:
#                 texts[0] = str(float(texts[0]) - self.interval)
#             self.setText(
#                 format_number(
#                     float("e".join(texts)),
#                     self.precision,
#                     self.upper_exponent,
#                     self.lower_exponent,
#                 )
#             )
#         except ValueError:
#             pass