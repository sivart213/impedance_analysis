# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from typing import Any, Literal, TypeAlias, overload
from dataclasses import field, dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
from PyQt5 import sip  # type: ignore
from PyQt5.QtCore import Qt, QEvent, QEventLoop, QAbstractListModel
from PyQt5.QtWidgets import (
    QMenu,
    QFrame,
    QAction,
    QDialog,
    QWidget,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QInputDialog,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)

from ..string_ops import safe_eval, find_common_str
from .data_handlers import DataManager

# from ..data_treatment.data_analysis import ComplexSystem
from ..z_system.system import ComplexSystem

# from .gui_workers import DataHandler
from .parameter_widgets import MultiEntryManager
from ..widgets.data_view import DataViewer
from ..widgets.generic_widgets import FormDialog

IsFalse: TypeAlias = Literal[False]
IsTrue: TypeAlias = Literal[True]


def get_valid_name(
    parent,
    default_name,
    existing_name,
    input_prompt="Enter the name for the new dataset:",
    overwrite_warning="The dataset '{}' will be overwritten. Do you want to continue?",
    input_title="Name Dataset",
    overwrite_title="Overwrite Dataset",
):
    """
    Helper function to get a valid name for a dataset.

    Args:
        parent (QWidget): The parent widget for dialogs.
        default_name (str): The default name to suggest in the input dialog.
        existing_name (str): The name of the existing dataset.
        input_prompt (str): The prompt text for the input dialog.
        overwrite_warning (str): The warning text for the overwrite confirmation dialog.
        input_title (str): The title of the input dialog.
        overwrite_title (str): The title of the overwrite confirmation dialog.

    Returns:
        str: The valid name provided by the user, or None if the user cancels.
    """
    while True:
        # Prompt the user for a new name
        new_name, ok = QInputDialog.getText(parent, input_title, input_prompt, text=default_name)
        if not ok:
            # User canceled the input dialog
            return None

        if new_name == existing_name:
            # Warn the user about overwriting
            reply = QMessageBox.question(
                parent,
                overwrite_title,
                overwrite_warning.format(existing_name),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                # User confirmed overwriting
                return new_name
        else:
            # User provided a new name
            return new_name


class MultiEntryWindow(MultiEntryManager):
    def __init__(
        self,
        root,
        entries=None,
        num_entries=1,
        has_value=True,
        has_check=True,
        callbacks=None,
    ):
        super().__init__(None, entries, num_entries, has_value, has_check)
        self.root: Any = root
        self.window: Any = None
        self.callbacks = callbacks if isinstance(callbacks, dict) else {}

    def show(self):
        """Create and show the multi-entry window."""
        self.window = QDialog(self.root)
        self.window.setWindowTitle("Add Entries")
        self.window.finished.connect(self.save)
        self.window.setMinimumSize(200, 100)

        main_layout = QVBoxLayout(self.window)

        self.frame = QFrame(self.window)

        self.reset_frame(self.frame)

        # Create a frame for the buttons
        button_frame = QFrame(self.window)
        button_layout = QVBoxLayout(button_frame)

        # Create frames for grouping buttons
        base_frame = QFrame(button_frame)
        base_layout = QHBoxLayout(base_frame)
        button_layout.addWidget(base_frame)

        # Create buttons
        save_button = QPushButton("Save", base_frame)
        save_button.clicked.connect(self.save)
        base_layout.addWidget(save_button)

        clear_button = QPushButton("Clear", base_frame)
        clear_button.clicked.connect(self.clear)
        base_layout.addWidget(clear_button)

        if self.callbacks:
            callback_frame = QFrame(button_frame)
            callback_layout = QVBoxLayout(callback_frame)
            button_layout.addWidget(callback_frame)

            for key in self.callbacks.keys():
                if key.startswith("button_"):
                    name = key.replace("button_", "").replace("_", " ").title()
                    button = QPushButton(name, callback_frame)
                    button.clicked.connect(self.callbacks[key])
                    callback_layout.addWidget(button)

        main_layout.addWidget(self.frame)
        main_layout.addWidget(button_frame)

        self.window.setLayout(main_layout)
        self.window.exec_()

    def save(self):
        """Save the entries to the parameters and verify their validity."""

        if self.callbacks.get("save"):
            res = self.callbacks["save"](np.array(self.values))
            if res is not None:
                self.values = res

        self.close()

    def close(self):
        """Close the entry window."""
        self.destroy()  # destroy the entries
        # Destroy entry window if it exists
        if self.window:
            self.window.close()
            self.window = None
        if self.frame:
            self.frame.close()
            self.frame = None

    def clear(self):
        """Clear the entries from the parameters."""
        clear_callback = self.callbacks.get("clear", None)
        checked = self.callbacks.get("clear_flag", "unchecked")
        defaults = None
        if callable(clear_callback):
            defaults = clear_callback(self.values)

        if defaults is None:
            defaults = np.array([entry.default_values for entry in self.entries])
        if checked is True or checked == "checked":
            self.checked_values = defaults
        elif checked is False or checked == "unchecked":
            self.unchecked_values = defaults
        else:
            self.values = defaults


@dataclass
class StrItem:
    string: str
    index: int = 0
    sub_strings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.sub_strings, str):
            self.sub_strings = [self.sub_strings]
        elif not isinstance(self.sub_strings, (list, tuple)):
            self.sub_strings = []

        if not self.sub_strings:
            self.sub_strings = [self.string]

        if not isinstance(self.info, dict):
            self.info = {}


class TableFrameBase(QFrame):
    """
    A generic table widget class that provides core table functionality.

    This base class handles table creation, row management, and basic interaction.
    """

    def __init__(self, columns_config, parent=None):
        """
        Initialize the TableFrameBase.

        Args:
            columns_config (dict): Configuration for each column with format:
                {
                    "column_id": {
                        "header": str,              # Column header text
                        "type": str,                # Column type (e.g., "text", "check", "readonly")
                        "width": int,               # Optional column width
                        "editable": bool            # Whether column is editable
                    },
                    ...
                }
            parent: Parent widget
        """
        super().__init__(parent)

        # Set up context menu actions dictionary
        self._context_menu_actions = {}

        # Set up internal data tracking
        self.table_data = {}
        self.changed = False
        self.items_table: Any = None
        self.columns_config = columns_config

        # Initialize the UI
        self.initUI()

    @property
    def context_menu_actions(self):
        """Property to get the context menu actions."""
        return self._context_menu_actions

    @context_menu_actions.setter
    def context_menu_actions(self, actions):
        """
        Setter for context menu actions. Validates the input and updates the internal dictionary.
        """
        if isinstance(actions, dict):
            # Ensure all keys are strings and all values are callables
            if all(isinstance(k, str) and callable(v) for k, v in actions.items()):
                self._context_menu_actions.update(actions)
            else:
                raise ValueError("All keys must be strings and all values must be callables.")
        elif isinstance(actions, (list, tuple)) and len(actions) == 2:
            # Ensure the first element is a string and the second is a callable
            if isinstance(actions[0], str) and callable(actions[1]):
                self._context_menu_actions[actions[0]] = actions[1]
            else:
                raise ValueError(
                    "The first element must be a string and the second must be a callable."
                )
        else:
            raise ValueError(
                "Actions must be a dict or a list/tuple of length 2 with a string and a callable."
            )

    def initUI(self):
        """Initialize the UI components."""
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create table widget
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(len(self.columns_config))

        # Set headers and column properties
        headers = []
        column_index = 0
        for col_id, config in self.columns_config.items():
            headers.append(config.get("header", col_id))
            if "width" in config:
                self.items_table.setColumnWidth(column_index, config["width"])
            column_index += 1

        self.items_table.setHorizontalHeaderLabels(headers)
        self.items_table.installEventFilter(self)
        self.items_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.items_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.items_table.itemChanged.connect(self.on_item_changed)

        # Enable context menu
        self.items_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.items_table.customContextMenuRequested.connect(self.show_context_menu)

        self.layout.addWidget(self.items_table)

        # Add Save button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.save_button.setFixedWidth(75)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.save_button)
        self.layout.addLayout(self.buttons_layout)

    def eventFilter(self, source, event):
        """Handles key press events for deleting items and reordering rows."""
        if event.type() == QEvent.KeyPress:
            if source is self.items_table:
                # Handle Shift + Up
                if event.key() == Qt.Key_Up and event.modifiers() == Qt.ShiftModifier:
                    self.move_up()
                    return True

                # Handle Shift + Down
                elif event.key() == Qt.Key_Down and event.modifiers() == Qt.ShiftModifier:
                    self.move_down()
                    return True

                # Handle Delete key for row deletion
                if event.key() == Qt.Key_Delete:
                    rows_to_delete = set()
                    for item in self.items_table.selectedItems():
                        rows_to_delete.add(item.row())
                    for row in sorted(rows_to_delete, reverse=True):
                        self.items_table.removeRow(row)
                    return True

        return super().eventFilter(source, event)

    def move_up(self):
        """Move the selected item up in the QTableWidget and update table_data."""
        selected_items = self.items_table.selectedItems()
        if selected_items:
            selected_rows = sorted(set(item.row() for item in selected_items))
            if min(selected_rows) > 0:  # Ensure the first selected row is not the first row
                for row in selected_rows:
                    # Swap rows in the QTableWidget
                    for col in range(self.items_table.columnCount()):
                        current_item = self.items_table.takeItem(row, col)
                        above_item = self.items_table.takeItem(row - 1, col)
                        self.items_table.setItem(row - 1, col, current_item)
                        self.items_table.setItem(row, col, above_item)

                # Update the internal data structure
                self._update_data_order()

                # Reselect the moved items
                self.items_table.clearSelection()
                for row in selected_rows:
                    for col in range(self.items_table.columnCount()):
                        item = self.items_table.item(row - 1, col)
                        if item:
                            item.setSelected(True)

    def move_down(self):
        """Move the selected item down in the QTableWidget and update table_data."""
        selected_items = self.items_table.selectedItems()
        if selected_items:
            selected_rows = sorted(set(item.row() for item in selected_items), reverse=True)
            if (
                max(selected_rows) < self.items_table.rowCount() - 1
            ):  # Ensure the last selected row is not the last row
                for row in selected_rows:
                    # Swap rows in the QTableWidget
                    for col in range(self.items_table.columnCount()):
                        current_item = self.items_table.takeItem(row, col)
                        below_item = self.items_table.takeItem(row + 1, col)
                        self.items_table.setItem(row + 1, col, current_item)
                        self.items_table.setItem(row, col, below_item)

                # Update the internal data structure
                self._update_data_order()

                # Reselect the moved items
                self.items_table.clearSelection()
                for row in selected_rows:
                    for col in range(self.items_table.columnCount()):
                        item = self.items_table.item(row + 1, col)
                        if item:
                            item.setSelected(True)

    def _update_data_order(self):
        """
        Update the internal data structure to reflect the new table order.
        Should be implemented by subclasses to handle specific data formats.
        """
        pass

    def show_context_menu(self, position):
        """Display the context menu for the table."""
        menu = QMenu(self)

        # Add custom actions from the context menu actions dict
        for action_name, action_function in self.context_menu_actions.items():
            custom_action = QAction(action_name.title(), self)
            custom_action.triggered.connect(action_function)
            menu.addAction(custom_action)

        # Show the menu at the cursor position
        menu.exec_(self.items_table.viewport().mapToGlobal(position))

    def initialize(self, parent=None):
        """Initialize or reinitialize the table widget."""
        if sip.isdeleted(self):
            super().__init__(parent)
            self.initUI()
        else:
            self.setParent(parent)

    def on_item_changed(self, item):
        """Handle item change events."""
        self.changed = True

    def populate_table(self):
        """
        Populate the table with data.
        This method should be implemented by subclasses.
        """
        pass

    def clear(self):
        """Clear the table widget."""
        self.items_table.setRowCount(0)
        self.table_data = {}
        self.changed = False

    def save(self):
        """
        Save the current table state.
        Default implementation just destroys the widget.
        Subclasses should override for specific save behavior.
        """
        self.destroy()


class DataTable(TableFrameBase):
    """
    Specialized table widget for dataset management in EIS analysis.
    Extends the generic TableFrameBase with additional functionality for
    handling specific data types and operations.
    """

    def __init__(self, data, parent=None):
        # Define column configuration for this specific data table
        columns_config = {
            "plot": {"header": "Plot", "type": "check", "width": 20, "editable": True},
            "dataset": {"header": "Dataset", "type": "readonly", "width": 150, "editable": False},
            "shape": {"header": "Shape", "type": "text", "width": 75, "editable": True},
            "mark": {"header": "Mark", "type": "check", "width": 20, "editable": True},
            "label": {"header": "Label", "type": "text", "width": 150, "editable": True},
        }

        # Initialize base class
        super().__init__(columns_config, parent)

        # Set up context menu actions specific to DataTable
        self._context_menu_actions = {
            "highlight": self.highlight,
            "rename": self.rename_item,
        }

        # Initialize specific attributes
        self.table_info = {}
        self.special_items = []
        self.checked_items = []
        self.str_index = 0

        # Set data
        self.set_info(data, False, False)

        # Add combine button (specific to DataTable)
        self.combine_button = QPushButton("Combine")
        self.combine_button.clicked.connect(self.combine_items)
        self.combine_button.setFixedWidth(75)
        self.buttons_layout.insertWidget(0, self.combine_button)

        # Initialize the table if needed
        if self.items_table.rowCount() == 0:
            self.populate_table()

    def populate_table(self, data=None, append=False):
        """Update the entries in the table widget."""
        self.set_info(data, append, False)

        if sip.isdeleted(self):
            self.initialize()

        checked_str = (
            self.get_checked() if self.items_table.rowCount() != 0 else self.special_items
        )
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        for key, value in self.table_info.items():
            row_position = self.items_table.rowCount()
            self.items_table.insertRow(row_position)
            check_item = QTableWidgetItem()
            check_item.setFlags(check_item.flags() | Qt.ItemIsUserCheckable)
            check_item.setCheckState(Qt.Checked if key in checked_str else Qt.Unchecked)
            self.items_table.setItem(row_position, 0, check_item)
            dataset_item = QTableWidgetItem(value["Dataset"].string)
            dataset_item.setFlags(
                dataset_item.flags() & ~Qt.ItemIsEditable
            )  # Make dataset_item un-editable
            self.items_table.setItem(row_position, 1, dataset_item)
            shape_item = QTableWidgetItem(value["Shape"])
            self.items_table.setItem(row_position, 2, shape_item)
            mark_item = QTableWidgetItem()
            mark_item.setFlags(mark_item.flags() | Qt.ItemIsUserCheckable)
            mark_item.setCheckState(Qt.Checked if value["Mark"] else Qt.Unchecked)
            self.items_table.setItem(row_position, 3, mark_item)
            label_item = QTableWidgetItem(value["Label"])
            self.items_table.setItem(row_position, 4, label_item)
            self.set_highlight(row_position)
        self.items_table.blockSignals(False)

    def _update_data_order(self):
        """Update the internal table_info dictionary to reflect the new order."""
        new_order = []
        for row in range(self.items_table.rowCount()):
            dataset_name = self.items_table.item(row, 1).text()
            new_order.append(dataset_name)

        # Reorder the table_info dictionary
        reordered_table_info = {
            name: self.table_info[name] for name in new_order if name in self.table_info
        }
        self.table_info = reordered_table_info

    def get_info(self):
        """Return the strings of the entries in the table widget."""
        return [self.items_table.item(row, 1).text() for row in range(self.items_table.rowCount())]

    def set_info(self, data=None, append=False, run_update_entries=True):
        """
        Update the entries in the table widget.

        Args:
            data (list | tuple | set | None): The data to update the table with. Each item in `data` should be:
                - A `StrItem` object, or
                - A tuple of the form `(string, shape)`, where:
                    - `string` is the name of the dataset.
                    - `shape` is a string representing the size or shape of the dataset.
            append (bool): Whether to append the new data to the existing table entries.
            run_update_entries (bool): Whether to refresh the table widget after updating the internal data.

        Returns:
            None
        """
        if data is None:
            return
        data = [data] if not isinstance(data, (list, tuple, set)) else data
        data = [(ds, "") if not isinstance(ds, (list, tuple)) else ds for ds in data]
        checked_str = (
            self.get_checked()
            if self.items_table is not None and self.items_table.rowCount() != 0
            else self.special_items
        )
        new_table_info = {}
        for ds, shp in data:
            s = ds.string if isinstance(ds, StrItem) else str(ds)
            new_table_info[s] = {
                "Plot": True if s in checked_str else False,
                "Dataset": StrItem(s, self.str_index),
                "Shape": str(shp),
                "Mark": True if s in self.special_items else False,
                "Label": f"_{s}" if s in self.special_items else s,
            }
            self.str_index += 1

        if append:
            self.table_info.update(new_table_info)
        else:
            self.table_info = new_table_info

        if run_update_entries:
            self.populate_table()

    def get_checked(self, column=0):
        """Return the strings of the checked entries."""
        column = column if column in (0, 3) else 0
        res = []
        for row in range(self.items_table.rowCount()):
            if self.items_table.item(row, column).checkState() == Qt.Checked:
                res.append(self.items_table.item(row, 1).text())
        if not res:
            return self.special_items
        return res

    def set_checked(self, strings):
        """Set the checked entries in the table widget."""
        strings = (
            [str(string) for string in strings]
            if isinstance(strings, (tuple, list, set))
            else [str(strings)]
        )
        if self.items_table.rowCount() == 0:
            self.populate_table()
        for row in range(self.items_table.rowCount()):
            item = self.items_table.item(row, 1)
            if item.text() in strings:
                self.items_table.item(row, 0).setCheckState(Qt.Checked)
            else:
                self.items_table.item(row, 0).setCheckState(Qt.Unchecked)

    def set_highlight(self, row, strings=None):
        """Set the highlight for the selected row."""
        strings = strings if strings is not None else self.special_items
        font = self.items_table.item(row, 1).font()
        bold = True if self.items_table.item(row, 1).text() in strings else False
        font.setBold(bold)
        self.items_table.item(row, 1).setFont(font)
        if bold:
            self.items_table.item(row, 0).setCheckState(Qt.Checked)
            self.items_table.item(row, 3).setCheckState(Qt.Checked)

    def save(self):
        """Save the checked items and destroy the widget."""
        self.checked_items = self.get_checked()
        self.destroy()

    def highlight(self, strings=None, reset=False):
        """Highlight the selected entry in the table widget."""
        if strings is None or not strings:
            # If no strings are provided, use the selected items from the table
            selected_items = self.items_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Warning", "No rows selected for highlighting.")
                return
            strings = [self.items_table.item(item.row(), 1).text() for item in selected_items]

        if isinstance(strings, int):
            strings = [self.items_table.item(strings, 1).text()]

        old_highlight = self.special_items.copy()
        self.special_items = (
            [str(string) for string in strings]
            if isinstance(strings, (tuple, list, set))
            else [str(strings)]
        )

        if self.items_table.rowCount() == 0:
            self.populate_table()

        self.items_table.blockSignals(True)
        for row in range(self.items_table.rowCount()):
            self.set_highlight(row, None)
            if self.items_table.item(row, 1).text() in self.special_items:
                self.items_table.item(row, 0).setCheckState(Qt.Checked)
                self.items_table.item(row, 3).setCheckState(Qt.Checked)
            elif self.items_table.item(row, 1).text() in old_highlight:
                self.items_table.item(row, 0).setCheckState(Qt.Unchecked)
                self.items_table.item(row, 3).setCheckState(Qt.Unchecked)
        self.items_table.blockSignals(False)

    def rename_item(self) -> tuple[Any, Any]:
        """Rename the selected entry via the context menu."""
        selected_items = self.items_table.selectedItems()
        if len(selected_items) != 5:
            QMessageBox.warning(self, "Warning", "Select one item to rename.")
            return None, None

        # Get the selected row
        selected_row = selected_items[0].row()
        name = self.items_table.item(selected_row, 1).text()
        new_name, ok = QInputDialog.getText(self, "Rename item", "Enter the new name:", text=name)
        if ok:
            self.items_table.item(selected_row, 1).setText(new_name)
            self.items_table.item(selected_row, 4).setText(f"_{new_name}")

            if name in self.table_info[name]["Dataset"].sub_strings:
                substrings = self.table_info[name]["Dataset"].sub_strings
            else:
                substrings = self.table_info[name]["Dataset"].sub_strings + [name]
            self.table_info[new_name] = self.table_info.pop(name)
            self.table_info[new_name]["Dataset"] = StrItem(new_name, self.str_index, substrings)
            self.table_info[new_name]["Label"] = f"_{new_name}"
            self.str_index += 1
            return new_name, substrings
        return None, None

    def combine_items(self):
        """Combine the selected entries in the table widget."""
        selected_items = self.items_table.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Warning", "Select at least two items.")
            return None, None
        selected_rows = list(set(item.row() for item in selected_items))
        strings = [self.items_table.item(row, 1).text() for row in selected_rows]
        autoname, _ = find_common_str(*strings)
        new_name, ok = QInputDialog.getText(
            self, "Combine items", "Enter the new name:", text=autoname
        )
        if ok:
            new_table_info = {}
            substrings = []
            inserted = False
            for key, value in self.table_info.items():
                if key not in strings:
                    new_table_info[key] = value
                elif not inserted:
                    new_table_info[new_name] = {
                        "Plot": Qt.Unchecked,
                        "Dataset": StrItem(new_name, self.str_index, substrings),
                        "Shape": "",
                        "Mark": Qt.Unchecked,
                        "Label": f"_{new_name}",
                    }
                    substrings.extend(value["Dataset"].sub_strings)
                    inserted = True
                else:
                    substrings.extend(value["Dataset"].sub_strings)
            new_table_info[new_name]["Dataset"].sub_strings = substrings
            self.table_info = new_table_info
            self.str_index += 1
            self.populate_table()
            return new_name, strings
        return None, None


class TableFrame(QFrame):
    """
    A class to provide table interaction via a QTableWidget.
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        # Protected dictionary for context menu actions
        self._context_menu_actions = {"highlight": self.highlight, "rename": self.rename_item}

        self.table_info = {}
        self.special_items = []
        self.checked_items = []
        self.str_index = 0
        self.changed = False
        self.items_table: QTableWidget = QTableWidget()
        self.set_info(data, False, False)

        self.initUI()

    @property
    def context_menu_actions(self):
        """Property to get the context menu actions."""
        # Ensure default actions are present
        if "highlight" not in self._context_menu_actions:
            self._context_menu_actions["highlight"] = self.highlight
        if "rename" not in self._context_menu_actions:
            self._context_menu_actions["rename"] = self.rename_item
        return self._context_menu_actions

    @context_menu_actions.setter
    def context_menu_actions(self, actions):
        """
        Setter for context menu actions. Validates the input and updates the internal dictionary.
        """
        if isinstance(actions, dict):
            # Ensure all keys are strings and all values are callables
            if all(isinstance(k, str) and callable(v) for k, v in actions.items()):
                self._context_menu_actions.update(actions)
            else:
                raise ValueError("All keys must be strings and all values must be callables.")
        elif isinstance(actions, (list, tuple)) and len(actions) == 2:
            # Ensure the first element is a string and the second is a callable
            if isinstance(actions[0], str) and callable(actions[1]):
                self._context_menu_actions[actions[0]] = actions[1]
            else:
                raise ValueError(
                    "The first element must be a string and the second must be a callable."
                )
        else:
            raise ValueError(
                "Actions must be a dict or a list/tuple of length 2 with a string and a callable."
            )

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Table Widget for items
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(5)
        self.items_table.setHorizontalHeaderLabels(["Plot", "Dataset", "Shape", "Mark", "Label"])
        self.items_table.installEventFilter(self)
        self.items_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.items_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.items_table.itemChanged.connect(self.on_item_changed)
        self.items_table.setColumnWidth(0, 20)
        self.items_table.setColumnWidth(1, 150)
        self.items_table.setColumnWidth(2, 75)
        self.items_table.setColumnWidth(3, 20)
        self.items_table.setColumnWidth(4, 150)

        # Enable context menu
        self.items_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.items_table.customContextMenuRequested.connect(self.show_context_menu)

        self.layout.addWidget(self.items_table)

        self.combine_button = QPushButton("Combine")
        self.combine_button.clicked.connect(self.combine_items)
        self.combine_button.setFixedWidth(75)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.save_button.setFixedWidth(75)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.combine_button)
        self.buttons_layout.addWidget(self.save_button)

        self.layout.addLayout(self.buttons_layout)

        if self.items_table.rowCount() == 0:
            self.populate_table()

    def populate_table(self, data=None, append=False):
        """Update the entries in the table widget."""
        self.set_info(data, append, False)

        if sip.isdeleted(self):
            self.initialize()

        checked_str = (
            self.get_checked() if self.items_table.rowCount() != 0 else self.special_items
        )
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        for key, value in self.table_info.items():
            row_position = self.items_table.rowCount()
            self.items_table.insertRow(row_position)

            check_item = QTableWidgetItem()
            check_item.setFlags(check_item.flags() | Qt.ItemIsUserCheckable)
            check_item.setCheckState(Qt.Checked if key in checked_str else Qt.Unchecked)
            self.items_table.setItem(row_position, 0, check_item)

            dataset_item = QTableWidgetItem(value["Dataset"].string)
            dataset_item.setFlags(dataset_item.flags() & ~Qt.ItemIsEditable)
            self.items_table.setItem(row_position, 1, dataset_item)

            shape_item = QTableWidgetItem(value["Shape"])
            self.items_table.setItem(row_position, 2, shape_item)

            mark_item = QTableWidgetItem()
            mark_item.setFlags(mark_item.flags() | Qt.ItemIsUserCheckable)
            mark_item.setCheckState(Qt.Checked if value["Mark"] else Qt.Unchecked)
            self.items_table.setItem(row_position, 3, mark_item)

            label_item = QTableWidgetItem(value["Label"])
            self.items_table.setItem(row_position, 4, label_item)
            self.set_highlight(row_position)
        self.items_table.blockSignals(False)

    def eventFilter(self, source, event):
        """Handles key press events for deleting items and reordering rows."""
        if event.type() == QEvent.KeyPress:
            if source is self.items_table:
                # Handle Shift + Up
                if event.key() == Qt.Key_Up and event.modifiers() == Qt.ShiftModifier:
                    # if selected_rows[0] > 0:  # Ensure the first selected row is not the first row
                    self.move_up()
                    return True

                # Handle Shift + Down
                elif event.key() == Qt.Key_Down and event.modifiers() == Qt.ShiftModifier:
                    # if selected_rows[-1] < self.items_table.rowCount() - 1:  # Ensure the last selected row is not the last row
                    self.move_down()
                    return True

                # Handle Delete key for row deletion
                if event.key() == Qt.Key_Delete:
                    rows_to_delete = set()
                    for item in self.items_table.selectedItems():
                        rows_to_delete.add(item.row())
                    for row in sorted(rows_to_delete, reverse=True):
                        self.items_table.removeRow(row)
                    return True

        return super().eventFilter(source, event)

    def move_up(self):
        """Move the selected item up in the QTableWidget and update table_info."""
        selected_items = self.items_table.selectedItems()
        if selected_items:
            selected_rows = sorted(set(item.row() for item in selected_items))
            if min(selected_rows) > 0:  # Ensure the first selected row is not the first row
                for row in selected_rows:
                    # Swap rows in the QTableWidget
                    for col in range(self.items_table.columnCount()):
                        current_item = self.items_table.takeItem(row, col)
                        above_item = self.items_table.takeItem(row - 1, col)
                        self.items_table.setItem(row - 1, col, current_item)
                        self.items_table.setItem(row, col, above_item)

                # Update the internal table_info dictionary
                self._update_table_info_order()

                # Reselect the moved items
                self.items_table.clearSelection()
                for row in selected_rows:
                    for col in range(self.items_table.columnCount()):
                        item = self.items_table.item(row - 1, col)
                        if item:
                            item.setSelected(True)

    def move_down(self):
        """Move the selected item down in the QTableWidget and update table_info."""
        selected_items = self.items_table.selectedItems()
        if selected_items:
            selected_rows = sorted(set(item.row() for item in selected_items), reverse=True)
            if (
                max(selected_rows) < self.items_table.rowCount() - 1
            ):  # Ensure the last selected row is not the last row
                for row in selected_rows:
                    # Swap rows in the QTableWidget
                    for col in range(self.items_table.columnCount()):
                        current_item = self.items_table.takeItem(row, col)
                        below_item = self.items_table.takeItem(row + 1, col)
                        self.items_table.setItem(row + 1, col, current_item)
                        self.items_table.setItem(row, col, below_item)

                # Update the internal table_info dictionary
                self._update_table_info_order()

                # Reselect the moved items
                self.items_table.clearSelection()
                for row in selected_rows:
                    for col in range(self.items_table.columnCount()):
                        item = self.items_table.item(row + 1, col)
                        if item:
                            item.setSelected(True)

    def _update_table_info_order(self):
        """Update the internal table_info dictionary to reflect the new order."""
        new_order = []
        for row in range(self.items_table.rowCount()):
            dataset_name = self.items_table.item(row, 1).text()
            new_order.append(dataset_name)

        # Reorder the table_info dictionary
        reordered_table_info = {
            name: self.table_info[name] for name in new_order if name in self.table_info
        }
        self.table_info = reordered_table_info

    def show_context_menu(self, position):
        """Display the context menu for the table."""
        menu = QMenu(self)

        # Add custom actions from the class variable
        for action_name, action_function in self.context_menu_actions.items():
            custom_action = QAction(action_name.title(), self)
            custom_action.triggered.connect(action_function)
            menu.addAction(custom_action)

        # Show the menu at the cursor position
        menu.exec_(self.items_table.viewport().mapToGlobal(position))

    def initialize(self, parent=None):
        """Initialize the table widget."""
        if sip.isdeleted(self):
            super().__init__(parent)
            self.initUI()
        else:
            self.setParent(parent)

    def on_item_changed(self, item):
        self.changed = True

    def get_info(self):
        """Return the strings of the entries in the table widget."""
        return [self.items_table.item(row, 1).text() for row in range(self.items_table.rowCount())]

    def set_info(self, data=None, append=False, run_update_entries=True):
        """
        Update the entries in the table widget.

        Args:
            data (list | tuple | set | None): The data to update the table with. Each item in `data` should be:
                - A `StrItem` object, or
                - A tuple of the form `(string, shape)`, where:
                    - `string` is the name of the dataset.
                    - `shape` is a string representing the size or shape of the dataset.
            append (bool): Whether to append the new data to the existing table entries.
            run_update_entries (bool): Whether to refresh the table widget after updating the internal data.

        Returns:
            None
        """
        if data is None:
            return
        data = [data] if not isinstance(data, (list, tuple, set)) else data
        data = [(ds, "") if not isinstance(ds, (list, tuple)) else ds for ds in data]
        checked_str = (
            self.get_checked()
            if self.items_table is not None and self.items_table.rowCount() != 0
            else self.special_items
        )
        new_table_info = {}
        for ds, shp in data:
            s = ds.string if isinstance(ds, StrItem) else str(ds)
            new_table_info[s] = {
                "Plot": True if s in checked_str else False,
                "Dataset": StrItem(s, self.str_index),
                "Shape": str(shp),
                "Mark": True if s in self.special_items else False,
                "Label": f"_{s}" if s in self.special_items else s,
            }
            self.str_index += 1

        if append:
            self.table_info.update(new_table_info)
        else:
            self.table_info = new_table_info

        if run_update_entries:
            self.populate_table()

    def get_checked(self, column=0):
        """Return the strings of the checked entries."""
        column = column if column in (0, 3) else 0
        res = []
        for row in range(self.items_table.rowCount()):
            if self.items_table.item(row, column).checkState() == Qt.Checked:
                res.append(self.items_table.item(row, 1).text())
        if not res:
            return self.special_items
        return res

    def set_checked(self, strings):
        """Set the checked entries in the table widget."""
        strings = (
            [str(string) for string in strings]
            if isinstance(strings, (tuple, list, set))
            else [str(strings)]
        )
        if self.items_table.rowCount() == 0:
            self.populate_table()
        for row in range(self.items_table.rowCount()):
            item = self.items_table.item(row, 1)
            if item.text() in strings:
                self.items_table.item(row, 0).setCheckState(Qt.Checked)
            else:
                self.items_table.item(row, 0).setCheckState(Qt.Unchecked)

    def set_highlight(self, row, strings=None):
        """Set the highlight for the selected row."""
        strings = strings if strings is not None else self.special_items
        font = self.items_table.item(row, 1).font()
        bold = True if self.items_table.item(row, 1).text() in strings else False
        font.setBold(bold)
        self.items_table.item(row, 1).setFont(font)
        if bold:
            self.items_table.item(row, 0).setCheckState(Qt.Checked)
            self.items_table.item(row, 3).setCheckState(Qt.Checked)

    def save(self):
        self.checked_items = self.get_checked()
        self.destroy()

    def clear(self):
        """Clear the table widget."""
        self.items_table.setRowCount(0)
        self.checked_items = []

    def highlight(self, strings=None, reset=False):
        """Highlight the selected entry in the table widget."""
        # insert logic from context_highlight for if strings is not a list of strings or None
        if strings is None or not strings:
            # If no strings are provided, use the selected items from the table
            selected_items = self.items_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Warning", "No rows selected for highlighting.")
                return
            strings = [self.items_table.item(item.row(), 1).text() for item in selected_items]

        if isinstance(strings, int):
            strings = [self.items_table.item(strings, 1).text()]

        old_highlight = self.special_items.copy()
        self.special_items = (
            [str(string) for string in strings]
            if isinstance(strings, (tuple, list, set))
            else [str(strings)]
        )

        if self.items_table.rowCount() == 0:
            self.populate_table()

        self.items_table.blockSignals(True)
        for row in range(self.items_table.rowCount()):
            self.set_highlight(row, None)
            if self.items_table.item(row, 1).text() in self.special_items:
                self.items_table.item(row, 0).setCheckState(Qt.Checked)
                self.items_table.item(row, 3).setCheckState(Qt.Checked)
            elif self.items_table.item(row, 1).text() in old_highlight:
                self.items_table.item(row, 0).setCheckState(Qt.Unchecked)
                self.items_table.item(row, 3).setCheckState(Qt.Unchecked)
        self.items_table.blockSignals(False)

    def rename_item(self) -> tuple[Any, Any]:
        """Rename the selected entry via the context menu."""
        selected_items = self.items_table.selectedItems()
        if len(selected_items) != 5:
            QMessageBox.warning(self, "Warning", "Select one item to rename.")
            return None, None

        # Get the selected row
        selected_row = selected_items[0].row()
        name = self.items_table.item(selected_row, 1).text()
        new_name, ok = QInputDialog.getText(self, "Rename item", "Enter the new name:", text=name)
        if ok:
            self.items_table.item(selected_row, 1).setText(new_name)
            self.items_table.item(selected_row, 4).setText(f"_{new_name}")

            if name in self.table_info[name]["Dataset"].sub_strings:
                substrings = self.table_info[name]["Dataset"].sub_strings
            else:
                substrings = self.table_info[name]["Dataset"].sub_strings + [name]
            self.table_info[new_name] = self.table_info.pop(name)
            self.table_info[new_name]["Dataset"] = StrItem(new_name, self.str_index, substrings)
            self.table_info[new_name]["Label"] = f"_{new_name}"
            self.str_index += 1
            return new_name, substrings
        return None, None

    def combine_items(self):
        """Combine the selected entries in the table widget."""
        selected_items = self.items_table.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Warning", "Select at least two items.")
            return None, None
        selected_rows = list(set(item.row() for item in selected_items))
        strings = [self.items_table.item(row, 1).text() for row in selected_rows]
        autoname, _ = find_common_str(*strings)
        new_name, ok = QInputDialog.getText(
            self, "Combine items", "Enter the new name:", text=autoname
        )
        if ok:
            new_table_info = {}
            substrings = []
            inserted = False
            for key, value in self.table_info.items():
                if key not in strings:
                    new_table_info[key] = value
                elif not inserted:
                    new_table_info[new_name] = {
                        "Plot": Qt.Unchecked,
                        "Dataset": StrItem(new_name, self.str_index, substrings),
                        "Shape": "",
                        "Mark": Qt.Unchecked,
                        "Label": f"_{new_name}",
                    }
                    substrings.extend(value["Dataset"].sub_strings)
                    inserted = True
                else:
                    substrings.extend(value["Dataset"].sub_strings)
            new_table_info[new_name]["Dataset"].sub_strings = substrings
            self.table_info = new_table_info
            self.str_index += 1
            self.populate_table()
            return new_name, strings
        return None, None


class DictKeysModel(QAbstractListModel):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self._dm = data_manager

    def rowCount(self, parent=None):
        # +1 for the sentinel
        return len(self._dm.raw) + 1

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if index.row() == 0:
                return "None"  # or "— None —" if you want a label
            return list(self._dm.raw.keys())[index.row() - 1]

    def keyAt(self, row):
        if row == 0:
            return None
        return list(self._dm.raw.keys())[row - 1]

    def refresh(self):
        self.layoutChanged.emit()


class DataController(DataManager):
    def __init__(self, *args, combo=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = DictKeysModel(self)
        self.var = combo
        if combo:
            combo.setModel(self._model)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._model.refresh()

    def clear(self):
        super().clear()
        self._model.refresh()

    def primary(self) -> str:
        if self.var is not None:
            return self.var.currentText()
        else:
            return ""

    def primary_sys(self) -> ComplexSystem | None:
        if self.primary() in self.raw:
            return self.raw[self.primary()]
        return None

    def primary_df(self) -> pd.DataFrame | None:
        if (system := self.primary_sys()) is not None:
            return system.get_df("freq", "real", "imag")
        return None


# class DataHandlerWidgets(DataHandler):
class DataHandlerWidgets(DataManager):
    """A class to handle the data widgets for the EIS analysis."""

    def __init__(self, parent=None, callback=None, **kwargs):
        # self.raw = {}
        super().__init__(**kwargs)
        self._raw: dict[str, pd.DataFrame] = {}
        self.root: Any = parent
        self.raw_list = TableFrame([(k, str(len(v))) for k, v in self.raw.items()], parent)
        self.raw_list.context_menu_actions = {"rename": self.rename_item}
        self.raw_list.context_menu_actions = {"simplify": self.simplify_item}
        self.raw_list.context_menu_actions = {"interpolate": self.interpolate_item}
        self.raw_list.context_menu_actions = {"smooth": self.smooth_item}
        self.raw_list.context_menu_actions = {"view_attrs": self.view_attrs}

        self.window: Any = None
        self.var: Any = None

        self.callback = callback

    def load_window(self):
        """Show the list frame window."""
        if not self.raw:
            QMessageBox.warning(self.root, "Warning", "No data to display.")
            return
        for key, system in self.raw.items():
            self._raw[key] = system.get_df("freq", "real", "imag")

        self.raw_archive = {**self.raw_archive.copy(), **self.raw.copy()}

        # self.raw_list.populate_list(list(self.raw.keys()))
        self.raw_list.populate_table([(k, str(len(v))) for k, v in self.raw.items()])

        self.window = QMainWindow(self.root)
        self.window.setWindowTitle("Loaded Datasets")
        # self.window.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        central_widget = QWidget()
        # self.raw_list.setParent(central_widget)
        self.raw_list.initialize(central_widget)

        # layout = QVBoxLayout(central_widget)
        self.layout.addWidget(self.raw_list)
        central_widget.setLayout(self.layout)

        self.raw_list.combine_button.clicked.disconnect()
        self.raw_list.combine_button.clicked.connect(self.combine_items)

        self.raw_list.save_button.clicked.disconnect()
        self.raw_list.save_button.clicked.connect(self.save_data)

        self.window.setCentralWidget(central_widget)

        self.window.closeEvent = self.closeEvent
        # set height by row
        self.window.resize(600, min(800, 100 + 30 * self.raw_list.items_table.rowCount()))

        self.window.show()

    def show(self):
        """Show the list frame window."""
        self.load_window()

    def closeEvent(self, event):
        self.raw_list.setParent(None)
        # self.window.hide()

        if self.raw_list.changed:
            reply = QMessageBox.question(
                self.raw_list,
                "Save Changes",
                "Do you want to save changes before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self.save_data()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()
            if not sip.isdeleted(self.window):
                self.window.deleteLater()
        else:
            event.accept()
            if not sip.isdeleted(self.window):
                self.window.deleteLater()

    def destroy(self):
        """Destroy the list frame window."""
        self.raw_list.changed = False
        if self.window is not None and not sip.isdeleted(self.window):
            self.window.close()
            self.window = None
        self.raw_list.setParent(None)

    def save_data(self):
        """Save the data to the raw dictionary."""
        self.raw_archive = {**self.raw_archive.copy(), **self.raw.copy()}
        self.raw = {}
        tmp_raw = {}
        for key in self.raw_list.get_info():
            self.update_system(
                key,
                self._raw[key],
                form="impedance",
            )
            tmp_raw[key] = self.raw[key].get_df("freq", "real", "imag")
        self._raw = tmp_raw
        if self.raw_list.changed:
            self.update_var()
        elif self.var is not None and not self.is_highlighted(self.primary()):
            self.var.setCurrentText(self.raw_list.special_items[0])
        self.raw_list.changed = False
        if self.callback is not None:
            self.callback()

    def update_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Update the raw data dictionary."""
        self._raw = data
        self.save_data()

    def update_var(self):
        """Update the variable in the list frame window."""
        if self.var is not None and self.raw:
            old_var = (
                self.raw_list.special_items[0]
                if self.raw_list.special_items
                else self.var.currentText()
            )
            self.var.blockSignals(True)
            self.var.clear()
            self.var.setCurrentText("")
            self.var.addItems(["None"] + list(self.raw.keys()))
            if old_var not in self.raw:
                old_var = next(iter(self.raw))
            self.highlight(old_var)
            self.var.setCurrentText(old_var)
            self.var.blockSignals(False)

    def set_var(self, var):
        """Set the variable in the list frame window."""
        if self.var is not None and var in self.raw:
            self.var.blockSignals(True)
            self.highlight(var)
            self.var.setCurrentText(var)
            self.var.blockSignals(False)

    def primary(self) -> str:
        if self.var is not None:
            return self.var.currentText()
        else:
            return ""

    def primary_sys(self) -> ComplexSystem | None:
        if self.primary() in self.raw:
            return self.raw[self.primary()]
        return None

    def primary_df(self) -> pd.DataFrame | None:
        if (system := self.primary_sys()) is not None:
            return system.get_df("freq", "real", "imag")
        return None

    def combine_items(self):
        """Combine the selected entries in the list widget."""
        new_key, keys = self.raw_list.combine_items()
        if new_key:
            data = {k: v for k, v in self._raw.items() if k in keys}  # type: ignore
            comb = pd.concat(data.values(), sort=False, keys=data.keys())
            comb.sort_values("freq", ignore_index=True)
            comb.attrs["thickness"] = np.mean([v.attrs["thickness"] for v in data.values()])
            comb.attrs["area"] = np.mean([v.attrs["area"] for v in data.values()])
            self._raw[new_key] = comb
            self.raw_list.changed = True
            self.raw_list.table_info[new_key]["Shape"] = str(len(comb))
            self.raw_list.populate_table()
            # self.update_var()

    def rename_item(self):
        """Rename the selected entry in the list widget."""
        new_key, keys = self.raw_list.rename_item()
        if new_key:
            for key in keys:
                if key in self._raw:
                    self._raw[new_key] = self._raw.pop(key)
                    self.raw_list.changed = True
                    # self.update_var()
                    break
            else:
                QMessageBox.warning(self.raw_list, "Warning", "Key not found.")

    @overload
    def _prepare_dataset_modificaton(
        self, as_df: IsTrue = True
    ) -> tuple[str, pd.DataFrame] | tuple[str, None]: ...
    @overload
    def _prepare_dataset_modificaton(
        self, as_df: IsFalse
    ) -> tuple[str, ComplexSystem] | tuple[str, None]: ...

    def _prepare_dataset_modificaton(
        self, as_df=True
    ) -> tuple[str, pd.DataFrame | ComplexSystem] | tuple[str, None]:
        """
        Helper method to validate the selected dataset.

        Returns:
            tuple: (dataset_name, dataset) if valid, otherwise None.
        """
        selected_items = self.raw_list.items_table.selectedItems()
        if len(selected_items) != 5:
            QMessageBox.warning(self.raw_list, "Warning", "Select one dataset for this action.")
            return "", None

        # Get the selected row and dataset name
        selected_row = selected_items[0].row()
        dataset_name = self.raw_list.items_table.item(selected_row, 1).text()

        # Check if the dataset exists in the raw data
        if dataset_name not in self._raw:
            QMessageBox.warning(self.raw_list, "Warning", f"Dataset '{dataset_name}' not found.")
            return "", None

        # Get the dataset
        if as_df:
            dataset = self._raw[dataset_name]

            # Check if "freq" column exists
            if "freq" not in dataset.columns:
                QMessageBox.warning(
                    self.raw_list,
                    "Warning",
                    f"Dataset '{dataset_name}' does not contain a 'freq' column.",
                )
                return "", None
        else:
            if dataset_name not in self.raw:
                in_data = self._raw[dataset_name]
                dataset = ComplexSystem(
                    in_data,  # df
                    thickness=in_data.attrs.get("thickness", 1.0),
                    area=in_data.attrs.get("area", 1.0),
                )
            else:
                dataset = self.raw[dataset_name].copy()

        return dataset_name, dataset

    def simplify_item(self):
        """Simplify the selected dataset by combining duplicate or nearly equivalent frequencies."""
        # Validate the dataset
        dataset_name, dataset = self._prepare_dataset_modificaton()
        if dataset is None:
            return

        # Get the initial tolerance from the user
        tolerance, ok = QInputDialog.getDouble(
            self.raw_list,
            "Set Tolerance",
            "Enter the tolerance for grouping frequencies:",
            value=1e-3,  # Default value
            min=1e-10,  # Minimum value
            max=1e2,  # Maximum value
            decimals=10,
        )
        if not ok:
            return

        # Sort the dataset by "freq"
        dataset = dataset.sort_values("freq").reset_index(drop=True)

        # Simplify the dataset in a loop
        while True:
            # Group frequencies based on the tolerance
            grouped_data = []
            current_group = [dataset.iloc[0]]
            for i in range(1, len(dataset)):
                current_freq = dataset.iloc[i]["freq"]
                previous_freq = current_group[-1]["freq"]

                # Check if the current frequency is within the tolerance of the previous frequency
                if (
                    abs(current_freq - previous_freq)
                    <= tolerance * (current_freq + previous_freq) / 2
                ):
                    current_group.append(dataset.iloc[i])
                else:
                    # Combine the current group and start a new group
                    grouped_data.append(pd.DataFrame(current_group).mean())
                    current_group = [dataset.iloc[i]]

            # Add the last group
            if current_group:
                grouped_data.append(pd.DataFrame(current_group).mean())

            # Combine the grouped data into a new DataFrame
            simplified_data = pd.DataFrame(grouped_data)
            simplified_data.sort_values("freq", ignore_index=True)
            simplified_data.attrs["thickness"] = dataset.attrs.get("thickness", 1)
            simplified_data.attrs["area"] = dataset.attrs.get("area", 1)

            # Check if the dataset length has changed significantly
            length_change = abs(len(dataset) - len(simplified_data)) / len(dataset)
            if length_change >= 0.05:  # At least 5% change
                break

            # Notify the user and ask for a new tolerance or to cancel
            reply = QMessageBox.question(
                self.raw_list,
                "Adjust Tolerance",
                f"The dataset length changed by only {length_change * 100:.2f}%. Enter a new tolerance or cancel.",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Cancel:
                return

            # Get a new tolerance from the user
            tolerance, ok = QInputDialog.getDouble(
                self.raw_list,
                "Set Tolerance",
                "Enter the tolerance for grouping frequencies:",
                value=tolerance,  # Default to the previous tolerance
                min=1e-10,
                max=1e2,
                decimals=10,
            )
            if not ok:
                return

        # Use the helper function to get a valid name
        new_name = get_valid_name(self.raw_list, dataset_name, dataset_name)
        if not new_name:
            # User canceled the operation
            return

        self._raw[new_name] = simplified_data

        # Prepare data for set_info
        data_to_update = [(new_name, str(len(simplified_data)))]
        if new_name != dataset_name:
            data_to_update.append((dataset_name, str(len(self._raw[dataset_name]))))

        # Update the table using set_info
        self.raw_list.set_info(data_to_update, append=True)

        # Mark the table as changed
        self.raw_list.changed = True

    def interpolate_item(self):
        """Interpolate the selected dataset based on user-defined parameters."""
        # Validate the dataset
        dataset_name, dataset = self._prepare_dataset_modificaton(False)
        if dataset is None:
            return

        try:
            # Define default inputs for the FormDialog
            interpolation_inputs = {
                "Start": "",
                "Stop": "",
                "Length": "",
                "smooth_first": True,
                "bc_type": "not-a-knot",
                "extrapolate": False,
            }

            # Create the FormDialog
            inputs, ok = FormDialog.getResult(
                self.raw_list,
                title="Interpolate Dataset",
                content=interpolation_inputs,
            )
            if not ok:
                return

            # # Retrieve user inputs
            # inputs = input_window.as_dict()
            start = inputs.pop("Start")
            stop = inputs.pop("Stop")
            length = inputs.pop("Length")
            # smooth_first = inputs["Smooth First (True/False)"]
            # bc_type = inputs["Boundary Condition (bc_type)"]
            # extrapolate = inputs["Extrapolate (True/False)"]

            # Handle default values for start, stop, and length
            if not start and not stop and not length:
                return  # User provided no input, cancel the operation

            # Fill in missing values from the current frequency
            # dataset = cast(ComplexSystem, dataset)
            # current_frequency:np.ndarray = dataset["freq"]
            current_frequency = dataset.frequency

            start = float(start) if start else current_frequency.min()
            stop = float(stop) if stop else current_frequency.max()
            length = int(length) if length else len(current_frequency)

            # Example: Generate the new x-axis (new_x)
            if start <= 0 or stop <= 0:
                new_x = np.logspace(start, stop, length)
            else:
                new_x = np.logspace(np.log10(start), np.log10(stop), length)

            # Initialize the interpolated_data DataFrame with the new frequency as the first column
            interpolated_data = pd.DataFrame({"freq": new_x})

            # Perform interpolation for the real and imaginary parts of impedance
            interpolated_data["real"] = dataset.copy().interpolated_form(
                dataset["impedance.real"], new_x=new_x, old_x=current_frequency, **inputs
            )
            interpolated_data["imag"] = dataset.copy().interpolated_form(
                dataset["impedance.imag"], new_x=new_x, old_x=current_frequency, **inputs
            )

            # Copy attributes from the original dataset
            interpolated_data.attrs["thickness"] = dataset["thickness"]
            interpolated_data.attrs["area"] = dataset["area"]

            # Use the helper function to get a valid name
            new_name = get_valid_name(self.raw_list, dataset_name, dataset_name)
            if not new_name:
                # User canceled the operation
                return

            # Save the interpolated dataset
            self._raw[new_name] = interpolated_data

            # Prepare data for set_info
            data_to_update = [(new_name, str(len(interpolated_data)))]
            if new_name != dataset_name:
                data_to_update.append((dataset_name, str(len(self._raw[dataset_name]))))

            # Update the table using set_info
            self.raw_list.set_info(data_to_update, append=True)

            # Mark the table as changed
            self.raw_list.changed = True
        except Exception as e:
            QMessageBox.critical(
                self.raw_list,
                "Error",
                f"An error occurred during interpolation:\n{str(e)}",
            )

    def smooth_item(self):
        """Interpolate the selected dataset based on user-defined parameters."""
        # Validate the dataset
        dataset_name, dataset = self._prepare_dataset_modificaton(False)
        if dataset is None:
            return
        try:
            # Define default inputs for the FormDialog
            smooth_inputs = dataset.savgol_kwargs.copy()

            # Create the FormDialog
            inputs, ok = FormDialog.getResult(
                self.raw_list,
                title="Smooth Dataset",
                content=smooth_inputs,
            )
            if not ok:
                return

            inputs = {k: safe_eval(v) for k, v in inputs.items()}

            # Initialize the interpolated_data DataFrame with the new frequency as the first column
            smooth_data = pd.DataFrame({"freq": dataset.frequency})

            # Perform interpolation for the real and imaginary parts of impedance
            smooth_data["real"] = dataset.copy().smoothed_form(dataset["impedance.real"], **inputs)
            smooth_data["imag"] = dataset.copy().smoothed_form(dataset["impedance.imag"], **inputs)

            # Copy attributes from the original dataset
            smooth_data.attrs["thickness"] = dataset["thickness"]
            smooth_data.attrs["area"] = dataset["area"]
            smooth_data.attrs |= dataset.attrs

            # Use the helper function to get a valid name
            new_name = get_valid_name(self.raw_list, dataset_name, dataset_name)
            if not new_name:
                # User canceled the operation
                return

            # Save the interpolated dataset
            self._raw[new_name] = smooth_data

            # Prepare data for set_info
            data_to_update = [(new_name, str(len(smooth_data)))]
            if new_name != dataset_name:
                data_to_update.append((dataset_name, str(len(self._raw[dataset_name]))))

            # Update the table using set_info
            self.raw_list.set_info(data_to_update, append=True)

            # Mark the table as changed
            self.raw_list.changed = True
        except Exception as e:
            QMessageBox.critical(
                self.raw_list,
                "Error",
                f"An error occurred during smoothing:\n{str(e)}",
            )

    def view_attrs(self):
        """View the selected dataset in a separate window."""
        # Get the selected dataset using _prepare_dataset_modificaton
        dataset_name, dataset = self._prepare_dataset_modificaton()
        if dataset is None:
            return

        # Prepare the attributes dictionary for editing
        if hasattr(dataset, "attrs"):
            attrs = dict(dataset.attrs)
        else:
            attrs = {}

        attr_window = DataViewer(
            attrs,
            self.raw_list,
            name=f"{dataset_name}_attributes",
        )

        loop = QEventLoop()
        attr_window.destroyed.connect(loop.quit)
        loop.exec_()

        # # If the user canceled, do not update
        # if attr_window.canceled:
        #     return

        # Update the dataset's attributes with the edited values
        # new_attrs = attr_window.as_dict()
        new_attrs = dict(attr_window.data)
        if hasattr(dataset, "attrs"):
            dataset.attrs.clear()
            dataset.attrs.update(new_attrs)

        # Optionally, update the _raw dict as well
        self._raw[dataset_name] = dataset

    def highlight(self, strings, reset=False):
        """Highlight the selected entry in the list widget."""
        self.raw_list.highlight(strings, reset)

    def is_highlighted(self, string):
        """Check if the string is highlighted."""
        return string in self.raw_list.special_items

    def get_checked(self):
        """Return the strings of the checked entries."""
        return self.raw_list.get_checked()

    def get_label(self, key, legend=False):
        """Return the label for the selected key."""
        try:
            valid_key = self.raw_list.table_info[key]["Label"]
        except KeyError:
            valid_key = f"_{key}"
        if legend:
            return valid_key.strip("_ ")
        return valid_key

    def get_mark(self, key):
        """Return the mark for the selected key."""
        if self.primary() and self.primary() == key:
            return True
        checks = self.raw_list.get_checked(3)
        if checks != self.raw_list.special_items:
            return key in checks
        try:
            return self.raw_list.table_info[key]["Mark"]
        except KeyError:
            return True

    def run_order(
        self,
        data: pd.Series | pd.DataFrame,
        dataset_col: str = "Dataset",
        key: Callable | None = lambda obj: np.max(obj.Z.real),
        ascending: bool = False,
    ) -> list[int]:
        """
        Generate a run order (list of row indices) based on a scoring function
        applied to each dataset in self.raw.

        Parameters
        ----------
        datasets : pd.Series or pd.DataFrame
            Either the 'Dataset' column or the full DataFrame.
        dataset_col : str, default="Dataset"
            Column name if a DataFrame is passed.
        key : Callable, optional
            Function mapping a dataset object -> numeric score.
            Defaults to max(obj.Z.real).
        ascending : bool, default=False
            Sort order (False = highest score first).

        Returns
        -------
        list[int]
            Row indices in the desired order.
        """
        # Compute max real impedance for each dataset
        # Normalize to a Series of dataset names
        if isinstance(data, pd.DataFrame):
            series = data[dataset_col]
        else:
            series = data

        if key is None:
            return series.index.tolist()

        scores = {name: key(self.raw[name]) for name in series.unique() if name in self.raw}

        return series.map(scores).sort_values(ascending=ascending).index.tolist()

        # max_real = {
        #     key: np.max(self.raw[key].Z.real)
        #     for key in df[dataset_col].unique()
        #     if key in self.raw
        # }

        # # Map each row to its dataset's max real impedance
        # scores = df[dataset_col].map(max_real)

        # # Sort indices by descending impedance
        # return scores.sort_values(ascending=False).index.tolist()
