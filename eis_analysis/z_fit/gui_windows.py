# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import sys

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QTabWidget,
    QTableWidget,
    QLineEdit,
    QPushButton,
    QFrame,
    QMessageBox,
    QFormLayout,
    QDialogButtonBox,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QMainWindow,
    QTableWidgetItem,
    QTextEdit,
    QProgressDialog,
)
from PyQt5.QtGui import QBrush, QColor, QFontMetrics

from ..string_ops import format_number

from .gui_widgets import MultiEntryManager


def create_progress_dialog(
    parent,
    title="Progress",
    label_text="Processing...",
    cancel=None,
    minimum=0,
    maximum=100,
):
    """Create and return a QProgressDialog."""
    progress_dialog = QProgressDialog(
        label_text, cancel, minimum, maximum, parent
    )
    progress_dialog.setWindowTitle(title)
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setMinimumDuration(0)
    return progress_dialog


class CustomSimpleDialog(QDialog):
    """Class to create a simple dialog window."""

    def __init__(
        self, parent, title=None, prompt=None, initialvalue="", width=50
    ):
        super().__init__(parent)
        self.prompt = prompt
        self.initialvalue = initialvalue
        self.width = width
        self.setWindowTitle(title)
        self.init_ui()

    def init_ui(self):
        """Create the body of the dialog window."""
        layout = QVBoxLayout()

        label = QLabel(self.prompt)
        layout.addWidget(label)

        self.entry = QLineEdit(self)
        self.entry.setText(self.initialvalue)
        self.entry.setFixedWidth(self.width)
        layout.addWidget(self.entry)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def accept(self):
        """Apply the changes made in the dialog window."""
        self.result = self.entry.text()
        super().accept()


class PrintWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = None
        self.text_area = None
        self.old_stdout = None
        self.history = ""

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.show()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

    def show(self):
        """Show the print window and redirect stdout."""
        self.window = QDialog(self.parent)
        self.window.setWindowTitle("Print Output")
        self.window.finished.connect(self.close)

        self.text_area = QTextEdit(self.window)
        self.text_area.setReadOnly(True)

        # Create a layout for the buttons
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear", self.window)
        clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(clear_button)

        close_button = QPushButton("Close", self.window)
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        # Create the main layout
        layout = QVBoxLayout(self.window)
        layout.addWidget(self.text_area)
        layout.addLayout(
            button_layout
        )  # Add the button layout to the main layout
        self.window.setLayout(layout)
        self.window.resize(800, 400)
        self.window.show()

        # Insert history into the text area
        self.text_area.setPlainText(self.history)

        # Redirect stdout to capture the output
        self.old_stdout = sys.stdout
        sys.stdout = self

    def close(self):
        """Restore stdout and close the print window."""
        self.text_area = None
        if self.old_stdout:
            sys.stdout = self.old_stdout
            self.old_stdout = None
        if self.window:
            self.window.close()
            self.window = None

    def clear_text(self):
        """Clear the text area and reset history."""
        self.history = ""
        if self.text_area:
            self.text_area.clear()

    def write(self, message, separate=False):
        """Write text to the text area and update history."""
        separator = "\n" + "=" * 150 + "\n"
        if separate:
            self.history += separator + message
        else:
            self.history += message
        if self.text_area:
            self.text_area.ensureCursorVisible()
            if separate:
                self.text_area.append(separator + message)
            else:
                self.text_area.append(message)
            # self.text_area.append(separator + message)
            self.text_area.ensureCursorVisible()
            QApplication.processEvents()  # Ensure the GUI updates in real-time

    def flush(self):
        """Flush the text area."""
        if self.text_area:
            self.text_area.ensureCursorVisible()


class DictWindow(dict):
    """Class to create an options window."""

    def __init__(self, parent, options, name=None):
        self.parent = parent
        if isinstance(options, (dict, DictWindow)) and all(
            isinstance(value, (dict, DictWindow)) for value in options.values()
        ):
            super().__init__({k: v.copy() for k, v in options.items()})
            self._defaults = {k: v.copy() for k, v in options.items()}
        elif isinstance(options, (dict, DictWindow)):
            super().__init__(options.copy())
            self._defaults = options.copy()
        else:
            super().__init__({})
            self._defaults = {}

        self.name = self._format_name(name)
        self.entries = {}
        self.qt_window = None
        self.tabs = None

    @property
    def defaults(self):
        """Get the default options."""
        return self._defaults

    def window(self):
        """Create the options window."""
        self.destroy()

        self.qt_window = QDialog(self.parent)
        self.qt_window.setWindowTitle(self.name)
        self.qt_window.finished.connect(self.destroy)
        layout = QVBoxLayout()

        def conv_items(key, value):
            label = QLabel(key)
            entry = QLineEdit()
            if isinstance(value, list):
                entry.setText(", ".join([format_number(v, 5) for v in value]))
            elif isinstance(value, (int, float)):
                entry.setText(format_number(value, 5))
            else:
                entry.setText(str(value))

            return label, entry

        if all(
            isinstance(value, (dict, DictWindow)) for value in self.values()
        ):
            self.tabs = QTabWidget()
            layout.addWidget(self.tabs)

            for key, sub_dict in self.items():
                tab = QWidget()
                tab_layout = QVBoxLayout()
                form_layout = QFormLayout()
                self.entries[key] = {}

                for sub_key, sub_value in sub_dict.items():
                    label, entry = conv_items(sub_key, sub_value)
                    form_layout.addRow(label, entry)
                    self.entries[key][sub_key] = entry

                tab_layout.addLayout(form_layout)
                tab.setLayout(tab_layout)
                self.tabs.addTab(tab, key)
        else:
            form_layout = QFormLayout()
            self.entries = {}

            for key, value in self.items():
                label, entry = conv_items(key, value)
                form_layout.addRow(label, entry)
                self.entries[key] = entry

            layout.addLayout(form_layout)

        button_layout = QHBoxLayout()

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save)
        button_layout.addWidget(save_button)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_option)
        button_layout.addWidget(add_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        button_layout.addWidget(reset_button)

        layout.addLayout(button_layout)
        self.qt_window.setLayout(layout)
        self.qt_window.exec_()

    def add_option(self):
        """Add a new option to the window."""
        dialog = CustomSimpleDialog(
            self.parent, "Add Option", "Enter the key:", width=50
        )
        if dialog.exec_() == QDialog.Accepted:
            key = dialog.result
            if key:
                if self.tabs is not None:
                    current_tab = self.tabs.tabText(self.tabs.currentIndex())
                    if current_tab:
                        self[current_tab][key] = ""
                else:
                    self[key] = ""
                self.window()

    def save(self):
        """Save the options to the dictionary."""

        def eval_value(value):
            if value.lower() == "none":
                return None
            elif value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                try:
                    val = eval(value, {}, {"inf": np.inf})
                    if isinstance(val, float) and val.is_integer():
                        return int(val)
                    return val
                except (NameError, SyntaxError):
                    return value

        for key, entry in self._get_dict(self.entries).items():
            value = entry.text()
            to_update = self._get_dict(self)
            if "," in value:
                to_update[key] = [
                    eval_value(v.strip()) for v in value.split(",")
                ]
            else:
                to_update[key] = eval_value(value)

        self.qt_window.accept()

    def reset(self):
        """Reset the options to the default values."""
        reply = QMessageBox.question(
            self.parent,
            "Confirm Reset",
            "Are you sure you want to reset to default options?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            to_reset = self._get_dict(self)
            to_reset.clear()
            to_reset.update(self._get_dict(self._defaults.copy()))
            self.window()  # Recreate the window to reflect the reset options

    def destroy(self):
        """Destroy the options window."""
        if self.qt_window is not None and self.qt_window.isVisible():
            self.qt_window.close()

    def _format_name(self, name):
        """Format the name of the options window."""
        if isinstance(name, str):
            return (
                name if "option" in name.lower() else f"{name.title()} Options"
            )
        else:
            return "Options"

    def _get_dict(self, source):
        if self.tabs is not None:
            return source[self.tabs.tabText(self.tabs.currentIndex())]
        else:
            return source


class DataTreeWindow:
    def __init__(
        self,
        root,
        df,
        tree_cols,
        add_row_callback=None,
        use_row_callback=None,
        **kwargs,
    ):
        self.root = root
        self.window = None
        self.tabs = None
        self.tree1 = None  # Initialize tree attribute
        self.tree2 = None  # Initialize tree attribute
        self.name = kwargs.get("name", "Pinned Results")
        self.buttons = []

        # Initialize properties
        self.df = df
        self.tree_cols = tree_cols

        # Set the callback functions
        self.add_row_callback = add_row_callback
        self.use_row_callback = use_row_callback
        self.graphing_callback = kwargs.get("graphing_callback", None)

        self.df_base_cols = kwargs.get("df_base_cols", [])
        self.df_sort_cols = kwargs.get("df_sort_cols", [])
        self.tree_gr_cols = kwargs.get("tree_gr_cols", [])

        self.wide_cols = kwargs.get("wide_cols", [])
        self.narrow_cols = kwargs.get("narrow_cols", [])

    @property
    def df(self):
        if not hasattr(self, "_df"):
            self._df = pd.DataFrame()
        return self._df

    @df.setter
    def df(self, values):
        if isinstance(values, pd.DataFrame):
            self._df = values
        elif isinstance(values, list) and all(
            isinstance(col, str) for col in values
        ):
            self._df = pd.DataFrame(columns=values)

    @property
    def df_base_cols(self):
        if not hasattr(self, "_df_base_cols"):
            self._df_base_cols = []
        return self._df_base_cols

    @df_base_cols.setter
    def df_base_cols(self, values):
        if isinstance(values, list) and all(
            isinstance(col, str) for col in values
        ):
            self._df_base_cols = values

    @property
    def df_sort_cols(self):
        if not hasattr(self, "_df_sort_cols"):
            self._df_sort_cols = []
        return self._df_sort_cols

    @df_sort_cols.setter
    def df_sort_cols(self, values):
        if isinstance(values, list) and all(
            isinstance(col, str) for col in values
        ):
            self._df_sort_cols = values

    @property
    def tree_cols(self):
        if not hasattr(self, "_tree_cols"):
            return self.df.columns.tolist()
        return self._tree_cols

    @tree_cols.setter
    def tree_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._tree_cols = columns
        elif isinstance(columns, str):
            self._tree_cols.append(columns)

    @property
    def tree_gr_cols(self):
        if not hasattr(self, "_tree_gr_cols"):
            self._tree_gr_cols = []
        return self._tree_gr_cols

    @tree_gr_cols.setter
    def tree_gr_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._tree_gr_cols = [
                col for col in columns if col in self.tree_cols
            ]
        elif isinstance(columns, str) and columns in self.tree_cols:
            if not hasattr(self, "_tree_gr_cols"):
                self._tree_gr_cols = []
            self._tree_gr_cols.append(columns)

    @property
    def wide_cols(self):
        if not hasattr(self, "_wide_cols"):
            return []
        return self._wide_cols

    @wide_cols.setter
    def wide_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._wide_cols = [col for col in columns if col in self.tree_cols]
        elif isinstance(columns, str) and columns in self.tree_cols:
            if not hasattr(self, "_wide_cols"):
                self._wide_cols = []
            self._wide_cols.append(columns)

    @property
    def narrow_cols(self):
        if not hasattr(self, "_narrow_cols"):
            return []
        return self._narrow_cols

    @narrow_cols.setter
    def narrow_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._narrow_cols = [
                col for col in columns if col in self.tree_cols
            ]
        elif isinstance(columns, str) and columns in self.tree_cols:
            if not hasattr(self, "_narrow_cols"):
                self._narrow_cols = []
            self._narrow_cols.append(columns)

    def show(self):
        """View the pinned results in a new window."""
        self.window = QMainWindow(self.root)
        self.window.setWindowTitle(self.name)

        # Create a central widget and set it as the central widget of the QMainWindow
        central_widget = QWidget(self.window)
        self.window.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create a tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ----------------- Create the TreeView -----------------
        tree_tab = QWidget()
        tree_layout = QVBoxLayout(tree_tab)

        # Create a TreeView to display the fit results
        self.tree1 = QTreeWidget()
        self.tree1.setColumnCount(len(self.tree_cols) + 1)
        self.tree1.setHeaderLabels([" "] + self.tree_cols)
        self.tree1.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Set column widths
        window_width = 0
        self.tree1.setColumnWidth(0, 5)
        for i, col in enumerate(self.tree_cols):
            if col in self.narrow_cols:
                self.tree1.setColumnWidth(i + 1, 50)
                window_width += 50
            elif col in self.wide_cols:
                self.tree1.setColumnWidth(i + 1, 500)
                window_width += 500
            else:
                if col in self.df.columns and not self.df[col].empty:
                    # Calculate the width of the longest string in the column
                    font_metrics = QFontMetrics(self.tree1.font())
                    max_width = (
                        self.df[col]
                        .astype(str)
                        .apply(font_metrics.width)
                        .max()
                    )
                    max_width *= 1.3
                    self.tree1.setColumnWidth(
                        i + 1, int(max_width)
                    )  # Add some padding
                    window_width += int(max_width)
                else:
                    self.tree1.setColumnWidth(
                        i + 1, 100
                    )  # Set default width to 100
                    window_width += 100

        self.window.resize(max(window_width, 1200), 600)

        tree_layout.addWidget(self.tree1)
        self.tabs.addTab(tree_tab, "Pinned Results")

        # ----------------- Create the DataFrame view -----------------
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        # Create a TreeView to display the DataFrame
        self.tree2 = QTreeWidget()
        self.tree2.setColumnCount(len(self.df.columns) + 1)
        self.tree2.setHeaderLabels([" "] + self.df.columns.tolist())

        self.tree2.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree2.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tree2.setEditTriggers(QAbstractItemView.DoubleClicked)

        self.tree2.setColumnWidth(0, 5)

        data_layout.addWidget(self.tree2)
        self.tabs.addTab(data_tab, "DataFrame View")

        def calc_width(num, width, frac):
            if frac < 1:
                frac = int(1 / frac)
            return int(((frac + 1) * num + 1) * width / frac)

        # Create a frame for the buttons
        button_frame = QFrame(central_widget)
        button_layout = QHBoxLayout(button_frame)

        b_width = 100
        # Create frames for grouping buttons
        edit_frame = QFrame(button_frame)
        edit_frame.setFixedWidth(calc_width(2, b_width, 4))
        edit_layout = QHBoxLayout(edit_frame)
        button_layout.addWidget(edit_frame)

        action_frame = QFrame(button_frame)
        action_frame.setFixedWidth(calc_width(3, b_width, 4))
        action_layout = QHBoxLayout(action_frame)
        button_layout.addWidget(action_frame)

        navigation_frame = QFrame(button_frame)
        navigation_frame.setFixedWidth(calc_width(2, b_width, 4))
        navigation_layout = QHBoxLayout(navigation_frame)
        button_layout.addWidget(navigation_frame)

        # Initialize the buttons list
        self.buttons = []

        # Add buttons to the edit frame
        add_button = QPushButton("Add", edit_frame)
        add_button.setFixedWidth(b_width)
        add_button.clicked.connect(self.add_row)
        if self.add_row_callback is None:
            add_button.setDisabled(True)
            add_button.setStyleSheet("color: grey;")
        edit_layout.addWidget(add_button)
        self.buttons.append(add_button)

        remove_button = QPushButton("Remove", edit_frame)
        remove_button.setFixedWidth(b_width)
        remove_button.clicked.connect(self.remove_rows)
        edit_layout.addWidget(remove_button)
        self.buttons.append(remove_button)

        # Add buttons to the action frame
        plot_button = QPushButton("Plot", action_frame)
        plot_button.setFixedWidth(b_width)
        plot_button.clicked.connect(self.plot_rows)
        action_layout.addWidget(plot_button)
        self.buttons.append(plot_button)

        clear_button = QPushButton("Clear", action_frame)
        clear_button.setFixedWidth(b_width)
        clear_button.clicked.connect(self.unplot_rows)
        action_layout.addWidget(clear_button)
        self.buttons.append(clear_button)

        use_row_button = QPushButton("Use Values", action_frame)
        use_row_button.setFixedWidth(b_width)
        use_row_button.clicked.connect(self.use_row)
        if self.use_row_callback is None:
            use_row_button.setDisabled(True)
            use_row_button.setStyleSheet("color: grey;")
        action_layout.addWidget(use_row_button)
        self.buttons.append(use_row_button)

        # Add navigation buttons to the navigation frame
        up_button = QPushButton("Move Up", navigation_frame)
        up_button.setFixedWidth(b_width)
        up_button.clicked.connect(self.move_up)
        navigation_layout.addWidget(up_button)
        self.buttons.append(up_button)

        down_button = QPushButton("Move Down", navigation_frame)
        down_button.setFixedWidth(b_width)
        down_button.clicked.connect(self.move_down)
        navigation_layout.addWidget(down_button)
        self.buttons.append(down_button)

        layout.addWidget(button_frame)

        self.tree1.itemDoubleClicked.connect(self.edit_item)
        self.item_changed_connected = False

        self._refresh()
        # Connect the tab change signal to a slot
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.window.show()

    def on_tab_changed(self, index):
        """Enable or disable buttons based on the selected tab."""
        if self.tabs.tabText(index) == "DataFrame View":
            for button in self.buttons:
                button.setEnabled(False)
                button.setStyleSheet("color: grey;")
        else:
            for button in self.buttons:
                button.setEnabled(True)
                button.setStyleSheet("color: black;")
        self._refresh()

    def edit_item(self, item, column):
        """Make the item editable on double-click."""
        # self._refresh()

        if self.tree_cols[column - 1] in self.tree_gr_cols:
            return

        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.tree1.editItem(item, column)

        self.item_changed_connected = True
        self.tree1.itemChanged.connect(self._update_dataframe)
        self.tree1.itemSelectionChanged.connect(self._refresh)

    def _update_dataframe(self, item, column):
        """Update the DataFrame when an item is edited."""
        item_index = self.tree1.indexOfTopLevelItem(item)
        new_value = item.text(column)
        self.df.loc[item_index, self.tree_cols[column - 1]] = new_value

        self._refresh()

    def _refresh(self):
        """Refresh the pinned results in the treeview."""
        # Refresh the treeview col.replace(self.df_sort_cols[0], '')
        self.tree1.clear()
        self.tree2.clear()

        self.tree2.setColumnCount(len(self.df.columns) + 1)
        self.tree2.setHeaderLabels([" "] + self.df.columns.tolist())

        self._drop_nan_columns()

        for index, (_, row) in enumerate(self.df.iterrows()):
            display_row = [str(index)] + [
                str(row[col]) if col in row else "" for col in self.tree_cols
            ]
            for gr_col in self.tree_gr_cols:
                # Create a dict from the filtered columns using the prefix parts of the columns
                filtered_dict = {
                    col.replace(f"_{gr_col.lower()}", ""): format_number(
                        row[col], 2
                    )
                    for col in row.index
                    if col.endswith(f"_{gr_col.lower()}")
                    and not pd.isnull(row[col])
                    and row[col]
                }
                # Convert the dict to a string representation
                joined_values = str(filtered_dict)
                display_row[self.tree_cols.index(gr_col) + 1] = joined_values
            item1 = QTreeWidgetItem(display_row)
            item2 = QTreeWidgetItem(
                [str(index)] + [format_number(col, 5) for col in row]
            )
            if index % 2 == 0:
                for col in range(item1.columnCount()):
                    item1.setBackground(col, QBrush(QColor("#E2DDEF")))
                for col in range(item2.columnCount()):
                    item2.setBackground(col, QBrush(QColor("#E2DDEF")))
            self.tree1.addTopLevelItem(item1)
            self.tree2.addTopLevelItem(item2)

        if self.item_changed_connected:
            self.tree1.itemChanged.disconnect(self._update_dataframe)
            self.tree1.itemSelectionChanged.disconnect(self._refresh)
            self.item_changed_connected = False

    def _drop_nan_columns(self):
        """Helper function to drop all NaN columns of self.df."""
        # Get all columns that end with self.df_sort_cols[0]
        if self.df_sort_cols:
            base_to_drop = [
                col.replace("_" + self.df_sort_cols[0], "")
                for col in self.df.columns
                if col.endswith(self.df_sort_cols[0])
                and self.df[col].isna().all()
            ]
            cols_to_drop = [
                base_col + "_" + suffix
                for suffix in self.df_sort_cols
                for base_col in base_to_drop
                if base_col + "_" + suffix in self.df.columns
            ]
            self.df = self.df.drop(columns=cols_to_drop)

    def move_up(self):
        """Move the selected item up in the DataFrame and refresh the Treeview."""
        selected_items = self.tree1.selectedItems()
        if selected_items:
            indices = [
                self.tree1.indexOfTopLevelItem(item) for item in selected_items
            ]
            if min(indices) > 0:
                for index in sorted(indices):
                    # Swap rows in the DataFrame
                    self.df.iloc[[index, index - 1]] = self.df.iloc[
                        [index - 1, index]
                    ]
                self._refresh()
                # Reselect the moved items
                new_selection = [
                    self.tree1.topLevelItem(index - 1) for index in indices
                ]
                for item in new_selection:
                    item.setSelected(True)

    def move_down(self):
        """Move the selected item down in the DataFrame and refresh the Treeview."""
        selected_items = self.tree1.selectedItems()
        if selected_items:
            indices = [
                self.tree1.indexOfTopLevelItem(item) for item in selected_items
            ]
            if max(indices) < self.tree1.topLevelItemCount() - 1:
                for index in sorted(indices, reverse=True):
                    # Swap rows in the DataFrame
                    self.df.iloc[[index, index + 1]] = self.df.iloc[
                        [index + 1, index]
                    ]
                self._refresh()
                # Reselect the moved items
                new_selection = [
                    self.tree1.topLevelItem(index + 1) for index in indices
                ]
                for item in new_selection:
                    item.setSelected(True)

    def append_df(self, values):
        """Utility function to assist in adding data to the internal DataFrame."""

        def col_sort(column_name):
            """Custom sort key that prioritizes n1, n2, and then reverse alphabetical order."""
            match = re.match(r"(^[a-zA-Z]+)_?([0-9_]+)(_std)?$", column_name)
            if match:
                a1 = len(match.group(1))
                a2 = len(match.group(3)) if match.group(3) is not None else 0
                n1, n2 = (
                    map(int, match.group(2).split("_"))
                    if "_" in match.group(2)
                    else (int(match.group(2)), 0)
                )
                return (a2, n1, n2, a1)
            return (0, 0, 0, column_name)

        def safe_eval(x):
            try:
                return eval(x, {}, {"inf": np.inf})
            except (SyntaxError, TypeError, ValueError, NameError):
                return x

        # Ensure df_base_cols are in the keys
        if isinstance(values, pd.Series):
            values = values.to_frame().T
        elif isinstance(values, dict):
            values = pd.DataFrame([values])

        if isinstance(values, pd.DataFrame):
            if not all(col in values.columns for col in self.df_base_cols):
                print(values.columns)
                raise ValueError(
                    "All df_base_cols must be in the DataFrame columns"
                )
            value_cols = values.columns.difference(self.df_base_cols)
            values[value_cols] = values[value_cols].map(safe_eval)

            # Sort columns based on df_sort_cols
            sort_base = []
            ind = 0
            while sort_base == [] and ind < len(self.df_sort_cols):
                sort_base = [
                    col.replace("_" + self.df_sort_cols[ind], "")
                    for col in values.columns
                    if self.df_sort_cols[ind] in col
                ]
                ind += 1
            # sort_base = [col.replace("_"+self.df_sort_cols[0], '') for col in self.df.columns if self.df_sort_cols[0] in col]
            sort_base = list(sorted(reversed(sorted(sort_base)), key=col_sort))
            if all(col in values.columns for col in sort_base):
                for base in sort_base:
                    values[f"{base}_{self.df_sort_cols[max(0,ind-2)]}"] = (
                        values[base]
                    )
                values.drop(columns=sort_base, inplace=True)

            self._df = pd.concat([self._df, values], ignore_index=True)
        else:
            raise ValueError(
                "Unsupported data type for appending to DataFrame"
            )

        # # Sort columns based on df_sort_cols
        sort_base = [
            col.replace("_" + self.df_sort_cols[0], "")
            for col in self.df.columns
            if self.df_sort_cols[0] in col
        ]
        sort_base = list(sorted(reversed(sorted(sort_base)), key=col_sort))

        sorted_columns = []
        for identifier in self.df_sort_cols:
            sorted_columns += [f"{base}_{identifier}" for base in sort_base]

        other_cols = self.df.columns.difference(
            self.df_base_cols + sorted_columns
        ).tolist()

        self._df = self._df[self.df_base_cols + other_cols + sorted_columns]
        self._df[self.df_base_cols + other_cols] = self._df[
            self.df_base_cols + other_cols
        ].fillna("")

    def add_row(self):
        """Add a new pinned result to the treeview."""
        if self.add_row_callback:
            self.add_row_callback()
            self._refresh()
        else:
            QMessageBox.warning(
                self.root,
                "No Callback",
                "A callback is needed for this function.",
            )

    def add_row_popup(self, default_name):
        """Add a new pinned result to the treeview."""

        # Get the desired name
        dialog = CustomSimpleDialog(
            self.root,
            "Enter Name",
            "Enter the name for the pinned result:",
            initialvalue=default_name,
            width=max(50, min(500, len(default_name) * 10)),
        )
        if dialog.exec_() == QDialog.Rejected:
            return
        return dialog.result

    def use_row(self):
        """Use the values of a pinned result."""
        if self.use_row_callback:
            selected_items = self.tree1.selectedItems()
            if not selected_items:
                QMessageBox.warning(self.root, "Warning", "No row selected.")
                return

            # Get the first selected row
            selected_item = selected_items[0]
            item_index = self.tree1.indexOfTopLevelItem(selected_item)

            # Get the corresponding row from the DataFrame
            selected_row = self.df.iloc[item_index].to_dict()

            self.use_row_callback(selected_row)
        else:
            QMessageBox.warning(
                self.root,
                "No Callback",
                "A callback is needed for this function.",
            )

    def plot_rows(self):
        """Plot the selected pinned results."""
        selected_items = self.tree1.selectedItems()
        if not selected_items:
            # Clear the existing plots
            self._refresh()
            return

        format_strings = [
            style + color
            for style in ["--", "-.", ":"]
            for color in ["g", "c", "m", "k"]
        ]
        used_format_strings = self.df["Show"].tolist()
        format_usage_count = {
            fmt: used_format_strings.count(fmt) for fmt in format_strings
        }

        for item in selected_items:
            # Find the least used format string
            min_usage = min(format_usage_count.values())
            for fmt in format_strings:
                if format_usage_count[fmt] == min_usage:
                    format_string = fmt
                    break

            # Update the format usage count
            format_usage_count[format_string] += 1

            self.df.loc[self.df["Name"] == item.text(1), "Show"] = (
                format_string
            )
            used_format_strings.append(format_string)

        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()

    def unplot_rows(self):
        """Remove the selected pinned results from the plots."""
        selected_items = self.tree1.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            # Update the "Show" column to None or ""
            self.df.loc[self.df["Name"] == item.text(1), "Show"] = ""

        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()

    def remove_rows(self):
        """Remove the selected pinned results from the treeview."""
        selected_items = self.tree1.selectedItems()

        if not selected_items:
            return

        selected_item_info = [
            f"Index: {self.tree1.indexOfTopLevelItem(item)}, Name: {item.text(1)}"
            for item in selected_items
        ]
        message = (
            "Are you sure you want to remove the following rows?\n\n"
            + "\n".join(selected_item_info)
        )
        reply = QMessageBox.question(
            self.root,
            "Remove Rows",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        for item in selected_items:

            self.df = self.df.drop(
                self.df[
                    (self.df.index == int(item.text(0)))
                    & (self.df["Name"] == item.text(1))
                ].index
            )

            tree_index = self.tree1.indexOfTopLevelItem(item)
            self.tree1.takeTopLevelItem(tree_index)

        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()


class CalcRCWindow:
    def __init__(self, main, root):
        self.main = main
        self.root = root
        self.window = None
        self.r_entry = None
        self.c_entry = None
        self.freq_entry = None

    def show(self):
        """Show the window to calculate R, C, or freq."""
        self.window = QDialog(self.root)
        self.window.setModal(False)
        self.window.setWindowTitle("Calculate RC")

        layout = QVBoxLayout(self.window)

        form_layout = QFormLayout()
        self.r_entry = QLineEdit()
        form_layout.addRow("R:", self.r_entry)

        self.c_entry = QLineEdit()
        form_layout.addRow("C:", self.c_entry)

        self.freq_entry = QLineEdit()
        form_layout.addRow("freq:", self.freq_entry)

        layout.addLayout(form_layout)

        calc_button = QPushButton("Calculate", self.window)
        calc_button.clicked.connect(self.calculate_rc_freq)
        layout.addWidget(calc_button)

        self.window.setLayout(layout)
        # self.window.exec_()
        self.window.show()

    def calculate_rc_freq(self):
        """Calculate R, C, or freq based on the other two values."""
        try:
            # Get the parameter names and values
            params = {p.name: p.values[0] for p in self.main.parameters}

            # Helper function to get the value from the entry or parameter
            def get_value(entry_value):
                try:
                    return (
                        float(eval(entry_value, params, {"inf": np.inf}))
                        if entry_value != ""
                        else None
                    )
                except (ValueError, SyntaxError):
                    return None

            # Get the values from the entries
            r_value = get_value(self.r_entry.text())
            c_value = get_value(self.c_entry.text())
            freq_value = get_value(self.freq_entry.text())

            if r_value is None:
                if c_value is not None and freq_value is not None:
                    res = 1 / (2 * np.pi * freq_value * c_value)
                    self.r_entry.setText(format_number(res, 6))
                else:
                    raise ValueError(
                        "Please provide values for C and freq to calculate R."
                    )
            elif c_value is None:
                if r_value is not None and freq_value is not None:
                    res = 1 / (2 * np.pi * freq_value * r_value)
                    self.c_entry.setText(format_number(res, 6))
                else:
                    raise ValueError(
                        "Please provide values for R and freq to calculate C."
                    )
            elif freq_value is None:
                if r_value is not None and c_value is not None:
                    res = 1 / (2 * np.pi * r_value * c_value)
                    self.freq_entry.setText(format_number(res, 6))
                else:
                    raise ValueError(
                        "Please provide values for R and C to calculate freq."
                    )
            else:
                raise ValueError(
                    "Please leave one field empty to calculate its value."
                )

        except ValueError as e:
            QMessageBox.critical(self.root, "Error", str(e))


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
        self.root = root
        self.window = None
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

        self.destroy()
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
            defaults = np.array(
                [entry.default_values for entry in self.entries]
            )
        if checked is True or checked == "checked":
            self.checked_values = defaults
        elif checked is False or checked == "unchecked":
            self.unchecked_values = defaults
        else:
            self.values = defaults


class DataViewer(QMainWindow):
    def __init__(self, data, parent=None, name=None):
        super().__init__(parent)
        self.data = data
        self.parent = parent
        self.name = name
        self.initUI()

    def initUI(self):
        self.setWindowTitle(
            f"Data Viewer - {self.name}" if self.name else "Data Viewer"
        )
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.setCentralWidget(central_widget)

        self.setStyleSheet(
            "background-color: #f0f0f0;"
        )  # Set overall background color

        self.populate_table(self.data)
        self.table.cellDoubleClicked.connect(self.get_value)
        self.table.itemChanged.connect(self.set_value)
        self.show()

    def populate_table(self, data):
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            self.table.setRowCount(data.shape[0])
            self.table.setColumnCount(data.shape[1])
            self.table.setHorizontalHeaderLabels(
                [str(col) for col in data.columns]
            )
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    item = QTableWidgetItem(str(data.iloc[row, col]))
                    item.setBackground(QColor(240, 255, 255))
                    self.table.setItem(row, col, item)
            self.table.horizontalHeader().setStretchLastSection(
                False
            )  # Do not stretch the last column
        else:
            self.table.setRowCount(len(data))
            self.table.setColumnCount(2)
            if isinstance(data, (list, tuple, set)):
                data = {str(i): v for i, v in enumerate(data)}
                self.table.setHorizontalHeaderLabels(["Index", "Value"])
            else:
                self.table.setHorizontalHeaderLabels(["Key", "Value"])

            for row, (key, value) in enumerate(data.items()):
                key_item = QTableWidgetItem(str(key))
                value_item = QTableWidgetItem(str(value))
                key_item.setBackground(
                    QColor(0, 255, 255)
                )  # Set background color for key column
                value_item.setBackground(QColor(240, 255, 255))
                self.table.setItem(row, 0, key_item)
                self.table.setItem(row, 1, value_item)
            self.table.horizontalHeader().setStretchLastSection(
                True
            )  # Stretch value column to fill width

    def get_value(self, row, column):
        if isinstance(self.data, np.ndarray):
            if len(self.data.shape) > 1:
                value = self.data[row, column]
            else:
                value = self.data[row]
        elif isinstance(self.data, pd.DataFrame):
            value = self.data.iloc[row, column]
        else:
            label = self.table.horizontalHeaderItem(0).text()
            key = self.table.item(row, 0).text()
            if label == "Key":
                value = self.data[key]
            else:
                value = list(self.data)[int(key)]
            if not isinstance(value, str) and hasattr(value, "__iter__"):
                self.viewer = DataViewer(
                    value, self, name=key
                )  # Use self.viewer and pass name
                self.viewer.show()

    def set_value(self, item):
        if isinstance(self.data, np.ndarray):
            if len(self.data.shape) > 1:
                dtype = type(self.data[item.row(), item.column()])
                self.data[item.row(), item.column()] = dtype(item.text())
            else:
                dtype = type(self.data[item.row()])
                self.data[item.row()] = dtype(item.text())
        elif isinstance(self.data, pd.DataFrame):
            dtype = type(self.data.iloc[item.row(), item.column()])
            self.data.iloc[item.row(), item.column()] = dtype(item.text())
        else:
            if item.column() == 0:
                return
            key = self.table.item(item.row(), 0).text()
            if self.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            dtype = (
                type(self.data[key])
                if not isinstance(self.data, set)
                else type(list(self.data)[key])
            )
            if dtype not in (int, float, str, np.number):
                return
            if isinstance(self.data, set):
                data_list = list(self.data)
                data_list[key] = dtype(item.text())
                self.data = set(data_list)
            else:
                self.data[key] = dtype(item.text())

        if self.parent and self.name:
            key = self.name
            if self.parent.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            self.parent.data[key] = self.data  # Update parent data

    def closeEvent(self, event):
        # Handle the close event to ensure the application closes properly
        if self.parent:
            self.parent.refresh_table()  # Refresh parent table on close
        event.accept()
        self.deleteLater()  # Ensure the widget is properly deleted

    def refresh_table(self):
        self.populate_table(self.data)
