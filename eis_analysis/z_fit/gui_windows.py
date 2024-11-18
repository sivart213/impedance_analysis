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

from PyQt5.QtCore import Qt, QEvent
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
    QListWidget,
    QListWidgetItem,
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
        try:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.tree1.editItem(item, column)

            self.item_changed_connected = True
            self.tree1.itemChanged.connect(self._update_dataframe)
            self.tree1.itemSelectionChanged.connect(self._refresh)
        except RuntimeError as exc:
            QMessageBox.critical(self.root, "Error", f"Error in editing of item: {exc}.")
            pass

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
        self.table.installEventFilter(self)
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
            self.table.setVerticalHeaderLabels(
                [str(i) for i in range(data.shape[0])]
            )
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
            self.table.setVerticalHeaderLabels(
                [str(i) for i in range(len(data))]
            )
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

        if self.parent and isinstance(self.parent, DataViewer) and self.name:
            key = self.name
            if self.parent.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            self.parent.data[key] = self.data  # Update parent data

    def closeEvent(self, event):
        # Handle the close event to ensure the application closes properly
        if self.parent and isinstance(self.parent, DataViewer):
            self.parent.refresh_table()  # Refresh parent table on close
        event.accept()
        self.deleteLater()  # Ensure the widget is properly deleted

    def refresh_table(self):
        self.populate_table(self.data)

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            if source is self.table:
                selected_rows = self.table.selectionModel().selectedRows()
                if selected_rows:
                    # for row in selected_rows:
                    #     self.delete_row(row.row())
                    # self.delete_row(selected_rows[0].row())
                    self.delete_row([row.row() for row in selected_rows])
                return True
        return super().eventFilter(source, event)

    def delete_row(self, rows):
        reply = QMessageBox.question(
            self,
            "Delete Row(s)",
            "Are you sure you want to delete data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            for row in sorted(rows, reverse=True):
                if isinstance(self.data, np.ndarray):
                    self.data = np.delete(self.data, row, axis=0)
                elif isinstance(self.data, pd.DataFrame):
                    self.data = self.data.drop(self.data.index[row]).reset_index(drop=True)
                else:
                    key = self.table.item(row, 0).text()
                    if self.table.horizontalHeaderItem(0).text() != "Key":
                        key = int(key)
                    del self.data[key]
                self.refresh_table()



class ListWindow(QMainWindow):
    """
    A class to convert MFIA files using a GUI.
    Attributes:
        is_xlsx (bool): Flag to indicate if the files are in xlsx format.
        files (list): List of files to be processed.
        in_path (Path): Input directory path.
        t_files (DataFrame): DataFrame containing target files information.
        data_in (dict): Dictionary to store input data.
        data_org (dict): Dictionary to store organized data.
        keywords_list (QLineEdit): Line edit for keywords.
        checked_columns (list): List of checked columns.
        tab_widget (QTabWidget): Tab widget for grouping iterations.
        patterns_list (QListWidget): List widget for file patterns.
        columns_list (QListWidget): List widget for columns.
        files_table (QTableWidget): Table widget for files.
        datasets_tree (QTreeWidget): Tree widget for datasets.
        t_files_tree (QTreeWidget): Tree widget for target files.
        files_datasets_tab_widget (QTabWidget): Tab widget for files and datasets.
        save_format_combo (QComboBox): Combo box for save format.
        get_all_checkbox (QCheckBox): Checkbox to get all files.
    Methods:
        initUI(): Initializes the user interface.
        add_grouping_tab(search_terms="", reject_terms="", keys=""): Adds a grouping tab.
        eventFilter(source, event): Handles key press events for deleting items.
        browse_in_path(): Opens a dialog to browse for input directory.
        add_pattern(): Adds a pattern to the patterns list.
        update_files(): Updates the files table with the selected patterns.
        set_data_org(): Sets the organized data.
        apply_group(tab=None): Applies grouping logic to the data.
        apply_all_groups(): Applies all grouping logic to the data.
        update_tree_view(): Updates the tree view with the organized data.
        update_t_files(): Updates the target files tree view.
        convert_files(): Converts the files based on the selected patterns.
        remove_dataset(item): Removes a dataset from the organized data.
        reset_columns_list(): Resets the columns list with unique columns from the dataframes.
        refresh_columns_list(): Refreshes the columns list keeping any checked list.
        sort_columns_list(): Sorts the columns list.
        concat_data(): Concatenates the data.
        simplify_data(): Simplifies the data for fitting.
        save(): Saves the converted data.
        save_settings(): Saves the current settings.
        load_empty_settings(): Loads empty settings.
        load_all_settings(): Loads all settings.
        load_settings(full=True): Loads the settings from a file.
    """

    def __init__(self):
        super().__init__()
        self.is_xlsx = True
        self.files = None
        self.in_path = None
        self.t_files = None
        self.data_in = None
        self.data_org = None
        self.keywords_list = None
        self.checked_columns = []

        self.setWindowTitle("MFIA File Conversion")

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

    

        # List Widget for Columns
        self.columns_list = QListWidget()
        self.columns_list.setDragDropMode(
            QAbstractItemView.InternalMove
        )  # Enable drag-and-drop reordering
        reset_columns_button = QPushButton("Reset")
        reset_columns_button.clicked.connect(self.reset_columns_list)
        reset_columns_button.setFixedWidth(75)
        refresh_columns_button = QPushButton("Refresh")
        refresh_columns_button.clicked.connect(self.refresh_columns_list)
        refresh_columns_button.setFixedWidth(75)
        sort_columns_button = QPushButton("Sort")
        sort_columns_button.clicked.connect(self.sort_columns_list)
        sort_columns_button.setFixedWidth(75)

        columns_layout = QVBoxLayout()
        columns_layout.addWidget(QLabel("<b>Columns:</b>"))
        columns_layout.addWidget(self.columns_list)
        columns_buttons_layout = QHBoxLayout()
        columns_buttons_layout.addWidget(reset_columns_button)
        columns_buttons_layout.addWidget(refresh_columns_button)
        columns_buttons_layout.addWidget(sort_columns_button)
        columns_layout.addLayout(columns_buttons_layout)

        columns_frame = QFrame()
        columns_frame.setLayout(columns_layout)

        layout.addWidget(columns_frame)
        

        self.setCentralWidget(central_widget)

        # needs update_entries, highlight, clear, checked_names, show

    def reset_columns_list(self):
        """Populate the columns list with unique columns from the dataframes ignoring any checked list."""
        if self.data_org is None:
            QMessageBox.warning(
                self, "No Data", "Please load and group data first."
            )
            return

        # Flatten the data organization dictionary
        flattened_data = flatten_dict(self.data_org)

        # Collect all unique columns from the dataframes
        unique_columns = set()
        for df in flattened_data.values():
            if isinstance(df, pd.DataFrame):
                unique_columns.update(df.columns)

        check = Qt.Checked if len(unique_columns) < 10 else Qt.Unchecked
        # Populate the list widget with unique columns
        self.columns_list.clear()
        for col in unique_columns:
            item = QListWidgetItem(str(col))
            item.setFlags(
                item.flags() | Qt.ItemIsUserCheckable
            )  # Allow item to be checkable
            item.setCheckState(check)  # Default to checked
            self.columns_list.addItem(item)

    def refresh_columns_list(self):
        """Populate the columns list with unique columns from the dataframes keeping any checked list."""
        if self.data_org is None and self.checked_columns == []:
            QMessageBox.warning(
                self, "No Data", "Please load and group data first."
            )
            return

        if self.data_org is None:
            for col in self.checked_columns:
                item = QListWidgetItem(str(col))
                item.setFlags(
                    item.flags() | Qt.ItemIsUserCheckable
                )  # Allow item to be checkable
                item.setCheckState(Qt.Checked)  # Default to checked
                self.columns_list.addItem(item)
            return

        if self.columns_list.count() == 0:
            self.reset_columns_list()

        # # Flatten the data organization dictionary
        # flattened_data = flatten_dict(self.data_org)

        # # Collect all unique columns from the dataframes
        # unique_columns = set()
        # for df in flattened_data.values():
        #     if isinstance(df, pd.DataFrame):
        #         unique_columns.update(df.columns)

        # # Make sure checked columns match the unique columns
        # clean_check = []
        # for col in self.checked_columns:
        #     if col in unique_columns:
        #         clean_check.append(col)
        #     elif any(str(col) in str(u_col) for u_col in unique_columns):
        #         clean_check.append(str(col))
        #     else:
        #         for u_col in unique_columns:
        #             if str(u_col) in str(col):
        #                 clean_check.append(u_col)
        #                 break
        # self.checked_columns = (
        #     clean_check  # if clean_check != [] else self.checked_columns
        # )

        # items = {}
        # c_items = {}
        # for i in range(self.columns_list.count()):
        #     item = self.columns_list.item(i)
        #     if item.checkState() == Qt.Checked:
        #         c_items[item.text()] = item
        #     else:
        #         items[item.text()] = item

        # if self.checked_columns == [] and c_items:
        #     self.checked_columns = list(c_items.keys())
        # elif self.checked_columns != [] and c_items:
        #     self.checked_columns = [
        #         c for c in self.checked_columns if c in c_items
        #     ]
        #     if items:
        #         self.checked_columns.extend(
        #             [
        #                 c
        #                 for c in c_items.keys()
        #                 if c not in self.checked_columns
        #             ]
        #         )
        #     else:
        #         for key, item in c_items.items():
        #             if key not in self.checked_columns:
        #                 item.setCheckState(Qt.Unchecked)
        # elif self.checked_columns != [] and not c_items:
        #     for key, item in items.items():
        #         if key in self.checked_columns:
        #             item.setCheckState(Qt.Checked)
        # self.sort_columns_list()

    def sort_columns_list(self):
        """Sort the columns list by check state first, then alphabetically."""
        if self.columns_list.count() == 0:
            self.reset_columns_list()

        items = [(c, Qt.Checked) for c in self.checked_columns]
        unc_items = []
        for i in range(self.columns_list.count()):
            if self.columns_list.item(i).text() in self.checked_columns:
                continue
            item = self.columns_list.item(i)
            unc_items.append((item.text(), item.checkState()))

        # Sort by check state first, then alphabetically
        unc_items.sort(key=lambda x: (x[1] == Qt.Unchecked, x[0].lower()))

        items.extend(unc_items)

        self.columns_list.clear()
        for text, checkState in items:
            item = QListWidgetItem(text)
            item.setCheckState(checkState)
            self.columns_list.addItem(item)



        # columns_frame.setMaximumWidth(max_width)

            # def create_separator(frame=None):
        #     separator = QFrame(frame)
        #     separator.setFrameShape(QFrame.HLine)
        #     separator.setFrameShadow(QFrame.Sunken)
        #     return separator

        # max_width = 300

        # # Input Path
        # self.in_path_label = QLabel("<b>Input Path:</b>")
        # in_path_button = QPushButton("Browse")
        # in_path_button.setFixedWidth(100)
        # in_path_button.clicked.connect(self.browse_in_path)
        # in_path_layout = QHBoxLayout()
        # in_path_layout.addWidget(self.in_path_label)
        # in_path_layout.addWidget(in_path_button)
        # in_path_frame = QFrame()
        # in_path_frame.setLayout(in_path_layout)
        # in_path_frame.setFrameShape(QFrame.StyledPanel)
        # layout.addWidget(in_path_frame)

        # layout.addWidget(create_separator())

        # # File Patterns
        # self.patterns_list = QListWidget()
        # self.patterns_list.setMaximumWidth(max_width)
        # self.patterns_list.installEventFilter(self)
        # self.patterns_edit = QLineEdit()
        # self.patterns_edit.setPlaceholderText("pattern")
        # self.patterns_edit.returnPressed.connect(self.add_pattern)

        # patterns_layout = QVBoxLayout()
        # patterns_layout.addWidget(QLabel("<b>File Patterns:</b>"))
        # patterns_layout.addWidget(self.patterns_list)

        # patterns_edit_layout = QHBoxLayout()
        # patterns_edit_layout.addWidget(QLabel("Add Item:"))
        # patterns_edit_layout.addWidget(self.patterns_edit)

        # patterns_layout.addLayout(patterns_edit_layout)

        # patterns_frame = QFrame()
        # patterns_frame.setLayout(patterns_layout)
        # patterns_frame.setMaximumWidth(max_width)

        # # Tab Widget for Grouping Iterations
        # self.tab_widget = QTabWidget()
        # self.add_grouping_tab()

        # grid1 = QGridLayout()
        # grid1.addWidget(patterns_frame, 0, 0)
        # grid1.addWidget(self.tab_widget, 0, 1)
        # grid1.addWidget(columns_frame, 0, 2)

        # layout.addLayout(grid1)

        # layout.addWidget(create_separator())

        # self.files_datasets_tab_widget = QTabWidget()

        # # Files Table
        # self.files_table = QTableWidget()
        # self.files_table.setColumnCount(3)
        # self.files_table.setHorizontalHeaderLabels(
        #     ["File Name", "Relative Path", "Path"]
        # )
        # self.files_table.setColumnWidth(0, 150)
        # self.files_table.setColumnWidth(1, 250)
        # self.files_table.setColumnWidth(2, 250)
        # files_update_button = QPushButton("Update Files")
        # files_update_button.clicked.connect(self.update_files)
        # files_update_button.setFixedWidth(150)
        # files_layout = QVBoxLayout()
        # files_layout.addWidget(self.files_table)
        # files_layout.addWidget(files_update_button)
        # files_frame = QFrame()
        # files_frame.setLayout(files_layout)
        # files_frame.setMinimumWidth(int(max_width * 1.5))

        # self.files_datasets_tab_widget.addTab(files_frame, "Loaded Files")

        # # Tree View for Datasets
        # datasets_layout = QVBoxLayout()

        # self.datasets_tree = QTreeWidget()
        # self.datasets_tree.setHeaderLabels(["Name", "Details"])
        # self.datasets_tree.setColumnWidth(0, 400)
        # self.datasets_tree.installEventFilter(self)
        # self.datasets_tree.itemDoubleClicked.connect(self.view_data)
        # datasets_layout.addWidget(self.datasets_tree)

        # datasets_buttons = QHBoxLayout()

        # reset_gr_button = QPushButton("Reset")
        # reset_gr_button.clicked.connect(self.set_data_org)
        # reset_gr_button.setFixedWidth(150)
        # datasets_buttons.addWidget(reset_gr_button)

        # apply_gr_button = QPushButton("Apply Groups")
        # apply_gr_button.clicked.connect(self.apply_all_groups)
        # apply_gr_button.setFixedWidth(150)
        # datasets_buttons.addWidget(apply_gr_button)
        
        # datasets_rename_button = QPushButton("Rename Datasets")
        # datasets_rename_button.clicked.connect(self.simplify_data_keys)
        # datasets_rename_button.setFixedWidth(150)
        # datasets_buttons.addWidget(datasets_rename_button)

        # datasets_concat_button = QPushButton("Combine Datasets")
        # datasets_concat_button.clicked.connect(self.concat_data)
        # datasets_concat_button.setFixedWidth(150)
        # datasets_buttons.addWidget(datasets_concat_button)

        # datasets_layout.addLayout(datasets_buttons)

        # self.datasets_frame = QFrame()
        # self.datasets_frame.setLayout(datasets_layout)
        # self.datasets_frame.setMinimumWidth(int(max_width * 1.5))

        # self.files_datasets_tab_widget.addTab(self.datasets_frame, "Datasets")

        # # Tree Widget for t_files
        # t_files_layout = QVBoxLayout()

        # self.t_files_tree = QTreeWidget()
        # self.t_files_tree.setHeaderLabels(["Name", "ID"])
        # t_files_layout.addWidget(self.t_files_tree)

        # t_files_update_button = QPushButton("Update Target Files")
        # t_files_update_button.clicked.connect(self.update_t_files)
        # t_files_update_button.setFixedWidth(150)
        # self.get_all_checkbox = QCheckBox("Get All")
        # convert_button = QPushButton("Convert")
        # convert_button.setFixedWidth(100)
        # convert_button.clicked.connect(self.convert_files)

        # check_update_layout = QHBoxLayout()
        # check_update_layout.addWidget(t_files_update_button)
        # check_update_layout.addWidget(self.get_all_checkbox)
        # check_update_layout.addWidget(convert_button)
        # t_files_layout.addLayout(check_update_layout)

        # self.t_files_frame = QFrame()
        # self.t_files_frame.setLayout(t_files_layout)
        # self.t_files_frame.setMinimumWidth(int(max_width * 1.5))

        # self.files_datasets_tab_widget.addTab(
        #     self.t_files_frame, "Target Files"
        # )
        # self.files_datasets_tab_widget.setTabVisible(2, False)

        # # layout.addLayout(grid2)
        # layout.addWidget(self.files_datasets_tab_widget)
        # layout.addWidget(create_separator())

        # bottom_layout = QHBoxLayout()
        # # Save Format
        # self.save_format_combo = QComboBox()
        # self.save_format_combo.addItems(
        #     ["Use Columns", "Use Freq, Real, Imag"]
        # )
        # self.save_format_combo.setFixedWidth(150)
        # save_button = QPushButton("Save")
        # save_button.setFixedWidth(100)
        # save_button.clicked.connect(self.save)

        # save_format_layout = QHBoxLayout()
        # save_format_layout.addWidget(QLabel("Save Format: "))
        # save_format_layout.addWidget(self.save_format_combo)
        # save_format_layout.addWidget(save_button)
        # save_format_layout.setAlignment(Qt.AlignLeft)

        # # Add Buttons for Save and Load Settings
        # save_settings_button = QPushButton("Save Settings")
        # save_settings_button.clicked.connect(self.save_settings)
        # load_settings_button = QPushButton("Load Settings")
        # load_settings_button.clicked.connect(self.load_settings)
        # # load_all_settings_button = QPushButton("Load All Settings")
        # # load_all_settings_button.clicked.connect(self.load_all_settings)

        # # Add buttons to the layout
        # settings_buttons_layout = QHBoxLayout()
        # settings_buttons_layout.addWidget(save_settings_button)
        # settings_buttons_layout.addWidget(load_settings_button)
        # # settings_buttons_layout.addWidget(load_all_settings_button)
        # settings_buttons_layout.setAlignment(Qt.AlignRight)

        # bottom_layout.addLayout(save_format_layout)
        # bottom_layout.addLayout(settings_buttons_layout)

        # layout.addLayout(bottom_layout)


