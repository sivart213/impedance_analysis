# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
from typing import Any

import numpy as np
import pandas as pd
from PyQt5 import sip  # type: ignore
from PyQt5.QtGui import QBrush, QColor, QFontMetrics
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QWidget,
    QTabWidget,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QVBoxLayout,
    QInputDialog,
    QTreeWidgetItem,
    QAbstractItemView,
)

from ..string_ops import safe_eval, format_number
from ..widgets.generic_widgets import SimpleDialog


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


class DataTreeWindow:
    def __init__(
        self,
        root,
        df,
        tree_cols,
        add_row_callback=None,
        use_row_callback=None,
        replace_row_callback=None,
        **kwargs,
    ):
        self.root: Any = root
        self.window: Any = None
        self.tabs: Any = None
        self.tree1: Any = None  # Initialize tree attribute
        self.tree2: Any = None  # Initialize tree attribute
        self.name: str = kwargs.get("name", "Pinned Results")
        self.item_changed_connected = False
        self.buttons = []
        self._df_base_cols = []
        self._df_sort_cols = []
        self._tree_gr_cols = []
        self._wide_cols = []
        self._narrow_cols = []
        self._tree_cols = []
        self._df = pd.DataFrame()

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
        elif isinstance(values, list) and all(isinstance(col, str) for col in values):
            self._df = pd.DataFrame(columns=values)

    @property
    def df_base_cols(self):
        if not hasattr(self, "_df_base_cols"):
            self._df_base_cols = []
        return self._df_base_cols

    @df_base_cols.setter
    def df_base_cols(self, values):
        if isinstance(values, list) and all(isinstance(col, str) for col in values):
            self._df_base_cols = values

    @property
    def df_sort_cols(self):
        if not hasattr(self, "_df_sort_cols"):
            self._df_sort_cols = []
        return self._df_sort_cols

    @df_sort_cols.setter
    def df_sort_cols(self, values):
        if isinstance(values, list) and all(isinstance(col, str) for col in values):
            self._df_sort_cols = values

    @property
    def tree_cols(self):
        if not self._tree_cols:
            return self.df.columns.tolist()
        return self._tree_cols

    @tree_cols.setter
    def tree_cols(self, columns):
        if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
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
        if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
            self._tree_gr_cols = [col for col in columns if col in self.tree_cols]
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
        if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
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
        if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
            self._narrow_cols = [col for col in columns if col in self.tree_cols]
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
                    max_width = self.df[col].astype(str).apply(font_metrics.width).max()
                    max_width *= 1.5
                    self.tree1.setColumnWidth(i + 1, int(max_width))  # Add some padding
                    window_width += int(max_width)
                else:
                    self.tree1.setColumnWidth(i + 1, 100)  # Set default width to 100
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

        self.tree2.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for i in range(4):
            self.tree2.setColumnWidth(i, self.tree1.columnWidth(i))

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

    def close(self):
        """Close the pinned results window."""
        if self.window is not None and not sip.isdeleted(self.window):
            self.window.close()
            self.window.deleteLater()

    def on_tab_changed(self, index):
        """Enable or disable buttons based on the selected tab."""
        if self.tabs.tabText(index) == "DataFrame View":
            # going from main view to df view
            for button in self.buttons:
                button.setEnabled(False)
                button.setStyleSheet("color: grey;")
            # Synchronize selection from tree1 to tree2
            selected_items = self.tree1.selectedItems()
            if selected_items:
                selected_row = self.tree1.indexOfTopLevelItem(selected_items[0])

                self._refresh()

                self.tree2.topLevelItem(selected_row).setSelected(True)

        else:
            # going from df view to main view
            for button in self.buttons:
                button.setEnabled(True)
                button.setStyleSheet("color: black;")
            # Synchronize selection from tree2 to tree1
            selected_items = self.tree2.selectedItems()
            if selected_items:
                selected_row = self.tree2.indexOfTopLevelItem(selected_items[0])

                self._refresh()

                self.tree1.topLevelItem(selected_row).setSelected(True)

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

        # Iterate through the df rows
        for index, row in self.df.iterrows():
            index = int(index)  # type: ignore
            # Will have cols of the tree1
            display_row = [str(index)] + [
                str(row[col]) if col in row else "" for col in self.tree_cols
            ]
            # Construct the grouped columns
            for gr_col in self.tree_gr_cols:
                # Create a dict from the filtered columns using the prefix parts of the columns
                filtered_dict = {
                    col.replace(f"_{gr_col.lower()}", ""): format_number(row[col], 2)
                    for col in row.index
                    if col.endswith(f"_{gr_col.lower()}") and not pd.isnull(row[col]) and row[col]  # type: ignore
                }
                # Convert the dict to a string representation
                joined_values = str(filtered_dict)

                # Replace the corresponding column in the display_row with the joined values
                display_row[self.tree_cols.index(gr_col) + 1] = joined_values

            # Item (row) of tree1
            item1 = QTreeWidgetItem(display_row)
            # Item (row) of tree2
            item2 = QTreeWidgetItem([str(index)] + [format_number(col, 5) for col in row])

            # Set the background color for every other row
            if index % 2 == 0:
                for col in range(item1.columnCount()):
                    item1.setBackground(col, QBrush(QColor("#E2DDEF")))
                for col in range(item2.columnCount()):
                    item2.setBackground(col, QBrush(QColor("#E2DDEF")))

            # add the formatted items to the treeviews
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
                if col.endswith(self.df_sort_cols[0]) and self.df[col].isna().all()
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
            indices = [self.tree1.indexOfTopLevelItem(item) for item in selected_items]
            if min(indices) > 0:
                for index in sorted(indices):
                    # Swap rows in the DataFrame
                    self.df.iloc[[index, index - 1]] = self.df.iloc[[index - 1, index]]
                self._refresh()
                # Reselect the moved items
                new_selection = [self.tree1.topLevelItem(index - 1) for index in indices]
                for item in new_selection:
                    item.setSelected(True)

    def move_down(self):
        """Move the selected item down in the DataFrame and refresh the Treeview."""
        selected_items = self.tree1.selectedItems()
        if selected_items:
            indices = [self.tree1.indexOfTopLevelItem(item) for item in selected_items]
            if max(indices) < self.tree1.topLevelItemCount() - 1:
                for index in sorted(indices, reverse=True):
                    # Swap rows in the DataFrame
                    self.df.iloc[[index, index + 1]] = self.df.iloc[[index + 1, index]]
                self._refresh()
                # Reselect the moved items
                new_selection = [self.tree1.topLevelItem(index + 1) for index in indices]
                for item in new_selection:
                    item.setSelected(True)

    def remove_rows(self):
        """Remove the selected pinned results from the treeview."""
        selected_items = self.tree1.selectedItems()

        if not selected_items:
            return

        selected_item_info = [
            f"Index: {self.tree1.indexOfTopLevelItem(item)}, Name: {item.text(1)}"
            for item in selected_items
        ]
        message = "Are you sure you want to remove the following rows?\n\n" + "\n".join(
            selected_item_info
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
                    (self.df.index == int(item.text(0))) & (self.df["Name"] == item.text(1))
                ].index
            )

            tree_index = self.tree1.indexOfTopLevelItem(item)
            self.tree1.takeTopLevelItem(tree_index)
        # # Collect indices of rows to remove
        # indices_to_remove = [
        #     self.tree1.indexOfTopLevelItem(item) for item in selected_items
        # ]

        # # Drop rows from the DataFrame and reindex
        # self.df.drop(indices_to_remove, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()

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

        # Ensure df_base_cols are in the keys
        if isinstance(values, pd.Series):
            values = values.to_frame().T
        elif isinstance(values, dict):
            values = pd.DataFrame([values])

        if not isinstance(values, pd.DataFrame):
            raise ValueError("Unsupported data type for appending to DataFrame")
        if not all(col in values.columns for col in self.df_base_cols):
            print(values.columns)
            raise ValueError("All df_base_cols must be in the DataFrame columns")

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
                values[f"{base}_{self.df_sort_cols[max(0,ind-2)]}"] = values[base]
            values.drop(columns=sort_base, inplace=True)

        values = self.sanitize_stds(values)

        self._df = pd.concat([self._df, values], ignore_index=True)

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

        other_cols = self.df.columns.difference(self.df_base_cols + sorted_columns).tolist()

        self._df = self._df[self.df_base_cols + other_cols + sorted_columns]
        self._df[self.df_base_cols + other_cols] = self._df[self.df_base_cols + other_cols].fillna(
            ""
        )

        # if window is active/open, refresh the treeview
        if self.window:
            self._refresh()

    def sanitize_stds(self, df: pd.DataFrame | None = None):
        """
        Sanitize the std values in the DataFrame based on their associated values.
        If abs(power of std) > 2 * power of value, set std to 0.
        """
        if df is None:
            df = self.df
        if df.empty:
            return

        for index, row in df.iterrows():
            for col in df.columns:
                if col.endswith("_std"):
                    # Get the associated value column
                    value_col = col.replace("_std", "_values")
                    if value_col in df.columns:
                        std = row[col]
                        value = row[value_col]
                        try:
                            # Ensure both std and value are numeric
                            if (
                                isinstance(std, (int, float))
                                and isinstance(value, (int, float))
                                and std != 0
                                and std != np.nan
                            ):
                                if value == 0:
                                    # If value is 0, set std to 0
                                    df.at[index, col] = 0  # type: ignore
                                else:
                                    # pos value indicates std larger than value
                                    relative_ratio = np.log10(abs(std / value))
                                    val_pow = max(int(abs(np.log10(abs(value)))), 6)
                                    # Check the condition and sanitize
                                    if (
                                        relative_ratio > val_pow * 0.75
                                        or relative_ratio < -1 * val_pow
                                    ):
                                        df.at[index, col] = 0  # type: ignore
                        except ValueError:
                            pass
        return df

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
        result, ok = SimpleDialog.getResult(
            self.root,
            "Enter Name",
            "Enter the name for the pinned result:",
            content=default_name,
            width=max(50, min(500, len(default_name) * 10)),
        )
        if not ok or not result:
            return ""
        return result

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
            style + color for style in ["--", "-.", ":"] for color in ["g", "c", "m", "k"]
        ]
        used_format_strings = self.df["Show"].tolist()
        format_usage_count = {fmt: used_format_strings.count(fmt) for fmt in format_strings}
        format_string = ""
        for item in selected_items:
            # Find the least used format string
            min_usage = min(format_usage_count.values())
            for fmt in format_strings:
                if format_usage_count[fmt] == min_usage:
                    format_string = fmt
                    break
            if not format_string:
                break
            # Update the format usage count
            format_usage_count[format_string] += 1

            self.df.loc[self.df["Name"] == item.text(1), "Show"] = format_string
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
