# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from typing import Any

import numpy as np
import pandas as pd
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (
    QMenu,
    QWidget,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QInputDialog,
    QTableWidget,
    QTableWidgetItem,
)


class DataViewer(QMainWindow):
    def __init__(self, data: Any, parent: Any = None, name: str = "", initialize: bool = True):
        super().__init__(parent)
        self.data: Any = data
        self.root: Any = parent
        self.name = name
        self.index = {}
        if initialize:
            self.initUI()

    def initUI(self):
        self.setWindowTitle(f"Data Viewer - {self.name}" if self.name else "Data Viewer")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.setCentralWidget(central_widget)

        self.setStyleSheet("background-color: #f0f0f0;")  # Set overall background color

        self.populate_table(self.data)

        self.table.cellDoubleClicked.connect(self.get_value)
        self.table.itemChanged.connect(self.set_value)

        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)

        self.table.installEventFilter(self)
        self.show()

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            if source is self.table and event.key() == Qt.Key_Delete:
                selected_rows = self.table.selectionModel().selectedRows()
                if selected_rows:
                    self.delete_row([row.row() for row in selected_rows])
                return True
            elif event.key() == Qt.Key_Insert:
                self.add_row()
                return True
        return super().eventFilter(source, event)

    def show_context_menu(self, position):
        menu = QMenu()

        add_action = menu.addAction("Add Row")
        add_action.triggered.connect(self.add_row)

        delete_action = menu.addAction("Delete Row")
        delete_action.triggered.connect(lambda: self.delete_row([self.table.currentRow()]))

        menu.exec_(self.table.viewport().mapToGlobal(position))

    def populate_table(self, data):
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            self.table.setRowCount(data.shape[0])
            self.table.setColumnCount(data.shape[1])
            self.table.setHorizontalHeaderLabels([str(col) for col in data.columns])
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    item = QTableWidgetItem(str(data.iloc[row, col]))
                    item.setBackground(QColor(240, 255, 255))
                    self.table.setItem(row, col, item)
            self.table.setVerticalHeaderLabels([str(i) for i in range(data.shape[0])])
            self.table.horizontalHeader().setStretchLastSection(False)
            self.index = {i: idx for i, idx in enumerate(data.index)}
        else:
            self.table.setRowCount(len(data))
            self.table.setColumnCount(2)
            if isinstance(data, (list, tuple, set)):
                data = {str(i): v for i, v in enumerate(data)}
                self.table.setHorizontalHeaderLabels(["Index", "Value"])
                self.index = {i: i for i in range(len(data))}
            else:
                self.table.setHorizontalHeaderLabels(["Key", "Value"])
                self.index = {i: k for i, k in enumerate(data.keys())}

            for row, (key, value) in enumerate(data.items()):
                key_item = QTableWidgetItem(str(key))
                value_item = QTableWidgetItem(str(value))
                key_item.setBackground(QColor(0, 255, 255))  # Set background color for key column
                value_item.setBackground(QColor(240, 255, 255))
                self.table.setItem(row, 0, key_item)
                self.table.setItem(row, 1, value_item)
            self.table.setVerticalHeaderLabels([str(i) for i in range(len(data))])
            self.table.horizontalHeader().setStretchLastSection(True)

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
                value = self.data[key]  # type: ignore
            else:
                value = list(self.data)[int(key)]
            if not isinstance(value, str) and hasattr(value, "__iter__"):
                self.viewer = DataViewer(value, self, name=key)  # Use self.viewer and pass name
                self.viewer.show()

    def set_value(self, item: Any):
        if isinstance(self.data, np.ndarray):
            if len(self.data.shape) > 1:
                dtype = type(self.data[item.row(), item.column()])
                self.data[item.row(), item.column()] = dtype(item.text())
            else:
                dtype = type(self.data[item.row()])
                self.data[item.row()] = dtype(item.text())
        elif isinstance(self.data, pd.DataFrame):
            dtype = type(self.data.iloc[item.row(), item.column()])
            self.data.iloc[item.row(), item.column()] = dtype(item.text())  # type: ignore
        elif item.column() == 0 and isinstance(self.data, dict):
            old_key = self.index[item.row()]
            if isinstance(self.data, dict):
                key = self.table.item(item.row(), 0).text()
                self.data = {k if k != old_key else key: self.data[k] for k in self.data}
        else:
            if item.column() == 0:
                old_key = self.index[item.row()]
                self.table.setItem(item.row(), 0, QTableWidgetItem(str(old_key)))
                return
            key = self.table.item(item.row(), 0).text()
            print(key)
            if self.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            print(key)
            dtype = (
                type(self.data[key])
                if not isinstance(self.data, set)
                else type(list(self.data)[key])
            )
            if dtype not in (int, float, str, np.number):
                print("Unsupported data type for editing.")
                return
            if isinstance(self.data, set):
                print("Editing set")
                data_list = list(self.data)
                data_list[key] = dtype(item.text())
                self.data = set(data_list)
            else:
                print("Editing dict or list")
                self.data[key] = dtype(item.text())

        if self.root and isinstance(self.root, DataViewer) and self.name:
            key = self.name
            print(key)
            if self.root.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            print("saving to root")
            self.root.data[key] = self.data  # Update parent data # type: ignore

    def closeEvent(self, event):
        # Handle the close event to ensure the application closes properly
        if self.root and isinstance(self.root, DataViewer):
            self.root.refresh_table()  # Refresh parent table on close
        event.accept()
        self.deleteLater()  # Ensure the widget is properly deleted

    def refresh_table(self):
        self.populate_table(self.data)

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
                    del self.data[key]  # type: ignore
                self.refresh_table()

    def add_row(self):
        """Add a new row to the data table."""
        if isinstance(self.data, np.ndarray):
            # For numpy arrays
            if len(self.data.shape) > 1:
                # 2D array
                empty_row = np.zeros((1, self.data.shape[1]), dtype=self.data.dtype)
                self.data = np.vstack([self.data, empty_row])
            else:
                # 1D array
                self.data = np.append(self.data, 0)

        elif isinstance(self.data, pd.DataFrame):
            # For pandas DataFrames
            self.data.loc[len(self.data)] = [None] * len(self.data.columns)

        else:
            # For dictionaries and other mappings
            if self.table.horizontalHeaderItem(0).text() == "Key":
                # Ask user for a new key
                key, ok = QInputDialog.getText(self, "New Entry", "Enter key for the new entry:")
                if ok and key:
                    self.data[key] = ""  # Add empty value
                else:
                    return  # User canceled
            else:
                # For list-like objects
                if isinstance(self.data, list):
                    self.data.append("")
                elif isinstance(self.data, tuple):
                    self.data = self.data + ("",)
                elif isinstance(self.data, set):
                    self.data.add("")

        # Refresh the table to show the new row
        self.refresh_table()
