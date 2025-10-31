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
    QWidget,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
)


class DataViewer(QMainWindow):
    def __init__(self, data: Any, parent: Any = None, name: str = ""):
        super().__init__(parent)
        self.data = data
        self.parent: Any = parent
        self.name = name
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
        self.table.installEventFilter(self)
        self.show()

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
                key_item.setBackground(QColor(0, 255, 255))  # Set background color for key column
                value_item.setBackground(QColor(240, 255, 255))
                self.table.setItem(row, 0, key_item)
                self.table.setItem(row, 1, value_item)
            self.table.setVerticalHeaderLabels([str(i) for i in range(len(data))])
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
                assert isinstance(self.data, (dict)), "Unsupported data type"
                value = self.data[key]
            else:
                assert isinstance(self.data, (list, tuple, set)), "Unsupported data type"
                value = list(self.data)[int(key)]
            if not isinstance(value, str) and hasattr(value, "__iter__"):
                self.viewer = DataViewer(value, self, name=key)  # Use self.viewer and pass name
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
            self.data.iloc[item.row(), item.column()] = dtype(item.text())  # type: ignore
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
            self.parent.data[key] = self.data  # Update parent data # type: ignore

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
                    del self.data[key]  # type: ignore
                self.refresh_table()
