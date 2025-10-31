# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney
pound !usr/bin/env python3
General function file
"""
# cSpell: ignore whos, linkk, stds, interp, savgol
# cSpell: ignoreRegExp /cmap[\w]?/gi
# cSpell:includeRegExp #.*
# cSpell:includeRegExp /(["]{3}|[']{3})[^\1]*?\1/g
# only comments and block strings will be checked for spelling.

from typing import Any

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QStackedLayout,
)

from .impedance_widgets import ImpedanceEntryWidget
from ..widgets.generic_widgets import AlignComboBox


def make_ax_labels(base: str, n: int, prefix: str = "") -> list[tuple[str, str]]:
    """Generate a list of names with indices based on the base name."""
    prefix = f"{prefix.strip()} " if prefix.strip() else ""

    if n <= 1:
        return [(base, f"{prefix}{base.upper()}")]

    long = ["Top", "Bottom"] if base.lower() == "y" else ["Left", "Right"]
    if n == 3:
        long = [long[0], "Middle", long[1]]
    elif n > 3:
        long = [long[0]] + [f"Mid ({i})" for i in range(1, n - 1)] + [long[1]]

    names = [(f"{base}{i+1}", f"{prefix}{name} {base.upper()}") for i, name in enumerate(long)]
    return names


class AxisWidgets(QWidget):
    """One row: Label, LineEdit, CheckBox"""

    def __init__(self, axis_name: str, parent=None):
        super().__init__(parent)
        self.axis_name = axis_name

        # layout = QHBoxLayout(self)
        self.label = QLabel(axis_name)
        self.var = ImpedanceEntryWidget(parent=parent)
        self.var.setAlignment(Qt.AlignCenter)
        self.check = QCheckBox("", parent)
        self.check.setToolTip(f"log({axis_name})")

        self.min_lim = QLineEdit()
        self.min_lim.setMinimumWidth(5)
        self.max_lim = QLineEdit()
        self.max_lim.setMinimumWidth(5)

    def clear_layout(self):
        if self.layout() is not None:
            old = self.layout()
            QWidget().setLayout(old)

    def arrange_horizontal(self, apply=True):
        """Simple row: label, value, checkbox"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.var)
        layout.addWidget(self.check)

        if apply:
            self.clear_layout()
            self.setLayout(layout)
        return layout

    def axisText(self):
        return self.var.text()

    def axisRawText(self):
        return self.var.get_raw_text()

    def axisLims(self, existing: list | tuple = (), set_placeholder: bool = True) -> list:
        if len(existing) < 2:
            existing = [np.nan, np.nan]
        try:
            min_lim = float(self.min_lim.text())
        except (ValueError, TypeError):
            min_lim = existing[0]
            if set_placeholder and not np.isnan(existing[0]):
                self.min_lim.setPlaceholderText(f"{existing[0]:.1e}")
        try:
            max_lim = float(self.max_lim.text())
        except (ValueError, TypeError):
            max_lim = existing[1]
            if set_placeholder and not np.isnan(existing[1]):
                self.max_lim.setPlaceholderText(f"{existing[1]:.1e}")
        return [min_lim, max_lim]

    def set_value(self, text: str):
        self.var.setText(text)

    def is_log(self):
        return self.check.isChecked()

    def set_p_text(self, text: str = "", target: str = ""):
        if not text:
            return
        widget: Any = getattr(self, target or "var", None)
        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(text)
            widget.parse_input()

    def set_p_lims(self, text: list | tuple = ()):
        if not text:
            return
        self.min_lim.setPlaceholderText(f"{text[0]:.1e}")
        self.max_lim.setPlaceholderText(f"{text[1]:.1e}")

    def p_text(self, target: str = ""):
        widget: Any = getattr(self, target or "var", None)
        if isinstance(widget, QLineEdit):
            return widget.placeholderText()


class PlotControl(QWidget):
    """
    Composite widget that builds axis control rows dynamically
    based on subplot dimensions [n_y, n_x].
    """

    def __init__(self, dimensions: tuple[int, int] = (1, 1), parent=None, name: str = ""):
        super().__init__(parent)
        n_y, n_x = dimensions
        self.ax: dict[str, AxisWidgets] = {}

        # layout = QGridLayout(self)

        self.name = name.strip().title() if name else "Plot"
        prefix = f"Enter {self.name}"

        # Create X-axis rows
        x_names = make_ax_labels("x", n_x, prefix=prefix)
        y_names = make_ax_labels("y", n_y, prefix=prefix)
        for nm, tip in x_names:
            row = AxisWidgets(nm)
            row.var.setToolTip(f"{tip} variable")
            self.ax[nm] = row

        # Create Y-axis rows
        for nm, tip in y_names:
            row = AxisWidgets(nm)
            row.var.setToolTip(f"{tip} variable")
            self.ax[nm] = row

        self.make_grid_stack(apply=True)

    def clear_layout(self):
        if self.layout() is not None:
            old = self.layout()
            QWidget().setLayout(old)

    def make_grid(self, apply=True):
        """Simple row: label, value, checkbox"""
        layout = QGridLayout()
        for i, row in enumerate(self.ax.values()):
            layout.addWidget(row.label, i, 0)
            layout.addWidget(row.var, i, 1)
            layout.addWidget(row.check, i, 2)

        if apply:
            self.clear_layout()
            self.setLayout(layout)
        return layout

    def make_stack(self, apply=True):
        """Simple column: label, value, checkbox"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # --- X axis ---
        x_sel = AlignComboBox()
        x_stack = QStackedLayout()
        x_stack.setContentsMargins(0, 0, 0, 0)
        y_sel = AlignComboBox()
        y_stack = QStackedLayout()
        y_stack.setContentsMargins(0, 0, 0, 0)
        for key, widget in self.ax.items():
            page = QWidget()
            page_layout = QGridLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.addWidget(widget.min_lim, 0, 0)
            page_layout.addWidget(widget.max_lim, 0, 1)
            if "x" in key.lower():
                x_sel.addItem(widget.axis_name)
                x_stack.addWidget(page)
            else:
                y_sel.addItem(widget.axis_name)
                y_stack.addWidget(page)

        x_sel.setCurrentIndex(0)
        x_stack.setCurrentIndex(0)
        y_sel.setCurrentIndex(0)
        y_stack.setCurrentIndex(0)

        x_sel.currentIndexChanged.connect(x_stack.setCurrentIndex)
        y_sel.currentIndexChanged.connect(y_stack.setCurrentIndex)

        # --- Assemble row ---
        layout.addWidget(x_sel)
        layout.addLayout(x_stack)
        layout.addWidget(y_sel)
        layout.addLayout(y_stack)

        if apply:
            self.clear_layout()
            self.setLayout(layout)
        return layout

    def make_grid_stack(self, apply=True):
        """Grid of label, value, checkbox + min/max edits."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.make_grid(apply=False))
        layout.addLayout(self.make_stack(apply=False))

        if apply:
            self.clear_layout()
            self.setLayout(layout)
        return layout

    # Convenience API
    def parse_axis(self) -> None:
        for row in self.ax.values():
            row.var.parse_input()

    def get_text(self, axis_name: str | list[str] = "") -> list[str]:
        if not axis_name:
            axis_name = list(self.ax.keys())
        if isinstance(axis_name, list):
            return [self.ax[name].axisText() for name in axis_name if name in self.ax]
        return [self.ax[axis_name].axisText()] if axis_name in self.ax else [""]

    def get_raw_text(self, axis_name: str | list[str] = "") -> list[str]:
        if not axis_name:
            axis_name = list(self.ax.keys())
        if isinstance(axis_name, list):
            return [self.ax[name].axisRawText() for name in axis_name if name in self.ax]
        return [self.ax[axis_name].axisRawText()] if axis_name in self.ax else [""]

    def set_axis_value(self, axis_name: str, value: str):
        self.ax[axis_name].set_value(value)

    def is_log(self, axis_name: str | list[str] = "") -> list[bool]:
        if not axis_name:
            axis_name = list(self.ax.keys())
        if isinstance(axis_name, list):
            return [self.ax[name].is_log() for name in axis_name if name in self.ax]
        return [self.ax[axis_name].is_log()] if axis_name in self.ax else [False]

    def set_p_texts(self, texts: list | tuple = (), target: str = ""):
        if len(texts) != len(self.ax):
            return
        for row, text in zip(self.ax.values(), texts):
            row.set_p_text(text, target=target)

    def p_text(self, target: str = ""):
        widget: Any = getattr(self, target or "var", None)
        if isinstance(widget, QLineEdit):
            return widget.placeholderText()

    def set_p_lims(self, texts: list | tuple = (), target: str = ""):
        if len(texts) != len(self.ax):
            return
        for row, text in zip(self.ax.values(), texts):
            row.set_p_text(text, target=target)


class PlotControlPanel(QWidget):
    """
    A panel containing Nyquist and Bode plot controls.
    Can be used as a standalone widget or embedded in other layouts.
    """

    def __init__(self, parent=None, plot_var=None, is_log=None):
        """
        Initialize the plot control panel.

        Args:
            parent (QWidget): Parent widget
            plot_var (dict): Dictionary of plot variable names
            is_log (dict): Dictionary of log scale flags
        """
        super().__init__(parent)
        self._create_widgets(plot_var, is_log)
        self._create_layout(apply=True)

    def _create_widgets(self, plot_var: dict | None = None, is_log: dict | None = None):
        """Create all the widgets for the panel."""
        # Create Nyquist plot control
        self.ny = PlotControl((1, 1), parent=self, name="Nyquist")
        self.bd = PlotControl((2, 1), parent=self, name="Bode")

        self.ny.set_p_texts(("Z'", "-Z''"), target="var")
        self.bd.set_p_texts(("Frequency", "Z'", "Z''"), target="var")

        if plot_var is not None and plot_var:
            self.ny.ax["x"].var.setText(plot_var.get("ny_x", "Z'"))
            self.ny.ax["y"].var.setText(plot_var.get("ny_y", "-Z''"))
            self.bd.ax["x"].var.setText(plot_var.get("bd_x", "Frequency"))
            self.bd.ax["y1"].var.setText(plot_var.get("bd_y1", "Z'"))
            self.bd.ax["y2"].var.setText(plot_var.get("bd_y2", "Z''"))
        if is_log is not None and is_log:
            self.ny.ax["x"].check.setChecked(is_log.get("ny_x", False))
            self.ny.ax["y"].check.setChecked(is_log.get("ny_y", False))
            self.bd.ax["x"].check.setChecked(is_log.get("bd_x", True))
            self.bd.ax["y1"].check.setChecked(is_log.get("bd_y1", False))
            self.bd.ax["y2"].check.setChecked(is_log.get("bd_y2", False))

        self.ny.ax["x"].min_lim.setText("0.0")
        self.ny.ax["y"].min_lim.setText("0.0")

        self.ny_band_check = QCheckBox("Nyq. band", self)
        self.bd_band_check = QCheckBox("Bode band", self)

    def _create_layout(self, apply=True):
        """
        Create the layout for the panel.

        Args:
            apply (bool): Whether to set the layout to this widget

        Returns:
            QLayout: The created layout
        """
        layout = QVBoxLayout()

        # Create labels with formatting
        nyq_label = QLabel("Nyquist")
        font = nyq_label.font()
        font.setPointSize(10)
        font.setBold(True)
        nyq_label.setFont(font)

        bode_label = QLabel("Bode")
        font = bode_label.font()
        font.setPointSize(10)
        font.setBold(True)
        bode_label.setFont(font)

        check_layout = QHBoxLayout()
        check_layout.addWidget(self.ny_band_check)
        check_layout.addWidget(self.bd_band_check)

        # Add widgets to layout
        layout.addWidget(nyq_label, alignment=Qt.AlignLeft)
        layout.addWidget(self.ny)
        layout.addWidget(bode_label, alignment=Qt.AlignLeft)
        layout.addWidget(self.bd)
        layout.addLayout(check_layout)

        if apply:
            self.clear_layout()
            self.setLayout(layout)

        return layout

    def clear_layout(self):
        """Clear the current layout."""
        if self.layout() is not None:
            old = self.layout()
            QWidget().setLayout(old)

    def parse_axes(self) -> None:
        """Parse all axis inputs."""
        self.ny.parse_axis()
        self.bd.parse_axis()

    def create_standalone_window(self, title="Plot Controls"):
        """
        Create a standalone window containing this panel.

        Args:
            title (str): Window title

        Returns:
            QWidget: Window containing the plot controls
        """
        window = QWidget(self.parent())
        window.setWindowTitle(title)
        layout = QVBoxLayout(window)
        layout.addWidget(self)
        window.setLayout(layout)
        return window
