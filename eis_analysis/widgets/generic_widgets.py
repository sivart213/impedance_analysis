# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QDialog,
    QSlider,
    QWidget,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QScrollBar,
    QFormLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QDoubleSpinBox,
    QDialogButtonBox,
)

from ..string_ops import MathEvaluator, format_number

T = TypeVar("T", bound=object, default=str)
C = TypeVar("C", bound=object, default=str)

WIDGETS = {
    "QLineEdit": QLineEdit,
    "QSpinBox": QSpinBox,
    "QDoubleSpinBox": QDoubleSpinBox,
    "QCheckBox": QCheckBox,
    "QComboBox": QComboBox,
    "QLabel": QLabel,
}


def get_widget_value(widget: QWidget) -> str | int | float | bool | None:
    """Universal-ish getter for common Qt widgets."""
    if isinstance(widget, QLineEdit):
        return widget.text()
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    elif isinstance(widget, QCheckBox):
        return widget.isChecked()
    elif isinstance(widget, QComboBox):
        return widget.currentText()
    elif hasattr(widget, "text"):
        return widget.text()
    elif hasattr(widget, "value"):
        return widget.value()
    return None


def set_widget_value(widget: QWidget, value: Any) -> QWidget:
    """Universal-ish setter for common Qt widgets."""
    if isinstance(widget, QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox) else int(value))
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QComboBox):
        if isinstance(value, (list, tuple)):
            widget.clear()
            widget.addItems([str(v) for v in value])
            if value:  # select first item
                widget.setCurrentIndex(0)
        else:
            text = str(value)
            idx = widget.findText(text)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            else:
                widget.addItem(text)
                widget.setCurrentIndex(widget.count() - 1)
    elif hasattr(widget, "setText"):
        widget.setText(str(value))
    elif hasattr(widget, "setValue"):
        widget.setValue(value)
    return widget
    # else: silently ignore


def set_widget_limits(widget: QWidget, *limits: Any) -> QWidget:
    """
    Apply limits/choices/range to a widget.

    For numeric widgets (QSpinBox, QDoubleSpinBox, QSlider, QScrollBar):
        - One value: sets maximum only
        - Two+ values: (min, max, [step?]), with None as a placeholder to skip
        - A single sequence [min, max, step?] is also accepted

    For QComboBox:
        - Sequence of values: replaces items with given list

    For other widgets: silently ignored.
    """
    if not limits:
        return widget
    if len(limits) == 1 and isinstance(limits[0], (list, tuple)):
        limits = tuple(limits[0])

    if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider, QScrollBar)):
        if len(limits) == 1:  # Not None is implied
            widget.setMaximum(limits[0])
        elif limits[0] is not None and limits[1] is not None:
            widget.setRange(limits[0], limits[1])
        elif limits[0] is not None:
            widget.setMinimum(limits[0])
        elif limits[1] is not None:
            widget.setMaximum(limits[1])

        if len(limits) >= 3 and limits[2] is not None:
            widget.setSingleStep(limits[2])

    elif isinstance(widget, QComboBox):
        widget.clear()
        widget.addItems([str(v) for v in limits])
    return widget


class WidgetFactory:
    """
    Declarative specification for a form field.
    Internally keeps only label, widget, and an options dict.
    """

    def __init__(
        self,
        label: str,
        widget: str | QWidget | type[QWidget] = QLineEdit,
        **kwargs: Any,
    ) -> None:
        self.label: str = label
        self.widget: QWidget = QLineEdit()
        self.options: dict[str, Any] = {}

        # Instantiate widget if a class was passed
        if isinstance(widget, QWidget):
            self.widget = widget
        elif isinstance(widget, type) and issubclass(widget, QWidget):
            self.widget = widget()
        elif isinstance(widget, str):
            self.widget = WIDGETS.get(widget, QLineEdit)()

        self.apply_options(**kwargs)

    def apply_options(self, **kwargs) -> None:
        """Apply options to the widget."""
        self.options |= kwargs

        if self.options.get("limits") is not None:
            set_widget_limits(self.widget, self.options["limits"])
        if self.options.get("default") is not None:
            set_widget_value(self.widget, self.options["default"])
        if tooltip := self.options.get("tooltip"):
            self.widget.setToolTip(tooltip)
        if self.options.get("enabled") is not None:
            self.widget.setEnabled(self.options["enabled"])
        if self.options.get("visible") is not None:
            self.widget.setVisible(self.options["visible"])
        if style := self.options.get("style"):
            self.widget.setStyleSheet(style)

    def rebuild(self, *args, **kwargs) -> None:
        """Reconstruct the widget to be a new instance of the same type."""
        try:
            self.widget = type(self.widget)(*args, **kwargs)  # Re-instantiate
            self.apply_options()
        except Exception as exc:
            raise ValueError(f"Failed to rebuild widget with given args/kwargs.\n{exc}") from exc

    @classmethod
    def parse(cls, label: str, spec: Any, **defaults: Any) -> "WidgetFactory":
        """
        Normalize a field spec into a WidgetFactory.
        """
        if isinstance(spec, QWidget):
            merged = defaults | {"widget": spec}
        elif isinstance(spec, type) and issubclass(spec, QWidget):
            merged = defaults | {"widget": spec()}
        elif isinstance(spec, str) and spec in WIDGETS:
            merged = defaults | {"widget": WIDGETS[spec]()}
        elif isinstance(spec, dict):
            merged = defaults | spec
        else:
            merged = defaults | {"default": spec}

        return cls(label, **merged)

    @classmethod
    def parse_dict(cls, input: dict[str, Any], **defaults: Any) -> dict[str, "WidgetFactory"]:
        """
        Normalize a dict of field specs into WidgetFactory objects.
        """
        return {label: cls.parse(label, spec, **defaults) for label, spec in input.items()}


class CollapsibleSection(QWidget):
    """A simple collapsible section widget with a toggle button."""

    def __init__(self, title: str, content_widget: QWidget):
        super().__init__()
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        # self.toggle_button.setStyleSheet("text-align: left;")
        self.toggle_button.setStyleSheet("font-weight: bold; font-size: 10pt;")

        self.frame = QFrame(content_widget)
        self.frame.setVisible(True)
        self.frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.frame)

        self.toggle_button.toggled.connect(self.frame.setVisible)


class tLabel(QLabel):
    """
    Composite widget: a bolded title label + a selectable/copyable text label.
    Layout can be horizontal (row) or vertical (column).
    Promotes text-related API from the value label.
    """

    def __init__(self, title: str = "", size: int | float = 0.0, parent=None):
        super().__init__(title, parent)
        # Title label (bold)
        font = self.font()
        if size > 0:
            font.setPointSize(int(size))
        font.setBold(True)
        self.setFont(font)


class fLabel(QLabel):
    """
    A LabeledText specialized for displaying floats.
    Uses an external `format_number` function to render values
    with configurable precision and exponent thresholds.
    """

    def __init__(
        self,
        value: str | float | int = "N/A",
        precision: int = 3,
        upper_exponent: int = 1,
        lower_exponent: int = -2,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)

        # Store formatting parameters
        self.precision = precision
        self.upper_exponent = upper_exponent
        self.lower_exponent = lower_exponent

        # Initialize display
        self.setText(value)

    def setText(self, value: str | float | int):
        # Intercept string input, try to parse as float
        # print("intercepted")
        try:
            f = float(value)
            formatted = format_number(
                f,
                precision=self.precision,
                upper_exponent=self.upper_exponent,
                lower_exponent=self.lower_exponent,
            )
            super().setText(formatted)
        except (ValueError, TypeError):
            super().setText("N/A")

    def text(self) -> float:
        """Return the current displayed value as a float (best effort)."""
        # raw = super().text()
        raw = super().text()
        try:
            return float(raw)
        except ValueError:
            return 0.0


# %% ---- Aligned ComboBox with Value Mapping ----
class AlignComboBox(QComboBox):
    """Class to create a combobox with custom alignment and value mapping."""

    def __init__(
        self, parent=None, items=None, values=None, init=None, width=None, alignment=None
    ):
        super(AlignComboBox, self).__init__(parent)
        self._value_map = {}  # Internal dictionary to map text to values
        if items is not None:
            self.addItems(items)
        if values is not None:
            self.setValues(values)
        if isinstance(init, (int, float)):
            self.setCurrentIndex(init)
        elif isinstance(init, str):
            self.setCurrentText(init)
        if width is not None:
            self.setFixedWidth(width)
        if alignment is not None:
            self.setTextAlignment(alignment)

    def setTextAlignment(self, alignment):
        """Set the alignment of the combobox."""
        self.setEditable(True)
        self.lineEdit().setAlignment(alignment)
        self.lineEdit().setReadOnly(True)

    def setValues(self, values):
        """Set the mapping of items to values."""
        if len(values) != self.count():
            raise ValueError(
                "The number of values must match the number of items in the combobox."
            )
        self._value_map = dict(zip([self.itemText(i) for i in range(self.count())], values))

    def addItem(self, text, value=None):
        """Add an item with an optional value."""
        super().addItem(text)
        if value is not None:
            self._value_map[text] = value

    def addItems(self, texts, values=None):
        """Add multiple items with optional values."""
        if values is None and isinstance(texts, dict):
            super().addItems(texts.keys())
            self._value_map.update(texts)
        else:
            super().addItems(texts)
            if values is not None:
                if len(texts) != len(values):
                    raise ValueError("The number of values must match the number of items.")
                self._value_map.update(dict(zip(texts, values)))

    def connectToggle(self, checkbox, state):
        """
        Connect the currentIndexChanged signal to toggle a checkbox.

        Args:
            checkbox (QCheckBox): The checkbox to toggle.
            state (bool): The state to set (True or False).
        """
        self.currentIndexChanged.connect(lambda: checkbox.setChecked(state))

    @property
    def currentValue(self):
        """Get the value corresponding to the current text."""
        return self._value_map.get(self.currentText(), None)

    @currentValue.setter
    def currentValue(self, value):
        """Set the current index based on the value."""
        for text, val in self._value_map.items():
            if val == value:
                self.setCurrentText(text)
                return
        raise ValueError(f"Value '{value}' not found in the value map.")


# %% ---- Numerical line edits ----
class SetIncLineEdit(QLineEdit):
    """
    A subclass of SLineEdit that provides a method to retrieve the float value
    of the text and validates the input to ensure it is a valid float or integer.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the IncLineEdit.

        Args:
            parent (QWidget): The parent widget.
        """
        self.lt_wigets = kwargs.pop("lt_wigets", [])
        self.gt_wigets = kwargs.pop("gt_wigets", [])
        # self.e_wigets = kwargs.pop("gt_wigets", [])

        super().__init__(*args, **kwargs)
        self._last_valid_number = 0.0  # Store the last valid number
        self.editingFinished.connect(self._check_str)

    def incr(self):
        """
        Retrieve the float value of the current text.

        Returns:
            float: The float value of the text.
        """
        return float(self.text())

    def _check_str(self):
        """
        Validate the current text to ensure it is a valid float or integer.
        If invalid, revert to the last valid number.
        """
        text = self.text().replace(" ", "")
        try:
            # Try to convert the text to a float
            self._last_valid_number = float(text)
            for widget in self.gt_wigets:
                # Widgets that this one should be greater than or equal to
                if isinstance(widget, SetIncLineEdit) and self._last_valid_number < widget.incr():
                    widget.setText(str(self._last_valid_number))
            for widget in self.lt_wigets:
                # Widgets that this one should be less than or equal to
                if isinstance(widget, SetIncLineEdit) and self._last_valid_number > widget.incr():
                    widget.setText(str(self._last_valid_number))
        except ValueError:
            # Revert to the last valid number if conversion fails
            self.setText(str(self._last_valid_number))


class TLineEdit(QLineEdit):
    """
    A QLineEdit subclass that debounces the textChanged signal.
    External users can still connect their functions to the textChanged signal.
    """

    textDelayedChanged = pyqtSignal(str)  # New signal for debounced text changes

    def __init__(self, parent=None, debounce_interval=120, **_):
        """
        Initialize the DebouncedLineEdit.

        Args:
            parent (QWidget): The parent widget.
            debounce_interval (int): The debounce interval in milliseconds.
        """
        super().__init__(parent)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._pending_request = False
        self._last_text = ""

        self.debounce_interval = debounce_interval

        # Connect the internal textChanged signal to the handler textEdited
        super().textChanged.connect(self._on_text_changed)

        # Connect the timer's timeout signal to the debounce handler
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

    def _on_text_changed(self, text):
        """
        Internal handler for the textChanged signal.
        Implements the debounce logic.
        """
        self._last_text = text
        if not self._debounce_timer.isActive():
            # Timer is not running, start it
            self._last_text = ""
            self.textDelayedChanged.emit(text)
            self._debounce_timer.start(self.debounce_interval)
        else:
            # Timer is running, set the pending request flag
            self._last_text = text
            self._pending_request = True

    def _on_debounce_timeout(self):
        """
        Handler for the debounce timer timeout.
        Emits the textDelayedChanged signal if there was a pending request.
        """
        # if self._pending_request or self._last_text:
        if self._pending_request:
            self.textDelayedChanged.emit(self._last_text)  # Emit the debounced signal
        # Reset the pending request flag
        self._pending_request = False

    def _on_forced_timeout(self):
        """
        Handler for the debounce timer timeout.
        Emits the textDelayedChanged signal if there was a pending request.
        """
        if self._debounce_timer.isActive():
            self._debounce_timer.stop()
        # if self._pending_request or self._last_text:
        if self._pending_request:
            self.textDelayedChanged.emit(self._last_text)  # Emit the debounced signal
        else:
            self.textDelayedChanged.emit(self.text())  # Emit the current text
        # Reset the pending request flag
        self._pending_request = False


class IncLineEdit(TLineEdit):
    """Class to create a custom line edit widget."""

    def __init__(self, *args, **kwargs):
        self.interval = kwargs.pop("interval", 1)
        self.fine_interval = kwargs.pop("fine_interval", self.interval / 10)
        self.precision = kwargs.pop("precision", 5)
        self.upper_exponent = kwargs.pop("upper_exponent", 1)
        self.lower_exponent = kwargs.pop("lower_exponent", -2)
        self.bounds = kwargs.pop("bounds", (-np.inf, np.inf))
        kwargs.setdefault("debounce_interval", 2000)
        super().__init__(*args, **kwargs)
        self.setText("0")
        self._eval = MathEvaluator()

    def setTextValue(self, value):
        """Set the text value with formatting."""
        try:
            val = np.clip(self._eval.parse(value), *self.bounds)
            self.setText(
                format_number(
                    val,
                    self.precision,
                    self.upper_exponent,
                    self.lower_exponent,
                )
            )
        except ValueError:
            pass

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Up:
            self.increment_value(event.modifiers() & Qt.ShiftModifier)
        elif event.key() == Qt.Key_Down:
            self.decrement_value(event.modifiers() & Qt.ShiftModifier)
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.setTextValue(self.text())
            self._on_forced_timeout()
            event.accept()
            # return True
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel events."""
        if self.hasFocus():  # Only respond if the widget is focused
            if event.angleDelta().y() > 0:  # Scroll up
                self.increment_value(event.modifiers() & Qt.ShiftModifier)
            elif event.angleDelta().y() < 0:  # Scroll down
                self.decrement_value(event.modifiers() & Qt.ShiftModifier)
        else:
            super().wheelEvent(event)

    def increment_value(self, fine=False):
        """Increment the value of the line edit."""
        try:
            shift = self.fine_interval if fine else self.interval

            texts = f"{self._eval.parse(self.text()):e}".split("e")
            texts[0] = str(float(texts[0]) + shift)

            self.setTextValue(float("e".join(texts)))

        except ValueError:
            pass

    def decrement_value(self, fine=False):
        """Decrement the value of the line edit."""
        try:
            shift = self.fine_interval if fine else self.interval

            texts = f"{self._eval.parse(self.text()):e}".split("e")
            if len(texts) == 2 and float(texts[0]) == shift:
                texts[0] = str(float(texts[0]) - shift / 10)
            else:
                texts[0] = str(float(texts[0]) - shift)

            self.setTextValue(float("e".join(texts)))

        except ValueError:
            pass


class BaseDialog(QDialog, Generic[T, C]):
    """
    Abstract base class for dialogs with a standard frame:
    - window title
    - optional prompt label
    - OK/Cancel button box

    Subclasses must implement:
      * `_insert_content` — to populate the dialog body with widgets
      * `values` — to extract the dialog's result value(s)

    Type parameters
    ---------------
    T : The return type of `values()` (e.g. str, dict[str, str])
    C : The type of the `content` argument used to initialize the dialog
    """

    def __init__(
        self, parent: Any = None, title: str = "", prompt: str = "", content: C = None, width=200
    ):
        super().__init__(parent)

        self.setWindowTitle(title or "Input")

        layout = QVBoxLayout(self)

        if prompt:
            label = QLabel(prompt)
            layout.addWidget(label)

        # Drop-in section for subclasses
        self._insert_content(layout, content)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setMinimumWidth(width)

    @abstractmethod
    def _insert_content(self, layout: QVBoxLayout, content: C) -> None:
        """
        Insert the main content widgets into the dialog.

        Parameters
        ----------
        layout : QVBoxLayout
            The parent layout provided by the base class. Subclasses
            should add their widgets or sub-layouts here.
        content : `C`
            Initialization data for the dialog body (type depends on subclass).
        """
        pass

    @abstractmethod
    def values(self) -> T:
        """
        Return the dialog's result value(s).

        Returns
        -------
        `T`
            The extracted result if the dialog was accepted.
            Subclasses define the exact type and semantics.
        """
        pass

    @classmethod
    def getResult(
        cls, parent: Any = None, title: str = "", prompt: str = "", content: C = None, width=200
    ) -> tuple[T, bool]:
        """
        Convenience method to run the dialog and return its result.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget for the dialog.
        title : str, optional
            Window title (default "Input").
        prompt : str, optional
            Optional label text shown above the content area.
        content : C, optional
            Initialization data for the dialog body.
        width : int, optional
            Minimum width of the dialog.

        Returns
        -------
        tuple[`T`, bool]
            A tuple of (result, accepted):
              * result : `T` — the dialog's extracted value(s) from `values()`
              * accepted : bool — True if the user pressed OK, False otherwise
        """
        dlg = cls(parent, title, prompt, content, width)
        ok = dlg.exec() == QDialog.Accepted
        return dlg.values(), ok


class SimpleDialog(BaseDialog[str, str | None]):
    """
    A simple dialog with a single QLineEdit text entry.

    Inherits the standard frame from BaseDialog and implements:
      * `_insert_content` — inserts a single line edit
      * `values` — returns the entered text as a string
    """

    def _insert_content(self, layout: QVBoxLayout, content: str | None) -> None:
        """
        Insert a single QLineEdit into the dialog.

        Parameters
        ----------
        layout : QVBoxLayout
            The parent layout provided by the base class.
        content : str
            Initial text to populate the line edit.
        """
        self.entry = QLineEdit(self)
        if content is not None:
            self.entry.setText(str(content))
        layout.addWidget(self.entry)

    def values(self) -> str:
        """
        Return the entered text.

        Returns
        -------
        str
            The text from the line edit if the dialog was accepted,
            or an empty string if it was cancelled.
        """
        if self.result() != QDialog.Accepted:
            return ""
        return self.entry.text()


class FormDialog(BaseDialog[dict[str, Any], dict | None]):
    """
    A dialog presenting multiple labeled text fields in a QFormLayout.

    Inherits the standard frame from BaseDialog and implements:
    * `_insert_content` — inserts a form of QLineEdits
    * `values` — returns a dict mapping labels to entered text
    """

    def _insert_content(self, layout: QVBoxLayout, content: dict | None) -> None:
        """
        Insert a QFormLayout of labeled QLineEdits.

        Parameters
        ----------
        layout : QVBoxLayout
            The parent layout provided by the base class.
        content : dict[str, str] or None
            Mapping of label -> initial text. If None, a single
            "Input" field is created.
        """
        form_layout = QFormLayout()
        self._widgets = {}

        if content is None:
            content = {"Input": ""}

        for label, widg in WidgetFactory.parse_dict(content).items():
            self._widgets[label] = widg.widget
            form_layout.addRow(label + ":", widg.widget)

        layout.addLayout(form_layout)

    def values(self) -> dict[str, Any]:
        """
        Return the entered values from the form.

        Returns
        -------
        dict[str, str]
            A mapping of label -> text for each field if the dialog
            was accepted, or an empty dict if it was cancelled.
        """
        if self.result() != QDialog.Accepted:
            return {}
        return {label: get_widget_value(widget) for label, widget in self._widgets.items()}


# class SimpleDialog(QDialog):
#     """A simple dialog window with a single text input."""

#     def __init__(self, parent=None, title=None, prompt="", initialvalue="", width=200):
#         super().__init__(parent)

#         self.setWindowTitle(title or "Input")

#         # --- Setup the UI ---
#         layout = QVBoxLayout(self)

#         if prompt:
#             label = QLabel(prompt)
#             layout.addWidget(label)

#         self.entry = QLineEdit(self)
#         self.entry.setText(str(initialvalue))
#         layout.addWidget(self.entry)

#         buttons = QDialogButtonBox(
#             QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
#         )
#         buttons.accepted.connect(self.accept)
#         buttons.rejected.connect(self.reject)
#         layout.addWidget(buttons)

#         self.setMinimumWidth(width)

#         # self.setLayout(layout)

#     def values(self) -> str:
#         """
#         Return the entered text if the dialog was accepted, else None.
#         """
#         if self.result() != QDialog.Accepted:
#             return ""
#         return self.entry.text()

#     @staticmethod
#     def getText(
#         parent=None, title="Input", prompt="", initialvalue="", width=200
#     ) -> tuple[str, bool]:
#         """
#         Static convenience method to get a string from the user.

#         Returns:
#             (str, bool): The entered text and whether the dialog was accepted.
#         """
#         dlg = SimpleDialog(parent, title, prompt, initialvalue, width)
#         ok = dlg.exec() == QDialog.Accepted
#         return dlg.values(), ok


# class FormDialog(QDialog):
#     def __init__(self, parent=None, title="Form", fields: dict | None = None, width=200):
#         """
#         Generic form dialog.

#         Parameters:
#         ----------
#         parent : QWidget, optional
#             Parent widget.
#         title : str, optional
#             Dialog title.
#         fields : dict, optional
#             Dictionary of field labels and default values or widgets. If a value is a QWidget,
#             it will be used directly; otherwise, a QLineEdit will be created with the value as text.
#         width : int, optional
#             Minimum width of the dialog.
#         """
#         super().__init__(parent)

#         if fields is None:
#             fields = {"Input": ""}

#         self.setWindowTitle(title)

#         self._widgets = {}
#         # --- Setup the UI ---
#         layout = QFormLayout(self)

#         # Build form rows
#         for label, spec in WidgetFactory.parse_dict(fields).items():
#             self._widgets[label] = spec.widget
#             layout.addRow(label + ":", spec.widget)

#         # OK/Cancel buttons
#         buttons = QDialogButtonBox(
#             QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
#         )
#         buttons.accepted.connect(self.accept)
#         buttons.rejected.connect(self.reject)
#         layout.addRow(buttons)

#         layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
#         layout.setLabelAlignment(Qt.AlignRight)
#         self.setMinimumWidth(width)

#     def values(self) -> dict:
#         """
#         Return a dict of field values, or None if canceled.
#         """
#         if self.result() != QDialog.Accepted:
#             return {}

#         return {label: get_widget_value(widget) for label, widget in self._widgets.items()}

#     @staticmethod
#     def getValues(
#         parent=None, title="Input", fields: dict | None = None, width=200
#     ) -> tuple[dict, bool]:
#         """
#         Static convenience method to get a string from the user.

#         Returns:
#             (str, bool): The entered text and whether the dialog was accepted.
#         """
#         dlg = FormDialog(parent, title, fields, width)
#         ok = dlg.exec() == QDialog.Accepted
#         return dlg.values(), ok
