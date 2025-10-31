# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from typing import Generic, TypeVar
from collections.abc import Callable

from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QEvent, QObject
from PyQt5.QtWidgets import (
    QFrame,
    QAction,
)

T = TypeVar("T")


# Create separators
def create_separator(frame=None, horizontal=True):
    separator = QFrame(frame)
    if horizontal:
        separator.setFrameShape(QFrame.HLine)
    else:
        separator.setFrameShape(QFrame.VLine)
    separator.setFrameShadow(QFrame.Sunken)
    return separator


def promote_from_children(cls):
    """
    Class decorator: looks for __promotions__ dict on the class and
    wires up forwarding attributes after __init__.

    Supports:
      - Methods / slots (callables)
      - Signals (Qt signal objects with .connect)
      - Properties (detected on the child's class)
    """
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        promotions = getattr(self, "__promotions__", {})

        for child_name, names in promotions.items():
            child = getattr(self, child_name)
            child_cls = type(child)

            for name in names:
                if hasattr(self, name):
                    continue  # don't overwrite existing

                # --- Case 1: Property on the child's class ---
                attr = getattr(child_cls, name, None)
                if isinstance(attr, property):
                    # Wrap as a delegating property
                    prop = property(
                        fget=lambda self, n=name, c=child: getattr(getattr(self, child_name), n),
                        fset=(
                            (
                                lambda self, v, n=name, c=child: setattr(
                                    getattr(self, child_name), n, v
                                )
                            )
                            if attr.fset
                            else None
                        ),
                        fdel=(
                            (lambda self, n=name, c=child: delattr(getattr(self, child_name), n))
                            if attr.fdel
                            else None
                        ),
                        doc=attr.__doc__,
                    )
                    setattr(cls, name, prop)
                    continue

                # --- Case 2: Signal (has .connect) ---
                member = getattr(child, name, None)
                if hasattr(member, "connect"):
                    # Expose as a property returning the signal
                    setattr(
                        cls,
                        name,
                        property(
                            lambda self, m=name, cn=child_name: getattr(getattr(self, cn), m)
                        ),
                    )
                    continue

                # --- Case 3: Callable (method/slot) ---
                if callable(member):
                    setattr(cls, name, member)
                    continue

                # --- Case 4: Fallback (plain attribute) ---
                setattr(cls, name, member)

    cls.__init__ = __init__
    return cls


class SelectionHelper(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.installEventFilter(self)

    def eventFilter(self, source, event):
        """
        Event filter to handle key press events for deleting items.
        """
        if source is self.parent and (
            event.type()
            in [QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick]
            and event.button() == Qt.LeftButton
        ):
            text = self.parent.text()
            sel_start = self.parent.selectionStart()
            sel_len = self.parent.selectionLength()
            if not self._eval_selection(text, sel_start, sel_len, event.pos()):
                # Not a special case -> return to default handler
                return False

            if event.type() == event.MouseButtonPress:
                self.parent.mousePressEvent(event)
            elif event.type() == event.MouseButtonRelease:
                self.parent.mouseReleaseEvent(event)
            elif event.type() == event.MouseButtonDblClick:
                # Expand selection ourselves; skip native double-click handler
                sel_start, sel_len = self.expand_selection(text, sel_start, sel_len)

            self.parent.setSelection(sel_start, sel_len)

            return True

        return False

    def _eval_selection(self, text, sel_start, sel_len, pos):
        """
        Evaluate the current text selection. It should match the following requirements:
            A) Current selection is somewhere between none and all of the text (exclusive)
            B) the cursor is within the selection (inclusive)
        """
        sel_end = sel_start + sel_len
        cursor = self.parent.cursorPositionAt(pos)
        txt_len = len(text)
        return 0 < sel_len < txt_len and max(1, sel_start) <= cursor <= min(sel_end, txt_len - 1)

    def expand_selection(self, text, sel_start, sel_len):
        if len(text[sel_start:]) > sel_len:  # Expand right
            return sel_start, sel_len + 1
        elif sel_start > 0:  # Expand left
            return sel_start - 1, sel_len + 1
        return 0, len(text)


class HistoryManager(QObject, Generic[T]):
    """
    A generic undo/redo history manager for Qt widgets.

    This class maintains a bounded history of values retrieved from a widget
    and allows stepping backward (undo) or forward (redo) through that history.
    It is agnostic to the type of value being tracked: you provide callables
    for retrieving the current state and restoring a prior state.

    Typical usage:
        history = HistoryManager(widget, widget.text, widget.setText)
        widget.textChanged.connect(history._update_history)

    Compatible widget types and common getter/setter pairs include:

    - QLineEdit, QPlainTextEdit, QTextEdit, *QLabel
        get: widget.text
        set: widget.setText

    - QSpinBox, QDoubleSpinBox, QSlider, QDial
        get: widget.value
        set: widget.setValue

    - QComboBox
        get: widget.currentText   (or currentIndex)
        set: widget.setCurrentText (or setCurrentIndex)

    - QCheckBox, QRadioButton, QAction (checkable)
        get: widget.isChecked
        set: widget.setChecked

    - QSplitter
        get: widget.sizes
        set: widget.setSizes

    Any widget exposing a stable getter and setter for its state can be used.
    * Applicable only if user-editable text interaction is enabled.

    Args:
        parent (QObject): The widget whose state is being tracked.
        get_method (callable): A zero-argument function for returning the current state.
        set_method (callable): A one-argument function for restoring a prior state.
        max_length (int): Maximum number of history entries to retain (default 30).

    Notes:
        - The manager installs itself as an event filter on the parent to
          intercept standard Undo/Redo key sequences (Ctrl+Z / Ctrl+Shift+Z).
        - To record changes, connect the widget’s change signal (e.g. textChanged,
          valueChanged, toggled) to `history.update_history`.
        - History values must be comparable (`!=` must work) and restorable
          via the provided setter.
    """

    def __init__(
        self, parent, get_method: Callable[[], T], set_method: Callable[[T], None], **kwargs
    ):
        if not callable(get_method) or not callable(set_method):
            raise ValueError("get_method and set_method must be callable functions.")

        super().__init__(parent)
        self.parent = parent

        if kwargs.get("install_event_filter", True):
            self.parent.installEventFilter(self)

        # Store the getter and setter functions
        self._get = get_method
        self._set = set_method

        # Initialize history and history components
        self._history = [self._get()]
        self._index = 0
        self._hlength = int(kwargs.get("max_len", 100)) - 2

    @property
    def history(self) -> list[T]:
        """Get the current history list."""
        return self._history.copy()

    @property
    def current(self) -> T:
        """Get the current value in history."""
        return self._history[self._index]

    def eventFilter(self, source, event):
        if source is self.parent and event.type() == QEvent.KeyPress:
            if event.matches(QKeySequence.Undo):
                self.undo()
                event.accept()
                return True
            elif event.matches(QKeySequence.Redo):
                self.redo()
                event.accept()
                return True
        return False  # let others handle

    def update_history(self, value: T) -> None:
        if value != self.current:
            self._history = self._history[max(0, self._index - self._hlength) : self._index + 1]
            self._history.append(value)
            self._index = len(self._history) - 1

    def undo(self) -> None:
        current = self._get()
        if current != self.current:
            # Restore most recent valid state
            self._set(self.current)
        elif self.isUndoAvailable():
            # Restore prior state
            self._index -= 1
            self._set(self.current)

    def redo(self) -> None:
        if self.isRedoAvailable():
            self._index += 1
            self._set(self.current)

    def isUndoAvailable(self) -> bool:
        return self._index > 0

    def isRedoAvailable(self) -> bool:
        return self._index + 1 < len(self._history)

    def create_or_patch_menu(self, menu, insert=False):
        """
        Ensure undo/redo actions in the given QMenu are replaced or added.

        - If existing undo/redo actions are found, they are always replaced.
        - If not found, new actions are added:
            * insert=True  → insert at the start of the menu
            * insert=False → append at the end (default)

        Args:
            menu (QMenu): The menu to modify.
            insert (bool): Controls where to add if actions don't exist.
        """
        actions = list(menu.actions())

        # Build fresh actions
        undo_action = QAction("&Undo", self.parent)
        undo_action.triggered.connect(self.undo)
        undo_action.setEnabled(self.isUndoAvailable())

        redo_action = QAction("&Redo", self.parent)
        redo_action.triggered.connect(self.redo)
        redo_action.setEnabled(self.isRedoAvailable())

        # Track whether we found existing ones
        found_undo = found_redo = False

        for action in actions:
            if action.text().startswith("&Undo"):
                undo_action.setText(action.text())
                menu.insertAction(action, undo_action)  # replace in place
                menu.removeAction(action)
                # menu.insertAction(actions[i + 1] if i + 1 < len(actions) else None, undo_action)
                found_undo = True
            elif action.text().startswith("&Redo"):
                redo_action.setText(action.text())
                menu.insertAction(action, redo_action)  # replace in place
                menu.removeAction(action)
                # menu.insertAction(actions[i + 1] if i + 1 < len(actions) else None, redo_action)
                found_redo = True

            if found_undo and found_redo:
                break

        insert = bool(insert and actions)
        # If not found, add according to insert flag
        if not found_undo:
            menu.insertAction(actions[0], undo_action) if insert else menu.addAction(undo_action)

        if not found_redo:
            menu.insertAction(actions[0], redo_action) if insert else menu.addAction(redo_action)

        return menu


class TextHistoryManager(HistoryManager[str]):
    """
    History manager for a text-based widget.

    Args:
        parent (QObject): The widget to manage.
        get_method (callable or None): Function to get current text.
            Defaults to parent.text.
        set_method (callable or None): Function to set text.
            Defaults to parent.setText.
    """

    def __init__(self, parent, get_method=None, set_method=None, **kwargs):

        get_method = get_method if callable(get_method) else parent.text
        set_method = set_method if callable(set_method) else parent.setText

        super().__init__(parent, get_method, set_method, **kwargs)
