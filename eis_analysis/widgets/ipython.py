# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import io
import sys
import inspect
import contextlib

from PyQt5 import sip  # type: ignore
from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow,
    QVBoxLayout,
)
from qtconsole.client import QtKernelClient
from ipykernel.connect import get_connection_file
from qtconsole.manager import QtKernelManager
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib.guisupport import get_app_qt4
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from traitlets.config.configurable import MultipleInstanceError


@contextlib.contextmanager
def suppress_console():
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def in_process_console(console_class=RichJupyterWidget, **kwargs):
    """Create a console widget, connected to an in-process Kernel"""
    # from qtconsole.inprocess import QtInProcessKernelManager
    # print("Creating in-process console")
    with suppress_console():
        km = QtInProcessKernelManager()
        km.start_kernel()

        kernel = km.kernel
        kernel.gui = "qt"

        client = km.client()
        client.start_channels()

        control = console_class() if inspect.isclass(console_class) else console_class
        control.kernel_manager = km
        control.kernel_client = client
        control.shell = kernel.shell
        control.shell.user_ns.update(**kwargs)  # type: ignore
        control.internel_namespace.update(**kwargs)
    return control


def connected_console(console_class=RichJupyterWidget, **kwargs):
    """Create a console widget, connected to another kernel running in
    the current process
    """
    # print("Creating connected console")
    shell = get_ipython()
    if shell is None:
        raise RuntimeError("There is no IPython kernel in this process")

    client = QtKernelClient(connection_file=get_connection_file())
    client.load_connection_file()
    client.start_channels()

    control = console_class() if inspect.isclass(console_class) else console_class
    control.kernel_manager = None
    control.kernel_client = client
    control.shell = shell
    control.shell.kernel.start()  # type: ignore
    control.shell.user_ns.update(**kwargs)
    control.internel_namespace.update(**kwargs)
    return control


def kerneled_console(console_class=RichJupyterWidget, **kwargs):
    """Create a console widget, connected to another kernel running in
    the current process
    """
    # from qtconsole.manager import QtKernelManager
    # print("Creating kerneled console")

    km = QtKernelManager(kernel_name="python3")
    km.start_kernel()

    client = km.client()
    client.start_channels()

    control = console_class() if inspect.isclass(console_class) else console_class
    control.kernel_manager = km
    control.kernel_client = client

    control.push({**get_ipython().user_ns, **kwargs})  # type: ignore

    return control
    # if control.shell is not None:
    #     print("Warning: overwriting existing shell")
    #     control.shell.user_ns.update(**kwargs)
    #     control.internel_namespace.update(**kwargs)
    # else:
    #     control.push({**get_ipython().user_ns, **kwargs})  # type: ignore
    # control.shell = get_ipython()
    # get_ipython().user_ns.update(**kwargs)  # type: ignore
    # control.push({**get_ipython().user_ns, **kwargs})  # type: ignore
    # control.push(kwargs)
    # control.shell = get_ipython()
    # control.shell.user_ns.update(**kwargs)  # type: ignore
    # control.internel_namespace.update(**kwargs)


# def ipython_terminal(console_class=None, **kwargs):
#     """Return a qt widget which embed an IPython interpreter."""
#     if console_class is None:
#         console_class = Terminal
#     shell = get_ipython()
#     if shell is None:
#         try:
#             print("r1")
#             return in_process_console(console_class, **kwargs)
#         except MultipleInstanceError:
#             print("r2")
#             return kerneled_console(console_class, **kwargs)
#     elif getattr(get_app_qt4(), "_in_event_loop", False):  # is_event_loop_running_qt4() and
#         print("r3")
#         return connected_console(console_class, **kwargs)
#     print("r4")
#     return kerneled_console(console_class, **kwargs)
#     # return in_process_console(console_class, **kwargs)


def ipython_terminal(console_class=None, **kwargs):
    """Return a qt widget which embed an IPython interpreter."""
    if console_class is None:
        console_class = Terminal
    if get_ipython() is not None and getattr(get_app_qt4(), "_in_event_loop", False):
        return connected_console(console_class, **kwargs)
    try:
        return in_process_console(console_class, **kwargs)
    except MultipleInstanceError:
        return kerneled_console(console_class, **kwargs)


class Terminal(RichJupyterWidget):
    """A terminal widget that can be used to run IPython code."""

    def __init__(self, **kwargs):
        super(Terminal, self).__init__(**kwargs)

        self.setAcceptDrops(True)
        self.shell = None
        self.kernel_manager = None
        self.kernel_client = None
        self.top_widget = None
        self.internel_namespace = {}

    @property
    def namespace(self):
        """Return the namespace of the shell."""
        return self.shell.user_ns if self.shell is not None else self.internel_namespace

    def push(self, kwargs):
        """Update the namespace of the shell with the given variables."""
        self.internel_namespace.update(kwargs)
        # print(f"Pushing to terminal: {list(kwargs.keys())}")
        if self.shell is not None:
            # print("Using local shell")
            self.shell.push(kwargs)
        elif self.kernel_client is not None:
            # print("Using kernel client")
            for key, value in kwargs.items():
                try:
                    if str(key).startswith("_") or str(key) == "In":
                        pass
                    elif callable(value) or inspect.ismodule(value) or inspect.isclass(value):
                        mod = inspect.getmodule(value).__name__  # type: ignore
                        if mod != "__main__" and hasattr(value, "__name__"):
                            if mod != value.__name__ and "." not in value.__name__:
                                self.kernel_client.execute(
                                    f"from {mod} import {value.__name__} as {key}",
                                    silent=True,
                                    stop_on_error=False,
                                )
                            else:
                                self.kernel_client.execute(
                                    f"import {value.__name__} as {key}",
                                    silent=True,
                                    stop_on_error=False,
                                )
                    elif isinstance(value, str):
                        self.kernel_client.execute(
                            f"{key} = {repr(value)}",
                            silent=True,
                            stop_on_error=False,
                        )
                    elif isinstance(value, (int, float, complex, bool)):
                        self.kernel_client.execute(
                            f"{key} = {repr(value)}",
                            silent=True,
                            stop_on_error=False,
                        )
                    elif isinstance(value, (list, tuple)) and all(
                        isinstance(v, (int, float, complex, bool, str)) for v in value
                    ):
                        self.kernel_client.execute(
                            f"{key} = {repr(value)}",
                            silent=True,
                            stop_on_error=False,
                        )
                    elif isinstance(value, dict) and all(
                        isinstance(v, (int, float, complex, bool, str)) for v in value.values()
                    ):
                        self.kernel_client.execute(
                            f"{key} = {repr(value)}",
                            silent=True,
                            stop_on_error=False,
                        )

                except (
                    TypeError,
                    NameError,
                    SyntaxError,
                    AttributeError,
                ) as e:
                    print(f"Failed to update {key}: {e}")

    def _is_complete(self, source, interactive):
        """Check if the source code is complete and can be executed."""
        return True, True

    def shutdown_kernel(self):
        """Shutdown the kernel and close the terminal."""
        # print("Shutting down kernel...")
        if self.kernel_manager is not None:
            self.kernel_client.stop_channels()  # type: ignore
            self.kernel_manager.shutdown_kernel()

            self.kernel_client = None
            self.kernel_manager = None
            if (
                self.shell is not None
                and isinstance(self.shell, InteractiveShell)
                and self.shell.__class__.__module__.startswith("ipykernel")
            ):
                self.shell.reset()
                self.shell = None
                InteractiveShell.clear_instance()
            # if self.shell is not None:
            #     self.internel_namespace |= self.shell.user_ns
            #     self.shell = None
            # print('Kernel off')

    def showEvent(self, event):
        """Handle the open event to ensure the application opens properly"""
        # print("In Terminal Show")
        self.initialize_terminal()
        event.accept()

    def closeEvent(self, event):
        """Handle the close event to ensure the application closes properly"""
        # print("In Terminal close")
        self.shutdown_kernel()
        event.accept()
        self.deleteLater()  # Ensure the widget is properly deleted

    def initialize_terminal(self):
        """Initialize the terminal with the given variables."""
        if self.kernel_client is None:
            # print(f"Re-initializing in terminal: {sip.isdeleted(self)}")
            ipython_terminal(self, **self.internel_namespace)


class MainConsole(QMainWindow):
    """A window that contains a single Qt console."""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        # self.parent = parent
        self.console = Terminal()
        self.console.push(kwargs)
        # ipython_terminal(self.console, **kwargs)

        self.setWindowTitle("IPython Console")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        layout.addWidget(self.console)

        self._full_shutdown = True
        if parent is not None:
            # print("parent connected to destroyed")
            self._full_shutdown = False
            # parent.close_children.connect(self.set_shutdown_logic)

    # def set_shutdown_logic(self):
    #     self.allow_shutdown = True

    def full_shutdown(self):
        self._full_shutdown = True
        if self and not sip.isdeleted(self):
            self.close()

    def closeEvent(self, event):
        """Handle the close event to ensure the application closes properly"""
        # print("Closing console...")
        if self._full_shutdown:
            self.console.shutdown_kernel()
            self.deleteLater()  # Ensure the widget is properly deleted
        # self.console.shutdown_kernel()
        # self.deleteLater()  # Ensure the widget is properly deleted
        event.accept()
