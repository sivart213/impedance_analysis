# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import inspect
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from IPython import get_ipython

from qtconsole.rich_jupyter_widget import RichJupyterWidget

from qtconsole.inprocess import QtInProcessKernelManager
from ipykernel.connect import get_connection_file
from qtconsole.client import QtKernelClient
from qtconsole.manager import QtKernelManager

# import cloudpickle
from traitlets.config.configurable import MultipleInstanceError
from IPython.lib.guisupport import get_app_qt4


def in_process_console(console_class=RichJupyterWidget, **kwargs):
    """Create a console widget, connected to an in-process Kernel"""
    # from qtconsole.inprocess import QtInProcessKernelManager
    # print("Creating in-process console")
    km = QtInProcessKernelManager()
    km.start_kernel()

    kernel = km.kernel
    kernel.gui = "qt"

    client = km.client()
    client.start_channels()

    control = (
        console_class() if inspect.isclass(console_class) else console_class
    )
    control.kernel_manager = km
    control.kernel_client = client
    control.shell = kernel.shell
    control.shell.user_ns.update(**kwargs)
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

    control = (
        console_class() if inspect.isclass(console_class) else console_class
    )
    control.kernel_manager = None
    control.kernel_client = client
    control.shell = shell
    control.shell.kernel.start()
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

    control = (
        console_class() if inspect.isclass(console_class) else console_class
    )
    control.kernel_manager = km
    control.kernel_client = client
    control.push({**get_ipython().user_ns, **kwargs})

    return control


def ipython_terminal(console_class=None, **kwargs):
    """Return a qt widget which embed an IPython interpreter."""
    if console_class is None:
        console_class = Terminal
    shell = get_ipython()
    if shell is None:
        try:
            return in_process_console(console_class, **kwargs)
        except MultipleInstanceError:
            return kerneled_console(console_class, **kwargs)
    elif getattr(
        get_app_qt4(), "_in_event_loop", False
    ):  # is_event_loop_running_qt4() and
        return connected_console(console_class, **kwargs)
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
        return (
            self.shell.user_ns
            if self.shell is not None
            else self.internel_namespace
        )

    def push(self, kwargs):
        """Update the namespace of the shell with the given variables."""
        self.internel_namespace.update(kwargs)
        if self.shell is not None:
            self.shell.push(kwargs)
        elif self.kernel_client is not None:
            for key, value in kwargs.items():
                try:
                    if str(key).startswith("_") or str(key) == "In":
                        pass
                    elif (
                        callable(value)
                        or inspect.ismodule(value)
                        or inspect.isclass(value)
                    ):
                        mod = inspect.getmodule(value).__name__
                        if mod != "__main__":
                            if (
                                mod != value.__name__
                                and "." not in value.__name__
                            ):
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
                        isinstance(v, (int, float, complex, bool, str))
                        for v in value
                    ):
                        self.kernel_client.execute(
                            f"{key} = {repr(value)}",
                            silent=True,
                            stop_on_error=False,
                        )
                    elif isinstance(value, dict) and all(
                        isinstance(v, (int, float, complex, bool, str))
                        for v in value.values()
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
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
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
            # print("Re-initializing in terminal")
            ipython_terminal(self, **self.namespace)


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

        self.allow_shutdown = True
        if parent is not None:
            # print("parent connected to destroyed")
            self.allow_shutdown = False
            parent.close_children.connect(self.set_shutdown_logic)

    def set_shutdown_logic(self):
        self.allow_shutdown = True

    def closeEvent(self, event):
        """Handle the close event to ensure the application closes properly"""
        # print("Closing console...")
        if self.allow_shutdown:
            self.console.shutdown_kernel()
            self.deleteLater()  # Ensure the widget is properly deleted
        event.accept()
