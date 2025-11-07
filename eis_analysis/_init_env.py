"""
Centralized environment/bootstrap helpers for EIS GUIs.

This module handles:
- Debug mode detection
- Spyder/IPython quirks
- Qt application initialization
"""

import sys
from collections.abc import Callable

from PyQt5.QtWidgets import QApplication
from IPython.core.getipython import get_ipython


def is_debug_mode() -> bool:
    """Return True if running under a debugger (e.g. Spyder debug)."""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def spyder_run(shell):
    try:
        from IPython.lib.guisupport import (
            start_event_loop_qt4,
            is_event_loop_running_qt4,
        )

        def start_event_loop_qt4_wrapper(app: QApplication):
            if not is_event_loop_running_qt4():
                shell.run_line_magic("gui", "qt")
            start_event_loop_qt4(app)

        return start_event_loop_qt4_wrapper
    except ImportError:
        # Fallback: just start normally
        return lambda a: a.exec_()


def init_qt_app() -> tuple[QApplication, bool, Callable[[QApplication], None]]:
    """
    Initialize a QApplication with Spyder/IPython quirks handled.

    Returns
    -------
    app : QApplication
        The Qt application instance (reused if already created).
    debug : bool
        Whether debug mode is active.
    """
    shell = get_ipython()
    debug = is_debug_mode()

    # Reuse existing app if one exists
    app = QApplication.instance() or QApplication(sys.argv)

    if shell is not None:
        # Running inside IPython/Spyder
        try:
            shell.run_line_magic("matplotlib", "inline")
        except Exception:
            pass

        # Spyder-specific: ensure Qt event loop is running
        if "SPYDER" in shell.__class__.__name__.upper():
            return app, debug, spyder_run(shell)

    return app, debug, lambda a: sys.exit(a.exec_())
