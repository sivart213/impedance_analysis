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
import re
import inspect
import logging

from PyQt5.QtWidgets import (
    QMainWindow,
    QMessageBox,
)

from ..utils.plot_factory import StylizedPlot
from ..system_utilities.log_config import setup_logging

# if get_ipython() is not None:
#     get_ipython().run_line_magic("matplotlib", "inline")
setup_logging()

logger = logging.getLogger("eis_analysis")
logger.setLevel(logging.INFO)


# # Redirect warnings to the logger
def log_warning(message, category, filename, lineno, *_, **__):
    """Custom warning handler to log warnings."""
    logger.warning("%s:%s: %s: %s", filename, lineno, category.__name__, message)


def set_style(style_name="seaborn-v0_8-whitegrid"):
    """Set the style for the plots."""
    try:
        StylizedPlot.set_style(style_name)
    except OSError as exc:
        logger.error("Error setting style: %s", exc)
        StylizedPlot.set_style("seaborn-v0_8-darkgrid")


class GraphGUIError(Exception):
    """Exception raised for errors in the GUI."""

    pass


def graceful_error_handler(popup=False, title=None):
    """Decorator to handle unexpected errors gracefully in the GUI."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if not args and not kwargs:
                    return func()
                elif not kwargs:
                    return func(*args)
                elif not args:
                    return func(**kwargs)
                return func(*args, **kwargs)
            except GraphGUIError as exc:
                # Check if ":" is in the message and adjust accordingly
                error_message = str(exc).strip() if ":" in str(exc) else ": " + str(exc).strip()

                # Construct the full error message
                full_message = (
                    f"{exc.__cause__.__class__.__name__} in {func.__name__} {error_message}"
                )

                # Log the error
                logger.error(full_message)

                # Optionally display the error message in a popup
                if popup:
                    # Use the provided title or default to the function name
                    error_title = (
                        str(title).strip().title() or func.__name__.replace("_", " ").title()
                    )
                    # Format the title for the popup
                    formatted_title = re.sub(r"error$", "", error_title).strip() + " Error"

                    # Display the error message
                    QMessageBox.critical(
                        args[0] if args and isinstance(args[0], QMainWindow) else None,
                        formatted_title,
                        re.sub(r":\s*", ":\n", full_message),
                    )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def construct_error_message(error, message="", identifier=None, identifier_joiner="on step"):
    """
    Construct a formatted error message.

    Args:
        error (Exception): The exception object.
        message (str): The context or description of the error.
        identifier (str, optional): An additional identifier to include in the message.
        identifier_joiner (str, optional): A string to join the identifier with the message.

    Returns:
        str: The constructed error message.
    """
    # Format the message and identifier
    message = " " + str(message).strip()
    identifier_str = (
        f" {str(identifier_joiner)} {identifier.strip()}"
        if isinstance(identifier, str) and len(identifier.strip()) > 1
        else ""
    )

    # Construct the error message
    return f"{message}{identifier_str}: {str(error)}"


def show_error_message(
    error,
    title="",
    message="",
    identifier=None,
    identifier_joiner="on step",
    popup=True,
    parent=None,
):
    """
    Display an error message in a QMessageBox and log the error.

    Args:
        error (Exception): The exception object.
        title (str): The title of the error message.
        message (str): The context or description of the error.
        identifier (str, optional): An additional identifier to include in the message. Defaults to None.
        identifier_joiner (str, optional): A string to join the identifier with the message. Defaults to "on step".
    """
    if title == "":
        stack = inspect.stack()
        for s in stack:
            # check for "show_error_message" in the code context and get
            if isinstance(s.code_context, list) and any(
                "show_error_message" in line for line in s.code_context
            ):
                caller_function = s.function
                caller_class = (
                    s.frame.f_locals.get("self", None).__class__.__name__
                    if "self" in s.frame.f_locals
                    else None
                )
                title = f"{caller_class}.{caller_function}" if caller_class else caller_function
                break

    # Pre-parse the title to remove "Error" if it ends with it
    l_title = str(title).lower().strip()
    l_title = re.sub(r"error$", "", l_title).strip()
    # l_title = l_title.replace(".", ": ")

    # Format the message and identifier
    message = " " + str(message).strip()
    identifier_str = (
        f" {str(identifier_joiner)} {identifier.strip()}"
        if isinstance(identifier, str) and len(identifier.strip()) > 1
        else ""
    )

    # Log the error
    logger.error(
        "%s in %s%s%s: %s",
        error.__class__.__name__,
        str(title),
        message,
        identifier_str,
        str(error),
    )
    if popup:
        # Display the error message
        QMessageBox.critical(
            parent,
            l_title.replace(".", ": ").title() + " Error",
            f"{error.__class__.__name__} in {l_title}{message}{identifier_str}:\n{str(error)}",
        )


# # Set the global warning handler
# warnings.showwarning = log_warning

# np.seterr(all="raise")


# def manage_settings_files(base_name="settings"):
#     """Manage settings files for the application."""
#     count = 0
#     for proc in psutil.process_iter(["pid", "name", "cmdline"]):
#         if "python" in proc.info["name"] and any("z_fit" in info for info in proc.info["cmdline"]):
#             count += 1
#         # return count
#     if count == 0:
#         count = 1
#     # instance_count = count_instances()
#     settings_file = Path(f"{base_name}_{count}.json")
#     copy_from_file = Path(f"{base_name}_{count - 1}.json") if count > 1 else None

#     # Initialize JSONSettings object
#     json_settings = JSONSettings(settings_file, copy_from=copy_from_file)

#     # Get the true path from JSONSettings
#     settings_dir = json_settings.settings_path.parent

#     # Remove any unwanted files
#     for file in settings_dir.glob(f"{base_name}_*.json"):
#         file_instance_number = int(file.stem.split("_")[-1])
#         if file_instance_number > count:
#             file.unlink()

#     return json_settings
