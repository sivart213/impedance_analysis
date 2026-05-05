import re
import sys
import inspect
from collections.abc import Iterable

from termcolor import colored

ANSI_CODES = {}
ANSI_CODES |= {
    "very_pale_blue": (215, 215, 255),
    "french_pass": (175, 255, 255),
    "cerulean": (0, 175, 215),
    "blue_bolt": (0, 175, 255),
    "teal": (0, 135, 135),
    "bondi_blue": (0, 175, 175),
    "dark_turquoise": (0, 215, 215),
    "aqua": (0, 255, 255),
    "light_cyan": (215, 255, 255),
    "very_pale_lime_green": (215, 255, 215),
    "orange": (255, 175, 0),
    "orage_red": (255, 95, 0),
    "strong_orange": (215, 95, 0),
    "dark_orange": (175, 95, 0),
    "bronze": (175, 135, 95),
    "brown": (135, 95, 0),
    "pirate_gold": (175, 135, 0),
    "light_gold": (175, 175, 0),
    "harvest_gold": (215, 135, 0),
    "grey": "dark_grey",  # (128, 128, 128),
    "mid_grey": (160, 160, 160),
}


ANSI_RE = re.compile(r"(?:\x1b|\x1B|\033)[^m]*m")


def color_wrap(
    text: object,
    color: str | tuple[int, int, int] | None = None,
    on_color: str | tuple[int, int, int] | None = None,
    attrs: Iterable[str] | None = None,
    *,
    no_color: bool | None = None,
    force_color: bool | None = None,
) -> str:
    """
    Wraps termcolor.colored(), expanding color options to any color names found in ANSI_CODES.
    """
    try:
        return colored(
            text,
            color=ANSI_CODES.get(color, color),
            on_color=ANSI_CODES.get(on_color, on_color),
            attrs=attrs,
            no_color=no_color,
            force_color=force_color,
        )
    except Exception:
        raise


def color_strip(text: str) -> str:
    """
    Remove ANSI (color) codes from text.
    """
    return ANSI_RE.sub("", text)


def group(name: str):
    def wrapper(func):
        setattr(func, "_test_group", name)
        return func

    return wrapper


def set_section(name):
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return

    module = sys.modules[frame.f_back.f_globals["__name__"]]
    lineno = frame.f_back.f_lineno

    sections = getattr(module, "_test_sections", None)
    if sections is None:
        sections = {}
        setattr(module, "_test_sections", sections)

    sections[lineno] = name


class BufferedStream:
    """
    A unified output stream that optionally buffers writes.

    This object implements the minimal stream interface required by
    logging.StreamHandler: a `write()` method and a `flush()` method.

    When buffering is enabled, writes are appended to an internal list
    and emitted only when `flush()` is called. When buffering is disabled,
    writes are forwarded directly to the underlying stream (default: sys.stdout).
    """

    def __init__(self, *, enabled=False):
        self.enabled = enabled
        self._buffer = []
        self.new_section = True

    def write(self, text):
        """
        Write text to the buffer or underlying stream.

        Parameters
        ----------
        text : str
            The text to write. Logging may call this with partial lines.
        """
        if not text:
            return

        if self.enabled:
            self._buffer.append(text)
        else:
            if self.new_section and not text.startswith("\n"):
                text = "\n" + text
            self.new_section = False
            sys.stdout.write(text)
            sys.stdout.flush()

    def flush(self):
        """
        Flush buffered text to the underlying stream.
        """
        if not self.enabled:
            # Underlying stream may still need flushing
            self._buffer.clear()

        if self._buffer:
            # Join without adding extra newlines; logging controls its own formatting
            text = "".join(self._buffer)
            if self.new_section and not text.startswith("\n"):
                text = "\n" + text
            sys.stdout.write(text)
            self._buffer.clear()
            self.new_section = False

        sys.stdout.flush()

    def print(self, *args, sep=" ", end="\n", **_):
        """
        Convenience method to print text with buffering behavior.
        """
        text = sep.join(str(a) for a in args)
        self.write(text + end)

    # Optional: logging may check for this
    def isatty(self):
        return False


# # Global instance used by both print wrappers and logging
BUFFERED_STREAM = BufferedStream(enabled=False)


def buffered_print(*args, **kwargs):
    """
    Thin wrapper around BUFFERED_STREAM.print().
    """
    BUFFERED_STREAM.print(*args, **kwargs)


def print_buffer():
    """
    Thin wrapper around BUFFERED_STREAM.flush().
    """
    BUFFERED_STREAM.flush()
