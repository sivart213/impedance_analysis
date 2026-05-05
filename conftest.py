import re
import bisect
import shutil
from typing import TextIO
from pathlib import Path

import pytest

from testing.helpers import BUFFERED_STREAM, color_wrap, color_strip, print_buffer

TEST_CONFIG = {}
TEST_CONFIG["default_reporter"] = False
TEST_CONFIG["headers"] = set()
TEST_CONFIG["line_parts"] = set()
TEST_CONFIG["current_folder"] = None
TEST_CONFIG["current_file"] = None
TEST_CONFIG["current_group"] = None
TEST_CONFIG["current_function"] = None
TEST_CONFIG["test_meta"] = {}
TEST_CONFIG["subtest_meta"] = {}
TEST_CONFIG["sep"] = "; "


def pytest_addoption(parser: pytest.Parser):
    p_group = parser.getgroup("custom-id-formatting")

    p_group.addoption(
        "--param-sep",
        action="store",
        default="; ",
        help="Separator used between parameter values in test IDs.",
    )

    p_group.addoption(
        "--param-show-keys",
        action="store_true",
        default=False,
        help="Include parameter keys in test IDs (e.g., a=1; b=2).",
    )

    p_group.addoption(
        "--default-reporter",
        "--default-report",
        action="store_true",
        default=False,
        help="Use the default pytest reporter instead of the custom clean reporter.",
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config):
    TEST_CONFIG["term_width"] = shutil.get_terminal_size().columns
    TEST_CONFIG["default_reporter"] = config.getoption("--default-reporter")
    TEST_CONFIG["verbosity"] = verb = int(config.getoption("verbose"))
    TEST_CONFIG["printout"] = printing = config.getoption("capture") in {"no", "tee-sys"}
    TEST_CONFIG["sep"] = str(config.getoption("--param-sep"))

    if TEST_CONFIG["default_reporter"]:
        BUFFERED_STREAM.enabled = False
        return  # user did not request custom reporter
    BUFFERED_STREAM.enabled = printing  # Sync the stream's enabled state with the flag

    if verb >= 2 or (verb == 1 and config.getoption("--param-show-keys")):
        # -vv and above
        TEST_CONFIG["headers"] = {"folder", "file", "group"}
        TEST_CONFIG["line_parts"] = {"testname", "param_keys", "param_values"}
    elif verb == 1:
        # -v (with or without -s)
        TEST_CONFIG["headers"] = {"folder", "group"}
        TEST_CONFIG["line_parts"] = {"filename", "testname", "param_values"}
    elif verb <= 0 and printing:
        # -s without -v, or -q
        TEST_CONFIG["headers"] = {"folder", "file", "group", "function"}
        TEST_CONFIG["line_parts"] = set()

    if not getattr(config, "slaveinput", None):
        old = config.pluginmanager.get_plugin("terminalreporter")
        if old is not None:
            new = CleanReporter(old.config)
            config.pluginmanager.unregister(old)
        else:
            new = CleanReporter(config)
        config.pluginmanager.register(new, "terminalreporter")  # "clean-reporter"


def assign_section_to_item(item):
    sections = getattr(item.module, "_test_sections", None)
    if not sections:
        return

    # Keys are in insertion order, which matches source order
    lines = list(sections.keys())
    test_lineno = item.function.__code__.co_firstlineno

    idx = bisect.bisect_right(lines, test_lineno) - 1
    if idx >= 0:
        section = sections[lines[idx]]
        setattr(item.function, "_test_group", section)


def pytest_collection_modifyitems(session, config, items):
    root_path = session.config.rootpath
    for item in items:
        path = Path(item.path)
        assign_section_to_item(item)

        TEST_CONFIG["test_meta"][item.nodeid] = {
            "path": path,  # full path
            "rel_folder": path.parent.relative_to(root_path),  # folder relative to root
            "filename": path.name,  # just the file
            "name": item.name.split("[")[0],  # base test name
            "function": item.function.__name__,  # function name
            "params": {},  # params dict or None callspec.params if callspec else
        }
        if hasattr(item, "callspec"):
            TEST_CONFIG["test_meta"][item.nodeid]["params"] = item.callspec.params
        if hasattr(item.module, "_current_test_section"):
            setattr(item.function, "_test_group", item.module._current_test_section)


def pytest_runtest_protocol(item, nextitem):
    if TEST_CONFIG["default_reporter"]:
        # print_buffer()  # Flush any buffered prints before the test starts
        return

    meta = TEST_CONFIG["test_meta"][item.nodeid]
    headers = TEST_CONFIG["headers"]
    width = min(TEST_CONFIG["term_width"], 100)

    if "folder" in headers:
        folder = meta["path"].parent
        if folder != TEST_CONFIG["current_folder"]:
            TEST_CONFIG["current_folder"] = folder
            rel_folder = meta["rel_folder"]
            print(
                color_wrap(f"\n{('= ' + str(rel_folder) + ' =').center(width, '=')}", "mid_grey"),
                end="",
            )

    if "file" in headers:
        path = meta["path"]
        if path != TEST_CONFIG["current_file"]:
            TEST_CONFIG["current_file"] = path
            print(
                color_wrap(f"\n{('+ ' + meta['filename'] + ' +').center(width, '+')}", "grey"),
                end="",
            )

    if "group" in headers:
        group = getattr(item.function, "_test_group", None)
        if group and group != TEST_CONFIG["current_group"]:
            TEST_CONFIG["current_group"] = group
            print(
                color_wrap(f"\n{('~ ' + group + ' ~').center(width, '~')}", "light_magenta"),
                end="",
            )

    if "function" in headers:
        func = meta["function"]
        if func != TEST_CONFIG["current_function"]:
            TEST_CONFIG["current_function"] = func
            print(
                color_wrap(f"\n{('- ' + func + ' -').center(width, '-')}", "blue_bolt"),
                end="\n",
            )

    return None


@pytest.hookimpl(tryfirst=True)
def pytest_report_teststatus(report, config):
    if report.when != "call":
        return None

    if isinstance(report, pytest.SubtestReport):
        keys = frozenset(report.context.kwargs.keys())

        old = TEST_CONFIG.get("subtest_meta", {})
        old_keys = old.get("param_keys", frozenset())

        if old_keys == keys and "splitter" in old:
            splitter = old["splitter"]
        else:
            key_pattern = "|".join(re.escape(k) for k in keys)
            splitter = re.compile(rf"({key_pattern})=")

        TEST_CONFIG["subtest_meta"] = {
            "description": report._sub_test_description(),
            "msg": report.context.msg,
            "params": dict(report.context.kwargs),
            "param_keys": keys,
            "splitter": splitter,
        }
    else:
        TEST_CONFIG["subtest_meta"] = {}

    if TEST_CONFIG["verbosity"] <= 0 and TEST_CONFIG["printout"]:
        print_buffer()  # Flush any buffered prints before the test starts
    BUFFERED_STREAM.new_section = True
    return None


class CleanReporter(pytest.TerminalReporter):  # type: ignore
    def __init__(self, config, tw: TextIO | None = None):
        super().__init__(config, tw)
        # self._orig = None  # original reporter
        self._verbose_text = ""

    def write_fspath_result(self, nodeid: str, res: str, **markup: bool) -> None:
        # Suppress the "relative path" prefix when verbosity == 0
        if self.verbosity == 0:
            return
        super().write_fspath_result(nodeid, res, **markup)

    def write_ensure_prefix(self, prefix: str, extra: str = "", **kwargs) -> None:
        """
        Override the prefix writer so pytest's percent alignment stays correct
        even when prefix contains ANSI colors.
        """
        custom_prefix = self._format_prefix_text(prefix.strip())

        custom_extra, revised_kwargs = self._format_extra_text(extra, kwargs)

        super().write_ensure_prefix(custom_prefix + " ", custom_extra, **revised_kwargs)

        self._tw._current_line = color_strip(self._tw._current_line)

        if custom_extra:
            print_buffer()  # Flush any buffered prints after the prefix line

    def _format_prefix_text(self, nodeid):
        meta = TEST_CONFIG["test_meta"].get(nodeid)
        if not meta:
            return nodeid  # fallback

        parts = TEST_CONFIG["line_parts"]
        prefix_parts = []
        name = ""

        if "filename" in parts:
            prefix_parts.append(color_wrap(meta["filename"], "grey"))
            name = color_wrap(meta["filename"], "grey") + "::"

        if "testname" in parts:
            prefix_parts.append(meta["name"])
            name += color_wrap(meta["name"], "blue_bolt")

        if "param_values" in parts and meta["params"]:
            if "param_keys" in parts:
                # key=value
                param_str = TEST_CONFIG["sep"].join(
                    f"{color_wrap(k, 'light_cyan')}={color_wrap(v, 'bondi_blue')}"
                    for k, v in meta["params"].items()
                )
            else:
                # values only
                param_str = TEST_CONFIG["sep"].join(
                    f"{color_wrap(v, 'bondi_blue')}" for v in meta["params"].values()
                )
            return f"{name}[{param_str}]"

        return name

    def _format_extra_text(
        self, extra: str, kwargs: dict[str, bool]
    ) -> tuple[str, dict[str, bool]]:
        """
        Transform pytest's 'extra' text for subtests into a structured,
        colorized form. If no subtest metadata is present or the description
        is not found in 'extra', return both unchanged.
        """
        if not extra or not (meta := TEST_CONFIG["subtest_meta"]):
            return extra, kwargs

        description = meta.get("description", "")

        # No subtest active → no changes
        if not description or description not in extra or description == "(<subtest>)":
            return extra, kwargs

        msg = meta.get("msg") or ""
        p_keys = meta.get("param_keys", frozenset())

        # 1. Isolate the clean outcome (e.g., "SUBPASSED", "SUBFAILED")
        clean = extra.replace(description, "").strip()

        # 2. Apply pytest's color markup to the clean outcome
        revised_kwargs = kwargs.copy()
        for color, enabled in list(kwargs.items()):
            if enabled:
                clean = color_wrap(clean, color)
                del revised_kwargs[color]
                break

        # 3. Insert the subtest message in blue inside square brackets
        if msg:
            clean += f"[{color_wrap(msg, 'dark_turquoise')}]"

        # 4. Format params using the same style as _format_prefix_text
        if p_keys:
            i = 0
            colored_parts = []
            tokens = meta["splitter"].split(description.replace(msg, "").strip())
            while i < len(tokens) - 1:
                key = tokens[i]
                val = tokens[i + 1]

                # Skip anything that isn't a real param key
                if key not in p_keys:
                    i += 1
                    continue

                # Remove exactly one trailing bracket OR one separator + optional space
                val = re.sub(r"(?:[\]\)\}\>,\.;:|!?]$)", "", val.strip())

                colored_parts.append(
                    f"{color_wrap(key.strip(), 'light_cyan')}="
                    f"{color_wrap(val.strip(), 'bondi_blue')}"
                )
                i += 2

            if colored_parts:
                clean += f" ({TEST_CONFIG['sep'].join(colored_parts)})"

        return clean, revised_kwargs
