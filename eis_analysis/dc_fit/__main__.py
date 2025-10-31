# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

from IPython.core.getipython import get_ipython

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

try:
    from .dc_data_post import (
        run_point_fitting,
    )
    from .extract_tools import (
        DEFAULT_DIR,
        load_dc_data,
        save_results,  # noqa: F401
    )
    from .dc_data_fitting import (
        run_fitting,
        run_fit_kwargs,
    )
    from .dc_data_cleaning import (
        pre_process,
        pre_process_kwargs,
    )
    from .dc_data_processing import (
        perm_jobs,
        run_perm_calcs,
    )

except ImportError:
    from eis_analysis.dc_fit.dc_data_post import (
        run_point_fitting,
    )
    from eis_analysis.dc_fit.extract_tools import (
        DEFAULT_DIR,
        load_dc_data,
        save_results,  # noqa: F401
    )
    from eis_analysis.dc_fit.dc_data_fitting import (
        run_fitting,
        run_fit_kwargs,
    )
    from eis_analysis.dc_fit.dc_data_cleaning import (
        pre_process,
        pre_process_kwargs,
    )
    from eis_analysis.dc_fit.dc_data_processing import (
        perm_jobs,
        run_perm_calcs,
    )


_CELL_NAMES = ["all", "clean", "fit", "perm"]
_parser = argparse.ArgumentParser(description="Evaluate DC data.")

_parser.add_argument(
    "--steps",
    nargs="+",
    choices=_CELL_NAMES,
    action="extend",
    dest="steps",
    help="Which steps to run (defaults to all). e.g. --steps clean fit",
)

for _i, _name in enumerate(_CELL_NAMES):
    _parser.add_argument(
        f"--run-{_name}",
        f"-r{_i}",
        action="append_const",
        const=_name,
        dest="steps",
        help=f"Shortcut for running the '{_name}' cell",
    )

_parser.add_argument(
    "--raw-path",
    type=Path,
    default=pre_process_kwargs.get("data_path", DEFAULT_DIR.parent),
    help="Define the load path for the raw files.",
)
_parser.add_argument(
    "--save-path",
    type=Path,
    default=DEFAULT_DIR,
    help="Define the save path for the results.",
)
_parser.add_argument(
    "-w",
    "--warnings",
    action="store_true",  # False unless called
    help="Show warnings (default: suppressed)",
)
_parser.add_argument(
    "-q",
    "--quiet",
    action="store_false",  # True unless called
    help="Prevents update messages (default: prints messages)",
)


_shell = get_ipython()
if _shell is not None:
    # Running in IPython environment (like Spyder)
    print("Running in IPython environment.")
    # Ask if user wants to provide command line arguments
    _use_custom = input("Enter command line arguments: ").split()
    _args = _parser.parse_args(_use_custom)
else:
    _args = _parser.parse_args()


run_steps = set(_args.steps or ["all"])
if "all" in run_steps:
    run_steps = set(_CELL_NAMES)

try:
    status = {}
    curves = globals()["curves"] if "curves" in globals() else {}
    points = globals()["points"] if "points" in globals() else {}
    params = globals()["params"] if "params" in globals() else {}
    perm_curves = {}
    perm_peaks = {}

    # %% Run initial data processing
    if "clean" in run_steps:
        # Use raw_path and save_path from _args
        pre_process_kwargs["data_path"] = _args.raw_path
        curves, rej_curves, points, rej_points = pre_process(
            verbose=_args.quiet,
            **pre_process_kwargs,
        )
        if "fit" not in run_steps:
            save_results(
                data_path=_args.save_path,
                curves=curves,
                save_mode="w",
                attrs=True,
                verbose=_args.quiet,
            )

    # %% Run data Fitting
    if "fit" in run_steps:
        points = points or load_dc_data(_args.save_path, "points")["points"]
        curves = curves or load_dc_data(_args.save_path, "curves")["curves"]
        points, curves, params = run_fitting(
            points=points,
            curves=curves,
            status=status,
            params=params,
            verbose=_args.quiet,
            **run_fit_kwargs,
        )
        # --- Run fit results Fitting ---
        results = run_point_fitting(
            points=points,
            params=params,
            curves=curves,
        )
        save_results(
            _args.save_path,
            attrs=True,
            verbose=_args.quiet,
            **results,  # type: ignore[call-arg]
        )
        # Save points and new/modified curves
        save_results(
            _args.save_path,
            curves=curves,
            save_mode="w",
            attrs=True,
            verbose=_args.quiet,
        )

    # %% Run Permitivitty analysis
    if "perm" in run_steps:
        curves = curves or load_dc_data(_args.save_path, "curves")["curves"]
        params = params or load_dc_data(_args.save_path, "params")["params"]
        perm_curves, perm_peaks = run_perm_calcs(
            curves=curves,
            perm_jobs=perm_jobs,
            perm_curves=perm_curves,
            perm_peaks=perm_peaks,
            params=params,
            suppress_warnings=not _args.warnings,
            verbose=_args.quiet,
        )
        # Save results
        save_results(
            data_path=_args.save_path,
            perm_curves=perm_curves,
            perm_peaks=perm_peaks,
            save_mode="w",
            attrs=True,
            verbose=_args.quiet,
        )
except KeyboardInterrupt:
    print("Process interrupted by user.")
sys.exit(0)
