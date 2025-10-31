import sys
import argparse

from PyQt5.QtWidgets import QApplication
from IPython.core.getipython import get_ipython

try:
    from .z_fit.__main__ import GraphGUI
    from .parse_files.__main__ import MFIAFileConverter
except ImportError:
    from eis_analysis.z_fit.__main__ import GraphGUI
    from eis_analysis.parse_files.__main__ import MFIAFileConverter

parser = argparse.ArgumentParser(description="Evaluate EIS files and/or data.")
parser.add_argument(
    "--gui",
    type=str,
    default="ask",
    help="Select the target GUI to run. Use 'main' for the main GUI, 'file' for the MFIA file converter.",
)

target = parser.parse_args().gui.lower()

if target == "ask":
    target = input("Use Main GUI or File loader? (return main or file)\n").strip().lower()

if target == "file":
    TargetGui = MFIAFileConverter
else:
    TargetGui = GraphGUI

shell = get_ipython()
is_debug = hasattr(sys, "gettrace") and sys.gettrace() is not None
if is_debug:
    # print("Running in debug mode.")
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")
        # shell.run_line_magic("gui", "qt")
    app = QApplication(sys.argv)
    window = TargetGui(debug=is_debug)
    window.show()
    sys.exit(app.exec_())
elif shell is not None and "SPYDER" in shell.__class__.__name__.upper():
    # print("Running in Spyder.")
    shell.run_line_magic("matplotlib", "inline")
    app = QApplication(sys.argv)
    window = TargetGui(debug=is_debug)
    window.show()
    try:
        from IPython.lib.guisupport import start_event_loop_qt4, is_event_loop_running_qt4

        if not is_event_loop_running_qt4():
            shell.run_line_magic("gui", "qt")
        start_event_loop_qt4(app)
    except ImportError:
        app.exec_()

elif "--terminal" in sys.argv:
    import subprocess

    subprocess.run(["python", __file__])
else:
    app = QApplication(sys.argv)
    window = TargetGui()
    window.show()
    sys.exit(app.exec_())

# if opt1_value.lower() == "file":
#     app = QApplication(sys.argv)
#     main_window = MFIAFileConverter()
#     main_window.show()
#     sys.exit(app.exec_())
# else:
#     app = QApplication(sys.argv)
#     main_window = QMainWindow()
#     ui = GraphGUI(main_window)
#     main_window.show()
#     sys.exit(app.exec_())
