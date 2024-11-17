import sys
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow
from .z_fit.__main__ import GraphGUI
from .parse_files.__main__ import MFIAFileConverter

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--gui", type=str, default="main", help="Select the target GUI to run. Use 'main' for the main GUI, 'file' for the MFIA file converter.")

args = parser.parse_args()

opt1_value = args.gui

if opt1_value.lower() == "file":
    app = QApplication(sys.argv)
    main_window = MFIAFileConverter()
    main_window.show()
    sys.exit(app.exec_())
else:
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = GraphGUI(main_window)
    main_window.show()
    sys.exit(app.exec_())

