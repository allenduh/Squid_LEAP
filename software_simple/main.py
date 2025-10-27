# set QT_API environment variable
import os 
import sys
import argparse
os.environ["QT_API"] = "pyqt5"

# qt libraries
from qtpy.QtWidgets import *

# app specific libraries
import control.gui as gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    app = QApplication([])
    app.setStyle('Fusion')

    win = gui.OctopiGUI()
    win.show()
    sys.exit(app.exec_())
