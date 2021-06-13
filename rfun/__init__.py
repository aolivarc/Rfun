from PyQt5 import QtWidgets
from .rfun_main_window import RfunMainWindow

def start_rfun():
    app = QtWidgets.QApplication([])
    window = RfunMainWindow()
    app.exec_()