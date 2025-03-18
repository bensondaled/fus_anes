##
from PyQt5 import QtWidgets
import sys

from fus_anes.util import setup_logging
from fus_anes.experiments import Controller

if __name__ == '__main__':
    setup_logging()
    app = QtWidgets.QApplication([])
    c = Controller(app=app)
    c.ui.show()
    sys.exit(app.exec())

##
