import sys
import h5py
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QGraphicsScene
from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt

from data.load_data import HDF5TreeModel
from data.assign_data import assign_data_from_file
from plots.plots import plot_raster
from gui.MatplotlibWidget import MatplotlibWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loadUi("gui/MainWindow.ui", self)
        ## Browse files
        self.browseFile.clicked.connect(self.browse_files)

        ## Set the activity variables
        self.btn_set_neuronal_activity.clicked.connect(self.set_neuronal_activity)

    def update_console_log(self, message):
        # Get current date and time
        current_date_time = QDateTime.currentDateTime().toString(Qt.DateFormat.ISODateWithMs)
        # Construct log entry with date and message
        log_entry = f"{current_date_time}: {message}"
        # Append the message to the console log
        self.console_log.appendPlainText(log_entry)
        # Ensure the console log updates immediately
        self.console_log.repaint()

    def browse_files(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.filenamePlain.setText(fname)
        self.update_console_log("Loading HDF5 file...")
        if fname:
            self.source_filename = fname
            # Load the HDF5 file
            self.update_console_log("Generating file structure...")
            hdf5_file = h5py.File(fname, 'r')
            self.model = HDF5TreeModel(hdf5_file)
            self.tree_view.setModel(self.model)
            self.update_console_log("Done loading file.")
            # Connect the clicked signal of the tree view to a slot
            self.tree_view.clicked.connect(self.item_clicked)
        else:
            self.update_console_log("HDF5 file not found.")

    def item_clicked(self, index):
        # Get the item from the index
        item = self.model.data_name(index)
        self.file_tree_selected = item

    def set_neuronal_activity(self):
        new_text = "Assigned"
        self.lbl_neuronal_activity_selected.setText(new_text)
        data_neuronal_activity = assign_data_from_file(self)
        print(data_neuronal_activity.shape)
        self.data_neuronal_activity = data_neuronal_activity

        # Create QGraphicsScene
        #self.scene = QGraphicsScene()
        #self.neuronal_raster_plot.setScene(self.scene)
        #plot_raster(self)

        # a figure instance to plot on
        #self.figure = plt.figure()
        #self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_widget = self.findChild(MatplotlibWidget, 'neuronal_raster_plot')
        # Example data to plot
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 4, 9, 16]

        # Plot the data
        self.plot_widget.plot(x, y)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()  