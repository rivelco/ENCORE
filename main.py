import sys
import h5py
import os
import csv
import scipy.io 
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QGraphicsScene
from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QStandardItemModel

from data.load_data import FileTreeModel
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
        # Connect the clicked signal of the tree view to a slot
        self.tree_view.clicked.connect(self.item_clicked)

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
            file_extension = os.path.splitext(fname)[1]

            self.update_console_log("Generating file structure...")
            if file_extension == '.h5' or file_extension == '.hdf5':
                hdf5_file = h5py.File(fname, 'r')
                self.file_model_type = "hdf5"
                self.file_model = FileTreeModel(hdf5_file, model_type="hdf5")
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading file.")
            elif file_extension == '.mat':
                mat_file = scipy.io.loadmat(fname)
                self.file_model_type = "mat"
                self.file_model = FileTreeModel(mat_file, model_type="mat")
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading matlab file.")
            elif file_extension == '.csv':
                self.file_model_type = "csv"
                with open(fname, 'r', newline='') as csvfile:
                    self.file_model = FileTreeModel(csvfile, model_type="csv")
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading csv file.") 
            else:
                print("Unsupported file format")
            
        else:
            self.update_console_log("File not found.")

    def item_clicked(self, index):
        # Get the item data from the index
        item_path = self.file_model.data_name(index)
        item_type = self.file_model.data_type(index)
        item_size = self.file_model.data_size(index)
        item_name = item_path.split('/')[-1]

        # Report description to UI
        new_text = f" {item_name} is a {item_type}"
        if item_type == "Dataset":
            new_text += f" with {item_size} shape."
        elif item_type == "Group":
            new_text += f" with {item_size} elements."
        else:
            new_text += f"."

        self.browser_var_info.setText(new_text)

        # Enable or disable the assign buttons
        if item_type == "Dataset":
            self.btn_set_dFFo.setEnabled(True)
            self.btn_set_neuronal_activity.setEnabled(True)
            self.btn_set_coordinates.setEnabled(True)
            self.btn_set_stim_data.setEnabled(True)
            self.btn_set_behavior_data.setEnabled(True)
        else:
            self.btn_set_dFFo.setEnabled(False)
            self.btn_set_neuronal_activity.setEnabled(False)
            self.btn_set_coordinates.setEnabled(False)
            self.btn_set_stim_data.setEnabled(False)
            self.btn_set_behavior_data.setEnabled(False)

        # Store data description temporally
        self.file_selected_var_path = item_path
        self.file_selected_var_type = item_type
        self.file_selected_var_size = item_size

    def set_neuronal_activity(self):
        data_neuronal_activity = assign_data_from_file(self)
        self.data_neuronal_activity = data_neuronal_activity

        new_text = "Assigned"
        self.lbl_neuronal_activity_select.setText(new_text)
        
        #self.plot_widget = self.findChild(MatplotlibWidget, 'neuronal_raster_plot')
        # Plot the data
        #self.plot_widget.raster_plot(data_neuronal_activity)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()  