import sys
import h5py
import os
import csv
import scipy.io 
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QGraphicsScene
from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QStandardItemModel, QTextCursor, QColor

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
        self.setWindowTitle('Ensembles GUI')
        ## Browse files
        self.browseFile.clicked.connect(self.browse_files)
        # Connect the clicked signal of the tree view to a slot
        self.tree_view.clicked.connect(self.item_clicked)

        ## Identify change of tab
        self.main_tabs.currentChanged.connect(self.onTabChange)

        ## Set the activity variables
        self.btn_set_dFFo.clicked.connect(self.set_dFFo)
        self.btn_set_neuronal_activity.clicked.connect(self.set_neuronal_activity)
        self.btn_set_coordinates.clicked.connect(self.set_coordinates)
        self.btn_set_stim.clicked.connect(self.set_stims)
        self.btn_set_behavior.clicked.connect(self.set_behavior)

        ## Set the clear buttons
        self.btn_clear_dFFo.clicked.connect(self.clear_dFFo)
        self.btn_clear_neuronal_activity.clicked.connect(self.clear_neuronal_activity)
        self.btn_clear_coordinates.clicked.connect(self.clear_coordinates)
        self.btn_clear_stim.clicked.connect(self.clear_stims)
        self.btn_clear_behavior.clicked.connect(self.clear_behavior)

        ## Set the clear buttons
        self.btn_view_dFFo.clicked.connect(self.view_dFFo)
        self.btn_view_neuronal_activity.clicked.connect(self.view_neuronal_activity)
        self.btn_view_coordinates.clicked.connect(self.view_coordinates)
        self.btn_view_stim.clicked.connect(self.view_stims)
        self.btn_view_behavior.clicked.connect(self.view_behavior)

        ## Edit actions
        self.btn_edit_transpose.clicked.connect(self.edit_transpose)

    def update_console_log(self, message, msg_type="log"):
        color_map = {"log": "#000000", "error": "#da1e28", "warning": "#ff832b", "complete": "#198038"}
        current_date_time = QDateTime.currentDateTime().toString(Qt.DateFormat.ISODateWithMs)

        log_entry = f"<span style=\"font-family:monospace; font-size:10pt; font-weight:600; color:{color_map[msg_type]};\">"
        log_entry += f"{current_date_time}: {message}"
        log_entry += "</span>"

        self.console_log.append(log_entry)
        # Scroll to the bottom to ensure the new message is visible
        self.console_log.moveCursor(QTextCursor.MoveOperation.End)
        self.console_log.repaint()

    def browse_files(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.filenamePlain.setText(fname)
        self.update_console_log("Loading file...")
        if fname:
            self.source_filename = fname
            file_extension = os.path.splitext(fname)[1]
            if file_extension == '.h5' or file_extension == '.hdf5' or file_extension == ".nwb":
                self.update_console_log("Generating file structure...")
                hdf5_file = h5py.File(fname, 'r')
                self.file_model_type = "hdf5"
                self.file_model = FileTreeModel(hdf5_file, model_type="hdf5")
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading file.", "complete")
            elif file_extension == '.mat':
                self.update_console_log("Generating file structure...")
                mat_file = scipy.io.loadmat(fname)
                self.file_model_type = "mat"
                self.file_model = FileTreeModel(mat_file, model_type="mat")
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading matlab file.", "complete")
            elif file_extension == '.csv':
                self.update_console_log("Generating file structure...")
                self.file_model_type = "csv"
                with open(fname, 'r', newline='') as csvfile:
                    self.file_model = FileTreeModel(csvfile, model_type="csv")
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading csv file.", "complete")
            else:
                self.update_console_log("Unsupported file format", "warning")
        else:
            self.update_console_log("File not found.", "error")

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
            valid = item_size[0] > 1
            self.btn_set_dFFo.setEnabled(valid)
            self.btn_set_neuronal_activity.setEnabled(valid)
            valid = len(item_size) > 1
            self.btn_set_coordinates.setEnabled(valid)
            self.btn_set_stim.setEnabled(True)
            self.btn_set_behavior.setEnabled(True)
        else:
            self.btn_set_dFFo.setEnabled(False)
            self.btn_set_neuronal_activity.setEnabled(False)
            self.btn_set_coordinates.setEnabled(False)
            self.btn_set_stim.setEnabled(False)
            self.btn_set_behavior.setEnabled(False)

        # Store data description temporally
        self.file_selected_var_path = item_path
        self.file_selected_var_type = item_type
        self.file_selected_var_size = item_size
        self.file_selected_var_name = item_name
    
    ## Identify the tab changes
    def onTabChange(self, index):
        if index == 1: # SVD tab
            if hasattr(self, "data_neuronal_activity"):
                self.lbl_sdv_spikes_selected.setText(f"Loaded")
            else:
                self.lbl_sdv_spikes_selected.setText(f"Nothing selected")
            if hasattr(self, "data_coordinates"):
                self.lbl_sdv_coordinates_selected.setText(f"Loaded")
            else:
                self.lbl_sdv_coordinates_selected.setText(f"Nothing selected")
            if hasattr(self, "data_dFFo"):
                self.lbl_sdv_dFFo_selected.setText(f"Loaded")
            else:
                self.lbl_sdv_dFFo_selected.setText(f"Nothing selected")

    ## Set variables from input file
    def set_dFFo(self):
        data_dFFo = assign_data_from_file(self)
        self.data_dFFo = data_dFFo
        self.btn_clear_dFFo.setEnabled(True)
        self.btn_view_dFFo.setEnabled(True)
        self.lbl_dffo_select.setText("Assigned")
        self.lbl_dffo_select_name.setText(self.file_selected_var_name)
    def set_neuronal_activity(self):
        data_neuronal_activity = assign_data_from_file(self)
        self.data_neuronal_activity = data_neuronal_activity
        self.btn_clear_neuronal_activity.setEnabled(True)
        self.btn_view_neuronal_activity.setEnabled(True)
        self.lbl_neuronal_activity_select.setText("Assigned")
        self.lbl_neuronal_activity_select_name.setText(self.file_selected_var_name)
    def set_coordinates(self):
        data_coordinates = assign_data_from_file(self)
        self.data_coordinates = data_coordinates[:, 0:2]
        self.btn_clear_coordinates.setEnabled(True)
        self.btn_view_coordinates.setEnabled(True)
        self.lbl_coordinates_select.setText("Assigned")
        self.lbl_coordinates_select_name.setText(self.file_selected_var_name)
    def set_stims(self):
        data_stims = assign_data_from_file(self)
        self.data_stims = data_stims
        self.btn_clear_stim.setEnabled(True)
        self.btn_view_stim.setEnabled(True)
        self.lbl_stim_select.setText("Assigned")
        self.lbl_stim_select_name.setText(self.file_selected_var_name)
    def set_behavior(self):
        data_behavior = assign_data_from_file(self)
        self.data_behavior = data_behavior
        self.btn_clear_behavior.setEnabled(True)
        self.btn_view_behavior.setEnabled(True)
        self.lbl_behavior_select.setText("Assigned")
        self.lbl_behavior_select_name.setText(self.file_selected_var_name)
    
    ## Clear variables 
    def clear_dFFo(self):
        delattr(self, "data_dFFo")
        self.btn_clear_dFFo.setEnabled(False)
        self.btn_view_dFFo.setEnabled(False)
        self.lbl_dffo_select.setText("Nothing")
        self.lbl_dffo_select_name.setText("")
    def clear_neuronal_activity(self):
        delattr(self, "data_neuronal_activity")
        self.btn_clear_neuronal_activity.setEnabled(False)
        self.btn_view_neuronal_activity.setEnabled(False)
        self.lbl_neuronal_activity_select.setText("Nothing")
        self.lbl_neuronal_activity_select_name.setText("")
    def clear_coordinates(self):
        delattr(self, "data_coordinates")
        self.btn_clear_coordinates.setEnabled(False)
        self.btn_view_coordinates.setEnabled(False)
        self.lbl_coordinates_select.setText("Nothing")
        self.lbl_coordinates_select_name.setText("")
    def clear_stims(self):
        delattr(self, "data_stims")
        self.btn_clear_stim.setEnabled(False)
        self.btn_view_stim.setEnabled(False)
        self.lbl_stim_select.setText("Nothing")
        self.lbl_stim_select_name.setText("")
    def clear_behavior(self):
        delattr(self, "data_behavior")
        self.btn_clear_behavior.setEnabled(False)
        self.btn_view_behavior.setEnabled(False)
        self.lbl_behavior_select.setText("Nothing")
        self.lbl_behavior_select_name.setText("")

    ## Set variables from input file
    def view_dFFo(self):
        self.currently_visualizing = "dFFo"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_dataset(self.data_dFFo)
    def view_neuronal_activity(self):
        self.currently_visualizing = "neuronal_activity"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.raster_plot(self.data_neuronal_activity)
    def view_coordinates(self):
        self.currently_visualizing = "coordinates"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_coordinates2D(self.data_coordinates)
    def view_stims(self):
        self.currently_visualizing = "stims"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_stims
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.plot_widget.preview_dataset(preview_data)
    def view_behavior(self):
        self.currently_visualizing = "behavior"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_behavior
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.plot_widget.preview_dataset(preview_data)

    ## Edit buttons
    def edit_transpose(self):
        to_edit = self.currently_visualizing
        if to_edit == "dFFo":
            self.data_dFFo = self.data_dFFo.T
            self.view_dFFo()
        elif to_edit == "neuronal_activity":
            self.data_neuronal_activity = self.data_neuronal_activity.T
            self.view_neuronal_activity()
        elif to_edit == "coordinates":
            self.data_coordinates = self.data_coordinates.T
            self.view_coordinates()
        elif to_edit == "stims":
            self.data_stims = self.data_stims.T
            self.view_stims()
        elif to_edit == "behavior":
            self.data_behavior = self.data_behavior.T
            self.view_behavior()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()  