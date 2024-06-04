import sys
import h5py
import os
import csv
import scipy.io 
import math
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QVBoxLayout, QGridLayout, QGraphicsScene
from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QTextCursor, QDoubleValidator

from data.load_data import FileTreeModel
from data.assign_data import assign_data_from_file
from plots.plots import plot_raster
from gui.MatplotlibWidget import MatplotlibWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matlab.engine
from matlab import double as matlab_double

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

        ## Numeric validator
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        double_validator.setRange(-1000000.0, 1000000.0, 10)

        # Set validators to QLineEdit widgets
        # For the SVD analysis
        self.svd_edit_pks.setValidator(double_validator)
        self.svd_edit_scut.setValidator(double_validator)
        self.svd_edit_hcut.setValidator(double_validator)
        self.svd_edit_statecut.setValidator(double_validator)
        # For the PCA analysis
        self.pca_edit_dc.setValidator(double_validator)
        self.pca_edit_npcs.setValidator(double_validator)
        self.pca_edit_minspk.setValidator(double_validator)
        self.pca_edit_nsur.setValidator(double_validator)
        self.pca_edit_prct.setValidator(double_validator)
        self.pca_edit_centthr.setValidator(double_validator)
        self.pca_edit_innercorr.setValidator(double_validator)
        self.pca_edit_minsize.setValidator(double_validator)
        

        ## SVD analysis
        self.btn_svd_run.clicked.connect(self.run_svd)

        ## PCA analysis
        self.btn_run_pca.clicked.connect(self.run_PCA)
        

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

            needed_data = ["data_neuronal_activity", "data_coordinates"]
            valid_data = True
            for req in needed_data:
                if not hasattr(self, req):
                    valid_data = False
            self.btn_svd_run.setEnabled(valid_data)

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

    ## Visualize variables from input file
    def view_dFFo(self):
        self.currently_visualizing = "dFFo"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_dataset(self.data_dFFo, ylabel='Neuron')
    def view_neuronal_activity(self):
        self.currently_visualizing = "neuronal_activity"
        self.btn_edit_transpose.setEnabled(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_dataset(self.data_neuronal_activity==0, ylabel='Neuron', cmap='gray')
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
        self.plot_widget.preview_dataset(preview_data==0, ylabel='Stim', cmap='gray')
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
        
    def add_matplotlib_widgets_to_tab(self, n, tab_index):
        # Access the tab at index tab_index
        tab = self.tabWidget.widget(tab_index)
        # Create a layout for the tab
        layout = QVBoxLayout(tab)
        # Create and add n MatplotlibWidgets to the layout
        for i in range(n):
            mw = MatplotlibWidget()
            mw.setObjectName(f"svd_plot_components_{i+1}")
            layout.addWidget(mw)
        # Set the layout for the tab
        tab.setLayout(layout)
        
    def run_svd(self):
        # Prepare data
        data = self.data_neuronal_activity
        spikes = matlab.double(data.tolist())
        data = self.data_coordinates
        coords = matlab.double(data.tolist())

        if hasattr(self, 'data_dFFo'):
            data = self.data_dFFo
        else:
            data = np.array([])
        FFo = matlab.double(data.tolist())

        input_value = self.svd_edit_pks.text()
        val_pks = np.array([float(input_value)]) if len(input_value) > 0 else np.array([]) 
        input_value = self.svd_edit_scut.text()
        val_scut = np.array([float(input_value)]) if len(input_value) > 0 else np.array([]) 
        input_value = self.svd_edit_hcut.text()
        val_hcut = float(input_value) if len(input_value) > 0 else 0.20
        input_value = self.svd_edit_statecut.text()
        val_statecut = float(input_value) if len(input_value) > 0 else 6
        val_idtfd = self.svd_check_tfidf.isChecked()

        print([val_pks, val_scut, val_hcut, val_statecut, val_idtfd])

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/SVD'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        self.update_console_log("Performing SVD...")
        answer = eng.Stoixeion(spikes, coords, FFo, val_pks, val_scut, val_hcut, val_statecut, val_idtfd)
        self.update_console_log("Done.", "complete")

        # Create plots for every result
        keys_list = list(answer.keys())
        print(keys_list)

        # Similarity map
        simmap = np.array(answer['S_index_ti'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_similaritymap')
        self.plot_widget.preview_dataset(simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='jet', aspect='equal')
        # Binary similarity map
        bin_simmap = np.array(answer['S_indexp'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_binarysimmap')
        self.plot_widget.preview_dataset(bin_simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='gray', aspect='equal')
        # Singular values plot
        singular_vals = np.array(answer['S_svd'])
        num_state = int(answer['num_state'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_singularvalues')
        self.plot_widget.plot_singular_values(singular_vals, num_state)

        # Components from the descomposition
        singular_vals = np.array(answer['svd_sig'])
        tab_index = 3 # Tab index within the tabs of SVD results
        tab = self.tabWidget.widget(tab_index)
        layout = QGridLayout()
        rows = math.ceil(math.sqrt(num_state))
        cols = math.ceil(num_state / rows)
        for state_idx in range(num_state):
            curent_comp = singular_vals[:, :, state_idx]
            row = state_idx // cols
            col = state_idx % cols
            mw = MatplotlibWidget()
            mw.setObjectName(f"svd_plot_components_{state_idx+1}")
            layout.addWidget(mw, row, col)
            mw.plot_states_from_svd(curent_comp, state_idx)
        tab.setLayout(layout)
            
        # Plot the ensembles timecourse
        Pks_Frame = np.array(answer['Pks_Frame'])
        sec_Pk_Frame = np.array(answer['sec_Pk_Frame'])
        frames = self.data_neuronal_activity.shape[1]
        ensembles_timecourse = np.zeros((num_state, frames))
        framesActiv = Pks_Frame.shape[1]
        for it in range(framesActiv):
            currentFrame = int(Pks_Frame[0, it])
            currentEns = int(sec_Pk_Frame[it, 0])
            if currentEns != 0: 
                ensembles_timecourse[currentEns-1, currentFrame-1] = 1
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_timecourse')
        self.plot_widget.plot_ensembles_timecourse(ensembles_timecourse)

    def run_PCA(self):
        data = self.data_neuronal_activity
        raster = matlab.double(data.tolist())

        input_value = self.pca_edit_dc.text()
        dc = float(input_value) if len(input_value) > 0 else 0.02
        input_value = self.pca_edit_npcs.text()
        npcs = float(input_value) if len(input_value) > 0 else 3
        input_value = self.pca_edit_minspk.text()
        minspk = float(input_value) if len(input_value) > 0 else 3
        input_value = self.pca_edit_nsur.text()
        nsur = float(input_value) if len(input_value) > 0 else 1000
        input_value = self.pca_edit_prct.text()
        prct = float(input_value) if len(input_value) > 0 else 99.9
        input_value = self.pca_edit_centthr.text()
        cent_thr = float(input_value) if len(input_value) > 0 else 99.9
        input_value = self.pca_edit_innercorr.text()
        inner_corr = float(input_value) if len(input_value) > 0 else 5
        input_value = self.pca_edit_minsize.text()
        minsize = float(input_value) if len(input_value) > 0 else 3

        # Pack data
        pars = {
            'dc': dc,
            'npcs': npcs,
            'minspk': minspk,
            'nsur': nsur,
            'prct': prct,
            'cent_thr': cent_thr,
            'inner_corr': inner_corr,
            'minsize': minsize
        }

        pars_matlab = {key: matlab.double([value]) if isinstance(value, (int, float)) else value for key, value in pars.items()}

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/NeuralEnsembles'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        self.update_console_log("Performing PCA...")
        answer = eng.raster2ens_by_density(raster, pars_matlab)
        self.update_console_log("Done.", "complete")

        # Create plots for every result
        keys_list = list(answer.keys())
        print(keys_list)

        ## Plot the results
        ## Plot the eigs
        eigs = np.array(answer['exp_var'])
        seleig = int(pars['npcs'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_eigs')
        self.plot_widget.plot_eigs(eigs, seleig)

        # Plot the PCA
        pcs = np.array(answer['pcs'])
        labels = np.array(answer['labels'])[0]
        Nens = int(answer['Nens'])
        ens_cols = plt.cm.tab10(range(Nens * 2))
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_pca')
        self.plot_widget.plot_pca(pcs, ens_labs=labels, ens_cols = ens_cols)

        # Plot the rhos vs deltas
        rho = np.array(answer['rho'])
        delta = np.array(answer['delta'])
        cents = np.array(answer['cents'])
        predbounds = np.array(answer['predbounds'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_rhodelta')
        self.plot_widget.plot_delta_rho(rho, delta, cents, predbounds, ens_cols)
        
        # Plot corr(n,e)
        ens_cel_corr = np.array(answer['ens_cel_corr'])
        ens_cel_corr_min = np.min(ens_cel_corr)
        ens_cel_corr_max = np.max(ens_cel_corr)
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_corrne')
        self.plot_widget.plot_core_cells(ens_cel_corr, [ens_cel_corr_min, ens_cel_corr_max])

        # Plot core cells
        core_cells = np.array(answer['core_cells'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_corecells')
        self.plot_widget.plot_core_cells(core_cells, [-1, 1])

        # Plot core cells
        ens_corr = np.array(answer["ens_corr"])[0]
        corr_thr = np.array(answer["corr_thr"])
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_innerens')
        self.plot_widget.plot_ens_corr(ens_corr, corr_thr, ens_cols)

        
    

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()  