import sys
import h5py
import os
import csv
import scipy.io 
import math
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QVBoxLayout, QVBoxLayout
from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QTextCursor, QDoubleValidator

from data.load_data import FileTreeModel
from data.assign_data import assign_data_from_file

import utils.metrics as metrics

from gui.MatplotlibWidget import MatplotlibWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matlab.engine

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

        ## Ensembles visualizer
        self.ensvis_btn_svd.clicked.connect(self.vis_ensembles_svd)
        self.ensvis_btn_pca.clicked.connect(self.vis_ensembles_pca)
        self.envis_slide_selectedens.valueChanged.connect(self.update_ensemble_visualization)

        ## Performance
        self.performance_btn_compare.clicked.connect(self.performance_compare)

        # Store analysis results
        self.results = {}
        

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
    
    def validate_needed_data(self, needed_data):
        valid_data = True
        for req in needed_data:
            if not hasattr(self, req):
                valid_data = False
        return valid_data
        
    def format_nums_to_string(self, numbers_list):
        txt = f""
        for member_id in range(len(numbers_list)):
            txt += f"{numbers_list[member_id]}, " if member_id < len(numbers_list)-1 else f"{numbers_list[member_id]}"
        return txt

    ## Identify the tab changes
    def onTabChange(self, index):
        if index > 0: # SVD tab
            if hasattr(self, "data_neuronal_activity"):
                self.lbl_sdv_spikes_selected.setText(f"Loaded")
                self.lbl_pca_spikes_selected.setText(f"Loaded")
            else:
                self.lbl_sdv_spikes_selected.setText(f"Nothing selected")
                self.lbl_pca_spikes_selected.setText(f"Nothing selected")

            # Validate data for SVD
            needed_data = ["data_neuronal_activity"]
            self.btn_svd_run.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for PCA
            needed_data = ["data_neuronal_activity"]
            self.btn_run_pca.setEnabled(self.validate_needed_data(needed_data))

    ## Set variables from input file
    def set_dFFo(self):
        data_dFFo = assign_data_from_file(self)
        self.data_dFFo = data_dFFo
        neus, frames = data_dFFo.shape
        self.btn_clear_dFFo.setEnabled(True)
        self.btn_view_dFFo.setEnabled(True)
        self.lbl_dffo_select.setText("Assigned")
        self.lbl_dffo_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set dFFo dataset - Identified {neus} cells and {frames} time points. Please, verify the data preview.", msg_type="complete")
        self.view_dFFo()
    def set_neuronal_activity(self):
        data_neuronal_activity = assign_data_from_file(self)
        self.data_neuronal_activity = data_neuronal_activity
        self.cant_neurons, self.cant_timepoints = data_neuronal_activity.shape
        self.btn_clear_neuronal_activity.setEnabled(True)
        self.btn_view_neuronal_activity.setEnabled(True)
        self.lbl_neuronal_activity_select.setText("Assigned")
        self.lbl_neuronal_activity_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Binary Neuronal Activity dataset - Identified {self.cant_neurons} cells and {self.cant_timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_neuronal_activity()
    def set_coordinates(self):
        data_coordinates = assign_data_from_file(self)
        self.data_coordinates = data_coordinates[:, 0:2]
        neus, dims = self.data_coordinates.shape
        self.btn_clear_coordinates.setEnabled(True)
        self.btn_view_coordinates.setEnabled(True)
        self.lbl_coordinates_select.setText("Assigned")
        self.lbl_coordinates_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Coordinates dataset - Identified {neus} cells and {dims} dimentions. Please, verify the data preview.", msg_type="complete")
        self.view_coordinates()
    def set_stims(self):
        data_stims = assign_data_from_file(self)
        self.data_stims = data_stims
        stims, timepoints = data_stims.shape
        self.btn_clear_stim.setEnabled(True)
        self.btn_view_stim.setEnabled(True)
        self.lbl_stim_select.setText("Assigned")
        self.lbl_stim_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Stimuli dataset - Identified {stims} stims and {timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_stims()
    def set_behavior(self):
        data_behavior = assign_data_from_file(self)
        self.data_behavior = data_behavior
        behaviors, timepoints = data_behavior.shape
        self.btn_clear_behavior.setEnabled(True)
        self.btn_view_behavior.setEnabled(True)
        self.lbl_behavior_select.setText("Assigned")
        self.lbl_behavior_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Behavior dataset - Identified {behaviors} behaviors and {timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_behavior()
    
    ## Clear variables 
    def clear_dFFo(self):
        delattr(self, "data_dFFo")
        self.btn_clear_dFFo.setEnabled(False)
        self.btn_view_dFFo.setEnabled(False)
        self.lbl_dffo_select.setText("Nothing")
        self.lbl_dffo_select_name.setText("")
        self.update_console_log(f"Deleted dFFo dataset", msg_type="complete")       
    def clear_neuronal_activity(self):
        delattr(self, "data_neuronal_activity")
        self.btn_clear_neuronal_activity.setEnabled(False)
        self.btn_view_neuronal_activity.setEnabled(False)
        self.lbl_neuronal_activity_select.setText("Nothing")
        self.lbl_neuronal_activity_select_name.setText("")
        self.update_console_log(f"Deleted Binary Neuronal Activity dataset", msg_type="complete")
    def clear_coordinates(self):
        delattr(self, "data_coordinates")
        self.btn_clear_coordinates.setEnabled(False)
        self.btn_view_coordinates.setEnabled(False)
        self.lbl_coordinates_select.setText("Nothing")
        self.lbl_coordinates_select_name.setText("")
        self.update_console_log(f"Deleted Coordinates dataset", msg_type="complete")
    def clear_stims(self):
        delattr(self, "data_stims")
        self.btn_clear_stim.setEnabled(False)
        self.btn_view_stim.setEnabled(False)
        self.lbl_stim_select.setText("Nothing")
        self.lbl_stim_select_name.setText("")
        self.update_console_log(f"Deleted Stimuli dataset", msg_type="complete")
    def clear_behavior(self):
        delattr(self, "data_behavior")
        self.btn_clear_behavior.setEnabled(False)
        self.btn_view_behavior.setEnabled(False)
        self.lbl_behavior_select.setText("Nothing")
        self.lbl_behavior_select_name.setText("")
        self.update_console_log(f"Deleted Behavior dataset", msg_type="complete")

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
            self.update_console_log(f"Updated dFFo dataset. Please, verify the data preview.", msg_type="complete")
            self.view_dFFo()
        elif to_edit == "neuronal_activity":
            self.data_neuronal_activity = self.data_neuronal_activity.T
            self.cant_neurons = self.data_neuronal_activity.shape[0]
            self.cant_timepoints = self.data_neuronal_activity.shape[1]
            self.update_console_log(f"Updated Binary Neuronal Activity dataset. Please, verify the data preview.", msg_type="complete")
            self.view_neuronal_activity()
        elif to_edit == "coordinates":
            self.data_coordinates = self.data_coordinates.T
            self.update_console_log(f"Updated Coordinates dataset. Please, verify the data preview.", msg_type="complete")
            self.view_coordinates()
        elif to_edit == "stims":
            self.data_stims = self.data_stims.T
            self.update_console_log(f"Updated Stims dataset. Please, verify the data preview.", msg_type="complete")
            self.view_stims()
        elif to_edit == "behavior":
            self.data_behavior = self.data_behavior.T
            self.update_console_log(f"Updated Behavior dataset. Please, verify the data preview.", msg_type="complete")
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
        #data = self.data_coordinates
        data = np.zeros((self.cant_neurons,2))
        coords_foo = matlab.double(data.tolist())

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

        #print([val_pks, val_scut, val_hcut, val_statecut, val_idtfd])

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/SVD'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        self.update_console_log("Performing SVD...")
        answer = eng.Stoixeion(spikes, coords_foo, FFo, val_pks, val_scut, val_hcut, val_statecut, val_idtfd)
        self.update_console_log("Done.", "complete")

        # Create plots for every result
        #keys_list = list(answer.keys())
        #print(keys_list)

        # Similarity map
        simmap = np.array(answer['S_index_ti'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_similaritymap')
        self.plot_widget.preview_dataset(simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='jet', aspect='equal')
        # Binary similarity map
        bin_simmap = np.array(answer['S_indexp'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_binarysimmap')
        self.plot_widget.preview_dataset(bin_simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='gray', aspect='equal')
        # Singular values plot
        singular_vals = np.diagonal(np.array(answer['S_svd']))
        num_state = int(answer['num_state'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_singularvalues')
        self.plot_widget.plot_singular_values(singular_vals, num_state)

        # Components from the descomposition
        singular_vals = np.array(answer['svd_sig'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_components')
        rows = math.ceil(math.sqrt(num_state))
        cols = math.ceil(num_state / rows)
        self.plot_widget.set_subplots(rows, cols)
        for state_idx in range(num_state):
            curent_comp = singular_vals[:, :, state_idx]
            row = state_idx // cols
            col = state_idx % cols
            self.plot_widget.plot_states_from_svd(curent_comp, state_idx, row, col)
            
        # Plot the ensembles timecourse
        Pks_Frame = np.array(answer['Pks_Frame'])
        sec_Pk_Frame = np.array(answer['sec_Pk_Frame'])
        ensembles_timecourse = np.zeros((num_state, self.cant_timepoints))
        framesActiv = Pks_Frame.shape[1]
        for it in range(framesActiv):
            currentFrame = int(Pks_Frame[0, it])
            currentEns = int(sec_Pk_Frame[it, 0])
            if currentEns != 0: 
                ensembles_timecourse[currentEns-1, currentFrame-1] = 1
        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_timecourse')
        self.plot_widget.plot_ensembles_timecourse(ensembles_timecourse)

        # Save the results
        self.results['svd'] = {}
        self.results['svd']['timecourse'] = ensembles_timecourse
        self.results['svd']['ensembles_cant'] = ensembles_timecourse.shape[0]
        Pools_coords = np.array(answer['Pools_coords'])
        # Identify the neurons that belongs to each ensamble
        neurons_in_ensembles = np.zeros((self.results['svd']['ensembles_cant'], self.cant_neurons))
        for ens in range(self.results['svd']['ensembles_cant']):
            cells_in_ens = Pools_coords[:, :, ens]
            for neu in range(self.cant_neurons):
                cell_id = int(cells_in_ens[neu][2])
                if cell_id == 0:
                    break
                else:
                    neurons_in_ensembles[ens, cell_id-1] = 1
        self.results['svd']['neus_in_ens'] = neurons_in_ensembles
        self.we_have_results()

        self.plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_cellsinens')
        self.plot_widget.plot_ensembles_timecourse(neurons_in_ensembles, xlabel="Neuron")

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
        #keys_list = list(answer.keys())
        #print(keys_list)

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

        # Save the results
        self.results['pca'] = {}
        self.results['pca']['timecourse'] = np.array(answer["sel_ensmat_out"])
        self.results['pca']['ensembles_cant'] = self.results['pca']['timecourse'].shape[0]
        self.results['pca']['neus_in_ens'] = np.array(answer["sel_core_cells"]).T
        self.we_have_results()

        # Plot ensembles timecourse
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_timecourse')
        self.plot_widget.plot_ensembles_timecourse(self.results['pca']['timecourse'])

        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_cellsinens')
        self.plot_widget.plot_ensembles_timecourse(self.results['pca']['neus_in_ens'])

    def we_have_results(self):
        self.performance_btn_compare.setEnabled(True)
        for analysis_name in self.results.keys():
            if analysis_name == 'svd':
                self.ensvis_btn_svd.setEnabled(True)
                self.performance_check_svd.setEnabled(True)
            elif analysis_name == 'pca':
                self.ensvis_btn_pca.setEnabled(True)
                self.performance_check_pca.setEnabled(True)

    def vis_ensembles_svd(self):
        if hasattr(self, "results"):
            self.ensemble_currently_shown = "svd"
            self.update_analysis_results()
    
    def vis_ensembles_pca(self):
        if hasattr(self, "results"):
            self.ensemble_currently_shown = "pca"
            self.update_analysis_results()

    def update_analysis_results(self):
        self.initialize_ensemble_view()
        self.update_ensvis_allbinary()
        self.update_ensvis_allens()
        if hasattr(self, "data_dFFo"):
            self.update_ensvis_alldFFo()
        if hasattr(self, "data_coordinates"):
            self.update_ensvis_allcoords()

    def initialize_ensemble_view(self):
        curr_show = self.ensemble_currently_shown 
        self.ensvis_lbl_currently.setText(f"{curr_show}".upper())
        # Show the number of identifies ensembles
        self.ensvis_edit_numens.setText(f"{self.results[curr_show]['ensembles_cant']}")
        # Activate the slider
        self.envis_slide_selectedens.setEnabled(True)
        # Update the slider used to select the ensemble to visualize
        self.envis_slide_selectedens.setMinimum(1)   # Set the minimum value
        self.envis_slide_selectedens.setMaximum(self.results[curr_show]['ensembles_cant']) # Set the maximum value
        self.envis_slide_selectedens.setValue(1)
        self.ensvis_lbl_currentens.setText(f"{1}")
        self.update_ensemble_visualization(1)

    def update_ensemble_visualization(self, value):
        curr_analysis = self.ensemble_currently_shown
        curr_ensemble = value
        self.ensvis_lbl_currentens.setText(f"{curr_ensemble}")

        # Get the members of this ensemble
        members = []
        ensemble = self.results[curr_analysis]['neus_in_ens'][value-1,:]
        members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
        members_txt = self.format_nums_to_string(members)
        self.ensvis_edit_members.setText(members_txt)

        # Get the exclusive members of this ensemble
        ens_mat = self.results[curr_analysis]['neus_in_ens']
        mask_e = ensemble == 1
        sum_mask = np.sum(ens_mat, axis=0)
        exc_elems = [cell+1 for cell in range(len(mask_e)) if mask_e[cell] and sum_mask[cell] == 1]
        exclusive_txt = self.format_nums_to_string(exc_elems)
        self.ensvis_edit_exclusive.setText(exclusive_txt)

        # Timepoints of activation
        ensemble_timecourse = self.results[curr_analysis]['timecourse'][curr_ensemble-1,:]
        ens_timepoints = [frame+1 for frame in range(len(ensemble_timecourse)) if ensemble_timecourse[frame]]
        ens_timepoints_txt = self.format_nums_to_string(ens_timepoints)
        self.ensvis_edit_timepoints.setText(ens_timepoints_txt)

        idx_corrected_members = [idx-1 for idx in members]
        idx_corrected_exclusive = [idx-1 for idx in exc_elems]
        
        if hasattr(self, "data_coordinates"):
            self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_map')
            self.plot_widget.plot_coordinates2D_highlight(self.data_coordinates, idx_corrected_members, idx_corrected_exclusive)

        if hasattr(self, "data_dFFo"):
            self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_raster')
            dFFo_ens = self.data_dFFo[idx_corrected_members, :]
            self.plot_widget.plot_ensemble_dFFo(dFFo_ens, idx_corrected_members, ensemble_timecourse)

    def update_ensvis_alldFFo(self):
        curr_analysis = self.ensemble_currently_shown
        cant_ensembles = self.results[curr_analysis]['ensembles_cant']

        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_alldffo')
        self.plot_widget.set_subplots(1, cant_ensembles)
        for current_ens in range(cant_ensembles):
            # Create subplot for each core
            ensemble = self.results[curr_analysis]['neus_in_ens'][current_ens,:]
            members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
            idx_corrected_members = [idx-1 for idx in members]
            dFFo_ens = self.data_dFFo[idx_corrected_members, :]
            self.plot_widget.plot_all_dFFo(dFFo_ens, idx_corrected_members, current_ens)

    def update_ensvis_allcoords(self):
        curr_analysis = self.ensemble_currently_shown
        cant_ensembles = self.results[curr_analysis]['ensembles_cant']
        
        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allspatial')
        
        rows = math.ceil(math.sqrt(cant_ensembles))
        cols = math.ceil(cant_ensembles / rows)
        self.plot_widget.set_subplots(rows, cols)

        for current_ens in range(cant_ensembles):
            row = current_ens // cols
            col = current_ens % cols
            # Create subplot for each core
            ensemble = self.results[curr_analysis]['neus_in_ens'][current_ens,:]
            members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
            idx_corrected_members = [idx-1 for idx in members]

            ens_mat = self.results[curr_analysis]['neus_in_ens']
            mask_e = ensemble == 1
            sum_mask = np.sum(ens_mat, axis=0)
            exc_elems = [cell+1 for cell in range(len(mask_e)) if mask_e[cell] and sum_mask[cell] == 1]
            idx_corrected_exclusive = [idx-1 for idx in exc_elems]
            
            self.plot_widget.plot_all_coords(self.data_coordinates, idx_corrected_members, idx_corrected_exclusive, row, col)

    def update_ensvis_allbinary(self):
        curr_analysis = self.ensemble_currently_shown
        cant_ensembles = self.results[curr_analysis]['ensembles_cant']
        
        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allbinary')
        self.plot_widget.set_subplots(1, cant_ensembles)
        for current_ens in range(cant_ensembles):
            ensemble = self.results[curr_analysis]['neus_in_ens'][current_ens,:]
            members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
            idx_corrected_members = [idx-1 for idx in members]
            activity = self.data_neuronal_activity[idx_corrected_members, :] == 0
            self.plot_widget.plot_all_binary(activity, members, current_ens, current_ens)

    def update_ensvis_allens(self):
        curr_analysis = self.ensemble_currently_shown
        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allens')
        self.plot_widget.plot_ensembles_timecourse(self.results[curr_analysis]['timecourse'])

    def performance_compare(self):
        methods_to_compare = []
        if self.performance_check_svd.isChecked():
            methods_to_compare.append("svd")
        if self.performance_check_pca.isChecked():
            methods_to_compare.append("pca")
        if self.performance_check_ica.isChecked():
            methods_to_compare.append("ica")
        if self.performance_check_sgc.isChecked():
            methods_to_compare.append("sgc")

        cant_methods_compare = len(methods_to_compare)

        if hasattr(self, "data_stims"):
            self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrstims')
            plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
            self.plot_widget.set_subplots(1, plot_colums)

            for m_idx, method in enumerate(methods_to_compare):
                # Calculate correlation with stimuli
                timecourse = self.results[method]['timecourse']
                stims = self.data_stims
                correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
                self.plot_widget.plot_perf_correlations_ens_stim(correlation, m_idx, title=f"{method}".upper())
        
        # Plot the correlation of cells between themselves
        self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrcells')
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        self.plot_widget.set_subplots(1, plot_colums)
        for m_idx, method in enumerate(methods_to_compare):
            cant_ens = self.results[method]['ensembles_cant']
            for ens_id, ens in enumerate(self.results[method]['neus_in_ens']):
                members = [c_idx for c_idx in range(len(ens)) if ens[c_idx] == 1]
                activity_neus_in_ens = self.data_neuronal_activity[members, :]
                correlation = metrics.compute_correlation_inside_ensemble(activity_neus_in_ens)

                self.plot_widget.plot_perf_correlations_ens_stim(correlation, m_idx, title=f"{method}".upper())
        
                
                #neus_in_ens = self.results[method]['neus_in_ens'][m_idx, :]

        # Get the algorithms to compare


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()  