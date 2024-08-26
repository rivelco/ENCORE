import sys
import h5py
import os
import scipy.io 
import math
import numpy as np
import scipy.stats as stats
from time import sleep

from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt6.QtWidgets import QTableWidgetItem, QColorDialog

from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QTextCursor, QDoubleValidator

from data.load_data import FileTreeModel
from data.assign_data import assign_data_from_file

import utils.metrics as metrics

from gui.MatplotlibWidget import MatplotlibWidget

import matplotlib.pyplot as plt

import matlab.engine

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        super(MainWindow, self).__init__()
        loadUi("gui/MainWindow.ui", self)
        self.setWindowTitle('Ensembles GUI')

        # Initialize the GUI
        self.reset_gui()
        ## Browse files
        self.browseFile.clicked.connect(self.browse_files)
        # Connect the clicked signal of the tree view to a slot
        self.tree_view.clicked.connect(self.item_clicked)

        ## Identify change of tab
        self.main_tabs.currentChanged.connect(self.main_tabs_change)

        ## Set the activity variables
        self.btn_set_dFFo.clicked.connect(self.set_dFFo)
        self.btn_set_neuronal_activity.clicked.connect(self.set_neuronal_activity)
        self.btn_set_coordinates.clicked.connect(self.set_coordinates)
        self.btn_set_stim.clicked.connect(self.set_stims)
        self.btn_set_cells.clicked.connect(self.set_cells)
        self.btn_set_behavior.clicked.connect(self.set_behavior)

        ## Set the clear buttons
        self.btn_clear_dFFo.clicked.connect(self.clear_dFFo)
        self.btn_clear_neuronal_activity.clicked.connect(self.clear_neuronal_activity)
        self.btn_clear_coordinates.clicked.connect(self.clear_coordinates)
        self.btn_clear_stim.clicked.connect(self.clear_stims)
        self.btn_clear_cells.clicked.connect(self.clear_cells)
        self.btn_clear_behavior.clicked.connect(self.clear_behavior)

        ## Set the preview buttons
        self.btn_view_dFFo.clicked.connect(self.view_dFFo)
        self.btn_view_neuronal_activity.clicked.connect(self.view_neuronal_activity)
        self.btn_view_coordinates.clicked.connect(self.view_coordinates)
        self.btn_view_stim.clicked.connect(self.view_stims)
        self.btn_view_cells.clicked.connect(self.view_cells)
        self.btn_view_behavior.clicked.connect(self.view_behavior)

        ## Edit actions
        self.btn_edit_transpose.clicked.connect(self.edit_transpose)
        self.edit_btn_bin.clicked.connect(self.edit_bin)
        self.edit_btn_trim.clicked.connect(self.edit_trimmatrix)
        self.btn_set_labels.clicked.connect(self.varlabels_save)
        self.btn_clear_labels.clicked.connect(self.varlabels_clear)

        ## Set default values for analysis
        defaults = {
            'pks': 3,
            'scut': 0.22,
            'hcut': 0.22,
            'state_cut': 6,
            'csi_start': 0.01,
            'csi_step': 0.01,
            'csi_end': 0.1,
            'tf_idf_norm': True
        }
        self.svd_defaults = defaults
        defaults = {
            'dc': 0.01,
            'npcs': 3,
            'minspk': 3,
            'nsur': 1000,
            'prct': 99.9,
            'cent_thr': 99.9,
            'inner_corr': 5,
            'minsize': 3
        }
        self.pca_defaults = defaults
        defaults = {
            'threshold': {
                'method': 'MarcenkoPastur',
                'permutations_percentile': 95,
                'number_of_permutations': 20
            },
            'Patterns': {
                'method': 'ICA',
                'number_of_iterations': 500
            }
        }
        self.ica_defaults = defaults
        defaults = {
            'network_bin': 1,
            'network_iterations': 1000,
            'network_significance': 0.05,
            'coactive_neurons_threshold': 2,
            'clustering_range_start': 3,
            'clustering_range_end': 10,
            'clustering_fixed': 0,
            'iterations_ensemble': 1000,
            'parallel_processing': False,
            'file_log': ''
        }
        self.x2p_defaults = defaults

        ## Numeric validator
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        # For the edit options
        double_validator.setRange(0, 1000000.0, 10)
        self.edit_edit_binsize.setValidator(double_validator)
        self.edit_edit_xstart.setValidator(double_validator)
        self.edit_edit_xend.setValidator(double_validator)
        self.edit_edit_ystart.setValidator(double_validator)
        self.edit_edit_yend.setValidator(double_validator)

        double_validator.setRange(-1000000.0, 1000000.0, 10)
        ## Set validators to QLineEdit widgets
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
        # For ICA analysis
        self.ica_edit_perpercentile.setValidator(double_validator)
        self.ica_edit_percant.setValidator(double_validator)
        self.ica_edit_iterations.setValidator(double_validator)
        # For X2P analysis
        self.x2p_edit_bin.setValidator(double_validator)
        self.x2p_edit_iterations.setValidator(double_validator)
        self.x2p_edit_significance.setValidator(double_validator)
        self.x2p_edit_threshold.setValidator(double_validator)
        self.x2p_edit_rangestart.setValidator(double_validator)
        self.x2p_edit_rangeend.setValidator(double_validator)
        self.x2p_edit_fixed.setValidator(double_validator)
        self.x2p_edit_itensemble.setValidator(double_validator)

        ## SVD analysis
        self.svd_btn_defaults.clicked.connect(self.load_defaults_svd)
        self.btn_svd_run.clicked.connect(self.run_svd)
        ## PCA analysis
        self.pca_btn_defaults.clicked.connect(self.load_defaults_pca)
        self.btn_run_pca.clicked.connect(self.run_PCA)
        ## ICA analysis
        self.ica_btn_defaults.clicked.connect(self.load_defaults_ica)
        self.btn_run_ica.clicked.connect(self.run_ICA)
        ## X2P analysis
        self.x2p_btn_defaults.clicked.connect(self.load_defaults_x2p)
        self.btn_run_x2p.clicked.connect(self.run_x2p)

        ## Ensembles visualizer
        self.ensvis_tabs.currentChanged.connect(self.ensvis_tabchange)
        self.ensvis_btn_svd.clicked.connect(self.vis_ensembles_svd)
        self.ensvis_btn_pca.clicked.connect(self.vis_ensembles_pca)
        self.ensvis_btn_ica.clicked.connect(self.vis_ensembles_ica)
        self.ensvis_btn_x2p.clicked.connect(self.vis_ensembles_x2p)
        self.envis_slide_selectedens.valueChanged.connect(self.update_ensemble_visualization)
        self.ensvis_check_onlyens.stateChanged.connect(self.update_ens_vis_coords)
        self.ensvis_check_onlycont.stateChanged.connect(self.update_ens_vis_coords)
        self.ensvis_check_cellnum.stateChanged.connect(self.update_ens_vis_coords)

        # Ensemble compare
        self.enscomp_slider_svd.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_pca.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_ica.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_x2p.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_visopts_setneusize.clicked.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_btn_color_svd.clicked.connect(self.get_color_svd)
        self.enscomp_btn_color_pca.clicked.connect(self.get_color_pca)
        self.enscomp_btn_color_ica.clicked.connect(self.get_color_ica)
        self.enscomp_btn_color_x2p.clicked.connect(self.get_color_x2p)
        self.enscomp_check_coords_svd.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_coords_pca.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_coords_ica.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_coords_x2p.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_visopts_showcells.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_ens_svd.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_neus_svd.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_ens_pca.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_neus_pca.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_ens_ica.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_neus_ica.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_ens_x2p.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_neus_x2p.stateChanged.connect(self.ensembles_compare_update_ensembles)

        ## Performance
        self.performance_tabs.currentChanged.connect(self.performance_tabchange)
        self.performance_check_svd.stateChanged.connect(self.performance_check_change)
        self.performance_check_pca.stateChanged.connect(self.performance_check_change)
        self.performance_check_ica.stateChanged.connect(self.performance_check_change)
        self.performance_check_x2p.stateChanged.connect(self.performance_check_change)
        self.performance_btn_compare.clicked.connect(self.performance_compare)

        # Saving
        self.save_btn_save.clicked.connect(self.save_results)
        
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

    def reset_gui(self):
        # Delete all previous results
        self.results = {}
        self.params = {}
        self.varlabels = {}
        self.tempvars = {}
        
        # Initialize buttons
        self.btn_svd_run.setEnabled(False)
        self.btn_run_pca.setEnabled(False)
        self.btn_run_ica.setEnabled(False)
        self.btn_run_x2p.setEnabled(False)

        self.ensvis_btn_svd.setEnabled(False)
        self.ensvis_btn_pca.setEnabled(False)
        self.ensvis_btn_ica.setEnabled(False)
        self.ensvis_btn_x2p.setEnabled(False)
        self.ensvis_btn_sgc.setEnabled(False)

        self.performance_check_svd.setEnabled(False)
        self.performance_check_pca.setEnabled(False)
        self.performance_check_ica.setEnabled(False)
        self.performance_check_x2p.setEnabled(False)
        self.performance_check_sgc.setEnabled(False)
        self.performance_btn_compare.setEnabled(False)

        # Save tab
        self.save_btn_save.setEnabled(False)

        # Clear the preview plots
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)

        # Clear the figures
        default_txt = "Perform the SVD analysis to see results"
        self.findChild(MatplotlibWidget, 'svd_plot_similaritymap').reset(default_txt)
        self.findChild(MatplotlibWidget, 'svd_plot_binarysimmap').reset(default_txt)
        self.findChild(MatplotlibWidget, 'svd_plot_singularvalues').reset(default_txt)
        self.findChild(MatplotlibWidget, 'svd_plot_components').reset(default_txt)
        self.findChild(MatplotlibWidget, 'svd_plot_timecourse').reset(default_txt)
        self.findChild(MatplotlibWidget, 'svd_plot_cellsinens').reset(default_txt)

        default_txt = "Perform the PCA analysis to see results"
        self.findChild(MatplotlibWidget, 'pca_plot_eigs').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_pca').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_rhodelta').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_corrne').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_corecells').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_innerens').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_timecourse').reset(default_txt)
        self.findChild(MatplotlibWidget, 'pca_plot_cellsinens').reset(default_txt)

        default_txt = "Perform the ICA analysis to see results"
        self.findChild(MatplotlibWidget, 'ica_plot_assemblys').reset(default_txt)
        self.findChild(MatplotlibWidget, 'ica_plot_activity').reset(default_txt)
        self.findChild(MatplotlibWidget, 'ica_plot_binary_patterns').reset(default_txt)
        self.findChild(MatplotlibWidget, 'ica_plot_binary_assemblies').reset(default_txt)

        default_txt = "Perform the Xsembles2P analysis to see results"
        self.findChild(MatplotlibWidget, 'x2p_plot_similarity').reset(default_txt)
        self.findChild(MatplotlibWidget, 'x2p_plot_epi').reset(default_txt)
        self.findChild(MatplotlibWidget, 'x2p_plot_onsemact').reset(default_txt)
        self.findChild(MatplotlibWidget, 'x2p_plot_offsemact').reset(default_txt)
        self.findChild(MatplotlibWidget, 'x2p_plot_activity').reset(default_txt)
        self.findChild(MatplotlibWidget, 'x2p_plot_onsemneu').reset(default_txt)
        self.findChild(MatplotlibWidget, 'x2p_plot_offsemneu').reset(default_txt)

        self.ensvis_edit_numens.setText("")
        self.envis_slide_selectedens.setMaximum(2)
        self.envis_slide_selectedens.setValue(1)
        self.ensvis_lbl_currentens.setText(f"{1}")
        self.ensvis_check_onlyens.setEnabled(False)
        self.ensvis_check_onlycont.setEnabled(False)
        self.ensvis_check_cellnum.setEnabled(False)
        self.ensvis_check_onlyens.setChecked(False)
        self.ensvis_check_onlycont.setChecked(False)
        self.ensvis_check_cellnum.setChecked(True)
        self.ensvis_edit_members.setText("")
        self.ensvis_edit_exclusive.setText("")
        self.ensvis_edit_timepoints.setText("")
        self.ensvis_tabs.setCurrentIndex(0)
        self.tempvars['ensvis_shown_results'] = False
        self.tempvars['ensvis_shown_tab1'] = False
        self.tempvars['ensvis_shown_tab2'] = False
        self.tempvars['ensvis_shown_tab3'] = False
        self.tempvars['ensvis_shown_tab4'] = False 

        # Ensembles compare
        self.enscomp_slider_svd.setEnabled(False)
        self.enscomp_slider_lbl_min_svd.setEnabled(False)
        self.enscomp_slider_lbl_max_svd.setEnabled(False)
        self.enscomp_slider_svd.setMinimum(1)
        self.enscomp_slider_svd.setMaximum(2)
        self.enscomp_slider_svd.setValue(1)
        self.enscomp_slider_lbl_min_svd.setText("1")
        self.enscomp_slider_lbl_max_svd.setText("1")
        self.enscomp_check_coords_svd.setEnabled(False)
        self.enscomp_check_ens_svd.setEnabled(False)
        self.enscomp_check_neus_svd.setEnabled(False)
        self.enscomp_btn_color_svd.setEnabled(False)
        self.enscomp_slider_pca.setEnabled(False)
        self.enscomp_slider_lbl_min_pca.setEnabled(False)
        self.enscomp_slider_lbl_max_pca.setEnabled(False)
        self.enscomp_slider_pca.setMinimum(1)
        self.enscomp_slider_pca.setMaximum(2)
        self.enscomp_slider_pca.setValue(1)
        self.enscomp_slider_lbl_min_pca.setText("1")
        self.enscomp_slider_lbl_max_pca.setText("1")
        self.enscomp_check_coords_pca.setEnabled(False)
        self.enscomp_check_ens_pca.setEnabled(False)
        self.enscomp_check_neus_pca.setEnabled(False)
        self.enscomp_btn_color_pca.setEnabled(False)
        self.enscomp_slider_ica.setEnabled(False)
        self.enscomp_slider_lbl_min_ica.setEnabled(False)
        self.enscomp_slider_lbl_max_ica.setEnabled(False)
        self.enscomp_slider_ica.setMinimum(1)
        self.enscomp_slider_ica.setMaximum(2)
        self.enscomp_slider_ica.setValue(1)
        self.enscomp_slider_lbl_min_ica.setText("1")
        self.enscomp_slider_lbl_max_ica.setText("1")
        self.enscomp_check_coords_ica.setEnabled(False)
        self.enscomp_check_ens_ica.setEnabled(False)
        self.enscomp_check_neus_ica.setEnabled(False)
        self.enscomp_btn_color_ica.setEnabled(False)
        self.enscomp_slider_x2p.setEnabled(False)
        self.enscomp_slider_lbl_min_x2p.setEnabled(False)
        self.enscomp_slider_lbl_max_x2p.setEnabled(False)
        self.enscomp_slider_x2p.setMinimum(1)
        self.enscomp_slider_x2p.setMaximum(2)
        self.enscomp_slider_x2p.setValue(1)
        self.enscomp_slider_lbl_min_x2p.setText("1")
        self.enscomp_slider_lbl_max_x2p.setText("1")
        self.enscomp_check_coords_x2p.setEnabled(False)
        self.enscomp_check_ens_x2p.setEnabled(False)
        self.enscomp_check_neus_x2p.setEnabled(False)
        self.enscomp_btn_color_x2p.setEnabled(False)
        if not hasattr(self, "data_stims"):
            self.enscomp_slider_stim.setEnabled(False)
            self.enscomp_slider_lbl_min_stim.setEnabled(False)
            self.enscomp_slider_lbl_max_stim.setEnabled(False)
            self.enscomp_slider_lbl_stim.setEnabled(False)
            self.enscomp_check_show_stim.setEnabled(False)
            self.enscomp_btn_color_stim.setEnabled(False)
        if not hasattr(self, "data_behavior"):
            self.enscomp_slider_behavior.setEnabled(False)
            self.enscomp_slider_lbl_min_behavior.setEnabled(False)
            self.enscomp_slider_lbl_max_behavior.setEnabled(False)
            self.enscomp_slider_lbl_behavior.setEnabled(False)
            self.enscomp_check_behavior_stim.setEnabled(False)
            self.enscomp_btn_color_behavior.setEnabled(False)

        self.enscom_colors = {
            'svd': 'red',
            'ica': 'blue',
            'pca': 'green',
            'x2p': 'orange'
        }
        self.enscomp_check_coords_svd.setChecked(True)
        self.enscomp_check_coords_pca.setChecked(True)
        self.enscomp_check_coords_ica.setChecked(True)
        self.enscomp_check_coords_x2p.setChecked(True)
        self.enscomp_check_ens_svd.setChecked(True)
        self.enscomp_check_ens_pca.setChecked(True)
        self.enscomp_check_ens_ica.setChecked(True)
        self.enscomp_check_ens_x2p.setChecked(True)
        self.enscomp_check_neus_svd.setChecked(False)
        self.enscomp_check_neus_pca.setChecked(False)
        self.enscomp_check_neus_ica.setChecked(False)
        self.enscomp_check_neus_x2p.setChecked(False)

        self.tempvars['performance_shown_results'] = False
        self.tempvars['performance_shown_tab0'] = False
        self.tempvars['performance_shown_tab1'] = False
        self.tempvars['performance_shown_tab2'] = False
        self.tempvars['performance_shown_tab3'] = False
        self.tempvars['performance_shown_tab4'] = False
        
        default_txt = "Perform an ensemble analysis first\nAnd load coordinates\nto see this panel"
        self.findChild(MatplotlibWidget, 'ensvis_plot_map').reset(default_txt)
        default_txt = "Perform an ensemble analysis first\nAnd load dFFo\nto see this panel"
        self.findChild(MatplotlibWidget, 'ensvis_plot_raster').reset(default_txt)
        default_txt = "Perform any analysis to see the identified ensembles\nAnd load coordinates\nto see this panel"
        self.findChild(MatplotlibWidget, 'ensvis_plot_allspatial').reset(default_txt)
        default_txt = "Perform any ensemble analysis\nto see the binary activity of the cells"
        self.findChild(MatplotlibWidget, 'ensvis_plot_allbinary').reset(default_txt)
        default_txt = "Perform any analysis to see the identified ensembles\nAnd load dFFo data\nto see this panel"
        self.findChild(MatplotlibWidget, 'ensvis_plot_alldffo').reset(default_txt)
        default_txt = "Perform any ensemble analysis\nto see the binary activity of the ensembles"
        self.findChild(MatplotlibWidget, 'ensvis_plot_allens').reset(default_txt)

        default_txt = "Perform and select at least one analysis\nand load stimulation data\nto see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_corrstims').reset(default_txt)
        default_txt = "Perform and select at least one analysis\nto see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_corrcells').reset(default_txt)
        #self.findChild(MatplotlibWidget, 'performance_plot_corrcells').canvas.setFixedHeight(400)
        default_txt = "Perform and select at least one analysis and load\n behavior data to see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_corrbehavior').reset(default_txt)
        #self.findChild(MatplotlibWidget, 'performance_plot_corrbehavior').canvas.setFixedHeight(400)
        default_txt = "Perform and select at least one analysis and load\n stimulation data to see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_crossensstim').reset(default_txt)
        #self.findChild(MatplotlibWidget, 'performance_plot_crossensstim').canvas.setFixedHeight(400)
        default_txt = "Perform and select at least one analysis and load\n behavior data to see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_crossensbehavior').reset(default_txt)
        #self.findChild(MatplotlibWidget, 'performance_plot_crossensbehavior').canvas.setFixedHeight(400)

    def browse_files(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.filenamePlain.setText(fname)
        self.update_console_log("Loading file...")
        if fname:
            self.reset_gui()
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
        if item_type == "Dataset" and len(item_size) < 3:
            valid = item_size[0] > 1
            self.btn_set_dFFo.setEnabled(valid)
            self.btn_set_neuronal_activity.setEnabled(valid)
            valid = len(item_size) > 1
            self.btn_set_coordinates.setEnabled(valid)
            self.btn_set_stim.setEnabled(True)
            self.btn_set_behavior.setEnabled(True)
            self.btn_set_cells.setEnabled(True)
        else:
            self.btn_set_dFFo.setEnabled(False)
            self.btn_set_neuronal_activity.setEnabled(False)
            self.btn_set_coordinates.setEnabled(False)
            self.btn_set_stim.setEnabled(False)
            self.btn_set_behavior.setEnabled(False)
            self.btn_set_cells.setEnabled(False)

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
    def main_tabs_change(self, index):
        if index > 0 and index < 5: # Analysis tabs
            if hasattr(self, "data_neuronal_activity"):
                self.lbl_sdv_spikes_selected.setText(f"Loaded")
                self.lbl_pca_spikes_selected.setText(f"Loaded")
                self.lbl_ica_spikes_selected.setText(f"Loaded")
                self.lbl_x2p_spikes_selected.setText(f"Loaded")
            else:
                self.lbl_sdv_spikes_selected.setText(f"Nothing selected")
                self.lbl_pca_spikes_selected.setText(f"Nothing selected")
                self.lbl_ica_spikes_selected.setText(f"Nothing selected")
                self.lbl_x2p_spikes_selected.setText(f"Nothing selected")

            # Validate data for SVD
            needed_data = ["data_neuronal_activity"]
            self.btn_svd_run.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for PCA
            needed_data = ["data_neuronal_activity"]
            self.btn_run_pca.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for ICA
            needed_data = ["data_neuronal_activity"]
            self.btn_run_ica.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for x2p
            needed_data = ["data_neuronal_activity"]
            self.btn_run_x2p.setEnabled(self.validate_needed_data(needed_data))
        if index == 6: #Ensembles compare tab
            if len(self.results) > 0:
                self.ensembles_compare_update_ensembles()

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
    def set_cells(self):
        data_cells = assign_data_from_file(self)
        self.data_cells = data_cells
        stims, cells = data_cells.shape
        self.btn_clear_cells.setEnabled(True)
        self.btn_view_cells.setEnabled(True)
        self.lbl_cells_select.setText("Assigned")
        self.lbl_cells_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Selected cells dataset - Identified {stims} groups and {cells} cells. Please, verify the data preview.", msg_type="complete")
        self.view_cells()
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
    
    def set_able_edit_options(self, boolval):
        # Transpose matrix
        self.btn_edit_transpose.setEnabled(boolval)
        # Binning options
        self.edit_btn_bin.setEnabled(boolval)
        self.edit_edit_binsize.setEnabled(boolval)
        self.edit_radio_sum.setEnabled(boolval)
        self.edit_radio_mean.setEnabled(boolval)
        # Trim options
        self.edit_btn_trim.setEnabled(boolval)
        self.edit_edit_xstart.setEnabled(boolval)
        self.edit_edit_xend.setEnabled(boolval)
        self.edit_edit_ystart.setEnabled(boolval)
        self.edit_edit_yend.setEnabled(boolval)

    ## Clear variables 
    def clear_dFFo(self):
        delattr(self, "data_dFFo")
        self.set_able_edit_options(False)
        self.btn_clear_dFFo.setEnabled(False)
        self.btn_view_dFFo.setEnabled(False)
        self.lbl_dffo_select.setText("Nothing")
        self.lbl_dffo_select_name.setText("")
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)
        self.update_console_log(f"Deleted dFFo dataset", msg_type="complete")       
    def clear_neuronal_activity(self):
        delattr(self, "data_neuronal_activity")
        self.set_able_edit_options(False)
        self.btn_clear_neuronal_activity.setEnabled(False)
        self.btn_view_neuronal_activity.setEnabled(False)
        self.lbl_neuronal_activity_select.setText("Nothing")
        self.lbl_neuronal_activity_select_name.setText("")
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)
        self.update_console_log(f"Deleted Binary Neuronal Activity dataset", msg_type="complete")
    def clear_coordinates(self):
        delattr(self, "data_coordinates")
        self.set_able_edit_options(False)
        self.btn_clear_coordinates.setEnabled(False)
        self.btn_view_coordinates.setEnabled(False)
        self.lbl_coordinates_select.setText("Nothing")
        self.lbl_coordinates_select_name.setText("")
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)
        self.update_console_log(f"Deleted Coordinates dataset", msg_type="complete")
    def clear_stims(self):
        delattr(self, "data_stims")
        self.set_able_edit_options(False)
        self.btn_clear_stim.setEnabled(False)
        self.btn_view_stim.setEnabled(False)
        self.lbl_stim_select.setText("Nothing")
        self.lbl_stim_select_name.setText("")
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)
        self.update_console_log(f"Deleted Stimuli dataset", msg_type="complete")
    def clear_cells(self):
        delattr(self, "data_cells")
        self.set_able_edit_options(False)
        self.btn_clear_cells.setEnabled(False)
        self.btn_view_cells.setEnabled(False)
        self.lbl_cells_select.setText("Nothing")
        self.lbl_cells_select_name.setText("")
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)
        self.update_console_log(f"Deleted Selected cells dataset", msg_type="complete")
    def clear_behavior(self):
        delattr(self, "data_behavior")
        self.set_able_edit_options(False)
        self.btn_clear_behavior.setEnabled(False)
        self.btn_view_behavior.setEnabled(False)
        self.lbl_behavior_select.setText("Nothing")
        self.lbl_behavior_select_name.setText("")
        default_txt = "Load or select a variable\nto see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)
        self.update_console_log(f"Deleted Behavior dataset", msg_type="complete")
        
    ## Visualize variables from input file
    def view_dFFo(self):
        self.currently_visualizing = "dFFo"
        self.set_able_edit_options(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_dataset(self.data_dFFo, ylabel='Cell')
        self.varlabels_setup_tab(self.data_dFFo.shape[0])
    def view_neuronal_activity(self):
        self.currently_visualizing = "neuronal_activity"
        self.set_able_edit_options(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_dataset(self.data_neuronal_activity==0, ylabel='Cell', cmap='gray')
        self.varlabels_setup_tab(self.data_neuronal_activity.shape[0])
    def view_coordinates(self):
        self.currently_visualizing = "coordinates"
        self.set_able_edit_options(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_coordinates2D(self.data_coordinates)
        self.varlabels_setup_tab(self.data_coordinates.shape[0])
    def view_stims(self):
        self.currently_visualizing = "stims"
        self.set_able_edit_options(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_stims
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        self.update_enscomp_options("stims")
        self.plot_widget.preview_dataset(preview_data==0, ylabel='Stim', cmap='gray')
    def view_cells(self):
        self.currently_visualizing = "cells"
        self.set_able_edit_options(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_cells
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        self.plot_widget.preview_dataset(preview_data==0, xlabel="Cell", ylabel='Group', cmap='gray')
    def view_behavior(self):
        self.currently_visualizing = "behavior"
        self.set_able_edit_options(True)
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_behavior
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        self.update_enscomp_options("behavior")
        self.plot_widget.preview_dataset(preview_data)

    ## Edit buttons
    def edit_transpose(self):
        to_edit = self.currently_visualizing
        if to_edit == "dFFo":
            self.data_dFFo = self.data_dFFo.T
            self.update_console_log(f"Updated dFFo dataset. Please, verify the data preview.", "warning")
            self.view_dFFo()
        elif to_edit == "neuronal_activity":
            self.data_neuronal_activity = self.data_neuronal_activity.T
            self.cant_neurons = self.data_neuronal_activity.shape[0]
            self.cant_timepoints = self.data_neuronal_activity.shape[1]
            self.update_console_log(f"Updated Binary Neuronal Activity dataset. Please, verify the data preview.", "warning")
            self.view_neuronal_activity()
        elif to_edit == "coordinates":
            self.data_coordinates = self.data_coordinates.T
            self.update_console_log(f"Updated Coordinates dataset. Please, verify the data preview.", "warning")
            self.view_coordinates()
        elif to_edit == "stims":
            self.data_stims = self.data_stims.T
            self.update_console_log(f"Updated Stims dataset. Please, verify the data preview.", "warning")
            self.view_stims()
        elif to_edit == "cells":
            self.data_cells = self.data_cells.T
            self.update_console_log(f"Updated Selected Cells dataset. Please, verify the data preview.", "warning")
            self.view_cells()
        elif to_edit == "behavior":
            self.data_behavior = self.data_behavior.T
            self.update_console_log(f"Updated Behavior dataset. Please, verify the data preview.", "warning")
            self.view_behavior()

    def bin_matrix(self, mat, bin_size, bin_method):
        elements, timepoints = mat.shape
        if bin_size >= timepoints:
            self.update_console_log(f"Enter a bin size smaller than the curren amount of timepoints. Nothing has been changed.", "warning")
            return mat   
        num_bins = timepoints // bin_size
        bin_mat = np.zeros((elements, num_bins))
        for i in range(num_bins):
            if bin_method == "mean":
                bin_mat[:, i] = np.mean(mat[:, i*bin_size:(i+1)*bin_size], axis=1)
            elif bin_method == "sum":
                bin_mat[:, i] = np.sum(mat[:, i*bin_size:(i+1)*bin_size], axis=1)
        return bin_mat 
    def edit_bin(self):
        to_edit = self.currently_visualizing
        bin_size = self.edit_edit_binsize.text()
        if len(bin_size) == 0:
            self.update_console_log(f"Set a positive and integer bin size to bin the matrix. Nothing has been changed.", "warning")
            return
        else:
            bin_size = int(bin_size)
        bin_method = ""
        if self.edit_radio_sum.isChecked():
            bin_method = "sum"
        else:
            bin_method = "mean"

        if to_edit == "dFFo":
            self.data_dFFo = self.bin_matrix(self.data_dFFo, bin_size, bin_method)
            self.update_console_log(f"Updated dFFo dataset. Please, verify the data preview.", "warning")
            self.view_dFFo()
        elif to_edit == "neuronal_activity":
            self.data_neuronal_activity = self.bin_matrix(self.data_neuronal_activity, bin_size, bin_method)
            print(self.data_neuronal_activity.shape)
            self.cant_neurons = self.data_neuronal_activity.shape[0]
            self.cant_timepoints = self.data_neuronal_activity.shape[1]
            self.update_console_log(f"Updated Binary Neuronal Activity dataset. Please, verify the data preview.", "warning")
            self.view_neuronal_activity()
        elif to_edit == "coordinates":
            self.data_coordinates = self.bin_matrix(self.data_coordinates, bin_size, bin_method)
            self.update_console_log(f"Updated Coordinates dataset. Please, verify the data preview.", "warning")
            self.view_coordinates()
        elif to_edit == "stims":
            self.data_stims = self.bin_matrix(self.data_stims, bin_size, bin_method)
            self.update_console_log(f"Updated Stims dataset. Please, verify the data preview.", "warning")
            self.view_stims()
        elif to_edit == "cells":
            self.data_cells = self.bin_matrix(self.data_cells, bin_size, bin_method)
            self.update_console_log(f"Updated Selected Cells dataset. Please, verify the data preview.", "warning")
            self.view_cells()
        elif to_edit == "behavior":
            self.data_behavior = self.bin_matrix(self.data_behavior, bin_size, bin_method)
            self.update_console_log(f"Updated Behavior dataset. Please, verify the data preview.", "warning")
            self.view_behavior()
        
    def edit_trimmatrix(self):
        # Basic aproach
        to_edit = self.currently_visualizing
        xstart = self.edit_edit_xstart.text()
        xend = self.edit_edit_xend.text()
        ystart = self.edit_edit_ystart.text()
        yend = self.edit_edit_yend.text()
        
        valid_x = len(xstart) and len(xend)
        valid_y = len(ystart) and len(yend)
        
        if valid_x:
            xstart = int(xstart)
            xend = int(xend)
        if valid_y:
            ystart = int(ystart)
            yend = int(yend)

        if to_edit == "dFFo":
            if valid_x:
                self.data_dFFo = self.data_dFFo[:, xstart:xend]
            if valid_y:
                self.data_dFFo = self.data_dFFo[ystart:yend, :]
            self.update_console_log(f"Updated dFFo dataset. Please, verify the data preview.", "warning")
            self.view_dFFo()
        elif to_edit == "neuronal_activity":
            if valid_x:
                self.data_neuronal_activity = self.data_neuronal_activity[:, xstart:xend]
            if valid_y:
                self.data_neuronal_activity = self.data_neuronal_activity[ystart:yend, :]
            print(self.data_neuronal_activity.shape)
            self.cant_neurons = self.data_neuronal_activity.shape[0]
            self.cant_timepoints = self.data_neuronal_activity.shape[1]
            self.update_console_log(f"Updated Binary Neuronal Activity dataset. Please, verify the data preview.", "warning")
            self.view_neuronal_activity()
        elif to_edit == "coordinates":
            if valid_x:
                self.data_coordinates = self.data_coordinates[:, xstart:xend]
            if valid_y:
                self.data_coordinates = self.data_coordinates[ystart:yend, :]
            self.update_console_log(f"Updated Coordinates dataset. Please, verify the data preview.", "warning")
            self.view_coordinates()
        elif to_edit == "stims":
            if valid_x:
                self.data_stims = self.data_stims[:, xstart:xend]
            if valid_y:
                self.data_stims = self.data_stims[ystart:yend, :]
            self.update_console_log(f"Updated Stims dataset. Please, verify the data preview.", "warning")
            self.view_stims()
        elif to_edit == "cells":
            if valid_x:
                self.data_cells = self.data_cells[:, xstart:xend]
            if valid_y:
                self.data_cells = self.data_cells[ystart:yend, :]
            self.update_console_log(f"Updated Selected Cells dataset. Please, verify the data preview.", "warning")
            self.view_cells()
        elif to_edit == "behavior":
            if valid_x:
                self.data_behavior = self.data_behavior[:, xstart:xend]
            if valid_y:
                self.data_behavior = self.data_behavior[ystart:yend, :]
            self.update_console_log(f"Updated Behavior dataset. Please, verify the data preview.", "warning")
            self.view_behavior()

    def varlabels_setup_tab(self, rows_cant):
        curr_view = self.currently_visualizing
        new_colum_start = ""
        label_family = ""
        if curr_view == "dFFo" or curr_view == "neuronal_activity" or curr_view == "coordinates":
            new_colum_start = "Cell"
            label_family = "cell"
        elif curr_view == "stims":
            new_colum_start = "Stimulus"
            label_family = "stim"
        elif curr_view == "cells":
            new_colum_start = "Selected cell"
            label_family = "selected_cell"
        elif curr_view == "behavior":
            new_colum_start = "Behavior"
            label_family = "behavior"
        self.table_setlabels.setRowCount(rows_cant)
        self.table_setlabels.setHorizontalHeaderLabels([f"{new_colum_start} index", "Label"])
        labels_registered = label_family in self.varlabels
        for row in range(rows_cant):
            self.table_setlabels.setItem(row, 0, QTableWidgetItem(str(row+1)))
            item = self.table_setlabels.item(row, 0)
            if item is not None: # Remove the ItemIsEditable flag
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if labels_registered:
                if row in self.varlabels[label_family]:
                    self.table_setlabels.setItem(row, 1, QTableWidgetItem(self.varlabels[label_family][row]))
            else:   # To clear out all the previous entries
                self.table_setlabels.setItem(row, 1, QTableWidgetItem(None))
    def varlabels_save(self):
        label_family = ""
        curr_view = self.currently_visualizing
        if curr_view == "dFFo" or curr_view == "neuronal_activity" or curr_view == "coordinates":
            label_family = "cell"
        elif curr_view == "stims":
            label_family = "stim"
        elif curr_view == "cells":
            label_family = "selected_cell"
        elif curr_view == "behavior":
            label_family = "behavior"
        if not label_family in self.varlabels:
            self.varlabels[label_family] = {}
        # Iterate through each row to get the value of the labels column
        for row in range(self.table_setlabels.rowCount()):
            item = self.table_setlabels.item(row, 1)
            new_label = str(row+1)
            if item is not None:
                if len(item.text()) > 0:
                    new_label = item.text()
            self.varlabels[label_family][row] = new_label
        self.update_console_log(f"Saved {label_family} labels. Please, verify the data preview.", "warning")
    def varlabels_clear(self):
        label_family = ""
        curr_view = self.currently_visualizing
        if curr_view == "dFFo" or curr_view == "neuronal_activity" or curr_view == "coordinates":
            label_family = "cell"
        elif curr_view == "stims":
            label_family = "stim"
        elif curr_view == "cells":
            label_family = "selected_cell"
        elif curr_view == "behavior":
            label_family = "behavior"
        if label_family in self.varlabels:
            del self.varlabels[label_family]
        self.varlabels_setup_tab(self.table_setlabels.rowCount())
        
    def dict_to_matlab_struct(self, pars_dict):
        matlab_struct = {}
        for key, value in pars_dict.items():
            if isinstance(value, dict):
                matlab_struct[key] = self.dict_to_matlab_struct(value)
            elif isinstance(value, (int, float)):
                matlab_struct[key] = matlab.double([value])
            else:
                matlab_struct[key] = value
        return matlab_struct

    def load_defaults_svd(self):
        defaults = self.svd_defaults
        self.svd_edit_pks.setText(f"{defaults['pks']}")
        self.svd_edit_scut.setText(f"{defaults['scut']}")
        self.svd_edit_hcut.setText(f"{defaults['hcut']}")
        self.svd_edit_statecut.setText(f"{defaults['state_cut']}")
        self.svd_edit_csistart.setText(f"{defaults['csi_start']}")
        self.svd_edit_csistep.setText(f"{defaults['csi_step']}")
        self.svd_edit_csiend.setText(f"{defaults['csi_end']}")
        self.svd_check_tfidf.setChecked(defaults['tf_idf_norm'])
        self.update_console_log("Loaded default SVD parameter values", "complete")
    def run_svd(self):
        # Prepare data
        data = self.data_neuronal_activity
        spikes = matlab.double(data.tolist())
        #Prepare dummy data
        data = np.zeros((self.cant_neurons,2))
        coords_foo = matlab.double(data.tolist())

        # Prepare parameters
        input_value = self.svd_edit_pks.text()
        val_pks = np.array([float(input_value)]) if len(input_value) > 0 else np.array([]) 
        input_value = self.svd_edit_scut.text()
        val_scut = np.array([float(input_value)]) if len(input_value) > 0 else np.array([]) 
        input_value = self.svd_edit_hcut.text()
        if len(input_value) > 0:
            val_hcut = float(input_value) 
        else:
            val_hcut = self.svd_defaults['hcut']
            self.svd_edit_hcut.setText(f"{val_hcut}")
        input_value = self.svd_edit_statecut.text()
        if len(input_value) > 0:
            val_statecut = float(input_value)
        else:
            val_statecut = self.svd_defaults['statecut']
            self.svd_edit_statecut.setText(f"{val_statecut}")
        input_value = self.svd_edit_csistart.text()
        if len(input_value) > 0:
            val_csistart = float(input_value)
        else:
            val_csistart = self.svd_defaults['csi_start']
            self.svd_edit_csistart.setText(f"{val_csistart}")
        input_value = self.svd_edit_csistep.text()
        if len(input_value) > 0:
            val_csistep = float(input_value)
        else:
            val_csistep = self.svd_defaults['statecut']
            self.svd_edit_csistep.setText(f"{val_csistep}")
        input_value = self.svd_edit_csiend.text()
        if len(input_value) > 0:
            val_csiend = float(input_value)
        else:
            val_csiend = self.svd_defaults['statecut']
            self.svd_edit_csiend.setText(f"{val_csiend}")
        val_idtfd = self.svd_check_tfidf.isChecked()

        # Pack parameters
        pars = {
            'pks': val_pks,
            'scut': val_scut,
            'hcut': val_hcut,
            'statecut': val_statecut,
            'tf_idf_norm': val_idtfd,
            'csi_start': val_csistart,
            'csi_step': val_csistep,
            'csi_end': val_csiend
        }
        self.params['svd'] = pars
        pars_matlab = self.dict_to_matlab_struct(pars)

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/SVD'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        # Clean all the figures in case there was something previously
        if 'svd' in self.results:
            del self.results['svd']
        algorithm_figs = ["svd_plot_similaritymap", "svd_plot_binarysimmap", "svd_plot_singularvalues", "svd_plot_components", "svd_plot_timecourse", "svd_plot_cellsinens"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing SVD...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        try:
            answer = eng.Stoixeion(spikes, coords_foo, pars_matlab)
        except:
            self.update_console_log("An error occurred while excecuting the algorithm. Check the Python console for more info.", msg_type="error")
            answer = None
        self.update_console_log("Done.", "complete")

        if answer != None:
            # Update pks and scut in case of automatic calculation
            self.svd_edit_pks.setText(f"{int(answer['pks'])}")
            self.svd_edit_scut.setText(f"{answer['scut']}")

            # Plotting results
            self.update_console_log("Plotting and saving results...")
            # For this method the saving occurs in the same plotting function to avoid recomputation
            self.plot_SVD_results(answer)
            self.update_console_log("Done plotting and saving...")
    def plot_SVD_results(self, answer):
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
        self.plot_widget.plot_ensembles_timecourse(neurons_in_ensembles, xlabel="Cell")

    def load_defaults_pca(self):
        defaults = self.pca_defaults
        self.pca_edit_dc.setText(f"{defaults['dc']}")
        self.pca_edit_npcs.setText(f"{defaults['npcs']}")
        self.pca_edit_minspk.setText(f"{defaults['minspk']}")
        self.pca_edit_nsur.setText(f"{defaults['nsur']}")
        self.pca_edit_prct.setText(f"{defaults['prct']}")
        self.pca_edit_centthr.setText(f"{defaults['cent_thr']}")
        self.pca_edit_innercorr.setText(f"{defaults['inner_corr']}")
        self.pca_edit_minsize.setText(f"{defaults['minsize']}")
        self.update_console_log("Loaded default PCA parameter values", "complete")
    def run_PCA(self):
        # Prepare data
        data = self.data_neuronal_activity
        raster = matlab.double(data.tolist())

        # Prepare parameters
        input_value = self.pca_edit_dc.text()
        dc = float(input_value) if len(input_value) > 0 else self.pca_defaults['dc']
        input_value = self.pca_edit_npcs.text()
        npcs = float(input_value) if len(input_value) > 0 else self.pca_defaults['npcs']
        input_value = self.pca_edit_minspk.text()
        minspk = float(input_value) if len(input_value) > 0 else self.pca_defaults['minspk']
        input_value = self.pca_edit_nsur.text()
        nsur = float(input_value) if len(input_value) > 0 else self.pca_defaults['nsur']
        input_value = self.pca_edit_prct.text()
        prct = float(input_value) if len(input_value) > 0 else self.pca_defaults['prct']
        input_value = self.pca_edit_centthr.text()
        cent_thr = float(input_value) if len(input_value) > 0 else self.pca_defaults['cent_thr']
        input_value = self.pca_edit_innercorr.text()
        inner_corr = float(input_value) if len(input_value) > 0 else self.pca_defaults['inner_corr']
        input_value = self.pca_edit_minsize.text()
        minsize = float(input_value) if len(input_value) > 0 else self.pca_defaults['minsize']

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
        self.params['pca'] = pars
        pars_matlab = self.dict_to_matlab_struct(pars)

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/NeuralEnsembles'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        # Clean all the figures in case there was something previously
        if 'pca' in self.results:
            del self.results['pca']
        algorithm_figs = ["pca_plot_eigs", "pca_plot_pca", "pca_plot_rhodelta", "pca_plot_corrne", "pca_plot_corecells", "pca_plot_innerens", "pca_plot_timecourse", "pca_plot_cellsinens"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing PCA...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        try:
            answer = eng.raster2ens_by_density(raster, pars_matlab)
        except:
            self.update_console_log("An error occurred while excecuting the algorithm. Check the Python console for more info.", msg_type="error")
            answer = None
        self.update_console_log("Done.", "complete")

        ## Plot the results
        if answer != None:
            self.update_console_log("Plotting results...")
            self.plot_PCA_results(pars, answer)
            self.update_console_log("Done plotting.", "complete")

            # Save the results
            self.update_console_log("Saving results...")
            if np.array(answer["sel_ensmat_out"]).shape[0] > 0:
                self.results['pca'] = {}
                self.results['pca']['timecourse'] = np.array(answer["sel_ensmat_out"]).astype(int)
                self.results['pca']['ensembles_cant'] = self.results['pca']['timecourse'].shape[0]
                self.results['pca']['neus_in_ens'] = np.array(answer["sel_core_cells"]).T.astype(float)
                self.we_have_results()
                self.update_console_log("Done saving", "complete")
            else:
                self.update_console_log("The algorithm didn't found any ensemble. Check the python console for more info.", "error")
    def plot_PCA_results(self, pars, answer):
        ## Plot the eigs
        eigs = np.array(answer['exp_var'])
        seleig = int(pars['npcs'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_eigs')
        self.plot_widget.plot_eigs(eigs, seleig)

        # Plot the PCA
        pcs = np.array(answer['pcs'])
        labels = np.array(answer['labels'])
        labels = labels[0] if len(labels) else None
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
        try:
            ens_cel_corr = np.array(answer['ens_cel_corr'])
            ens_cel_corr_min = np.min(ens_cel_corr)
            ens_cel_corr_max = np.max(ens_cel_corr)
            self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_corrne')
            self.plot_widget.plot_core_cells(ens_cel_corr, [ens_cel_corr_min, ens_cel_corr_max])
        except:
            print("Error plotting the correlation of cells vs ensembles. Check the other plots and console for more info.")

        # Plot core cells
        core_cells = np.array(answer['core_cells'])
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_corecells')
        self.plot_widget.plot_core_cells(core_cells, [-1, 1])

        # Plot core cells
        try:
            ens_corr = np.array(answer["ens_corr"])[0]
            corr_thr = np.array(answer["corr_thr"])
            self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_innerens')
            self.plot_widget.plot_ens_corr(ens_corr, corr_thr, ens_cols)
        except:
            print("Error plotting the core cells. Check the other plots and console for more info.")

        # Plot ensembles timecourse
        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_timecourse')
        self.plot_widget.plot_ensembles_timecourse(np.array(answer["sel_ensmat_out"]))

        self.plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_cellsinens')
        self.plot_widget.plot_ensembles_timecourse(np.array(answer["sel_core_cells"]).T)

    def load_defaults_ica(self):
        defaults = self.ica_defaults
        self.ica_radio_method_marcenko.setChecked(True)
        self.ica_edit_perpercentile.setText(f"{defaults['threshold']['permutations_percentile']}")
        self.ica_edit_percant.setText(f"{defaults['threshold']['number_of_permutations']}")
        self.ica_radio_method_ica.setChecked(True)
        self.ica_edit_iterations.setText(f"{defaults['Patterns']['number_of_iterations']}")
        self.update_console_log("Loaded default ICA parameter values", "complete")
    def run_ICA(self):
        # Prepare data
        data = self.data_neuronal_activity
        spikes = matlab.double(data.tolist())

        # Prepare parameters
        if self.ica_radio_method_marcenko.isChecked():
            threshold_method = "MarcenkoPastur"
        elif self.ica_radio_method_shuffling.isChecked():
            threshold_method = "binshuffling"
        elif self.ica_radio_method_shift.isChecked():
            threshold_method = "circularshift"

        input_value = self.ica_edit_perpercentile.text()
        val_per_percentile = float(input_value) if len(input_value) > 0 else self.ica_defaults['threshold']['permutations_percentile']
        input_value = self.ica_edit_percant.text()
        val_per_cant = float(input_value) if len(input_value) > 0 else self.ica_defaults['threshold']['number_of_permutations']

        if self.ica_radio_method_ica.isChecked():
            patterns_method = "ICA"
        elif self.ica_radio_method_pca.isChecked():
            patterns_method = "PCA"
        input_value = self.ica_edit_iterations.text()
        val_iteartions = float(input_value) if len(input_value) > 0 else self.ica_defaults['Patterns']['number_of_iterations']

        # Pack parameters
        pars = {
            'threshold': {
                'method': threshold_method,
                'permutations_percentile': val_per_percentile,
                'number_of_permutations': val_per_cant
            },
            'Patterns': {
                'method': patterns_method,
                'number_of_iterations': val_iteartions
            }
        }
        self.params['ica'] = pars
        pars_matlab = self.dict_to_matlab_struct(pars)

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/Cell-Assembly-Detection'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        # Clean all the figures in case there was something previously
        if 'ica' in self.results:
            del self.results['ica']
        algorithm_figs = ["ica_plot_assemblys", "ica_plot_activity", "ica_plot_binary_patterns", "ica_plot_binary_assemblies"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing ICA...")
        self.update_console_log("Looking for patterns...")
        try:
            answer = eng.assembly_patterns(spikes, pars_matlab)
        except:
            self.update_console_log("An error occurred while excecuting the algorithm. Check the Python console for more info.", msg_type="error")
            answer = None
        self.update_console_log("Done looking for patterns...", "complete")

        if answer != None:
            assembly_templates = np.array(answer['AssemblyTemplates']).T

            self.update_console_log("Looking for assembly activity...")
            try:
                answer = eng.assembly_activity(answer['AssemblyTemplates'],spikes)
            except:
                self.update_console_log("An error occurred while excecuting the algorithm. Check the Python console for more info.", msg_type="error")
                answer = None
            self.update_console_log("Done looking for assembly activity...", "complete")
        self.update_console_log("Done.", "complete")

        if answer != None:
            time_projection = np.array(answer["time_projection"])
            
            ## Identify the significative values to binarize the matrix
            threshold = 1.96    # p < 0.05 for the z-score
            binary_assembly_templates = np.zeros(assembly_templates.shape)
            for a_idx, assembly in enumerate(assembly_templates):
                z_scores = stats.zscore(assembly)
                tmp = np.abs(z_scores) > threshold
                binary_assembly_templates[a_idx,:] = [int(v) for v in tmp]

            binary_time_projection = np.zeros(time_projection.shape)
            for a_idx, assembly in enumerate(time_projection):
                z_scores = stats.zscore(assembly)
                tmp = np.abs(z_scores) > threshold
                binary_time_projection[a_idx,:] = [int(v) for v in tmp]

            answer = {
                'assembly_templates': assembly_templates,
                'time_projection': time_projection,
                'binary_assembly_templates': binary_assembly_templates,
                'binary_time_projection': binary_time_projection
            }
            self.plot_ICA_results(answer)

            self.update_console_log("Saving results...")
            self.results['ica'] = {}
            self.results['ica']['timecourse'] = binary_time_projection
            self.results['ica']['ensembles_cant'] = binary_time_projection.shape[0]
            self.results['ica']['neus_in_ens'] = binary_assembly_templates
            self.we_have_results()
            self.update_console_log("Done saving", "complete")
    def plot_ICA_results(self, answer):
        # Plot the assembly templates
        self.plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_assemblys')
        self.plot_widget.set_subplots(answer['assembly_templates'].shape[0], 1)
        total_assemblies = answer['assembly_templates'].shape[0]
        for e_idx, ens in enumerate(answer['assembly_templates']):
            plot_xaxis = e_idx == total_assemblies-1
            self.plot_widget.plot_assembly_patterns(ens, e_idx, title=f"Ensemble {e_idx+1}", plot_xaxis=plot_xaxis)

        # Plot the time projection
        self.plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_activity')
        self.plot_widget.plot_cell_assemblies_activity(answer['time_projection'])

        # Plot binary assembly templates
        self.plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_binary_patterns')
        self.plot_widget.plot_ensembles_timecourse(answer['binary_assembly_templates'], xlabel="Cell")

        self.plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_binary_assemblies')
        self.plot_widget.plot_ensembles_timecourse(answer['binary_time_projection'], xlabel="Timepoint")

    def load_defaults_x2p(self):
        defaults = self.x2p_defaults
        self.x2p_edit_bin.setText(f"{defaults['network_bin']}")
        self.x2p_edit_iterations.setText(f"{defaults['network_iterations']}")
        self.x2p_edit_significance.setText(f"{defaults['network_significance']}")
        self.x2p_edit_threshold.setText(f"{defaults['coactive_neurons_threshold']}")
        self.x2p_edit_rangestart.setText(f"{defaults['clustering_range_start']}")
        self.x2p_edit_rangeend.setText(f"{defaults['clustering_range_end']}")
        self.x2p_edit_fixed.setText(f"{defaults['clustering_fixed']}")
        self.x2p_edit_itensemble.setText(f"{defaults['iterations_ensemble']}")
        self.x2p_check_parallel.setChecked(defaults['parallel_processing'])
        self.update_console_log("Loaded default Xsembles2P parameter values", "complete")
    def run_x2p(self):
        # Prepare data
        data = self.data_neuronal_activity
        raster = matlab.logical(data.tolist())

        # Prepare parameters
        input_value = self.x2p_edit_bin.text()
        val_network_bin = float(input_value) if len(input_value) > 0 else self.x2p_defaults['network_bin']
        input_value = self.x2p_edit_iterations.text()
        val_network_iterations = float(input_value) if len(input_value) > 0 else self.x2p_defaults['network_iterations']
        input_value = self.x2p_edit_significance.text()
        val_network_significance = float(input_value) if len(input_value) > 0 else self.x2p_defaults['network_significance']
        input_value = self.x2p_edit_threshold.text()
        val_coactive_neurons_threshold = float(input_value) if len(input_value) > 0 else self.x2p_defaults['coactive_neurons_threshold']
        input_value = self.x2p_edit_rangestart.text()
        val_clustering_range_start = float(input_value) if len(input_value) > 0 else self.x2p_defaults['clustering_range_start']
        input_value = self.x2p_edit_rangeend.text()
        val_clustering_range_end = float(input_value) if len(input_value) > 0 else self.x2p_defaults['clustering_range_end']
        val_clustering_range = range(int(val_clustering_range_start), int(val_clustering_range_end)+1)
        val_clustering_range = matlab.double(val_clustering_range)
        input_value = self.x2p_edit_fixed.text()
        val_clustering_fixed = float(input_value) if len(input_value) > 0 else self.x2p_defaults['clustering_fixed']
        input_value = self.x2p_edit_itensemble.text()
        val_iterations_ensemble = float(input_value) if len(input_value) > 0 else self.x2p_defaults['iterations_ensemble']
        parallel = matlab.logical(self.x2p_check_parallel.isChecked())

        # Pack parameters
        pars = {
            'NetworkBin': val_network_bin,
            'NetworkIterations': val_network_iterations,
            'NetworkSignificance': val_network_significance,
            'CoactiveNeuronsThreshold': val_coactive_neurons_threshold,
            'ClusteringRange': val_clustering_range,
            'ClusteringFixed': val_clustering_fixed,
            'EnsembleIterations': val_iterations_ensemble,
            'ParallelProcessing': parallel,
            'FileLog': ''
        }
        self.params['x2p'] = pars
        pars_matlab = self.dict_to_matlab_struct(pars)

        self.update_console_log("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        self.update_console_log("Loaded MATLAB engine.", "complete")

        # Adding to path
        relative_folder_path = 'analysis/Xsembles2P'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng.genpath(folder_path)
        eng.addpath(folder_path_with_subfolders, nargout=0)

        # Clean all the figures in case there was something previously
        #if 'ica' in self.results:
        #    del self.results['ica']
        #algorithm_figs = ["ica_plot_assemblys", "ica_plot_activity", "ica_plot_binary_patterns", "ica_plot_binary_assemblies"] 
        #for fig_name in algorithm_figs:
        #    self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing Xsembles2P...")
        try:
            answer = eng.Get_Xsembles(raster, pars_matlab)
        except:
            self.update_console_log("An error occurred while excecuting the algorithm. Check the Python console for more info.", msg_type="error")
            answer = None
        self.update_console_log("Done.", "complete")

        if answer != None:
            clean_answer = {}
            clean_answer['similarity'] = np.array(answer['Clustering']['Similarity'])
            clean_answer['EPI'] = np.array(answer['Ensembles']['EPI'])
            clean_answer['OnsembleActivity'] = np.array(answer['Ensembles']['OnsembleActivity'])
            clean_answer['OffsembleActivity'] = np.array(answer['Ensembles']['OffsembleActivity'])
            clean_answer['Activity'] = np.array(answer['Ensembles']['Activity'])
            cant_ens = int(answer['Ensembles']['Count'])
            clean_answer['Count'] = cant_ens
            ## Format the onsemble and offsemble neurons
            clean_answer['OnsembleNeurons'] = np.zeros((cant_ens, self.cant_neurons))
            for ens_it in range(cant_ens):
                members = np.array(answer['Ensembles']['OnsembleNeurons'][ens_it]) - 1
                members = members.astype(int)
                clean_answer['OnsembleNeurons'][ens_it, members] = 1
            clean_answer['OffsembleNeurons'] = np.zeros((cant_ens, self.cant_neurons))
            for ens_it in range(cant_ens):
                members = np.array(answer['Ensembles']['OffsembleNeurons'][ens_it]) - 1
                members = members.astype(int)
                clean_answer['OffsembleNeurons'][ens_it, members] = 1

            self.plot_X2P_results(clean_answer)

            self.update_console_log("Saving results...")
            self.results['x2p'] = {}
            self.results['x2p']['timecourse'] = clean_answer['Activity']
            self.results['x2p']['ensembles_cant'] = cant_ens
            self.results['x2p']['neus_in_ens'] = clean_answer['OnsembleNeurons']
            self.we_have_results()
            self.update_console_log("Done saving", "complete")
            
            #import pprint
            #with open('text_output.txt', 'w') as file:
            #    pprint.pprint(answer, stream=file)
            #with h5py.File("ensembles_output.h5", 'w') as hdf_file:
            #    tmp = {"results": answer}
            #    self.save_data_to_hdf5(hdf_file, tmp)
            #print("done saving")
    def plot_X2P_results(self, answer):
        # Similarity map
        dataset = answer['similarity']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_similarity')
        plot_widget.preview_dataset(dataset, xlabel="Vector #", ylabel="Vector #", cmap='jet', aspect='equal')
        # EPI
        dataset = answer['EPI']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_epi')
        plot_widget.preview_dataset(dataset, xlabel="Neuron", ylabel="Ensemble", cmap='jet')
        # Onsemble activity
        dataset = answer['OnsembleActivity']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_onsemact')
        plot_widget.preview_dataset(dataset, xlabel="Timepoint", ylabel="Ensemble", cmap='jet')
        # Onsemble activity
        dataset = answer['OffsembleActivity']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_offsemact')
        plot_widget.preview_dataset(dataset, xlabel="Timepoint", ylabel="Ensemble", cmap='jet')
        # Activity
        dataset = answer['Activity']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_activity')
        plot_widget.plot_ensembles_timecourse(dataset)
        # Onsemble neurons
        dataset = answer['OnsembleNeurons']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_onsemneu')
        plot_widget.plot_ensembles_timecourse(dataset, xlabel="Cell")
        # Offsemble neurons
        dataset = answer['OffsembleNeurons']
        plot_widget = self.findChild(MatplotlibWidget, 'x2p_plot_offsemneu')
        plot_widget.plot_ensembles_timecourse(dataset, xlabel="Cell")


    def we_have_results(self):
        self.save_btn_save.setEnabled(True)
        for analysis_name in self.results.keys():
            if analysis_name == 'svd':
                self.ensvis_btn_svd.setEnabled(True)
                self.performance_check_svd.setEnabled(True)
                self.ensembles_compare_update_opts('svd')
            elif analysis_name == 'pca':
                self.ensvis_btn_pca.setEnabled(True)
                self.performance_check_pca.setEnabled(True)
                self.ensembles_compare_update_opts('pca')
            elif analysis_name == 'ica':
                self.ensvis_btn_ica.setEnabled(True)
                self.performance_check_ica.setEnabled(True)
                self.ensembles_compare_update_opts('ica')
            elif analysis_name == 'x2p':
                self.ensvis_btn_x2p.setEnabled(True)
                self.performance_check_x2p.setEnabled(True)
                self.ensembles_compare_update_opts('x2p')

    def ensvis_tabchange(self, index):
        if self.tempvars['ensvis_shown_results']:
            if index == 0:  # General
                pass
            elif index == 1:    # Spatial distributions
                if hasattr(self, "data_coordinates"):
                    if not self.tempvars['ensvis_shown_tab1']:
                        self.tempvars['ensvis_shown_tab1'] = True
                        self.update_ensvis_allcoords()
            elif index == 2:    # Binary activations
                if not self.tempvars['ensvis_shown_tab2']:
                    self.tempvars['ensvis_shown_tab2'] = True
                    self.update_ensvis_allbinary()
            elif index == 3:    # dFFo
                if hasattr(self, "data_dFFo"):
                    if not self.tempvars['ensvis_shown_tab3']:
                        self.tempvars['ensvis_shown_tab3'] = True
                        self.update_ensvis_alldFFo()
            elif index == 4:    # Ensemble activations
                if not self.tempvars['ensvis_shown_tab4']:
                    self.tempvars['ensvis_shown_tab4'] = True
                    self.update_ensvis_allens()

    def vis_ensembles_svd(self):
        self.ensemble_currently_shown = "svd"
        self.update_analysis_results()
    def vis_ensembles_pca(self):
        self.ensemble_currently_shown = "pca"
        self.update_analysis_results()
    def vis_ensembles_ica(self):
        self.ensemble_currently_shown = "ica"
        self.update_analysis_results()
    def vis_ensembles_x2p(self):
        self.ensemble_currently_shown = "x2p"
        self.update_analysis_results()

    def update_analysis_results(self):
        self.initialize_ensemble_view()   
        self.tempvars['ensvis_shown_tab1'] = False
        self.tempvars['ensvis_shown_tab2'] = False
        self.tempvars['ensvis_shown_tab3'] = False
        self.tempvars['ensvis_shown_tab4'] = False 

    def initialize_ensemble_view(self):
        self.tempvars['ensvis_shown_results'] = True
        self.ensvis_tabs.setCurrentIndex(0)
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

        self.current_idx_corrected_members = idx_corrected_members
        self.current_idx_corrected_exclusive = idx_corrected_exclusive
        
        if hasattr(self, "data_coordinates"):
            self.ensvis_check_onlyens.setEnabled(True)
            self.ensvis_check_onlycont.setEnabled(True)
            self.ensvis_check_cellnum.setEnabled(True)
            self.update_ens_vis_coords()

        if hasattr(self, "data_dFFo"):
            self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_raster')
            dFFo_ens = self.data_dFFo[idx_corrected_members, :]
            self.plot_widget.plot_ensemble_dFFo(dFFo_ens, idx_corrected_members, ensemble_timecourse)
    
    def update_ens_vis_coords(self):
        only_ens = self.ensvis_check_onlyens.isChecked()
        only_contours = self.ensvis_check_onlycont.isChecked()
        show_numbers = self.ensvis_check_cellnum.isChecked()
        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_map')
        self.plot_widget.plot_coordinates2D_highlight(self.data_coordinates, self.current_idx_corrected_members, self.current_idx_corrected_exclusive, only_ens, only_contours, show_numbers)

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
        self.plot_widget.canvas.setFixedHeight(300*rows)

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

    def ensembles_compare_update_opts(self, algorithm):
        if algorithm == 'svd':
            ens_selector = self.enscomp_slider_svd
            selector_label_min = self.enscomp_slider_lbl_min_svd
            selector_label_max = self.enscomp_slider_lbl_max_svd
            check_coords = self.enscomp_check_coords_svd
            check_ensemble = self.enscomp_check_ens_svd
            check_neurons = self.enscomp_check_neus_svd
            color_button = self.enscomp_btn_color_svd
        elif algorithm == 'pca':
            ens_selector = self.enscomp_slider_pca
            selector_label_min = self.enscomp_slider_lbl_min_pca
            selector_label_max = self.enscomp_slider_lbl_max_pca
            check_coords = self.enscomp_check_coords_pca
            check_ensemble = self.enscomp_check_ens_pca
            check_neurons = self.enscomp_check_neus_pca
            color_button = self.enscomp_btn_color_pca
        elif algorithm == 'ica':
            ens_selector = self.enscomp_slider_ica
            selector_label_min = self.enscomp_slider_lbl_min_ica
            selector_label_max = self.enscomp_slider_lbl_max_ica
            check_coords = self.enscomp_check_coords_ica
            check_ensemble = self.enscomp_check_ens_ica
            check_neurons = self.enscomp_check_neus_ica
            color_button = self.enscomp_btn_color_ica
        elif algorithm == 'x2p':
            ens_selector = self.enscomp_slider_x2p
            selector_label_min = self.enscomp_slider_lbl_min_x2p
            selector_label_max = self.enscomp_slider_lbl_max_x2p
            check_coords = self.enscomp_check_coords_x2p
            check_ensemble = self.enscomp_check_ens_x2p
            check_neurons = self.enscomp_check_neus_x2p
            color_button = self.enscomp_btn_color_x2p
        # Activate the slider
        ens_selector.setEnabled(True)
        ens_selector.setMinimum(1)   # Set the minimum value
        ens_selector.setMaximum(self.results[algorithm]['ensembles_cant']) # Set the maximum value
        ens_selector.setValue(1)
        selector_label_min.setEnabled(True)
        selector_label_min.setText(f"{1}")
        selector_label_max.setEnabled(True)
        selector_label_max.setText(f"{self.results[algorithm]['ensembles_cant']}")
        # Update the toolbox options
        check_coords.setEnabled(True)
        check_ensemble.setEnabled(True)
        check_neurons.setEnabled(True)
        color_button.setEnabled(True)
    
    def update_enscomp_options(self, exp_data):
        if exp_data == "stims":
            slider = self.enscomp_slider_stim
            lbl_min = self.enscomp_slider_lbl_min_stim
            lbl_max = self.enscomp_slider_lbl_max_stim
            lbl_label = self.enscomp_slider_lbl_stim
            check_show = self.enscomp_check_show_stim
            color_pick = self.enscomp_btn_color_stim
            shp = self.data_stims.shape
            max_val = shp[0] if len(shp) > 1 else 1
        elif exp_data == "behavior":
            slider = self.enscomp_slider_behavior
            lbl_min = self.enscomp_slider_lbl_min_behavior
            lbl_max = self.enscomp_slider_lbl_max_behavior
            lbl_label = self.enscomp_slider_lbl_behavior
            check_show = self.enscomp_check_behavior_stim
            color_pick = self.enscomp_btn_color_behavior
            shp = self.data_behavior.shape
            max_val = shp[0] if len(shp) > 1 else 1
        # Activate the slider
        slider.setEnabled(True)
        slider.setMinimum(1)   # Set the minimum value
        slider.setMaximum(max_val) # Set the maximum value
        slider.setValue(1)
        lbl_min.setText(f"{1}")
        lbl_min.setEnabled(True)
        lbl_label.setText(f"{1}")
        lbl_label.setEnabled(True)
        lbl_max.setText(f"{max_val}")
        lbl_max.setEnabled(True)
        # Update the toolbox options
        check_show.setEnabled(True)
        color_pick.setEnabled(True)
        
    def ensembles_compare_update_ensembles(self):
        ensembles_to_compare = {}
        ens_selector = {
            "svd": self.enscomp_slider_svd,
            "pca": self.enscomp_slider_pca,
            "ica": self.enscomp_slider_ica,
            "x2p": self.enscomp_slider_x2p
        }
        for key, slider in ens_selector.items():
            if slider.isEnabled():
                ens_idx = slider.value()
                ensembles_to_compare[key] = {}
                ensembles_to_compare[key]["ens_idx"] = ens_idx-1
                ensembles_to_compare[key]["neus_in_ens"] = self.results[key]['neus_in_ens'][ens_idx-1,:].copy()
                ensembles_to_compare[key]["timecourse"] = self.results[key]['timecourse'][ens_idx-1,:].copy()
        
        self.enscomp_colorflag_svd.setStyleSheet(f'background-color: {self.enscom_colors["svd"]};')
        self.enscomp_colorflag_pca.setStyleSheet(f'background-color: {self.enscom_colors["pca"]};')
        self.enscomp_colorflag_ica.setStyleSheet(f'background-color: {self.enscom_colors["ica"]};')
        self.enscomp_colorflag_x2p.setStyleSheet(f'background-color: {self.enscom_colors["x2p"]};')

        self.ensembles_compare_update_map(ensembles_to_compare)
        self.ensembles_compare_update_timecourses(ensembles_to_compare)

    def ensembles_compare_update_map(self, ensembles_to_compare):
        ens_show = {
            "svd": self.enscomp_check_coords_svd,
            "pca": self.enscomp_check_coords_pca,
            "ica": self.enscomp_check_coords_ica,
            "x2p": self.enscomp_check_coords_x2p
        }
        # Stablish the dimention of the map
        max_x = np.max(self.data_coordinates[:, 0])
        max_y = np.max(self.data_coordinates[:, 1])
        lims = [max_x, max_y]

        mixed_ens = []
        colors = self.enscom_colors

        list_colors_freq = [[] for l in range(self.cant_neurons)] 

        for key, ens_data in ensembles_to_compare.items():
            if ens_show[key].isChecked():
                new_members = ens_data["neus_in_ens"].copy()
                if len(mixed_ens) == 0:
                    mixed_ens = new_members
                else:
                    mixed_ens += new_members
                for cell_idx in range(len(new_members)):
                    if new_members[cell_idx] > 0:
                        list_colors_freq[cell_idx].append(colors[key])

        members_idx = [idx for idx in range(len(mixed_ens)) if mixed_ens[idx] > 0]
        members_freq = [member for member in mixed_ens if member > 0]
        members_colors = [colors_list for colors_list in list_colors_freq if len(colors_list) > 0]

        members_coords = [[],[]]
        members_coords[0] = self.data_coordinates[members_idx, 0]
        members_coords[1] = self.data_coordinates[members_idx, 1]

        neuron_size = float(self.enscomp_visopts_neusize.text())

        members_idx = []
        if self.enscomp_visopts_showcells.isChecked():
            members_idx = [idx for idx in range(len(mixed_ens)) if mixed_ens[idx] > 0]

        map_plot = self.findChild(MatplotlibWidget, 'enscomp_plot_map')
        map_plot.enscomp_update_map(lims, members_idx, members_freq, members_coords, members_colors, neuron_size)

    def ensembles_compare_update_timecourses(self, ensembles_to_compare):
        requested = {
            "svd": [self.enscomp_check_ens_svd, self.enscomp_check_neus_svd],
            "pca": [self.enscomp_check_ens_pca, self.enscomp_check_neus_pca],
            "ica": [self.enscomp_check_ens_ica, self.enscomp_check_neus_ica],
            "x2p": [self.enscomp_check_ens_x2p, self.enscomp_check_neus_x2p]
        }
        colors = []
        timecourses = []
        cells_activities = []
        new_ticks = []
        for key, ens_data in ensembles_to_compare.items():
            if requested[key][0].isEnabled() and requested[key][0].isChecked():
                new_timecourse = ens_data["timecourse"].copy()
            else:
                new_timecourse = []
            timecourses.append(new_timecourse)

            if requested[key][1].isEnabled() and requested[key][1].isChecked():
                new_members = ens_data["neus_in_ens"].copy()
                cells_activity_mat = self.data_neuronal_activity[new_members.astype(bool), :]
                cells_activity_count = np.sum(cells_activity_mat, axis=0)
            else:
                cells_activity_count = []
            cells_activities.append(cells_activity_count)

            colors.append(self.enscom_colors[key])
            new_ticks.append(key)
        
        cells_activities.reverse()
        timecourses.reverse()
        colors.reverse()
        new_ticks.reverse()

        plot_widget = self.findChild(MatplotlibWidget, 'enscomp_plot_neusact')
        plot_widget.enscomp_update_timelines(new_ticks, cells_activities, [], timecourses, colors, self.cant_timepoints)

    def get_color_svd(self):
        # Open the QColorDialog to select a color
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            self.enscom_colors['svd'] = color_hex
            self.ensembles_compare_update_ensembles()
    def get_color_pca(self):
        # Open the QColorDialog to select a color
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            self.enscom_colors['pca'] = color_hex
            self.ensembles_compare_update_ensembles()
    def get_color_ica(self):
        # Open the QColorDialog to select a color
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            self.enscom_colors['ica'] = color_hex
            self.ensembles_compare_update_ensembles()
    def get_color_x2p(self):
        # Open the QColorDialog to select a color
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            self.enscom_colors['x2p'] = color_hex
            self.ensembles_compare_update_ensembles()

    def performance_tabchange(self, index):
        if self.tempvars['performance_shown_results']:
            if index == 0:  # Correlation with ensemble presentation
                if hasattr(self, "data_stims"):
                    if not self.tempvars['performance_shown_tab0']:
                        self.tempvars['performance_shown_tab0'] = True
                        self.update_corr_stim()
            elif index == 1:    # Correlations between cells
                if not self.tempvars['performance_shown_tab1']:
                    self.tempvars['performance_shown_tab1'] = True
                    self.update_correlation_cells()
            elif index == 2:    # Cross correlations ensembles and stims
                if hasattr(self, "data_stims"):
                    if not self.tempvars['performance_shown_tab2']:
                        self.tempvars['performance_shown_tab2'] = True
                        self.update_cross_ens_stim()
            elif index == 3:    # Correlation with behavior
                if hasattr(self, "data_behavior"):
                    if not self.tempvars['performance_shown_tab3']:
                        self.tempvars['performance_shown_tab3'] = True
                        self.update_corr_behavior()
            elif index == 4:    # Cross Correlation with behavior
                if hasattr(self, "data_behavior"):
                    if not self.tempvars['performance_shown_tab4']:
                        self.tempvars['performance_shown_tab4'] = True
                        self.update_cross_behavior()
        
    def performance_check_change(self):
        methods_to_compare = []
        if self.performance_check_svd.isChecked():
            methods_to_compare.append("svd")
        if self.performance_check_pca.isChecked():
            methods_to_compare.append("pca")
        if self.performance_check_ica.isChecked():
            methods_to_compare.append("ica")
        if self.performance_check_x2p.isChecked():
            methods_to_compare.append("x2p")
        if self.performance_check_sgc.isChecked():
            methods_to_compare.append("sgc")
        self.tempvars['methods_to_compare'] = methods_to_compare
        self.tempvars['cant_methods_compare'] = len(methods_to_compare)
        if self.tempvars['cant_methods_compare'] > 0:
            self.performance_btn_compare.setEnabled(True)
        else:
            self.performance_btn_compare.setEnabled(False)

    def performance_compare(self):
        self.tempvars['performance_shown_results'] = True
        self.tempvars['performance_shown_tab0'] = False
        self.tempvars['performance_shown_tab1'] = False
        self.tempvars['performance_shown_tab2'] = False
        self.tempvars['performance_shown_tab3'] = False
        self.tempvars['performance_shown_tab4'] = False
        self.performance_tabs.setCurrentIndex(0)
        self.update_corr_stim()

    def update_corr_stim(self):
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        # Calculate correlation with stimuli
        self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrstims')
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        self.plot_widget.set_subplots(1, plot_colums)
        for m_idx, method in enumerate(methods_to_compare):
            timecourse = self.results[method]['timecourse']
            stims = self.data_stims
            correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
            self.plot_widget.plot_perf_correlations_ens_stim(correlation, m_idx, title=f"{method}".upper())            
    
    def update_correlation_cells(self):
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        # Plot the correlation of cells between themselves
        self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrcells')
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        # Find the greatest number of ensembles
        max_ens = 0
        for method in methods_to_compare:
            max_ens = max(self.results[method]['ensembles_cant'], max_ens)
        self.plot_widget.canvas.setFixedHeight(450*max_ens)

        self.plot_widget.set_subplots(max_ens, plot_colums)
        for col_idx, method in enumerate(methods_to_compare):
            for row_idx, ens in enumerate(self.results[method]['neus_in_ens']):
                members = [c_idx for c_idx in range(len(ens)) if ens[c_idx] == 1]
                activity_neus_in_ens = self.data_neuronal_activity[members, :]
                cells_names = [member+1 for member in members]
                correlation = metrics.compute_correlation_inside_ensemble(activity_neus_in_ens)
                self.plot_widget.plot_perf_correlations_cells(correlation, cells_names, col_idx, row_idx, title=f"Cells in ensemble {row_idx+1} - Method " + f"{method}".upper())

    def update_cross_ens_stim(self):
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        # Calculate cross-correlation
        self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_crossensstim')
        max_ens = 0
        for method in methods_to_compare:
            max_ens = max(self.results[method]['ensembles_cant'], max_ens)
        self.plot_widget.canvas.setFixedHeight(400*max_ens)
        self.plot_widget.set_subplots(max_ens, plot_colums)
        for m_idx, method in enumerate(methods_to_compare):
            for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                cross_corrs = []
                for stimtime in self.data_stims:
                    cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                    cross_corrs.append(cross_corr)
                self.plot_widget.plot_perf_cross_ens_stims(cross_corrs, lags, m_idx, ens_idx, title=f"Cross correlation Ensemble {ens_idx+1} and stimuli - Method " + f"{method}".upper())          

    def update_corr_behavior(self):
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        # Calculate correlation with stimuli
        self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrbehavior')
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        self.plot_widget.set_subplots(1, plot_colums)
        for m_idx, method in enumerate(methods_to_compare):
            timecourse = self.results[method]['timecourse']
            stims = self.data_behavior
            correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
            self.plot_widget.plot_perf_correlations_ens_stim(correlation, m_idx, title=f"{method}".upper())

    def update_cross_behavior(self):
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        # Calculate cross-correlation
        self.plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_crossensbehavior')
        max_ens = 0
        for method in methods_to_compare:
            max_ens = max(self.results[method]['ensembles_cant'], max_ens)
        self.plot_widget.canvas.setFixedHeight(400*max_ens)
        self.plot_widget.set_subplots(max_ens, plot_colums)
        for m_idx, method in enumerate(methods_to_compare):
            for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                cross_corrs = []
                for stimtime in self.data_behavior:
                    cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                    cross_corrs.append(cross_corr)
                self.plot_widget.plot_perf_cross_ens_stims(cross_corrs, lags, m_idx, ens_idx, title=f"Cross correlation Ensemble {ens_idx+1} and behavior - Method " + f"{method}".upper())          \
                    
    def save_results(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HDF5 Results File", "", "HDF5 Files (*.h5);;All files(*)")
        if file_path:
            self.update_console_log("Saving results file...")
            with h5py.File(file_path, 'w') as hdf_file:
                tmp = {"results": self.results}
                self.save_data_to_hdf5(hdf_file, tmp)
                tmp = {"parameters": self.params}
                self.save_data_to_hdf5(hdf_file, tmp)
            self.update_console_log("Done saving.", "complete")

    def save_data_to_hdf5(self, group, data):
        for key, value in data.items():
            print(key)
            if isinstance(value, dict):
                subgroup = group.create_group(str(key))
                self.save_data_to_hdf5(subgroup, value)
            elif isinstance(value, list):
                try:
                    group.create_dataset(key, data=value)
                except:
                    print(" -> error")
            else:
                group[key] = value

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()  