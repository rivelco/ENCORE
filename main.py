import sys
import h5py
import os
import scipy.io 
import math
import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import time
from datetime import datetime
import pickle

from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt6.QtWidgets import QTableWidgetItem, QColorDialog

from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor, QDoubleValidator, QIntValidator

from data.load_data import FileTreeModel
from data.assign_data import assign_data_from_file

import utils.metrics as metrics
from utils.text_formatting import format_nums_to_string

from gui.MatplotlibWidget import MatplotlibWidget

import matplotlib.pyplot as plt

import matlab.engine

from pprint import pprint

class WorkerSignals(QObject):
    """
    Signals used by the worker thread.

    :ivar result_ready: Signal emitted when the long-running function finishes execution and returns a result.
    :vartype result_ready: pyqtSignal(object)
    """
    result_ready = pyqtSignal(object)

class WorkerRunnable(QRunnable):
    """
    Runnable task to execute a long-running function in a separate thread.

    :param long_running_function: The function to be executed in the thread.
    :type long_running_function: callable
    :param args: Positional arguments to pass to the function.
    :type args: tuple
    :param kwargs: Keyword arguments to pass to the function.
    :type kwargs: dict
    :ivar long_running_function: The function to execute in the thread.
    :vartype long_running_function: callable
    :ivar args: Positional arguments for the function.
    :vartype args: tuple
    :ivar kwargs: Keyword arguments for the function.
    :vartype kwargs: dict
    :ivar signals: Signals for communicating the result of the function execution.
    :vartype signals: WorkerSignals
    """
    def __init__(self, long_running_function, *args, **kwargs):
        """
        Initialize the WorkerRunnable.

        :param long_running_function: The function to execute in the thread.
        :type long_running_function: callable
        :param args: Positional arguments to pass to the function.
        :type args: tuple
        :param kwargs: Keyword arguments to pass to the function.
        :type kwargs: dict
        """
        super().__init__()
        self.long_running_function = long_running_function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """
        Execute the long-running function with the provided arguments.

        :raises Exception: Propagates any exception raised by the long-running function.
        :return: Emits the result of the function execution via the `result_ready` signal.
        :rtype: None
        """
        result = self.long_running_function(*self.args, **self.kwargs)
        self.signals.result_ready.emit(result)

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        super(MainWindow, self).__init__()
        loadUi("gui/MainWindow.ui", self)
        self.setWindowTitle('Ensembles GUI')

        self.ensgui_desc = {
            "analyzer": "EnsemblesGUI",
            "date": "",
            "gui_version": 2.0
        }

        self.threadpool = QThreadPool()

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
        self.svd_defaults = {
            'pks': 3,
            'scut': 0.22,
            'hcut': 0.22,
            'state_cut': 6,
            'csi_start': 0.01,
            'csi_step': 0.01,
            'csi_end': 0.1,
            'tf_idf_norm': True,
            'parallel_processing': False
        }
        self.pca_defaults = {
            'dc': 0.01,
            'npcs': 3,
            'minspk': 3,
            'nsur': 1000,
            'prct': 99.9,
            'cent_thr': 99.9,
            'inner_corr': 5,
            'minsize': 3
        }
        self.ica_defaults = {
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
        self.x2p_defaults = {
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
        self.sgc_defaults = {
            'use_first_derivative': False,
            'standard_deviations_threshold': 2,
            'shuffling_rounds': 1000,
            'coactivity_significance_level': 0.05,
            'montecarlo_rounds': 5,
            'montecarlo_steps': 10000,
            'affinity_threshold': 0.2
        }

        ## Numeric validator
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.Notation.StandardNotation)

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
        self.btn_run_svd.clicked.connect(self.run_svd)
        ## PCA analysis
        self.pca_btn_defaults.clicked.connect(self.load_defaults_pca)
        self.btn_run_pca.clicked.connect(self.run_PCA)
        ## ICA analysis
        self.ica_btn_defaults.clicked.connect(self.load_defaults_ica)
        self.btn_run_ica.clicked.connect(self.run_ICA)
        ## X2P analysis
        self.x2p_btn_defaults.clicked.connect(self.load_defaults_x2p)
        self.btn_run_x2p.clicked.connect(self.run_x2p)
        ## SGC analysis
        self.sgc_btn_defaults.clicked.connect(self.load_defaults_sgc)
        self.btn_run_sgc.clicked.connect(self.run_sgc)

        ## Ensembles visualizer
        self.ensvis_tabs.currentChanged.connect(self.ensvis_tabchange)
        self.ensvis_btn_svd.clicked.connect(self.vis_ensembles_svd)
        self.ensvis_btn_pca.clicked.connect(self.vis_ensembles_pca)
        self.ensvis_btn_ica.clicked.connect(self.vis_ensembles_ica)
        self.ensvis_btn_x2p.clicked.connect(self.vis_ensembles_x2p)
        self.ensvis_btn_sgc.clicked.connect(self.vis_ensembles_sgc)
        self.envis_slide_selectedens.valueChanged.connect(self.update_ensemble_visualization)
        self.ensvis_check_onlyens.stateChanged.connect(self.update_ens_vis_coords)
        self.ensvis_check_onlycont.stateChanged.connect(self.update_ens_vis_coords)
        self.ensvis_check_cellnum.stateChanged.connect(self.update_ens_vis_coords)

        # Ensemble compare
        self.enscomp_slider_svd.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_pca.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_ica.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_x2p.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_sgc.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_slider_stim.valueChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_visopts_setneusize.clicked.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_visopts_showcells.stateChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_btn_color.clicked.connect(self.enscomp_get_color)
        self.enscomp_check_coords.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_ens.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_neus.stateChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_check_show_stim.stateChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_combo_select_result.currentTextChanged.connect(self.ensembles_compare_update_combo_results)

        # Populate the similarity maps combo box
        double_validator.setRange(0.0, 100.0, 2)
        self.enscomp_visopts_neusize.setValidator(double_validator)

        similarity_items = ["Neurons", "Timecourses"]
        similarity_methods = ["Cosine", "Euclidean", "Correlation", "Jaccard"]
        similarity_colors = ["viridis", "plasma", "coolwarm", "magma", "Spectral"]
        for item in similarity_items:
            self.enscomp_combo_select_simil.addItem(item)
        for method in similarity_methods:
            self.enscomp_combo_select_simil_method.addItem(method)
        for color in similarity_colors:
            self.enscomp_combo_select_simil_colormap.addItem(color)
        self.enscomp_combo_select_simil.setEnabled(False)
        self.enscomp_combo_select_simil_method.setEnabled(False)
        self.enscomp_combo_select_simil_colormap.setEnabled(False)
        self.enscomp_combo_select_simil.currentTextChanged.connect(self.ensembles_compare_similarity_update_combbox)
        
        self.enscomp_combo_select_simil.setCurrentText("Neurons")
        self.enscomp_combo_select_simil_method.setCurrentText("Jaccard")
        self.enscomp_combo_select_simil_colormap.setCurrentText("viridis")

        # Connect the combo box to a function that handles selection changes
        self.enscomp_combo_select_simil_method.currentTextChanged.connect(self.ensembles_compare_similarity)
        self.enscomp_combo_select_simil_colormap.currentTextChanged.connect(self.ensembles_compare_similarity)
        self.enscomp_tabs.currentChanged.connect(self.ensembles_compare_tabchange)

        ## Performance
        self.performance_tabs.currentChanged.connect(self.performance_tabchange)
        self.performance_check_svd.stateChanged.connect(self.performance_check_change)
        self.performance_check_pca.stateChanged.connect(self.performance_check_change)
        self.performance_check_ica.stateChanged.connect(self.performance_check_change)
        self.performance_check_x2p.stateChanged.connect(self.performance_check_change)
        self.performance_check_sgc.stateChanged.connect(self.performance_check_change)
        self.performance_btn_compare.clicked.connect(self.performance_compare)

        # Saving
        self.save_btn_hdf5.clicked.connect(self.save_results_hdf5)
        self.save_btn_pkl.clicked.connect(self.save_results_pkl)
        self.save_btn_mat.clicked.connect(self.save_results_mat)
        
    def update_console_log(self, message, msg_type="log"):
        """
        Updates the console log with a new message, formatted with a specific color based on the message type.

        :param message: The message to be displayed in the console log.
        :type message: str
        :param msg_type: The type of the message, which determines its color. 
                        It can be one of "log", "error", "warning", or "complete". 
                        Defaults to "log".
        :type msg_type: str, optional
        :return: None

        This method formats the log entry by including the current timestamp and the provided message. The message 
        is displayed in a monospace font with different colors based on the message type. The new message is appended 
        to the console log, and the view is updated to scroll to the bottom to ensure the message is visible.
        """
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
        """
        Reset the GUI, delete all the variables and analysis results.
        Also clears all the figures and restores the analysis and display options.
        The result is the GUI just like the first time you opened it.
        """
        # Delete all previous results
        self.results = {}
        self.algotrithm_results = {}
        self.params = {}
        self.varlabels = {}
        self.tempvars = {}
        
        # Initialize buttons
        self.btn_run_svd.setEnabled(False)
        self.btn_run_pca.setEnabled(False)
        self.btn_run_ica.setEnabled(False)
        self.btn_run_x2p.setEnabled(False)

        self.ensvis_btn_svd.setEnabled(False)
        self.ensvis_btn_pca.setEnabled(False)
        self.ensvis_btn_ica.setEnabled(False)
        self.ensvis_btn_x2p.setEnabled(False)
        self.ensvis_btn_sgc.setEnabled(False)

        # Ensemble performance selectors
        check_boxes = [self.performance_check_svd,
                       self.performance_check_pca,
                       self.performance_check_ica,
                       self.performance_check_x2p,
                       self.performance_check_sgc]
        for obj in check_boxes:
            obj.setEnabled(False)
            obj.setChecked(False)
        self.performance_btn_compare.setEnabled(False)

        # Save tab
        save_itms = [self.save_check_input,
                self.save_check_minimal,
                self.save_check_params,
                self.save_check_full,
                self.save_check_enscomp,
                self.save_check_perf]
        for itm in save_itms:
            itm.setChecked(True)
            itm.setEnabled(False)
        save_btns = [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]
        for btn in save_btns:
            btn.setEnabled(False)

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

        default_txt = "Perform the SGC analysis to see results"
        self.findChild(MatplotlibWidget, 'sgc_plot_timecourse').reset(default_txt)
        self.findChild(MatplotlibWidget, 'sgc_plot_cellsinens').reset(default_txt)

        self.ensvis_edit_numens.setText("")
        self.envis_slide_selectedens.blockSignals(True)
        self.envis_slide_selectedens.setMaximum(2)
        self.envis_slide_selectedens.setValue(1)
        self.envis_slide_selectedens.blockSignals(False)
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
        
        self.tempvars['ensvis_shown_results'] = False
        self.tempvars['ensvis_shown_tab1'] = False
        self.tempvars['ensvis_shown_tab2'] = False
        self.tempvars['ensvis_shown_tab3'] = False
        self.tempvars['ensvis_shown_tab4'] = False
        self.ensvis_tabs.setCurrentIndex(0)

        # Ensembles compare
        self.enscomp_visopts = {
            "svd": {'enscomp_check_coords': True, 'enscomp_check_ens': True, 'enscomp_check_neus': False, 'color': 'red', 'enabled': False},
            "pca": {'enscomp_check_coords': True, 'enscomp_check_ens': True, 'enscomp_check_neus': False, 'color': 'blue', 'enabled': False},
            "ica": {'enscomp_check_coords': True, 'enscomp_check_ens': True, 'enscomp_check_neus': False, 'color': 'green', 'enabled': False},
            "x2p": {'enscomp_check_coords': True, 'enscomp_check_ens': True, 'enscomp_check_neus': False, 'color': 'orange', 'enabled': False},
            "sgc": {'enscomp_check_coords': True, 'enscomp_check_ens': True, 'enscomp_check_neus': False, 'color': 'pink', 'enabled': False},
            "stims": {'color': 'black'},
            "behavior": {'color': 'yellow'},
            "sim_neus": {'method': 'Jaccard', 'colormap': 'viridis'},
            "sim_time": {'method': 'Cosine', 'colormap': 'plasma'},
        }
        self.tempvars["showed_sim_maps"] = False

        # The general options
        options_objs = [self.enscomp_visopts_showcells, 
                        self.enscomp_visopts_neusize, 
                        self.enscomp_visopts_setneusize]
        for obj in options_objs:
            obj.blockSignals(True)
            obj.setEnabled(False)
            obj.blockSignals(False)

        # Clean the combo box of results
        elems_in_combox = self.enscomp_combo_select_result.count()
        self.enscomp_combo_select_result.blockSignals(True)
        for elem in range(elems_in_combox):
            self.enscomp_combo_select_result.removeItem(elem)
        self.enscomp_combo_select_result.blockSignals(False)

        self.enscomp_combo_select_simil.setEnabled(False)
        self.enscomp_combo_select_simil_method.setEnabled(False)
        self.enscomp_combo_select_simil_colormap.setEnabled(False)

        self.enscomp_check_coords.setEnabled(False)
        self.enscomp_check_ens.setEnabled(False)
        self.enscomp_check_neus.setEnabled(False)
        self.enscomp_btn_color.setEnabled(False)

        options_objs = [(self.enscomp_check_coords, True), 
                        (self.enscomp_check_ens, True), 
                        (self.enscomp_check_neus, False)]
        for obj, val in options_objs:
            obj.blockSignals(True)
            obj.setChecked(val)
            obj.blockSignals(False)
        
        # Sliders
        sliders = [self.enscomp_slider_svd, 
                   self.enscomp_slider_pca, 
                   self.enscomp_slider_ica,
                   self.enscomp_slider_x2p]
        for obj in sliders:
            obj.blockSignals(True)
            obj.setEnabled(False)
            obj.setMinimum(1)
            obj.setMaximum(2)
            obj.setValue(1)
            obj.blockSignals(False)

        slider_labels = [self.enscomp_slider_lbl_min_svd,
                         self.enscomp_slider_lbl_max_svd,
                         self.enscomp_slider_lbl_min_pca,
                         self.enscomp_slider_lbl_max_pca,
                         self.enscomp_slider_lbl_min_ica,
                         self.enscomp_slider_lbl_max_ica,
                         self.enscomp_slider_lbl_min_x2p,
                         self.enscomp_slider_lbl_max_x2p]
        for obj in slider_labels:
            obj.setEnabled(False)
            obj.setText("1")
            
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

        default_txt = "Perform and select at least one analysis\n to see the metrics"
        self.findChild(MatplotlibWidget, 'enscomp_plot_map').reset(default_txt)
        self.findChild(MatplotlibWidget, 'enscomp_plot_neusact').reset(default_txt)
        self.findChild(MatplotlibWidget, 'enscomp_plot_sim_elements').reset(default_txt)
        self.findChild(MatplotlibWidget, 'enscomp_plot_sim_times').reset(default_txt)

    def browse_files(self):
        """
        Opens a file browing dialog to open a new file.
        
        This function also loads the file tree model and shows it in the variable browser widget.
        """
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
            elif file_extension == ".pkl":
                self.update_console_log("Generating file structure...")
                with open(fname, 'rb') as file:
                    pkl_file = pickle.load(file)
                self.file_model_type = "pkl"
                self.file_model = FileTreeModel(pkl_file, model_type="pkl")
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
        """
        Handles the click event on a variable in the :attr:`MainWindow.tree_view`, displaying relevant information 
        about the selected variable and enabling or disabling assign and clear buttons in the UI.

        :param index: The index of the clicked item in the file model.
        :type index: QModelIndex

        :return: None

        This method retrieves information about the selected variable, including its path in the file, type, and size. 
        The item's name is extracted from the path and displayed along with its description in the UI. 
        Depending on the type and size of the item, it enables or disables specific buttons related 
        to the assigning to a dataset. The relevant information is also stored for further processing.
        """
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
        """
        Validates that all required data for a given analysis are present in the current session.

        :param needed_data: A list of strings representing the names of the required attributes.
        :type needed_data: list of str

        :return: True if all required attributes are present, False otherwise.
        :rtype: bool

        This method checks whether the object has the necessary attributes as specified in the 
        `needed_data` list. If any required attribute is missing, it returns False, indicating 
        that the data is not valid. Otherwise, it returns True.
        """
        valid_data = True
        for req in needed_data:
            if not hasattr(self, req):
                valid_data = False
        return valid_data

    ## Identify the tab changes
    def main_tabs_change(self, index):
        """
        Identifies the change in tabs to load some data.

        This function allows the asynchronous verification and loading of some tabs identified by index.
        For the analysis tabs, the necessary data is evaluated and then change the text accordingly.
        For the ensembles compare tab the visualizations are only loaded when the user reaches this tab.

        :param index: Index of the currently open tab, 0 indexing.
        :type index: int
        """
        if index > 0 and index < 6: # Analysis tabs
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
            if hasattr(self, "data_dFFo"):
                self.lbl_sgc_spikes_selected.setText(f"Loaded")
            else:
                self.lbl_sgc_spikes_selected.setText(f"Nothing selected")

            # Validate data for SVD
            needed_data = ["data_neuronal_activity"]
            self.btn_run_svd.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for PCA
            needed_data = ["data_neuronal_activity"]
            self.btn_run_pca.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for ICA
            needed_data = ["data_neuronal_activity"]
            self.btn_run_ica.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for x2p
            needed_data = ["data_neuronal_activity"]
            self.btn_run_x2p.setEnabled(self.validate_needed_data(needed_data))

            # Validate needed data for sgc
            needed_data = ["data_dFFo"]
            self.btn_run_sgc.setEnabled(self.validate_needed_data(needed_data))

        if index == 7: #Ensembles compare tab
            if len(self.results) > 0:
                self.ensembles_compare_update_ensembles()

    ## Set variables from input file
    def set_dFFo(self):
        """
        Sets the :attr:`MainWindow.dFFo` dataset by assigning data from a file and updating the relevant UI components.

        This method loads the dFFo dataset from the selected file.
        It then updates the UI to reflect that the dataset has been
        assigned, enables buttons for further edition, and logs the update message.

        :return: None
        """
        self.data_dFFo = assign_data_from_file(self)
        neus, frames = self.data_dFFo.shape
        self.btn_clear_dFFo.setEnabled(True)
        self.btn_view_dFFo.setEnabled(True)
        self.lbl_dffo_select.setText("Assigned")
        self.lbl_dffo_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set dFFo dataset - Identified {neus} cells and {frames} time points. Please, verify the data preview.", msg_type="complete")
        self.view_dFFo()
        self.save_check_input.setEnabled(True)
        for btn in [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]:
            btn.setEnabled(True)
    def set_neuronal_activity(self):
        """
        Sets the :attr:`MainWindow.data_neuronal_activity` dataset by assigning data from a file and updating the relevant UI components.

        This method loads the data_neuronal_activity dataset from the selected file and extracts the number of cells 
        (cant_neurons) and time points (cant_timepoints). It then updates the UI to reflect that the dataset has been
        assigned, enables buttons for further edition, and logs the update message.

        :return: None
        """
        self.data_neuronal_activity = assign_data_from_file(self)
        self.cant_neurons, self.cant_timepoints = self.data_neuronal_activity.shape
        self.btn_clear_neuronal_activity.setEnabled(True)
        self.btn_view_neuronal_activity.setEnabled(True)
        self.lbl_neuronal_activity_select.setText("Assigned")
        self.lbl_neuronal_activity_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Binary Neuronal Activity dataset - Identified {self.cant_neurons} cells and {self.cant_timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_neuronal_activity()
        self.save_check_input.setEnabled(True)
        for btn in [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]:
            btn.setEnabled(True)
    def set_coordinates(self):
        """
        Sets the value of the :attr:`MainWindow.data_coordinates` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        Only the first two elements of the second dimention are assigned.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_coordinates = assign_data_from_file(self)
        self.data_coordinates = data_coordinates[:, 0:2]
        neus, dims = self.data_coordinates.shape
        self.btn_clear_coordinates.setEnabled(True)
        self.btn_view_coordinates.setEnabled(True)
        self.lbl_coordinates_select.setText("Assigned")
        self.lbl_coordinates_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Coordinates dataset - Identified {neus} cells and {dims} dimentions. Please, verify the data preview.", msg_type="complete")
        self.view_coordinates()
        self.save_check_input.setEnabled(True)
        for btn in [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]:
            btn.setEnabled(True)
    def set_stims(self):
        """
        Sets the value of the :attr:`MainWindow.data_stims` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_stims = assign_data_from_file(self)
        self.data_stims = data_stims
        stims, timepoints = data_stims.shape
        self.btn_clear_stim.setEnabled(True)
        self.btn_view_stim.setEnabled(True)
        self.lbl_stim_select.setText("Assigned")
        self.lbl_stim_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Stimuli dataset - Identified {stims} stims and {timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_stims()
        self.save_check_input.setEnabled(True)
        for btn in [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]:
            btn.setEnabled(True)
    def set_cells(self):
        """
        Sets the value of the :attr:`MainWindow.data_cells` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_cells = assign_data_from_file(self)
        self.data_cells = data_cells
        stims, cells = data_cells.shape
        self.btn_clear_cells.setEnabled(True)
        self.btn_view_cells.setEnabled(True)
        self.lbl_cells_select.setText("Assigned")
        self.lbl_cells_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Selected cells dataset - Identified {stims} groups and {cells} cells. Please, verify the data preview.", msg_type="complete")
        self.view_cells()
        self.save_check_input.setEnabled(True)
        for btn in [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]:
            btn.setEnabled(True)
    def set_behavior(self):
        """
        Sets the value of the :attr:`MainWindow.data_behavior` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_behavior = assign_data_from_file(self)
        self.data_behavior = data_behavior
        behaviors, timepoints = data_behavior.shape
        self.btn_clear_behavior.setEnabled(True)
        self.btn_view_behavior.setEnabled(True)
        self.lbl_behavior_select.setText("Assigned")
        self.lbl_behavior_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Behavior dataset - Identified {behaviors} behaviors and {timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_behavior()
        self.save_check_input.setEnabled(True)
        for btn in [self.save_btn_hdf5, self.save_btn_pkl, self.save_btn_mat]:
            btn.setEnabled(True)
    
    def set_able_edit_options(self, boolval):
        """
        Changes the enabled status of the editing options.

        :param boolval: When true, all the buttons for editing are enabled, dissabled otherwise.
        :type boolval: bool
        """
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
        """
        Deletes the value of the :attr:`MainWindow.data_dFFo` variable.

        This function also updates the buttons related to load and clear the variable
        At the end the function clears visualization in the :attr:`MainWindow.data_preview` widget.
        """
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
        """
        Deletes the value of the :attr:`MainWindow.data_neuronal_activity` variable.

        This function also updates the buttons related to load and clear the variable
        At the end the function clears visualization in the :attr:`MainWindow.data_preview` widget.
        """
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
        """
        Deletes the value of the :attr:`MainWindow.data_coordinates` variable.

        This function also updates the buttons related to load and clear the variable
        At the end the function clears visualization in the :attr:`MainWindow.data_preview` widget.
        """
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
        """
        Deletes the value of the :attr:`MainWindow.data_stims` variable.

        This function also updates the buttons related to load and clear the variable
        At the end the function clears visualization in the :attr:`MainWindow.data_preview` widget.
        """
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
        """
        Deletes the value of the :attr:`MainWindow.data_cells` variable.

        This function also updates the buttons related to load and clear the variable
        At the end the function clears visualization in the :attr:`MainWindow.data_preview` widget.
        """
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
        """
        Deletes the value of the :attr:`MainWindow.data_behavior` variable.

        This function also updates the buttons related to load and clear the variable
        At the end the function clears visualization in the :attr:`MainWindow.data_preview` widget.
        """
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
        """
        Displays the data saved in the :attr:`MainWindow.dFFo` variable.

        This function also runs update the status for editing the variable
        using :meth:`MainWindow.set_able_edit_options` and updates validators
        using :meth:`MainWindow.update_edit_validators`.
        This function also triggers the setup of the variable labels tab
        using :meth:`MainWindow.varlabels_setup_tab`.
        """
        self.currently_visualizing = "dFFo"
        self.set_able_edit_options(True)
        self.update_edit_validators(lim_sup_x=self.data_dFFo.shape[1], lim_sup_y=self.data_dFFo.shape[0])
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        cell_labels = list(self.varlabels["cell"].values()) if "cell" in self.varlabels else []
        plot_widget.preview_dataset(self.data_dFFo, ylabel='Cell', yitems_labels=cell_labels)
        self.varlabels_setup_tab(self.data_dFFo.shape[0])
    def view_neuronal_activity(self):
        """
        Displays the data saved in the :attr:`MainWindow.neuronal_activity` variable.

        This function also runs update the status for editing the variable
        using :meth:`MainWindow.set_able_edit_options` and updates validators
        using :meth:`MainWindow.update_edit_validators`.
        This function also triggers the setup of the variable labels tab
        using :meth:`MainWindow.varlabels_setup_tab`.
        """
        self.currently_visualizing = "neuronal_activity"
        self.set_able_edit_options(True)
        self.update_edit_validators(lim_sup_x=self.data_neuronal_activity.shape[1], lim_sup_y=self.data_neuronal_activity.shape[0])
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        cell_labels = list(self.varlabels["cell"].values()) if "cell" in self.varlabels else []
        plot_widget.preview_dataset(self.data_neuronal_activity==0, ylabel='Cell', cmap='gray', yitems_labels=cell_labels)
        self.varlabels_setup_tab(self.data_neuronal_activity.shape[0])
    def view_coordinates(self):
        """
        Displays the data saved in the :attr:`MainWindow.coordinates` variable.

        This function also runs update the status for editing the variable
        using :meth:`MainWindow.set_able_edit_options` and updates validators
        using :meth:`MainWindow.update_edit_validators`.
        This function also triggers the setup of the variable labels tab
        using :meth:`MainWindow.varlabels_setup_tab`.
        """
        self.currently_visualizing = "coordinates"
        self.set_able_edit_options(True)
        self.update_edit_validators(lim_sup_x=2, lim_sup_y=self.data_coordinates.shape[0])
        self.plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        self.plot_widget.preview_coordinates2D(self.data_coordinates)
        self.varlabels_setup_tab(self.data_coordinates.shape[0])
    def view_stims(self):
        """
        Displays the data saved in the :attr:`MainWindow.stims` variable.

        This function also runs update the status for editing the variable
        using :meth:`MainWindow.set_able_edit_options` and updates validators
        using :meth:`MainWindow.update_edit_validators`.
        This function also triggers the setup of the variable labels tab
        using :meth:`MainWindow.varlabels_setup_tab`.
        """
        self.currently_visualizing = "stims"
        self.set_able_edit_options(True)
        self.update_edit_validators(lim_sup_x=self.data_stims.shape[1], lim_sup_y=self.data_stims.shape[0])
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_stims
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        self.update_enscomp_options("stims")
        stim_labels = list(self.varlabels["stim"].values()) if "stim" in self.varlabels else []
        plot_widget.preview_dataset(preview_data==0, ylabel='Stim', cmap='gray', yitems_labels=stim_labels)
    def view_cells(self):
        """
        Displays the data saved in the :attr:`MainWindow.cells` variable.

        This function also runs update the status for editing the variable
        using :meth:`MainWindow.set_able_edit_options` and updates validators
        using :meth:`MainWindow.update_edit_validators`.
        This function also triggers the setup of the variable labels tab
        using :meth:`MainWindow.varlabels_setup_tab`.
        """
        self.currently_visualizing = "cells"
        self.set_able_edit_options(True)
        self.update_edit_validators(lim_sup_x=self.data_cells.shape[1], lim_sup_y=self.data_cells.shape[0])
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_cells
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        selectcell_labels = list(self.varlabels["selected_cell"].values()) if "selected_cell" in self.varlabels else []
        plot_widget.preview_dataset(preview_data==0, xlabel="Cell", ylabel='Group', cmap='gray', yitems_labels=selectcell_labels)
    def view_behavior(self):
        """
        Displays the data saved in the :attr:`MainWindow.behavior` variable.

        This function also runs update the status for editing the variable
        using :meth:`MainWindow.set_able_edit_options` and updates validators
        using :meth:`MainWindow.update_edit_validators`.
        This function also triggers the setup of the variable labels tab
        using :meth:`MainWindow.varlabels_setup_tab`.
        """
        self.currently_visualizing = "behavior"
        self.set_able_edit_options(True)
        self.update_edit_validators(lim_sup_x=self.data_behavior.shape[1], lim_sup_y=self.data_behavior.shape[0])
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_behavior
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        self.update_enscomp_options("behavior")
        behavior_labels = list(self.varlabels["behavior"].values()) if "behavior" in self.varlabels else []
        plot_widget.preview_dataset(preview_data, ylabel='Behavior', yitems_labels=behavior_labels)

    ## Edit buttons
    def edit_transpose(self):
        """
        Transpose the currently visualized variable.

        Simple transpose of the data currently viewed in the MainWindow.data_preview widget.
        This method changes both the stored data and the visualization.
        Also, a message is shown to the user about the interpretation of the transpose.
        """
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

    def update_edit_validators(self, lim_sup_x=10000000, lim_sup_y=10000000):
        """
        Update the validators for bining and slicing.

        :param lim_sup_x: Maximum value possible for the x dimention, defaults to 10000000
        :type lim_sup_x: int, optional
        :param lim_sup_y: Mavimum value possible for the y dimention, defaults to 10000000
        :type lim_sup_y: int, optional
        """
        # For the edit options
        int_validator = QIntValidator(0, lim_sup_x)
        self.edit_edit_binsize.setValidator(int_validator)
        self.edit_edit_xstart.setValidator(int_validator)
        self.edit_edit_xend.setValidator(int_validator)
        int_validator = QIntValidator(0, lim_sup_y)
        self.edit_edit_ystart.setValidator(int_validator)
        self.edit_edit_yend.setValidator(int_validator)

    def bin_matrix(self, mat, bin_size, bin_method):
        """
        Bins the input matrix along the timepoints dimension using the specified binning method.

        :param mat: The input matrix to be binned, where each row represents an element and each column represents a timepoint.
        :type mat: numpy.ndarray of shape (n, t)
        :param bin_size: The number of timepoints to include in each bin. Must be smaller than the number of timepoints in `mat`.
        :type bin_size: int
        :param bin_method: The method used to bin the data. Options are "mean" to compute the mean of each bin, or "sum" to compute the sum.
        :type bin_method: str
        :raises ValueError: If `bin_method` is not one of the accepted values ("mean" or "sum").
        :return: A matrix with the same number of rows as `mat` but with columns reduced by the binning operation.
        :rtype: numpy.ndarray of shape (n, b)

        :Example:

            >>> mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> bin_size = 2
            >>> bin_method = "mean"
            >>> binned_mat = self.bin_matrix(mat, bin_size, bin_method)
            >>> print(binned_mat)
            array([[1.5, 3.5],
                [5.5, 7.5]])

        """
        elements, timepoints = mat.shape
        if bin_size >= timepoints:
            self.update_console_log(f"Enter a bin size smaller than the current amount of timepoints. Nothing has been changed.", "warning")
            return mat   
        num_bins = timepoints // bin_size
        bin_mat = np.zeros((elements, num_bins))
        for i in range(num_bins):
            if bin_method == "mean":
                bin_mat[:, i] = np.mean(mat[:, i*bin_size:(i+1)*bin_size], axis=1)
            elif bin_method == "sum":
                bin_mat[:, i] = np.sum(mat[:, i*bin_size:(i+1)*bin_size], axis=1)
            else:
                raise ValueError("Invalid bin_method. Use 'mean' or 'sum'.")
        return bin_mat 
    def edit_bin(self):
        """
        Reads the binning parameters and performs the binning in the selected dataset.

        Loads the bin size and bin method from the UI, and shows feedback to the user
        about the performed operation. If the input is invalid nothing is changed.
        """
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
        """
        Edits the currently visualized matrix by trimming its rows and/or columns based on user-specified indices.

        :return: None

        This method uses the start and end indices entered by the user to trim the matrix currently selected for editing.
        Depending on the selection, it will trim one of the following datasets: dFFo, neuronal_activity, coordinates, stims, 
        cells, or behavior. For each dataset, if valid start and end indices are provided for the x-axis or y-axis, 
        it slices the data accordingly and updates the view for that dataset.

        After trimming, the console log is updated to notify the user, and the appropriate preview function is called to 
        display the updated dataset.
        """
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
        """
        Sets up the table for labeling variables according to the currently visualized dataset.

        :param rows_cant: The number of rows to display in the table.
        :type rows_cant: int
        :return: None

        Based on the dataset currently selected for viewing (:attr:`MainWindow.currently_visualizing`), this function configures the label 
        for the index column and initializes the label family. It sets the row count of :attr:`MainWindow.table_setlabels` to `rows_cant`, 
        populates the table with index values in the first column, and sets up editable label fields in the second column. 
        If labels have already been registered for the selected dataset type, these are pre-filled in the table; 
        otherwise, previous entries are cleared.

        The first column items are set as non-editable to prevent user modification of index values.
        """
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
            self.table_setlabels.setItem(row, 0, QTableWidgetItem(str(row)))
            item = self.table_setlabels.item(row, 0)
            if item is not None: # Remove the ItemIsEditable flag
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if labels_registered:
                if row in self.varlabels[label_family]:
                    self.table_setlabels.setItem(row, 1, QTableWidgetItem(self.varlabels[label_family][row]))
            else:   # To clear out all the previous entries
                self.table_setlabels.setItem(row, 1, QTableWidgetItem(None))
    def varlabels_save(self):
        """
        Saves the labels entered by the user for the currently visualized dataset.

        :return: None

        Based on the current dataset selected (:attr:`MainWindow.currently_visualizing`), this function assigns a label family, retrieves 
        the label values entered by the user in the second column of :attr:`MainWindow.table_setlabels`, and stores them in the :attr:`MainWindow.varlabels` 
        dictionary under the corresponding label family. If a label is missing for any row, the row index is used as the 
        default label.

        After saving, the function updates the view of the current dataset to reflect any changes and logs a message 
        to notify the user.
        """
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
            new_label = str(row)
            if item is not None:
                if len(item.text()) > 0:
                    new_label = item.text()
            self.varlabels[label_family][row] = new_label
        if curr_view == "dFFo":
            self.view_dFFo()
        elif curr_view == "neuronal_activity":
            self.view_neuronal_activity()
        elif curr_view == "stims":
            self.view_stims()
        elif curr_view == "cells":
            self.view_cells()
        elif curr_view == "behavior":
            self.view_behavior()
        self.update_console_log(f"Saved {label_family} labels. Please, verify the data preview.", "warning")
    def varlabels_clear(self):
        """
        Clears the saved labels for the currently visualized dataset.

        :return: None

        Determines the label family based on the currently visualized dataset (:attr:`MainWindow.currently_visualizing`) and removes 
        the corresponding entries from the :attr:`MainWindow.varlabels` dictionary if they exist. This action clears all custom labels 
        previously assigned to the dataset.

        After clearing the labels, the function updates the view of the current dataset to reflect the removal of labels 
        and reinitializes the :attr:`MainWindow.table_setlabels` widget to show empty label entries.
        """
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
            
        if curr_view == "dFFo":
            self.view_dFFo()
        elif curr_view == "neuronal_activity":
            self.view_neuronal_activity()
        elif curr_view == "stims":
            self.view_stims()
        elif curr_view == "cells":
            self.view_cells()
        elif curr_view == "behavior":
            self.view_behavior()
        self.varlabels_setup_tab(self.table_setlabels.rowCount())
        
    def dict_to_matlab_struct(self, pars_dict):
        """
        Converts a Python dictionary to a MATLAB struct.

        :param pars_dict: A dictionary where keys represent the names of fields in the MATLAB struct and the values 
                        represent the corresponding field values.
        :type pars_dict: dict

        :return: A MATLAB struct where the keys are the field names and the values are converted to the appropriate MATLAB data type.
        :rtype: dict

        This function recursively converts a Python dictionary to a MATLAB struct. It handles nested dictionaries by 
        recursively calling the conversion function. Numeric values (integers or floats) are converted into MATLAB 
        double type, while other data types are kept unchanged.
        """
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
        """
        Loads default SVD parameter values into the UI fields.

        This method retrieves the default parameter values for SVD analysis from `MainWindow.svd_defaults` 
        and sets the corresponding values in the UI fields. It also updates the console log to 
        indicate that the default values have been successfully loaded.

        :return: None
        """
        defaults = self.svd_defaults
        self.svd_edit_pks.setText(f"{defaults['pks']}")
        self.svd_edit_scut.setText(f"{defaults['scut']}")
        self.svd_edit_hcut.setText(f"{defaults['hcut']}")
        self.svd_edit_statecut.setText(f"{defaults['state_cut']}")
        self.svd_edit_csistart.setText(f"{defaults['csi_start']}")
        self.svd_edit_csistep.setText(f"{defaults['csi_step']}")
        self.svd_edit_csiend.setText(f"{defaults['csi_end']}")
        self.svd_check_tfidf.setChecked(defaults['tf_idf_norm'])
        self.svd_check_parallel.setChecked(defaults['parallel_processing'])
        self.update_console_log("Loaded default SVD parameter values", "complete")
    def run_svd(self):
        """
        Retrieves user-defined parameters for Singular Value Decomposition (SVD) from the GUI, applies default values 
        if fields are empty, and initiates the SVD analysis in parallel. The function also updates the console log 
        with messages about the current status and resets any previously displayed SVD figures.

        :return: None
        :rtype: None
        """
        # Temporarly disable the button
        self.btn_run_svd.setEnabled(False)
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
        parallel_computing = self.svd_check_parallel.isChecked()

        # Pack parameters
        pars = {
            'pks': val_pks,
            'scut': val_scut,
            'hcut': val_hcut,
            'statecut': val_statecut,
            'tf_idf_norm': val_idtfd,
            'csi_start': val_csistart,
            'csi_step': val_csistep,
            'csi_end': val_csiend,
            'parallel_processing': parallel_computing
        }
        self.params['svd'] = pars
        pars_matlab = self.dict_to_matlab_struct(pars)

        # Clean all the figures in case there was something previously
        if 'svd' in self.results:
            del self.results['svd']
        algorithm_figs = ["svd_plot_similaritymap", "svd_plot_binarysimmap", "svd_plot_singularvalues", "svd_plot_components", "svd_plot_timecourse", "svd_plot_cellsinens"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        # Run the SVD in parallel
        self.update_console_log("Performing SVD...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        worker_svd = WorkerRunnable(self.run_svd_parallel, spikes, coords_foo, pars_matlab)
        worker_svd.signals.result_ready.connect(self.run_svd_parallel_end)
        self.threadpool.start(worker_svd)
    def run_svd_parallel(self, spikes, coords_foo, pars_matlab):
        """
        Initializes and runs the MATLAB engine to execute the SVD algorithm on neural activity data in parallel. 
        This function also handles MATLAB path setup, updates parameter values in the GUI, and plots the results.

        :param spikes: Matrix of neural activity data to be processed.
        :type spikes: matlab.double
        :param coords_foo: Matrix of dummy coordinates used as input for the SVD function.
        :type coords_foo: matlab.double
        :param pars_matlab: MATLAB structure of parameters for the SVD algorithm.
        :type pars_matlab: dict
        :return: List of times taken for MATLAB engine setup, SVD execution, and plotting.
        :rtype: list[float]
        """
        log_flag = "GUI SVD:"
        print(f"{log_flag} Starting MATLAB engine...")
        start_time = time.time()
        eng_svd = matlab.engine.start_matlab()
        # Adding to path
        relative_folder_path = 'analysis/SVD'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng_svd.genpath(folder_path)
        eng_svd.addpath(folder_path_with_subfolders, nargout=0)
        end_time = time.time()
        engine_time = end_time - start_time
        print(f"{log_flag} Loaded MATLAB engine.")
        start_time = time.time()
        try:
            answer = eng_svd.Stoixeion(spikes, coords_foo, pars_matlab)
        except:
            print(f"{log_flag} An error occurred while excecuting the algorithm. Check console logs for more info.")
            answer = None
        end_time = time.time()
        algorithm_time = end_time - start_time
        print(f"{log_flag} Done.")
        print(f"{log_flag} Terminating MATLAB engine...")
        eng_svd.quit()
        print(f"{log_flag} Done.")
        plot_times = 0
        if answer != None:
            self.algotrithm_results['svd'] = answer
            # Update pks and scut in case of automatic calculation
            self.svd_edit_pks.setText(f"{int(answer['pks'])}")
            self.svd_edit_scut.setText(f"{answer['scut']}")
            # Plotting results
            print(f"{log_flag} Plotting and saving results...")
            # For this method the saving occurs in the same plotting function to avoid recomputation
            start_time = time.time()
            self.plot_SVD_results(answer)
            end_time = time.time()
            plot_times = end_time - start_time
            print(f"{log_flag} Done plotting and saving...")
        return [engine_time, algorithm_time, plot_times]
    def run_svd_parallel_end(self, times):
        """
        Finalizes the SVD execution process, logging timing information for each stage of the computation, 
        and re-enables the SVD run button.

        :param times: List containing the time taken for MATLAB engine loading, algorithm execution, 
                    and plotting in seconds.
        :type times: list[float]
        :return: None
        :rtype: None
        """
        self.update_console_log("Done executing the SVD algorithm", "complete") 
        self.update_console_log(f"- Loading the engine took {times[0]:.2f} seconds") 
        self.update_console_log(f"- Running the algorithm took {times[1]:.2f} seconds") 
        self.update_console_log(f"- Plotting and saving results took {times[2]:.2f} seconds")
        self.btn_run_svd.setEnabled(True)
    def plot_SVD_results(self, answer):
        """
        Plots and saves the results of the SVD algorithm, including similarity maps, singular values, 
        components, and ensemble timecourses.

        :param answer: Dictionary containing the SVD output data from MATLAB, including matrices for 
                    similarity maps, singular values, component vectors, ensemble timecourses, 
                    and neuron groupings.
        :type answer: dict
        :return: None
        :rtype: None
        """
        # Similarity map
        simmap = np.array(answer['S_index_ti'])
        plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_similaritymap')
        plot_widget.preview_dataset(simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='jet', aspect='equal')
        # Binary similarity map
        bin_simmap = np.array(answer['S_indexp'])
        plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_binarysimmap')
        plot_widget.preview_dataset(bin_simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='gray', aspect='equal')
        # Singular values plot
        singular_vals = np.diagonal(np.array(answer['S_svd']))
        num_state = int(answer['num_state'])
        plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_singularvalues')
        plot_widget.plot_singular_values(singular_vals, num_state)

        # Components from the descomposition
        singular_vals = np.array(answer['svd_sig'])
        plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_components')
        rows = math.ceil(math.sqrt(num_state))
        cols = math.ceil(num_state / rows)
        plot_widget.set_subplots(rows, cols)
        for state_idx in range(num_state):
            curent_comp = singular_vals[:, :, state_idx]
            row = state_idx // cols
            col = state_idx % cols
            plot_widget.plot_states_from_svd(curent_comp, state_idx, row, col)
            
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
        plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_timecourse')
        plot_widget.plot_ensembles_timecourse(ensembles_timecourse)

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

        plot_widget = self.findChild(MatplotlibWidget, 'svd_plot_cellsinens')
        plot_widget.plot_ensembles_timecourse(neurons_in_ensembles, xlabel="Cell")

    def load_defaults_pca(self):
        """
        Loads default PCA parameter values into the UI fields.

        This method retrieves the default parameter values for PCA analysis from `MainWindow.pca_defaults` 
        and sets the corresponding values in the UI fields. It also updates the console log to 
        indicate that the default values have been successfully loaded.

        :return: None
        """
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
        """
        Retrieves user-defined parameters for PCA from the GUI, applies default values 
        if fields are empty, and initiates the PCA analysis in parallel. The function also updates the console log 
        with messages about the current status and resets any previously displayed PCA figures.

        :return: None
        :rtype: None
        """
        # Temporarly disable the button
        self.btn_run_pca.setEnabled(False)
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

        # Clean all the figures in case there was something previously
        if 'pca' in self.results:
            del self.results['pca']
        algorithm_figs = ["pca_plot_eigs", "pca_plot_pca", "pca_plot_rhodelta", "pca_plot_corrne", "pca_plot_corecells", "pca_plot_innerens", "pca_plot_timecourse", "pca_plot_cellsinens"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing PCA...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        worker_pca = WorkerRunnable(self.run_pca_parallel, raster, pars_matlab, pars)
        worker_pca.signals.result_ready.connect(self.run_pca_parallel_end)
        self.threadpool.start(worker_pca) 
    def run_pca_parallel(self, raster, pars_matlab, pars):
        """
        Initializes and runs the MATLAB engine to execute the PCA algorithm on neural activity data in parallel. 
        This function also handles MATLAB path setup, updates parameter values in the GUI, and plots the results.

        :param raster: Matrix of neural activity data to be processed.
        :type raster: matlab.double
        :param pars_matlab: MATLAB structure of parameters for the PCA algorithm.
        :type pars_matlab: dict
        :param pars: Python dictionary of parameters for the PCA algorithm, used for plotting.
        :type pars: dict
        :return: List of times taken for MATLAB engine setup, PCA execution, and plotting.
        :rtype: list[float]
        """
        log_flag = "GUI PCA:"
        start_time = time.time()
        print(f"{log_flag} Starting MATLAB engine...")
        eng_pca = matlab.engine.start_matlab()
        # Adding to path
        relative_folder_path = 'analysis/NeuralEnsembles'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng_pca.genpath(folder_path)
        eng_pca.addpath(folder_path_with_subfolders, nargout=0)
        end_time = time.time()
        engine_time = end_time - start_time
        print(f"{log_flag} Loaded MATLAB engine.")
        start_time = time.time()
        try:
            answer = eng_pca.raster2ens_by_density(raster, pars_matlab)
        except:
            print(f"{log_flag} An error occurred while excecuting the algorithm. Check the Python console for more info.")
            answer = None
        end_time = time.time()
        algorithm_time = end_time - start_time
        print(f"{log_flag} Done.")
        print(f"{log_flag} Terminating MATLAB engine...")
        eng_pca.quit()
        print(f"{log_flag} Done.")
        plot_times = 0
        # Plot the results
        if answer != None:
            self.algotrithm_results['pca'] = answer
            print(f"{log_flag} Plotting results...")
            start_time = time.time()
            self.plot_PCA_results(pars, answer)
            print(f"{log_flag} Done plotting.")
            # Save the results
            print(f"{log_flag} Saving results...")
            if np.array(answer["sel_ensmat_out"]).shape[0] > 0:
                self.results['pca'] = {}
                self.results['pca']['timecourse'] = np.array(answer["sel_ensmat_out"]).astype(int)
                self.results['pca']['ensembles_cant'] = self.results['pca']['timecourse'].shape[0]
                self.results['pca']['neus_in_ens'] = np.array(answer["sel_core_cells"]).T.astype(float)
                self.we_have_results()
                print(f"{log_flag} Done saving")
            else:
                print(f"{log_flag} The algorithm didn't found any ensemble. Check the python console for more info.")
            end_time = time.time()
            plot_times = end_time - start_time
            print(f"{log_flag} Done plotting and saving...")
        return [engine_time, algorithm_time, plot_times]
    def run_pca_parallel_end(self, times):
        """
        Runs when the PCA execution process finishes, logging timing information for each stage of the computation, 
        and re-enables the PCA run button.

        :param times: List containing the time taken for MATLAB engine loading, algorithm execution, 
                    and plotting in seconds.
        :type times: list[float]
        :return: None
        :rtype: None
        """
        self.update_console_log("Done executing the PCA algorithm", "complete") 
        self.update_console_log(f"- Loading the engine took {times[0]:.2f} seconds") 
        self.update_console_log(f"- Running the algorithm took {times[1]:.2f} seconds") 
        self.update_console_log(f"- Plotting and saving results took {times[2]:.2f} seconds")
        self.btn_run_pca.setEnabled(True)
    def plot_PCA_results(self, pars, answer):
        """
        Plots the results of the PCA algorithm, including eigen values, principal components, 
        rho and delta values, correlation of cells, core cells and ensembles time course.

        :param answer: Dictionary containing the PCA output data from MATLAB, including
                        eigen values, principal components, rho and delta values, correlation of cells, 
                        core cells and ensembles time course.
        :type answer: dict
        :return: None
        :rtype: None
        """
        ## Plot the eigs
        eigs = np.array(answer['exp_var'])
        seleig = int(pars['npcs'])
        plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_eigs')
        plot_widget.plot_eigs(eigs, seleig)

        # Plot the PCA
        pcs = np.array(answer['pcs'])
        labels = np.array(answer['labels'])
        labels = labels[0] if len(labels) else None
        Nens = int(answer['Nens'])
        ens_cols = plt.cm.tab10(range(Nens * 2))
        plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_pca')
        plot_widget.plot_pca(pcs, ens_labs=labels, ens_cols = ens_cols)

        # Plot the rhos vs deltas
        rho = np.array(answer['rho'])
        delta = np.array(answer['delta'])
        cents = np.array(answer['cents'])
        predbounds = np.array(answer['predbounds'])
        plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_rhodelta')
        plot_widget.plot_delta_rho(rho, delta, cents, predbounds, ens_cols)
        
        # Plot corr(n,e)
        try:
            ens_cel_corr = np.array(answer['ens_cel_corr'])
            ens_cel_corr_min = np.min(ens_cel_corr)
            ens_cel_corr_max = np.max(ens_cel_corr)
            plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_corrne')
            plot_widget.plot_core_cells(ens_cel_corr, [ens_cel_corr_min, ens_cel_corr_max])
        except:
            print("Error plotting the correlation of cells vs ensembles. Check the other plots and console for more info.")

        # Plot core cells
        core_cells = np.array(answer['core_cells'])
        plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_corecells')
        plot_widget.plot_core_cells(core_cells, [-1, 1])

        # Plot core cells
        try:
            ens_corr = np.array(answer["ens_corr"])[0]
            corr_thr = np.array(answer["corr_thr"])
            plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_innerens')
            plot_widget.plot_ens_corr(ens_corr, corr_thr, ens_cols)
        except:
            print("Error plotting the core cells. Check the other plots and console for more info.")

        # Plot ensembles timecourse
        plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_timecourse')
        plot_widget.plot_ensembles_timecourse(np.array(answer["sel_ensmat_out"]))

        plot_widget = self.findChild(MatplotlibWidget, 'pca_plot_cellsinens')
        plot_widget.plot_ensembles_timecourse(np.array(answer["sel_core_cells"]).T)

    def load_defaults_ica(self):
        """
        Loads default ICA parameter values into the UI fields.

        This method retrieves the default parameter values for ICA analysis from `MainWindow.ica_defaults` 
        and sets the corresponding values in the UI fields. It also updates the console log to 
        indicate that the default values have been successfully loaded.

        :return: None
        """
        defaults = self.ica_defaults
        self.ica_radio_method_marcenko.setChecked(True)
        self.ica_edit_perpercentile.setText(f"{defaults['threshold']['permutations_percentile']}")
        self.ica_edit_percant.setText(f"{defaults['threshold']['number_of_permutations']}")
        self.ica_radio_method_ica.setChecked(True)
        self.ica_edit_iterations.setText(f"{defaults['Patterns']['number_of_iterations']}")
        self.update_console_log("Loaded default ICA parameter values", "complete")
    def run_ICA(self):
        """
        Retrieves user-defined parameters for ICA from the GUI, applies default values 
        if fields are empty, and initiates the ICA analysis in parallel. The function also updates the console log 
        with messages about the current status and resets any previously displayed ICA figures.

        :return: None
        :rtype: None
        """
        # Temporarly disable the button
        self.btn_run_ica.setEnabled(False)
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

        # Clean all the figures in case there was something previously
        if 'ica' in self.results:
            del self.results['ica']
        algorithm_figs = ["ica_plot_assemblys", "ica_plot_activity", "ica_plot_binary_patterns", "ica_plot_binary_assemblies"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing ICA...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        worker_ica = WorkerRunnable(self.run_ica_parallel, spikes, pars_matlab)
        worker_ica.signals.result_ready.connect(self.run_ica_parallel_end)
        self.threadpool.start(worker_ica)
    def run_ica_parallel(self, spikes, pars_matlab):
        """
        Initializes and runs the MATLAB engine to execute the ICA algorithm on neural activity data in parallel. 
        This function also handles MATLAB path setup, updates parameter values in the GUI, and plots the results.

        :param spikes: Matrix of neural activity data to be processed.
        :type spikes: matlab.double
        :param pars_matlab: MATLAB structure of parameters for the ICA algorithm.
        :type pars_matlab: dict
        :return: List of times taken for MATLAB engine setup, ICA execution, and plotting.
        :rtype: list[float]
        """
        log_flag = "GUI ICA:"
        print(f"{log_flag} Starting MATLAB engine...")
        start_time = time.time()
        eng_ica = matlab.engine.start_matlab()
        # Adding to path
        relative_folder_path = 'analysis/Cell-Assembly-Detection'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng_ica.genpath(folder_path)
        eng_ica.addpath(folder_path_with_subfolders, nargout=0)
        end_time = time.time()
        engine_time = end_time - start_time
        print(f"{log_flag} Loaded MATLAB engine.")
        print(f"{log_flag} Looking for patterns...")
        start_time = time.time()
        try:
            answer = eng_ica.assembly_patterns(spikes, pars_matlab)
        except:
            print(f"{log_flag} An error occurred while excecuting the algorithm. Check the Python console for more info.")
            answer = None
        print(f"{log_flag} Done looking for patterns...")

        if answer != None:
            self.algotrithm_results['ica'] = {}
            self.algotrithm_results['ica']['patterns'] = answer
            assembly_templates = np.array(answer['AssemblyTemplates']).T
            print(f"{log_flag} Looking for assembly activity...")
            try:
                answer = eng_ica.assembly_activity(answer['AssemblyTemplates'],spikes)
            except:
                print(f"{log_flag} An error occurred while excecuting the algorithm. Check the Python console for more info.")
                answer = None
            print(f"{log_flag} Done looking for assembly activity...")
        end_time = time.time()
        algorithm_time = end_time - start_time
        print(f"{log_flag} Done.")
        print(f"{log_flag} Terminating MATLAB engine...")
        eng_ica.quit()
        print(f"{log_flag} Done.")
        plot_times = 0
        if answer != None:
            self.algotrithm_results['ica']['assembly_activity'] = answer
            start_time = time.time()
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

            print(f"{log_flag} Saving results...")
            self.results['ica'] = {}
            self.results['ica']['timecourse'] = binary_time_projection
            self.results['ica']['ensembles_cant'] = binary_time_projection.shape[0]
            self.results['ica']['neus_in_ens'] = binary_assembly_templates
            self.we_have_results()
            end_time = time.time()
            plot_times = end_time - start_time
            print(f"{log_flag} Done plotting and saving...")
        return [engine_time, algorithm_time, plot_times]
    def run_ica_parallel_end(self, times):
        """
        Runs when the ICA execution process finishes, logging timing information for each stage of the computation, 
        and re-enables the ICA run button.

        :param times: List containing the time taken for MATLAB engine loading, algorithm execution, 
                    and plotting in seconds.
        :type times: list[float]
        :return: None
        :rtype: None
        """
        self.update_console_log("Done executing the ICA algorithm", "complete") 
        self.update_console_log(f"- Loading the engine took {times[0]:.2f} seconds") 
        self.update_console_log(f"- Running the algorithm took {times[1]:.2f} seconds") 
        self.update_console_log(f"- Plotting and saving results took {times[2]:.2f} seconds")
        self.btn_run_ica.setEnabled(True)
    def plot_ICA_results(self, answer):
        """
        Plots the results of the ICA algorithm, including assembly templates, time projections, 
        binary assembly templates, core cells and binary assemblies.

        :param answer: Dictionary containing the ICA output data from MATLAB, including
                        assembly templates, time projections, binary assembly templates, 
                        and binary assemblies.
        :type answer: dict
        :return: None
        :rtype: None
        """
        # Plot the assembly templates
        plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_assemblys')
        plot_widget.set_subplots(answer['assembly_templates'].shape[0], 1)
        total_assemblies = answer['assembly_templates'].shape[0]
        for e_idx, ens in enumerate(answer['assembly_templates']):
            plot_xaxis = e_idx == total_assemblies-1
            plot_widget.plot_assembly_patterns(ens, e_idx, title=f"Ensemble {e_idx+1}", plot_xaxis=plot_xaxis)

        # Plot the time projection
        plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_activity')
        plot_widget.plot_cell_assemblies_activity(answer['time_projection'])

        # Plot binary assembly templates
        plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_binary_patterns')
        plot_widget.plot_ensembles_timecourse(answer['binary_assembly_templates'], xlabel="Cell")

        plot_widget = self.findChild(MatplotlibWidget, 'ica_plot_binary_assemblies')
        plot_widget.plot_ensembles_timecourse(answer['binary_time_projection'], xlabel="Timepoint")

    def load_defaults_x2p(self):
        """
        Loads default Xsembles2P parameter values into the UI fields.

        This method retrieves the default parameter values for Xsembles2P analysis from `MainWindow.x2p_defaults` 
        and sets the corresponding values in the UI fields. It also updates the console log to 
        indicate that the default values have been successfully loaded.

        :return: None
        """
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
        """
        Retrieves user-defined parameters for Xsembles2P from the GUI, applies default values 
        if fields are empty, and initiates the X2P analysis in parallel. The function also updates the console log 
        with messages about the current status and resets any previously displayed X2P figures.

        :return: None
        :rtype: None
        """
        # Temporarly disable the button
        self.btn_run_x2p.setEnabled(False)
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

        # Clean all the figures in case there was something previously
        if 'x2p' in self.results:
            del self.results['x2p']
        algorithm_figs = ["x2p_plot_similarity", "x2p_plot_epi", "x2p_plot_onsemact", "x2p_plot_offsemact", "x2p_plot_activity", "x2p_plot_onsemneu", "x2p_plot_offsemneu"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing Xsembles2P...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        worker_x2p = WorkerRunnable(self.run_x2p_parallel, raster, pars_matlab)
        worker_x2p.signals.result_ready.connect(self.run_x2p_parallel_end)
        self.threadpool.start(worker_x2p)
    def run_x2p_parallel(self, raster, pars_matlab):
        """
        Initializes and runs the MATLAB engine to execute the X2P algorithm on neural activity data in parallel. 
        This function also handles MATLAB path setup, updates parameter values in the GUI, and plots the results.

        :param raster: Matrix of neural activity data to be processed.
        :type raster: matlab.double
        :param pars_matlab: MATLAB structure of parameters for the X2P algorithm.
        :type pars_matlab: dict
        :return: List of times taken for MATLAB engine setup, X2P execution, and plotting.
        :rtype: list[float]
        """
        log_flag = "GUI X2P:"
        print(f"{log_flag} Starting MATLAB engine...")
        start_time = time.time()
        eng_x2p = matlab.engine.start_matlab()
        # Adding to path
        relative_folder_path = 'analysis/Xsembles2P'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng_x2p.genpath(folder_path)
        eng_x2p.addpath(folder_path_with_subfolders, nargout=0)
        end_time = time.time()
        engine_time = end_time - start_time
        print(f"{log_flag} Loaded MATLAB engine.")
        start_time = time.time()
        try:
            answer = eng_x2p.Get_Xsembles(raster, pars_matlab)
        except:
            print(f"{log_flag} An error occurred while excecuting the algorithm. Check the Python console for more info.")
            answer = None
        end_time = time.time()
        algorithm_time = end_time - start_time
        print(f"{log_flag} Done.")
        print(f"{log_flag} Terminating MATLAB engine...")
        eng_x2p.quit()
        print(f"{log_flag} Done.")
        plot_times = 0
        if answer != None:
            start_time = time.time()
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
            answer['Ensembles']['OnsembleNeurons'] = clean_answer['OnsembleNeurons']
            clean_answer['OffsembleNeurons'] = np.zeros((cant_ens, self.cant_neurons))
            for ens_it in range(cant_ens):
                members = np.array(answer['Ensembles']['OffsembleNeurons'][ens_it]) - 1
                members = members.astype(int)
                clean_answer['OffsembleNeurons'][ens_it, members] = 1
            answer['Ensembles']['OffsembleNeurons'] = clean_answer['OffsembleNeurons']
            # Clean other variables for the h5 save file
            new_clean = {}
            new_clean['Durations'] = {}
            new_clean['Indices'] = {}
            new_clean['Vectors'] = {}
            for ens_it in range(cant_ens):
                new_clean['Durations'][f"{ens_it}"] = np.array(answer['Ensembles']['Durations'][ens_it])
                new_clean['Indices'][f"{ens_it}"] = np.array(answer['Ensembles']['Indices'][ens_it])
                new_clean['Vectors'][f"{ens_it}"] = np.array(answer['Ensembles']['Vectors'][ens_it])
            answer['Ensembles']['Vectors'] = new_clean['Vectors']
            answer['Ensembles']['Indices'] = new_clean['Indices']
            answer['Ensembles']['Durations'] = new_clean['Durations']

            self.algotrithm_results['x2p'] = answer
            self.plot_X2P_results(clean_answer)

            print(f"{log_flag} Saving results...")
            self.results['x2p'] = {}
            self.results['x2p']['timecourse'] = clean_answer['Activity']
            self.results['x2p']['ensembles_cant'] = cant_ens
            self.results['x2p']['neus_in_ens'] = clean_answer['OnsembleNeurons']
            self.we_have_results()
            end_time = time.time()
            plot_times = end_time - start_time
            print(f"{log_flag} Done plotting and saving...")
        return [engine_time, algorithm_time, plot_times]
    def run_x2p_parallel_end(self, times):
        """
        Runs when the X2P execution process finishes, logging timing information for each stage of the computation, 
        and re-enables the X2P run button.

        :param times: List containing the time taken for MATLAB engine loading, algorithm execution, 
                    and plotting in seconds.
        :type times: list[float]
        :return: None
        :rtype: None
        """
        self.update_console_log("Done executing the Xsembles2P algorithm", "complete") 
        self.update_console_log(f"- Loading the engine took {times[0]:.2f} seconds") 
        self.update_console_log(f"- Running the algorithm took {times[1]:.2f} seconds") 
        self.update_console_log(f"- Plotting and saving results took {times[2]:.2f} seconds")
        self.btn_run_x2p.setEnabled(True)
    def plot_X2P_results(self, answer):
        """
        Plots the results of the X2P algorithm, including similarity map, EPI, 
        onsemble activity, offsemble activity, activity, onsembles neurons and offsemble neurons.

        :param answer: Dictionary containing the X2P output data from MATLAB, including similaty map,
                        EPI, onsemble activity, offsemble activity, activity, onnsembles neurons
                        and offsemble neurons.
        :type answer: dict
        :return: None
        :rtype: None
        """
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
        # Offsemble activity
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

    def load_defaults_sgc(self):
        """
        Loads default SGC parameter values into the UI fields.

        This method retrieves the default parameter values for SGC analysis from `MainWindow.sgc_defaults` 
        and sets the corresponding values in the UI fields. It also updates the console log to 
        indicate that the default values have been successfully loaded.

        :return: None
        """
        defaults = self.sgc_defaults
        self.sgc_check_firstderiv.setChecked(defaults['use_first_derivative'])
        self.sgc_edit_stdthreshold.setText(f"{defaults['standard_deviations_threshold']}")
        self.sgc_edit_shuff.setText(f"{defaults['shuffling_rounds']}")
        self.sgc_edit_sig.setText(f"{defaults['coactivity_significance_level']}")
        self.sgc_edit_monterounds.setText(f"{defaults['montecarlo_rounds']}")
        self.sgc_edit_montesteps.setText(f"{defaults['montecarlo_steps']}")
        self.sgc_edit_affthres.setText(f"{defaults['affinity_threshold']}")
        self.update_console_log("Loaded default SGC parameter values", "complete")
    def run_sgc(self):
        """
        Retrieves user-defined parameters for SGC from the GUI, applies default values 
        if fields are empty, and initiates the SGC analysis in parallel. The function also updates the console log 
        with messages about the current status and resets any previously displayed SGC figures.

        :return: None
        :rtype: None
        """
        # Temporarly disable the button
        self.btn_run_sgc.setEnabled(False)
        # Prepare data
        #data = self.data_neuronal_activity
        #spikes = matlab.double(data.tolist())
        self.cant_neurons, self.cant_timepoints = self.data_dFFo.shape
        data = self.data_dFFo
        dFFo = matlab.double(data.tolist())
        # Check for the first derivative flag
        use_first_derivative = self.sgc_check_firstderiv.isChecked()
        if use_first_derivative:
            dx = np.gradient(data, axis=1) # Axis 1 to get the derivative of the signal of every neuron
            dFFo = matlab.double(dx.tolist())
        # Prepare parameters
        input_value = self.sgc_edit_stdthreshold.text()
        val_std_threshold = float(input_value) if len(input_value) > 0 else self.sgc_defaults['standard_deviations_threshold']
        input_value = self.sgc_edit_shuff.text()
        val_shuffling_rounds = int(input_value) if len(input_value) > 0 else self.sgc_defaults['shuffling_rounds']
        input_value = self.sgc_edit_sig.text()
        val_coactivity_significance_level = float(input_value) if len(input_value) > 0 else self.sgc_defaults['coactivity_significance_level']
        input_value = self.sgc_edit_monterounds.text()
        val_montecarlo_rounds = int(input_value) if len(input_value) > 0 else self.sgc_defaults['montecarlo_rounds']
        input_value = self.sgc_edit_montesteps.text()
        val_montecarlo_steps = int(input_value) if len(input_value) > 0 else self.sgc_defaults['montecarlo_steps']
        input_value = self.sgc_edit_affthres.text()
        val_affinity_threshold = float(input_value) if len(input_value) > 0 else self.sgc_defaults['affinity_threshold']

        # Pack parameters
        pars = {
            'use_first_derivative': use_first_derivative, 
            'standard_deviations_threshold': val_std_threshold,
            'shuffling_rounds': val_shuffling_rounds,
            'coactivity_significance_level': val_coactivity_significance_level,
            'montecarlo_rounds': val_montecarlo_rounds,
            'montecarlo_steps': val_montecarlo_steps,
            'affinity_threshold': val_affinity_threshold
        }
        self.params['sgc'] = pars
        pars_matlab = self.dict_to_matlab_struct(pars)

        # Clean all the figures in case there was something previously
        if 'sgc' in self.results:
            del self.results['sgc']
        algorithm_figs = ["sgc_plot_timecourse", "sgc_plot_cellsinens"] 
        for fig_name in algorithm_figs:
            self.findChild(MatplotlibWidget, fig_name).reset("Loading new plots...")

        self.update_console_log("Performing SGC...")
        self.update_console_log("Look in the Python console for additional logs.", "warning")
        worker_sgc = WorkerRunnable(self.run_sgc_parallel, dFFo, pars_matlab)
        worker_sgc.signals.result_ready.connect(self.run_sgc_parallel_end)
        self.threadpool.start(worker_sgc)
    def run_sgc_parallel(self, dFFo, pars_matlab):
        """
        Initializes and runs the MATLAB engine to execute the SGC algorithm on neural activity data in parallel. 
        This function also handles MATLAB path setup, updates parameter values in the GUI, and plots the results.

        :param spikes: Matrix of neural activity data to be processed.
        :type spikes: matlab.double
        :param dFFo: Matrix of neural flourescence data to be processed
        :type dFFo: matlab.double
        :param pars_matlab: MATLAB structure of parameters for the SGC algorithm.
        :type pars_matlab: dict
        :return: List of times taken for MATLAB engine setup, SGC execution, and plotting.
        :rtype: list[float]
        """
        log_flag = "GUI SGC:"
        print(f"{log_flag} Starting MATLAB engine...")
        start_time = time.time()
        eng_sgc = matlab.engine.start_matlab()
        # Adding to path
        relative_folder_path = 'analysis/SGC_neural_assembly_detection'
        folder_path = os.path.abspath(relative_folder_path)
        folder_path_with_subfolders = eng_sgc.genpath(folder_path)
        eng_sgc.addpath(folder_path_with_subfolders, nargout=0)
        end_time = time.time()
        engine_time = end_time - start_time
        print(f"{log_flag} Loaded MATLAB engine.")
        start_time = time.time()
        try:
            answer = eng_sgc.EnsemblesGUI_linker_SGC(dFFo, pars_matlab)
        except:
            print(f"{log_flag} An error occurred while excecuting the algorithm. Check console logs for more info.")
            answer = None
        end_time = time.time()
        algorithm_time = end_time - start_time
        print(f"{log_flag} Done.")
        print(f"{log_flag} Terminating MATLAB engine...")
        eng_sgc.quit()
        print(f"{log_flag} Done.")
        plot_times = 0
        if answer != None:
            self.algotrithm_results['sgc'] = answer
            # Plotting results
            print(f"{log_flag} Plotting and saving results...")
            # For this method the saving occurs in the same plotting function to avoid recomputation
            start_time = time.time()
            self.plot_sgc_results(answer)
            end_time = time.time()
            plot_times = end_time - start_time
            print(f"{log_flag} Done plotting and saving...")
            self.we_have_results()
        return [engine_time, algorithm_time, plot_times]
    def run_sgc_parallel_end(self, times):
        """
        Runs when the SGC execution process finishes, logging timing information for each stage of the computation, 
        and re-enables the SGC run button.

        :param times: List containing the time taken for MATLAB engine loading, algorithm execution, 
                    and plotting in seconds.
        :type times: list[float]
        :return: None
        :rtype: None
        """
        self.update_console_log("Done executing the SGC algorithm", "complete") 
        self.update_console_log(f"- Importing the functions took {times[0]:.2f} seconds") 
        self.update_console_log(f"- Running the algorithm took {times[1]:.2f} seconds") 
        self.update_console_log(f"- Plotting and saving results took {times[2]:.2f} seconds")
        self.btn_run_sgc.setEnabled(True)
    def plot_sgc_results(self, answer):
        # Extracting neurons in ensembles
        assemblies_raw = answer['assemblies']
        assemblies = [np.array(list(assembly[0])) for assembly in assemblies_raw]
        num_states = len(assemblies)
        neurons_in_ensembles = np.zeros((num_states, self.cant_neurons))
        for ens_idx, ensmeble in enumerate(assemblies):
            ensemble_fixed = [cell-1 for cell in ensmeble]
            neurons_in_ensembles[ens_idx, ensemble_fixed] = 1
        # Extracting timepoints of activations
        activity_raster_peaks_raw = answer['activity_raster_peaks']
        activity_raster_peaks = np.array([int(np.array(raster_peak)[0][0])-1 for raster_peak in activity_raster_peaks_raw])
        i_assembly_patterns_raw = answer['assembly_pattern_detection']['assemblyIActivityPatterns']
        i_assembly_patterns = [np.array(list(assembly[0])).astype(int) for assembly in i_assembly_patterns_raw]
        # Formatting ensmbles timecourse 
        activations = [activity_raster_peaks[I-1] for I in i_assembly_patterns]
        ensembles_timecourse = np.zeros((num_states, self.cant_timepoints))
        for ens_idx, ensmeble_activations in enumerate(activations):
            ensembles_timecourse[ens_idx, ensmeble_activations] = 1
        # Saving results
        self.results['sgc'] = {}
        self.results['sgc']['timecourse'] = ensembles_timecourse
        self.results['sgc']['ensembles_cant'] = ensembles_timecourse.shape[0]
        self.results['sgc']['neus_in_ens'] = neurons_in_ensembles
        # Correct the answer saved
        self.algotrithm_results['sgc']['assemblies'] = {}
        for idx, assembly in enumerate(assemblies):
            self.algotrithm_results['sgc']['assemblies'][f"{idx}"] = assembly
        self.algotrithm_results['sgc']['assembly_pattern_detection']['assemblyIActivityPatterns'] = {}
        for idx, act_patt in enumerate(i_assembly_patterns):
            self.algotrithm_results['sgc']['assembly_pattern_detection']['assemblyIActivityPatterns'][f"{idx}"] = act_patt


        # Plot the cells in ensembles
        plot_widget = self.findChild(MatplotlibWidget, 'sgc_plot_cellsinens')
        plot_widget.plot_ensembles_timecourse(neurons_in_ensembles, xlabel="Cell")
        # Plot the ensmbles activations
        plot_widget = self.findChild(MatplotlibWidget, 'sgc_plot_timecourse')
        plot_widget.plot_ensembles_timecourse(ensembles_timecourse, xlabel="Timepoint")


    def we_have_results(self):
        """
        Updates the UI for the ensembles compare and performance analysis given the available analysis results.

        :return: None
        :rtype: None
        """
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
            elif analysis_name == 'sgc':
                self.ensvis_btn_sgc.setEnabled(True)
                self.performance_check_sgc.setEnabled(True)
                self.ensembles_compare_update_opts('sgc')
        save_itms = [self.save_check_minimal,
                self.save_check_params,
                self.save_check_full,
                self.save_check_enscomp,
                self.save_check_perf]
        for itm in save_itms:
            itm.setEnabled(True)
        self.tempvars["showed_sim_maps"] = False

    def ensvis_tabchange(self, index):
        """
        Identifies the tab change in the ensamble visualizer for asynchronous loading of plots.

        The loading of all the comparations and plots could be slow when loaded all at the same time.
        For this, the plots are loaded only when the user reaches the relevant tab.

        :param index: Index of the tab opened by the user.
        :type index: int
        """
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
        """
        Loads the results of the SVD into the ensembles visualizer.

        This is done this way to use only one function to update the ensembles visualizer.
        The variable :attr:`MainWindow.ensemble_currently_shown` is used to set the algorithm to show.
        Then the global funtion :meth:`MainWindow.update_analysis_results` is executed.
        """
        self.ensemble_currently_shown = "svd"
        self.update_analysis_results()
    def vis_ensembles_pca(self):
        """
        Loads the results of the PCA into the ensembles visualizer.

        This is done this way to use only one function to update the ensembles visualizer.
        The variable :attr:`MainWindow.ensemble_currently_shown` is used to set the algorithm to show.
        Then the global funtion :meth:`MainWindow.update_analysis_results` is executed.
        """
        self.ensemble_currently_shown = "pca"
        self.update_analysis_results()
    def vis_ensembles_ica(self):
        """
        Loads the results of the ICA into the ensembles visualizer.

        This is done this way to use only one function to update the ensembles visualizer.
        The variable :attr:`MainWindow.ensemble_currently_shown` is used to set the algorithm to show.
        Then the global funtion :meth:`MainWindow.update_analysis_results` is executed.
        """
        self.ensemble_currently_shown = "ica"
        self.update_analysis_results()
    def vis_ensembles_x2p(self):
        """
        Loads the results of the X2P into the ensembles visualizer.

        This is done this way to use only one function to update the ensembles visualizer.
        The variable :attr:`MainWindow.ensemble_currently_shown` is used to set the algorithm to show.
        Then the global funtion :meth:`MainWindow.update_analysis_results` is executed.
        """
        self.ensemble_currently_shown = "x2p"
        self.update_analysis_results()
    def vis_ensembles_sgc(self):
        """
        Loads the results of the SGC into the ensembles visualizer.

        This is done this way to use only one function to update the ensembles visualizer.
        The variable :attr:`MainWindow.ensemble_currently_shown` is used to set the algorithm to show.
        Then the global funtion :meth:`MainWindow.update_analysis_results` is executed.
        """
        self.ensemble_currently_shown = "sgc"
        self.update_analysis_results()

    def update_analysis_results(self):
        """
        Pre-loads the `General` tab in the ensembles visualizer.

        :return: None
        :rtype: None

        This function also saves the state of shown/not-shown of the other tabs for
        asynchronous loading and caching.
        """
        self.initialize_ensemble_view()   
        self.tempvars['ensvis_shown_tab1'] = False
        self.tempvars['ensvis_shown_tab2'] = False
        self.tempvars['ensvis_shown_tab3'] = False
        self.tempvars['ensvis_shown_tab4'] = False 

    def initialize_ensemble_view(self):
        """
        Loads the slider for ensemble selection for the chosen analysis and sets its limits.

        :return: None
        :rtype: None

        This function makes available the ensemble selector.
        This function also loads the visualization of the first ensemble.
        """
        self.tempvars['ensvis_shown_results'] = True
        self.ensvis_tabs.setCurrentIndex(0)
        curr_show = self.ensemble_currently_shown 
        self.ensvis_lbl_currently.setText(f"{curr_show}".upper())
        # Show the number of identified ensembles
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
        """
        Loads the visualization of the selected ensemble.

        This function shows the members of the ensemble, exclusive members and timepoints of
        activation of the selected ensemble in text boxes. If coordinates are provided, it also 
        loads the visualization of the coordinates and enables customization buttons
        If dFFo is provided it also plots the dFFo traces.

        :param value: Index of th eensamble to be shown, indexed at 1.
        :type value: int
        :return: None
        :rtype: None
        """
        curr_analysis = self.ensemble_currently_shown
        curr_ensemble = value
        self.ensvis_lbl_currentens.setText(f"{curr_ensemble}")

        # Get the members of this ensemble
        members = []
        ensemble = self.results[curr_analysis]['neus_in_ens'][value-1,:]
        members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
        members_txt = format_nums_to_string(members)
        self.ensvis_edit_members.setText(members_txt)

        # Get the exclusive members of this ensemble
        ens_mat = self.results[curr_analysis]['neus_in_ens']
        mask_e = ensemble == 1
        sum_mask = np.sum(ens_mat, axis=0)
        exc_elems = [cell+1 for cell in range(len(mask_e)) if mask_e[cell] and sum_mask[cell] == 1]
        exclusive_txt = format_nums_to_string(exc_elems)
        self.ensvis_edit_exclusive.setText(exclusive_txt)

        # Timepoints of activation
        ensemble_timecourse = self.results[curr_analysis]['timecourse'][curr_ensemble-1,:]
        ens_timepoints = [frame+1 for frame in range(len(ensemble_timecourse)) if ensemble_timecourse[frame]]
        ens_timepoints_txt = format_nums_to_string(ens_timepoints)
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
            plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_raster')
            dFFo_ens = self.data_dFFo[idx_corrected_members, :]
            plot_widget.plot_ensemble_dFFo(dFFo_ens, idx_corrected_members, ensemble_timecourse)
    
    def update_ens_vis_coords(self):
        """
        Shows the neurons maps in the ensembles visualizer and enables the visualizer buttons.

        This is a separated function to be executed also when the visualization options are updated
        and show the visualization with the new options in real time.

        :return: None
        :rtype: None
        """
        only_ens = self.ensvis_check_onlyens.isChecked()
        only_contours = self.ensvis_check_onlycont.isChecked()
        show_numbers = self.ensvis_check_cellnum.isChecked()
        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_map')
        self.plot_widget.plot_coordinates2D_highlight(self.data_coordinates, self.current_idx_corrected_members, self.current_idx_corrected_exclusive, only_ens, only_contours, show_numbers)

    def update_ensvis_alldFFo(self):
        """
        Plot the dFFo of the neurons in every ensemble.

        This function creates a subplot for every ensemble and shows the dFFo traces for every
        neuron in every ensemble.

        :return: None
        :rtype: None
        """
        curr_analysis = self.ensemble_currently_shown
        cant_ensembles = self.results[curr_analysis]['ensembles_cant']

        plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_alldffo')
        plot_widget.set_subplots(1, max(cant_ensembles,2))
        for current_ens in range(cant_ensembles):
            # Create subplot for each core
            ensemble = self.results[curr_analysis]['neus_in_ens'][current_ens,:]
            members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
            idx_corrected_members = [idx-1 for idx in members]
            dFFo_ens = self.data_dFFo[idx_corrected_members, :]
            plot_widget.plot_all_dFFo(dFFo_ens, idx_corrected_members, current_ens)

    def update_ensvis_allcoords(self):
        """
        Plot the coordiantes of the neurons in every ensemble.

        This function creates a subplot for every ensemble and shows the neuron maps
        for every ensemble.

        :return: None
        :rtype: None
        """
        curr_analysis = self.ensemble_currently_shown
        cant_ensembles = self.results[curr_analysis]['ensembles_cant']
        
        plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allspatial')
        
        rows = math.ceil(math.sqrt(cant_ensembles))
        cols = math.ceil(cant_ensembles / rows)
        plot_widget.set_subplots(max(rows, 2), max(cols, 2))
        plot_widget.canvas.setFixedHeight(300*rows)

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
            
            plot_widget.plot_all_coords(self.data_coordinates, idx_corrected_members, idx_corrected_exclusive, row, col)

    def update_ensvis_allbinary(self):
        """
        Plot the binary activations of every neurons in every ensemble.

        This function creates a subplot for every ensemble and shows the activations of
        every neuron in every ensemble.

        :return: None
        :rtype: None
        """
        curr_analysis = self.ensemble_currently_shown
        cant_ensembles = self.results[curr_analysis]['ensembles_cant']
        
        plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allbinary')
        plot_widget.set_subplots(1, max(cant_ensembles, 2))
        for current_ens in range(cant_ensembles):
            ensemble = self.results[curr_analysis]['neus_in_ens'][current_ens,:]
            members = [cell+1 for cell in range(len(ensemble)) if ensemble[cell] > 0]
            idx_corrected_members = [idx-1 for idx in members]
            activity = self.data_neuronal_activity[idx_corrected_members, :] == 0
            plot_widget.plot_all_binary(activity, members, current_ens, current_ens)

    def update_ensvis_allens(self):
        """
        Plot the activations of every ensemble.

        This function creates figure with the timecourse of every ensemble identified.

        :return: None
        :rtype: None
        """
        curr_analysis = self.ensemble_currently_shown
        self.plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allens')
        self.plot_widget.plot_ensembles_timecourse(self.results[curr_analysis]['timecourse'])

    def ensembles_compare_update_opts(self, algorithm):
        """
        Updates the option buttons with the data from the given analysis.

        This function receives the name of some algorithm and then updates the options
        in the `Ensemble Compare` tab. In this function the slider and labels of each 
        analysis is assigned and then the general buttons are loaded.

        :param algorithm: String with the name of the algorithm, in lower case and three chars syntax.
        :type algorithm: string
        :return: None
        :rtype: None
        """
        if algorithm == 'svd':
            ens_selector = self.enscomp_slider_svd
            selector_label_min = self.enscomp_slider_lbl_min_svd
            selector_label_max = self.enscomp_slider_lbl_max_svd
        elif algorithm == 'pca':
            ens_selector = self.enscomp_slider_pca
            selector_label_min = self.enscomp_slider_lbl_min_pca
            selector_label_max = self.enscomp_slider_lbl_max_pca
        elif algorithm == 'ica':
            ens_selector = self.enscomp_slider_ica
            selector_label_min = self.enscomp_slider_lbl_min_ica
            selector_label_max = self.enscomp_slider_lbl_max_ica
        elif algorithm == 'x2p':
            ens_selector = self.enscomp_slider_x2p
            selector_label_min = self.enscomp_slider_lbl_min_x2p
            selector_label_max = self.enscomp_slider_lbl_max_x2p
        elif algorithm == 'sgc':
            ens_selector = self.enscomp_slider_sgc
            selector_label_min = self.enscomp_slider_lbl_min_sgc
            selector_label_max = self.enscomp_slider_lbl_max_sgc

        # Enable the general visualization options
        self.enscomp_visopts_showcells.setEnabled(True)
        self.enscomp_visopts_neusize.setEnabled(True)
        self.enscomp_visopts_setneusize.setEnabled(True)
        
        # Only add the new algorithm to the combobox selector if it's not already there
        combo_string = algorithm.upper()
        index_match = self.enscomp_combo_select_result.findText(combo_string)
        if index_match == -1:
            self.enscomp_combo_select_result.addItem(combo_string)

        # Activate the slider to select an ensemble of the given algorithm
        ens_selector.setEnabled(True)
        ens_selector.setMinimum(1)   # Set the minimum value
        ens_selector.setMaximum(self.results[algorithm]['ensembles_cant']) # Set the maximum value
        ens_selector.setValue(1)
        selector_label_min.setEnabled(True)
        selector_label_min.setText(f"{1}")
        selector_label_max.setEnabled(True)
        selector_label_max.setText(f"{self.results[algorithm]['ensembles_cant']}")
        # Update the toolbox options
        self.enscomp_visopts[algorithm]['enabled'] = True
        self.enscomp_combo_select_simil.setEnabled(True)
        self.enscomp_combo_select_simil_method.setEnabled(True)
        self.enscomp_combo_select_simil_colormap.setEnabled(True)

    
    def ensembles_compare_update_combo_results(self, text):
        """
        Updates the visualization options for the selected algorithm.

        This function is executed when the combo box value is changed. Loads the visualization
        options for the current algorithm.

        :param text: Name of the selected analysis as shown in the combo box.
        :type text: string
        :return: None
        :rtype: None
        """
        # Teporarly block the signals of the options to not trigger their functions when reasigned.
        self.enscomp_check_coords.blockSignals(True)
        self.enscomp_check_ens.blockSignals(True)
        self.enscomp_check_neus.blockSignals(True)
        method_selected = text.lower()
        # Change enabled status for this option
        self.enscomp_check_coords.setEnabled(self.enscomp_visopts[method_selected]['enabled'])
        self.enscomp_check_ens.setEnabled(self.enscomp_visopts[method_selected]['enabled'])
        self.enscomp_check_neus.setEnabled(self.enscomp_visopts[method_selected]['enabled'])
        self.enscomp_btn_color.setEnabled(self.enscomp_visopts[method_selected]['enabled'])
        # Change the boxes values
        self.enscomp_check_coords.setChecked(self.enscomp_visopts[method_selected]['enscomp_check_coords'])
        self.enscomp_check_ens.setChecked(self.enscomp_visopts[method_selected]['enscomp_check_ens'])
        self.enscomp_check_neus.setChecked(self.enscomp_visopts[method_selected]['enscomp_check_neus'])
        self.enscomp_check_coords.blockSignals(False)
        self.enscomp_check_ens.blockSignals(False)
        self.enscomp_check_neus.blockSignals(False)
    
    def update_enscomp_options(self, exp_data):
        """
        Loads the behavior or stimulation data into the `Ensemble Compare` tab.

        This function is executed when a dataset of behavior or stimulation is assigned.
        The data of this dataset is loaded in the corresponding field in the ensembles compare tab.

        :param exp_data: Kind of dataset, options are "behavior" and "stims".
        :type exp_data: string.
        """
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
        slider.blockSignals(True)
        slider.setEnabled(True)
        slider.setMinimum(1)
        slider.setMaximum(max_val)
        slider.setValue(1)
        lbl_min.setText(f"{1}")
        lbl_min.setEnabled(True)
        lbl_label.setText(f"{1}")
        lbl_label.setEnabled(True)
        lbl_max.setText(f"{max_val}")
        lbl_max.setEnabled(True)
        slider.blockSignals(False)
        # Update the toolbox options
        check_show.setEnabled(True)
        color_pick.setEnabled(True)
        
    def ensembles_compare_update_ensembles(self):
        """
        Updates the ensembles comparison settings and visualizations.

        This method collects the selected ensembles from various methods (SVD, PCA, ICA, X2P),
        updates their corresponding properties (index, neurons in ensemble, and timecourse),
        and applies the visualization options to the GUI. The method also updates the comparison
        map and timecourse visualizations.

        :return: None
        :rtype: None
        """
        ensembles_to_compare = {}
        ens_selector = {
            "svd": self.enscomp_slider_svd,
            "pca": self.enscomp_slider_pca,
            "ica": self.enscomp_slider_ica,
            "x2p": self.enscomp_slider_x2p,
            "sgc": self.enscomp_slider_sgc,
            "stims": self.enscomp_slider_stim
        }
        for key, slider in ens_selector.items():
            if slider.isEnabled():
                ens_idx = slider.value()
                if key == 'stims':
                    ensembles_to_compare[key] = {}
                    ensembles_to_compare[key]["ens_idx"] = ens_idx-1
                    ensembles_to_compare[key]["timecourse"] = self.data_stims[ens_idx-1,:].copy()
                else:
                    ensembles_to_compare[key] = {}
                    ensembles_to_compare[key]["ens_idx"] = ens_idx-1
                    ensembles_to_compare[key]["neus_in_ens"] = self.results[key]['neus_in_ens'][ens_idx-1,:].copy()
                    ensembles_to_compare[key]["timecourse"] = self.results[key]['timecourse'][ens_idx-1,:].copy()
        
        self.enscomp_colorflag_svd.setStyleSheet(f"background-color: {self.enscomp_visopts['svd']['color']};")
        self.enscomp_colorflag_pca.setStyleSheet(f"background-color: {self.enscomp_visopts['pca']['color']};")
        self.enscomp_colorflag_ica.setStyleSheet(f"background-color: {self.enscomp_visopts['ica']['color']};")
        self.enscomp_colorflag_x2p.setStyleSheet(f"background-color: {self.enscomp_visopts['x2p']['color']};")
        self.enscomp_colorflag_sgc.setStyleSheet(f"background-color: {self.enscomp_visopts['sgc']['color']};")
        self.enscomp_colorflag_stims.setStyleSheet(f"background-color: {self.enscomp_visopts['stims']['color']};")
        self.enscomp_colorflag_behavior.setStyleSheet(f"background-color: {self.enscomp_visopts['behavior']['color']};")
        
        # Update the visualization options
        current_method = self.enscomp_combo_select_result.currentText().lower()
        self.enscomp_visopts[current_method]['enscomp_check_coords'] = self.enscomp_check_coords.isChecked()
        self.enscomp_visopts[current_method]['enscomp_check_ens'] = self.enscomp_check_ens.isChecked()
        self.enscomp_visopts[current_method]['enscomp_check_neus'] = self.enscomp_check_neus.isChecked()

        self.ensembles_compare_update_map(ensembles_to_compare)
        self.ensembles_compare_update_timecourses(ensembles_to_compare)

    def ensembles_compare_update_map(self, ensembles_to_compare):
        """
        Updates the spatial map in ensembles compare.

        This function calculates the shared neurons between every ensemble according to the
        current visualization options, and asigns colors for each neuron in the map.

        :param ensembles_to_compare: Dictionary with where each key is the name of an algorithm,
                        the value is another dictionary. The key used here is `neus_in_ens` with a
                        binary numpy matrix with shape with the members of each ensemble.
        :type ensembles_to_compare: dict
        :return: None
        :rtype: None
        """
        if not hasattr(self, "data_coordinates"):
            self.data_coordinates = np.random.randint(1, 351, size=(self.cant_neurons, 2))
        # Stablish the dimention of the map
        max_x = np.max(self.data_coordinates[:, 0])
        max_y = np.max(self.data_coordinates[:, 1])
        lims = [max_x, max_y]

        mixed_ens = []

        list_colors_freq = [[] for l in range(self.cant_neurons)] 

        for key, ens_data in ensembles_to_compare.items():
            if key == "stims":
                continue
            if self.enscomp_visopts[key]['enabled'] and self.enscomp_visopts[key]['enscomp_check_coords']:
                new_members = ens_data["neus_in_ens"].copy()
                if len(mixed_ens) == 0:
                    mixed_ens = new_members
                else:
                    mixed_ens += new_members
                for cell_idx in range(len(new_members)):
                    if new_members[cell_idx] > 0:
                        list_colors_freq[cell_idx].append(self.enscomp_visopts[key]['color'])

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
        """
        Updates the timecourse plot in the ensembles compare tab.

        This function extracts the timecourse of the selected ensembles accordingly with the
        current visualization options and then plots those in the corresponding figure.

        :param ensembles_to_compare: Dictionary with where each key is the name of an algorithm,
                        the value is another dictionary. The keys used here are `"neus_in_ens"` with a
                        binary numpy matrix with shape with the members of each ensemble and `"timecourse"`
                        with a binary matrix describing the moment of activation of the selected ensembles.
        :type ensembles_to_compare: dict
        :return: None
        :rtype: None
        """
        colors = []
        timecourses = []
        cells_activities = []
        new_ticks = []
        for key, ens_data in ensembles_to_compare.items():
            if key == 'stims':
                if self.enscomp_check_show_stim.isChecked():
                    # Get the currently selected stimulation
                    selected_stim = ens_data["ens_idx"]
                    # Get the timecourse
                    timecourses.append(ens_data["timecourse"].copy())
                    # Get the label, if any
                    if "stim" in self.varlabels:
                        stim_labels = list(self.varlabels["stim"].values())
                        stim_label = f"Stim {stim_labels[selected_stim]}"
                    else:
                        stim_label = f"Stim {selected_stim}"
                    new_ticks.append(f"{stim_label}")
                    colors.append(self.enscomp_visopts['stims']['color'])
            else:
                if self.enscomp_visopts[key]['enabled'] and self.enscomp_visopts[key]['enscomp_check_ens']:
                    new_timecourse = ens_data["timecourse"].copy()
                else:
                    new_timecourse = []
                timecourses.append(new_timecourse)

                if self.enscomp_visopts[key]['enabled'] and self.enscomp_visopts[key]['enscomp_check_neus']:
                    new_members = ens_data["neus_in_ens"].copy()
                    cells_activity_mat = self.data_neuronal_activity[new_members.astype(bool), :]
                    cells_activity_count = np.sum(cells_activity_mat, axis=0)
                else:
                    cells_activity_count = []
                cells_activities.append(cells_activity_count)

                colors.append(self.enscomp_visopts[key]['color'])
                new_ticks.append(key)
        
        cells_activities.reverse()
        timecourses.reverse()
        colors.reverse()
        new_ticks.reverse()

        plot_widget = self.findChild(MatplotlibWidget, 'enscomp_plot_neusact')
        plot_widget.enscomp_update_timelines(new_ticks, cells_activities, [], timecourses, colors, self.cant_timepoints)

    def enscomp_get_color(self):
        """
        Opens the QColorDialog to select a color for the selected analysis.

        The selected color will be applied to the elements of the algorithm selected
        in the combo box for the Ensemble Compare tab.
        :return: None
        :rtype: None
        """
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            current_method = self.enscomp_combo_select_result.currentText().lower()
            self.enscomp_visopts[current_method]['color'] = color_hex
            self.ensembles_compare_update_ensembles()

    def ensembles_compare_get_elements_labels(self, criteria):
        """
        Retrieves elements and their labels for ensembles comparison based on a given criterion.

        This method extracts elements from the results dictionary according to the specified criterion 
        (e.g., neurons in ensemble or timecourse). It also generates labels for these elements, 
        indicating the algorithm and the ensemble index.

        :param criteria: The key used to extract elements from each algorithm's results.
        :type criteria: string
        :return: A tuple containing the array of elements and their corresponding labels.
        :rtype: tuple (numpy.ndarray, list of string)
        """

        labels = []
        all_elements = []
        for algorithm in list(self.results.keys()):
            elements = self.results[algorithm][criteria]
            for e_idx, element in enumerate(elements):
                all_elements.append(element)
                labels.append(f"{algorithm}-E{e_idx+1}")
        # Convert to numpy array
        all_elements = np.array(all_elements)
        return all_elements, labels
    
    def ensembles_compare_get_simmatrix(self, method, all_elements):
        """
        Calculates the similarity matrix for a set of elements using the specified method.

        This method computes pairwise similarity or distance metrics (e.g., cosine, Euclidean, 
        correlation, Jaccard) and formats the result as a similarity matrix.

        :param method: The similarity or distance metric to use. Valid options are:
                    'Cosine', 'Euclidean', 'Correlation', 'Jaccard'.
        :type method: string
        :param all_elements: A numpy array containing the elements to compare. Each row represents 
                            a single element, and columns represent features.
        :type all_elements: numpy.ndarray
        :raises ValueError: If an unsupported method is provided.
        :return: A similarity matrix where each entry represents the pairwise similarity between elements.
        :rtype: numpy.ndarray
        """

        similarity_matrix = []
        if method == 'Cosine':
            similarity_matrix = cosine_similarity(all_elements)
        elif method == 'Euclidean':
            similarity_matrix = squareform(pdist(all_elements, metric='euclidean'))
        elif method == 'Correlation':
            similarity_matrix = np.corrcoef(all_elements)
        elif method == 'Jaccard':
            jaccard_distances = pdist(all_elements, metric='jaccard')
            similarity_matrix = 1 - squareform(jaccard_distances)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        return similarity_matrix

    def ensembles_compare_similarity(self, component=None, first_show=False):
        """
        Computes and visualizes the similarity matrix for ensembles based on a selected component.

        This method calculates the similarity matrix for ensembles using either the "Neurons" or 
        "Timecourses" component. It updates the visualization options and plots the results.

        :param component: Specifies the component to compute the similarity for ("Neurons" or "Timecourses").
                        If not provided, the currently selected component in the GUI is used.
        :type component: str, optional
        :param first_show: Indicates whether this is the initial visualization, which determines default 
                        visualization settings.
        :type first_show: bool, optional
        :return: None
        :rtype: None
        """
        for i in range(2):
            if component == "Neurons":
                criteria = 'neus_in_ens'
                key = "sim_neus"
            elif component == "Timecourses":
                criteria = 'timecourse'
                key = "sim_time"
            else:
                component = self.enscomp_combo_select_simil.currentText()

        # Create the labels and the big matrix
        all_elements, labels = self.ensembles_compare_get_elements_labels(criteria)

        if not first_show:
            method = self.enscomp_combo_select_simil_method.currentText()
            color = self.enscomp_combo_select_simil_colormap.currentText()
        else:
            method = self.enscomp_visopts[key]['method']
            color = self.enscomp_visopts[key]['colormap']

        similarity_matrix = self.ensembles_compare_get_simmatrix(method, all_elements)

        if component == "Neurons":
            plot_widget = self.findChild(MatplotlibWidget, 'enscomp_plot_sim_elements')
            self.enscomp_visopts["sim_neus"]['method'] = method
            self.enscomp_visopts["sim_neus"]['colormap'] = color
        elif component == "Timecourses":
            plot_widget = self.findChild(MatplotlibWidget, 'enscomp_plot_sim_times')
            self.enscomp_visopts["sim_time"]['method'] = method
            self.enscomp_visopts["sim_time"]['colormap'] = color
        
        plot_widget.enscomp_plot_similarity(similarity_matrix, labels, color)
    
    def ensembles_compare_similarity_update_combbox(self, text):
        """
        Updates the combo boxes for similarity method and colormap based on the selected component.

        This method adjusts the currently displayed options in the similarity method and colormap 
        combo boxes to match the visualization settings for the selected component ("Neurons" or "Timecourses").

        :param text: The selected component, either "Neurons" or "Timecourses".
        :type text: string
        :return: None
        :rtype: None
        """

        self.enscomp_combo_select_simil_method.blockSignals(True)
        self.enscomp_combo_select_simil_colormap.blockSignals(True)

        if text == "Neurons":
            key = "sim_neus"
        elif text == "Timecourses":
            key = "sim_time"
        self.enscomp_combo_select_simil_method.setCurrentText(self.enscomp_visopts[key]['method'])
        self.enscomp_combo_select_simil_colormap.setCurrentText(self.enscomp_visopts[key]['colormap'])

        self.enscomp_combo_select_simil_method.blockSignals(False)
        self.enscomp_combo_select_simil_colormap.blockSignals(False)

    
    def ensembles_compare_tabchange(self, index):
        """
        Handles tab changes in the GUI, updating the interface and displaying similarity maps as needed.

        This method is triggered when the user switches tabs in the GUI. It updates the state 
        of dropdown menus and enables/disables specific components depending on the selected tab.
        If the user navigates to the tabs for "Neurons" or "Timecourses", it ensures that the 
        similarity maps are displayed for the first time.

        :param index: The index of the currently selected tab.
                    - `2`: Neurons tab
                    - `3`: Timecourses tab
        :type index: int
        :return: None
        :rtype: None
        """
        if len(self.results) > 0:
            if index == 2:
                self.enscomp_combo_select_simil.setCurrentText("Neurons")
            elif index == 3:
                self.enscomp_combo_select_simil.setCurrentText("Timecourses")
            if index == 2 or index == 3:
                self.enscomp_combo_select_simil.setEnabled(True)
                self.enscomp_combo_select_simil_method.setEnabled(True)
                self.enscomp_combo_select_simil_colormap.setEnabled(True)
                if not self.tempvars["showed_sim_maps"]:
                    self.ensembles_compare_similarity(component="Neurons", first_show=True)
                    self.ensembles_compare_similarity(component="Timecourses", first_show=True)
                    self.tempvars["showed_sim_maps"] = True

    def performance_tabchange(self, index):
        """
        Handles tab changes in the performance analysis section of the GUI, triggering updates for the selected tab.

        This method ensures that specific updates are performed when the user switches tabs in the 
        performance analysis section. Each tab corresponds to a different type of performance metric 
        or visualization.

        :param index: The index of the currently selected tab.
                    - `0`: Correlation with ensemble presentation
                    - `1`: Correlations between cells
                    - `2`: Cross correlations between ensembles and stimuli
                    - `3`: Correlation with behavior
                    - `4`: Cross correlations with behavior
        :type index: int
        :return: None
        :rtype: None
        """

        if self.tempvars['performance_shown_results']:
            if index == 0:  # Correlation with ensemble presentation
                if hasattr(self, "data_stims"):
                    if not self.tempvars['performance_shown_tab0']:
                        self.tempvars['performance_shown_tab0'] = True
                        self.update_corr_stim()
            elif index == 1:    # Correlations between cells
                if hasattr(self, "data_neuronal_activity") or hasattr(self, "data_dFFo"):
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
        """
        Updates the list of selected methods for performance comparison based on user input.

        This method checks the state of multiple checkboxes corresponding to different analysis methods 
        (SVD, PCA, ICA, X2P, SGC) and updates the list of methods to compare. It also enables or disables 
        the compare button based on whether any methods are selected.

        :return: None
        :rtype: None
        """

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
        """
        Performs performance comparison and initializes relevant performance tabs.

        This method sets up temporary variables to track which performance tabs have been displayed, 
        sets the default tab to the first tab, and updates the correlations with stimulus data 
        if such data is available.

        :return: None
        :rtype: None
        """

        self.tempvars['performance_shown_results'] = True
        self.tempvars['performance_shown_tab0'] = False
        self.tempvars['performance_shown_tab1'] = False
        self.tempvars['performance_shown_tab2'] = False
        self.tempvars['performance_shown_tab3'] = False
        self.tempvars['performance_shown_tab4'] = False
        self.performance_tabs.setCurrentIndex(0)
        if hasattr(self, 'data_stims'):
            self.update_corr_stim()
            self.tempvars['performance_shown_tab0'] = True

    def update_corr_stim(self):
        """
        Updates the plot of correlation with stimulus data by initiating a parallel worker thread.

        This method retrieves the plot widget for displaying correlation with stimuli, 
        and starts a worker thread to perform the correlation update in parallel, 
        which is then rendered in the plot widget.

        :return: None
        :rtype: None
        """
        plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrstims')
        worker_corrstim = WorkerRunnable(self.update_corr_stim_parallel, plot_widget)
        self.threadpool.start(worker_corrstim) 
    def update_corr_stim_parallel(self, plot_widget):
        """
        Calculates and updates the correlation between the ensemble timecourses and stimulus data.

        This method computes the correlation for each selected method (e.g., SVD, PCA, ICA, etc.) 
        between the ensemble timecourses and the stimulus data, and then updates the plot widget 
        with the results. It handles the display by organizing the plots into a specified layout 
        based on the number of methods being compared.

        :param plot_widget: The widget used to display the correlation plots.
        :type plot_widget: MatplotlibWidget
        :return: None
        :rtype: None
        """

        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        # Calculate correlation with stimuli
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        plot_widget.set_subplots(1, plot_colums)
        stim_labels = self.varlabels["stim"].values() if "stim" in self.varlabels else []
        for m_idx, method in enumerate(methods_to_compare):
            timecourse = self.results[method]['timecourse']
            stims = self.data_stims
            correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
            if np.isscalar(correlation):
                    correlation = np.array([[correlation]])
            plot_widget.plot_perf_correlations_ens_group(correlation, m_idx, title=f"{method}".upper(), xlabel="Stims", group_labels=stim_labels)            

    def update_correlation_cells(self):
        """
        Updates the plot of correlation between the activations of the neurons 
        in the same ensemble by initiating a parallel worker thread.

        This method retrieves the plot widget for displaying correlation between the neurons 
        in the ensembles, and starts a worker thread to perform the correlation update in parallel, 
        which is then rendered in the plot widget.

        :return: None
        :rtype: None
        """
        plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrcells')
        worker_corrcells = WorkerRunnable(self.update_correlation_cells_parallel, plot_widget)
        self.threadpool.start(worker_corrcells) 
    def update_correlation_cells_parallel(self, plot_widget):
        """
        Calculates and updates the correlation between the activation of the cells in the ensemble.

        This method computes the correlation for each selected method (e.g., SVD, PCA, ICA, etc.) 
        between the neurons in the ensemble, and then updates the plot widget with the results. 
        It handles the display by organizing the plots into a specified layout 
        based on the number of methods being compared.

        :param plot_widget: The widget used to display the correlation plots.
        :type plot_widget: MatplotlibWidget
        :return: None
        :rtype: None
        """
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        # Plot the correlation of cells between themselves
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        # Find the greatest number of ensembles
        max_ens = 0
        for method in methods_to_compare:
            max_ens = max(self.results[method]['ensembles_cant'], max_ens)
        plot_widget.canvas.setFixedHeight(450*max_ens)

        if hasattr(self, "data_neuronal_activity"):
            activity_to_correlate = self.data_neuronal_activity
        elif hasattr(self, "data_dFFo"):
            activity_to_correlate = self.data_dFFo

        plot_widget.set_subplots(max_ens, plot_colums)
        for col_idx, method in enumerate(methods_to_compare):
            for row_idx, ens in enumerate(self.results[method]['neus_in_ens']):
                members = [c_idx for c_idx in range(len(ens)) if ens[c_idx] == 1]
                activity_neus_in_ens = activity_to_correlate[members, :]
                cells_names = [member+1 for member in members]
                correlation = metrics.compute_correlation_inside_ensemble(activity_neus_in_ens)
                # Convert scalar to a 1x1 matrix for plotting if the ensamble contained only one neuron
                if np.isscalar(correlation):
                    correlation = np.array([[correlation]])
                plot_widget.plot_perf_correlations_cells(correlation, cells_names, col_idx, row_idx, title=f"Cells in ensemble {row_idx+1} - Method " + f"{method}".upper())

    def update_cross_ens_stim(self):
        """
        Updates the plot of cross-correlation with stimulus data by initiating a parallel worker thread.

        This method retrieves the plot widget for displaying cross-correlation with stimuli, 
        and starts a worker thread to perform the cross-correlation update in parallel, 
        which is then rendered in the plot widget.

        :return: None
        :rtype: None
        """
        plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_crossensstim')
        worker_crosstim = WorkerRunnable(self.update_cross_ens_stim_parallel, plot_widget)
        #worker_crosstim.signals.result_ready.connect(self.update_cross_ens_stim_end)
        self.threadpool.start(worker_crosstim) 
    def update_cross_ens_stim_parallel(self, plot_widget):
        """
        Calculates and updates the cross-correlation between the ensemble timecourses and stimulus data.

        This method computes the cross-correlation for each selected method (e.g., SVD, PCA, ICA, etc.) 
        between the ensemble timecourses and the stimulus data, and then updates the plot widget 
        with the results. It handles the display by organizing the plots into a specified layout 
        based on the number of methods being compared.

        :param plot_widget: The widget used to display the cross-correlation plots.
        :type plot_widget: MatplotlibWidget
        :return: None
        :rtype: None
        """ 
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        # Calculate cross-correlation
        max_ens = 0
        for method in methods_to_compare:
            max_ens = max(self.results[method]['ensembles_cant'], max_ens)
        plot_widget.canvas.setFixedHeight(400*max_ens)
        plot_widget.set_subplots(max_ens, plot_colums)
        stim_labels = list(self.varlabels["stim"].values()) if "stim" in self.varlabels else []
        for m_idx, method in enumerate(methods_to_compare):
            for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                cross_corrs = []
                for stimtime in self.data_stims:
                    cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                    cross_corrs.append(cross_corr)
                cross_corrs = np.array(cross_corrs)
                plot_widget.plot_perf_cross_ens_stims(cross_corrs, lags, m_idx, ens_idx, group_prefix="Stim", title=f"Cross correlation Ensemble {ens_idx+1} and stimuli - Method " + f"{method}".upper(), group_labels=stim_labels)          

    def update_corr_behavior(self):
        """
        Updates the plot of correlation with behavior data by initiating a parallel worker thread.

        This method retrieves the plot widget for displaying correlation with behavior, 
        and starts a worker thread to perform the correlation update in parallel, 
        which is then rendered in the plot widget.

        :return: None
        :rtype: None
        """
        plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_corrbehavior')
        worker_corrbeha = WorkerRunnable(self.update_corr_behavior_parallel, plot_widget)
        #worker_corrbeha.signals.result_ready.connect(self.update_cross_ens_stim_end)
        self.threadpool.start(worker_corrbeha) 
    def update_corr_behavior_parallel(self, plot_widget):
        """
        Calculates and updates the correlation between the ensemble timecourses and behavior data.

        This method computes the correlation for each selected method (e.g., SVD, PCA, ICA, etc.) 
        between the ensemble timecourses and the behavior data, and then updates the plot widget 
        with the results. It handles the display by organizing the plots into a specified layout 
        based on the number of methods being compared.

        :param plot_widget: The widget used to display the correlation plots.
        :type plot_widget: MatplotlibWidget
        :return: None
        :rtype: None
        """
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        # Calculate correlation with stimuli 
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        plot_widget.set_subplots(1, plot_colums)
        behavior_labels = self.varlabels["behavior"].values() if "behavior" in self.varlabels else []
        for m_idx, method in enumerate(methods_to_compare):
            timecourse = self.results[method]['timecourse']
            stims = self.data_behavior
            correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
            if np.isscalar(correlation):
                correlation = np.array([[correlation]])
            plot_widget.plot_perf_correlations_ens_group(correlation, m_idx, title=f"{method}".upper(), xlabel="Behavior", group_labels=behavior_labels)

    def update_cross_behavior(self):
        """
        Updates the plot of cross-correlation with behavior data by initiating a parallel worker thread.

        This method retrieves the plot widget for displaying cross-correlation with behavior, 
        and starts a worker thread to perform the cross-correlation update in parallel, 
        which is then rendered in the plot widget.

        :return: None
        :rtype: None
        """
        plot_widget = self.findChild(MatplotlibWidget, 'performance_plot_crossensbehavior')
        worker_crossbeha = WorkerRunnable(self.update_cross_behavior_parallel, plot_widget)
        #worker_crossbeha.signals.result_ready.connect(self.update_cross_ens_stim_end)
        self.threadpool.start(worker_crossbeha)
    def update_cross_behavior_parallel(self, plot_widget):
        """
        Calculates and updates the cross-correlation between the ensemble timecourses and behavior data.

        This method computes the cross-correlation for each selected method (e.g., SVD, PCA, ICA, etc.) 
        between the ensemble timecourses and the behavior data, and then updates the plot widget 
        with the results. It handles the display by organizing the plots into a specified layout 
        based on the number of methods being compared.

        :param plot_widget: The widget used to display the cross-correlation plots.
        :type plot_widget: MatplotlibWidget
        :return: None
        :rtype: None
        """
        methods_to_compare = self.tempvars['methods_to_compare']
        cant_methods_compare = self.tempvars['cant_methods_compare']
        plot_colums = 2 if cant_methods_compare == 1 else cant_methods_compare
        # Calculate cross-correlation
        max_ens = 0
        for method in methods_to_compare:
            max_ens = max(self.results[method]['ensembles_cant'], max_ens)
        plot_widget.canvas.setFixedHeight(400*max_ens)
        plot_widget.set_subplots(max_ens, plot_colums)
        behavior_labels = list(self.varlabels["behavior"].values()) if "behavior" in self.varlabels else []
        for m_idx, method in enumerate(methods_to_compare):
            for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                cross_corrs = []
                for stimtime in self.data_behavior:
                    cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                    cross_corrs.append(cross_corr)
                cross_corrs = np.array(cross_corrs)
                plot_widget.plot_perf_cross_ens_stims(cross_corrs, lags, m_idx, ens_idx, group_prefix="Beha", title=f"Cross correlation Ensemble {ens_idx+1} and behavior - Method " + f"{method}".upper(), group_labels=behavior_labels)

    def get_data_to_save(self):
        """
        Prepares data for saving based on the selected checkboxes in the GUI.

        This method compiles data from various sources, including input data, results, analysis parameters, 
        algorithms results, ensembles comparison, and performance metrics. It gathers and organizes the data 
        to be saved into a dictionary format, which can later be written to a file. The specific data included 
        depends on the user's selections in the GUI checkboxes.

        It collects the following data:
        - Input data (e.g., dFFo, neuronal activity, stimuli, coordinates, behavior, etc.)
        - Results from the ensembles comparison
        - Parameters used in the analysis
        - Full algorithm results
        - Performance metrics for ensembles, including correlation with stimuli, behavior, and cross-correlation

        The method also generates a timestamp and stores it in the data under the key "EnsemblesGUI".

        :return: A dictionary containing the data to be saved.
        :rtype: dict
        """

        data = {}
        now = datetime.now()
        formatted_time = now.strftime("%d%m%y_%H%M%S")
        self.ensgui_desc["date"] = formatted_time
        data["EnsemblesGUI"] = self.ensgui_desc
        if self.save_check_input.isChecked() and self.save_check_input.isEnabled():
            print("GUI Save: Getting input data...")
            data['input_data'] = {}
            if hasattr(self, "data_dFFo"):
                data['input_data']["dFFo"] = self.data_dFFo
            if hasattr(self, "data_neuronal_activity"):
                data['input_data']["neuronal_activity"] = self.data_neuronal_activity
            if hasattr(self, "data_coordinates"):
                data['input_data']["coordinates"] = self.data_coordinates
            if hasattr(self, "data_stims"):
                data['input_data']["stims"] = self.data_stims
            if hasattr(self, "data_cells"):
                data['input_data']["cells"] = self.data_cells
            if hasattr(self, "data_behavior"):
                data['input_data']["behavior"] = self.data_behavior
        if self.save_check_minimal.isChecked() and self.save_check_minimal.isEnabled():
            print("GUI Save: Getting minimal results...")
            data['results'] = self.results
        if self.save_check_params.isChecked() and self.save_check_params.isEnabled():
            print("GUI Save: Getting analysis parameters...")
            data["parameters"] = self.params
        if self.save_check_full.isChecked() and self.save_check_full.isEnabled():
            print("GUI Save: Getting algorithms full results...")
            data['algorithms_results'] = self.algotrithm_results
        if self.save_check_enscomp.isChecked() and self.save_check_enscomp.isEnabled():
            print("GUI Save: Getting ensembles compare...")
            data["ensembles_compare"] = {}
            for criteria in ["neus_in_ens", "timecourse"]:
                data["ensembles_compare"][criteria] = {}
                all_elements, labels = self.ensembles_compare_get_elements_labels(criteria)
                for method in ["Cosine", "Euclidean", "Correlation", "Jaccard"]:
                    similarity_matrix = self.ensembles_compare_get_simmatrix(method, all_elements)
                    data["ensembles_compare"][criteria][method] = similarity_matrix
            data["ensembles_compare"]["labels"] = labels
        if self.save_check_perf.isChecked() and self.save_check_perf.isEnabled():
            print("GUI Save: Getting ensembles performance...")
            data["ensembles_performance"] = {}

            data["ensembles_performance"]["correlation_cells"] = {}
            for method in list(self.results.keys()):
                data["ensembles_performance"]["correlation_cells"][method] = {}
                for ens_idx, ens in enumerate(self.results[method]['neus_in_ens']):
                    members = [c_idx for c_idx in range(len(ens)) if ens[c_idx] == 1]
                    activity_neus_in_ens = self.data_neuronal_activity[members, :]
                    correlation = metrics.compute_correlation_inside_ensemble(activity_neus_in_ens)
                    data["ensembles_performance"]["correlation_cells"][method][f"Ensemble {ens_idx+1}"] = correlation

            if hasattr(self, "data_stims"):
                data["ensembles_performance"]["correlation_ensembles_stimuli"] = {}
                stims = self.data_stims
                for method in list(self.results.keys()):
                    timecourse = self.results[method]['timecourse']
                    correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
                    data["ensembles_performance"]["correlation_ensembles_stimuli"][method] = correlation
                
                data["ensembles_performance"]["crosscorr_ensembles_stimuli"] = {}
                for method in self.results.keys():
                    data["ensembles_performance"]["crosscorr_ensembles_stimuli"][method] = {}
                    for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                        cross_corrs = []
                        for stimtime in self.data_stims:
                            cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                            cross_corrs.append(cross_corr)
                        data["ensembles_performance"]["crosscorr_ensembles_stimuli"][method][f"Ensemble {ens_idx+1}"] = cross_corrs
            
            if hasattr(self, "data_behavior"):
                data["ensembles_performance"]["correlation_ensembles_behavior"] = {}
                behavior = self.data_behavior
                for method in list(self.results.keys()):
                    timecourse = self.results[method]['timecourse']
                    correlation = metrics.compute_correlation_with_stimuli(timecourse, behavior)
                    data["ensembles_performance"]["correlation_ensembles_behavior"][method] = correlation
                
                data["ensembles_performance"]["crosscorr_ensembles_behavior"] = {}
                for method in self.results.keys():
                    data["ensembles_performance"]["crosscorr_ensembles_behavior"][method] = {}
                    for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                        cross_corrs = []
                        for stimtime in behavior:
                            cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                            cross_corrs.append(cross_corr)
                        data["ensembles_performance"]["crosscorr_ensembles_behavior"][method][f"Ensemble {ens_idx+1}"] = cross_corrs
        return data

    def save_data_to_hdf5(self, group, data):
        """
        Recursively saves data to an HDF5 file group.

        This method iterates through a dictionary and saves its contents to the provided HDF5 group. 
        If a value in the dictionary is another dictionary, it creates a subgroup and recursively saves 
        its contents. 
        If the value is a list, it attempts to create a dataset in the group, catching exceptions 
        if the data cannot be saved. 
        For other data types, the method directly stores the value in the group.

        :param group: The HDF5 group to which the data will be saved.
        :type group: h5py.Group
        :param data: The data to be saved, which can be a dictionary, list, or other types.
        :type data: dict
        """

        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(str(key))
                self.save_data_to_hdf5(subgroup, value)
            elif isinstance(value, list):
                try:
                    group.create_dataset(key, data=value)
                except:
                    print(f" GUI Saving: Could not save a variable called {key}, maybe it is not a matrix nor scalar.")
            else:
                group[key] = value
    def save_results_hdf5(self):
        """
        Saves the current results to an HDF5 file.

        This method retrieves the data to be saved using :meth:`MainWindow.get_data_to_save()`, 
        prompts the user to choose a location and name for the file, and then saves the data in HDF5 format. 
        The file is saved using the :meth:`MainWindow.save_data_to_hdf5` method to recursively write 
        the data into the file.

        The file is named based on the current date and a prefix "EnsGUI", and the user is prompted 
        to choose a location and file name via a file dialog.

        :raises IOError: If the file could not be saved.
        """

        data_to_save = self.get_data_to_save()
        proposed_name = f"EnsGUI_{data_to_save['EnsemblesGUI']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HDF5 Results File", proposed_name, "HDF5 Files (*.h5);;All files(*)")
        if file_path:
            try:
                self.update_console_log("Saving results in HDF5 file...")
                with h5py.File(file_path, 'w') as hdf_file:
                    self.save_data_to_hdf5(hdf_file, data_to_save)
                self.update_console_log("Done saving.", "complete")
            except Exception as e:
                self.update_console_log(f"Error saving file: {str(e)}", "error")
                raise IOError(f"Could not save the file to {file_path}.")

    def save_results_pkl(self):
        """
        Saves the current results to a Pickle (.pkl) file.

        This method retrieves the data to be saved using the :meth:`MainWindow.get_data_to_save()` method, 
        prompts the user to choose a location and file name using a file dialog, and then saves 
        the retrieved data into a Pickle (.pkl) file. 
        The Pickle format is a binary format used for serializing Python objects.

        The default file name is based on the current date and a prefix "EnsGUI". 
        If the user cancels or does not provide a file name, the function will not proceed.

        If an error occurs while saving the file, an `IOError` is raised to notify the user that 
        the file could not be saved.

        :raises IOError: If there is an error saving the file, for example, if the file path is invalid 
            or the file cannot be written to.
        """
        data_to_save = self.get_data_to_save()
        proposed_name = f"EnsGUI_{data_to_save['EnsemblesGUI']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PKL Results File", proposed_name, "Pickle Files (*.pkl);;All files(*)")
        if file_path:
            try:
                self.update_console_log("Saving results in Python Pickle file...")
                with open(file_path, 'wb') as pkl_file:
                    pickle.dump(data_to_save, pkl_file)
                self.update_console_log("Done saving.", "complete")
            except Exception as e:
                self.update_console_log(f"Error saving file: {str(e)}", "error")
                raise IOError(f"Could not save the file to {file_path}.")

    def save_results_mat(self):
        """
        Saves the current results to a MATLAB (.mat) file.

        This method retrieves the data to be saved using the :meth:`MainWindow.get_data_to_save` method, 
        prompts the user to choose a location and file name using a file dialog, and then saves 
        the retrieved data into a MATLAB .mat file using the `scipy.io.savemat()` function.

        The default file name is based on the current date and a prefix "EnsGUI". 
        If the user cancels or does not provide a file name, the function will not proceed.

        If an error occurs while saving the file, an `IOError` is raised to notify the user that the 
        file could not be saved.

        :raises IOError: If there is an error saving the file, for example, if the file path is invalid 
            or the file cannot be written to.
        """
        
        data_to_save = self.get_data_to_save()
        proposed_name = f"EnsGUI_{data_to_save['EnsemblesGUI']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save MATLAB Results File", proposed_name, "MATLAB Files (*.mat);;All files(*)")
        if file_path:
            try:
                self.update_console_log("Saving results in MATLAB file...")
                scipy.io.savemat(file_path, data_to_save)
                self.update_console_log("Done saving.", "complete")
            except Exception as e:
                self.update_console_log(f"Error saving file: {str(e)}", "error")
                raise IOError(f"Could not save the file to {file_path}.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()  
    #sys.exit(app.exec())