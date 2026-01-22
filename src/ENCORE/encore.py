import sys
import h5py
import os
import scipy.io 
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import time
from datetime import datetime
import pickle
import yaml
import qdarktheme
import importlib

from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt6.QtWidgets import QTableWidgetItem, QColorDialog

from PyQt6.uic import loadUi
from PyQt6.QtCore import QDateTime, Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal, QSize
from PyQt6.QtGui import QTextCursor, QDoubleValidator, QFont, QIcon

from data.load_data import FileTreeModel
from data.assign_data import assign_data_from_file

import utils.metrics as metrics
from utils.text_formatting import format_nums_to_string

from gui.MatplotlibWidget import MatplotlibWidget

import plotters.encore_plots as encore_plots

import validators.algorithm_results as validate_results

from PyQt6.QtWidgets import (
    QWidget,
    QTabWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
    QScrollArea
)

class QtLoggerAdapter:
    def __init__(self, log_signal):
        self.log_signal = log_signal
    def __call__(self, message, level="log"):
        self.log_signal.emit(message, level)

class WorkerSignals(QObject):
    """
    Signals used by the worker thread.

    :ivar result_ready: Signal emitted when the long-running function finishes execution and returns a result.
    :vartype result_ready: pyqtSignal(object)
    """
    result_ready = pyqtSignal(object)
    log = pyqtSignal(str, str)

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
        logger = QtLoggerAdapter(self.signals.log)
        result = self.long_running_function(*self.args, **self.kwargs, logger=logger)
        self.signals.result_ready.emit(result)

class MainWindow(QMainWindow):
    def __init__(self, gui_colors={}, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        super(MainWindow, self).__init__()
        loadUi("gui/MainWindow.ui", self)
        self.setWindowTitle('ENCORE - Ensembles Comparison and Recognition')

        self.ensgui_desc = {
            "analyzer": "ENCORE",
            "date": "",
            "gui_version": "2.0.0"
        }
        
        self.gui_colors = gui_colors

        self.threadpool = QThreadPool()
        
        # Initialize algorithms
        self.initialize_user_algorithms()

        # Initialize the GUI
        self.reset_gui()
        
        # Check if MATLAB is available
        try:
            import matlab
            self.matlab_available = True
        except ImportError as exc:
            self.update_console_log("Could not load MATLAB engine.", "error")
            self.update_console_log("The algorithms that requires MATLAB are not available.", "warning")
            self.matlab_available = False
        
        # Dark mode button
        self.dark_mode.clicked.connect(self.set_theme)
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

        ## Ensembles visualizer
        self.ensvis_tabs.currentChanged.connect(self.ensvis_tabchange)
        self.envis_slide_selectedens.editingFinished.connect(self.update_ensemble_visualization)
        self.envis_slide_selectedens.valueChanged.connect(self.update_ensemble_visualization)
        self.ensvis_check_onlyens.stateChanged.connect(self.update_ens_vis_coords)
        self.ensvis_check_onlycont.stateChanged.connect(self.update_ens_vis_coords)
        self.ensvis_check_cellnum.stateChanged.connect(self.update_ens_vis_coords)

        # Ensemble compare
        self.enscomp_spinbox_stim.valueChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_spinbox_behavior.valueChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_visopts_setneusize.clicked.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_visopts_showcells.stateChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_btn_color.clicked.connect(self.enscomp_get_color)
        self.enscomp_check_coords.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_ens.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_check_neus.stateChanged.connect(self.ensembles_compare_update_ensembles)

        self.enscomp_check_show_stim.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_btn_color_stim.clicked.connect(self.enscomp_get_color_stims)

        self.enscomp_check_show_behavior.stateChanged.connect(self.ensembles_compare_update_ensembles)
        self.enscomp_btn_color_behavior.clicked.connect(self.enscomp_get_color_behavior)

        self.enscomp_combo_select_result.currentTextChanged.connect(self.ensembles_compare_update_combo_results)

        ## Numeric validator
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.Notation.StandardNotation)

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

        # Saving
        self.save_btn_hdf5.clicked.connect(self.save_results_hdf5)
        self.save_btn_npz.clicked.connect(self.save_results_npz)
        self.save_btn_pkl.clicked.connect(self.save_results_pkl)
        self.save_btn_mat.clicked.connect(self.save_results_mat)

    def reset_gui(self):
        """
        Reset the GUI, delete all the variables and analysis results.
        Also clears all the figures and restores the analysis and display options.
        The result is the GUI just like the first time you opened it.
        """
        # Delete all previous results
        self.results = {}
        self.algorithm_results = {}
        self.params = {}
        self.varlabels = {}
        self.tempvars = {}
        if hasattr(self, "data_coordinates_generated"):
            delattr(self, "data_coordinates")
            self.data_coordinates_generated = False
            
        # Update buttons for ensemble visualization, performance and comparison
        for algorithm_key, algorithm_cfg in self.algorithms_config.items():
            button = self.findChild(QWidget, f'ensvis_btn_{algorithm_key}')
            if button:
                button.setEnabled(False)
            check = self.findChild(QWidget, f'performance_check_{algorithm_key}')
            if check:
                check.setEnabled(False)
            spinbox = self.findChild(QWidget, f"enscomp_spinbox_{algorithm_key}")
            if spinbox:
                spinbox.setValue(0)
                spinbox.setEnabled(False)
            label_with_max = self.findChild(QWidget, f"enscomp_spinbox_lbl_max_{algorithm_key}")
            if label_with_max:
                label_with_max.setText("0")
                label_with_max.setEnabled(False)

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
        self.save_btns = [self.save_btn_hdf5, 
                     self.save_btn_npz,
                     self.save_btn_pkl, 
                     self.save_btn_mat]
        for btn in self.save_btns:
            btn.setEnabled(False)

        # Clear the preview plots
        default_txt = "Load or select a variable to see a preview here"
        self.findChild(MatplotlibWidget, 'data_preview').reset(default_txt)

        self.ensvis_edit_numens.setText("0")
        self.envis_slide_selectedens.setEnabled(False)
        self.envis_slide_selectedens.blockSignals(True)
        self.envis_slide_selectedens.setMaximum(2)
        self.envis_slide_selectedens.setValue(1)
        self.envis_slide_selectedens.blockSignals(False)
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
            "stims": {'color': 'black'},
            "behavior": {'color': 'yellow'},
            "sim_neus": {'method': 'Jaccard', 'colormap': 'viridis'},
            "sim_time": {'method': 'Cosine', 'colormap': 'plasma'},
        }
        # Update visualizations options for the ensembles comparison
        for algorithm_key, algorithm_cfg in self.algorithms_config.items():
            selected_color = algorithm_cfg.get("ensemble_color", "black")
            self.enscomp_visopts[algorithm_key] = {
                'enscomp_check_coords': True, 
                'enscomp_check_ens': True, 
                'enscomp_check_neus': False, 
                'color': selected_color, 
                'enabled': False
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
        sliders = [self.enscomp_spinbox_stim,
                   self.enscomp_spinbox_behavior]
        for obj in sliders:
            obj.blockSignals(True)
            obj.setEnabled(False)
            obj.setRange(0, 0)
            obj.setValue(0)
            obj.blockSignals(False)
            
        if not hasattr(self, "data_stims"):
            self.enscomp_spinbox_lbl_max_stim.setEnabled(False)
            self.enscomp_spinbox_lbl_max_stim.setText("0")
            self.enscomp_spinbox_lbl_stim.setEnabled(False)
            self.enscomp_spinbox_lbl_stim.setText("Label")
            self.enscomp_check_show_stim.setEnabled(False)
            self.enscomp_btn_color_stim.setEnabled(False)
        if not hasattr(self, "data_behavior"):
            self.enscomp_spinbox_lbl_max_behavior.setEnabled(False)
            self.enscomp_spinbox_lbl_max_behavior.setText("0")
            self.enscomp_spinbox_lbl_behavior.setEnabled(False)
            self.enscomp_spinbox_lbl_behavior.setText("Label")
            self.enscomp_check_show_behavior.setEnabled(False)
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
        default_txt = "Perform and select at least one analysis and load\n behavior data to see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_corrbehavior').reset(default_txt)
        default_txt = "Perform and select at least one analysis and load\n stimulation data to see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_crossensstim').reset(default_txt)
        default_txt = "Perform and select at least one analysis and load\n behavior data to see the metrics"
        self.findChild(MatplotlibWidget, 'performance_plot_crossensbehavior').reset(default_txt)

        default_txt = "Perform and select at least one analysis\n to see the metrics"
        self.findChild(MatplotlibWidget, 'enscomp_plot_map').reset(default_txt)
        self.findChild(MatplotlibWidget, 'enscomp_plot_neusact').reset(default_txt)
        self.findChild(MatplotlibWidget, 'enscomp_plot_sim_elements').reset(default_txt)
        self.findChild(MatplotlibWidget, 'enscomp_plot_sim_times').reset(default_txt)

    # Create and initialize the elements for each algorithm
    def initialize_user_algorithms(self):
        """
        Build algorithm tabs dynamically from YAML configuration.
        """
        # Read the config file
        config = {}
        config_yaml_path = "config\encore_runners_config.yaml"
        
        if os.path.exists(config_yaml_path):
            with open(config_yaml_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            self.update_console_log(f"YAML config file not found in {config_yaml_path}", "error")
            self.update_console_log(f"No algorithm will be loaded", "error")
        
        # Extract the runners from config and keep only the enabled ones
        runners = config.get("encore_runners", {})
        to_delete = []
        for runner_key, runner_cfg in runners.items():
            enabled = runner_cfg.get("enabled", False)
            if not enabled:
                to_delete.append(runner_key)
        for key in to_delete:
            del runners[key]
            
        self.update_console_log(f"{len(runners)} algorithms will be loaded.", "log")
        self.algorithms_config = dict(runners)
        
        self.update_console_log("Loading algorithms defined in config file...", "log")
        # Create one tab per algorithm
        encore_algorithms_tab = self.findChild(QWidget, 'main_encore_algorithms_tab')
        if encore_algorithms_tab:
            self.encore_algorithms_tab_layout = QVBoxLayout(encore_algorithms_tab)
        
            tabs = QTabWidget()
            tabs.setObjectName("encore_algorithms_tabs")
            self.encore_algorithms_tab_layout.addWidget(tabs)

            for algorithm_key, algorithm_cfg in runners.items():
                algorithm_cfg['short_name'] = algorithm_key
                self._create_algorithm_tab(algorithm_cfg)
        else:
            raise RuntimeError("The tab 'main_encore_algorithms_tab' could not be found, make sure it's present in the main tabs.")
        
        # Initialize buttons for the ensembles visualization
        self._initialize_visualization_buttons(runners)
        # Initialize check boxes for ensembles performance comparison
        self._initialize_performance_checks(runners)
        # Initialize selectors in ensembles comparisons
        self._initialize_comparisons_selectors(runners)
        
        self.update_console_log("Done loading algorithms.", "complete")
    def _create_algorithm_tab(self, algorithm_cfg: dict):
        tab = QWidget()
        main_layout = QHBoxLayout(tab)

        # Analysis parameters on the left
        parameters_box = self._create_parameters_box(algorithm_cfg)
        main_layout.addWidget(parameters_box, stretch=1)

        # Results visualization on the right
        figures_box = self._create_figures_box(algorithm_cfg.get("figures", []))
        main_layout.addWidget(figures_box, stretch=2)

        # Add the tab with the new algorithm
        encore_algorithms_tabs = self.findChild(QWidget, 'encore_algorithms_tabs')
        encore_algorithms_tabs.addTab(tab, algorithm_cfg.get("full_name", "Algorithm"))  
    def _create_parameters_box(self, algorithm_cfg: dict) -> QGroupBox:
        """
        Creates the 'Analysis parameters' box for one algorithm.
        """
        short_name = algorithm_cfg.get("short_name", "algo")
        
        analysis_box = QGroupBox("Analysis parameters")
        analysis_box.setMinimumSize(QSize(370, 0))
        analysis_box.setMaximumSize(QSize(370, 16777215))
        analysis_layout = QVBoxLayout(analysis_box)
        analysis_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Input data box
        input_data_box = QGroupBox("Input data")
        input_form = QFormLayout(input_data_box)
        input_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        needed_data = algorithm_cfg.get("needed_data", [])
        
        needed_data_names = {
            'data_neuronal_activity': 'Binary Neuronal Activity',
            'data_dFFo': 'dFFo Activity',
            'data_coordinates': 'Coordinates',
            'data_stims': 'Stimulation data',
            'data_cells': 'Cells data',
            'data_behavior': 'Behavior data'
        }
        
        for data_key in needed_data:
            data_name = needed_data_names.get(data_key, f"Unknown: {data_key}")
            left_label = QLabel(data_name)
            bold_font = QFont()
            bold_font.setBold(True)
            left_label.setFont(bold_font)
            if hasattr(self, data_key):
                right_label = QLabel("Loaded")
            else:
                right_label = QLabel("Nothing selected")
            right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_label.setObjectName(f"{short_name}_{data_key}_status_label")

            input_form.addRow(left_label, right_label)

        analysis_layout.addWidget(input_data_box)

        # Parameters to adjust
        params_box = QGroupBox("Parameters to adjust")
        params_layout = QVBoxLayout(params_box)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        parameters = algorithm_cfg.get("parameters", {})

        for param_key, param_cfg in parameters.items():
            label = QLabel(param_cfg["display_name"])
            widget = self._create_parameter_widget(param_cfg)
            form_layout.addRow(label, widget)

        params_layout.addLayout(form_layout)
        
        # Load defaults button
        load_defaults_btn = QPushButton("Load default values")
        load_defaults_btn.clicked.connect(
            lambda _, cfg=algorithm_cfg: self.load_algorithm_defaults(cfg)
        )
        params_layout.addWidget(load_defaults_btn, alignment=Qt.AlignmentFlag.AlignRight)
        analysis_layout.addWidget(params_box)
        analysis_layout.addStretch()

        # Run analysis button
        run_button = QPushButton("Run analysis")
        run_button.setObjectName(f"{short_name}_run_analysis_button")
        run_button.setEnabled(False)
        run_button.clicked.connect(
            lambda _, cfg=algorithm_cfg: self.run_algorithm(cfg)
        )

        analysis_layout.addWidget(run_button)

        # Algorithm source text
        description = algorithm_cfg.get("source", "")
        description_box = QPlainTextEdit()
        description_box.setReadOnly(True)
        description_box.setPlainText(description)
        description_box.setMaximumSize(QSize(16777215, 75))
        analysis_layout.addWidget(description_box)

        return analysis_box
    def _create_parameter_widget(self, cfg: dict):
        default = cfg.get("default_value")
        min_val = cfg.get("min_value")
        max_val = cfg.get("max_value")
        
        MAX_INT = 250000000
        
        if cfg.get("type", "") == "enum":
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)

            button_group = QButtonGroup(widget)
            button_group.setExclusive(True)

            default = cfg.get("default_value")

            for option in cfg.get("options", []):
                radio = QRadioButton(option["label"])
                value = option["value"]

                # Store value in Qt user data
                radio.setProperty("value", value)

                if value == default:
                    radio.setChecked(True)
                
                # Add a unique name for each radial button
                button_name = cfg.get("object_name", "") + f"_{value}"
                radio.setObjectName(button_name)

                button_group.addButton(radio)
                layout.addWidget(radio)

            # Keep reference for later retrieval
            widget.button_group = button_group
        
        # Boolean
        elif isinstance(default, bool):
            widget = QCheckBox()
            widget.setChecked(default)

        # Integer
        elif isinstance(default, int):
            widget = QSpinBox()
            if min_val is not None:
                widget.setMinimum(min_val)
            if max_val is not None:
                if max_val == 'MAX_INT':
                    max_val = MAX_INT
                widget.setMaximum(max_val)
            widget.setValue(default)

        # Float
        elif isinstance(default, float):
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            if min_val is not None:
                widget.setMinimum(min_val)
            if max_val is not None:
                if max_val == 'MAX_INT':
                    max_val = MAX_INT
                widget.setMaximum(max_val)
            widget.setValue(default)

        # Fallback: string
        else:
            widget = QLineEdit()
            widget.setText(str(default))
        widget.setObjectName(cfg.get("object_name", ""))
        widget.setToolTip(cfg.get("description", ""))

        return widget
    def _create_figures_box(self, figures: list) -> QGroupBox:
        figures_box = QGroupBox("Results visualization")
        figures_layout = QVBoxLayout(figures_box)

        tabs = QTabWidget()

        for fig in figures:
            # Container widget (goes inside scroll area)
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)

            plot = MatplotlibWidget()
            plot.reset("Run this analysis to see results here")
            plot.setObjectName(fig["name"])

            # Important for resizing behavior
            plot.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )

            content_layout.addWidget(plot)

            # Scroll area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content_widget)

            tabs.addTab(scroll, fig["display_name"])

        figures_layout.addWidget(tabs)
        return figures_box
    def _initialize_visualization_buttons(self, runners: dict):
        # Update buttons for ensemble visualization
        buttons_container = self.findChild(QWidget, 'ensvis_algorithm_buttons_box')
        if buttons_container:
            buttons_layout = QHBoxLayout(buttons_container)
            buttons_layout.setObjectName('ensvis_algorithm_buttons_box_layout')
            for algorithm_key, algorithm_cfg in runners.items():
                button_name = f'ensvis_btn_{algorithm_key}'
                button = QPushButton(algorithm_key.upper())
                button.setObjectName(button_name)
                button.clicked.connect(
                    lambda _, cfg=algorithm_cfg: self.visualize_ensembles(cfg)
                )
                button.setEnabled(False)
                buttons_layout.addWidget(button)
    def _initialize_performance_checks(self, runners: dict):
        # Update check boxes for performance comparison
        checks_container = self.findChild(QWidget, 'performance_checks_box')
        if checks_container:
            checks_layout = QHBoxLayout(checks_container)
            checks_layout.setObjectName('performance_checks_box_layout')
            for algorithm_key, algorithm_cfg in runners.items():
                check_name = f'performance_check_{algorithm_key}'
                check = QCheckBox(algorithm_key.upper())
                check.setObjectName(check_name)
                check.setEnabled(False)
                check.stateChanged.connect(self.performance_check_change)
                checks_layout.addWidget(check)
        
            button = QPushButton("Compare")
            button.setObjectName('performance_btn_compare')
            button.clicked.connect(self.performance_compare)
            button.setEnabled(False)
            checks_layout.addWidget(button)
    def _initialize_comparisons_selectors(self, runners: dict):
        # Update the ensembles selectors in ensembles comparison
        enscomp_box = self.findChild(QWidget, 'enscomp_selector_box')
        if enscomp_box:
            big_font = QFont()
            big_font.setPointSize(12)
            big_font.setBold(True)
            
            size_policy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
            size_policy.setHorizontalStretch(0)
            size_policy.setVerticalStretch(0)
        
            enscomp_layout = QVBoxLayout(enscomp_box)
            enscomp_layout.setObjectName('enscomp_selector_box_layout')
            size_policy.setHeightForWidth(enscomp_box.sizePolicy().hasHeightForWidth())
            enscomp_box.setSizePolicy(size_policy)
            for algorithm_key, algorithm_cfg in runners.items():
                # Small container for the elements
                container_widget = QWidget(enscomp_box)
                container_widget_layout = QHBoxLayout(container_widget)
                container_widget_layout.setContentsMargins(0, 0, 0, 0)
                
                # Label for the name of the algorithm
                label_with_name = QLabel(algorithm_key.upper())
                label_with_name.setObjectName(f"enscomp_algo_lbl_{algorithm_key}")
                label_with_name.setFont(big_font)
                label_with_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
                size_policy.setHeightForWidth(label_with_name.sizePolicy().hasHeightForWidth())
                label_with_name.setSizePolicy(size_policy)
                container_widget_layout.addWidget(label_with_name)
                
                # Spinbox to select the ensemble
                spinbox = QSpinBox()
                spinbox.setObjectName(f"enscomp_spinbox_{algorithm_key}")
                spinbox.setEnabled(False)
                spinbox.valueChanged.connect(self.ensembles_compare_update_ensembles)
                container_widget_layout.addWidget(spinbox)
                
                # Simple label just to separate the spinbox and the total ensembles label
                label_separator = QLabel("/")
                label_separator.setObjectName(f"enscomp_separe_lbl_{algorithm_key}")
                label_separator.setFont(big_font)
                container_widget_layout.addWidget(label_separator)
                
                # Label for the maximum amount of ensembles to select from
                label_with_max = QLabel("0")
                label_with_max.setObjectName(f"enscomp_spinbox_lbl_max_{algorithm_key}")
                label_with_max.setEnabled(False)
                container_widget_layout.addWidget(label_with_max)
                
                # Empty widget to show the color of the current ensemble
                color_flag = QWidget()
                color_flag.setObjectName(f"enscomp_colorflag_{algorithm_key}")
                color_flag.setMinimumSize(QSize(10, 0))
                color_flag.setMaximumSize(QSize(10, 16777215))
                color_flag.setAutoFillBackground(False)
                container_widget_layout.addWidget(color_flag)
                
                enscomp_layout.addWidget(container_widget)

    ## Theme for the UI
    def set_theme(self):
        set_dark_mode = self.dark_mode.isChecked()
        if set_dark_mode:
            qdarktheme.setup_theme(
                custom_colors=self.gui_colors
            )
        else:
            qdarktheme.setup_theme("light",
                custom_colors=self.gui_colors
            )
    
    ## Console log  
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
    
    ## File browsing and variable selector
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
                with h5py.File(fname, 'r') as hdf_file:
                    self.file_model_type = "hdf5"
                    self.file_model = FileTreeModel(hdf_file, model_type=self.file_model_type, MATLAB_available=self.matlab_available)
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading file.", "complete")
            elif file_extension == ".npz":
                self.update_console_log("Generating file structure...")
                numpy_file = np.load(fname)
                file_structure = {}
                for key in numpy_file.files:
                    file_structure[key] = numpy_file[key]
                self.file_model_type = "np_flatten"
                self.file_model = FileTreeModel(file_structure, model_type=self.file_model_type, MATLAB_available=self.matlab_available)
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading file.", "complete")
            elif file_extension == ".pkl":
                self.update_console_log("Generating file structure...")
                with open(fname, 'rb') as file:
                    pkl_file = pickle.load(file)
                self.file_model_type = "pkl"
                self.file_model = FileTreeModel(pkl_file, model_type=self.file_model_type, MATLAB_available=self.matlab_available)
                self.tree_view.setModel(self.file_model)
                self.update_console_log("Done loading file.", "complete")
            elif file_extension == '.mat':
                if self.matlab_available:
                    self.update_console_log("Generating file structure...")
                    mat_file = scipy.io.loadmat(fname)
                    self.file_model_type = "mat"
                    self.file_model = FileTreeModel(mat_file, model_type=self.file_model_type, MATLAB_available=self.matlab_available)
                    self.tree_view.setModel(self.file_model)
                    self.update_console_log("Done loading matlab file.", "complete")
                else:
                    self.update_console_log("Loading of MATLAB files is currently unavailable", "warning")
            elif file_extension == '.csv':
                self.update_console_log("Generating file structure...")
                self.file_model_type = "csv"
                with open(fname, 'r', newline='') as csvfile:
                    self.file_model = FileTreeModel(csvfile, model_type=self.file_model_type, MATLAB_available=self.matlab_available)
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
        current_tab_name = self.main_tabs.tabText(index)

        if current_tab_name == "Ensembles compare":
            if len(self.results) > 0:
                self.ensembles_compare_update_ensembles()
        elif current_tab_name == "ENCORE algorithms":
            self.update_user_analysis_requirements()
    
    ## Cehck algorithm requirements  
    def update_user_analysis_requirements(self):
        algorithms_config = self.algorithms_config
        
        for algorithm in algorithms_config.values():
            if algorithm.get("language", "") == 'MATLAB':
                needed_loaded = self.matlab_available
            else:
                needed_loaded = True
                    
            short_name = algorithm.get("short_name")
            needed_data = algorithm.get("needed_data", [])
            for data in needed_data:
                if hasattr(self, data):
                    status_label = "Loaded"
                else:
                    status_label = "Not loaded"
                    needed_loaded = False
                    
                label_name = f"{short_name}_{data}_status_label"
                status_label_widget = self.findChild(QLabel, label_name)
                if status_label_widget:
                    status_label_widget.setText(status_label)

            run_button = self.findChild(QWidget, f"{short_name}_run_analysis_button")
            if run_button:
                run_button.setEnabled(needed_loaded)

    ## Set variables from input file
    def set_dFFo(self):
        """
        Sets the :attr:`MainWindow.dFFo` dataset by assigning data from a file and updating the relevant UI components.

        This method loads the dFFo dataset from the selected file.
        It then updates the UI to reflect that the dataset has been
        assigned, enables buttons for further edition, and logs the update message.

        :return: None
        """
        self.data_dFFo = assign_data_from_file(self.file_selected_var_path, self.source_filename, self.file_model_type)
        neus, frames = self.data_dFFo.shape
        self.btn_clear_dFFo.setEnabled(True)
        self.btn_view_dFFo.setEnabled(True)
        self.lbl_dffo_select.setText("Assigned")
        self.lbl_dffo_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set dFFo dataset - Identified {neus} cells and {frames} time points. Please, verify the data preview.", msg_type="complete")
        self.view_dFFo()
        self.save_check_input.setEnabled(True)
        for btn in self.save_btns:
            btn.setEnabled(True)
    def set_neuronal_activity(self):
        """
        Sets the :attr:`MainWindow.data_neuronal_activity` dataset by assigning data from a file and updating the relevant UI components.

        This method loads the data_neuronal_activity dataset from the selected file and extracts the number of cells 
        (cant_neurons) and time points (cant_timepoints). It then updates the UI to reflect that the dataset has been
        assigned, enables buttons for further edition, and logs the update message.

        :return: None
        """
        self.data_neuronal_activity = assign_data_from_file(self.file_selected_var_path, self.source_filename, self.file_model_type)
        self.cant_neurons, self.cant_timepoints = self.data_neuronal_activity.shape
        self.btn_clear_neuronal_activity.setEnabled(True)
        self.btn_view_neuronal_activity.setEnabled(True)
        self.lbl_neuronal_activity_select.setText("Assigned")
        self.lbl_neuronal_activity_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Binary Neuronal Activity dataset - Identified {self.cant_neurons} cells and {self.cant_timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_neuronal_activity()
        self.save_check_input.setEnabled(True)
        for btn in self.save_btns:
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
        data_coordinates = assign_data_from_file(self.file_selected_var_path, self.source_filename, self.file_model_type)
        self.data_coordinates = data_coordinates[:, 0:2]
        neus, dims = self.data_coordinates.shape
        self.btn_clear_coordinates.setEnabled(True)
        self.btn_view_coordinates.setEnabled(True)
        self.lbl_coordinates_select.setText("Assigned")
        self.lbl_coordinates_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Coordinates dataset - Identified {neus} cells and {dims} dimentions. Please, verify the data preview.", msg_type="complete")
        self.view_coordinates()
        self.save_check_input.setEnabled(True)
        for btn in self.save_btns:
            btn.setEnabled(True)
    def set_stims(self):
        """
        Sets the value of the :attr:`MainWindow.data_stims` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_stims = assign_data_from_file(self.file_selected_var_path, self.source_filename, self.file_model_type)
        self.data_stims = data_stims
        stims, timepoints = data_stims.shape
        self.btn_clear_stim.setEnabled(True)
        self.btn_view_stim.setEnabled(True)
        self.lbl_stim_select.setText("Assigned")
        self.lbl_stim_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Stimuli dataset - Identified {stims} stims and {timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_stims()
        self.save_check_input.setEnabled(True)
        for btn in self.save_btns:
            btn.setEnabled(True)
    def set_cells(self):
        """
        Sets the value of the :attr:`MainWindow.data_cells` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_cells = assign_data_from_file(self.file_selected_var_path, self.source_filename, self.file_model_type)
        self.data_cells = data_cells
        stims, cells = data_cells.shape
        self.btn_clear_cells.setEnabled(True)
        self.btn_view_cells.setEnabled(True)
        self.lbl_cells_select.setText("Assigned")
        self.lbl_cells_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Selected cells dataset - Identified {stims} groups and {cells} cells. Please, verify the data preview.", msg_type="complete")
        self.view_cells()
        self.save_check_input.setEnabled(True)
        for btn in self.save_btns:
            btn.setEnabled(True)
    def set_behavior(self):
        """
        Sets the value of the :attr:`MainWindow.data_behavior` variable.

        The assigned value is the one of the selected variable in the variable broswer.
        This function also updates the buttons related to load and clear the variable
        and triggers the visualization function.
        At the end the function shows the loaded data in the :attr:`MainWindow.data_preview` widget.
        """
        data_behavior = assign_data_from_file(self.file_selected_var_path, self.source_filename, self.file_model_type)
        self.data_behavior = data_behavior
        behav_shape = data_behavior.shape
        if len(behav_shape) > 1:
            behaviors, timepoints = data_behavior.shape
        else:
            timepoints = data_behavior.shape[0]
            behaviors = 1
        self.btn_clear_behavior.setEnabled(True)
        self.btn_view_behavior.setEnabled(True)
        self.lbl_behavior_select.setText("Assigned")
        self.lbl_behavior_select_name.setText(self.file_selected_var_name)
        self.update_console_log(f"Set Behavior dataset - Identified {behaviors} behaviors and {timepoints} time points. Please, verify the data preview.", msg_type="complete")
        self.view_behavior()
        self.save_check_input.setEnabled(True)
        for btn in self.save_btns:
            btn.setEnabled(True)
    
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
        encore_plots.preview_dataset(plot_widget, self.data_dFFo, ylabel='Cell', yitems_labels=cell_labels)
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
        encore_plots.preview_dataset(plot_widget, self.data_neuronal_activity==0, ylabel='Cell', cmap='gray', yitems_labels=cell_labels)
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
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        encore_plots.preview_coordinates2D(plot_widget, self.data_coordinates)
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
        encore_plots.preview_dataset(plot_widget, preview_data==0, ylabel='Stim', cmap='gray', yitems_labels=stim_labels)
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
        encore_plots.preview_dataset(plot_widget, preview_data==0, xlabel="Cell", ylabel='Group', cmap='gray', yitems_labels=selectcell_labels)
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
        if len(self.data_behavior.shape) > 1:
            behaviors, timepoints = self.data_behavior.shape
        else:
            timepoints = self.data_behavior.shape[0]
            behaviors = 1
        self.update_edit_validators(lim_sup_x=timepoints, lim_sup_y=behaviors)
        plot_widget = self.findChild(MatplotlibWidget, 'data_preview')
        preview_data = self.data_behavior
        if len(preview_data.shape) == 1:
            zeros_array = np.zeros_like(preview_data)
            preview_data = np.row_stack((preview_data, zeros_array))
        self.varlabels_setup_tab(preview_data.shape[0])
        self.update_enscomp_options("behavior")
        behavior_labels = list(self.varlabels["behavior"].values()) if "behavior" in self.varlabels else []
        encore_plots.preview_dataset(plot_widget, preview_data, ylabel='Behavior', yitems_labels=behavior_labels)

    ## Edit buttons
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
        self.edit_edit_binsize.setRange(1, lim_sup_x)
        self.edit_edit_xstart.setRange(0, lim_sup_x)
        self.edit_edit_xend.setRange(1, lim_sup_x)
        self.edit_edit_ystart.setRange(0, lim_sup_y)
        self.edit_edit_yend.setRange(1, lim_sup_y)
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
        mat = np.atleast_2d(mat)  # ensures its 2D (1, N) even if originally 1D
        elements, timepoints = mat.shape

        if bin_size >= timepoints:
            self.update_console_log("Enter a bin size smaller than the current amount of timepoints. Nothing has been changed.", "warning")
            return mat

        num_bins = timepoints // bin_size
        bin_mat = np.zeros((elements, num_bins))

        for i in range(num_bins):
            window = mat[:, i * bin_size:(i + 1) * bin_size]
            if bin_method == "mean":
                bin_mat[:, i] = np.mean(window, axis=1)
            elif bin_method == "sum":
                bin_mat[:, i] = np.sum(window, axis=1)
            else:
                self.update_console_log("Invalid bin_method. Use 'mean' or 'sum'.", "error")

        # If input was 1D, return 1D output
        if mat.shape[0] == 1:
            return bin_mat.flatten()
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

    ## Labels for the variables
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

    ## Running algorithms
    def load_algorithm_defaults(self, algorithm_cfg: dict):
        """
        Load default values from config into parameter widgets.
        """
        parameters = algorithm_cfg.get("parameters", {})
        short_name = algorithm_cfg.get("short_name", "Algo").upper()
        for param_cfg in parameters.values():
            obj_name = param_cfg.get("object_name")
            default = param_cfg.get("default_value")
            possible_type = param_cfg.get("type", "")
            
            if possible_type == "enum":
                obj_name = f"{obj_name}_{default}"
                default = True

            widget = self.findChild(QWidget, obj_name)
            if widget:
                if hasattr(widget, "setValue"):
                    widget.setValue(default)
                elif hasattr(widget, "setChecked"):
                    widget.setChecked(default)
                elif hasattr(widget, "setText"):
                    widget.setText(str(default))
        self.update_console_log(f"Loaded default values for {short_name}.")
    def collect_algorithm_parameters(self, algorithm_cfg: dict) -> dict:
        """
        Collect parameters for one algorithm from the UI using YAML configuration.

        :param algorithm_cfg: Algorithm configuration dictionary from YAML
        :return: Dictionary of parameter_name -> value
        """
        collected_params = {}

        parameters = algorithm_cfg.get("parameters", {})

        for param_name, param_cfg in parameters.items():
            obj_name = param_cfg.get("object_name")
            default = param_cfg.get("default_value")

            widget = self.findChild(QWidget, obj_name)

            # Widget missing so fallback to default
            if widget is None:
                collected_params[param_name] = default
                continue

            # Type aware extraction
            try:
                if param_cfg.get("type", "") == "enum":
                    for btn in widget.button_group.buttons():
                        if btn.isChecked():
                            value = btn.property("value")
                            break
                    else:
                        value = default
                elif isinstance(default, bool) and isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(default, int) and isinstance(widget, QSpinBox):
                    value = widget.value()
                elif isinstance(default, float) and isinstance(widget, QDoubleSpinBox):
                    value = widget.value()
                elif isinstance(default, str) and isinstance(widget, QLineEdit):
                    value = widget.text()
                # Fallback: try generic accessors
                elif hasattr(widget, "value"):
                    value = widget.value()
                elif hasattr(widget, "text"):
                    value = widget.text()
                else:
                    value = default
            except Exception:
                value = default

            collected_params[param_name] = value

        return collected_params
    def collect_algorithm_data(self, algorithm_cfg: dict) -> list:
        requested_data = []
        needed_data = algorithm_cfg.get("needed_data", [])
        for data_key in needed_data:
            if hasattr(self, data_key):
                requested_data.append(getattr(self, data_key))
                if data_key == 'data_dFFo':
                    self.cant_neurons, self.cant_timepoints = self.data_dFFo.shape
                elif data_key == 'data_neuronal_activity':
                    self.cant_neurons, self.cant_timepoints = self.data_neuronal_activity.shape
            else:
                self.update_console_log(f"The requested data key '{data_key}' has not been loaded", "error")
            
        return requested_data
    def run_algorithm(self, algorithm_cfg: dict):
        # Deactivate the running button while running 
        short_name = algorithm_cfg.get('short_name', 'Algorithm')
        run_button = self.findChild(QWidget, f"{short_name}_run_analysis_button")
        if run_button:
            run_button.setEnabled(False)
            
        params = self.collect_algorithm_parameters(algorithm_cfg)
        data = self.collect_algorithm_data(algorithm_cfg)
        
        # Save the parameters used
        self.params[short_name] = params

        # Clean all the figures in case there was something previously
        if short_name in self.results:
            del self.results[short_name]
        figures_list = algorithm_cfg.get("figures", [])
        for figure_info in figures_list:
            object_name = figure_info.get("name", None)
            if object_name:
                gui_object = self.findChild(MatplotlibWidget, object_name)
                if object_name:
                    gui_object.reset("Waiting for new plots...")

        worker = WorkerRunnable(
            self.run_analysis_function,
            algorithm_cfg,
            params,
            data,
        )

        worker.signals.result_ready.connect(
            lambda result, cfg=algorithm_cfg: self.run_algorithm_end(cfg, result)
        )
        worker.signals.log.connect(self.update_console_log)

        self.threadpool.start(worker)
    def run_analysis_function(self, algorithm_cfg: dict, params: dict, data: np.ndarray, logger=None):
        """
        Dynamically load and execute an analysis function.

        :param algorithm_cfg: Algorithm configuration from YAML
        :param params: Validated parameters dictionary
        :param data: NumPy data matrix
        :raises RuntimeError: If function cannot be loaded or executed
        """
        function_name = algorithm_cfg.get("analysis_function")
        code_folder_path = algorithm_cfg.get("folder_path")
        short_name = algorithm_cfg.get("short_name")

        if not function_name:
            logger("No analysis_function defined in algorithm config", "error")

        module_path = "runners.encore"

        # Import module
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger(f"Analysis module '{module_path}' could not be imported", "error")
            logger(str(exc), "error")

        # Retrieve function
        func = getattr(module, function_name, None)
        if func is None or not callable(func):
            logger(f"Function '{function_name}' not found or not callable in '{module_path}'", "error")
        
        # Execute
        try:
            result = func(
                data,
                params,
                relative_folder_path=code_folder_path,
                include_answer=True,
                logger=logger
            )
            
            # Validate the result dictionary
            validated_result = validate_results.validate_analysis_output(result, neurons=self.cant_neurons, timepoints=self.cant_timepoints)
            result = validated_result.model_dump()
            
            if result["success"]:
                # Check if the analysis found any ensemble
                num_ensembles = result['results']['ensembles_cant']
                if num_ensembles > 0:
                    # Update parameters that where calculated
                    parameters_info = algorithm_cfg.get("parameters", {})
                    params_to_update = result.get('update_params', {})
                    for param_name, param_value in params_to_update.items():
                        param_info = parameters_info.get(param_name, {})
                        param_object_name = param_info.get("object_name")
                        gui_object = self.findChild(QWidget, param_object_name)
                        if hasattr(gui_object, "setValue"):
                            gui_object.setValue(param_value)
                        elif hasattr(gui_object, "setChecked"):
                            gui_object.setChecked(param_value)
                        elif hasattr(gui_object, "setText"):
                            gui_object.setText(str(param_value))

                    # Plotting results
                    logger(f"{short_name.upper()} Plotting and saving results...", "log")
                    start_time = time.time()
                    # Save results
                    self.algorithm_results[short_name] = result["answer"]
                    self.results[short_name] = result["results"]
                    # Plot the results
                    try:
                        self.plot_algorithm_plots(algorithm_cfg, result["answer"])
                    except Exception as exc:
                        logger(
                            f"Error while plotting the results of the algorithm for '{function_name}'",
                            "error"
                        )
                        logger(str(exc), "error")
                        
                    # Update the GUI
                    self.we_have_results()
                    end_time = time.time()
                    plot_times = end_time - start_time
                
                    logger(f"{short_name.upper()} Done plotting and saving...", "complete")
                else:
                    logger(f"{short_name.upper()} Plotting results...", "log")
                    start_time = time.time()
                    try:
                        self.plot_algorithm_plots(algorithm_cfg, result["answer"])
                    except Exception as exc:
                        logger(
                            f"Error while plotting the results of the algorithm for '{function_name}'",
                            "error"
                        )
                        logger(str(exc), "error")
                    end_time = time.time()
                    plot_times = end_time - start_time
                    logger(f"{short_name.upper()} Done plotting...", "complete")
                
            return [result["engine_time"], result["algorithm_time"], plot_times, num_ensembles]
        except Exception as exc:
            logger(
                f"Error while executing analysis function '{function_name}'",
                "error"
            )
            logger(str(exc), "error")
    def plot_algorithm_plots(self, algorithm_cfg: dict, answer: dict):
        function_name = algorithm_cfg.get("plot_function")

        if not function_name:
            raise RuntimeError("No plot_function defined in algorithm config")

        module_path = "plotters.encore"

        # Import module
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise RuntimeError(f"Plot module '{module_path}' could not be imported") from exc

        # Retrieve function
        func = getattr(module, function_name, None)
        if func is None or not callable(func):
            raise RuntimeError(f"Function '{function_name}' not found or not callable in '{module_path}'")

        # Collect the plots for the selected algorithm
        figures_list = algorithm_cfg.get("figures", [])
        figures_dict = {}
        for figure_info in figures_list:
            object_name = figure_info.get("name", None)
            if object_name:
                gui_object = self.findChild(MatplotlibWidget, object_name)
                if object_name:
                    figures_dict[object_name] = gui_object
                        
        # Plot
        try:
            func(figures_dict, answer)
        except Exception as exc:
            raise RuntimeError(f"{exc}")
    def run_algorithm_end(self, algorithm_cfg, times):
        if times is None:
            return
        short_name = algorithm_cfg.get("short_name", "").upper()
        self.update_console_log(f"Done executing the {short_name} algorithm", "complete") 
        self.update_console_log(f"- Loading the engine took {times[0]:.2f} seconds") 
        self.update_console_log(f"- Running the algorithm took {times[1]:.2f} seconds") 
        self.update_console_log(f"- Plotting and saving results took {times[2]:.2f} seconds")
        if times[3] > 0:
            self.update_console_log(f"The {short_name} analysis found {times[3]} ensembles", "complete")
        else:
            self.update_console_log(f"The {short_name} analysis didn't found any ensembles. Try changing the selected parameters.", "warning")
        
        # Rectivate the button once the algorithm finishes
        short_name = algorithm_cfg.get('short_name', 'Algorithm')
        run_button = self.findChild(QWidget, f"{short_name}_run_analysis_button")
        if run_button:
            run_button.setEnabled(True)

    def we_have_results(self):
        """
        Updates the UI for the ensembles compare and performance analysis given the available analysis results.

        :return: None
        :rtype: None
        """
        for analysis_name in self.results.keys():
            ensvis_button_name = f'ensvis_btn_{analysis_name}'
            ensvis_button = self.findChild(QWidget, ensvis_button_name)
            if ensvis_button:
                ensvis_button.setEnabled(True)
            
            performance_check_name = f'performance_check_{analysis_name}'
            performance_check = self.findChild(QWidget, performance_check_name)
            if performance_check:
                performance_check.setEnabled(True)
            
            self.ensembles_compare_update_opts(analysis_name)
    
        save_itms = [self.save_check_minimal,
                self.save_check_params,
                self.save_check_full,
                self.save_check_enscomp,
                self.save_check_perf]
        for itm in save_itms:
            itm.setEnabled(True)
        self.tempvars["showed_sim_maps"] = False
    
    ## Ensembles visualizer
    def ensvis_tabchange(self, index):
        """
        Identifies the tab change in the ensamble visualizer for asynchronous loading of plots.

        The loading of all the comparations and plots could be slow when loaded all at the same time.
        For this, the plots are loaded only when the user reaches the relevant tab.

        :param index: Index of the tab opened by the user.
        :type index: int
        """
        current_tab_name = self.ensvis_tabs.tabText(index)
        if self.tempvars['ensvis_shown_results']:
            if current_tab_name == 'General':
                pass
            elif current_tab_name == 'All the spatial distributions':
                if hasattr(self, "data_coordinates"):
                    if not self.tempvars['ensvis_shown_tab1']:
                        self.tempvars['ensvis_shown_tab1'] = True
                        self.update_ensvis_allcoords()
            elif current_tab_name == 'All the binary activations':
                if not self.tempvars['ensvis_shown_tab2']:
                    self.tempvars['ensvis_shown_tab2'] = True
                    self.update_ensvis_allbinary()
            elif current_tab_name == 'All the dFFo':
                if hasattr(self, "data_dFFo"):
                    if not self.tempvars['ensvis_shown_tab3']:
                        self.tempvars['ensvis_shown_tab3'] = True
                        self.update_ensvis_alldFFo()
            elif current_tab_name == 'All the ensembles activations':
                if not self.tempvars['ensvis_shown_tab4']:
                    self.tempvars['ensvis_shown_tab4'] = True
                    self.update_ensvis_allens()
    def visualize_ensembles(self, algorithm_cfg:dict):
        """
        Loads the results of the SVD into the ensembles visualizer.

        This is done this way to use only one function to update the ensembles visualizer.
        The variable :attr:`MainWindow.ensemble_currently_shown` is used to set the algorithm to show.
        Then the global funtion :meth:`MainWindow.update_analysis_results` is executed.
        """
        self.ensemble_currently_shown = algorithm_cfg['short_name']
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
        self.update_ensemble_visualization(1)
    def update_ensemble_visualization(self, value=-1):
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
        if value == -1:
            value = self.envis_slide_selectedens.value()
            
        curr_analysis = self.ensemble_currently_shown
        curr_ensemble = value

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
            encore_plots.plot_ensemble_dFFo(plot_widget, dFFo_ens, idx_corrected_members, ensemble_timecourse)
    
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
        plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_map')
        encore_plots.plot_coordinates2D_highlight(plot_widget, self.data_coordinates, self.current_idx_corrected_members, self.current_idx_corrected_exclusive, only_ens, only_contours, show_numbers)
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
            encore_plots.plot_all_dFFo(plot_widget, dFFo_ens, idx_corrected_members, current_ens)
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
            
            encore_plots.plot_all_coords(plot_widget, self.data_coordinates, idx_corrected_members, idx_corrected_exclusive, row, col)
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
            encore_plots.plot_all_binary(plot_widget, activity, members, current_ens, current_ens)
    def update_ensvis_allens(self):
        """
        Plot the activations of every ensemble.

        This function creates figure with the timecourse of every ensemble identified.

        :return: None
        :rtype: None
        """
        curr_analysis = self.ensemble_currently_shown
        plot_widget = self.findChild(MatplotlibWidget, 'ensvis_plot_allens')
        encore_plots.plot_ensembles_timecourse(plot_widget, self.results[curr_analysis]['timecourse'])

    ## Ensembles comparison
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
        ens_selector_name = f"enscomp_spinbox_{algorithm}"
        ens_selector = self.findChild(QWidget, ens_selector_name)
        if not ens_selector:
            raise RuntimeError(f"The ensemble selector for {algorithm} is missing")
        
        selector_label_max_name = f"enscomp_spinbox_lbl_max_{algorithm}"
        selector_label_max = self.findChild(QWidget, selector_label_max_name)
        if not selector_label_max:
            raise RuntimeError(f"The ensemble selector label for {algorithm} is missing")

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
        ens_selector.setRange(1, self.results[algorithm]['ensembles_cant']) # Set the maximum value
        ens_selector.setValue(1)
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
            spinbox = self.enscomp_spinbox_stim
            lbl_max = self.enscomp_spinbox_lbl_max_stim
            lbl_label = self.enscomp_spinbox_lbl_stim
            check_show = self.enscomp_check_show_stim
            color_pick = self.enscomp_btn_color_stim
            shp = self.data_stims.shape
            max_val = shp[0] if len(shp) > 1 else 1
        elif exp_data == "behavior":
            spinbox = self.enscomp_spinbox_behavior
            lbl_max = self.enscomp_spinbox_lbl_max_behavior
            lbl_label = self.enscomp_spinbox_lbl_behavior
            check_show = self.enscomp_check_show_behavior
            color_pick = self.enscomp_btn_color_behavior
            shp = self.data_behavior.shape
            max_val = shp[0] if len(shp) > 1 else 1
        # Activate the spinbox
        spinbox.blockSignals(True)
        spinbox.setEnabled(True)
        spinbox.setRange(1, max_val)
        spinbox.setValue(1)
        lbl_label.setText(f"{1}")
        lbl_label.setEnabled(True)
        lbl_max.setText(f"{max_val}")
        lbl_max.setEnabled(True)
        spinbox.blockSignals(False)
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
            "stims": self.enscomp_spinbox_stim,
            "behavior": self.enscomp_spinbox_behavior
        }
        for algorithm_key in self.algorithms_config.keys():
            enscomp_spinbox_name = f"enscomp_spinbox_{algorithm_key}"
            enscomp_spinbox = self.findChild(QWidget, enscomp_spinbox_name)
            if enscomp_spinbox:
                ens_selector[algorithm_key] = enscomp_spinbox
                
                # Update the color of the ensemble
                color_flag_name = f"enscomp_colorflag_{algorithm_key}"
                color_flag = self.findChild(QWidget, color_flag_name)
                if color_flag:
                    color_flag.setStyleSheet(f"background-color: {self.enscomp_visopts[algorithm_key]['color']};")
                
        for key, spinbox in ens_selector.items():
            if spinbox.isEnabled():
                ens_idx = spinbox.value()
                if key == 'stims':
                    if hasattr(self, "data_stims"):
                        ensembles_to_compare[key] = {}
                        ensembles_to_compare[key]["ens_idx"] = ens_idx-1
                        ensembles_to_compare[key]["timecourse"] = self.data_stims[ens_idx-1,:].copy()
                elif key == "behavior":
                    if hasattr(self, "data_behavior"):
                        ensembles_to_compare[key] = {}
                        ensembles_to_compare[key]["ens_idx"] = ens_idx-1
                        ensembles_to_compare[key]["timecourse"] = self.data_behavior[ens_idx-1,:].copy()
                else:
                    if key in self.results:
                        ensembles_to_compare[key] = {}
                        ensembles_to_compare[key]["ens_idx"] = ens_idx-1
                        ensembles_to_compare[key]["neus_in_ens"] = self.results[key]['neus_in_ens'][ens_idx-1,:].copy()
                        ensembles_to_compare[key]["timecourse"] = self.results[key]['timecourse'][ens_idx-1,:].copy()
        
        # Update the labels indicator
        if ens_selector['stims'].isEnabled():
            selected_stim = ens_selector['stims'].value()-1
            if "stim" in self.varlabels:
                stim_labels = list(self.varlabels["stim"].values())
                stim_label = f"{stim_labels[selected_stim]}"
            else:
                stim_label = f"{selected_stim}"
            self.enscomp_spinbox_lbl_stim.setText(stim_label)
        if ens_selector['behavior'].isEnabled():
            selected_behavior = ens_selector['behavior'].value()-1
            if "behavior" in self.varlabels:
                behavior_labels = list(self.varlabels["behavior"].values())
                behavior_label = f"{behavior_labels[selected_behavior]}"
            else:
                behavior_label = f"{selected_behavior}"
            self.enscomp_spinbox_lbl_behavior.setText(behavior_label)

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
            self.data_coordinates_generated = True
        # Stablish the dimention of the map
        max_x = np.max(self.data_coordinates[:, 0])
        max_y = np.max(self.data_coordinates[:, 1])
        lims = [max_x, max_y]

        mixed_ens = []

        list_colors_freq = [[] for l in range(self.cant_neurons)] 

        for key, ens_data in ensembles_to_compare.items():
            if key == "stims":
                continue
            if key == "behavior":
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
        encore_plots.enscomp_update_map(map_plot, lims, members_idx, members_freq, members_coords, members_colors, neuron_size)
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
            elif key == "behavior":
                if self.enscomp_check_show_behavior.isChecked():
                    # Get the currently selected stimulation
                    selected_behavior = ens_data["ens_idx"]
                    # Get the timecourse
                    timecourses.append(ens_data["timecourse"].copy())
                    # Get the label, if any
                    if "behavior" in self.varlabels:
                        behavior_labels = list(self.varlabels["behavior"].values())
                        behavior_label = f"Behav {behavior_labels[selected_behavior]}"
                    else:
                        behavior_label = f"Behav {selected_behavior}"
                    new_ticks.append(f"{behavior_label}")
                    colors.append(self.enscomp_visopts['behavior']['color'])
            else:
                if self.enscomp_visopts[key]['enabled'] and self.enscomp_visopts[key]['enscomp_check_ens']:
                    new_timecourse = ens_data["timecourse"].copy()
                else:
                    new_timecourse = []
                timecourses.append(new_timecourse)

                if self.enscomp_visopts[key]['enabled'] and self.enscomp_visopts[key]['enscomp_check_neus']:
                    new_members = ens_data["neus_in_ens"].copy()
                    if hasattr(self, "data_dFFo"):
                        cells_activity_mat = self.data_dFFo[new_members.astype(bool), :]
                    else:
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
        encore_plots.enscomp_update_timelines(plot_widget, new_ticks, cells_activities, [], timecourses, colors, self.cant_timepoints)
    def ensembles_compare_get_elements_labels(self, criteria):
        """
        Retrieves elements and their labels for ensembles comparison based on a given criterion.

        This method extracts elements from the results dictionary according to the specified criterion 
        (e.g., neurons in ensemble or timecourse). It also generates labels for these elements, 
        indicating the algorithm and the ensemble index.

        If a stimulation matrix is provided, each stimulus is added along with their label.

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
        if criteria == "timecourse":
            if hasattr(self, "data_stims"):
                stims, timepoints = self.data_stims.shape
                for stim in range(stims):
                    all_elements.append(self.data_stims[stim,:])
                    if "stim" in self.varlabels:
                        stim_labels = list(self.varlabels["stim"].values())
                        stim_label = f"Stim {stim_labels[stim]}"
                    else:
                        stim_label = f"Stim {stim}"
                    labels.append(stim_label)
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
        
        encore_plots.enscomp_plot_similarity(plot_widget, similarity_matrix, labels, color)
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

    ## Handle colors
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
            # Convert the color to a Matplotlib compatible format (hex string)
            color_hex = color.name()
            current_method = self.enscomp_combo_select_result.currentText().lower()
            self.enscomp_visopts[current_method]['color'] = color_hex
            self.ensembles_compare_update_ensembles()
    def enscomp_get_color_stims(self):
        """
        Opens the QColorDialog to select a color for the stimulation.

        The selected color will be applied to the elements of the stimulation.
        :return: None
        :rtype: None
        """
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            self.enscomp_visopts['stims']['color'] = color_hex
            self.ensembles_compare_update_ensembles()
    def enscomp_get_color_behavior(self):
        """
        Opens the QColorDialog to select a color for the behavior.

        The selected color will be applied to the elements of the behavior.
        :return: None
        :rtype: None
        """
        color = QColorDialog.getColor()
        # Check if a color was selected
        if color.isValid():
            # Convert the color to a Matplotlib-compatible format (hex string)
            color_hex = color.name()
            self.enscomp_visopts['behavior']['color'] = color_hex
            self.ensembles_compare_update_ensembles()

    ## Ensembles performance
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
        for algorithm_key in self.algorithms_config.keys():
            check_name = f"performance_check_{algorithm_key}"
            check = self.findChild(QWidget, check_name)
            if check:
                if check.isChecked():
                    methods_to_compare.append(algorithm_key)

        self.tempvars['methods_to_compare'] = methods_to_compare
        self.tempvars['cant_methods_compare'] = len(methods_to_compare)
        compare_button = self.findChild(QWidget, 'performance_btn_compare')
        if compare_button:
            if self.tempvars['cant_methods_compare'] > 0:
                compare_button.setEnabled(True)
            else:
                compare_button.setEnabled(False)
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
        worker_corrstim.signals.log.connect(self.update_console_log)
        self.threadpool.start(worker_corrstim) 
    def update_corr_stim_parallel(self, plot_widget, logger=None):
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
        logger("Plotting correlations with stimuli...", "log")
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
            encore_plots.plot_perf_correlations_ens_group(plot_widget, correlation, m_idx, title=f"{method}".upper(), xlabel="Stims", group_labels=stim_labels)            
        logger("Done plotting correlations with stimuli.", "complete")
        
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
        worker_corrcells.signals.log.connect(self.update_console_log)
        self.threadpool.start(worker_corrcells) 
    def update_correlation_cells_parallel(self, plot_widget, logger=None):
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
        logger("Plotting correlations between cells...", "log")
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
                encore_plots.plot_perf_correlations_cells(plot_widget, correlation, cells_names, col_idx, row_idx, title=f"Cells in ensemble {row_idx+1} - Method " + f"{method}".upper())
        logger("Done plotting correlations between cells.", "complete")

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
        worker_crosstim.signals.log.connect(self.update_console_log)
        self.threadpool.start(worker_crosstim) 
    def update_cross_ens_stim_parallel(self, plot_widget, logger=None):
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
        logger("Plotting cross correlation with stimuli...", "log")
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
                encore_plots.plot_perf_cross_ens_stims(plot_widget, cross_corrs, lags, m_idx, ens_idx, group_prefix="Stim", title=f"Cross correlation Ensemble {ens_idx+1} and stimuli - Method " + f"{method}".upper(), group_labels=stim_labels)          
        logger("Done plotting cross correlation with stimuli.", "complete")

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
        worker_corrbeha.signals.log.connect(self.update_console_log)
        self.threadpool.start(worker_corrbeha) 
    def update_corr_behavior_parallel(self, plot_widget, logger=None):
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
        logger("Plotting correlation with behavior...", "log")
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
            encore_plots.plot_perf_correlations_ens_group(plot_widget, correlation, m_idx, title=f"{method}".upper(), xlabel="Behavior", group_labels=behavior_labels)
        logger("Done plotting correlation with behavior.", "complete")

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
        worker_crossbeha.signals.log.connect(self.update_console_log)
        self.threadpool.start(worker_crossbeha)
    def update_cross_behavior_parallel(self, plot_widget, logger=None):
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
        logger("Plotting cross correlation with behavior...", "log")
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
                encore_plots.plot_perf_cross_ens_stims(plot_widget, cross_corrs, lags, m_idx, ens_idx, group_prefix="Beha", title=f"Cross correlation Ensemble {ens_idx+1} and behavior - Method " + f"{method}".upper(), group_labels=behavior_labels)
        logger("Done plotting cross correlation with behavior.", "complete")

    ## Saving
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
        data["ENCORE"] = {}
        data["ENCORE"]["info"] = self.ensgui_desc
        if self.save_check_input.isChecked() and self.save_check_input.isEnabled():
            self.update_console_log(" - Saving: Getting input data...")
            data["ENCORE"]['input_data'] = {}
            if hasattr(self, "data_dFFo"):
                data["ENCORE"]['input_data']["dFFo"] = self.data_dFFo
            if hasattr(self, "data_neuronal_activity"):
                data["ENCORE"]['input_data']["neuronal_activity"] = self.data_neuronal_activity
            if hasattr(self, "data_coordinates"):
                data["ENCORE"]['input_data']["coordinates"] = self.data_coordinates
            if hasattr(self, "data_stims"):
                data["ENCORE"]['input_data']["stims"] = self.data_stims
            if hasattr(self, "data_cells"):
                data["ENCORE"]['input_data']["cells"] = self.data_cells
            if hasattr(self, "data_behavior"):
                data["ENCORE"]['input_data']["behavior"] = self.data_behavior
        if self.save_check_minimal.isChecked() and self.save_check_minimal.isEnabled():
            self.update_console_log(" - Saving: Getting minimal results...")
            data["ENCORE"]['results'] = self.results
        if self.save_check_params.isChecked() and self.save_check_params.isEnabled():
            self.update_console_log(" - Saving: Getting analysis parameters...")
            data["ENCORE"]["parameters"] = self.params
        if self.save_check_full.isChecked() and self.save_check_full.isEnabled():
            self.update_console_log(" - Saving: Getting algorithms full results...")
            data["ENCORE"]['algorithms_results'] = self.algorithm_results
        if self.save_check_enscomp.isChecked() and self.save_check_enscomp.isEnabled():
            self.update_console_log(" - Saving: Getting ensembles compare...")
            data["ENCORE"]["ensembles_compare"] = {}
            for criteria in ["neus_in_ens", "timecourse"]:
                data["ENCORE"]["ensembles_compare"][criteria] = {}
                all_elements, labels = self.ensembles_compare_get_elements_labels(criteria)
                for method in ["Cosine", "Euclidean", "Correlation", "Jaccard"]:
                    similarity_matrix = self.ensembles_compare_get_simmatrix(method, all_elements)
                    data["ENCORE"]["ensembles_compare"][criteria][method] = similarity_matrix
            data["ENCORE"]["ensembles_compare"]["labels"] = labels
        if self.save_check_perf.isChecked() and self.save_check_perf.isEnabled():
            self.update_console_log(" - Saving: Getting ensembles performance...")
            data["ENCORE"]["ensembles_performance"] = {}

            data["ENCORE"]["ensembles_performance"]["correlation_cells"] = {}
            for method in list(self.results.keys()):
                data["ENCORE"]["ensembles_performance"]["correlation_cells"][method] = {}
                for ens_idx, ens in enumerate(self.results[method]['neus_in_ens']):
                    members = [c_idx for c_idx in range(len(ens)) if ens[c_idx] == 1]
                    activity_neus_in_ens = self.data_neuronal_activity[members, :]
                    correlation = metrics.compute_correlation_inside_ensemble(activity_neus_in_ens)
                    data["ENCORE"]["ensembles_performance"]["correlation_cells"][method][f"Ensemble {ens_idx+1}"] = correlation

            if hasattr(self, "data_stims"):
                data["ENCORE"]["ensembles_performance"]["correlation_ensembles_stimuli"] = {}
                stims = self.data_stims
                for method in list(self.results.keys()):
                    timecourse = self.results[method]['timecourse']
                    correlation = metrics.compute_correlation_with_stimuli(timecourse, stims)
                    data["ENCORE"]["ensembles_performance"]["correlation_ensembles_stimuli"][method] = correlation
                
                data["ENCORE"]["ensembles_performance"]["crosscorr_ensembles_stimuli"] = {}
                for method in self.results.keys():
                    data["ENCORE"]["ensembles_performance"]["crosscorr_ensembles_stimuli"][method] = {}
                    for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                        cross_corrs = []
                        for stimtime in self.data_stims:
                            cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                            cross_corrs.append(cross_corr)
                        data["ENCORE"]["ensembles_performance"]["crosscorr_ensembles_stimuli"][method][f"Ensemble {ens_idx+1}"] = cross_corrs
            
            if hasattr(self, "data_behavior"):
                data["ENCORE"]["ensembles_performance"]["correlation_ensembles_behavior"] = {}
                behavior = self.data_behavior
                for method in list(self.results.keys()):
                    timecourse = self.results[method]['timecourse']
                    correlation = metrics.compute_correlation_with_stimuli(timecourse, behavior)
                    data["ENCORE"]["ensembles_performance"]["correlation_ensembles_behavior"][method] = correlation
                
                data["ENCORE"]["ensembles_performance"]["crosscorr_ensembles_behavior"] = {}
                for method in self.results.keys():
                    data["ENCORE"]["ensembles_performance"]["crosscorr_ensembles_behavior"][method] = {}
                    for ens_idx, enstime in enumerate(self.results[method]['timecourse']):
                        cross_corrs = []
                        for stimtime in behavior:
                            cross_corr, lags = metrics.compute_cross_correlations(enstime, stimtime)
                            cross_corrs.append(cross_corr)
                        data["ENCORE"]["ensembles_performance"]["crosscorr_ensembles_behavior"][method][f"Ensemble {ens_idx+1}"] = cross_corrs
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
            if key in group:
                del group[key]
                
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
        proposed_name = f"ENCORE_{data_to_save['ENCORE']['info']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HDF5 Results File", proposed_name, "HDF5 Files (*.h5);;All files(*)")
        if file_path:
            from pathlib import Path
            file_path = Path(file_path)
            if file_path.exists:
                # Save on place
                try:
                    self.update_console_log("Saving results in HDF5 file...")
                    with h5py.File(file_path, 'a') as hdf_file:
                        self.save_data_to_hdf5(hdf_file, data_to_save)
                    self.update_console_log("Done saving.", "complete")
                except Exception as e:
                    self.update_console_log(f"Error saving file: {str(e)}", "error")
                    raise IOError(f"Could not save the file to {file_path}.")
            else:
                try:
                    self.update_console_log("Saving results in HDF5 file...")
                    with h5py.File(file_path, 'w') as hdf_file:
                        self.save_data_to_hdf5(hdf_file, data_to_save)
                    self.update_console_log("Done saving.", "complete")
                except Exception as e:
                    self.update_console_log(f"Error saving file: {str(e)}", "error")
                    raise IOError(f"Could not save the file to {file_path}.")
    def flatten_dict(self, d, parent_key="", sep="/"):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, (int, float, str, np.ndarray)):
                items[new_key] = v
        return items
    def save_results_npz(self):
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
        proposed_name = f"ENCORE_{data_to_save['ENCORE']['info']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save NPZ Results File", proposed_name, "Numpy Files (*.npz);;All files(*)")
        if file_path:
            try:
                self.update_console_log("Saving results in Numpy NPZ file...")
                flat = self.flatten_dict(data_to_save)
                np.savez(file_path, **flat)
                self.update_console_log("Done saving.", "complete")
            except Exception as e:
                self.update_console_log(f"Error saving file: {str(e)}", "error")
                #raise IOError(f"Could not save the file to {file_path}.")
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
        proposed_name = f"ENCORE_{data_to_save['ENCORE']['info']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PKL Results File", proposed_name, "Pickle Files (*.pkl);;All files(*)")
        if file_path:
            try:
                self.update_console_log("Saving results in Python Pickle file...")
                with open(file_path, 'wb') as pkl_file:
                    pickle.dump(data_to_save, pkl_file)
                self.update_console_log("Done saving.", "complete")
            except Exception as e:
                self.update_console_log(f"Error saving file: {str(e)}", "error")
                #raise IOError(f"Could not save the file to {file_path}.")
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
        proposed_name = f"ENCORE_{data_to_save['ENCORE']['info']['date']}_"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save MATLAB Results File", proposed_name, "MATLAB Files (*.mat);;All files(*)")
        if file_path:
            try:
                self.update_console_log("Saving results in MATLAB file...")
                scipy.io.savemat(file_path, data_to_save)
                self.update_console_log("Done saving.", "complete")
            except Exception as e:
                self.update_console_log(f"Error saving file: {str(e)}", "error")
                #raise IOError(f"Could not save the file to {file_path}.")

if __name__ == "__main__":
    qdarktheme.enable_hi_dpi()
    app = QApplication(sys.argv)
    custom_colors = {
        "[dark]": {
            "primary": "#9a44dc",
            "toolbar.background": "#736083"
        },
        "[light]": {
            "primary": "#652d90",
            "toolbar.background": "#c3b5cf"
        }
    }
    qdarktheme.setup_theme("light",
        custom_colors=custom_colors
    )
    app.setWindowIcon(QIcon("gui/ENCORE_logo.png")) 
    window = MainWindow(gui_colors=custom_colors)
    window.show()
    app.exec()