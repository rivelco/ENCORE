Adding New Algorithms to ENCORE
===============================

ENCORE is designed to be **extensible**: users can add new neuronal ensemble identification algorithms with minimal changes to the codebase.

This page describes the required steps and conventions for integrating a new algorithm so that it is automatically available in the GUI, CLI, and Python API.

Overview
--------

Adding a new algorithm to ENCORE requires:

1. Defining the algorithm configuration in a YAML file
2. Implementing the analysis function
3. (Optionally) implementing a plotting function
4. Registering the algorithm outputs for visualization

All algorithm metadata and parameters are defined declaratively using YAML, while the computational and visualization logic is implemented in Python.


Clone or download the repo
--------------------------

Use git to clone the repo or download it from the website. As stated in their webpage "Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency."[#]_. If you don't have it already, install git in your computer, check the available installers at `the official download site <https://git-scm.com/downloads>`_.

Once installed git, run the following command and then change to the directory of the repository.

.. code-block:: console

    git clone https://github.com/rivelco/ENCORE.git
    cd ENCORE

If you do not want to use the git command, you can download the repository by going to the repository `<https://github.com/rivelco/ENCORE>`_ and using the Download button.

This will download the repository as a compressed zip file. Uncompress the file to extract the folder. It's highly recommended to use an exclusive python environment for adding new algorithms.

Once the desired environment is active install the the package by running:

.. code-block:: console

    pip install -e .

Now the changes in the code will be reflected on the execution. As normally, you can launch the GUI by doing:

.. code-block:: console

    encore

Or also:

.. code-block:: console

    python -m encore


Algorithm Configuration
-----------------------

Algorithm definitions are stored in:

.. code-block:: bash
    
    encore/src/encore/config/encore_runners_config.yaml


Each algorithm is defined as an entry under the ``encore_runners`` field.

Example configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml
    
    encore_runners:
    example:
        enabled: True
        full_name: Example Algorithm
        language: Python
        folder_path: ''
        analysis_function: run_example
        plot_function: plot_example
        ensemble_color: pink

        needed_data:
        - data_dFFo
        - data_neuronal_activity

        parameters:
        int_parameter_ensembles:
            object_name: example_int_ensembles
            display_name: Number of ensembles
            description: Integer values from 0 to 8.
            default_value: 5
            min_value: 1
            max_value: 8

        int_parameter_A:
            object_name: example_int_parameter_A
            display_name: Integer parameter A
            description: Integer values from -10 to 50.
            default_value: 5
            min_value: -10
            max_value: 50

        selection_parameter:
            object_name: example_selection
            display_name: Multiple selection parameter
            description: This parameter selects one option.
            type: enum
            default_value: SUM
            options:
            - value: SUM
                label: Sum the parameters A and B
            - value: MEAN
                label: Mean of the parameters A and B

        figures:
        - name: example_plot_raster
            display_name: Raster of the activity
        - name: example_plot_dFFo
            display_name: dFFo of a neuron

        source: ENCORE example algorithm

Key configuration fields
------------------------

- ``analysis_function``  
  Name of the Python function that performs the analysis.

- ``plot_function``  
  Name of the function responsible for generating figures.

- ``needed_data``  
  Identifiers of the data objects required by the algorithm. Available values are: 'data_neuronal_activity', 'data_dFFo', 'data_coordinates', 'data_stims', 'data_cells' and 'data_behavior', corresponding with the variables in the ENCORE documentation and GUI.

- ``parameters``  
  User-configurable parameters that are automatically rendered in the GUI and
  validated before execution.

- ``figures``  
  Declares figure placeholders that will be passed to the plotting function.

Implementing the Analysis Function
----------------------------------

Analysis functions must be implemented in:

.. code-block:: bash
    
    encore/src/encore/runners/encore.py
    

Function signature
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def run_example(
        input_data: list,
        params: dict,
        code_folder_name: str = '',
        include_answer: bool = True,
        logger=None
    ):
        ...

Arguments
~~~~~~~~~

- ``input_data``  
  List of NumPy arrays or data objects specified in ``needed_data``.

- ``params``  
  Dictionary containing validated parameter values from the GUI or CLI.

- ``code_folder_name``  
  Optional relative path for external code (e.g. MATLAB scripts).

- ``include_answer``  
  If ``True``, results are returned and made available for plotting and saving.

- ``logger``  
  Optional logging callback. If provided, use it to report progress and messages.

Logging
~~~~~~~

To support both GUI and CLI usage, logging should always be conditional:

.. code-block:: python

    if logger:
        logger("Starting analysis...", "log")


This allows the same function to run silently in batch or CLI mode.

Return value
~~~~~~~~~~~~

Analysis functions should return a dictionary containing all results that may be used for plotting or saving, for example:

.. code-block:: python

    return {
        "raster": timecourse_thresholded,
        "neuron_dFFo": data_for_neuron_dFFo,
        "other_dFFo": secondary_dFFo,
        "many_neurons_dFFo": many_dFFo
    }

Implementing the Plotting Function
----------------------------------

Plotting functions must be implemented in:

.. code-block:: bash

    encore/src/encore/plotters/encore.py
    

Function signature
~~~~~~~~~~~~~~~~~~

.. code-block:: python
    
    def plot_example(figures: dict, answer: dict):
        ...

Arguments
~~~~~~~~~

- ``figures``  
  Dictionary mapping figure names (defined in YAML) to plot widgets.

- ``answer``  
  Output dictionary returned by the analysis function.

Example
~~~~~~~

.. code-block:: python

    def plot_example(figures, answer):
        raster = answer["raster"]

        plot_widget = figures.get("example_plot_raster")
        if plot_widget:
            encore_plots.preview_dataset(
                plot_widget,
                raster,
                xlabel="Timepoint",
                ylabel="Ensembles"
            )
            

Plot widgets expose Matplotlib-compatible axes and canvases and can be used directly for custom visualizations.

Execution and Integration
-------------------------

Once the configuration and functions are defined:

- The algorithm automatically appears in the GUI
- Parameters are validated and rendered dynamically
- Results can be plotted, logged, and saved using existing infrastructure
- The same algorithm can be executed via CLI or Python API

Example Algorithm
-----------------

ENCORE includes a fully working **example algorithm** that demonstrates all features described on this page. Contributors are encouraged to use it as a template when implementing new methods.

Community Contributions
-----------------------

ENCORE welcomes contributions of new ensemble identification algorithms.
When contributing:

- Follow the structure described in this document
- Include clear parameter descriptions
- Cite original sources where applicable

This design ensures that new methods remain easy to use, reproducible, and accessible to the community.

References
----------

.. [#] `<https://git-scm.com/>`_.