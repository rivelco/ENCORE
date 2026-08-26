Changelog
=========

ENCORE v3.0.0
-------------

This version introduces new parameters for the **SVD** and **ICA** algorithms, providing greater flexibility for exploratory analysis and parameter selection. These changes modify the parameters expected by the Python API and may therefore **break compatibility with existing API code**.

Please check :doc:`/extending/algorithms_config_file` for the updated algorithm parameters and their default values.

New features
~~~~~~~~~~~~

* **Parallel analysis of multiple databases through the Python API.** Different databases (e.g recording sessions) can now be analyzed in parallel. Users can specify different algorithms and parameter sets for each database, while also defining a default set of parameters to be used as a fallback. See the documentation for details.

Additions to ensemble algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following additions were introduced to facilitate exploratory data analysis and provide greater flexibility in parameter selection. These are **ENCORE-specific extensions to the original implementations of the algorithms**.

**ICA**

* **Configurable minimum and maximum number of ensembles.** Users can specify the minimum and maximum number of ensembles to extract from the principal components of the activity matrix. When specified, these values override the number of ensembles determined by the original shuffling-based selection procedure.

* **Eigenvalues included in the algorithm output.** The eigenvalues of the activity matrix are now included in the algorithm's response structure and are therefore available when saving the analysis results.

* **New ensemble weight plot.** Added a plot showing the contribution/weight of each neuron to each detected ensemble, providing a visual aid for evaluating ensemble membership.

* **New neuron inclusion threshold.** Added a threshold parameter controlling which neurons are included in each ensemble. The new ensemble weight plot can be used to help determine an appropriate threshold.

**SVD**

* **Fixed number of ensembles.** Added a parameter that allows users to explicitly specify the number of ensembles to identify. When enabled, this overrides the standard selection pipeline and uses the specified number of singular values/components as ensembles.

* **New coordinated-activity plot.** Added a plot showing the spikes contributing to coordinated activity, providing a visual aid for selecting an appropriate value for the `pks` parameter.

Fixes
~~~~~

* Fixed a crash in the **Performance** tab when starting a second analysis.

* Fixed crashes that occurred when **loading 1D arrays** into certain variables.

* Fixed an issue where the UI could become **unresponsive after errors in MATLAB-based algorithms**.

* Added **missing similarity comparisons** between ensemble activity and behavioral variables.

* Fixed the **automatic calculation of `pks` and `scut`** for SVD analysis. Setting either parameter to `0` now triggers its automatic calculation.

* Fixed **display errors in the "Ensembles compare" plots**. When behavioral or stimulation data were included in the time-course plot, some plots could overlap or be displayed in incorrect positions.

* Fixed an additional **display error in the "Ensembles compare" plots**. When **Show neurons activity** was enabled, neuronal activity could be displayed at an incorrect position in the plot.

* Fixed the **ICA "Assembly patterns" plot** when a large number of ensembles are detected. The plot size is now adjusted dynamically to maintain appropriate visibility and readability.

Enhancements
~~~~~~~~~~~~

* The **Ensembles visualizer** now displays the binary activity of individual neurons when dF/F data have not been loaded.

* Improved the **parameter descriptions and legends** for the **SVD**, **ICA**, and **X2P** algorithms to make their configuration and interpretation clearer.

* Changed the default **X2P parameter** that limited the maximum number of ensembles detected. The new default no longer imposes this limitation unless the user explicitly sets it.

* Improved code formatting using **Black** in several modules. Formatting will be progressively extended to the rest of the project.

* **Enum parameters** for algorithms are now displayed as **Combo Boxes**, providing a cleaner and more compact user interface.

* Added a dedicated function for **saving HDF5 files from dictionary structures**, which is particularly useful when performing analyses in parallel through the Python API.

* Added a dedicated function for **computing similarity matrices**, which can be used when performing analyses in parallel through the Python API.

ENCORE v2.1.0
-------------

This release introduces major improvements to ENCORE (Ensembles Comparison and Recognition), focusing on usability, extensibility, and robustness.

Simplified installation via PyPI  `pip install encore-toolkit` and more.

New features
~~~~~~~~~~~~

- ENCORE is now distributed as a standard PyPI package `pip install encore-toolkit` with no mandatory MATLAB dependency.
- Added a Python API enabling batch analyses and programmatic use without the GUI.
- Introduced a plugin-style system to add new algorithms via a YAML config + analysis function (no GUI recompilation).
- Significantly expanded documentation with full API reference, tutorials, and examples.
- Added Pydantic-based validation for data, parameters, and user-defined algorithms to prevent crashes and improve error reporting.


ENCORE v1.0.0
-------------

Features
~~~~~~~~

- Unified and user-friendly platform for identifying neuronal ensembles from calcium imaging and electrophysiology data. 
- Five validated algorithms into a single graphical interface, enabling streamlined analysis and direct comparison of neuronal ensembles and activation dynamics. 
- Browse and format variables from .mat, .pkl, .h5, and .nwb files interactively
- Visualize ensemble activity through descriptive plots, and assess the biological relevance of results using behavioral or stimulation data. 
- Flexible export options support downstream processing and reporting, making ENCORE a versatile tool for neuroscience research.

For a detailed description of each algorithm included check the `book <https://link.springer.com/book/10.1007/978-1-0716-4208-5>`_.
For a deeper understanding of the code, read the `documentation <https://rivelco.github.io/ENCORE/>`_.