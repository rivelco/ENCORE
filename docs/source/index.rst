.. ENCORE documentation master file, created by
   sphinx-quickstart on Fri Oct  4 13:38:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ENCORE Documentation
====================

**ENCORE** is an open-source software tool for the identification and comparison
of neuronal ensembles from population activity data, including calcium imaging
and electrophysiological recordings.

The software provides:

- A graphical user interface (GUI) for interactive analysis
- A modular architecture that facilitates the integration of new algorithms
- Supports running the algorithms of ENCORE directly from Python scripts
- Support for multiple published ensemble detection methods

ENCORE is designed to support both exploratory data analysis and reproducible
scientific research, with a strong emphasis on transparency, extensibility, and
community contributions.

.. note::
   ENCORE is under active development. The current version already supports
   multiple published ensemble identification algorithms, with ongoing efforts
   to expand functionality and documentation.

Getting Started
---------------

If you are new to ENCORE, start here:

- :doc:`installation`
- :doc:`quickstart`
- :doc:`user_guide/usage`

Conceptual Background
---------------------

To understand the theoretical and methodological foundations of the software,
see:

- :doc:`concepts/overview`

API Reference
-------------

For developers and advanced users, the API documentation describes the internal
structure of ENCORE and its main components:

- :doc:`api/main_window`
- :doc:`api/runners`
- :doc:`api/data/index`
- :doc:`api/plotters/index`
- :doc:`api/validators/index`
- :doc:`api/utils`

Adding New Algorithms
---------------------

If you're interested in adding new algorithms to ENCORE go to:

- :doc:`extending/adding_algorithms`

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   quickstart
   user_guide/usage

   concepts/overview

   api/main_window
   api/runners
   api/data/index
   api/plotters/index
   api/validators/index
   api/utils

   extending/adding_algorithms

   acknowledgments

