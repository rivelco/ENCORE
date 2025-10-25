# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = 'ENCORE'
copyright = '2025, rivelco'
author = 'rivelco'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
]

# Mock MATLAB during doc build
autodoc_mock_imports = ["matlab", "matlab.engine", "qdarktheme"]

templates_path = ['_templates']
exclude_patterns = []

html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://www.kavlifoundation.org/",
            "html": """
                <small>
                    This project is funded by The Kavli Foundation.
                </small>
            """,
            "class": "",
        },
    ],
}