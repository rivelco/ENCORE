Upgrading
=========

Upgrading ENCORE depend on your installation method.

.. seealso::

    Check the :doc:`/changelog` to review the main changes across versions. This is specially important when using the python API.


From a pip installation
-----------------------

If you installed ENCORE doing:

.. code:: bash

    pip install encore-toolkit

Then the upgrading process is very straightforward:

1. Activate the environment where ENCORE is installed:

.. code:: bash

    conda activate encore

2. Run the standard pip upgrade:

.. code:: bash

    pip install encore-toolkit --upgrade


From the source code
--------------------

If you installed ENCORE from the source code, for example cloning the `GitHub repository <https://github.com/rivelco/ENCORE>`_ and then doing ``pip install .`` or ``pip install -e .``, then the recommended way to upgrade is:

A. Make a `git pull` from the source repository (recommended).
B. Download the repository again. 

