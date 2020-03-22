.. CovidOut documentation master file, created by
   sphinx-quickstart on Sun Mar 22 03:42:14 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CovidOut's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Module ``spreading_model``
+++++++++++++++++++++++++++

We provide a short tutorial how to run this bunch of code, see: `src.ipynb`.

I/O operations
______________

.. automodule:: spreading_model.ioops
   :members:
   :undoc-members:

Spreading model definition and experiments
___________________________________________

.. automodule:: spreading_model.spreading
   :members:
   :undoc-members:


Graphs visualisations
______________________

.. automodule:: spreading_model.visualisations
   :members:
   :undoc-members:


Module ``date_generation``
+++++++++++++++++++++++++++

Database generation
______________

.. automodule:: data_generation.gen_database
   :members:
   :undoc-members:

Run `fill_database.py` to fill generated database.


Brownian movements simulaiton
______________

.. automodule:: data_generation.gen_graphs
   :members:
   :undoc-members:


.. automodule:: data_generation.simulation
   :members:
   :undoc-members:


.. automodule:: data_generation.visualize_movement_simulation
   :members:
   :undoc-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
