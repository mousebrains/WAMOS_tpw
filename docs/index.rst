wamos_tpw Documentation
=======================

**wamos_tpw** is a Python package for processing WAMOS marine radar data.

WAMOS (Wave and Meteorological Observation System) is a radar-based wave
measurement system. This package provides tools for:

- Loading and parsing polar radar files (.pol)
- Coordinate transformations (polar to cartesian, ship to earth reference frames)
- Data corrections (deramp, destreak)
- Shadow region detection and bearing refinement
- Export to NetCDF/Zarr formats
- Animation generation

Quick Start
-----------

.. code-block:: python

   from wamos_tpw import WamosDataset

   # Load and process data
   ds = WamosDataset('/path/to/POLAR', '2022-04-05', '2022-04-06')
   ds.load()
   ds.process()

   # Export to NetCDF
   ds.to_netcdf('output.nc')

   # Create animation
   ds.animate('output.mp4')

Installation
------------

.. code-block:: bash

   pip install wamos_tpw

Or from source:

.. code-block:: bash

   git clone https://github.com/your-repo/wamos_tpw.git
   cd wamos_tpw
   pip install -e ".[dev]"

Command Line Interface
----------------------

The package provides a CLI for common operations:

.. code-block:: bash

   # List available files
   wamos list "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR

   # Parse a single file
   wamos parse /path/to/file.pol --show-header

   # Process and combine frames
   wamos combine "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR --plot

   # Dry-run mode
   wamos --dry-run combine "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   algorithms
   cli

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
