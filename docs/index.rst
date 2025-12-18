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

   from wamos_tpw import ProcessedFrames, Combine

   # Load and process data
   with ProcessedFrames(
       stime='2022-04-05 14:00',
       etime='2022-04-05 15:00',
       polar_path='/path/to/POLAR'
   ) as pf:
       for period, frames in pf.itergroups():
           frames = list(frames)
           pf.process(frames)
           combine = Combine(frames)
           combine.save_frame('output.png')

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
