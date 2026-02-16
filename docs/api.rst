API Reference
=============

This is the complete API reference for the wamos_tpw package.

Core Classes
------------

Frame
~~~~~

The fundamental data container for a single radar scan.

.. automodule:: wamos_tpw.frame
   :members:
   :undoc-members:
   :show-inheritance:

Bearing
~~~~~~~

Radar beam angle calculations in ship and earth reference frames.

.. automodule:: wamos_tpw.bearing
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

YAML-based configuration management with validation.

.. automodule:: wamos_tpw.config
   :members:
   :undoc-members:
   :show-inheritance:

Processing
----------

Deramp
~~~~~~

Range-dependent intensity correction.

.. automodule:: wamos_tpw.deramp
   :members:
   :undoc-members:
   :show-inheritance:

Destreak
~~~~~~~~

Radial streak artifact removal.

.. automodule:: wamos_tpw.destreak
   :members:
   :undoc-members:
   :show-inheritance:

File I/O
--------

Polar Files
~~~~~~~~~~~

Parser for .pol file format (supports compression).

.. automodule:: wamos_tpw.polarfile
   :members:
   :undoc-members:
   :show-inheritance:

Filenames
~~~~~~~~~

File discovery with time-based filtering.

.. automodule:: wamos_tpw.filenames
   :members:
   :undoc-members:
   :show-inheritance:

Files
~~~~~

High-level file loading interface with grouping.

.. automodule:: wamos_tpw.files
   :members:
   :undoc-members:
   :show-inheritance:

Pipelines
---------

Frame Pipeline
~~~~~~~~~~~~~~

Per-frame processing pipeline.

.. automodule:: wamos_tpw.frame_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Files Pipeline
~~~~~~~~~~~~~~

Multi-file processing pipeline with time windowing.

.. automodule:: wamos_tpw.files_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Grid
~~~~

UTM grid computation and projection.

.. automodule:: wamos_tpw.grid
   :members:
   :undoc-members:
   :show-inheritance:

Window
~~~~~~

Time window creation and frame accumulation.

.. automodule:: wamos_tpw.window
   :members:
   :undoc-members:
   :show-inheritance:

Merged Image
~~~~~~~~~~~~

Data structures for merged radar images.

.. automodule:: wamos_tpw.merged_image
   :members:
   :undoc-members:
   :show-inheritance:

Merged Viewer
~~~~~~~~~~~~~

Visualization for merged images.

.. automodule:: wamos_tpw.merged_viewer
   :members:
   :undoc-members:
   :show-inheritance:

Output Writers
~~~~~~~~~~~~~~

Output format writers (NetCDF, PNG, GeoTIFF, KML/KMZ).

.. automodule:: wamos_tpw.output_writers
   :members:
   :undoc-members:
   :show-inheritance:

Interpolator
~~~~~~~~~~~~

Multi-frame interpolation for motion correction.

.. automodule:: wamos_tpw.interpolator
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Constants
~~~~~~~~~

Physical constants for radar calculations.

.. automodule:: wamos_tpw.constants
   :members:
   :undoc-members:
   :show-inheritance:

CLI Utilities
~~~~~~~~~~~~~

CLI boilerplate utilities.

.. automodule:: wamos_tpw.cli_utils
   :members:
   :undoc-members:
   :show-inheritance:

Logging
~~~~~~~

Logging configuration utilities.

.. automodule:: wamos_tpw.logging_config
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
~~~~~~~~

Common plotting utilities and helpers.

.. automodule:: wamos_tpw.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Projection
~~~~~~~~~~

UTM projection and coordinate transformation.

.. automodule:: wamos_tpw.projection
   :members:
   :undoc-members:
   :show-inheritance:

Range
~~~~~

Slant and ground range calculations.

.. automodule:: wamos_tpw.range
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

Custom exception hierarchy for error handling.

.. automodule:: wamos_tpw.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
