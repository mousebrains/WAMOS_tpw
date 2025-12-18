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

Combine
~~~~~~~

Earth-referenced image combination with ship motion compensation.

.. automodule:: wamos_tpw.combine
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

Processed Frames
~~~~~~~~~~~~~~~~

Processed frame collection with deramp/destreak pipeline.

.. automodule:: wamos_tpw.processed
   :members:
   :undoc-members:
   :show-inheritance:

Movie Generation
~~~~~~~~~~~~~~~~

MP4 movie generation from radar frame sequences.

.. automodule:: wamos_tpw.combine_movie
   :members:
   :undoc-members:
   :show-inheritance:

Shadow Detection
~~~~~~~~~~~~~~~~

Shadow region detection and bearing refinement.

.. automodule:: wamos_tpw.combine_shadow
   :members:
   :undoc-members:
   :show-inheritance:

Streaming Processing
~~~~~~~~~~~~~~~~~~~~

Memory-efficient streaming frame processing and gridding.

.. automodule:: wamos_tpw.combine_streaming
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
~~~~~~~~

Interactive viewer and frame plotting utilities.

.. automodule:: wamos_tpw.combine_plot
   :members:
   :undoc-members:
   :show-inheritance:

NetCDF Export
~~~~~~~~~~~~~

NetCDF file writer for radar data export.

.. automodule:: wamos_tpw.combine_netcdf
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Export
~~~~~~~~~~~~~~

xarray Dataset creation and export to multiple formats.

.. automodule:: wamos_tpw.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Timestamp
~~~~~~~~~

Timestamp parsing and validation utilities.

.. automodule:: wamos_tpw.timestamp
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

Exceptions
----------

Custom exception hierarchy for error handling.

.. automodule:: wamos_tpw.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Type Definitions
----------------

Type aliases for scientific computing.

.. automodule:: wamos_tpw._types
   :members:
   :undoc-members:
   :show-inheritance:

Protocols
---------

Runtime-checkable protocol definitions.

.. automodule:: wamos_tpw.protocols
   :members:
   :undoc-members:
   :show-inheritance:
