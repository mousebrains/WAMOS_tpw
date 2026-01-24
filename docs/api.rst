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

File Pipeline
~~~~~~~~~~~~~

Single file processing pipeline.

.. automodule:: wamos_tpw.file_pipeline
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

Frame Pipeline
~~~~~~~~~~~~~~

Per-frame processing pipeline.

.. automodule:: wamos_tpw.frame_pipeline
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

Type Definitions
----------------

Type aliases for scientific computing.

.. automodule:: wamos_tpw._types
   :members:
   :undoc-members:
   :show-inheritance:
