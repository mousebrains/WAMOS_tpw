"""WAMOS marine radar data processing pipeline."""

from importlib.metadata import version as _version

__version__ = _version("wamos_tpw")

from wamos_tpw.bearing import MultiBearing as Bearing
from wamos_tpw.bearing import MultiTheta as Theta
from wamos_tpw.config import Config
from wamos_tpw.exceptions import (
    ConfigError,
    PolarFileError,
    ProcessingError,
    ValidationError,
    WamosError,
)
from wamos_tpw.filenames import Filenames
from wamos_tpw.files import Files
from wamos_tpw.frame import Frame
from wamos_tpw.pipeline import MergePipeline, create_pipeline
from wamos_tpw.polarfile import PolarFile

__all__ = [
    "__version__",
    # Core classes
    "Frame",
    "PolarFile",
    "Filenames",
    "Files",
    "Theta",
    "Bearing",
    "Config",
    # Pipeline
    "MergePipeline",
    "create_pipeline",
    # Exceptions
    "WamosError",
    "PolarFileError",
    "ConfigError",
    "ProcessingError",
    "ValidationError",
]
