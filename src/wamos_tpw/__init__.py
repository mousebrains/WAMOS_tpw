"""WAMOS marine radar data processing pipeline."""

__version__ = "0.2.0"

from wamos_tpw.frame import Frame
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.filenames import Filenames
from wamos_tpw.files import Files
from wamos_tpw.bearing import MultiTheta as Theta, MultiBearing as Bearing
from wamos_tpw.config import Config
from wamos_tpw.exceptions import (
    WamosError,
    PolarFileError,
    ConfigError,
    ProcessingError,
    ValidationError,
)
from wamos_tpw.pipeline import MergePipeline, create_pipeline

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
