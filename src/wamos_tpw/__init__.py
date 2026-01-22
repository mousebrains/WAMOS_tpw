"""WAMOS marine radar data processing pipeline."""

__version__ = "0.2.0"

from wamos_tpw.frame import Frame
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.filenames import Filenames
from wamos_tpw.files import Files
from wamos_tpw.multi_theta import MultiTheta as Theta, MultiBearing as Bearing
from wamos_tpw.timestamp import Timestamp
from wamos_tpw.config import Config
from wamos_tpw.processed import ProcessedFrames
from wamos_tpw.dataset import WamosDataset
from wamos_tpw.exceptions import (
    WamosError,
    PolarFileError,
    ConfigError,
    ProcessingError,
    ValidationError,
)

__all__ = [
    "__version__",
    # High-level API
    "WamosDataset",
    # Core classes
    "Frame",
    "PolarFile",
    "Filenames",
    "Files",
    "Theta",
    "Bearing",
    "Timestamp",
    "Config",
    "ProcessedFrames",
    # Exceptions
    "WamosError",
    "PolarFileError",
    "ConfigError",
    "ProcessingError",
    "ValidationError",
]
