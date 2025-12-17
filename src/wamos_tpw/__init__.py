"""WAMOS marine radar data processing pipeline."""

__version__ = "0.1.0"

from wamos_tpw.frame import Frame
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.filenames import Filenames
from wamos_tpw.files import Files
from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.timestamp import Timestamp
from wamos_tpw.config import WamosConfig
from wamos_tpw.processed import ProcessedFrames

__all__ = [
    "__version__",
    "Frame",
    "PolarFile",
    "Filenames",
    "Files",
    "Theta",
    "Bearing",
    "Timestamp",
    "WamosConfig",
    "ProcessedFrames",
]
