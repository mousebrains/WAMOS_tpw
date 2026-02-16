"""Shared CF-1.13 compliant NetCDF writer for instrument data."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

__all__ = ["write_cf_netcdf"]

logger = logging.getLogger(__name__)


def write_cf_netcdf(
    filepath: Path,
    time: np.ndarray,
    variables: dict[str, tuple[np.ndarray, dict]],
    global_attrs: dict,
    encoding: dict | None = None,
) -> Path:
    """Write instrument data to a CF-1.13 compliant NetCDF file.

    Args:
        filepath: Output file path.
        time: Array of datetime64[ns] timestamps.
        variables: Mapping of variable name to (data_array, attributes_dict).
            Attributes should include at minimum ``units`` and ``long_name``.
        global_attrs: Additional global attributes (merged with CF defaults).
        encoding: Optional per-variable encoding overrides.

    Returns:
        The path to the written file.
    """
    import xarray as xr

    # Build time coordinate
    time_attrs = {
        "standard_name": "time",
        "axis": "T",
    }

    coords = {"time": ("time", time, time_attrs)}

    data_vars = {}
    default_encoding: dict[str, dict[str, object]] = {}

    for name, (data, attrs) in variables.items():
        data_vars[name] = ("time", data, attrs)
        # Default encoding: zlib compression, NaN fill for floats
        var_enc = {"zlib": True, "complevel": 4}
        if np.issubdtype(data.dtype, np.floating):
            var_enc["_FillValue"] = np.nan
        else:
            var_enc["_FillValue"] = None
        default_encoding[name] = var_enc

    # Time encoding: float64 seconds since epoch, no fill value
    default_encoding["time"] = {
        "units": "seconds since 1970-01-01T00:00:00Z",
        "calendar": "standard",
        "dtype": "float64",
        "_FillValue": None,
    }

    # Merge user encoding overrides
    if encoding:
        for key, val in encoding.items():
            if key in default_encoding:
                default_encoding[key].update(val)
            else:
                default_encoding[key] = val

    # CF-1.13 global attributes
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    cf_attrs = {
        "Conventions": "CF-1.13",
        "institution": "Scripps Institution of Oceanography",
        "history": f"Created {now} by wamos instruments",
        "references": "",
        "platform": "R/V Roger Revelle",
        "platform_vocabulary": "https://vocab.nerc.ac.uk/collection/C17/current/",
        "cruise": "RR2203 ARCTERX",
    }
    cf_attrs.update(global_attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=cf_attrs)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(filepath, encoding=default_encoding)
    logger.info("Wrote %s (%d records)", filepath, len(time))
    return filepath
