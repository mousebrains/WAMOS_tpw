#! /usr/bin/env python3
#
# Task handlers for parallel tile-level current extraction
#
# Contains the worker function that runs CurrentExtractor on a single tile
# sub-cube, reading the cube data from shared memory.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _do_extract_tile(task):
    """Extract surface current from a single tile sub-cube.

    The full cube intensity is in shared memory. This handler copies
    the tile slice, builds a minimal FrameCube, runs CurrentExtractor,
    and returns the result dict.
    """
    from multiprocessing import shared_memory

    from wamos_tpw.config import Config
    from wamos_tpw.current import CurrentExtractor, FrameCube
    from wamos_tpw.priority_executor import Result

    (
        cube_shm_name,
        cube_shape,
        cube_dtype,
        x_start,
        x_end,
        y_start,
        y_end,
        ix,
        iy,
        cube_id,
        dt,
        grid_spacing,
        x_centers,
        y_centers,
        center_lat,
        center_lon,
        config_dict,
    ) = task.data

    result_base = {
        "cube_id": cube_id,
        "ix": ix,
        "iy": iy,
    }

    try:
        # Attach to shared memory and copy tile slice
        shm = shared_memory.SharedMemory(name=cube_shm_name)
        cube_array = np.ndarray(cube_shape, dtype=cube_dtype, buffer=shm.buf)
        tile_data = cube_array[:, y_start:y_end, x_start:x_end].copy()
        shm.close()

        # Build minimal FrameCube for the tile
        tile_x = x_centers[x_start:x_end]
        tile_y = y_centers[y_start:y_end]
        n_t = tile_data.shape[0]
        timestamps = np.arange(n_t).astype("timedelta64[ms]") * int(dt * 1000) + np.datetime64(
            "2022-01-01"
        )

        tile_cube = FrameCube(
            intensity=tile_data,
            timestamps=timestamps,
            dt=dt,
            x_centers=tile_x,
            y_centers=tile_y,
            grid_spacing=grid_spacing,
            center_lat=center_lat,
            center_lon=center_lon,
        )

        # Reconstruct config
        config = Config()
        if config_dict:
            config._config = config_dict

        # Run extraction
        extractor = CurrentExtractor(tile_cube, config=config)
        est = extractor.estimate

        result_base.update(
            {
                "ux": est.ux,
                "uy": est.uy,
                "speed": est.speed,
                "direction": est.direction,
                "snr": est.snr,
                "depth": est.depth,
                "center_x": est.center_x,
                "center_y": est.center_y,
                "peak_ratio": est.peak_ratio,
                "fom": est.fom,
                "ux_err": est.ux_err,
                "uy_err": est.uy_err,
                "n_ls_points": est.n_ls_points,
                "ls_rms": est.ls_rms,
            }
        )
    except Exception as e:
        logger.debug("Tile extraction failed for (%d, %d): %s", ix, iy, e)
        result_base["error"] = str(e)

    return Result(
        task_type="extract_tile",
        task_id=task.task_id,
        data=result_base,
    )


CURRENT_TASK_HANDLERS = {
    "extract_tile": _do_extract_tile,
}
