#! /usr/bin/env python3
#
# Output writers for merged radar images
#
# Supports NetCDF, PNG, MP4, GeoTIFF, KML, and KMZ output formats.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.merged_image import MergedImage

logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================


def _draw_range_rings(ax, extent: list, ring_interval: float = 1000.0) -> None:
    """
    Draw range rings centered at origin.

    Args:
        ax: Matplotlib axes
        extent: [xmin, xmax, ymin, ymax] of the plot
        ring_interval: Distance between rings in meters (default 1000m)
    """
    from matplotlib.patches import Circle

    xmin, xmax, ymin, ymax = extent

    # Compute max range needed to cover the plot
    max_range = max(
        abs(xmin),
        abs(xmax),
        abs(ymin),
        abs(ymax),
        np.sqrt(xmin**2 + ymin**2),
        np.sqrt(xmax**2 + ymin**2),
        np.sqrt(xmin**2 + ymax**2),
        np.sqrt(xmax**2 + ymax**2),
    )

    # Draw rings at regular intervals
    n_rings = int(max_range / ring_interval) + 1
    for i in range(1, n_rings + 1):
        radius = i * ring_interval
        circle = Circle(
            (0, 0),
            radius,
            fill=False,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.5,
            linestyle="--",
        )
        ax.add_patch(circle)

        # Add range label at top of circle (if visible)
        if -radius <= xmax and radius >= xmin and radius <= ymax:
            ax.text(
                0,
                radius,
                f"{radius / 1000:.0f}km",
                ha="center",
                va="bottom",
                fontsize=7,
                color="white",
                alpha=0.7,
            )


def _compute_latlon_bounds(merged: "MergedImage") -> dict:
    """
    Compute lat/lon bounds for a merged image.

    Args:
        merged: MergedImage with UTM grid

    Returns:
        Dictionary with north, south, east, west bounds in degrees
    """
    from pyproj import CRS, Transformer

    # Create transformers
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={merged.utm_zone} +{merged.hemisphere} +datum=WGS84")
    crs_wgs84 = CRS.from_epsg(4326)
    transformer_to_ll = Transformer.from_crs(utm_crs, crs_wgs84, always_xy=True)
    transformer_to_utm = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    # Get center in UTM
    center_x, center_y = transformer_to_utm.transform(merged.center_lon, merged.center_lat)

    # Compute corners in UTM
    x_min = center_x + merged.x_edges[0]
    x_max = center_x + merged.x_edges[-1]
    y_min = center_y + merged.y_edges[0]
    y_max = center_y + merged.y_edges[-1]

    # Transform corners to lat/lon
    corners_x = [x_min, x_max, x_min, x_max]
    corners_y = [y_min, y_min, y_max, y_max]
    lons, lats = transformer_to_ll.transform(corners_x, corners_y)

    return {
        "north": max(lats),
        "south": min(lats),
        "east": max(lons),
        "west": min(lons),
    }


def _write_overlay_png(
    merged: "MergedImage",
    output_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Write a PNG image suitable for KML overlay (with transparency).

    Args:
        merged: MergedImage to write
        output_path: Output PNG file path
        cmap: Colormap name
        vmin: Minimum intensity value
        vmax: Maximum intensity value
    """
    import matplotlib.pyplot as plt

    intensity = merged.intensity.copy()

    if vmin is None:
        vmin = float(np.nanpercentile(intensity, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(intensity, 98))

    # Normalize to 0-1 range
    normalized = (intensity - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    rgba = colormap(normalized)

    # Set NaN pixels to transparent
    mask = np.isnan(intensity)
    rgba[mask, 3] = 0

    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Flip vertically for correct orientation (image origin is top-left)
    rgba_uint8 = np.flipud(rgba_uint8)

    # Write PNG
    try:
        from PIL import Image

        img = Image.fromarray(rgba_uint8, mode="RGBA")
        img.save(output_path)
    except ImportError:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgba_uint8, origin="upper")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)


# ============================================================
# NetCDF Output
# ============================================================


def write_merged_netcdf(merged: "MergedImage", output_dir: str) -> str:
    """
    Write merged image to NetCDF file with CF-1.8 conventions.

    Args:
        merged: MergedImage to write
        output_dir: Output directory

    Returns:
        Path to created file
    """
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not installed, skipping NetCDF output")
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from time range
    start_str = (
        np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = np.datetime_as_string(merged.end_time, unit="s").replace(":", "-").replace("T", "_")
    filename = f"merged_{start_str}_to_{end_str}.nc"
    filepath = output_dir / filename

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "intensity": (
                ["y", "x"],
                merged.intensity.astype(np.float32),
                {
                    "long_name": "Merged radar intensity",
                    "units": "counts",
                    "coordinates": "x y",
                },
            ),
        },
        coords={
            "x": (
                ["x"],
                merged.x_centers,
                {
                    "long_name": "Distance east from center",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                ["y"],
                merged.y_centers,
                {
                    "long_name": "Distance north from center",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "time_start": merged.start_time,
            "time_end": merged.end_time,
        },
        attrs={
            "title": "WAMOS merged radar image",
            "institution": "WAMOS TPW",
            "source": "wamos files-pipeline",
            "history": f"Created {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            # Grid metadata
            "grid_spacing_m": merged.grid_spacing,
            "utm_zone": merged.utm_zone,
            "hemisphere": merged.hemisphere,
            "center_latitude": merged.center_lat,
            "center_longitude": merged.center_lon,
            "crs": f"EPSG:326{merged.utm_zone:02d}"
            if merged.hemisphere == "north"
            else f"EPSG:327{merged.utm_zone:02d}",
            # Window metadata
            "n_frames": merged.n_frames,
            "window_index": merged.window_index,
            "mean_ship_heading_deg": merged.mean_heading,
        },
    )

    # Add optional metadata
    if merged.mean_ship_speed is not None:
        ds.attrs["mean_ship_speed_m_s"] = merged.mean_ship_speed
    if merged.mean_wind_speed is not None:
        ds.attrs["mean_wind_speed_m_s"] = merged.mean_wind_speed
    if merged.mean_wind_direction is not None:
        ds.attrs["mean_wind_direction_deg"] = merged.mean_wind_direction

    # Write with compression
    encoding = {"intensity": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(filepath, encoding=encoding)

    logger.debug("Wrote merged image to %s", filepath)
    return str(filepath)


# ============================================================
# PNG Output
# ============================================================


def write_merged_png(
    merged: "MergedImage",
    output_dir: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Write merged image to PNG file.

    Args:
        merged: MergedImage to write
        output_dir: Output directory
        cmap: Colormap name
        vmin: Minimum intensity value (auto if None)
        vmax: Maximum intensity value (auto if None)

    Returns:
        Path to created file
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    start_str = (
        np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = np.datetime_as_string(merged.end_time, unit="s").replace(":", "-").replace("T", "_")
    filename = f"merged_{start_str}_to_{end_str}.png"
    filepath = output_dir / filename

    # Auto-scale
    intensity = merged.intensity
    if vmin is None:
        vmin = float(np.nanpercentile(intensity, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(intensity, 98))

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.pcolormesh(
        merged.x_edges,
        merged.y_edges,
        intensity,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    ax.set_xlabel("Distance East (m)")
    ax.set_ylabel("Distance North (m)")
    ax.set_aspect("equal")

    fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

    # Title
    start_time = np.datetime_as_string(merged.start_time, unit="s")
    end_time = np.datetime_as_string(merged.end_time, unit="s")
    ax.set_title(
        f"Merged Image: {merged.n_frames} frames\n"
        f"{start_time} to {end_time}\n"
        f"Center: {abs(merged.center_lat):.4f}°{'N' if merged.center_lat >= 0 else 'S'}, "
        f"{abs(merged.center_lon):.4f}°{'E' if merged.center_lon >= 0 else 'W'}"
    )

    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Wrote merged image to %s", filepath)
    return str(filepath)


# ============================================================
# MP4 Movie Output
# ============================================================


def write_mp4_movie(
    merged_images: list["MergedImage"],
    output_path: str,
    fps: float = 2.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (10, 8),
    range_rings: bool = True,
) -> str:
    """
    Generate an MP4 movie from merged images.

    Args:
        merged_images: List of MergedImage objects (in time order)
        output_path: Output MP4 file path
        fps: Frames per second (default 2.0)
        cmap: Colormap name
        vmin: Minimum intensity value (auto if None)
        vmax: Maximum intensity value (auto if None)
        dpi: Output resolution
        figsize: Figure size in inches
        range_rings: Draw range rings on frames

    Returns:
        Path to created file
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    if not merged_images:
        logger.warning("No merged images for movie")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute global intensity range
    if vmin is None or vmax is None:
        all_valid = []
        for merged in merged_images:
            valid_data = merged.intensity[~np.isnan(merged.intensity)]
            if len(valid_data) > 0:
                all_valid.extend(valid_data.ravel())
        if all_valid:
            if vmin is None:
                vmin = float(np.percentile(all_valid, 2))
            if vmax is None:
                vmax = float(np.percentile(all_valid, 98))
        else:
            vmin, vmax = 0, 1

    # Find global data bounds for consistent framing
    global_bounds = {"xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf}
    for merged in merged_images:
        valid_mask = ~np.isnan(merged.intensity)
        valid_rows = np.any(valid_mask, axis=1)
        valid_cols = np.any(valid_mask, axis=0)
        if np.any(valid_rows) and np.any(valid_cols):
            row_min, row_max = np.where(valid_rows)[0][[0, -1]]
            col_min, col_max = np.where(valid_cols)[0][[0, -1]]
            global_bounds["xmin"] = min(global_bounds["xmin"], merged.x_edges[col_min])
            global_bounds["xmax"] = max(global_bounds["xmax"], merged.x_edges[col_max + 1])
            global_bounds["ymin"] = min(global_bounds["ymin"], merged.y_edges[row_min])
            global_bounds["ymax"] = max(global_bounds["ymax"], merged.y_edges[row_max + 1])

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.tight_layout()

    # Create writer
    writer = FFMpegWriter(fps=fps, metadata={"title": "WAMOS Radar Movie"})

    logger.info("Generating MP4 movie with %d frames at %.1f fps", len(merged_images), fps)

    with writer.saving(fig, str(output_path), dpi=dpi):
        for i, merged in enumerate(merged_images):
            ax.clear()

            # Find valid data bounds
            valid_mask = ~np.isnan(merged.intensity)
            valid_rows = np.any(valid_mask, axis=1)
            valid_cols = np.any(valid_mask, axis=0)

            if np.any(valid_rows) and np.any(valid_cols):
                row_min, row_max = np.where(valid_rows)[0][[0, -1]]
                col_min, col_max = np.where(valid_cols)[0][[0, -1]]
                cropped = merged.intensity[row_min : row_max + 1, col_min : col_max + 1]
                extent = [
                    merged.x_edges[col_min],
                    merged.x_edges[col_max + 1],
                    merged.y_edges[row_min],
                    merged.y_edges[row_max + 1],
                ]
            else:
                cropped = merged.intensity
                extent = [
                    merged.x_edges[0],
                    merged.x_edges[-1],
                    merged.y_edges[0],
                    merged.y_edges[-1],
                ]

            ax.imshow(
                cropped,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="lower",
                aspect="equal",
            )

            if range_rings:
                _draw_range_rings(ax, extent)

            # Use global bounds for consistent framing
            ax.set_xlim(global_bounds["xmin"], global_bounds["xmax"])
            ax.set_ylim(global_bounds["ymin"], global_bounds["ymax"])

            ax.set_xlabel("Distance East (m)")
            ax.set_ylabel("Distance North (m)")

            start_time = np.datetime_as_string(merged.start_time, unit="s")
            end_time = np.datetime_as_string(merged.end_time, unit="s")
            ax.set_title(
                f"Frame {i + 1}/{len(merged_images)}: {merged.n_frames} frames\n"
                f"{start_time} to {end_time}"
            )

            writer.grab_frame()

    plt.close(fig)
    logger.info("Wrote MP4 movie to %s", output_path)
    return str(output_path)


# ============================================================
# GeoTIFF Output
# ============================================================


def write_geotiff(
    merged: "MergedImage",
    output_dir: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Write merged image as a georeferenced GeoTIFF file.

    Args:
        merged: MergedImage to write
        output_dir: Output directory
        cmap: Colormap name for RGB conversion
        vmin: Minimum intensity value (auto if None)
        vmax: Maximum intensity value (auto if None)

    Returns:
        Path to created file
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
    except ImportError:
        logger.warning("rasterio not installed, skipping GeoTIFF output")
        return ""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    start_str = (
        np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = np.datetime_as_string(merged.end_time, unit="s").replace(":", "-").replace("T", "_")
    filename = f"merged_{start_str}_to_{end_str}.tif"
    filepath = output_dir / filename

    # Get intensity data and compute scaling
    intensity = merged.intensity.copy()
    if vmin is None:
        vmin = float(np.nanpercentile(intensity, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(intensity, 98))

    # Normalize to 0-1 range
    normalized = (intensity - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap to get RGBA
    colormap = plt.get_cmap(cmap)
    rgba = colormap(normalized)  # Shape: (h, w, 4)

    # Convert to uint8
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Handle NaN values - make them transparent
    mask = np.isnan(intensity)
    alpha = np.where(mask, 0, 255).astype(np.uint8)

    # Stack RGB + Alpha
    rgba_uint8 = np.dstack([rgb, alpha])

    # Compute bounds in UTM coordinates
    # The grid is centered, so we need to convert to absolute UTM
    from pyproj import CRS as ProjCRS
    from pyproj import Transformer

    utm_crs = ProjCRS.from_proj4(
        f"+proj=utm +zone={merged.utm_zone} +{merged.hemisphere} +datum=WGS84"
    )
    crs_wgs84 = ProjCRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    # Get center in UTM
    center_x, center_y = transformer.transform(merged.center_lon, merged.center_lat)

    # Compute absolute bounds
    x_min = center_x + merged.x_edges[0]
    x_max = center_x + merged.x_edges[-1]
    y_min = center_y + merged.y_edges[0]
    y_max = center_y + merged.y_edges[-1]

    # Create transform (note: rasterio uses top-left origin, so y is flipped)
    height, width = intensity.shape
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Determine EPSG code
    if merged.hemisphere == "north":
        epsg = 32600 + merged.utm_zone
    else:
        epsg = 32700 + merged.utm_zone

    # Write GeoTIFF
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,  # RGBA
        dtype=np.uint8,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        compress="lzw",
    ) as dst:
        # Write each band (rasterio expects bands first)
        for i in range(4):
            # Flip vertically since rasterio uses top-left origin
            dst.write(np.flipud(rgba_uint8[:, :, i]), i + 1)

        # Add metadata
        dst.update_tags(
            title="WAMOS Merged Radar Image",
            start_time=str(merged.start_time),
            end_time=str(merged.end_time),
            n_frames=str(merged.n_frames),
            center_lat=str(merged.center_lat),
            center_lon=str(merged.center_lon),
        )

    logger.debug("Wrote GeoTIFF to %s", filepath)
    return str(filepath)


# ============================================================
# KML/KMZ Output
# ============================================================


def write_kml(
    merged_images: list["MergedImage"],
    output_path: str,
    image_dir: str | None = None,
    image_format: str = "png",
) -> str:
    """
    Write a KML file with ground overlays for merged images.

    If image_dir is provided, generates PNG images for each frame.
    The KML file references these images as ground overlays with proper
    geographic positioning.

    Args:
        merged_images: List of MergedImage objects
        output_path: Output KML file path
        image_dir: Directory for overlay images (if None, uses output_path directory)
        image_format: Image format for overlays ("png" or "tiff")

    Returns:
        Path to created KML file
    """
    from xml.etree.ElementTree import Element, ElementTree, SubElement  # nosec B405

    if not merged_images:
        logger.warning("No merged images for KML")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        image_dir = output_path.parent / "images"
    else:
        image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Compute global intensity range for consistent coloring
    all_valid = []
    for merged in merged_images:
        valid_data = merged.intensity[~np.isnan(merged.intensity)]
        if len(valid_data) > 0:
            all_valid.extend(valid_data.ravel())
    if all_valid:
        vmin = float(np.percentile(all_valid, 2))
        vmax = float(np.percentile(all_valid, 98))
    else:
        vmin, vmax = 0, 1

    # Create KML structure
    kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(kml, "Document")

    # Add document name and description
    name = SubElement(document, "name")
    name.text = "WAMOS Radar Images"

    description = SubElement(document, "description")
    description.text = f"Merged radar images: {len(merged_images)} frames"

    # Create folder for overlays
    folder = SubElement(document, "Folder")
    folder_name = SubElement(folder, "name")
    folder_name.text = "Radar Overlays"

    # Generate images and add overlays
    for i, merged in enumerate(merged_images):
        # Generate image filename
        start_str = (
            np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
        )
        image_filename = f"overlay_{i:04d}_{start_str}.png"
        image_path = image_dir / image_filename

        # Write overlay image (PNG with transparency)
        _write_overlay_png(merged, image_path, vmin=vmin, vmax=vmax)

        # Compute geographic bounds
        bounds = _compute_latlon_bounds(merged)

        # Create GroundOverlay element
        overlay = SubElement(folder, "GroundOverlay")

        overlay_name = SubElement(overlay, "name")
        overlay_name.text = f"Frame {i + 1}: {start_str}"

        # Time span (for time-enabled KML viewers)
        timespan = SubElement(overlay, "TimeSpan")
        begin = SubElement(timespan, "begin")
        begin.text = np.datetime_as_string(merged.start_time, unit="s")
        end = SubElement(timespan, "end")
        end.text = np.datetime_as_string(merged.end_time, unit="s")

        # Icon (image reference)
        icon = SubElement(overlay, "Icon")
        href = SubElement(icon, "href")
        # Use relative path
        href.text = f"images/{image_filename}"

        # LatLonBox for positioning
        latlonbox = SubElement(overlay, "LatLonBox")
        north = SubElement(latlonbox, "north")
        north.text = str(bounds["north"])
        south = SubElement(latlonbox, "south")
        south.text = str(bounds["south"])
        east = SubElement(latlonbox, "east")
        east.text = str(bounds["east"])
        west = SubElement(latlonbox, "west")
        west.text = str(bounds["west"])

    # Write KML file
    tree = ElementTree(kml)
    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    logger.info("Wrote KML file to %s with %d overlays", output_path, len(merged_images))
    return str(output_path)


def write_kmz(
    merged_images: list["MergedImage"],
    output_path: str,
) -> str:
    """
    Write a KMZ file (compressed KML with embedded images).

    A KMZ file is a ZIP archive containing:
    - doc.kml: The main KML file
    - images/: Directory with overlay PNG images

    This creates a self-contained package that can be opened directly
    in Google Earth without external file dependencies.

    Args:
        merged_images: List of MergedImage objects
        output_path: Output KMZ file path

    Returns:
        Path to created KMZ file
    """
    import shutil
    import tempfile
    import zipfile

    if not merged_images:
        logger.warning("No merged images for KMZ")
        return ""

    output_path = Path(output_path)
    if not output_path.suffix.lower() == ".kmz":
        output_path = output_path.with_suffix(".kmz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for KML and images
    temp_dir = Path(tempfile.mkdtemp(prefix="wamos_kmz_"))
    try:
        kml_path = temp_dir / "doc.kml"
        image_dir = temp_dir / "images"

        # Generate KML and images using existing function
        write_kml(merged_images, str(kml_path), image_dir=str(image_dir))

        # Package into KMZ (ZIP file)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as kmz:
            # Add doc.kml at root
            kmz.write(kml_path, "doc.kml")

            # Add all images in images/ folder
            for image_file in image_dir.iterdir():
                if image_file.is_file():
                    kmz.write(image_file, f"images/{image_file.name}")

        logger.info(
            "Wrote KMZ file to %s (%d overlays, %.1f MB)",
            output_path,
            len(merged_images),
            output_path.stat().st_size / (1024 * 1024),
        )
        return str(output_path)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
