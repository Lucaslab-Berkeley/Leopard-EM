r"""Standalone napari helper to choose filament Euler-box orientation constraints.

Draw a line along a filament and four corner points for its search region, then
commit that pair as a colored overlay. Repeat for each filament. The ± Euler
box is shared; each region keeps its own in-plane ``psi``.

Dependencies: napari, numpy, mrcfile, h5py (qtpy ships with napari). This
script does not import leopard_em.

The template filament axis is assumed to lie along Z. Roma ``'ZYZ'`` is
intrinsic with angles ``(phi, theta, psi)``. The drawn line sets ``psi``
(in-plane rotation of the projection); ``phi`` is searched over 360° (roll
around the tube); ``theta`` is an Euler box around 90° (side-on). The
opposite polarity along the same line is a second pole at ``psi + 180°``.

Example
-------
::

    python napari_choose_constraint.py \\
        --micrograph /path/to/micrograph.mrc \\
        --pixel-size 0.9194 \\
        --output filament_constraint.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_NAPARI_INSTALL_MESSAGE = (
    "This program needs napari, numpy, mrcfile, and h5py.\n"
    "Install with:  pip install napari mrcfile numpy h5py"
)
_ANGLE_DECIMALS = 4
REGION_COLORS = (
    "#ff6b6b",
    "#4dabf7",
    "#69db7c",
    "#ffd43b",
    "#da77f2",
    "#ff922b",
    "#22b8cf",
    "#e599f7",
)


def region_color(index: int) -> str:
    """Return a stable overlay color for committed region ``index``."""
    return REGION_COLORS[int(index) % len(REGION_COLORS)]


def last_filament_line(shapes_layer) -> tuple[float, float, float, float]:
    """Return ``(y0, x0, y1, x1)`` for the last line in a napari Shapes layer."""
    lines = []
    for data, shape_type in zip(shapes_layer.data, shapes_layer.shape_type):
        if shape_type == "line" or len(data) == 2:
            lines.append(np.asarray(data, dtype=float))

    if not lines:
        raise ValueError("Draw a line along the filament first (two-point line).")

    points = lines[-1]
    return (
        float(points[0, 0]),
        float(points[0, 1]),
        float(points[1, 0]),
        float(points[1, 1]),
    )


def last_search_corners(points_layer) -> np.ndarray:
    """Return four ``(y, x)`` corners from a napari Points layer.

    Points are ordered around their centroid so click order does not matter.
    """
    points = np.asarray(points_layer.data, dtype=float)
    if points.ndim != 2 or points.shape[0] != 4 or points.shape[1] < 2:
        n_pts = 0 if points.size == 0 else int(points.shape[0])
        raise ValueError(
            f"Place exactly four search-region corners (currently {n_pts})."
        )
    return order_polygon_vertices(points[:, :2])


def order_polygon_vertices(vertices: np.ndarray) -> np.ndarray:
    """Order vertices counter-clockwise around their centroid."""
    verts = np.asarray(vertices, dtype=float).reshape(-1, 2)
    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 0] - center[0], verts[:, 1] - center[1])
    return verts[np.argsort(angles)]


def points_in_polygon(height: int, width: int, vertices: np.ndarray) -> np.ndarray:
    """Pixel-center even-odd fill of a closed polygon, inclusive of edges."""
    verts = order_polygon_vertices(vertices)
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    inside = np.zeros((height, width), dtype=bool)
    n_verts = verts.shape[0]
    prev = n_verts - 1
    for index in range(n_verts):
        y_i, x_i = verts[index, 0], verts[index, 1]
        y_j, x_j = verts[prev, 0], verts[prev, 1]
        straddles = (y_i > yy) != (y_j > yy)
        denom = y_j - y_i
        denom = np.where(denom == 0.0, 1.0, denom)
        x_intersect = x_i + (yy - y_i) * (x_j - x_i) / denom
        inside ^= straddles & (xx < x_intersect)
        prev = index
    atol = 1e-6
    on_edge = np.zeros((height, width), dtype=bool)
    for index in range(n_verts):
        y0, x0 = verts[index]
        y1, x1 = verts[(index + 1) % n_verts]
        dy = y1 - y0
        dx = x1 - x0
        scale = max(1.0, abs(dy) + abs(dx))
        cross = (yy - y0) * dx - (xx - x0) * dy
        collinear = np.abs(cross) <= atol * scale
        in_segment = (
            (yy >= min(y0, y1) - atol)
            & (yy <= max(y0, y1) + atol)
            & (xx >= min(x0, x1) - atol)
            & (xx <= max(x0, x1) + atol)
        )
        on_edge |= collinear & in_segment
    return inside | on_edge


def estimate_n_orientations(blocks: list[dict]) -> int:
    """Rough Euler-grid size from YAML-style orientation blocks."""
    total = 0
    for block in blocks:
        psi_step = float(block["psi_step"])
        theta_step = float(block["theta_step"])
        n_psi = max(1, int(round((block["psi_max"] - block["psi_min"]) / psi_step)))
        n_theta = max(
            1,
            int(round((block["theta_max"] - block["theta_min"]) / theta_step)) + 1,
        )
        n_phi = max(
            1,
            int(round((block["phi_max"] - block["phi_min"]) / theta_step)) + 1,
        )
        total += n_psi * n_theta * n_phi
    return total


def rasterize_search_box(
    image_shape: tuple[int, int],
    corners: np.ndarray,
    n_orientations: int,
    region_id: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``eligible``, ``region_id``, ``n_orientations`` maps for one quad."""
    height, width = image_shape
    inside = points_in_polygon(height, width, corners)
    eligible = inside.astype(np.uint8)
    region_ids = np.where(inside, int(region_id), 0).astype(np.int16)
    n_orients = np.where(inside, int(n_orientations), 0).astype(np.int32)
    return eligible, region_ids, n_orients


def combine_search_boxes(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union rasterized boxes. Overlapping pixels take the later region."""
    if not parts:
        raise ValueError("Need at least one search box to combine.")
    eligible, region_id, n_orients = (array.copy() for array in parts[0])
    for next_eligible, next_region_id, next_n_orients in parts[1:]:
        if next_eligible.shape != eligible.shape:
            raise ValueError(
                "Cannot combine search boxes of shape "
                f"{next_eligible.shape} with {eligible.shape}."
            )
        inside = next_eligible > 0
        eligible = np.where(inside, next_eligible, eligible)
        region_id = np.where(inside, next_region_id, region_id)
        n_orients = np.where(inside, next_n_orients, n_orients)
    return (
        eligible.astype(np.uint8),
        region_id.astype(np.int16),
        n_orients.astype(np.int32),
    )


def write_spatial_constraint_hdf5(
    path: str,
    eligible: np.ndarray,
    region_id: np.ndarray,
    n_orientations: np.ndarray,
    region_payloads: list[dict],
    pixel_size_angstrom: float | None = None,
) -> None:
    """Write the constraint HDF5 sidecar (standalone; no leopard_em import)."""
    import h5py

    if not region_payloads:
        raise ValueError("Need at least one region to write a constraint HDF5.")

    compression = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(path, "w") as handle:
        handle.attrs["leopard_em_version"] = "uninstalled"
        handle.attrs["coordinate_frame"] = "pos_xy_img"
        handle.attrs["micrograph_shape"] = np.array(eligible.shape, dtype=np.int32)
        if pixel_size_angstrom is not None:
            handle.attrs["pixel_size_angstrom"] = float(pixel_size_angstrom)

        maps_group = handle.create_group("maps")
        maps_group.create_dataset("eligible", data=eligible, **compression)
        maps_group.create_dataset("region_id", data=region_id, **compression)
        maps_group.create_dataset("n_orientations", data=n_orientations, **compression)

        regions_group = handle.create_group("regions")
        for index, payload in enumerate(region_payloads, start=1):
            region = regions_group.create_group(f"{index:04d}")
            _write_hdf5_region(region, payload, region_id=index)


def _write_hdf5_region(region, payload: dict, region_id: int) -> None:
    region.attrs["cone_half_angle_deg"] = payload["cone_half_angle_deg"]
    region.attrs["theta_center_deg"] = payload["theta_center_deg"]
    region.attrs["phi_min"] = payload["phi_min"]
    region.attrs["phi_max"] = payload["phi_max"]
    region.attrs["psi_step"] = payload["psi_step"]
    region.attrs["theta_step"] = payload["theta_step"]
    region.attrs["base_grid_method"] = payload["base_grid_method"]
    region.attrs["region_id"] = int(region_id)
    region.attrs["filament_angle_deg"] = payload["filament_angle_deg"]
    line = payload["line"]
    region.create_dataset(
        "line",
        data=np.array(
            [line["y0"], line["x0"], line["y1"], line["x1"]], dtype=np.float64
        ),
    )
    box = payload["spatial_box"]["corners"]
    region.create_dataset("box", data=np.asarray(box, dtype=np.float64).reshape(-1, 2))
    cfg_root = region.create_group("orientation_configs")
    for index, block in enumerate(
        payload["orientation_search_config"]["orientation_configs"]
    ):
        cfg_grp = cfg_root.create_group(str(index))
        for key, value in block.items():
            if value is None:
                continue
            cfg_grp.attrs[key] = value


def filament_psi_from_image_line(y0: float, x0: float, y1: float, x1: float) -> float:
    """Return in-plane ``psi`` in degrees from an image-space line (y down, x right)."""
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        raise ValueError("Filament line has zero length.")
    return math.degrees(math.atan2(-dy, dx)) % 360.0


def periodic_euler_box_intervals(
    center_deg: float, half_width_deg: float
) -> list[tuple[float, float]]:
    """Split ``center ± half_width`` into ``[0, 360]`` intervals."""
    if half_width_deg >= 180.0:
        return [(0.0, 360.0)]

    lo = center_deg - half_width_deg
    hi = center_deg + half_width_deg
    lo = lo % 360.0
    hi = hi % 360.0
    lo_r = round(lo, _ANGLE_DECIMALS)
    hi_r = round(hi, _ANGLE_DECIMALS)

    if math.isclose(lo, hi, abs_tol=1e-12):
        return [(lo_r, lo_r)]
    if hi == 0.0:
        return [(lo_r, 360.0)]
    if lo < hi:
        return [(lo_r, hi_r)]

    intervals = [(lo_r, 360.0)]
    if hi > 0.0:
        intervals.append((0.0, hi_r))
    return intervals


def load_mrc_image(path: str) -> tuple[np.ndarray, float | None]:
    """Load a 2D MRC as float32 (y, x) and pixel size in Angstroms if present."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        image = np.asarray(mrc.data, dtype=np.float32).squeeze()
        pixel_size = _mrc_pixel_size_angstrom(mrc)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D MRC micrograph, got shape {image.shape}.")
    return image, pixel_size


def _mrc_pixel_size_angstrom(mrc) -> float | None:
    """Pixel size from voxel_size or cella/nx, in Angstroms."""
    voxel = float(getattr(mrc.voxel_size, "x", 0.0) or 0.0)
    if voxel > 1e-6:
        return voxel
    nx = int(mrc.header.nx)
    cella = float(mrc.header.cella.x)
    if nx > 0 and cella > 1e-6:
        return cella / nx
    return None


def fourier_lowpass(
    image: np.ndarray,
    pixel_size_angstrom: float,
    resolution_angstrom: float,
) -> np.ndarray:
    """Soft Fourier low-pass at ``resolution_angstrom``."""
    ny, nx = image.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.rfftfreq(nx)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    freq = np.sqrt(kx * kx + ky * ky) / pixel_size_angstrom
    cutoff = 1.0 / resolution_angstrom
    falloff = max(0.05 / pixel_size_angstrom, cutoff * 0.1)
    weight = np.ones_like(freq, dtype=np.float32)
    taper = (freq > cutoff) & (freq < cutoff + falloff)
    t = (freq[taper] - cutoff) / falloff
    weight[taper] = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    weight[freq >= cutoff + falloff] = 0.0
    filtered = np.fft.irfft2(
        np.fft.rfft2(image) * weight, s=image.shape, axes=(-2, -1)
    )
    return np.asarray(filtered, dtype=np.float32)


def imagej_contrast_limits(
    image: np.ndarray, saturated_percent: float = 0.35
) -> tuple[float, float]:
    """Display range matching ImageJ Enhance Contrast (no Normalize).

    ImageJ splits ``saturated_percent`` across both histogram tails, so the
    default 0.35% uses the 0.175th and 99.825th percentiles.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if saturated_percent <= 0.0:
        return float(np.min(finite)), float(np.max(finite))
    tail = min(saturated_percent / 2.0, 50.0)
    low, high = np.percentile(finite, (tail, 100.0 - tail))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return float(np.min(finite)), float(np.max(finite))
    return float(low), float(high)


def raise_constraint_layers(viewer) -> None:
    """Keep committed overlays, scratch outline, corners, then filament on top."""
    names = [layer.name for layer in viewer.layers]
    n_layers = len(viewer.layers)

    def _move_to_top(name: str) -> None:
        nonlocal names, n_layers
        if name not in names:
            return
        idx = names.index(name)
        if idx != n_layers - 1:
            viewer.layers.move(idx, n_layers)
            names = [layer.name for layer in viewer.layers]
            n_layers = len(viewer.layers)

    _move_to_top("committed_regions")
    _move_to_top("search_region")
    _move_to_top("search_corners")
    _move_to_top("filament")


def raise_filament_layer(viewer) -> None:
    """Keep constraint shapes above the micrograph."""
    raise_constraint_layers(viewer)


def add_or_replace_micrograph(
    viewer,
    micrograph_path: str,
    lowpass_angstrom: float | None = 20.0,
    saturated_percent: float = 0.35,
    pixel_size_angstrom: float | None = None,
) -> str:
    """Load an MRC, optionally low-pass and set ImageJ-style contrast."""
    image, header_pixel_size = load_mrc_image(micrograph_path)
    pixel_size = pixel_size_angstrom or header_pixel_size
    notes: list[str] = []
    if lowpass_angstrom is not None and lowpass_angstrom > 0.0:
        if pixel_size is None:
            notes.append(
                "Lowpass skipped: no pixel size. Pass --pixel-size (Å/px) "
                "or set Pixel size in the panel."
            )
        else:
            image = fourier_lowpass(image, pixel_size, lowpass_angstrom)
            source = "given" if pixel_size_angstrom else "MRC header"
            notes.append(
                f"Lowpass {lowpass_angstrom:g} Å "
                f"(pixel size {pixel_size:.4g} Å/px, {source})."
            )
    low, high = imagej_contrast_limits(image, saturated_percent)
    notes.append(f"Enhance contrast: {saturated_percent:g}% saturated pixels (ImageJ).")

    existing = [layer for layer in viewer.layers if layer.name == "micrograph"]
    for layer in existing:
        viewer.layers.remove(layer)

    viewer.add_image(
        image,
        name="micrograph",
        colormap="gray",
        contrast_limits=(low, high),
    )
    micrograph_layer = viewer.layers["micrograph"]
    micrograph_layer.contrast_limits = (low, high)
    raise_filament_layer(viewer)
    return " ".join(notes)


def _yaml_scalar(value: object) -> str:
    """Format a YAML scalar without PyYAML."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{round(float(value), _ANGLE_DECIMALS):.4f}".rstrip("0").rstrip(".")
    return json.dumps(str(value))


def build_constraint_payload(
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    cone_half_angle_deg: float,
    psi_step: float,
    theta_step: float,
    theta_center_deg: float,
    micrograph_path: str | None,
    spatial_box: np.ndarray | None = None,
    spatial_constraint_path: str | None = None,
) -> dict:
    """Build the sidecar dict consumed by FilamentConstraint.from_yaml."""
    psi = round(filament_psi_from_image_line(y0, x0, y1, x1), _ANGLE_DECIMALS)
    theta_min = round(max(0.0, theta_center_deg - cone_half_angle_deg), _ANGLE_DECIMALS)
    theta_max = round(
        min(180.0, theta_center_deg + cone_half_angle_deg), _ANGLE_DECIMALS
    )
    pole_1 = psi % 360.0
    pole_2 = (psi + 180.0) % 360.0

    if cone_half_angle_deg >= 180.0:
        interval_groups = [[(0.0, 360.0)]]
    else:
        interval_groups = [
            periodic_euler_box_intervals(pole, cone_half_angle_deg)
            for pole in (pole_1, pole_2)
        ]

    blocks: list[dict] = []
    for intervals in interval_groups:
        for psi_min, psi_max in intervals:
            blocks.append(
                {
                    "psi_step": psi_step,
                    "theta_step": theta_step,
                    "symmetry": None,
                    "phi_min": 0.0,
                    "phi_max": 360.0,
                    "theta_min": theta_min,
                    "theta_max": theta_max,
                    "psi_min": psi_min,
                    "psi_max": psi_max,
                    "base_grid_method": "uniform",
                }
            )

    payload: dict = {
        "filament_angle_deg": psi,
        "cone_half_angle_deg": cone_half_angle_deg,
        "theta_center_deg": theta_center_deg,
        "phi_min": 0.0,
        "phi_max": 360.0,
        "psi_step": psi_step,
        "theta_step": theta_step,
        "base_grid_method": "uniform",
        "line": {"y0": y0, "x0": x0, "y1": y1, "x1": x1},
        "orientation_search_config": {"orientation_configs": blocks},
    }
    if micrograph_path:
        payload["micrograph_path"] = micrograph_path
    if spatial_box is not None:
        corners = np.asarray(spatial_box, dtype=float).reshape(-1, 2)
        payload["spatial_box"] = {"corners": [[float(y), float(x)] for y, x in corners]}
    if spatial_constraint_path:
        payload["spatial_constraint_path"] = spatial_constraint_path
    return payload


def preview_text(payload: dict) -> str:
    """Human-readable summary of the Euler boxes."""
    psi = payload["filament_angle_deg"]
    cone = payload["cone_half_angle_deg"]
    pole_1 = psi % 360.0
    pole_2 = (psi + 180.0) % 360.0
    blocks = payload["orientation_search_config"]["orientation_configs"]
    theta_min = blocks[0]["theta_min"]
    theta_max = blocks[0]["theta_max"]
    lines = [
        f"psi pole 1: {pole_1:.2f}°",
        f"psi pole 2: {pole_2:.2f}°",
        f"Euler box ±{cone:g}°",
        f"  theta: [{theta_min:.2f}, {theta_max:.2f}]",
        "  phi:   [0.00, 360.00]",
    ]
    for i, pole in enumerate((pole_1, pole_2), start=1):
        intervals = periodic_euler_box_intervals(pole, cone)
        interval_str = ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in intervals)
        lines.append(f"  psi pole {i}: {interval_str}")
    return "\n".join(lines)


def preview_regions_text(region_payloads: list[dict]) -> str:
    """Summary of one or more committed regions."""
    if not region_payloads:
        return "No regions yet."
    if len(region_payloads) == 1:
        return preview_text(region_payloads[0])
    first = region_payloads[0]
    cone = first["cone_half_angle_deg"]
    blocks = first["orientation_search_config"]["orientation_configs"]
    lines = [
        f"{len(region_payloads)} regions",
        f"Euler box ±{cone:g}°",
        f"  theta: [{blocks[0]['theta_min']:.2f}, {blocks[0]['theta_max']:.2f}]",
        "  phi:   [0.00, 360.00]",
    ]
    for index, payload in enumerate(region_payloads, start=1):
        psi = payload["filament_angle_deg"] % 360.0
        lines.append(f"region {index}: psi {psi:.2f}° / {(psi + 180.0) % 360.0:.2f}°")
    return "\n".join(lines)


def region_from_line_and_corners(
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    corners: np.ndarray,
) -> dict:
    """Snapshot a scratch line + quad as a committed region dict."""
    return {
        "y0": float(y0),
        "x0": float(x0),
        "y1": float(y1),
        "x1": float(x1),
        "corners": np.asarray(corners, dtype=float).reshape(-1, 2),
        "filament_angle_deg": round(
            filament_psi_from_image_line(y0, x0, y1, x1), _ANGLE_DECIMALS
        ),
    }


def sidecar_payload_from_regions(
    region_payloads: list[dict],
    spatial_constraint_path: str | None = None,
) -> dict:
    """Build a YAML sidecar dict from one or more region payloads."""
    if not region_payloads:
        raise ValueError("Need at least one region.")
    payloads = [dict(payload) for payload in region_payloads]
    if spatial_constraint_path:
        for payload in payloads:
            payload["spatial_constraint_path"] = spatial_constraint_path
    if len(payloads) == 1:
        return payloads[0]
    first = payloads[0]
    sidecar: dict = {
        "cone_half_angle_deg": first["cone_half_angle_deg"],
        "theta_center_deg": first["theta_center_deg"],
        "phi_min": first["phi_min"],
        "phi_max": first["phi_max"],
        "psi_step": first["psi_step"],
        "theta_step": first["theta_step"],
        "base_grid_method": first["base_grid_method"],
        "regions": [
            {
                "filament_angle_deg": payload["filament_angle_deg"],
                "line": payload["line"],
                "spatial_box": payload["spatial_box"],
            }
            for payload in payloads
        ],
    }
    if "micrograph_path" in first:
        sidecar["micrograph_path"] = first["micrograph_path"]
    if spatial_constraint_path:
        sidecar["spatial_constraint_path"] = spatial_constraint_path
    if first.get("stats_from_valid_orientations_defocus"):
        sidecar["stats_from_valid_orientations_defocus"] = True
    return sidecar


def dump_constraint_yaml(payload: dict) -> str:
    """Serialize the sidecar as YAML text."""
    if payload.get("regions"):
        return _dump_multi_region_yaml(payload)
    return _dump_single_region_yaml(payload)


def _dump_shared_header(payload: dict) -> list[str]:
    lines = [
        f"cone_half_angle_deg: {_yaml_scalar(payload['cone_half_angle_deg'])}",
        f"theta_center_deg: {_yaml_scalar(payload['theta_center_deg'])}",
        f"phi_min: {_yaml_scalar(payload['phi_min'])}",
        f"phi_max: {_yaml_scalar(payload['phi_max'])}",
        f"psi_step: {_yaml_scalar(payload['psi_step'])}",
        f"theta_step: {_yaml_scalar(payload['theta_step'])}",
        f"base_grid_method: {_yaml_scalar(payload['base_grid_method'])}",
    ]
    if "micrograph_path" in payload:
        lines.append(f"micrograph_path: {_yaml_scalar(payload['micrograph_path'])}")
    if "spatial_constraint_path" in payload:
        lines.append(
            "spatial_constraint_path: "
            f"{_yaml_scalar(payload['spatial_constraint_path'])}"
        )
    if payload.get("stats_from_valid_orientations_defocus"):
        lines.append("stats_from_valid_orientations_defocus: true")
    return lines


def _dump_line_block(line: dict, indent: int) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}line:"]
    for key in ("y0", "x0", "y1", "x1"):
        lines.append(f"{pad}  {key}: {_yaml_scalar(line[key])}")
    return lines


def _dump_spatial_box_block(box: dict, indent: int) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}spatial_box:"]
    if "corners" in box:
        lines.append(f"{pad}  corners:")
        for y_pt, x_pt in box["corners"]:
            lines.append(f"{pad}    - [{_yaml_scalar(y_pt)}, {_yaml_scalar(x_pt)}]")
        return lines
    for key in ("y0", "x0", "y1", "x1"):
        lines.append(f"{pad}  {key}: {_yaml_scalar(box[key])}")
    return lines


def _dump_orientation_search_config(payload: dict, indent: int = 0) -> list[str]:
    pad = " " * indent
    item_pad = " " * (indent + 4)
    cont_pad = " " * (indent + 6)
    lines = [
        f"{pad}orientation_search_config:",
        f"{pad}  orientation_configs:",
    ]
    keys = (
        "psi_step",
        "theta_step",
        "symmetry",
        "phi_min",
        "phi_max",
        "theta_min",
        "theta_max",
        "psi_min",
        "psi_max",
        "base_grid_method",
    )
    for block in payload["orientation_search_config"]["orientation_configs"]:
        first = True
        for key in keys:
            prefix = f"{item_pad}- " if first else cont_pad
            lines.append(f"{prefix}{key}: {_yaml_scalar(block[key])}")
            first = False
    return lines


def _dump_single_region_yaml(payload: dict) -> str:
    lines = [
        f"filament_angle_deg: {_yaml_scalar(payload['filament_angle_deg'])}",
        *_dump_shared_header(payload),
        *_dump_line_block(payload["line"], 0),
    ]
    if "spatial_box" in payload:
        lines.extend(_dump_spatial_box_block(payload["spatial_box"], 0))
    lines.extend(_dump_orientation_search_config(payload))
    return "\n".join(lines) + "\n"


def _dump_multi_region_yaml(payload: dict) -> str:
    lines = _dump_shared_header(payload)
    lines.append("regions:")
    for region in payload["regions"]:
        lines.append(
            f"  - filament_angle_deg: {_yaml_scalar(region['filament_angle_deg'])}"
        )
        lines.extend(_dump_line_block(region["line"], 4))
        if "spatial_box" in region:
            lines.extend(_dump_spatial_box_block(region["spatial_box"], 4))
    return "\n".join(lines) + "\n"


def build_viewer(
    micrograph_path: str | None,
    output_path: str,
    pixel_size_angstrom: float | None = None,
) -> None:
    """Create the napari viewer, constraint layers, and widgets."""
    try:
        import napari
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(_NAPARI_INSTALL_MESSAGE) from exc

    viewer = napari.Viewer(title="Choose filament constraint")
    committed = viewer.add_shapes(
        name="committed_regions",
        ndim=2,
        edge_color="tomato",
        edge_width=3,
        face_color="transparent",
    )
    committed.editable = False
    search_region = viewer.add_shapes(
        name="search_region",
        ndim=2,
        edge_color="yellow",
        edge_width=4,
        face_color="transparent",
    )
    search_region.editable = False
    search_corners = viewer.add_points(
        name="search_corners",
        ndim=2,
        size=18,
        face_color="yellow",
    )
    search_corners.mode = "add"
    shapes = viewer.add_shapes(
        name="filament",
        ndim=2,
        edge_color="cyan",
        edge_width=8,
        face_color="transparent",
    )
    shapes.mode = "add_line"

    committed_regions: list[dict] = []

    panel = QWidget()
    layout = QVBoxLayout(panel)

    layout.addWidget(
        QLabel(
            "1. Load a micrograph\n"
            "2. Draw a line along the filament\n"
            "3. Click four corners for allowed particle centers\n"
            "4. Click Add region (repeat for each filament)\n"
            "5. Set the shared ± Euler-box range\n"
            "6. Export YAML + HDF5"
        )
    )

    def _path_row(label: str, default: str, save: bool) -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit(default)
        browse = QPushButton("Browse")

        def on_browse() -> None:
            if save:
                path, _ = QFileDialog.getSaveFileName(
                    panel, label, edit.text(), "YAML (*.yaml *.yml)"
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    panel, label, edit.text(), "MRC (*.mrc)"
                )
            if path:
                edit.setText(path)

        browse.clicked.connect(on_browse)
        row.addWidget(edit)
        row.addWidget(browse)
        layout.addWidget(QLabel(label))
        layout.addLayout(row)
        return edit

    micrograph_edit = _path_row("Micrograph MRC", micrograph_path or "", save=False)

    display_form = QFormLayout()
    lowpass_check = QCheckBox("Lowpass filter")
    lowpass_check.setChecked(True)
    lowpass_spin = QDoubleSpinBox()
    lowpass_spin.setRange(1.0, 200.0)
    lowpass_spin.setSingleStep(1.0)
    lowpass_spin.setValue(20.0)
    lowpass_spin.setSuffix(" Å")
    pixel_size_spin = QDoubleSpinBox()
    pixel_size_spin.setRange(0.0, 20.0)
    pixel_size_spin.setDecimals(4)
    pixel_size_spin.setSingleStep(0.01)
    pixel_size_spin.setSpecialValueText("from MRC")
    pixel_size_spin.setValue(float(pixel_size_angstrom) if pixel_size_angstrom else 0.0)
    pixel_size_spin.setSuffix(" Å/px")
    saturated_spin = QDoubleSpinBox()
    saturated_spin.setRange(0.0, 50.0)
    saturated_spin.setDecimals(2)
    saturated_spin.setSingleStep(0.05)
    saturated_spin.setValue(0.35)
    saturated_spin.setSuffix(" %")
    display_form.addRow(lowpass_check, lowpass_spin)
    display_form.addRow("Pixel size", pixel_size_spin)
    display_form.addRow("Saturated pixels (ImageJ)", saturated_spin)
    layout.addLayout(display_form)
    lowpass_check.toggled.connect(lowpass_spin.setEnabled)

    load_button = QPushButton("Load micrograph")
    layout.addWidget(load_button)

    form = QFormLayout()
    cone_spin = QDoubleSpinBox()
    cone_spin.setRange(0.1, 90.0)
    cone_spin.setSingleStep(0.5)
    cone_spin.setValue(10.0)
    cone_spin.setSuffix(" °")
    psi_step_spin = QDoubleSpinBox()
    psi_step_spin.setRange(0.1, 30.0)
    psi_step_spin.setSingleStep(0.1)
    psi_step_spin.setValue(1.5)
    psi_step_spin.setSuffix(" °")
    theta_step_spin = QDoubleSpinBox()
    theta_step_spin.setRange(0.1, 30.0)
    theta_step_spin.setSingleStep(0.1)
    theta_step_spin.setValue(2.5)
    theta_step_spin.setSuffix(" °")
    theta_center_spin = QDoubleSpinBox()
    theta_center_spin.setRange(0.0, 180.0)
    theta_center_spin.setSingleStep(1.0)
    theta_center_spin.setValue(90.0)
    theta_center_spin.setSuffix(" °")
    form.addRow("± Euler box", cone_spin)
    form.addRow("psi step", psi_step_spin)
    form.addRow("theta step", theta_step_spin)
    form.addRow("theta center", theta_center_spin)
    layout.addLayout(form)

    layout.addWidget(QLabel("Regions"))
    region_list = QListWidget()
    region_list.setMinimumHeight(90)
    layout.addWidget(region_list)
    region_buttons = QHBoxLayout()
    add_region_button = QPushButton("Add region")
    delete_region_button = QPushButton("Delete")
    focus_region_button = QPushButton("Focus")
    region_buttons.addWidget(add_region_button)
    region_buttons.addWidget(delete_region_button)
    region_buttons.addWidget(focus_region_button)
    layout.addLayout(region_buttons)

    output_edit = _path_row("Output YAML", output_path, save=True)
    export_button = QPushButton("Export constraint YAML")
    layout.addWidget(export_button)

    help_text = (
        "Draw a line along the filament and four corners for allowed centers.\n"
        "Click Add region to keep that pair, then draw the next filament.\n"
        "The line sets psi (in-plane angle of the tube in the image).\n"
        "The four points define a quadrilateral in pos_x_img / pos_y_img\n"
        "(particle center); click order does not matter.\n"
        "phi is searched 0-360° around the tube.\n"
        "A second pole at psi+180° covers the flip.\n"
        "A complete scratch pair is included on export even without Add."
    )
    preview = QTextEdit()
    preview.setReadOnly(True)
    preview.setPlainText(help_text)
    layout.addWidget(preview)

    def current_micrograph_path() -> str | None:
        text = micrograph_edit.text().strip()
        if text and Path(text).is_file():
            return text
        return None

    def payload_kwargs() -> dict:
        return {
            "cone_half_angle_deg": float(cone_spin.value()),
            "psi_step": float(psi_step_spin.value()),
            "theta_step": float(theta_step_spin.value()),
            "theta_center_deg": float(theta_center_spin.value()),
            "micrograph_path": current_micrograph_path(),
        }

    def payload_from_region(region: dict) -> dict:
        return build_constraint_payload(
            y0=region["y0"],
            x0=region["x0"],
            y1=region["y1"],
            x1=region["x1"],
            spatial_box=region["corners"],
            **payload_kwargs(),
        )

    def sync_search_outline() -> None:
        """Draw the quadrilateral connecting the four corner points."""
        try:
            corners = last_search_corners(search_corners)
        except ValueError:
            search_region.data = []
            return
        search_region.data = [corners]
        search_region.shape_type = ["polygon"]

    def sync_committed_overlay() -> None:
        data = []
        shape_types = []
        colors = []
        widths = []
        focused_row = region_list.currentRow()
        for index, region in enumerate(committed_regions):
            color = region_color(index)
            width = 6 if index == focused_row else 3
            data.append(np.asarray(region["corners"], dtype=float))
            shape_types.append("polygon")
            colors.append(color)
            widths.append(width)
            data.append(
                np.array(
                    [
                        [region["y0"], region["x0"]],
                        [region["y1"], region["x1"]],
                    ],
                    dtype=float,
                )
            )
            shape_types.append("line")
            colors.append(color)
            widths.append(width + 2)
        if not data:
            committed.data = []
            return
        committed.data = data
        committed.shape_type = shape_types
        committed.edge_color = colors
        committed.edge_width = widths
        committed.editable = False

    def refresh_region_list() -> None:
        current = region_list.currentRow()
        region_list.blockSignals(True)
        region_list.clear()
        for index, region in enumerate(committed_regions):
            region_list.addItem(f"{index + 1}  psi {region['filament_angle_deg']:.1f}°")
        if committed_regions:
            region_list.setCurrentRow(min(max(current, 0), len(committed_regions) - 1))
        region_list.blockSignals(False)
        sync_committed_overlay()

    def current_payload() -> dict:
        y0, x0, y1, x1 = last_filament_line(shapes)
        try:
            box = last_search_corners(search_corners)
        except ValueError:
            box = None
        return build_constraint_payload(
            y0=y0,
            x0=x0,
            y1=y1,
            x1=x1,
            spatial_box=box,
            **payload_kwargs(),
        )

    def refresh_preview() -> None:
        sync_search_outline()
        committed_payloads = [
            payload_from_region(region) for region in committed_regions
        ]
        scratch_payload = None
        scratch_error = None
        try:
            scratch_payload = current_payload()
        except (ValueError, IndexError) as exc:
            scratch_error = str(exc)
        chunks: list[str] = []
        if committed_payloads:
            chunks.append(preview_regions_text(committed_payloads))
        if scratch_payload is not None and "spatial_box" in scratch_payload:
            scratch_text = preview_text(scratch_payload)
            if committed_payloads:
                chunks.append(f"Scratch (click Add region to keep):\n{scratch_text}")
            else:
                chunks.append(scratch_text)
        elif scratch_error and not committed_payloads:
            chunks.append(scratch_error)
        preview.setPlainText("\n\n".join(chunks) if chunks else help_text)

    def clear_scratch() -> None:
        shapes.data = []
        search_corners.data = []
        search_region.data = []
        shapes.mode = "add_line"
        search_corners.mode = "add"

    def on_add_region() -> None:
        try:
            y0, x0, y1, x1 = last_filament_line(shapes)
            corners = last_search_corners(search_corners)
        except ValueError as exc:
            preview.setPlainText(str(exc))
            return
        committed_regions.append(region_from_line_and_corners(y0, x0, y1, x1, corners))
        clear_scratch()
        refresh_region_list()
        region_list.setCurrentRow(len(committed_regions) - 1)
        refresh_preview()

    def on_delete_region() -> None:
        row = region_list.currentRow()
        if row < 0 or row >= len(committed_regions):
            preview.setPlainText("Select a region to delete.")
            return
        committed_regions.pop(row)
        refresh_region_list()
        refresh_preview()

    def on_focus_region() -> None:
        row = region_list.currentRow()
        if row < 0 or row >= len(committed_regions):
            preview.setPlainText("Select a region to focus.")
            return
        corners = np.asarray(committed_regions[row]["corners"], dtype=float)
        cy, cx = corners.mean(axis=0)
        center = list(viewer.camera.center)
        center[-2] = float(cy)
        center[-1] = float(cx)
        viewer.camera.center = tuple(center)
        sync_committed_overlay()

    def load_display_kwargs() -> tuple[float | None, float, float | None]:
        lowpass = float(lowpass_spin.value()) if lowpass_check.isChecked() else None
        px = float(pixel_size_spin.value())
        pixel_size = px if px > 0.0 else None
        return lowpass, float(saturated_spin.value()), pixel_size

    def on_load() -> None:
        path = current_micrograph_path()
        if path is None:
            preview.setPlainText("Choose an existing micrograph MRC path first.")
            return
        try:
            lowpass, saturated, pixel_size = load_display_kwargs()
            notes = add_or_replace_micrograph(
                viewer,
                path,
                lowpass_angstrom=lowpass,
                saturated_percent=saturated,
                pixel_size_angstrom=pixel_size,
            )
        except (ValueError, OSError) as exc:
            preview.setPlainText(str(exc))
            return
        shapes.mode = "add_line"
        search_corners.mode = "add"
        raise_constraint_layers(viewer)
        refresh_preview()
        current = preview.toPlainText()
        if current and current != help_text:
            preview.setPlainText(f"{notes}\n\n{current}")
        else:
            preview.setPlainText(notes)

    def regions_for_export() -> list[dict]:
        payloads = [payload_from_region(region) for region in committed_regions]
        try:
            scratch = current_payload()
        except (ValueError, IndexError):
            scratch = None
        if scratch is not None and "spatial_box" in scratch:
            payloads.append(scratch)
        if not payloads:
            raise ValueError(
                "Add at least one region: draw a line and four corners, "
                "then Add region (or export the current scratch pair)."
            )
        return payloads

    def on_export() -> None:
        out = output_edit.text().strip()
        if not out:
            preview.setPlainText("Choose an output YAML path first.")
            return
        try:
            region_payloads = regions_for_export()
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            h5_path = out_path.with_suffix(".h5")

            micrograph_layer = next(
                (layer for layer in viewer.layers if layer.name == "micrograph"),
                None,
            )
            if micrograph_layer is None:
                raise ValueError("Load a micrograph before exporting.")
            image_shape = tuple(int(v) for v in micrograph_layer.data.shape[-2:])
            parts = []
            for index, payload in enumerate(region_payloads, start=1):
                n_orient = estimate_n_orientations(
                    payload["orientation_search_config"]["orientation_configs"]
                )
                corners = np.asarray(payload["spatial_box"]["corners"], dtype=float)
                parts.append(
                    rasterize_search_box(
                        image_shape,
                        corners,
                        n_orientations=n_orient,
                        region_id=index,
                    )
                )
            eligible, region_id, n_orients = combine_search_boxes(parts)
            px = float(pixel_size_spin.value())
            pixel_size = px if px > 0.0 else None
            write_spatial_constraint_hdf5(
                str(h5_path),
                eligible,
                region_id,
                n_orients,
                region_payloads,
                pixel_size_angstrom=pixel_size,
            )
            sidecar = sidecar_payload_from_regions(
                region_payloads, spatial_constraint_path=str(h5_path)
            )
            out_path.write_text(dump_constraint_yaml(sidecar), encoding="utf-8")
        except (ValueError, IndexError, OSError) as exc:
            preview.setPlainText(str(exc))
            return
        preview.setPlainText(
            f"Wrote {out_path}\nWrote {h5_path}\n\n"
            f"{preview_regions_text(region_payloads)}"
        )
        print(f"Wrote filament constraint to {out_path}")
        print(f"Wrote spatial constraint maps to {h5_path}")

    load_button.clicked.connect(on_load)
    add_region_button.clicked.connect(on_add_region)
    delete_region_button.clicked.connect(on_delete_region)
    focus_region_button.clicked.connect(on_focus_region)
    region_list.currentRowChanged.connect(lambda _: sync_committed_overlay())
    export_button.clicked.connect(on_export)
    cone_spin.valueChanged.connect(lambda _: refresh_preview())
    psi_step_spin.valueChanged.connect(lambda _: refresh_preview())
    theta_step_spin.valueChanged.connect(lambda _: refresh_preview())
    theta_center_spin.valueChanged.connect(lambda _: refresh_preview())
    shapes.events.data.connect(lambda _: refresh_preview())
    search_corners.events.data.connect(lambda _: refresh_preview())

    viewer.window.add_dock_widget(panel, name="Filament constraint", area="right")

    # Load after the Qt window is up. Adding the image before napari.run()
    # lets the canvas reset contrast and layer order on first paint.
    if micrograph_path:
        QTimer.singleShot(0, on_load)

    napari.run()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the napari helper."""
    parser = argparse.ArgumentParser(
        description=(
            "Draw filament lines and four search-region corners in napari, "
            "commit each pair as a region, then export an Euler-box constraint YAML."
        )
    )
    parser.add_argument(
        "--micrograph",
        type=str,
        default=None,
        help="Optional path to a 2D MRC micrograph to load on startup.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Pixel size in Angstroms. Used for the lowpass if the MRC header has none."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="filament_constraint.yaml",
        help="Default output path for the exported constraint YAML.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the napari filament-constraint helper."""
    args = parse_args(argv)
    if args.micrograph is not None and not Path(args.micrograph).is_file():
        print(f"Micrograph not found: {args.micrograph}", file=sys.stderr)
        raise SystemExit(1)
    build_viewer(
        micrograph_path=args.micrograph,
        output_path=args.output,
        pixel_size_angstrom=args.pixel_size,
    )


if __name__ == "__main__":
    main()
