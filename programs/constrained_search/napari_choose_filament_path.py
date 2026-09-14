"""Draw CURVED filament paths in napari and export a per-pixel angular constraint.

``napari_choose_constraint.py`` handles a straight filament: one drawn line fixes a
single ``psi``, and a quadrilateral of four clicked corners fixes where it applies. A
bent microtubule has no single ``psi``, so that sidecar cannot describe it.

This tool draws **polylines** instead, in an editable napari Shapes layer (vertices
can be added, moved and deleted after the fact), fits a smooth spline through each,
and gives each path its own **width**. Export writes the continuous form:

    maps/psi_center   the in-plane angle of the local spline TANGENT, per pixel
    maps/eligible     pixels within width/2 of that path's centreline
    maps/region_id    which path owns the pixel -- nearest centreline wins

Leopard-EM already consumes this: ``_orient_ok_from_psi_center`` in
``leopard_em/backend/utils.py`` expands the Euler box per pixel about ``psi_center`` on
the GPU. What did not exist was any way to *produce* the field for a filament, since
``rasterize_spatial_maps`` cannot, and the only writer was the membrane exporter.

⚠️ **The membrane exporter stores the local NORMAL**; a filament needs the local
**TANGENT**. Both use ``atan2(-dy, dx)``, so the formula looks identical and copying it
without changing the vector would put every allowed ``psi`` at 90 degrees to the tube.

Only ``psi`` is per-pixel. ``theta`` and ``phi`` stay a single global Euler box shared
by every region, so this describes a tube bending **in the image plane**. A tube
bending out of plane would need a per-pixel ``theta``, which Leopard-EM does not have.

Standalone by design -- it does not import ``leopard_em``, so it runs in a napari-only
environment.

Usage:
    python napari_choose_filament_path.py --micrograph frame.mrc --output sidecar.yaml
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_NAPARI_INSTALL_MESSAGE = (
    "napari and its Qt bindings are required:\n"
    "    pip install 'napari[all]'"
)
_ANGLE_DECIMALS = 4
_PATH_LAYER = "filament paths"
_PREVIEW_LAYER = "constraint preview"
REGION_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
)

DEFAULT_WIDTH_PX = 320.0          # a microtubule is ~250 A = 272 px at 0.92 A/px
DEFAULT_CONE_HALF_ANGLE_DEG = 10.0
DEFAULT_THETA_CENTER_DEG = 90.0
DEFAULT_PSI_STEP = 1.5
DEFAULT_THETA_STEP = 2.5


def region_color(index: int) -> str:
    """Stable colour for a region, cycling through the palette."""
    return REGION_COLORS[index % len(REGION_COLORS)]


# --------------------------------------------------------------------------- geometry


def spline_through_points(
    points_yx: np.ndarray, samples_per_pixel: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a drawn polyline as a smooth curve, with its unit tangent.

    Returns ``(points, tangents)``, both ``(n, 2)`` in ``(y, x)``, sampled at roughly
    one point per pixel of arc length. Two clicked points give a straight segment; three
    or more are fitted with an interpolating spline of the highest order the point count
    allows.
    """
    points = np.asarray(points_yx, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("A filament path needs at least two points.")

    # Approximate length from the polyline, to choose how densely to sample.
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0.0:
        raise ValueError("A filament path must have non-zero length.")
    n_samples = max(int(total_length * samples_per_pixel), 2)

    if len(points) == 2:
        t = np.linspace(0.0, 1.0, n_samples)[:, None]
        curve = points[0][None, :] + t * (points[1] - points[0])[None, :]
        tangent = np.repeat((points[1] - points[0])[None, :], n_samples, axis=0)
    else:
        from scipy.interpolate import splev, splprep

        order = min(3, len(points) - 1)
        (tck, _u) = splprep([points[:, 0], points[:, 1]], s=0.0, k=order)
        u = np.linspace(0.0, 1.0, n_samples)
        y, x = splev(u, tck)
        dy, dx = splev(u, tck, der=1)
        curve = np.column_stack([y, x])
        tangent = np.column_stack([dy, dx])

    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return curve, tangent / norms


def psi_from_tangent(tangent_yx: np.ndarray) -> np.ndarray:
    """In-plane ``psi`` in degrees from a TANGENT vector, image convention (y down).

    Matches ``filament_psi_from_image_line``: ``atan2(-dy, dx)``. Note this is the
    tangent, *not* the normal the membrane exporter uses.
    """
    tangent = np.asarray(tangent_yx, dtype=np.float64).reshape(-1, 2)
    return np.degrees(np.arctan2(-tangent[:, 0], tangent[:, 1])) % 360.0


def paint_path_maps(
    shape: tuple[int, int],
    paths: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize drawn paths into eligibility, region id, psi_center and distance.

    Each path contributes its centreline; a Euclidean distance transform then assigns
    every pixel to its **nearest** centreline, which is how overlapping paths are
    resolved (the same rule the membrane exporter uses, and the opposite of the
    later-region-wins rule in the straight-filament rasterizer).

    Eligibility is then per-path, because each path carries its own width: a pixel is in
    play when its distance to the *owning* centreline is within that path's half-width.
    """
    from scipy.ndimage import distance_transform_edt

    height, width = shape
    centreline_region = np.zeros(shape, dtype=np.int16)
    centreline_psi = np.zeros(shape, dtype=np.float32)

    for index, path in enumerate(paths, start=1):
        curve, tangent = spline_through_points(path["points"])
        psi = psi_from_tangent(tangent)
        rows = np.clip(np.rint(curve[:, 0]).astype(int), 0, height - 1)
        cols = np.clip(np.rint(curve[:, 1]).astype(int), 0, width - 1)
        centreline_region[rows, cols] = index
        centreline_psi[rows, cols] = psi.astype(np.float32)

    if not np.any(centreline_region):
        raise ValueError("No path pixels landed inside the micrograph.")

    distance, (nearest_y, nearest_x) = distance_transform_edt(
        centreline_region == 0, return_indices=True
    )
    region_id = centreline_region[nearest_y, nearest_x].astype(np.int16)
    psi_center = centreline_psi[nearest_y, nearest_x].astype(np.float32)

    half_widths = np.array(
        [0.0] + [float(path["width_px"]) / 2.0 for path in paths], dtype=np.float64
    )
    eligible = (distance <= half_widths[region_id]).astype(np.uint8)
    region_id = np.where(eligible > 0, region_id, 0).astype(np.int16)
    psi_center = np.where(eligible > 0, psi_center, 0.0).astype(np.float32)
    return eligible, region_id, psi_center, distance.astype(np.float32)


def estimate_n_orientations(
    eligible: np.ndarray,
    cone_half_angle_deg: float,
    psi_step: float,
    theta_step: float,
    theta_center_deg: float = DEFAULT_THETA_CENTER_DEG,
) -> np.ndarray:
    """A placeholder orientation count for the HDF5.

    Leopard-EM recomputes this at load time against the run's own Euler grid --
    ``SpatialConstraintMaps.expand_orientation_against_grid`` takes the ``psi_center``
    branch and calls ``count_orientations_from_psi_center``. The value written here is
    only for display and for tools that read the sidecar without a search config.
    """
    n_psi = max(int(round(2.0 * cone_half_angle_deg / max(psi_step, 1e-6))), 1)
    theta_span = min(180.0, theta_center_deg + cone_half_angle_deg) - max(
        0.0, theta_center_deg - cone_half_angle_deg
    )
    n_theta = max(int(round(theta_span / max(theta_step, 1e-6))), 1)
    # Two poles, unless polarity narrows it later.
    per_pixel = 2 * n_psi * n_theta
    return (np.asarray(eligible, dtype=np.int32) > 0).astype(np.int32) * per_pixel


# ----------------------------------------------------------------------------- export


def _yaml_scalar(value: object) -> str:
    """Render a scalar for the hand-written YAML sidecar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.{_ANGLE_DECIMALS}f}".rstrip("0").rstrip(".")
    return str(value)


def build_sidecar_payload(
    paths: list[dict],
    cone_half_angle_deg: float,
    theta_center_deg: float,
    psi_step: float,
    theta_step: float,
    micrograph_path: str | None,
    spatial_constraint_path: str,
) -> dict:
    """Sidecar dict for ``FilamentConstraint.from_yaml``, continuous form.

    There is deliberately no ``filament_angle_deg``: a curved filament has no single
    one, and ``FilamentConstraint._require_angle_or_regions`` permits its absence when
    ``spatial_constraint_path`` is set. The per-pixel field in the HDF5 carries the
    angle instead.
    """
    return {
        "cone_half_angle_deg": cone_half_angle_deg,
        "theta_center_deg": theta_center_deg,
        "phi_min": 0.0,
        "phi_max": 360.0,
        "psi_step": psi_step,
        "theta_step": theta_step,
        "base_grid_method": "uniform",
        "micrograph_path": micrograph_path,
        "spatial_constraint_path": spatial_constraint_path,
        "regions": [
            {"region_id": index, "polarity": path.get("polarity", "both")}
            for index, path in enumerate(paths, start=1)
        ],
    }


def dump_sidecar_yaml(payload: dict) -> str:
    """Serialize the sidecar as YAML text, matching the membrane sidecar layout."""
    lines = [
        f"cone_half_angle_deg: {_yaml_scalar(payload['cone_half_angle_deg'])}",
        f"theta_center_deg: {_yaml_scalar(payload['theta_center_deg'])}",
        f"phi_min: {_yaml_scalar(payload['phi_min'])}",
        f"phi_max: {_yaml_scalar(payload['phi_max'])}",
        f"psi_step: {_yaml_scalar(payload['psi_step'])}",
        f"theta_step: {_yaml_scalar(payload['theta_step'])}",
        f"base_grid_method: {_yaml_scalar(payload['base_grid_method'])}",
    ]
    if payload.get("micrograph_path"):
        lines.append(f"micrograph_path: {_yaml_scalar(payload['micrograph_path'])}")
    lines.append(
        f"spatial_constraint_path: {_yaml_scalar(payload['spatial_constraint_path'])}"
    )
    lines.append("regions:")
    for region in payload["regions"]:
        lines.append(f"  - region_id: {region['region_id']}")
        lines.append(f"    polarity: {region['polarity']}")
    return "\n".join(lines) + "\n"


def write_constraint_hdf5(
    path: str,
    eligible: np.ndarray,
    region_id: np.ndarray,
    psi_center: np.ndarray,
    n_orientations: np.ndarray,
    signed_distance: np.ndarray | None,
    paths: list[dict],
    cone_half_angle_deg: float,
    theta_center_deg: float,
    psi_step: float,
    theta_step: float,
    pixel_size_angstrom: float | None = None,
) -> None:
    """Write the continuous constraint sidecar (standalone; no leopard_em import).

    Layout matches ``leopard_em.pydantic_models.config.spatial_constraint``: a ``maps``
    group of ``(H, W)`` arrays plus a ``regions`` table of Euler-box attributes. The
    pole mask is deliberately NOT written -- Leopard-EM derives it at load time from
    ``region_id`` crossed with each region's ``polarity``, so flipping polarity is a
    YAML edit with no re-rasterization.
    """
    import h5py

    if not paths:
        raise ValueError("Need at least one path to write a constraint HDF5.")

    compression = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(path, "w") as handle:
        handle.attrs["leopard_em_version"] = "uninstalled"
        handle.attrs["coordinate_frame"] = "pos_xy_img"
        handle.attrs["micrograph_shape"] = np.array(eligible.shape, dtype=np.int32)
        if pixel_size_angstrom is not None:
            handle.attrs["pixel_size_angstrom"] = float(pixel_size_angstrom)

        maps_group = handle.create_group("maps")
        maps_group.create_dataset(
            "eligible", data=np.asarray(eligible, dtype=np.uint8), **compression
        )
        maps_group.create_dataset(
            "region_id", data=np.asarray(region_id, dtype=np.int16), **compression
        )
        maps_group.create_dataset(
            "n_orientations",
            data=np.asarray(n_orientations, dtype=np.int32),
            **compression,
        )
        maps_group.create_dataset(
            "psi_center", data=np.asarray(psi_center, dtype=np.float32), **compression
        )
        if signed_distance is not None:
            maps_group.create_dataset(
                "signed_distance",
                data=np.asarray(signed_distance, dtype=np.float32),
                **compression,
            )

        regions_group = handle.create_group("regions")
        for index, drawn in enumerate(paths, start=1):
            region = regions_group.create_group(f"{index:04d}")
            region.attrs["region_id"] = int(index)
            region.attrs["polarity"] = drawn.get("polarity", "both")
            region.attrs["cone_half_angle_deg"] = float(cone_half_angle_deg)
            region.attrs["theta_center_deg"] = float(theta_center_deg)
            region.attrs["phi_min"] = 0.0
            region.attrs["phi_max"] = 360.0
            region.attrs["psi_step"] = float(psi_step)
            region.attrs["theta_step"] = float(theta_step)
            region.attrs["base_grid_method"] = "uniform"
            region.attrs["width_px"] = float(drawn["width_px"])
            region.create_dataset(
                "path", data=np.asarray(drawn["points"], dtype=np.float64)
            )


def save_paths_json(path: str, micrograph_path: str, paths: list[dict]) -> None:
    """Save the drawn paths themselves, separately from the derived constraint.

    The paths are the thing worth keeping: they can be reloaded and edited, and the
    constraint re-derived at a different width without redrawing. Writing them needs no
    HDF5 support, so it works in a bare napari environment.
    """
    import json

    payload = {
        "micrograph_path": micrograph_path,
        "paths": [
            {
                "points": np.asarray(drawn["points"], dtype=float).tolist(),
                "width_px": float(drawn["width_px"]),
                "polarity": drawn.get("polarity", "both"),
            }
            for drawn in paths
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_paths_json(path: str) -> tuple[str | None, list[dict]]:
    """Read paths written by :func:`save_paths_json`."""
    import json

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    paths = [
        {
            "points": np.asarray(drawn["points"], dtype=float).reshape(-1, 2),
            "width_px": float(drawn.get("width_px", DEFAULT_WIDTH_PX)),
            "polarity": drawn.get("polarity", "both"),
        }
        for drawn in payload.get("paths", [])
    ]
    return payload.get("micrograph_path"), paths


def preview_text(paths: list[dict], eligible: np.ndarray, pixel_size: float | None,
                 cone_half_angle_deg: float) -> str:
    """Human-readable summary of what would be exported."""
    lines = [
        f"{len(paths)} path(s), psi per pixel from the local tangent, "
        f"±{cone_half_angle_deg:g}° cone",
    ]
    for index, path in enumerate(paths, start=1):
        points = np.asarray(path["points"], dtype=np.float64)
        try:
            curve, tangent = spline_through_points(points)
        except ValueError as error:
            lines.append(f"  path {index}: {error}")
            continue
        length_px = float(
            np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()
        )
        psi = psi_from_tangent(tangent)
        turn = float(
            np.abs(np.degrees(np.angle(np.exp(1j * np.radians(psi[-1] - psi[0])))))
        )
        length = (
            f"{length_px * pixel_size / 10.0:.0f} nm"
            if pixel_size
            else f"{length_px:.0f} px"
        )
        lines.append(
            f"  path {index}: {len(points)} points, {length}, "
            f"width {path['width_px']:.0f} px, "
            f"psi {psi[0]:.1f}° → {psi[-1]:.1f}° (turns {turn:.1f}°), "
            f"polarity {path.get('polarity', 'both')}"
        )
    covered = int(np.count_nonzero(eligible))
    lines.append(
        f"  eligible pixels: {covered:,} of {eligible.size:,} "
        f"({100.0 * covered / eligible.size:.2f}%)"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- viewer


def load_mrc_image(path: str) -> tuple[np.ndarray, float | None]:
    """Load a 2-D MRC as float32 ``(y, x)`` and its pixel size if the header has one."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        image = np.asarray(mrc.data, dtype=np.float32).squeeze()
        voxel = float(getattr(mrc.voxel_size, "x", 0.0) or 0.0)
        pixel_size = voxel if voxel > 1e-6 else None
        if pixel_size is None:
            nx = int(mrc.header.nx)
            cella = float(mrc.header.cella.x)
            pixel_size = cella / nx if nx > 0 and cella > 1e-6 else None
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D MRC micrograph, got shape {image.shape}.")
    return image, pixel_size


def display_image(
    image: np.ndarray, pixel_size: float | None, lowpass_angstrom: float | None
) -> np.ndarray:
    """Low-pass for display only, so faint tubes are visible while drawing."""
    if not lowpass_angstrom or not pixel_size:
        return image
    ny, nx = image.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.rfftfreq(nx)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    freq = np.sqrt(kx * kx + ky * ky) / pixel_size
    cutoff = 1.0 / lowpass_angstrom
    falloff = max(cutoff * 0.1, 1e-6)
    weight = np.ones_like(freq, dtype=np.float32)
    taper = (freq > cutoff) & (freq < cutoff + falloff)
    weight[taper] = 0.5 * (
        1.0 + np.cos(np.pi * (freq[taper] - cutoff) / falloff)
    ).astype(np.float32)
    weight[freq >= cutoff + falloff] = 0.0
    return np.asarray(
        np.fft.irfft2(np.fft.rfft2(image) * weight, s=image.shape), dtype=np.float32
    )


def contrast_limits(image: np.ndarray, saturated_percent: float = 0.35) -> tuple:
    """Display range matching ImageJ's Enhance Contrast, split across both tails."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    tail = min(saturated_percent / 2.0, 50.0)
    low, high = np.percentile(finite, (tail, 100.0 - tail))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return float(np.min(finite)), float(np.max(finite))
    return float(low), float(high)


def paths_from_layer(shapes_layer, default_width: float) -> list[dict]:
    """Read the drawn polylines and their per-shape widths out of the Shapes layer."""
    paths: list[dict] = []
    widths = shapes_layer.features.get("width_px") if len(shapes_layer.data) else None
    polarities = (
        shapes_layer.features.get("polarity") if len(shapes_layer.data) else None
    )
    for index, data in enumerate(shapes_layer.data):
        points = np.asarray(data, dtype=np.float64).reshape(-1, 2)
        if len(points) < 2:
            continue
        width = (
            float(widths.iloc[index])
            if widths is not None and index < len(widths)
            else default_width
        )
        polarity = (
            str(polarities.iloc[index])
            if polarities is not None and index < len(polarities)
            else "both"
        )
        paths.append({"points": points, "width_px": width, "polarity": polarity})
    return paths


# pylint: disable=too-many-locals,too-many-statements
def build_viewer(
    micrograph_path: str,
    output_path: str,
    width_px: float,
    cone_half_angle_deg: float,
    theta_center_deg: float,
    psi_step: float,
    theta_step: float,
    lowpass_angstrom: float | None,
    pixel_size_angstrom: float | None,
    initial_paths: list[dict] | None = None,
):
    """Open napari with an editable path layer and an export dock."""
    try:
        import napari
        from qtpy.QtWidgets import (
            QComboBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(_NAPARI_INSTALL_MESSAGE) from error

    image, header_pixel_size = load_mrc_image(micrograph_path)
    pixel_size = pixel_size_angstrom or header_pixel_size
    shown = display_image(image, pixel_size, lowpass_angstrom)

    name = pathlib.Path(micrograph_path).name
    viewer = napari.Viewer(title=f"Filament paths — {name}")
    viewer.add_image(
        shown,
        name="micrograph",
        colormap="gray",
        contrast_limits=contrast_limits(shown),
    )

    shapes = viewer.add_shapes(
        name=_PATH_LAYER,
        shape_type="path",
        edge_color="#ff7f0e",
        edge_width=6,
        features={"width_px": np.empty(0, dtype=float),
                  "polarity": np.empty(0, dtype=object)},
        feature_defaults={"width_px": width_px, "polarity": "both"},
    )
    shapes.mode = "add_path"

    for drawn in initial_paths or []:
        shapes.add_paths(np.asarray(drawn["points"], dtype=float))
        index = len(shapes.data) - 1
        shapes.features.loc[index, "width_px"] = float(drawn["width_px"])
        shapes.features.loc[index, "polarity"] = drawn.get("polarity", "both")

    panel = QWidget()
    layout = QVBoxLayout(panel)

    layout.addWidget(QLabel(
        "Draw a path along each filament (click to add vertices, Esc to finish).\n"
        "Select a path and use the box below to set ITS width."
    ))

    width_row = QHBoxLayout()
    width_row.addWidget(QLabel("width (px)"))
    width_box = QDoubleSpinBox()
    width_box.setRange(4.0, 4000.0)
    width_box.setDecimals(0)
    width_box.setSingleStep(10.0)
    width_box.setValue(width_px)
    width_row.addWidget(width_box)
    apply_width = QPushButton("apply to selected")
    width_row.addWidget(apply_width)
    layout.addLayout(width_row)

    polarity_row = QHBoxLayout()
    polarity_row.addWidget(QLabel("polarity"))
    polarity_box = QComboBox()
    polarity_box.addItems(["both", "positive", "negative"])
    polarity_row.addWidget(polarity_box)
    apply_polarity = QPushButton("apply to selected")
    polarity_row.addWidget(apply_polarity)
    layout.addLayout(polarity_row)

    summary = QTextEdit()
    summary.setReadOnly(True)
    summary.setMinimumHeight(180)
    layout.addWidget(summary)

    preview_button = QPushButton("Preview")
    export_button = QPushButton("Export sidecar + HDF5")
    layout.addWidget(preview_button)
    layout.addWidget(export_button)

    def current_paths() -> list[dict]:
        return paths_from_layer(shapes, width_box.value())

    def rasterize(paths: list[dict]):
        return paint_path_maps(image.shape, paths)

    def on_apply_width() -> None:
        selected = list(shapes.selected_data)
        if not selected:
            summary.setPlainText("Select one or more paths first.")
            return
        for index in selected:
            shapes.features.loc[index, "width_px"] = width_box.value()
        shapes.feature_defaults["width_px"] = width_box.value()
        on_preview()

    def on_apply_polarity() -> None:
        selected = list(shapes.selected_data)
        if not selected:
            summary.setPlainText("Select one or more paths first.")
            return
        for index in selected:
            shapes.features.loc[index, "polarity"] = polarity_box.currentText()
        shapes.feature_defaults["polarity"] = polarity_box.currentText()
        on_preview()

    def on_preview() -> None:
        paths = current_paths()
        if not paths:
            summary.setPlainText("No paths drawn yet.")
            return
        try:
            eligible, region_id, psi_center, _distance = rasterize(paths)
        except ValueError as error:
            summary.setPlainText(f"Cannot rasterize: {error}")
            return
        summary.setPlainText(
            preview_text(paths, eligible, pixel_size, cone_half_angle_deg)
        )
        # Show psi_center where eligible, so a wrong tangent is obvious at a glance.
        preview = np.where(eligible > 0, psi_center, np.nan)
        if _PREVIEW_LAYER in viewer.layers:
            viewer.layers[_PREVIEW_LAYER].data = preview
        else:
            viewer.add_image(
                preview, name=_PREVIEW_LAYER, colormap="hsv", opacity=0.55,
                contrast_limits=(0.0, 360.0),
            )
        viewer.layers.selection.active = shapes

    def on_export() -> None:
        paths = current_paths()
        if not paths:
            summary.setPlainText("No paths drawn yet.")
            return
        try:
            eligible, region_id, psi_center, distance = rasterize(paths)
        except ValueError as error:
            summary.setPlainText(f"Cannot rasterize: {error}")
            return

        yaml_path = pathlib.Path(output_path).expanduser().resolve()
        hdf5_path = yaml_path.with_suffix(".h5")
        json_path = yaml_path.with_suffix(".paths.json")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # The drawn paths always get saved, even where HDF5 support is missing.
        save_paths_json(
            str(json_path), str(pathlib.Path(micrograph_path).resolve()), paths
        )
        written = [str(json_path)]

        try:
            n_orientations = estimate_n_orientations(
                eligible, cone_half_angle_deg, psi_step, theta_step, theta_center_deg
            )
            write_constraint_hdf5(
                str(hdf5_path), eligible, region_id, psi_center, n_orientations,
                signed_distance=distance, paths=paths,
                cone_half_angle_deg=cone_half_angle_deg,
                theta_center_deg=theta_center_deg,
                psi_step=psi_step, theta_step=theta_step,
                pixel_size_angstrom=pixel_size,
            )
            payload = build_sidecar_payload(
                paths, cone_half_angle_deg, theta_center_deg, psi_step, theta_step,
                micrograph_path=str(pathlib.Path(micrograph_path).resolve()),
                spatial_constraint_path=str(hdf5_path),
            )
            yaml_path.write_text(dump_sidecar_yaml(payload), encoding="utf-8")
            written += [str(yaml_path), str(hdf5_path)]
            note = ""
        except ImportError:
            note = (
                "\n\nh5py is not available here, so only the paths were saved.\n"
                "Build the constraint in an environment that has it:\n"
                f"    python make_filament_constraint.py {json_path} "
                f"-o {yaml_path}"
            )

        summary.setPlainText(
            preview_text(paths, eligible, pixel_size, cone_half_angle_deg)
            + "\n\n" + "\n".join(f"wrote {p}" for p in written) + note
        )

    apply_width.clicked.connect(on_apply_width)
    apply_polarity.clicked.connect(on_apply_polarity)
    preview_button.clicked.connect(on_preview)
    export_button.clicked.connect(on_export)

    viewer.window.add_dock_widget(panel, area="right", name="filament paths")
    return viewer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micrograph", required=True)
    parser.add_argument("--output", required=True,
                        help="sidecar YAML path; the HDF5 is written alongside it")
    parser.add_argument("--width-px", type=float, default=DEFAULT_WIDTH_PX,
                        help="default path width in pixels (per-path in the GUI)")
    parser.add_argument("--cone-half-angle-deg", type=float,
                        default=DEFAULT_CONE_HALF_ANGLE_DEG)
    parser.add_argument("--theta-center-deg", type=float,
                        default=DEFAULT_THETA_CENTER_DEG)
    parser.add_argument("--psi-step", type=float, default=DEFAULT_PSI_STEP)
    parser.add_argument("--theta-step", type=float, default=DEFAULT_THETA_STEP)
    parser.add_argument("--lowpass-angstrom", type=float, default=30.0,
                        help="display-only low-pass; 0 disables")
    parser.add_argument("--pixel-size-angstrom", type=float, default=None,
                        help="override the MRC header (cropped files carry none)")
    parser.add_argument("--paths", default=None,
                        help="reload paths written by a previous session "
                             "(<output>.paths.json) to keep editing them")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Open the viewer and block until it closes."""
    args = parse_args(argv)
    try:
        import napari
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(_NAPARI_INSTALL_MESSAGE) from error

    build_viewer(
        micrograph_path=args.micrograph,
        output_path=args.output,
        width_px=args.width_px,
        cone_half_angle_deg=args.cone_half_angle_deg,
        theta_center_deg=args.theta_center_deg,
        psi_step=args.psi_step,
        theta_step=args.theta_step,
        lowpass_angstrom=args.lowpass_angstrom or None,
        pixel_size_angstrom=args.pixel_size_angstrom,
        initial_paths=load_paths_json(args.paths)[1] if args.paths else None,
    )
    napari.run()


if __name__ == "__main__":
    sys.exit(main())
