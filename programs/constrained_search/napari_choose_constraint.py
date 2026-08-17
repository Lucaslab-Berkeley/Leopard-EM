r"""Standalone napari helper to choose a filament Euler-box orientation constraint.

Draw a line along a filament in a micrograph, set the ± Euler-box range, and
export a YAML sidecar for ``run_constrained_match_template.py``.

Dependencies: napari, numpy, mrcfile, h5py (qtpy ships with napari). This
script does not import leopard_em.

The template filament axis is assumed to lie along Z. Roma ``'ZYZ'`` is
intrinsic with angles ``(phi, theta, psi)``. The drawn line sets ``phi``
(azimuth of the tube in the image); ``psi`` is searched over 360° (roll
around the tube); ``theta`` is an Euler box around 90° (side-on). The
opposite polarity along the same line is a second pole at ``phi + 180°``.

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


def last_search_box(shapes_layer) -> tuple[float, float, float, float]:
    """Return ``(y0, x0, y1, x1)`` bounds of the last rectangle in a Shapes layer."""
    rects = []
    for data, shape_type in zip(shapes_layer.data, shapes_layer.shape_type):
        arr = np.asarray(data, dtype=float)
        if shape_type in ("rectangle", "polygon") or arr.shape[0] >= 4:
            rects.append(arr)
    if not rects:
        raise ValueError("Draw a search-region rectangle first.")
    points = rects[-1]
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


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
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    n_orientations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``eligible``, ``region_id``, ``n_orientations`` maps for one box."""
    height, width = image_shape
    ymin, ymax = min(y0, y1), max(y0, y1)
    xmin, xmax = min(x0, x1), max(x0, x1)
    y_idx = np.arange(height)[:, None]
    x_idx = np.arange(width)[None, :]
    inside = (y_idx >= ymin) & (y_idx <= ymax) & (x_idx >= xmin) & (x_idx <= xmax)
    eligible = inside.astype(np.uint8)
    region_id = np.where(inside, 1, 0).astype(np.int16)
    n_orients = np.where(inside, int(n_orientations), 0).astype(np.int32)
    return eligible, region_id, n_orients


def write_spatial_constraint_hdf5(
    path: str,
    eligible: np.ndarray,
    region_id: np.ndarray,
    n_orientations: np.ndarray,
    payload: dict,
    box: tuple[float, float, float, float],
    pixel_size_angstrom: float | None = None,
) -> None:
    """Write the constraint HDF5 sidecar (standalone; no leopard_em import)."""
    import h5py

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

        region = handle.create_group("regions").create_group("0001")
        region.attrs["cone_half_angle_deg"] = payload["cone_half_angle_deg"]
        region.attrs["theta_center_deg"] = payload["theta_center_deg"]
        region.attrs["psi_min"] = payload["psi_min"]
        region.attrs["psi_max"] = payload["psi_max"]
        region.attrs["psi_step"] = payload["psi_step"]
        region.attrs["theta_step"] = payload["theta_step"]
        region.attrs["base_grid_method"] = payload["base_grid_method"]
        region.attrs["region_id"] = 1
        line = payload["line"]
        region.create_dataset(
            "line",
            data=np.array(
                [line["y0"], line["x0"], line["y1"], line["x1"]], dtype=np.float64
            ),
        )
        region.create_dataset("box", data=np.array(box, dtype=np.float64))
        cfg_root = region.create_group("orientation_configs")
        for index, block in enumerate(
            payload["orientation_search_config"]["orientation_configs"]
        ):
            cfg_grp = cfg_root.create_group(str(index))
            for key, value in block.items():
                if value is None:
                    continue
                cfg_grp.attrs[key] = value


def filament_phi_from_image_line(y0: float, x0: float, y1: float, x1: float) -> float:
    """Return ``phi`` in degrees from an image-space line (y down, x right)."""
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        raise ValueError("Filament line has zero length.")
    return math.degrees(math.atan2(-dy, dx)) % 360.0


def phi_euler_box_intervals(
    center_deg: float, half_width_deg: float
) -> list[tuple[float, float]]:
    """Split ``center ± half_width`` into ``[0, 360]`` phi intervals."""
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
    filtered = np.fft.irfftn(np.fft.rfftn(image) * weight, s=image.shape)
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
    """Keep search-region then filament above the micrograph."""
    names = [layer.name for layer in viewer.layers]
    n_layers = len(viewer.layers)
    if "search_region" in names:
        idx = names.index("search_region")
        if idx != n_layers - 1:
            viewer.layers.move(idx, n_layers)
            names = [layer.name for layer in viewer.layers]
            n_layers = len(viewer.layers)
    if "filament" in names:
        idx = names.index("filament")
        if idx != n_layers - 1:
            viewer.layers.move(idx, n_layers)


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
    spatial_box: tuple[float, float, float, float] | None = None,
    spatial_constraint_path: str | None = None,
) -> dict:
    """Build the sidecar dict consumed by FilamentConstraint.from_yaml."""
    phi = round(filament_phi_from_image_line(y0, x0, y1, x1), _ANGLE_DECIMALS)
    theta_min = round(max(0.0, theta_center_deg - cone_half_angle_deg), _ANGLE_DECIMALS)
    theta_max = round(
        min(180.0, theta_center_deg + cone_half_angle_deg), _ANGLE_DECIMALS
    )
    pole_1 = phi % 360.0
    pole_2 = (phi + 180.0) % 360.0

    if cone_half_angle_deg >= 180.0:
        interval_groups = [[(0.0, 360.0)]]
    else:
        interval_groups = [
            phi_euler_box_intervals(pole, cone_half_angle_deg)
            for pole in (pole_1, pole_2)
        ]

    blocks: list[dict] = []
    for intervals in interval_groups:
        for phi_min, phi_max in intervals:
            blocks.append(
                {
                    "psi_step": psi_step,
                    "theta_step": theta_step,
                    "symmetry": None,
                    "phi_min": phi_min,
                    "phi_max": phi_max,
                    "theta_min": theta_min,
                    "theta_max": theta_max,
                    "psi_min": 0.0,
                    "psi_max": 360.0,
                    "base_grid_method": "uniform",
                }
            )

    payload: dict = {
        "filament_angle_deg": phi,
        "cone_half_angle_deg": cone_half_angle_deg,
        "theta_center_deg": theta_center_deg,
        "psi_min": 0.0,
        "psi_max": 360.0,
        "psi_step": psi_step,
        "theta_step": theta_step,
        "base_grid_method": "uniform",
        "line": {"y0": y0, "x0": x0, "y1": y1, "x1": x1},
        "orientation_search_config": {"orientation_configs": blocks},
    }
    if micrograph_path:
        payload["micrograph_path"] = micrograph_path
    if spatial_box is not None:
        payload["spatial_box"] = {
            "y0": spatial_box[0],
            "x0": spatial_box[1],
            "y1": spatial_box[2],
            "x1": spatial_box[3],
        }
    if spatial_constraint_path:
        payload["spatial_constraint_path"] = spatial_constraint_path
    return payload


def preview_text(payload: dict) -> str:
    """Human-readable summary of the Euler boxes."""
    phi = payload["filament_angle_deg"]
    cone = payload["cone_half_angle_deg"]
    pole_1 = phi % 360.0
    pole_2 = (phi + 180.0) % 360.0
    blocks = payload["orientation_search_config"]["orientation_configs"]
    theta_min = blocks[0]["theta_min"]
    theta_max = blocks[0]["theta_max"]
    lines = [
        f"phi pole 1: {pole_1:.2f}°",
        f"phi pole 2: {pole_2:.2f}°",
        f"Euler box ±{cone:g}°",
        f"  theta: [{theta_min:.2f}, {theta_max:.2f}]",
        "  psi:   [0.00, 360.00]",
    ]
    for i, pole in enumerate((pole_1, pole_2), start=1):
        intervals = phi_euler_box_intervals(pole, cone)
        interval_str = ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in intervals)
        lines.append(f"  phi pole {i}: {interval_str}")
    return "\n".join(lines)


def dump_constraint_yaml(payload: dict) -> str:
    """Serialize the sidecar as YAML text."""
    lines = [
        f"filament_angle_deg: {_yaml_scalar(payload['filament_angle_deg'])}",
        f"cone_half_angle_deg: {_yaml_scalar(payload['cone_half_angle_deg'])}",
        f"theta_center_deg: {_yaml_scalar(payload['theta_center_deg'])}",
        f"psi_min: {_yaml_scalar(payload['psi_min'])}",
        f"psi_max: {_yaml_scalar(payload['psi_max'])}",
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
    if payload.get("stats_from_valid_orientations"):
        lines.append("stats_from_valid_orientations: true")
    line = payload["line"]
    lines.append("line:")
    for key in ("y0", "x0", "y1", "x1"):
        lines.append(f"  {key}: {_yaml_scalar(line[key])}")
    if "spatial_box" in payload:
        box = payload["spatial_box"]
        lines.append("spatial_box:")
        for key in ("y0", "x0", "y1", "x1"):
            lines.append(f"  {key}: {_yaml_scalar(box[key])}")
    lines.append("orientation_search_config:")
    lines.append("  orientation_configs:")
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
            prefix = "    - " if first else "      "
            lines.append(f"{prefix}{key}: {_yaml_scalar(block[key])}")
            first = False
    return "\n".join(lines) + "\n"


def build_viewer(
    micrograph_path: str | None,
    output_path: str,
    pixel_size_angstrom: float | None = None,
) -> None:
    """Create the napari viewer, shapes layer, and constraint widgets."""
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
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(_NAPARI_INSTALL_MESSAGE) from exc

    viewer = napari.Viewer(title="Choose filament constraint")
    search_region = viewer.add_shapes(
        name="search_region",
        ndim=2,
        edge_color="yellow",
        edge_width=4,
        face_color="transparent",
    )
    search_region.mode = "add_rectangle"
    shapes = viewer.add_shapes(
        name="filament",
        ndim=2,
        edge_color="cyan",
        edge_width=8,
        face_color="transparent",
    )
    shapes.mode = "add_line"

    panel = QWidget()
    layout = QVBoxLayout(panel)

    layout.addWidget(
        QLabel(
            "1. Load a micrograph\n"
            "2. Draw a line along the filament\n"
            "3. Draw a rectangle for allowed particle centers\n"
            "4. Set the ± Euler-box range\n"
            "5. Export YAML + HDF5"
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
    pixel_size_spin.setValue(
        float(pixel_size_angstrom) if pixel_size_angstrom else 0.0
    )
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

    output_edit = _path_row("Output YAML", output_path, save=True)
    export_button = QPushButton("Export constraint YAML")
    layout.addWidget(export_button)

    preview = QTextEdit()
    preview.setReadOnly(True)
    preview.setPlainText(
        "Draw a line along the filament and a rectangle for allowed centers.\n"
        "The line sets phi (tube azimuth in the image).\n"
        "The box is pos_x_img / pos_y_img (particle center).\n"
        "psi is searched 0-360° around the tube.\n"
        "A second pole at phi+180° covers the flip."
    )
    layout.addWidget(preview)

    def current_micrograph_path() -> str | None:
        text = micrograph_edit.text().strip()
        if text and Path(text).is_file():
            return text
        return None

    def current_payload() -> dict:
        y0, x0, y1, x1 = last_filament_line(shapes)
        try:
            box = last_search_box(search_region)
        except ValueError:
            box = None
        return build_constraint_payload(
            y0=y0,
            x0=x0,
            y1=y1,
            x1=x1,
            cone_half_angle_deg=float(cone_spin.value()),
            psi_step=float(psi_step_spin.value()),
            theta_step=float(theta_step_spin.value()),
            theta_center_deg=float(theta_center_spin.value()),
            micrograph_path=current_micrograph_path(),
            spatial_box=box,
        )

    def refresh_preview() -> None:
        try:
            payload = current_payload()
        except (ValueError, IndexError) as exc:
            preview.setPlainText(str(exc))
            return
        preview.setPlainText(preview_text(payload))

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
        search_region.mode = "add_rectangle"
        raise_constraint_layers(viewer)
        try:
            preview.setPlainText(f"{notes}\n\n{preview_text(current_payload())}")
        except (ValueError, IndexError):
            preview.setPlainText(notes)

    def on_export() -> None:
        out = output_edit.text().strip()
        if not out:
            preview.setPlainText("Choose an output YAML path first.")
            return
        try:
            box = last_search_box(search_region)
            payload = current_payload()
            if "spatial_box" not in payload:
                raise ValueError("Draw a search-region rectangle first.")
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            h5_path = out_path.with_suffix(".h5")
            payload["spatial_constraint_path"] = str(h5_path)

            micrograph_layer = next(
                (layer for layer in viewer.layers if layer.name == "micrograph"),
                None,
            )
            if micrograph_layer is None:
                raise ValueError("Load a micrograph before exporting.")
            image_shape = tuple(int(v) for v in micrograph_layer.data.shape[-2:])
            n_orient = estimate_n_orientations(
                payload["orientation_search_config"]["orientation_configs"]
            )
            eligible, region_id, n_orients = rasterize_search_box(
                image_shape, *box, n_orientations=n_orient
            )
            px = float(pixel_size_spin.value())
            pixel_size = px if px > 0.0 else None
            write_spatial_constraint_hdf5(
                str(h5_path),
                eligible,
                region_id,
                n_orients,
                payload,
                box,
                pixel_size_angstrom=pixel_size,
            )
            out_path.write_text(dump_constraint_yaml(payload), encoding="utf-8")
        except (ValueError, IndexError, OSError) as exc:
            preview.setPlainText(str(exc))
            return
        preview.setPlainText(
            f"Wrote {out_path}\nWrote {h5_path}\n\n{preview_text(payload)}"
        )
        print(f"Wrote filament constraint to {out_path}")
        print(f"Wrote spatial constraint maps to {h5_path}")

    load_button.clicked.connect(on_load)
    export_button.clicked.connect(on_export)
    cone_spin.valueChanged.connect(lambda _: refresh_preview())
    psi_step_spin.valueChanged.connect(lambda _: refresh_preview())
    theta_step_spin.valueChanged.connect(lambda _: refresh_preview())
    theta_center_spin.valueChanged.connect(lambda _: refresh_preview())
    shapes.events.data.connect(lambda _: refresh_preview())
    search_region.events.data.connect(lambda _: refresh_preview())

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
            "Draw a filament line in napari and export an Euler-box "
            "orientation constraint YAML."
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
            "Pixel size in Angstroms. Used for the lowpass if the MRC "
            "header has none."
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
