"""Filament orientation constraint: Euler-box search around an in-plane line.

Assumes the template microtubule (or other filament) axis is along Z. Roma
``'ZYZ'`` is intrinsic, with angles ordered ``(phi, theta, psi)``. torch-so3
treats ``(phi, theta)`` as the view on the sphere and ``psi`` as the in-plane
rotation of that projection:

* ``psi`` — in-plane angle of the tube in the image (the drawn line)
* ``theta`` — tilt of the tube vs the beam (centered at 90° for side-on)
* ``phi`` — roll around the tube (searched over a full 360°)

A line drawn on the micrograph sets ``psi``. The opposite polarity along the
same line is the second pole at ``psi + 180°``. The ± range is an Euler box
in ``theta`` and ``psi`` around each pole, not a true spherical cone.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import yaml
from pydantic import AliasChoices, ConfigDict, Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM

from .orientation_search import MultipleOrientationConfig, OrientationSearchConfig
from .spatial_constraint import (
    SpatialBox,
    SpatialConstraintMaps,
    _as_euler_array,
    _in_angle_range,
    rasterize_rectangle,
    read_spatial_constraint_hdf5,
    write_spatial_constraint_hdf5,
)

_ANGLE_DECIMALS = 4


class FilamentLine(BaseModel2DTM):
    """A 2-point line in image coordinates.

    Coordinates follow numpy / napari convention: ``(y, x)`` is ``(row, column)``
    with ``y`` increasing downward.
    """

    y0: float
    x0: float
    y1: float
    x1: float


class FilamentConstraint(BaseModel2DTM):
    """Sidecar that subsets a YAML orientation search, not a replacement grid.

    The match-template YAML still owns the FFT / default mean-variance Euler
    list. This model stores an Euler box around an in-plane filament line. At
    runtime that box is intersected with the YAML angles to build a
    per-pixel (or broadcast 1-D) eligibility mask.

    Attributes
    ----------
    filament_angle_deg : float
        In-plane filament angle in degrees, used as ``psi`` for pole 1. See
        ``filament_psi_from_image_line``.
    cone_half_angle_deg : float
        Half-width of the Euler box around each pole, in degrees. Applied to
        both ``theta`` and ``psi``.
    theta_center_deg : float
        Center of the ``theta`` box. Default 90° (side-on).
    phi_min, phi_max : float
        Roll around the filament axis, in degrees. Default is a full 360°.
    psi_step, theta_step : float
        Angular sampling used when dumping a preview ``OrientationSearchConfig``.
        The match-template YAML still controls the actual search sampling.
    base_grid_method : str
        Sampling method forwarded to the preview ``OrientationSearchConfig``.
    line : FilamentLine, optional
        Image-space line the angle was measured from.
    micrograph_path : str, optional
        Micrograph the line was drawn on.
    spatial_constraint_path : str, optional
        Path to a per-pixel constraint HDF5 written by the napari helper.
    spatial_box : SpatialBox, optional
        Image-space quadrilateral of four ``(y, x)`` corners (particle-center
        coordinates). Stored for round-trip; the HDF5 is authoritative at
        runtime. Legacy ``y0, x0, y1, x1`` YAML is still accepted.
    stats_from_valid_orientations_defocus : bool
        If True, mean/variance use only eligible (pixel, orientation, defocus)
        tuples. Default False keeps mean/variance from all searched angles and
        defocus values. The older YAML key
        ``stats_from_valid_orientations`` is still accepted.
    """

    # Sidecar YAML may also contain a generated orientation_search_config block.
    model_config: ClassVar = ConfigDict(extra="ignore", populate_by_name=True)

    filament_angle_deg: float
    cone_half_angle_deg: Annotated[float, Field(gt=0.0, le=180.0)] = 10.0
    theta_center_deg: float = 90.0
    phi_min: float = 0.0
    phi_max: float = 360.0
    psi_step: Annotated[float, Field(gt=0.0)] = 1.5
    theta_step: Annotated[float, Field(gt=0.0)] = 2.5
    base_grid_method: Literal["uniform", "healpix", "cartesian"] = "uniform"
    line: FilamentLine | None = None
    micrograph_path: str | None = None
    spatial_constraint_path: str | None = None
    spatial_box: SpatialBox | None = None
    stats_from_valid_orientations_defocus: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "stats_from_valid_orientations_defocus",
            "stats_from_valid_orientations",
        ),
    )

    @classmethod
    def from_line(
        cls,
        y0: float,
        x0: float,
        y1: float,
        x1: float,
        **kwargs: Any,
    ) -> FilamentConstraint:
        """Build a constraint from an image-space line (y down, x right)."""
        angle = filament_psi_from_image_line(y0, x0, y1, x1)
        return cls(
            filament_angle_deg=angle,
            line=FilamentLine(y0=y0, x0=x0, y1=y1, x1=x1),
            **kwargs,
        )

    def pole_psi_angles_deg(self) -> tuple[float, float]:
        """Return the two in-plane poles ``(psi, psi + 180)`` in ``[0, 360)``."""
        pole_1 = _wrap_360(self.filament_angle_deg)
        pole_2 = _wrap_360(self.filament_angle_deg + 180.0)
        return pole_1, pole_2

    def theta_range_deg(self) -> tuple[float, float]:
        """Return the clamped ``(theta_min, theta_max)`` Euler-box range."""
        return _clamp_theta_range(self.theta_center_deg, self.cone_half_angle_deg)

    def to_orientation_config(self) -> MultipleOrientationConfig:
        """Build the two-pole (possibly wrap-split) orientation search."""
        theta_min, theta_max = self.theta_range_deg()
        configs: list[OrientationSearchConfig] = []

        if self.cone_half_angle_deg >= 180.0:
            configs.append(
                self._make_orientation_config(
                    psi_min=0.0,
                    psi_max=360.0,
                    theta_min=theta_min,
                    theta_max=theta_max,
                )
            )
            return MultipleOrientationConfig(orientation_configs=configs)

        for pole_psi in self.pole_psi_angles_deg():
            for psi_min, psi_max in periodic_euler_box_intervals(
                pole_psi, self.cone_half_angle_deg
            ):
                configs.append(
                    self._make_orientation_config(
                        psi_min=psi_min,
                        psi_max=psi_max,
                        theta_min=theta_min,
                        theta_max=theta_max,
                    )
                )

        return MultipleOrientationConfig(orientation_configs=configs)

    def orientation_allowed_mask(self, euler_angles: Any) -> np.ndarray:
        """Return a ``(N,)`` bool mask of YAML angles inside this Euler box.

        ``euler_angles`` is ``(N, 3)`` with columns ``(phi, theta, psi)`` in
        degrees (roma ``'ZYZ'``). The mask is True where the angle falls in
        the ``phi`` roll range, the ``theta`` box, and either pole's ``psi``
        interval. This is the search-grid subset, not a new grid.
        """
        angles = _as_euler_array(euler_angles)
        phi = angles[:, 0]
        theta = angles[:, 1]
        psi = angles[:, 2]
        theta_min, theta_max = self.theta_range_deg()
        theta_ok = _in_angle_range(theta, theta_min, theta_max, period=None)
        phi_ok = _in_angle_range(phi, self.phi_min, self.phi_max, period=360.0)
        if self.cone_half_angle_deg >= 180.0:
            psi_ok = np.ones(psi.shape, dtype=bool)
        else:
            psi_ok = np.zeros(psi.shape, dtype=bool)
            half = float(self.cone_half_angle_deg)
            for pole in self.pole_psi_angles_deg():
                delta = np.abs(((psi - pole + 180.0) % 360.0) - 180.0)
                psi_ok |= delta <= half + 1e-3
        return phi_ok & theta_ok & psi_ok

    def preview_text(self) -> str:
        """Human-readable summary of the Euler boxes for the GUI."""
        pole_1, pole_2 = self.pole_psi_angles_deg()
        theta_min, theta_max = self.theta_range_deg()
        lines = [
            f"psi pole 1: {_round_angle(pole_1):.2f}°",
            f"psi pole 2: {_round_angle(pole_2):.2f}°",
            f"Euler box ±{self.cone_half_angle_deg:g}°",
            f"  theta: [{theta_min:.2f}, {theta_max:.2f}]",
            f"  phi:   [{self.phi_min:.2f}, {self.phi_max:.2f}]",
        ]
        for i, pole_psi in enumerate((pole_1, pole_2), start=1):
            intervals = periodic_euler_box_intervals(
                pole_psi, self.cone_half_angle_deg
            )
            interval_str = ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in intervals)
            lines.append(f"  psi pole {i}: {interval_str}")
        return "\n".join(lines)

    def to_sidecar_dict(self) -> dict[str, Any]:
        """Serialize constraint fields plus the generated orientation config."""
        payload = self.model_dump(exclude_none=True)
        payload["orientation_search_config"] = self.to_orientation_config().model_dump()
        return payload

    def save_sidecar(self, path: str) -> None:
        """Write a YAML sidecar that match-template can consume."""
        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(
                self.to_sidecar_dict(),
                handle,
                default_flow_style=False,
                sort_keys=False,
            )

    def load_spatial_maps(self) -> SpatialConstraintMaps | None:
        """Load per-pixel maps from ``spatial_constraint_path``, if set.

        ``n_orientations`` is whatever was stored at write time. Call
        ``expand_orientation_against_grid`` (via ``stats_maps_for_template``)
        to count YAML angles inside this Euler box.
        """
        if not self.spatial_constraint_path:
            return None
        return read_spatial_constraint_hdf5(self.spatial_constraint_path)

    def stats_maps_for_template(
        self,
        image_shape: tuple[int, int],
        template_width: int,
        defocus_values: Any | None = None,
        euler_angles: Any | None = None,
    ) -> SpatialConstraintMaps | None:
        """Return maps in stats-map (``pos_xy``) coordinates, or None.

        If ``defocus_values`` / ``euler_angles`` are given, optional HDF5
        bounds are expanded against this run's YAML grids. A missing
        orientation mask is filled from this constraint's Euler box.
        """
        maps = self.load_spatial_maps()
        if maps is None:
            return None
        half_width = int(template_width) // 2
        stats_shape = (
            int(image_shape[0]) - int(template_width) + 1,
            int(image_shape[1]) - int(template_width) + 1,
        )
        if stats_shape[0] <= 0 or stats_shape[1] <= 0:
            raise ValueError(
                "Template is larger than the micrograph; cannot convert "
                "spatial constraint maps to stats-map coordinates."
            )
        maps = maps.to_stats_map_coords(half_width, stats_shape)
        if defocus_values is not None:
            maps.expand_defocus_against_grid(defocus_values)
        if euler_angles is not None:
            maps.expand_orientation_against_grid(euler_angles)
            # YAML Euler box is authoritative. Region-table boxes in an older
            # HDF5 may still describe the inverted phi/psi assignment.
            if (
                maps.orientation_eligible is None
                or np.asarray(maps.orientation_eligible).ndim == 1
            ):
                maps.apply_orientation_mask_1d(
                    self.orientation_allowed_mask(euler_angles)
                )
        return maps

    def write_spatial_hdf5(
        self,
        path: str,
        image_shape: tuple[int, int],
        n_orientations: int | None = None,
        pixel_size_angstrom: float | None = None,
        leopard_em_version: str = "uninstalled",
    ) -> SpatialConstraintMaps:
        """Rasterize ``spatial_box`` and write the constraint HDF5."""
        if self.spatial_box is None:
            raise ValueError("spatial_box is required to write a constraint HDF5.")
        if n_orientations is None:
            n_orientations = int(self.to_orientation_config().euler_angles.shape[0])
        maps = rasterize_rectangle(
            image_shape=image_shape,
            box=self.spatial_box,
            n_orientations=n_orientations,
        )
        maps.pixel_size_angstrom = pixel_size_angstrom
        maps.regions = [
            {
                "cone_half_angle_deg": self.cone_half_angle_deg,
                "theta_center_deg": self.theta_center_deg,
                "phi_min": self.phi_min,
                "phi_max": self.phi_max,
                "psi_step": self.psi_step,
                "theta_step": self.theta_step,
                "base_grid_method": self.base_grid_method,
                "region_id": 1,
                "filament_angle_deg": self.filament_angle_deg,
                "box": self.spatial_box.corner_array(),
                "line": None
                if self.line is None
                else (self.line.y0, self.line.x0, self.line.y1, self.line.x1),
                "orientation_configs": [
                    block.model_dump()
                    for block in self.to_orientation_config().orientation_configs
                ],
            }
        ]
        write_spatial_constraint_hdf5(path, maps, leopard_em_version=leopard_em_version)
        return maps

    def _make_orientation_config(
        self,
        psi_min: float,
        psi_max: float,
        theta_min: float,
        theta_max: float,
    ) -> OrientationSearchConfig:
        return OrientationSearchConfig(
            symmetry=None,
            psi_step=self.psi_step,
            theta_step=self.theta_step,
            phi_min=_round_angle(self.phi_min),
            phi_max=_round_angle(self.phi_max),
            theta_min=_round_angle(theta_min),
            theta_max=_round_angle(theta_max),
            psi_min=_round_angle(psi_min),
            psi_max=_round_angle(psi_max),
            base_grid_method=self.base_grid_method,
        )


def filament_psi_from_image_line(
    y0: float,
    x0: float,
    y1: float,
    x1: float,
) -> float:
    """Return in-plane ``psi`` in degrees from an image-space line.

    Image coordinates are numpy / napari ``(y, x)`` with ``y`` increasing
    downward. torch-so3 ``'ZYZ'`` treats ``psi`` as the in-plane rotation of
    the projection, so the image-space line ``(dx, dy)`` maps to
    ``psi = atan2(-dy, dx)``.

    The line is undirected: ``psi`` and ``psi + 180°`` are the two poles of
    the same filament.

    Parameters
    ----------
    y0, x0, y1, x1 : float
        Line endpoints in image pixels.

    Returns
    -------
    float
        ``psi`` for pole 1, in ``[0, 360)``.

    Raises
    ------
    ValueError
        If the line has zero length.
    """
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        raise ValueError("Filament line has zero length.")
    return _wrap_360(math.degrees(math.atan2(-dy, dx)))


# Deprecated alias; the image-plane angle is ZYZ psi, not phi.
filament_phi_from_image_line = filament_psi_from_image_line


def periodic_euler_box_intervals(
    center_deg: float,
    half_width_deg: float,
) -> list[tuple[float, float]]:
    """Split ``center ± half_width`` into ``[0, 360]`` intervals.

    A range that wraps past 0/360 is returned as two intervals so it can be
    represented by ``OrientationSearchConfig`` min/max fields.
    """
    if half_width_deg >= 180.0:
        return [(0.0, 360.0)]

    lo = _wrap_360(center_deg - half_width_deg)
    hi = _wrap_360(center_deg + half_width_deg)

    if math.isclose(lo, hi, abs_tol=1e-12):
        return [(lo, lo)]

    if hi == 0.0:
        return [(_round_angle(lo), 360.0)]

    if lo < hi:
        return [(_round_angle(lo), _round_angle(hi))]

    intervals: list[tuple[float, float]] = [(_round_angle(lo), 360.0)]
    if hi > 0.0:
        intervals.append((0.0, _round_angle(hi)))
    return intervals


phi_euler_box_intervals = periodic_euler_box_intervals


def _clamp_theta_range(center_deg: float, half_width_deg: float) -> tuple[float, float]:
    theta_min = max(0.0, center_deg - half_width_deg)
    theta_max = min(180.0, center_deg + half_width_deg)
    if theta_min > theta_max:
        raise ValueError(
            f"Invalid theta Euler box: center={center_deg}, ±{half_width_deg}."
        )
    return _round_angle(theta_min), _round_angle(theta_max)


def _wrap_360(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def _round_angle(angle_deg: float) -> float:
    return round(float(angle_deg), _ANGLE_DECIMALS)
