"""Per-pixel spatial constraint maps for match template.

HDF5 layout::

    attrs: leopard_em_version, coordinate_frame, micrograph_shape
    maps/eligible          uint8  (H, W)
    maps/region_id         int16  (H, W)
    maps/n_orientations    int32  (H, W)
    maps/n_defocus         int32  (H, W)            # optional
    maps/defocus_min       float32 (H, W)           # optional, relative Å
    maps/defocus_max       float32 (H, W)           # optional, relative Å
    maps/defocus_eligible  uint8  (n_defocus, H, W) # optional, kernel-ready
    maps/orientation_eligible uint8 (n_orient, H, W) or (n_orient,)  # optional
    regions/0001/...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
from pydantic import field_validator, model_validator

from leopard_em.pydantic_models.custom_types import BaseModel2DTM

_MAPS_GROUP = "maps"
_REGIONS_GROUP = "regions"
_COORDINATE_FRAME = "pos_xy_img"


class SpatialBox(BaseModel2DTM):
    """Quadrilateral search region in image (particle-center) coordinates.

    Four corners in numpy / napari ``(y, x)`` (``y`` increasing downward).
    The region is the interior of the closed polygon those corners define,
    with edges drawn between consecutive corners (and the last back to the
    first). Click order does not matter: corners are ordered around their
    centroid.

    The older axis-aligned YAML form ``y0, x0, y1, x1`` is still accepted
    and converted to four rectangle corners.
    """

    corners: list[tuple[float, float]]

    @model_validator(mode="before")
    @classmethod
    def _legacy_axis_aligned(cls, data: Any) -> Any:
        if isinstance(data, dict) and "corners" not in data and "y0" in data:
            y0 = float(data["y0"])
            x0 = float(data["x0"])
            y1 = float(data["y1"])
            x1 = float(data["x1"])
            ymin, ymax = min(y0, y1), max(y0, y1)
            xmin, xmax = min(x0, x1), max(x0, x1)
            return {
                "corners": [
                    (ymin, xmin),
                    (ymin, xmax),
                    (ymax, xmax),
                    (ymax, xmin),
                ]
            }
        return data

    @field_validator("corners")
    @classmethod
    def _four_corners(cls, value: list[Any]) -> list[tuple[float, float]]:
        if len(value) != 4:
            raise ValueError("spatial_box.corners must contain four (y, x) points.")
        return [(float(point[0]), float(point[1])) for point in value]

    def corner_array(self) -> np.ndarray:
        """Return corners as a ``(4, 2)`` array of ``(y, x)``."""
        return np.asarray(self.corners, dtype=np.float64)

    def as_ymin_xmin_ymax_xmax(self) -> tuple[float, float, float, float]:
        """Return the axis-aligned bounding box of the four corners."""
        corners = self.corner_array()
        return (
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        )


@dataclass
class SpatialConstraintMaps:
    """In-memory per-pixel constraint maps.

    Attributes
    ----------
    eligible : np.ndarray
        ``uint8`` mask, 1 = in play.
    region_id : np.ndarray
        ``int16`` region id, 0 = none.
    n_orientations : np.ndarray
        ``int32`` allowed orientations at each pixel (0 if not eligible).
    micrograph_shape : tuple[int, int]
        ``(H, W)`` of the maps, in ``pos_xy_img`` coordinates unless converted.
    coordinate_frame : str
        ``"pos_xy_img"`` (particle center) or ``"pos_xy"`` (stats-map / template
        top-left).
    pixel_size_angstrom : float, optional
        Display metadata.
    regions : list[dict]
        Region table (Euler-box parameters, geometry, orientation configs).
    n_defocus : np.ndarray, optional
        ``int32`` allowed defocus steps at each pixel.
    defocus_min : np.ndarray, optional
        Per-pixel relative-defocus lower bound, Angstroms.
    defocus_max : np.ndarray, optional
        Per-pixel relative-defocus upper bound, Angstroms.
    defocus_eligible : np.ndarray, optional
        ``uint8`` kernel-ready mask of shape ``(n_defocus, H, W)``.
    orientation_eligible : np.ndarray, optional
        ``uint8`` kernel-ready mask of shape ``(n_orient, H, W)`` or
        ``(n_orient,)``. A 1-D mask is the same allowed YAML angles at every
        in-play pixel (broadcast with ``eligible``).
    """

    eligible: np.ndarray
    region_id: np.ndarray
    n_orientations: np.ndarray
    micrograph_shape: tuple[int, int]
    coordinate_frame: str = _COORDINATE_FRAME
    pixel_size_angstrom: float | None = None
    regions: list[dict[str, Any]] = field(default_factory=list)
    n_defocus: np.ndarray | None = None
    defocus_min: np.ndarray | None = None
    defocus_max: np.ndarray | None = None
    defocus_eligible: np.ndarray | None = None
    orientation_eligible: np.ndarray | None = None

    def fill_n_orientations(self, n_orient: int) -> None:
        """Set ``n_orientations`` to ``n_orient`` inside the eligible mask."""
        self.n_orientations = np.where(self.eligible > 0, int(n_orient), 0).astype(
            np.int32
        )

    def expand_defocus_against_grid(self, defocus_values: Any) -> None:
        """Build ``defocus_eligible`` / ``n_defocus`` against this run's grid.

        The YAML defocus grid is the source of allowed values. HDF5 maps only
        subset that grid. Missing datasets mean every in-play pixel keeps the
        full list; in that case ``defocus_eligible`` stays ``None`` so the
        kernel does not allocate a 3-D mask.

        Precedence: ``defocus_eligible`` as stored, else per-pixel
        ``defocus_min`` / ``defocus_max``, else region-table bounds.
        """
        if hasattr(defocus_values, "detach"):
            values = np.asarray(
                defocus_values.detach().cpu(), dtype=np.float64
            ).reshape(-1)
        else:
            values = np.asarray(defocus_values, dtype=np.float64).reshape(-1)
        n_grid = int(values.size)
        if n_grid == 0:
            raise ValueError("defocus_values must contain at least one sample.")
        eligible = self.eligible > 0

        if self.defocus_eligible is not None:
            ok = np.asarray(self.defocus_eligible, dtype=bool)
            if ok.shape != (n_grid, *self.eligible.shape):
                raise ValueError(
                    "maps/defocus_eligible shape "
                    f"{tuple(ok.shape)} does not match this run's defocus grid "
                    f"({n_grid}, {self.eligible.shape[0]}, {self.eligible.shape[1]}). "
                    "Store defocus_min/defocus_max instead if the grid can change."
                )
            ok = ok & eligible[None, :, :]
        elif self.defocus_min is not None and self.defocus_max is not None:
            ok = (
                values[:, None, None] >= np.asarray(self.defocus_min, dtype=np.float64)
            ) & (
                values[:, None, None] <= np.asarray(self.defocus_max, dtype=np.float64)
            )
            ok = ok & eligible[None, :, :]
        else:
            ok = self._defocus_ok_from_regions(values)
            if ok is None:
                self.n_defocus = np.where(eligible, n_grid, 0).astype(np.int32)
                self.defocus_eligible = None
                return
            ok = ok & eligible[None, :, :]

        self.defocus_eligible = ok.astype(np.uint8)
        self.n_defocus = ok.sum(axis=0).astype(np.int32)

    def expand_orientation_against_grid(self, euler_angles: Any) -> None:
        """Build ``orientation_eligible`` / ``n_orientations`` against this run.

        The YAML Euler list is the source of allowed angles. HDF5 maps only
        subset that list. Missing datasets and missing region Euler boxes mean
        every in-play pixel keeps the full list; in that case
        ``orientation_eligible`` stays ``None`` so the kernel does not allocate
        a 3-D mask.

        Precedence: ``orientation_eligible`` as stored, else per-region Euler
        boxes in the region table.
        """
        angles = _as_euler_array(euler_angles)
        n_grid = int(angles.shape[0])
        if n_grid == 0:
            raise ValueError("euler_angles must contain at least one sample.")
        eligible = self.eligible > 0

        if self.orientation_eligible is not None:
            ok = np.asarray(self.orientation_eligible, dtype=bool)
            if ok.ndim == 1:
                if ok.shape != (n_grid,):
                    raise ValueError(
                        "maps/orientation_eligible shape "
                        f"{tuple(ok.shape)} does not match this run's Euler "
                        f"grid ({n_grid},)."
                    )
                self.orientation_eligible = ok.astype(np.uint8)
                self.n_orientations = np.where(eligible, int(ok.sum()), 0).astype(
                    np.int32
                )
                return
            expected = (n_grid, *self.eligible.shape)
            if ok.shape != expected:
                raise ValueError(
                    "maps/orientation_eligible shape "
                    f"{tuple(ok.shape)} does not match this run's Euler grid "
                    f"{expected}. Store region Euler boxes instead if the "
                    "grid can change."
                )
            ok = ok & eligible[None, :, :]
            self.orientation_eligible = ok.astype(np.uint8)
            self.n_orientations = ok.sum(axis=0).astype(np.int32)
            return

        ok = self._orientation_ok_from_regions(angles)
        if ok is None:
            self.n_orientations = np.where(eligible, n_grid, 0).astype(np.int32)
            self.orientation_eligible = None
            return

        if ok.ndim == 1:
            self.orientation_eligible = ok.astype(np.uint8)
            self.n_orientations = np.where(eligible, int(ok.sum()), 0).astype(np.int32)
            return

        ok = ok & eligible[None, :, :]
        self.orientation_eligible = ok.astype(np.uint8)
        self.n_orientations = ok.sum(axis=0).astype(np.int32)

    def apply_orientation_mask_1d(self, mask_1d: Any) -> None:
        """Restrict in-play pixels to a shared 1-D YAML-angle mask."""
        ok = np.asarray(mask_1d, dtype=bool).reshape(-1)
        eligible = self.eligible > 0
        self.orientation_eligible = ok.astype(np.uint8)
        self.n_orientations = np.where(eligible, int(ok.sum()), 0).astype(np.int32)

    def _orientation_ok_from_regions(self, angles: np.ndarray) -> np.ndarray | None:
        """Piecewise-constant Euler boxes from the region table, if any."""
        boxed = [
            region
            for region in self.regions
            if _region_orientation_mask(region, angles) is not None
        ]
        if not boxed:
            return None

        masks = [_region_orientation_mask(region, angles) for region in boxed]
        unique_masks: list[np.ndarray] = []
        for mask in masks:
            assert mask is not None
            if not any(np.array_equal(mask, seen) for seen in unique_masks):
                unique_masks.append(mask)
        if len(unique_masks) == 1:
            return unique_masks[0].astype(bool)

        ok = np.ones((angles.shape[0], *self.eligible.shape), dtype=bool)
        for region, mask in zip(boxed, masks):
            assert mask is not None
            rid = int(region.get("region_id", 1))
            pixel_mask = self.region_id == rid
            ok[:, pixel_mask] = mask[:, np.newaxis]
        return ok

    def _defocus_ok_from_regions(self, values: np.ndarray) -> np.ndarray | None:
        """Piecewise-constant defocus bounds from the region table, if any."""
        bounded = [
            region
            for region in self.regions
            if region.get("defocus_min") is not None
            and region.get("defocus_max") is not None
        ]
        if not bounded:
            return None

        ok = np.ones((values.size, *self.eligible.shape), dtype=bool)
        for region in bounded:
            rid = int(region.get("region_id", 1))
            pixel_mask = self.region_id == rid
            allowed = (values >= float(region["defocus_min"])) & (
                values <= float(region["defocus_max"])
            )
            ok[:, pixel_mask] = allowed[:, np.newaxis]
        return ok

    def to_stats_map_coords(
        self,
        half_template_width: int,
        stats_shape: tuple[int, int],
    ) -> SpatialConstraintMaps:
        """Index image-center maps at stats-map (template top-left) coordinates.

        Stats-map pixel ``(y, x)`` corresponds to particle center
        ``(y + half_template_width, x + half_template_width)`` in the
        micrograph.
        """
        if self.coordinate_frame == "pos_xy":
            return self

        height, width = stats_shape
        img_h, img_w = self.eligible.shape
        ys = np.arange(height) + int(half_template_width)
        xs = np.arange(width) + int(half_template_width)
        ys = np.clip(ys, 0, img_h - 1)
        xs = np.clip(xs, 0, img_w - 1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        in_bounds = (np.arange(height)[:, None] + half_template_width < img_h) & (
            np.arange(width)[None, :] + half_template_width < img_w
        )

        eligible = _sample_and_mask(self.eligible, yy, xx, in_bounds, 0).astype(
            np.uint8
        )
        region_id = _sample_and_mask(self.region_id, yy, xx, in_bounds, 0).astype(
            np.int16
        )
        n_orientations = _sample_and_mask(
            self.n_orientations, yy, xx, in_bounds, 0
        ).astype(np.int32)
        n_defocus = _sample_optional_map(self.n_defocus, yy, xx, in_bounds, 0)
        if n_defocus is not None:
            n_defocus = n_defocus.astype(np.int32)
        defocus_min = _sample_optional_map(self.defocus_min, yy, xx, in_bounds, 0.0)
        defocus_max = _sample_optional_map(self.defocus_max, yy, xx, in_bounds, 0.0)
        defocus_eligible = None
        if self.defocus_eligible is not None:
            sampled = self.defocus_eligible[:, yy, xx]
            defocus_eligible = np.where(in_bounds[None, :, :], sampled, 0).astype(
                np.uint8
            )
        orientation_eligible = None
        if self.orientation_eligible is not None:
            stored = np.asarray(self.orientation_eligible)
            if stored.ndim == 1:
                orientation_eligible = stored.astype(np.uint8)
            else:
                sampled = stored[:, yy, xx]
                orientation_eligible = np.where(
                    in_bounds[None, :, :], sampled, 0
                ).astype(np.uint8)

        return SpatialConstraintMaps(
            eligible=eligible,
            region_id=region_id,
            n_orientations=n_orientations,
            micrograph_shape=stats_shape,
            coordinate_frame="pos_xy",
            pixel_size_angstrom=self.pixel_size_angstrom,
            regions=self.regions,
            n_defocus=n_defocus,
            defocus_min=defocus_min,
            defocus_max=defocus_max,
            defocus_eligible=defocus_eligible,
            orientation_eligible=orientation_eligible,
        )

    def allowed_num_ccg(self, n_defocus: int | None = None, n_cs: int = 1) -> int:
        """Return the number of allowed (pixel, orientation, defocus, Cs) tuples.

        Uses ``sum(n_orientations * n_defocus_map) * n_cs`` when a per-pixel
        defocus count is present, otherwise ``sum(n_orientations) * n_defocus * n_cs``.
        """
        n_orients = self.n_orientations.astype(np.int64)
        if self.n_defocus is not None:
            n_def = self.n_defocus.astype(np.int64)
            return int((n_orients * n_def).sum()) * int(n_cs)
        if n_defocus is None:
            raise ValueError("n_defocus is required when maps.n_defocus is not set.")
        return int(n_orients.sum()) * int(n_defocus) * int(n_cs)


def order_polygon_vertices(vertices: np.ndarray) -> np.ndarray:
    """Order vertices counter-clockwise around their centroid.

    For a convex quadrilateral this yields a simple (non-self-intersecting)
    boundary regardless of click order.
    """
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if verts.shape[0] < 3:
        raise ValueError("A search region needs at least three vertices.")
    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 0] - center[0], verts[:, 1] - center[1])
    return verts[np.argsort(angles)]


def points_in_polygon(height: int, width: int, vertices: np.ndarray) -> np.ndarray:
    """Return a ``(H, W)`` mask of pixel centers inside a closed polygon.

    Uses even-odd fill, then includes points that lie on an edge so the
    boundary is inclusive (matching the old axis-aligned rectangle).
    """
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
    inside |= _points_on_polygon_edges(yy, xx, verts)
    return inside


def _points_on_polygon_edges(
    yy: np.ndarray,
    xx: np.ndarray,
    verts: np.ndarray,
    atol: float = 1e-6,
) -> np.ndarray:
    """True where a pixel center lies on a polygon edge (inclusive)."""
    on_edge = np.zeros(yy.shape, dtype=bool)
    n_verts = verts.shape[0]
    for index in range(n_verts):
        y0, x0 = verts[index]
        y1, x1 = verts[(index + 1) % n_verts]
        dy = y1 - y0
        dx = x1 - x0
        scale = max(1.0, abs(dy) + abs(dx))
        cross = (yy - y0) * dx - (xx - x0) * dy
        collinear = np.abs(cross) <= atol * scale
        ymin, ymax = (min(y0, y1), max(y0, y1))
        xmin, xmax = (min(x0, x1), max(x0, x1))
        in_segment = (
            (yy >= ymin - atol)
            & (yy <= ymax + atol)
            & (xx >= xmin - atol)
            & (xx <= xmax + atol)
        )
        on_edge |= collinear & in_segment
    return on_edge


def _corners_from_box(
    box: SpatialBox | tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return ``(N, 2)`` ``(y, x)`` corners from a SpatialBox or AABB tuple."""
    if isinstance(box, SpatialBox):
        return box.corner_array()
    array = np.asarray(box, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 2:
        return array
    if array.ndim == 1 and array.size == 4:
        ymin, xmin = min(array[0], array[2]), min(array[1], array[3])
        ymax, xmax = max(array[0], array[2]), max(array[1], array[3])
        return np.array(
            [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]],
            dtype=np.float64,
        )
    raise ValueError(
        "spatial box must be four (y, x) corners or an axis-aligned "
        "(y0, x0, y1, x1) tuple."
    )


def rasterize_polygon(
    image_shape: tuple[int, int],
    corners: np.ndarray | SpatialBox | list[tuple[float, float]],
    n_orientations: int,
    region_id: int = 1,
) -> SpatialConstraintMaps:
    """Paint a closed polygon onto ``(H, W)`` maps."""
    height, width = image_shape
    verts = _corners_from_box(corners)
    inside = points_in_polygon(height, width, verts)
    eligible = inside.astype(np.uint8)
    region = np.where(inside, region_id, 0).astype(np.int16)
    n_orients = np.where(inside, int(n_orientations), 0).astype(np.int32)
    return SpatialConstraintMaps(
        eligible=eligible,
        region_id=region,
        n_orientations=n_orients,
        micrograph_shape=(height, width),
        regions=[
            {
                "box": [tuple(row) for row in verts.tolist()],
                "region_id": region_id,
            }
        ],
    )


def rasterize_rectangle(
    image_shape: tuple[int, int],
    box: SpatialBox | tuple[float, float, float, float] | np.ndarray,
    n_orientations: int,
    region_id: int = 1,
) -> SpatialConstraintMaps:
    """Paint a search region (quadrilateral or axis-aligned box) onto maps."""
    return rasterize_polygon(
        image_shape=image_shape,
        corners=_corners_from_box(box),
        n_orientations=n_orientations,
        region_id=region_id,
    )


def combine_spatial_maps(
    parts: list[SpatialConstraintMaps],
) -> SpatialConstraintMaps:
    """Union several region maps. Overlapping pixels take the later region."""
    if not parts:
        raise ValueError("Need at least one spatial map to combine.")
    first = parts[0]
    shape = first.eligible.shape
    eligible = np.zeros(shape, dtype=np.uint8)
    region_id = np.zeros(shape, dtype=np.int16)
    n_orientations = np.zeros(shape, dtype=np.int32)
    regions: list[dict[str, Any]] = []
    pixel_size = first.pixel_size_angstrom
    for maps in parts:
        if maps.eligible.shape != shape:
            raise ValueError(
                "Cannot combine spatial maps of shape "
                f"{maps.eligible.shape} with {shape}."
            )
        inside = maps.eligible > 0
        eligible = np.where(inside, np.uint8(1), eligible)
        region_id = np.where(inside, maps.region_id, region_id).astype(np.int16)
        n_orientations = np.where(inside, maps.n_orientations, n_orientations).astype(
            np.int32
        )
        regions.extend(maps.regions)
        if maps.pixel_size_angstrom is not None:
            pixel_size = maps.pixel_size_angstrom
    return SpatialConstraintMaps(
        eligible=eligible,
        region_id=region_id,
        n_orientations=n_orientations,
        micrograph_shape=first.micrograph_shape,
        coordinate_frame=first.coordinate_frame,
        pixel_size_angstrom=pixel_size,
        regions=regions,
    )


def write_spatial_constraint_hdf5(
    path: str,
    maps: SpatialConstraintMaps,
    leopard_em_version: str = "uninstalled",
    compress: bool = True,
) -> None:
    """Write constraint maps and the region table to ``path``."""
    compression_kwargs: dict = (
        {"compression": "gzip", "compression_opts": 4} if compress else {}
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["leopard_em_version"] = leopard_em_version
        handle.attrs["coordinate_frame"] = maps.coordinate_frame
        handle.attrs["micrograph_shape"] = np.array(
            maps.micrograph_shape, dtype=np.int32
        )
        if maps.pixel_size_angstrom is not None:
            handle.attrs["pixel_size_angstrom"] = float(maps.pixel_size_angstrom)

        maps_group = handle.create_group(_MAPS_GROUP)
        maps_group.create_dataset(
            "eligible",
            data=np.asarray(maps.eligible, dtype=np.uint8),
            **compression_kwargs,
        )
        maps_group.create_dataset(
            "region_id",
            data=np.asarray(maps.region_id, dtype=np.int16),
            **compression_kwargs,
        )
        maps_group.create_dataset(
            "n_orientations",
            data=np.asarray(maps.n_orientations, dtype=np.int32),
            **compression_kwargs,
        )
        if maps.n_defocus is not None:
            maps_group.create_dataset(
                "n_defocus",
                data=np.asarray(maps.n_defocus, dtype=np.int32),
                **compression_kwargs,
            )
        if maps.defocus_min is not None:
            maps_group.create_dataset(
                "defocus_min",
                data=np.asarray(maps.defocus_min, dtype=np.float32),
                **compression_kwargs,
            )
        if maps.defocus_max is not None:
            maps_group.create_dataset(
                "defocus_max",
                data=np.asarray(maps.defocus_max, dtype=np.float32),
                **compression_kwargs,
            )
        if maps.defocus_eligible is not None:
            maps_group.create_dataset(
                "defocus_eligible",
                data=np.asarray(maps.defocus_eligible, dtype=np.uint8),
                **compression_kwargs,
            )
        if maps.orientation_eligible is not None:
            maps_group.create_dataset(
                "orientation_eligible",
                data=np.asarray(maps.orientation_eligible, dtype=np.uint8),
                **compression_kwargs,
            )

        regions_group = handle.create_group(_REGIONS_GROUP)
        for index, region in enumerate(maps.regions, start=1):
            grp = regions_group.create_group(f"{index:04d}")
            _write_region_group(grp, region)


def read_spatial_constraint_hdf5(path: str) -> SpatialConstraintMaps:
    """Load constraint maps written by ``write_spatial_constraint_hdf5``."""
    with h5py.File(path, "r") as handle:
        coordinate_frame = str(handle.attrs.get("coordinate_frame", _COORDINATE_FRAME))
        shape_attr = handle.attrs.get("micrograph_shape")
        maps_group = handle[_MAPS_GROUP]
        eligible = np.asarray(maps_group["eligible"][:], dtype=np.uint8)
        region_id = np.asarray(maps_group["region_id"][:], dtype=np.int16)
        n_orientations = np.asarray(maps_group["n_orientations"][:], dtype=np.int32)
        n_defocus = _read_optional_dataset(maps_group, "n_defocus", np.int32)
        defocus_min = _read_optional_dataset(maps_group, "defocus_min", np.float32)
        defocus_max = _read_optional_dataset(maps_group, "defocus_max", np.float32)
        defocus_eligible = _read_optional_dataset(
            maps_group, "defocus_eligible", np.uint8
        )
        orientation_eligible = _read_optional_dataset(
            maps_group, "orientation_eligible", np.uint8
        )
        if shape_attr is None:
            micrograph_shape = (int(eligible.shape[0]), int(eligible.shape[1]))
        else:
            micrograph_shape = (int(shape_attr[0]), int(shape_attr[1]))
        pixel_size = handle.attrs.get("pixel_size_angstrom")
        pixel_size_angstrom = float(pixel_size) if pixel_size is not None else None

        regions: list[dict[str, Any]] = []
        if _REGIONS_GROUP in handle:
            regions_group = handle[_REGIONS_GROUP]
            for name in sorted(regions_group.keys()):
                regions.append(_read_region_group(regions_group[name]))

    return SpatialConstraintMaps(
        eligible=eligible,
        region_id=region_id,
        n_orientations=n_orientations,
        micrograph_shape=micrograph_shape,
        coordinate_frame=coordinate_frame,
        pixel_size_angstrom=pixel_size_angstrom,
        regions=regions,
        n_defocus=n_defocus,
        defocus_min=defocus_min,
        defocus_max=defocus_max,
        defocus_eligible=defocus_eligible,
        orientation_eligible=orientation_eligible,
    )


def _read_optional_dataset(
    group: h5py.Group, name: str, dtype: Any
) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][:], dtype=dtype)


def _sample_and_mask(
    array: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    in_bounds: np.ndarray,
    fill: float,
) -> np.ndarray:
    return np.where(in_bounds, array[yy, xx], fill)


def _sample_optional_map(
    array: np.ndarray | None,
    yy: np.ndarray,
    xx: np.ndarray,
    in_bounds: np.ndarray,
    fill: float,
) -> np.ndarray | None:
    if array is None:
        return None
    sampled = _sample_and_mask(array, yy, xx, in_bounds, fill)
    return np.asarray(sampled, dtype=array.dtype)


def _write_region_group(grp: h5py.Group, region: dict[str, Any]) -> None:
    for key in (
        "cone_half_angle_deg",
        "theta_center_deg",
        "phi_min",
        "phi_max",
        "psi_min",
        "psi_max",
        "psi_step",
        "theta_step",
        "region_id",
        "filament_angle_deg",
        "base_grid_method",
        "defocus_min",
        "defocus_max",
    ):
        if key in region and region[key] is not None:
            grp.attrs[key] = region[key]

    if region.get("line") is not None:
        grp.create_dataset("line", data=np.asarray(region["line"], dtype=np.float64))
    if region.get("box") is not None:
        grp.create_dataset("box", data=np.asarray(region["box"], dtype=np.float64))

    configs = region.get("orientation_configs") or []
    if configs:
        cfg_root = grp.create_group("orientation_configs")
        for cfg_index, config in enumerate(configs):
            cfg_grp = cfg_root.create_group(str(cfg_index))
            for key, value in config.items():
                if value is None:
                    continue
                cfg_grp.attrs[key] = value


def _box_dataset_to_python(array: np.ndarray) -> Any:
    """Decode a region ``box`` dataset as corners or a legacy AABB tuple."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1 and arr.size == 4:
        return tuple(float(v) for v in arr.tolist())
    corners = arr.reshape(-1, 2)
    return [tuple(float(v) for v in row) for row in corners.tolist()]


def _read_region_group(grp: h5py.Group) -> dict[str, Any]:
    region: dict[str, Any] = {
        key: _from_h5_attr(value) for key, value in grp.attrs.items()
    }
    if "line" in grp:
        region["line"] = tuple(np.asarray(grp["line"][:], dtype=np.float64).tolist())
    if "box" in grp:
        region["box"] = _box_dataset_to_python(np.asarray(grp["box"][:]))
    if "orientation_configs" in grp:
        cfg_root = grp["orientation_configs"]
        configs = []
        for name in sorted(cfg_root.keys(), key=int):
            configs.append(
                {
                    key: _from_h5_attr(value)
                    for key, value in cfg_root[name].attrs.items()
                }
            )
        region["orientation_configs"] = configs
    return region


def _from_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_euler_array(euler_angles: Any) -> np.ndarray:
    """Return Euler angles as ``(N, 3)`` float64 ``(phi, theta, psi)``."""
    if hasattr(euler_angles, "detach"):
        angles = np.asarray(euler_angles.detach().cpu(), dtype=np.float64)
    else:
        angles = np.asarray(euler_angles, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(
            "euler_angles must have shape (n_orientations, 3) with columns "
            "(phi, theta, psi)."
        )
    return angles


def euler_angles_in_boxes(
    euler_angles: np.ndarray,
    configs: list[dict[str, Any]],
    atol: float = 1e-3,
) -> np.ndarray:
    """Return a ``(N,)`` mask of YAML angles that fall in any Euler box."""
    phi = euler_angles[:, 0]
    theta = euler_angles[:, 1]
    psi = euler_angles[:, 2]
    ok = np.zeros(euler_angles.shape[0], dtype=bool)
    for config in configs:
        phi_ok = _in_angle_range(
            phi, config.get("phi_min"), config.get("phi_max"), period=360.0, atol=atol
        )
        theta_ok = _in_angle_range(
            theta,
            config.get("theta_min"),
            config.get("theta_max"),
            period=None,
            atol=atol,
        )
        psi_ok = _in_angle_range(
            psi, config.get("psi_min"), config.get("psi_max"), period=360.0, atol=atol
        )
        ok |= phi_ok & theta_ok & psi_ok
    return ok


def _in_angle_range(
    values: np.ndarray,
    lo: Any,
    hi: Any,
    period: float | None,
    atol: float = 1e-3,
) -> np.ndarray:
    """Inclusive range test. ``None`` bounds mean the axis is unrestricted."""
    if lo is None or hi is None:
        return np.ones(values.shape, dtype=bool)
    low = float(lo)
    high = float(hi)
    if period is None:
        return (values >= low - atol) & (values <= high + atol)
    if high + atol >= period and low <= atol:
        return np.ones(values.shape, dtype=bool)
    if low <= high:
        return (values >= low - atol) & (values <= high + atol)
    return (values >= low - atol) | (values <= high + atol)


def _region_orientation_mask(
    region: dict[str, Any],
    angles: np.ndarray,
) -> np.ndarray | None:
    """Return a ``(N,)`` mask for a region Euler box, or None if unspecified."""
    configs = region.get("orientation_configs") or []
    usable = [
        config
        for config in configs
        if any(
            config.get(key) is not None
            for key in (
                "phi_min",
                "phi_max",
                "theta_min",
                "theta_max",
                "psi_min",
                "psi_max",
            )
        )
    ]
    if not usable:
        return None
    return euler_angles_in_boxes(angles, usable)
