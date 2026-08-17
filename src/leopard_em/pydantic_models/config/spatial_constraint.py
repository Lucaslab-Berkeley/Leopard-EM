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
    regions/0001/...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from leopard_em.pydantic_models.custom_types import BaseModel2DTM

_MAPS_GROUP = "maps"
_REGIONS_GROUP = "regions"
_COORDINATE_FRAME = "pos_xy_img"


class SpatialBox(BaseModel2DTM):
    """Axis-aligned rectangle in image (particle-center) coordinates.

    Coordinates follow numpy / napari convention: ``(y, x)`` with ``y``
    increasing downward. ``y0, x0`` is one corner and ``y1, x1`` the opposite.
    """

    y0: float
    x0: float
    y1: float
    x1: float

    def as_ymin_xmin_ymax_xmax(self) -> tuple[float, float, float, float]:
        """Return ``(ymin, xmin, ymax, xmax)`` regardless of corner order."""
        return (
            min(self.y0, self.y1),
            min(self.x0, self.x1),
            max(self.y0, self.y1),
            max(self.x0, self.x1),
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


def rasterize_rectangle(
    image_shape: tuple[int, int],
    box: SpatialBox | tuple[float, float, float, float],
    n_orientations: int,
    region_id: int = 1,
) -> SpatialConstraintMaps:
    """Paint one axis-aligned box onto ``(H, W)`` maps."""
    height, width = image_shape
    if isinstance(box, SpatialBox):
        ymin, xmin, ymax, xmax = box.as_ymin_xmin_ymax_xmax()
        box_tuple = (box.y0, box.x0, box.y1, box.x1)
    else:
        ymin, xmin, ymax, xmax = (
            min(box[0], box[2]),
            min(box[1], box[3]),
            max(box[0], box[2]),
            max(box[1], box[3]),
        )
        box_tuple = box

    y_idx = np.arange(height)[:, None]
    x_idx = np.arange(width)[None, :]
    # Inclusive on both ends in pixel-center coordinates.
    inside = (y_idx >= ymin) & (y_idx <= ymax) & (x_idx >= xmin) & (x_idx <= xmax)

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
                "box": box_tuple,
                "region_id": region_id,
            }
        ],
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
        "psi_min",
        "psi_max",
        "psi_step",
        "theta_step",
        "region_id",
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


def _read_region_group(grp: h5py.Group) -> dict[str, Any]:
    region: dict[str, Any] = {
        key: _from_h5_attr(value) for key, value in grp.attrs.items()
    }
    if "line" in grp:
        region["line"] = tuple(np.asarray(grp["line"][:], dtype=np.float64).tolist())
    if "box" in grp:
        region["box"] = tuple(np.asarray(grp["box"][:], dtype=np.float64).tolist())
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
