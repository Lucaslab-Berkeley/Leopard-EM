"""Per-pixel spatial constraint maps for match template.

HDF5 layout::

    attrs: leopard_em_version, coordinate_frame, micrograph_shape
    maps/eligible          uint8  (H, W)
    maps/region_id         int16  (H, W)
    maps/n_orientations    int32  (H, W)
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
    """

    eligible: np.ndarray
    region_id: np.ndarray
    n_orientations: np.ndarray
    micrograph_shape: tuple[int, int]
    coordinate_frame: str = _COORDINATE_FRAME
    pixel_size_angstrom: float | None = None
    regions: list[dict[str, Any]] = field(default_factory=list)

    def fill_n_orientations(self, n_orient: int) -> None:
        """Set ``n_orientations`` to ``n_orient`` inside the eligible mask."""
        self.n_orientations = np.where(
            self.eligible > 0, int(n_orient), 0
        ).astype(np.int32)

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

        eligible = self.eligible[yy, xx]
        region_id = self.region_id[yy, xx]
        n_orientations = self.n_orientations[yy, xx]
        # Pixels whose center would fall outside the micrograph are ineligible.
        in_bounds = (
            (np.arange(height)[:, None] + half_template_width < img_h)
            & (np.arange(width)[None, :] + half_template_width < img_w)
        )
        eligible = np.where(in_bounds, eligible, 0).astype(np.uint8)
        region_id = np.where(in_bounds, region_id, 0).astype(np.int16)
        n_orientations = np.where(in_bounds, n_orientations, 0).astype(np.int32)

        return SpatialConstraintMaps(
            eligible=eligible,
            region_id=region_id,
            n_orientations=n_orientations,
            micrograph_shape=stats_shape,
            coordinate_frame="pos_xy",
            pixel_size_angstrom=self.pixel_size_angstrom,
            regions=self.regions,
        )

    def allowed_num_ccg(self, n_defocus: int, n_cs: int = 1) -> int:
        """Return ``sum(n_orientations) * n_defocus * n_cs``."""
        return (
            int(self.n_orientations.astype(np.int64).sum())
            * int(n_defocus)
            * int(n_cs)
        )


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
    )


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
