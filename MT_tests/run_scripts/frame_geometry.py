"""Pixel size and table-to-micrograph offset for a correlation table -- never assumed.

A correlation table's ``x``/``y`` are correlation-map coordinates. The micrograph
position of a detection is ``x + half`` with ``half = template_box // 2``, and every
distance in Angstroms needs the pixel size. Scripts here used to hard-code both
(``HALF = 300``, ``PIXEL_SIZE = 0.9194``), which is right for the in vitro searches and
silently wrong for anything else: the in situ set is 1.056 A/px with a 528 px template,
and even two of the early in vitro crop searches used a 512 px template (half 256).

Tables written from now on record their geometry, so it is read from the table.
Older tables do not; for those the caller names the template MAP the search actually
used (its header gives the box) and the pixel size. Nothing falls back to a default.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import mrcfile

from leopard_em.pydantic_models.results.correlation_table import (
    correlation_table_geometry,
)


@dataclass(frozen=True)
class FrameGeometry:
    """How to read one correlation table.

    Attributes
    ----------
    pixel_size : float
        Angstroms per pixel of the searched micrograph.
    half_width : int
        Table ``x``/``y`` plus this is the micrograph position.
    source : str
        Where these came from, for the log.
    correlations_per_pixel : int
        Orientations x defocus planes the search scored at each pixel -- what
        ``refine_template``'s ``total_correlations`` means. Read from the table's own
        search space, so it cannot drift from the search that made it.
    """

    pixel_size: float
    half_width: int
    source: str
    correlations_per_pixel: int

    def describe(self) -> str:
        """One line for a script's output."""
        return (f"frame geometry: {self.pixel_size:.4f} A/px, table-to-micrograph "
                f"offset {self.half_width} px, {self.correlations_per_pixel:,} "
                f"correlations per pixel ({self.source})")


def template_half_width(template_path: str | Path) -> int:
    """``box // 2`` from a template map's header -- what ``match_template`` uses."""
    with mrcfile.open(str(template_path), header_only=True, permissive=True) as handle:
        return int(handle.header.nx) // 2


def correlations_per_pixel(table_path: str | Path) -> int:
    """Orientations x defocus planes, from the table's stored search space."""
    with h5py.File(str(table_path), "r") as handle:
        space = handle["search_space"]
        n_defocus = int(space["defocus_offsets"].shape[0])
        if "euler_angles" in space:
            n_orient = int(space["euler_angles"].shape[0])
        else:
            n_orient = int(space["phi_theta_angles"].shape[0]) * int(
                space["psi_angles"].shape[0])
    return n_orient * n_defocus


def frame_geometry(
    table_path: str | Path,
    legacy_template: str | Path | None = None,
    legacy_pixel_size: float | None = None,
) -> FrameGeometry:
    """Geometry for ``table_path``: recorded in the table, else from the caller.

    Parameters
    ----------
    table_path : str or Path
        The correlation table.
    legacy_template : str or Path, optional
        The template map the search used. Consulted ONLY when the table records no
        geometry, so pass the map that actually made this table.
    legacy_pixel_size : float, optional
        Likewise, the search's pixel size, consulted only for such tables.

    Raises
    ------
    ValueError
        When the table records nothing and the caller supplies nothing.
    """
    recorded = correlation_table_geometry(str(table_path))
    half = recorded["template_half_width"]
    pixel = recorded["pixel_size"]
    count = correlations_per_pixel(table_path)
    if half is not None and pixel is not None:
        return FrameGeometry(pixel, half, "recorded in the table", count)

    if legacy_template is None or legacy_pixel_size is None:
        raise ValueError(
            f"{table_path} records no search geometry (it predates that). Pass the "
            "template map this search used and its pixel size; they are not assumed."
        )
    return FrameGeometry(
        float(pixel if pixel is not None else legacy_pixel_size),
        int(half if half is not None else template_half_width(legacy_template)),
        f"table predates recorded geometry; box from {Path(legacy_template).name}",
        count,
    )


# The in vitro searches (GMPCPP, GDP, the synthetics) all ran at 0.9194 A/px on maps
# named <stem>_0.9194_bscale0.5.mrc. These describe tables that predate recorded
# geometry and nothing else: a table that records its own always wins.
LEGACY_PIXEL_SIZE = 0.9194
MT_ROOT = Path(__file__).resolve().parents[1]


def map_for(stem: str, pixel_size: float) -> Path:
    """The map for template ``stem`` simulated at ``pixel_size``.

    ``simulate_lattice_competition.py --pixel-size`` names them this way, so a later
    stage picks the rival or partner map matching the search from the pixel size the
    search's own stack carries.
    """
    return MT_ROOT / "maps" / f"{stem}_{float(pixel_size)}_bscale0.5.mrc"


def legacy_map(stem: str) -> Path:
    """The map an in vitro search used for template ``stem``."""
    return map_for(stem, LEGACY_PIXEL_SIZE)


def legacy_frame_geometry(table_path: str | Path, stem: str) -> FrameGeometry:
    """``frame_geometry`` for a table from an in vitro search with template ``stem``."""
    return frame_geometry(table_path, legacy_template=legacy_map(stem),
                          legacy_pixel_size=LEGACY_PIXEL_SIZE)
