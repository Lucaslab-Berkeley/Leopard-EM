"""Recovering filament geometry from orientation-aware 2DTM peaks.

A helical filament template is built with its axis along the template z-axis, so once
a peak's orientation is known the matched Euler angles carry real geometric meaning:

- ``psi`` is the in-plane direction of the filament in the image, and its two poles
  (180 degrees apart) are the two possible polarities.
- ``phi`` is a roll about the filament axis, i.e. the **azimuth** around the tube --
  which protofilament, and near wall versus far wall.
- ``theta`` is the tilt of the axis out of the image plane (90 degrees = in-plane).

The one piece that is not in the angles is where the filament axis sits *inside* the
template box. For a template built from a complete ring the axis passes through the box
centre, but a template built from a patch of the lattice is offset, and that offset
rotates with ``phi``. :class:`TemplateLatticeGeometry` carries this, which lets every
peak predict a point on the filament axis independently -- and the scatter of those
predictions about a single line is a free check that the whole chain is consistent.
"""

from dataclasses import dataclass

import numpy as np
import roma
import torch

__all__ = [
    "FilamentAssignment",
    "FilamentAxis",
    "LatticeModelScore",
    "LatticeRise",
    "LatticeSites",
    "PolarityEstimate",
    "TemplateLatticeGeometry",
    "assign_detections_to_filaments",
    "axis_points_from_peaks",
    "estimate_lattice_rise",
    "estimate_polarity",
    "estimate_protofilament_number",
    "extract_lattice_sites",
    "filament_coordinates",
    "filament_direction_from_angles",
    "fit_filament_axis",
    "geometry_columns",
    "image_offset_from_template_point",
    "lattice_sharpness",
    "rise_from_template_autocorrelation",
    "score_lattice_model",
    "subunit_contrast",
    "template_axial_autocorrelation",
    "unwrap_helical_axial_coordinate",
]


def _parse_pdb_chain_centroids(pdb_path: str) -> dict[str, np.ndarray]:
    """Mean (x, y, z) of the ATOM records of each chain, in Angstroms."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    with open(pdb_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            position = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
            if chain in sums:
                sums[chain] += position
                counts[chain] += 1
            else:
                sums[chain] = position
                counts[chain] = 1

    if not sums:
        raise ValueError(f"No ATOM records found in '{pdb_path}'.")

    return {chain: sums[chain] / counts[chain] for chain in sums}


def _fit_circle(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic circle fit; returns (centre, radius)."""
    if points_xy.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 points to fit a circle, got {points_xy.shape[0]}."
        )

    design = np.column_stack(
        [2.0 * points_xy[:, 0], 2.0 * points_xy[:, 1], np.ones(len(points_xy))]
    )
    solution, *_ = np.linalg.lstsq(design, (points_xy**2).sum(axis=1), rcond=None)
    centre = solution[:2]
    radius = float(np.sqrt(solution[2] + centre @ centre))

    return centre, radius


@dataclass(frozen=True)
class TemplateLatticeGeometry:
    """Where a helical filament sits inside its template box, and its lattice.

    Attributes
    ----------
    axis_offset_angstrom : tuple[float, float]
        Position of the filament axis in the template's (x, y), relative to the centre
        of the template box, in Angstroms. Zero for a template built from a complete
        ring, or for a patch deliberately translated so the axis is at the origin;
        substantial for a patch centred on its own subunits instead.
    subunit_radius_angstrom : float
        Radius at which the repeating subunits sit, in Angstroms.
    n_protofilaments : int
        Number of protofilaments (longitudinal strands) around the tube.
    rise_angstrom : float
        Axial rise of the repeating unit along one protofilament, in Angstroms.
    n_start : int
        Helical start number: how many subunits the lattice rises going once around
        the tube. A microtubule is conventionally 3-start.
    lateral_axial_offset_angstrom : float
        Axial shift between neighbouring protofilaments, in Angstroms, signed by the
        handedness of the helix. Small (about 9 Angstroms for a microtubule) but it
        accumulates all the way around the tube, so pooling protofilaments without
        removing it smears the axial lattice into mush. Zero means a lattice with no
        helical offset at all.
    """

    axis_offset_angstrom: tuple[float, float]
    subunit_radius_angstrom: float
    n_protofilaments: int
    rise_angstrom: float
    n_start: int = 3
    lateral_axial_offset_angstrom: float = 0.0

    @property
    def axial_shift_per_turn_angstrom(self) -> float:
        """Axial shift accumulated going once around the tube."""
        return self.lateral_axial_offset_angstrom * self.n_protofilaments

    @property
    def turns_are_closed_in_repeats(self) -> float:
        """Axial shift per turn measured in lattice repeats.

        A half-integer value means the lattice cannot close with whole repeats, which
        is the geometric origin of a seam: a microtubule rises three monomers -- one
        and a half dimers -- per turn, so one lateral contact is forced out of register.

        The value depends on which repeat ``rise_angstrom`` holds. Built with the dimer
        repeat a microtubule gives -1.5 (a seam); built with the monomer repeat it gives
        -3, an integer, because the *monomer* lattice does close. That is precisely why
        monomer positions alone carry no seam signature, and why locating a seam needs
        the two halves of the repeat to be told apart.
        """
        if self.rise_angstrom == 0:
            return float("nan")
        return self.axial_shift_per_turn_angstrom / self.rise_angstrom

    @property
    def protofilament_angular_spacing_deg(self) -> float:
        """Azimuthal separation between neighbouring protofilaments, in degrees."""
        return 360.0 / self.n_protofilaments

    @property
    def axis_offset_magnitude_angstrom(self) -> float:
        """Distance from the template box centre to the filament axis."""
        return float(np.hypot(*self.axis_offset_angstrom))

    def suggested_angular_radius_deg(self, lobe_half_width_deg: float = 10.0) -> float:
        """A peak-suppression angular radius consistent with this lattice.

        Placed midway between the angular width of a single particle's correlation
        lobe and the azimuthal spacing of neighbouring protofilaments, which are
        respectively the lower and upper bounds on a sensible choice.

        Parameters
        ----------
        lobe_half_width_deg : float
            Half-width of one particle's correlation lobe in azimuth. Measure it by
            holding position and every other angle fixed and varying phi.

        Returns
        -------
        float
            Suggested ``angular_radius_deg`` for ``find_peaks_orientation_aware``.

        Raises
        ------
        ValueError
            If the lobe is already as wide as the protofilament spacing, in which case
            no radius can both merge a particle and separate protofilaments.
        """
        spacing = self.protofilament_angular_spacing_deg
        if lobe_half_width_deg >= spacing:
            raise ValueError(
                f"Correlation lobe half-width ({lobe_half_width_deg:.1f} deg) is not "
                f"smaller than the protofilament spacing ({spacing:.1f} deg), so no "
                "angular radius can separate protofilaments while still merging a "
                "single particle's own detections."
            )
        return 0.5 * (lobe_half_width_deg + spacing)

    def suggested_xy_radius_px(
        self, pixel_size_angstrom: float, lobe_half_width_px: float = 6.0
    ) -> float:
        """A peak-suppression positional radius consistent with this lattice.

        Bounded below by the correlation lobe and above by the axial rise, which is
        the closest spacing of distinct subunits along a protofilament. Neighbouring
        protofilaments can project far closer than this -- arbitrarily close near the
        edge of the tube -- but those are separated by azimuth, not position, which is
        exactly what the angular half of the suppression rule is for.

        Parameters
        ----------
        pixel_size_angstrom : float
            Pixel size of the micrograph.
        lobe_half_width_px : float
            Half-width of one particle's correlation lobe, in pixels.

        Returns
        -------
        float
            Suggested ``xy_radius_px`` for ``find_peaks_orientation_aware``.
        """
        rise_px = self.rise_angstrom / pixel_size_angstrom
        # Sit just above the lobe rather than midway: the lobe has a long tail, while
        # approaching the rise risks merging neighbouring subunits.
        return float(min(2.0 * lobe_half_width_px, 0.5 * rise_px))

    @classmethod
    def from_chain_centroids(
        cls,
        centroids: np.ndarray,
        n_protofilaments: int | None = None,
        n_start: int = 3,
        subunits_per_repeat: int = 1,
    ) -> "TemplateLatticeGeometry":
        """Derive the geometry from the subunit centroids of a template model.

        Fits a circle to the centroids projected onto the plane perpendicular to the
        filament axis, which gives both the axis position within the box and the
        subunit radius. The protofilament count and rise are then read off the
        azimuthal and axial spacings.

        Parameters
        ----------
        centroids : np.ndarray
            Subunit centroids, shape (n, 3), in Angstroms, in the template's own frame
            with the filament axis along z and the box centred on the origin.
        n_protofilaments : int | None
            Override the protofilament count instead of inferring it.
        n_start : int
            Helical start number.
        subunits_per_repeat : int
            How many modelled subunits make up one repeating unit. Pass 2 for an
            alpha/beta tubulin model with one chain per monomer to get the dimer rise,
            or leave at 1 to get the monomer rise.

        Returns
        -------
        TemplateLatticeGeometry
        """
        centroids = np.asarray(centroids, dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"centroids must have shape (n, 3), got {centroids.shape}."
            )

        centre, radius = _fit_circle(centroids[:, :2])
        # Measured against the coordinate ORIGIN, not the centroid of the supplied
        # subunits: a simulator that keeps absolute coordinates (ttsim3d with
        # center_atoms=False) centres the box on the origin, so that is the box centre.
        # The two coincide only when the model was already centroid-centred.
        offset = centre

        azimuths = np.degrees(
            np.arctan2(centroids[:, 1] - centre[1], centroids[:, 0] - centre[0])
        )
        azimuths %= 360.0

        if n_protofilaments is None:
            n_protofilaments = _infer_protofilament_count(azimuths)

        rise = _infer_rise(centroids[:, 2], azimuths, subunits_per_repeat)
        lateral = _infer_lateral_offset(
            centroids[:, 2], azimuths, int(n_protofilaments)
        )

        return cls(
            axis_offset_angstrom=(float(offset[0]), float(offset[1])),
            subunit_radius_angstrom=radius,
            n_protofilaments=int(n_protofilaments),
            rise_angstrom=rise,
            n_start=n_start,
            lateral_axial_offset_angstrom=lateral,
        )

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str,
        n_protofilaments: int | None = None,
        n_start: int = 3,
        subunits_per_repeat: int = 1,
    ) -> "TemplateLatticeGeometry":
        """Derive the geometry from a PDB model, one subunit per chain.

        The model must be in the same frame the template volume was simulated in: the
        filament axis along z and the box centred on the origin.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file.
        n_protofilaments : int | None
            Override the protofilament count instead of inferring it.
        n_start : int
            Helical start number.
        subunits_per_repeat : int
            How many chains make up one repeating unit. Pass 2 for an alpha/beta
            tubulin model with one chain per monomer to get the dimer rise.

        Returns
        -------
        TemplateLatticeGeometry
        """
        centroids = np.array(list(_parse_pdb_chain_centroids(pdb_path).values()))
        return cls.from_chain_centroids(
            centroids, n_protofilaments, n_start, subunits_per_repeat
        )


def _infer_protofilament_count(azimuths_deg: np.ndarray, max_count: int = 20) -> int:
    """Protofilament count from the azimuth distribution of the subunits.

    Subunits sit on protofilaments evenly spaced around the tube, so the azimuth
    distribution has a strong harmonic at exactly the protofilament count. Scoring
    harmonics directly is robust to a template that samples only part of the tube,
    where counting distinct azimuths would undercount.
    """
    radians = np.radians(azimuths_deg)
    orders = np.arange(2, max_count + 1)
    # Resultant length of each circular harmonic; peaks at the true count.
    power = [np.abs(np.exp(1j * order * radians).mean()) for order in orders]

    return int(orders[int(np.argmax(power))])


def _infer_rise(
    z_angstrom: np.ndarray, azimuths_deg: np.ndarray, subunits_per_repeat: int = 1
) -> float:
    """Axial repeat along a single protofilament, in Angstroms.

    ``subunits_per_repeat`` is how many modelled subunits make up one repeating unit:
    1 when each subunit is itself the repeat, 2 for an alpha/beta dimer modelled as two
    chains. Getting it wrong scales the answer by exactly that factor.

    Notes
    -----
    Prefer :func:`rise_from_template_autocorrelation` when the simulated volume is to
    hand: it measures the template as actually rendered, and returns pixels, so the
    ratio against a measurement carries no pixel-size assumption. This function is for
    when only the model is available.

    The two halves of a repeat are generally *not* evenly spaced -- in tubulin the
    alpha-to-beta step is about 41 Angstroms and beta-to-alpha about 43. Neither the
    median step nor a plain linear fit recovers the true mean from that: the median
    returns one of the two alternates, and a linear fit is biased whenever the number
    of steps is odd. The alternation is therefore fitted explicitly, as
    ``height = a + b * index + A * (-1) ** index``, and ``b`` -- which is unbiased --
    is the per-subunit rise.
    """
    order = np.argsort(azimuths_deg)
    grouped: list[list[int]] = []
    for index in order:
        if grouped and abs(azimuths_deg[index] - azimuths_deg[grouped[-1][0]]) < 3.0:
            grouped[-1].append(int(index))
        else:
            grouped.append([int(index)])

    best = max(grouped, key=len)
    if len(best) < 3:
        raise ValueError(
            "Could not infer the axial rise: no protofilament in the model has at "
            "least three subunits. Pass rise_angstrom explicitly."
        )

    heights = np.sort(z_angstrom[np.asarray(best, dtype=int)])
    step = max(int(subunits_per_repeat), 1)
    if heights.size <= step:
        raise ValueError(
            f"A protofilament has only {heights.size} subunits, too few to measure a "
            f"repeat spanning {step} of them."
        )

    spacing = float(np.median(np.diff(heights)))
    if spacing <= 0:
        raise ValueError("Subunit heights are not increasing; cannot index them.")

    # A template built from separate patches of lattice has gaps between them that are
    # not whole numbers of subunits, so the whole column cannot be indexed as one
    # lattice. Fit the longest run that *is* contiguous instead.
    steps = np.diff(heights)
    # 10% admits the alternation between the two halves of a repeat (about 5%
    # for tubulin) while rejecting a gap between separate patches of lattice.
    breaks = np.nonzero(np.abs(steps - spacing) > 0.10 * spacing)[0]
    bounds = [0, *(breaks + 1), len(heights)]
    runs = [
        heights[start:end] for start, end in zip(bounds, bounds[1:]) if end - start >= 3
    ]
    if not runs:
        raise ValueError(
            "No run of at least three evenly spaced subunits was found in any "
            "protofilament; the model may not be a continuous lattice."
        )
    heights = max(runs, key=len)
    indices = np.round((heights - heights[0]) / spacing)

    design = [np.ones_like(indices), indices]
    if heights.size >= 4:
        design.append((-1.0) ** indices)
    solution, *_ = np.linalg.lstsq(np.stack(design, axis=1), heights, rcond=None)

    return float(solution[1]) * step


def image_offset_from_template_point(
    point_angstrom: np.ndarray,
    phi_deg: np.ndarray,
    theta_deg: np.ndarray,
    psi_deg: np.ndarray,
) -> np.ndarray:
    """Where a point of the template lands in the image, relative to the box centre.

    The projection applies the rotation to the sampling grid rather than the object, so
    the object is rotated by the transpose and the image offset of a template point
    ``p`` is ``(R^T p)`` with the z-component dropped. Verified against rendered
    projections to better than 0.1 pixels.

    Parameters
    ----------
    point_angstrom : np.ndarray
        Template-frame point, shape (3,), in Angstroms relative to the box centre.
    phi_deg, theta_deg, psi_deg : np.ndarray
        Per-peak ZYZ Euler angles in degrees, each shape (n,).

    Returns
    -------
    np.ndarray
        Offsets in Angstroms, shape (n, 2), as (dx, dy) in image axis order.
    """
    angles = np.stack(
        [
            np.asarray(phi_deg, dtype=np.float64),
            np.asarray(theta_deg, dtype=np.float64),
            np.asarray(psi_deg, dtype=np.float64),
        ],
        axis=1,
    )
    rotations = roma.euler_to_rotmat(
        "ZYZ", torch.as_tensor(angles, dtype=torch.float64), degrees=True
    ).numpy()

    point = np.asarray(point_angstrom, dtype=np.float64).reshape(3)
    # (R^T p) for each rotation, keeping the two in-plane components.
    offsets = np.einsum("nji,j->ni", rotations, point)

    return offsets[:, :2]


def geometry_columns(peaks, prefer_refined: bool = True) -> dict[str, str]:
    """Pick the position and orientation columns to read geometry from.

    ``refine_template`` writes its results into ``refined_``-prefixed columns beside
    the originals, and those are what any lattice measurement should use: the coarse
    search grid is far too coarse for the axial precision a rise measurement needs.
    This mirrors the same preference the particle-stack accessors apply.

    Parameters
    ----------
    peaks : pd.DataFrame
        Peaks, refined or not.
    prefer_refined : bool
        Use ``refined_`` columns when every one of them is present. Set False to force
        the pre-refinement values, e.g. to compare the two.

    Returns
    -------
    dict[str, str]
        Maps ``"x"``, ``"y"``, ``"phi"``, ``"theta"``, ``"psi"`` to column names.
    """
    refined = {
        "x": "refined_pos_x",
        "y": "refined_pos_y",
        "phi": "refined_phi",
        "theta": "refined_theta",
        "psi": "refined_psi",
    }
    if prefer_refined and all(name in peaks.columns for name in refined.values()):
        return refined

    plain = {"x": "x", "y": "y", "phi": "phi", "theta": "theta", "psi": "psi"}
    if all(name in peaks.columns for name in plain.values()):
        return plain

    return {
        "x": "pos_x",
        "y": "pos_y",
        "phi": "phi",
        "theta": "theta",
        "psi": "psi",
    }


def axis_points_from_peaks(
    peaks,
    geometry: TemplateLatticeGeometry,
    pixel_size_angstrom: float,
    position_columns: tuple[str, str] | None = None,
    prefer_refined: bool = True,
) -> np.ndarray:
    """Each peak's independent prediction of a point on the filament axis.

    A peak reports where the *template box centre* sits. The filament axis is offset
    from that centre by an amount that rotates with the peak's azimuth, so undoing it
    turns every peak -- whichever protofilament and whichever wall of the tube it came
    from -- into a prediction of the same underlying axis.

    Parameters
    ----------
    peaks : pd.DataFrame
        Peaks with ``phi``, ``theta``, ``psi`` and the two position columns.
    geometry : TemplateLatticeGeometry
        Calibration of the template.
    pixel_size_angstrom : float
        Pixel size of the micrograph.
    position_columns : tuple[str, str] | None
        Names of the (x, y) columns, overriding the automatic choice. The result is in
        the same frame as these, so passing stats-map coordinates returns stats-map
        coordinates.
    prefer_refined : bool
        Read ``refined_`` columns when they are present. See :func:`geometry_columns`.

    Returns
    -------
    np.ndarray
        Predicted axis points, shape (n, 2), in pixels.
    """
    columns = geometry_columns(peaks, prefer_refined)
    if position_columns is not None:
        columns = {**columns, "x": position_columns[0], "y": position_columns[1]}

    x_column, y_column = columns["x"], columns["y"]
    missing = [name for name in columns.values() if name not in peaks.columns]
    if missing:
        raise ValueError(f"peaks is missing required columns: {missing}")

    axis_point = np.array([*geometry.axis_offset_angstrom, 0.0], dtype=np.float64)
    offsets = image_offset_from_template_point(
        axis_point,
        peaks[columns["phi"]].to_numpy(),
        peaks[columns["theta"]].to_numpy(),
        peaks[columns["psi"]].to_numpy(),
    )

    positions = np.stack(
        [
            peaks[x_column].to_numpy(dtype=np.float64),
            peaks[y_column].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )

    return positions + offsets / pixel_size_angstrom


@dataclass(frozen=True)
class FilamentAxis:
    """A straight filament axis fitted in the image plane.

    Attributes
    ----------
    origin : np.ndarray
        A point on the axis, shape (2,), in pixels.
    direction : np.ndarray
        Unit vector along the axis, shape (2,).
    residual_rms_px : float
        Root-mean-square perpendicular distance of the fitted points from the axis.
        Small values mean every peak agreed about where the axis is, which is a strong
        end-to-end check on the orientation decode and template calibration.
    """

    origin: np.ndarray
    direction: np.ndarray
    residual_rms_px: float

    @property
    def angle_deg(self) -> float:
        """In-plane direction of the axis in degrees, measured as atan2(-dy, dx)."""
        return float(
            np.degrees(np.arctan2(-self.direction[1], self.direction[0])) % 360.0
        )


def fit_filament_axis(
    points_px: np.ndarray,
    weights: np.ndarray | None = None,
    reference_direction: np.ndarray | None = None,
) -> FilamentAxis:
    """Fit a straight axis through predicted axis points.

    Uses a weighted total-least-squares (principal axis) fit, which minimises
    perpendicular distance rather than distance in one coordinate and so does not care
    how the filament is oriented in the image.

    Parameters
    ----------
    points_px : np.ndarray
        Points, shape (n, 2), in pixels.
    weights : np.ndarray | None
        Per-point weights, shape (n,); peak scores work well. Uniform if None.
    reference_direction : np.ndarray | None
        Orient the fitted direction to agree with this, shape (2,). A principal-axis
        fit determines the axis only up to sign, and anything that cares which way
        along the filament is "forwards" -- notably
        :func:`unwrap_helical_axial_coordinate`, whose correction is handed -- needs
        that sign pinned. Pass :func:`filament_direction_from_angles`. Without it the
        sign is arbitrary and may flip between runs on the same data.

    Returns
    -------
    FilamentAxis
    """
    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_px must have shape (n, 2), got {points.shape}.")
    if len(points) < 2:
        raise ValueError(f"Need at least 2 points to fit an axis, got {len(points)}.")

    if weights is None:
        weights = np.ones(len(points))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    origin = (weights[:, None] * points).sum(axis=0)
    centred = points - origin
    covariance = (weights[:, None] * centred).T @ centred

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]

    if reference_direction is not None:
        reference = np.asarray(reference_direction, dtype=np.float64).reshape(2)
        if direction @ reference < 0:
            direction = -direction

    perpendicular = centred @ np.array([-direction[1], direction[0]])
    residual = float(np.sqrt((weights * perpendicular**2).sum()))

    return FilamentAxis(origin=origin, direction=direction, residual_rms_px=residual)


def filament_coordinates(
    points_px: np.ndarray, axis: FilamentAxis
) -> tuple[np.ndarray, np.ndarray]:
    """Convert image points into along-axis and across-axis coordinates.

    Parameters
    ----------
    points_px : np.ndarray
        Points, shape (n, 2), in pixels.
    axis : FilamentAxis
        The fitted axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(s, t)``: distance along the axis from its origin, and signed perpendicular
        offset, both shape (n,) and in pixels.
    """
    centred = np.asarray(points_px, dtype=np.float64) - axis.origin
    along = centred @ axis.direction
    across = centred @ np.array([-axis.direction[1], axis.direction[0]])

    return along, across


@dataclass(frozen=True)
class PolarityEstimate:
    """Which way a filament points, and how confidently.

    Attributes
    ----------
    psi_deg : float
        Score-weighted mean in-plane angle of the winning pole, in degrees.
    score_ratio : float
        Summed score of the winning pole divided by that of the opposite pole. Large
        values mean the template genuinely discriminates the two ends; a ratio near 1
        means it does not, and no polarity can be assigned from these data.
    n_winning, n_opposite : int
        Detections assigned to each pole.
    """

    psi_deg: float
    score_ratio: float
    n_winning: int
    n_opposite: int


def estimate_polarity(
    detections,
    score_column: str = "z_score",
    score_threshold: float | None = None,
) -> PolarityEstimate:
    """Decide which of the two in-plane poles the filament actually points along.

    A filament template searched without prior knowledge of polarity is given two psi
    ranges 180 degrees apart. If the subunits are distinguishable end-to-end -- for a
    microtubule, if alpha- and beta-tubulin differ enough at the working resolution --
    one pole wins decisively and that is the polarity. If the two poles score alike,
    the template cannot tell them apart and this reports a ratio near 1 rather than an
    arbitrary answer.

    Parameters
    ----------
    detections : pd.DataFrame
        Detections or peaks with ``psi`` and ``score_column``. Using the full detection
        table rather than picked peaks gives a much better-determined ratio.
    score_column : str
        Column to sum.
    score_threshold : float | None
        Ignore detections at or below this score.

    Returns
    -------
    PolarityEstimate
    """
    working = detections
    if score_threshold is not None:
        working = working[working[score_column] > score_threshold]
    if working.empty:
        raise ValueError("No detections survive the score threshold.")

    psi = working["psi"].to_numpy(dtype=np.float64)
    score = working[score_column].to_numpy(dtype=np.float64)

    # Split on the circular mean of the doubled angle, which is insensitive to where
    # the two poles happen to fall relative to the 0/360 wrap.
    doubled = np.angle(np.exp(2j * np.radians(psi)).mean()) / 2.0
    axis_deg = np.degrees(doubled) % 180.0
    is_positive = np.cos(np.radians(psi - axis_deg)) >= 0.0

    positive_score = float(score[is_positive].sum())
    negative_score = float(score[~is_positive].sum())

    if positive_score >= negative_score:
        winning, opposite = is_positive, ~is_positive
        winning_score, opposite_score = positive_score, negative_score
    else:
        winning, opposite = ~is_positive, is_positive
        winning_score, opposite_score = negative_score, positive_score

    weights = score[winning]
    mean_psi = (
        np.degrees(np.angle((weights * np.exp(1j * np.radians(psi[winning]))).sum()))
        % 360.0
    )

    ratio = float("inf") if opposite_score == 0 else winning_score / opposite_score

    return PolarityEstimate(
        psi_deg=float(mean_psi),
        score_ratio=ratio,
        n_winning=int(winning.sum()),
        n_opposite=int(opposite.sum()),
    )


def estimate_protofilament_number(
    peaks,
    max_count: int = 20,
    score_column: str = "z_score",
) -> tuple[int, np.ndarray]:
    """Protofilament count from the azimuth distribution of the peaks.

    Peaks sit on protofilaments evenly spaced around the tube, so the distribution of
    matched azimuths carries a harmonic at exactly the protofilament count. Reading the
    harmonic rather than counting distinct azimuths is robust to only part of the tube
    being detected, and to peaks being unevenly distributed around it.

    Parameters
    ----------
    peaks : pd.DataFrame
        Peaks with ``phi`` and ``score_column``.
    max_count : int
        Largest count to consider.
    score_column : str
        Weights the harmonic sum.

    Returns
    -------
    tuple[int, np.ndarray]
        The best-scoring count, and the harmonic power for orders ``2..max_count`` so
        the margin over the runner-up can be inspected.

    Notes
    -----
    This is a *geometric* estimate and degrades when few protofilaments are detected or
    when the azimuths are clustered on one wall. Treat a narrow margin as inconclusive
    and fall back to competing whole templates of different protofilament number.
    """
    radians = np.radians(peaks["phi"].to_numpy(dtype=np.float64))
    weights = peaks[score_column].to_numpy(dtype=np.float64)
    weights = weights / weights.sum()

    orders = np.arange(2, max_count + 1)
    power = np.array(
        [np.abs((weights * np.exp(1j * order * radians)).sum()) for order in orders]
    )

    return int(orders[int(np.argmax(power))]), power


@dataclass(frozen=True)
class LatticeRise:
    """A measured axial repeat, reported both absolutely and against a reference.

    Both numbers depend on the assumed pixel size -- a 2DTM lattice measurement is
    fundamentally the product (true spacing) x (assumed pixel size), and nothing in a
    single measurement separates a mis-calibrated pixel size from a genuinely expanded
    or compacted lattice. They fail differently, though, which is why both are given:

    - ``rise_angstrom`` is what you quote, and carries the pixel-size assumption openly.
    - ``ratio`` is the form in which a later calibration is a single multiplication, and
      in which errors common to the template and the data partly cancel, so it is the
      quantity to compare between datasets processed the same way.

    Attributes
    ----------
    rise_angstrom : float
        Fitted axial repeat.
    standard_error_angstrom : float
        Statistical error on the fit only. Systematics -- pixel-size calibration,
        magnification anisotropy, tilt foreshortening -- are usually far larger.
    reference_angstrom : float
        Repeat this is being compared against, normally the template's own.
    rise_pixels : float
        The fitted repeat in pixels, the one quantity here that involves no pixel-size
        assumption at all.
    foreshortening : float
        ``sin(theta)`` averaged over the peaks: the factor by which projection shortens
        the filament's true repeat. ``rise_angstrom`` is already divided by it, so it is
        recorded here only so the size of the correction is visible. 1.0 means the
        filament lay in the image plane, or that no angles were supplied.
    """

    rise_angstrom: float
    standard_error_angstrom: float
    reference_angstrom: float
    rise_pixels: float
    foreshortening: float = 1.0

    @property
    def ratio(self) -> float:
        """Measured repeat divided by the reference."""
        return self.rise_angstrom / self.reference_angstrom

    @property
    def ratio_standard_error(self) -> float:
        """Statistical error on :attr:`ratio`."""
        return self.standard_error_angstrom / self.reference_angstrom

    @property
    def percent_difference(self) -> float:
        """Difference from the reference, as a percentage."""
        return 100.0 * (self.ratio - 1.0)

    @property
    def implied_pixel_size_angstrom(self) -> float:
        """Pixel size that would make the measurement equal the reference.

        Only meaningful if you are willing to assume the lattice really does match the
        reference; otherwise this is the expansion, not a calibration.
        """
        if self.rise_pixels == 0:
            return float("nan")
        return self.reference_angstrom / self.rise_pixels

    def summary(self) -> str:
        """One-line report of both the absolute value and the ratio."""
        return (
            f"{self.rise_angstrom:.2f} +/- {self.standard_error_angstrom:.2f} A "
            f"({self.rise_pixels:.3f} px) = {self.ratio:.4f} +/- "
            f"{self.ratio_standard_error:.4f} x reference "
            f"{self.reference_angstrom:.2f} A ({self.percent_difference:+.2f}%)"
            + (
                ""
                if self.foreshortening == 1.0
                else f"; tilt correction {100 * (1 / self.foreshortening - 1):+.2f}%"
            )
        )


def estimate_lattice_rise(
    along_axis_px: np.ndarray,
    pixel_size_angstrom: float,
    initial_rise_angstrom: float,
    weights: np.ndarray | None = None,
    max_iterations: int = 5,
    reference_rise_angstrom: float | None = None,
    theta_deg: np.ndarray | None = None,
    search_fraction: float = 0.1,
) -> LatticeRise:
    """Refine the axial repeat by indexing peaks against a lattice.

    Assigns each peak an integer lattice index from an initial guess, fits position
    against index, and iterates. Fitting a line through many indexed positions is far
    more precise than measuring individual gaps: the standard error of the slope falls
    as ``M**-1.5`` in the number of repeats spanned, not ``M**-0.5``.

    Parameters
    ----------
    along_axis_px : np.ndarray
        Along-axis coordinates of the peaks, shape (n,), in pixels.
    pixel_size_angstrom : float
        Pixel size of the micrograph.
    initial_rise_angstrom : float
        Starting guess, e.g. the template's own rise. The global scan searches
        ``search_fraction`` either side of it, so it need only be roughly right.
    weights : np.ndarray | None
        Per-peak weights, shape (n,).
    max_iterations : int
        Indexing iterations.
    reference_rise_angstrom : float | None
        Repeat to report the ratio against. Defaults to ``initial_rise_angstrom``,
        which is normally the template's own rise.
    search_fraction : float
        Fractional window about ``initial_rise_angstrom`` scanned for the repeat that
        best concentrates the positions, before local refinement. Widen it if the true
        repeat may be far from the guess; narrow it to pin the search near a known one.
    theta_deg : np.ndarray | None
        Per-peak out-of-plane tilt in degrees, shape (n,). A filament tilted away from
        the image plane is projected shorter by ``sin(theta)``; supplying this divides
        it out so the result is the true three-dimensional repeat. The effect is small
        for a filament searched near theta = 90 but is not always negligible: at
        theta = 80 it is already 1.5%, comparable to the lattice changes being measured.

    Returns
    -------
    LatticeRise
        The fitted repeat, in Angstroms, in pixels, and as a ratio to the reference.

    Raises
    ------
    ValueError
        If fewer than three peaks are supplied, or they span less than one repeat.
    """
    positions = np.asarray(along_axis_px, dtype=np.float64) * pixel_size_angstrom
    if positions.size < 3:
        raise ValueError(f"Need at least 3 peaks to fit a rise, got {positions.size}.")
    span = positions.max() - positions.min()
    if span < initial_rise_angstrom:
        raise ValueError(
            f"Peaks span {span:.0f} A, less than one repeat of "
            f"{initial_rise_angstrom:.0f} A; the rise is not determined."
        )

    if weights is None:
        weights = np.ones_like(positions)
    weights = np.asarray(weights, dtype=np.float64)

    # Global scan before local refinement. Indexing against an assumed repeat has many
    # self-consistent solutions: an error d accumulates to N*d over N repeats, so
    # aliases sit only about rise**2 / span apart and local refinement cannot leave the
    # one it starts in. On a long track that makes the answer a readout of the initial
    # guess. Pick the repeat that actually concentrates the positions first.
    rise = _scan_for_best_repeat(
        positions, weights, float(initial_rise_angstrom), search_fraction
    )
    origin = float(positions.min())
    slope, intercept = rise, origin

    for _ in range(max_iterations):
        indices = np.round((positions - origin) / rise)
        design = np.column_stack([indices, np.ones_like(indices)])
        weighted = design * weights[:, None]
        solution, *_ = np.linalg.lstsq(weighted, positions * weights, rcond=None)
        slope, intercept = float(solution[0]), float(solution[1])
        if abs(slope - rise) < 1e-9:
            rise, origin = slope, intercept
            break
        rise, origin = slope, intercept

    indices = np.round((positions - origin) / rise)
    residuals = positions - (slope * indices + intercept)
    degrees_of_freedom = max(len(positions) - 2, 1)
    scatter = float(np.sqrt((residuals**2).sum() / degrees_of_freedom))
    index_spread = float(np.sqrt(((indices - indices.mean()) ** 2).sum()))
    standard_error = scatter / index_spread if index_spread > 0 else float("inf")

    reference = (
        float(initial_rise_angstrom)
        if reference_rise_angstrom is None
        else float(reference_rise_angstrom)
    )

    foreshortening = 1.0
    if theta_deg is not None:
        sines = np.sin(np.radians(np.asarray(theta_deg, dtype=np.float64)))
        foreshortening = float((weights * sines).sum() / weights.sum())
        # Dividing by sin(theta) amplifies everything by 1 / sin(theta), so a filament
        # pointing near the viewing direction is not merely foreshortened, its axial
        # lattice is unmeasurable. Refuse rather than return a 10x-inflated repeat.
        if foreshortening < 0.1:
            raise ValueError(
                f"Mean sin(theta) is {foreshortening:.3f}: the filament lies within "
                "about 6 degrees of the viewing direction, so its axial repeat is not "
                "recoverable from a projection."
            )

    corrected = float(rise) / foreshortening

    return LatticeRise(
        rise_angstrom=corrected,
        standard_error_angstrom=standard_error / foreshortening,
        reference_angstrom=reference,
        rise_pixels=corrected / pixel_size_angstrom,
        foreshortening=foreshortening,
    )


def _infer_lateral_offset(
    z_angstrom: np.ndarray, azimuths_deg: np.ndarray, n_protofilaments: int
) -> float:
    """Axial shift between neighbouring protofilaments, in Angstroms.

    Bins subunits onto protofilaments, takes the lowest subunit of each as that
    protofilament's axial phase, and fits phase against protofilament index. The seam
    makes one lateral step an outlier, so the median step is used rather than the mean.
    """
    spacing = 360.0 / n_protofilaments
    phases: dict[int, float] = {}
    for azimuth, height in zip(azimuths_deg, z_angstrom):
        index = int(round(azimuth / spacing)) % n_protofilaments
        phases[index] = min(phases.get(index, np.inf), float(height))

    if len(phases) < 3:
        return 0.0

    indices = np.array(sorted(phases))
    heights = np.array([phases[i] for i in indices])
    steps = np.diff(heights) / np.diff(indices)

    return float(np.median(steps))


def unwrap_helical_axial_coordinate(
    along_axis_angstrom: np.ndarray,
    azimuth_deg: np.ndarray,
    geometry: TemplateLatticeGeometry,
) -> np.ndarray:
    """Remove the helical offset so every protofilament shares one axial lattice.

    Subunits on neighbouring protofilaments sit at slightly different heights, and that
    offset accumulates around the tube -- a full turn of a microtubule rises one and a
    half dimers. Pooling protofilaments without removing it spreads what should be a
    sharp axial repeat across the whole repeat distance, and any rise fitted from the
    pooled positions is meaningless.

    Parameters
    ----------
    along_axis_angstrom : np.ndarray
        Along-axis coordinates, shape (n,), in Angstroms.
    azimuth_deg : np.ndarray
        Matched ``phi`` angle of each peak, shape (n,), in degrees. Note this is the
        roll applied by the search, which runs *opposite* to azimuth in the template's
        own frame; the sign is handled here.
    geometry : TemplateLatticeGeometry
        Supplies the lateral axial offset and protofilament spacing.

    Returns
    -------
    np.ndarray
        Axial coordinates with the helical offset removed, shape (n,), in Angstroms.
        Peaks from every protofilament now fall on a common lattice of period
        ``geometry.rise_angstrom``.
    """
    # The projection rotates the sampling grid, so the object is rolled by -phi: a
    # matched phi corresponds to template azimuth -phi. The subunit height therefore
    # runs as -(lateral / spacing) * phi, and removing it *adds* that term back.
    per_degree = (
        geometry.lateral_axial_offset_angstrom
        / geometry.protofilament_angular_spacing_deg
    )

    return np.asarray(along_axis_angstrom, dtype=np.float64) + per_degree * np.asarray(
        azimuth_deg, dtype=np.float64
    )


def filament_direction_from_angles(
    phi_deg: np.ndarray, theta_deg: np.ndarray, psi_deg: np.ndarray
) -> np.ndarray:
    """Direction the filament axis points in the image, from matched orientations.

    The filament axis is the template z-axis, so its image direction is the projection
    of that axis under the object rotation. It depends on ``theta`` and ``psi`` but not
    on ``phi``, which is a roll about the axis itself -- so every peak on a filament
    agrees on it regardless of which protofilament it came from.

    Unlike a principal-axis fit this is *signed*: it points from the filament's minus
    end towards its plus end as the matched polarity defines them, which is what pins
    the sign of the helical unwrap.

    Parameters
    ----------
    phi_deg, theta_deg, psi_deg : np.ndarray
        Per-peak ZYZ Euler angles in degrees, each shape (n,).

    Returns
    -------
    np.ndarray
        Mean unit direction, shape (2,).
    """
    directions = image_offset_from_template_point(
        np.array([0.0, 0.0, 1.0]), phi_deg, theta_deg, psi_deg
    )
    mean = directions.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError(
            "The peak orientations cancel out, so no filament direction is defined; "
            "they probably span both polarities and should be split first."
        )

    return mean / norm


def template_axial_autocorrelation(volume: np.ndarray) -> np.ndarray:
    """Autocorrelation of a filament template along its own axis.

    Computes the axis line of the template's 3-D autocorrelation, normalised to 1 at
    zero lag. Each column of the volume is autocorrelated along z and the results are
    summed, rather than summing the volume first: a helical lattice puts neighbouring
    protofilaments at different heights, so collapsing them before autocorrelating
    would wash the axial repeat out entirely.

    Parameters
    ----------
    volume : np.ndarray
        Template volume, shape (nz, ny, nx), with the filament axis along the first
        (z) axis, as Leopard-EM's filament templates are built.

    Returns
    -------
    np.ndarray
        Autocorrelation against axial lag in pixels, shape (nz,), with element 0 the
        zero-lag value of 1.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}.")

    power = (np.abs(np.fft.rfft(volume, axis=0)) ** 2).sum(axis=(1, 2))
    profile = np.fft.irfft(power, n=volume.shape[0])

    if profile[0] == 0:
        raise ValueError("Template autocorrelation is zero; is the volume empty?")

    return profile / profile[0]


def _refine_peak(profile: np.ndarray, index: int) -> float:
    """Sub-pixel peak position by parabolic interpolation about ``index``."""
    if index <= 0 or index >= len(profile) - 1:
        return float(index)

    left, centre, right = profile[index - 1], profile[index], profile[index + 1]
    denominator = left - 2.0 * centre + right
    if denominator == 0:
        return float(index)

    return float(index) + 0.5 * (left - right) / denominator


def rise_from_template_autocorrelation(
    volume: np.ndarray,
    approximate_rise_px: float,
    search_fraction: float = 0.3,
) -> float:
    """Measure a template's own axial repeat, in pixels, from its autocorrelation.

    This is the reference a measured lattice spacing should be quoted against. It is
    better than reading the repeat off the deposited model for three reasons: it
    measures the template *as actually rendered*, at its own pixel size and B-factor;
    it uses the same estimator applied to the data, so estimator bias partly cancels;
    and because both are then in pixels, the resulting ratio carries no pixel-size
    assumption at all.

    Parameters
    ----------
    volume : np.ndarray
        Template volume, shape (nz, ny, nx), filament axis along z.
    approximate_rise_px : float
        Rough expected repeat in pixels; the nearest autocorrelation peak is taken.
    search_fraction : float
        Fractional window around ``approximate_rise_px`` to search.

    Returns
    -------
    float
        Axial repeat in pixels, refined to sub-pixel precision.

    Raises
    ------
    ValueError
        If no autocorrelation peak is found in the search window.
    """
    profile = template_axial_autocorrelation(volume)

    low = max(1, int(approximate_rise_px * (1.0 - search_fraction)))
    high = min(
        len(profile) - 2, int(np.ceil(approximate_rise_px * (1 + search_fraction)))
    )
    if high <= low:
        raise ValueError(
            f"Search window [{low}, {high}] px is empty; check approximate_rise_px "
            f"({approximate_rise_px}) against the volume depth ({len(profile)})."
        )

    window = profile[low : high + 1]
    # Require a genuine maximum, not a point on a plateau: a flat stretch (an empty
    # region of a non-periodic template autocorrelates to a run of zeros) satisfies
    # ">= both neighbours" everywhere and would otherwise yield a bogus repeat.
    peaks = [
        low + i
        for i in range(1, len(window) - 1)
        if window[i] >= window[i - 1]
        and window[i] >= window[i + 1]
        and (window[i] > window[i - 1] or window[i] > window[i + 1])
        and window[i] > 1e-6
    ]
    if not peaks:
        raise ValueError(
            f"No autocorrelation peak between {low} and {high} px. The template may "
            "not be a periodic filament, or approximate_rise_px may be far off."
        )

    return _refine_peak(profile, max(peaks, key=lambda i: profile[i]))


def subunit_contrast(volume: np.ndarray, repeat_px: float) -> float:
    """How distinguishable the two halves of a two-subunit repeat are, as rendered.

    A microtubule repeat is an alpha/beta tubulin dimer, and locating the seam means
    telling alpha from beta. If shifting the template by half a repeat correlates as
    well as shifting by a whole one, the two subunits are interchangeable *in the
    rendered template* and no amount of data can separate them -- the limit is the
    simulation, not the micrograph.

    Parameters
    ----------
    volume : np.ndarray
        Template volume, shape (nz, ny, nx), filament axis along z.
    repeat_px : float
        Full repeat (the dimer) in pixels.

    Returns
    -------
    float
        Autocorrelation at half the repeat divided by that at the full repeat. Near 1
        means the subunits are indistinguishable; well below 1 means they differ and
        the register is in principle recoverable.
    """
    profile = template_axial_autocorrelation(volume)
    half = profile[int(round(repeat_px / 2.0))]
    full = profile[int(round(repeat_px))]

    if full == 0:
        return float("inf")

    return float(half / full)


def lattice_sharpness(
    positions_angstrom: np.ndarray,
    repeat_angstrom: float,
    weights: np.ndarray | None = None,
) -> float:
    """How tightly positions concentrate at one phase of an assumed repeat.

    The magnitude of the first circular harmonic of the positions taken modulo the
    repeat: 1 when every position sits at the same phase, 0 when they are spread
    uniformly. Useful both for choosing between candidate repeats and for judging
    whether a fitted repeat means anything at all -- a small value says the lattice is
    not resolved, however tight the formal error on the fit.

    Parameters
    ----------
    positions_angstrom : np.ndarray
        Axial positions, shape (n,), in Angstroms.
    repeat_angstrom : float
        Repeat to test.
    weights : np.ndarray | None
        Per-position weights, shape (n,).

    Returns
    -------
    float
        Concentration in [0, 1].

    Notes
    -----
    This is maximal at the true repeat *and* at every sub-harmonic of it: positions on
    an 84 Angstrom lattice also sit perfectly on a 42 Angstrom one, with alternate
    sites empty. It therefore cannot pick a period out of a search window wide enough
    to span a factor of two. Low occupancy is the tell that a sub-harmonic was chosen.
    """
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(positions)
    weights = np.asarray(weights, dtype=np.float64)
    if repeat_angstrom <= 0:
        raise ValueError(f"repeat_angstrom must be positive, got {repeat_angstrom}.")

    phasor = (weights * np.exp(2j * np.pi * positions / repeat_angstrom)).sum()

    return float(np.abs(phasor) / weights.sum())


def _scan_for_best_repeat(
    positions: np.ndarray,
    weights: np.ndarray,
    initial: float,
    search_fraction: float,
) -> float:
    """Repeat near ``initial`` that best concentrates the positions."""
    span = float(positions.max() - positions.min())
    if span <= 0:
        return initial

    # Resolve the aliases, which are about initial**2 / span apart.
    step = max(initial**2 / span / 20.0, 1e-4)
    low = initial * (1.0 - search_fraction)
    high = initial * (1.0 + search_fraction)
    grid = np.arange(low, high + step, step)

    sharpness = np.array([lattice_sharpness(positions, r, weights) for r in grid])

    return float(grid[int(np.argmax(sharpness))])


@dataclass(frozen=True)
class LatticeSites:
    """Detections assigned to the sites of a fitted lattice.

    Attributes
    ----------
    detection_index : np.ndarray
        Row of the source detection table occupying each site, shape (n_occupied,).
    axial_index : np.ndarray
        Lattice index along the filament for each occupied site, shape (n_occupied,).
    protofilament : np.ndarray | None
        Protofilament index for each occupied site, or None for a pooled
        (protofilament-blind) extraction.
    score : np.ndarray
        Score of the detection at each occupied site, shape (n_occupied,).
    axial_deviation_angstrom : np.ndarray
        How far each detection sits from its predicted axial position. Compare against
        the search window -- deviations far below it mean the positions are set by the
        data rather than by the model.
    rise_angstrom : float
        Fitted axial repeat, corrected for out-of-plane tilt. This is an AXIAL repeat,
        not a projected one: the fit divides out the score-weighted mean of sin(theta),
        so it is directly comparable to a repeat measured from an atomic model.
    axis : FilamentAxis
        Axis the sites were measured against.
    n_predicted : int
        Total sites the model predicts over the track, occupied or not.
    """

    detection_index: np.ndarray
    axial_index: np.ndarray
    protofilament: np.ndarray | None
    score: np.ndarray
    axial_deviation_angstrom: np.ndarray
    rise_angstrom: float
    axis: FilamentAxis
    n_predicted: int

    @property
    def occupancy(self) -> float:
        """Fraction of predicted sites that a detection was found at."""
        if self.n_predicted == 0:
            return float("nan")
        return len(self.detection_index) / self.n_predicted

    def occupancy_by_protofilament(self) -> dict[int, float]:
        """Occupancy of each protofilament separately.

        A single protofilament scoring or filling consistently below the others is the
        signature the seam would produce if the real filament is structurally different
        there from the symmetrised template. Requires a protofilament-resolved
        extraction.

        Returns
        -------
        dict[int, float]
            Protofilament index to occupied fraction.

        Raises
        ------
        ValueError
            If the extraction pooled protofilaments.
        """
        if self.protofilament is None:
            raise ValueError(
                "This extraction pooled protofilaments; re-run with "
                "resolve_protofilaments=True to get a per-protofilament breakdown."
            )

        counts = np.bincount(self.protofilament)
        per_protofilament = self.n_predicted / len(counts)

        return {int(k): float(c / per_protofilament) for k, c in enumerate(counts)}

    def mean_score_by_protofilament(self) -> dict[int, float]:
        """Mean score of the occupied sites of each protofilament.

        Raises
        ------
        ValueError
            If the extraction pooled protofilaments.
        """
        if self.protofilament is None:
            raise ValueError(
                "This extraction pooled protofilaments; re-run with "
                "resolve_protofilaments=True to get a per-protofilament breakdown."
            )

        return {
            int(k): float(self.score[self.protofilament == k].mean())
            for k in np.unique(self.protofilament)
        }


# pylint: disable=too-many-locals
def extract_lattice_sites(
    detections,
    geometry: TemplateLatticeGeometry,
    pixel_size_angstrom: float,
    bootstrap_score_threshold: float,
    initial_rise_angstrom: float | None = None,
    score_column: str = "z_score",
    resolve_protofilaments: bool = False,
    n_iterations: int = 3,
    xy_radius_px: float = 12.0,
    angular_radius_deg: float = 15.0,
) -> LatticeSites:
    """Fit a lattice from the strongest detections, then read every site it predicts.

    Thresholding globally forces a bad trade: cut high and real sites are lost, cut low
    and noise corrupts the axis fit, which biases everything measured along it. This
    avoids the trade by using the model to say *where* to look. A rough lattice is
    bootstrapped from the strongest peaks only, and then the best detection within half
    a repeat of each predicted site is taken, however weak. Noise away from the lattice
    is never examined, so purity and completeness stop competing.

    The obvious worry is circularity -- predicting sites from a repeat and then
    measuring the repeat from them. Two things guard against it: the search window is a
    full half-repeat, so a site's position is free to land anywhere, and
    ``axial_deviation_angstrom`` records how far detections actually sat from their
    predictions. Deviations far below the window mean the data, not the model, set the
    positions. Confirm by re-running from different bootstrap thresholds and starting
    repeats; the answer should not move.

    Parameters
    ----------
    detections : pd.DataFrame
        Detections with ``x``, ``y``, ``phi``, ``theta``, ``psi`` and ``score_column``.
        Pass everything available -- the point is to reach below any sensible threshold.
    geometry : TemplateLatticeGeometry
        Template calibration.
    pixel_size_angstrom : float
        Pixel size of the micrograph.
    bootstrap_score_threshold : float
        Score cut for the initial fit only. Set it high; this should see only confident
        detections, and the extraction reaches below it afterwards.
    initial_rise_angstrom : float | None
        Starting repeat. Defaults to the template's.
    score_column : str
        Column ranked when choosing between detections at one site.
    resolve_protofilaments : bool
        Assign sites to individual protofilaments as well as axial positions. Only
        meaningful for a template whose azimuth is well determined: a complete ring is
        N-fold pseudo-symmetric, so its matched ``phi`` does not identify a
        protofilament and the breakdown will be meaningless.
    n_iterations : int
        Refit-and-reassign rounds. Convergence is typically immediate.
    xy_radius_px, angular_radius_deg
        Suppression radii for the bootstrap peak finding.

    Returns
    -------
    LatticeSites

    Raises
    ------
    ValueError
        If too few detections survive the bootstrap threshold to fit an axis.
    """
    # Local import: correlation_peaks is a sibling and does not import this module.
    from leopard_em.analysis.correlation_peaks import find_peaks_orientation_aware

    rise = (
        float(geometry.rise_angstrom)
        if initial_rise_angstrom is None
        else float(initial_rise_angstrom)
    )
    spacing = geometry.protofilament_angular_spacing_deg

    bootstrap = find_peaks_orientation_aware(
        detections,
        xy_radius_px=xy_radius_px,
        angular_radius_deg=angular_radius_deg,
        score_column=score_column,
        score_threshold=bootstrap_score_threshold,
    )
    if len(bootstrap) < 3:
        raise ValueError(
            f"Only {len(bootstrap)} detections exceed the bootstrap threshold of "
            f"{bootstrap_score_threshold}; too few to fit an axis. Lower it."
        )

    boot_weights = bootstrap[score_column].to_numpy(dtype=np.float64)
    boot_phi = bootstrap["phi"].to_numpy(dtype=np.float64)
    axis = fit_filament_axis(
        axis_points_from_peaks(bootstrap, geometry, pixel_size_angstrom),
        boot_weights,
        filament_direction_from_angles(
            boot_phi,
            bootstrap["theta"].to_numpy(dtype=np.float64),
            bootstrap["psi"].to_numpy(dtype=np.float64),
        ),
    )

    all_points = axis_points_from_peaks(detections, geometry, pixel_size_angstrom)
    # Read the angles through the same column convention axis_points_from_peaks uses, so
    # a table carrying refined_* columns does not end up with refined axis points and
    # unrefined angles.
    angle_columns = geometry_columns(detections)
    all_phi = detections[angle_columns["phi"]].to_numpy(dtype=np.float64)
    all_theta = detections[angle_columns["theta"]].to_numpy(dtype=np.float64)
    scores = detections[score_column].to_numpy(dtype=np.float64)
    boot_points = axis_points_from_peaks(bootstrap, geometry, pixel_size_angstrom)

    selected = np.empty(0, dtype=np.int64)
    axial = np.empty(0, dtype=np.int64)
    protofilament = None
    deviation = np.empty(0)
    n_predicted = 0

    for _ in range(max(n_iterations, 1)):
        along, _ = filament_coordinates(all_points, axis)
        unwrapped = unwrap_helical_axial_coordinate(
            along * pixel_size_angstrom, all_phi, geometry
        )
        boot_along, _ = filament_coordinates(boot_points, axis)
        boot_unwrapped = unwrap_helical_axial_coordinate(
            boot_along * pixel_size_angstrom, boot_phi, geometry
        )

        phase = (
            np.angle((boot_weights * np.exp(2j * np.pi * boot_unwrapped / rise)).sum())
            / (2 * np.pi)
            * rise
        )
        fractional = (unwrapped - phase) / rise
        axial_all = np.round(fractional).astype(np.int64)

        low = int(np.floor(fractional.min()))
        high = int(np.ceil(fractional.max()))
        n_axial = high - low + 1

        if resolve_protofilaments:
            pf_all = np.round(all_phi / spacing).astype(np.int64) % (
                geometry.n_protofilaments
            )
            key = (axial_all - low) * geometry.n_protofilaments + pf_all
            n_predicted = n_axial * geometry.n_protofilaments
        else:
            pf_all = None
            key = axial_all - low
            n_predicted = n_axial

        # Best-scoring detection at each site: sort by site then descending score and
        # keep the first of each run.
        order = np.lexsort((-scores, key))
        is_first = np.concatenate(([True], key[order][1:] != key[order][:-1]))
        selected = order[is_first]

        axial = axial_all[selected]
        protofilament = None if pf_all is None else pf_all[selected]
        deviation = (fractional[selected] - axial) * rise

        rise = estimate_lattice_rise(
            unwrapped[selected] / pixel_size_angstrom,
            pixel_size_angstrom,
            rise,
            scores[selected],
            # ALWAYS correct the out-of-plane tilt. A filament at theta is projected
            # shorter by sin(theta), so without this the returned repeat is a projected
            # one, not an axial one, and it reads systematically LOW. The size of it is
            # easy to underestimate: 1/sin(mean theta) is not the correction applied --
            # estimate_lattice_rise divides by the score-weighted mean of sin(theta),
            # and because sin is concave that is the larger number. On a real filament
            # spanning theta 79-99 degrees with a mean of 88.8, the naive figure is
            # +0.02% while the correction actually applied is +0.45% -- twenty times
            # bigger, and comparable to the chord-vs-arc bias on a bent tube. Against a
            # 2.5% expanded-versus-compacted separation that is not negligible.
            theta_deg=all_theta[selected],
        ).rise_angstrom

    return LatticeSites(
        detection_index=selected,
        axial_index=axial,
        protofilament=protofilament,
        score=scores[selected],
        axial_deviation_angstrom=deviation,
        rise_angstrom=rise,
        axis=axis,
        n_predicted=n_predicted,
    )


@dataclass(frozen=True)
class LatticeModelScore:
    """How well one candidate lattice model accounts for a fixed set of detections.

    Produced by :func:`score_lattice_model`. The comparison is only meaningful between
    scores computed with the same detections, the same windows and the same threshold.

    Attributes
    ----------
    n_protofilaments, rise_angstrom : int, float
        The model that was scored.
    n_predicted, n_found : int
        Sites the model predicts within the observed span, and how many were filled.
    occupancy : float
        ``n_found / n_predicted``. The headline statistic.
    mean_score, median_score : float
        Over the filled sites only. ``nan`` when nothing was found.
    score_threshold : float
        The cutoff actually applied, so two scores can be checked for comparability.
    phase_offset_angstrom : float
        Global axial shift chosen by the refinement. A value near +/- rise/2 means the
        model preferred a different registration than the anchor implied -- treat the
        result with suspicion.
    axial_deviation_angstrom : np.ndarray
        Signed distance from each filled site to its prediction. Small values relative
        to the window mean the detections, not the model, set the positions.
    site_protofilament, site_axial_index : np.ndarray
        Indices identifying every predicted site, filled or not.
    detection_index : np.ndarray
        Row of the source table filling each site, or -1 where the site was empty.
    """

    n_protofilaments: int
    rise_angstrom: float
    n_predicted: int
    n_found: int
    occupancy: float
    mean_score: float
    median_score: float
    score_threshold: float
    phase_offset_angstrom: float
    axial_deviation_angstrom: np.ndarray
    site_protofilament: np.ndarray
    site_axial_index: np.ndarray
    detection_index: np.ndarray

    def occupancy_by_protofilament(self) -> dict[int, float]:
        """Filled fraction per protofilament, to expose an unevenly sampled tube."""
        filled = self.detection_index >= 0
        return {
            int(p): float(filled[self.site_protofilament == p].mean())
            for p in np.unique(self.site_protofilament)
        }


def _predict_lattice_sites(
    geometry: TemplateLatticeGeometry,
    axis: FilamentAxis,
    pixel_size_angstrom: float,
    n_protofilaments: int,
    rise_angstrom: float,
    anchor_along_angstrom: float,
    anchor_phi_deg: float,
    theta_deg: float,
    psi_deg: float,
    axial_index_range: tuple[int, int],
    along_span_angstrom: tuple[float, float] | None = None,
    foreshortening: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predicted (x, y) in pixels and azimuth for every site the model implies.

    Inverts the two mappings the forward analysis uses, in order:

    * ``unwrap_helical_axial_coordinate`` places sites at a constant *unwrapped*
      coordinate, so a site's along-axis coordinate runs back as
      ``s = s0 + m * rise - (turn_shift / 360) * (phi - phi0)``, where ``turn_shift`` is
      the template's measured axial shift per turn scaled to the model's repeat;
    * ``axis_points_from_peaks`` maps a detection to its axis point by *adding*
      ``(R^T p)[:2]``, so predicting the detection from the axis point *subtracts* it.

    The template point ``p`` is the template's own axis offset and is deliberately held
    fixed across models: it is a property of the template that produced the detections,
    not of the model under test. This is why the test cannot see tube radius.
    """
    spacing = 360.0 / n_protofilaments
    # Preserve the template's measured axial shift per turn, scaled to the model's
    # repeat, rather than re-deriving it from an assumed subunit rise -- the geometry
    # stores a repeat that may be a dimer, so n_start * rise / N would be out by the
    # number of subunits per repeat. Note per_degree = turn_shift / 360 is therefore
    # independent of N, which is why this test cannot separate N by axial position.
    turn_shift = geometry.axial_shift_per_turn_angstrom * (
        rise_angstrom / geometry.rise_angstrom
    )
    per_degree = turn_shift / 360.0

    protofilaments = np.arange(n_protofilaments)
    phi = (anchor_phi_deg + protofilaments * spacing) % 360.0
    # Azimuth relative to the anchor, wrapped into [0, 360) rather than +/-180. The
    # helical offset accumulates monotonically the whole way round the tube -- going
    # once round genuinely rises by the turn shift -- so a signed difference would flip
    # the offset's sign past 180 degrees and displace half the protofilaments.
    phi_relative = protofilaments * spacing

    axial = np.arange(axial_index_range[0], axial_index_range[1] + 1)

    pf_grid, m_grid = np.meshgrid(protofilaments, axial, indexing="ij")
    phi_grid = phi[pf_grid.ravel()]
    along = (
        anchor_along_angstrom
        + m_grid.ravel() * rise_angstrom
        - per_degree * phi_relative[pf_grid.ravel()]
    )

    offsets = image_offset_from_template_point(
        np.array([*geometry.axis_offset_angstrom, 0.0]),
        phi_grid,
        np.full(phi_grid.shape, float(theta_deg)),
        np.full(phi_grid.shape, float(psi_deg)),
    )
    axis_points = (
        axis.origin[None, :]
        + (along[:, None] * foreshortening / pixel_size_angstrom)
        * axis.direction[None, :]
    )
    predicted_xy = axis_points - offsets / pixel_size_angstrom

    site_pf, site_m = pf_grid.ravel(), m_grid.ravel()
    if along_span_angstrom is not None:
        # Sites beyond the ends of the observed filament are not real predictions.
        # Keeping them would depress occupancy by an amount that depends on how far
        # the model's helical offset spreads sites, which differs between models.
        inside = (along >= along_span_angstrom[0]) & (along <= along_span_angstrom[1])
        predicted_xy, phi_grid = predicted_xy[inside], phi_grid[inside]
        site_pf, site_m = site_pf[inside], site_m[inside]

    return predicted_xy, phi_grid, site_pf, site_m


def _match_sites_to_detections(
    predicted_xy: np.ndarray,
    predicted_phi: np.ndarray,
    phi: np.ndarray,
    scores: np.ndarray,
    tree,
    rows: np.ndarray,
    xy_radius_px: float,
    angular_radius_deg: float,
) -> np.ndarray:
    """Best detection for each predicted site, one detection to at most one site.

    Returns an index per site, or -1 where nothing qualified. The assignment is greedy
    by descending score: without it a single strong detection can satisfy several
    neighbouring predictions at once, which inflates the occupancy of models whose
    sites are packed more closely and would systematically favour large ``N`` and the
    compacted repeat.
    """
    # pylint: disable=import-outside-toplevel
    from leopard_em.analysis.correlation_peaks import _circular_distance_deg

    assigned = np.full(len(predicted_xy), -1, dtype=np.int64)
    if rows.size == 0:
        return assigned

    candidates: list[tuple[float, int, int]] = []
    for site, (xy, phi_pred) in enumerate(zip(predicted_xy, predicted_phi)):
        for local in tree.query_ball_point(xy, xy_radius_px):
            row = int(rows[local])
            if _circular_distance_deg(phi[row], phi_pred) <= angular_radius_deg:
                candidates.append((float(scores[row]), site, row))

    taken_sites: set[int] = set()
    taken_rows: set[int] = set()
    for _, site, row in sorted(candidates, key=lambda c: -c[0]):
        if site in taken_sites or row in taken_rows:
            continue
        assigned[site] = row
        taken_sites.add(site)
        taken_rows.add(row)

    return assigned


def score_lattice_model(
    detections,
    geometry: TemplateLatticeGeometry,
    axis: FilamentAxis,
    pixel_size_angstrom: float,
    n_protofilaments: int,
    rise_angstrom: float,
    score_threshold: float,
    anchor_index: int | None = None,
    xy_radius_px: float = 12.0,
    angular_radius_deg: float | None = None,
    theta_tolerance_deg: float = 3.5,
    psi_tolerance_deg: float = 2.5,
    score_column: str = "z_score",
    refine_phase: bool = True,
    phase_step_angstrom: float = 0.5,
    azimuth_offset_deg: float = 0.0,
    detection_index_cache: dict | None = None,
) -> LatticeModelScore:
    """Ask how much of a candidate lattice a fixed set of detections actually supports.

    Complementary to competing whole templates. That varies the template and so carries
    template-quality and template-size confounds; this holds the detections completely
    fixed and varies only the *model*, predicting where every subunit should appear in
    ``(x, y)`` and in azimuth and asking whether a detection is there.

    Two properties make the comparison interpretable. Sites a model predicts but that
    hold nothing are counted, so occupancy is a real statistic rather than the near
    tautology it becomes when sites are defined by the detections themselves. And a
    detection can fill at most one site, so a model cannot win by predicting sites so
    densely that one strong detection satisfies several.

    What it can and cannot see
    --------------------------
    The template's axis offset is held fixed across models -- it belongs to the template
    that produced the detections, not to the model being tested -- so **tube radius is
    not testable here**. Competing whole templates is the radius measurement.

    For a closed lattice the unwrap coefficient ``n_start * rise / 360`` is independent
    of ``n_protofilaments``, so the along-axis positions for a given azimuth do not
    depend on ``N`` either. The protofilament axis of the comparison therefore rests on
    *azimuth quantisation* alone -- which azimuths are occupied -- and re-expresses the
    same information as the azimuth harmonic, in a stricter form. The repeat axis is
    much stronger, because a wrong repeat accumulates along the whole track.

    Parameters
    ----------
    detections : pd.DataFrame
        Detections with position, orientation and ``score_column``. Pass everything.
    geometry : TemplateLatticeGeometry
        Calibration of the template that produced ``detections``. Only its axis offset
        and ``n_start`` are used; its own protofilament count is not.
    axis : FilamentAxis
        Fitted filament axis, e.g. from :func:`extract_lattice_sites`.
    pixel_size_angstrom : float
        Pixel size of the micrograph.
    n_protofilaments, rise_angstrom : int, float
        The model to score.
    score_threshold : float
        A site holds nothing unless a detection exceeds this. Use the same value for
        every model being compared, or models predicting fewer sites get an easier cut.
    anchor_index : int | None
        Row anchoring the lattice phase and supplying theta/psi. Defaults to the
        best-scoring detection.
    xy_radius_px : float
        Positional tolerance about each prediction.
    angular_radius_deg : float | None
        Azimuthal tolerance. Defaults to ``0.4 * 360 / n_protofilaments``, which stays
        below half the protofilament spacing; a wider gate makes neighbouring models
        indistinguishable. Do not pass ``suggested_angular_radius_deg``, which is sized
        for peak suppression, not for separating models.
    theta_tolerance_deg, psi_tolerance_deg : float
        Tolerances about the anchor's tilt and in-plane angle, defaulting to one search
        grid step. The psi tolerance is what excludes the opposite polarity pole.
    score_column : str
        Column ranked when choosing between detections.
    refine_phase : bool
        Scan a single global axial offset and keep the best. One parameter per model,
        so no model is penalised for the arbitrary choice of anchor.
    phase_step_angstrom : float
        Step of that scan.
    azimuth_offset_deg : float
        Rotate the whole predicted azimuth grid. Randomising this is how to build a
        null: occupancy has a large floor that depends on the model, because a coarser
        azimuth grid is more likely to find something in window whatever the truth, so
        raw occupancy cannot be compared between models without one.
    detection_index_cache : dict | None
        Pass the same empty dict to every call that shares detections, anchor and
        thresholds, and the eligibility mask and spatial index are built once instead of
        per call. Comparing eight models against a null of many draws otherwise rebuilds
        a million-point tree hundreds of times, which dominates the runtime.

    Returns
    -------
    LatticeModelScore

    Raises
    ------
    ValueError
        If the angular tolerance is too wide to separate adjacent protofilaments.
    """
    spacing = 360.0 / n_protofilaments
    if angular_radius_deg is None:
        angular_radius_deg = 0.4 * spacing
    if angular_radius_deg >= 0.5 * spacing:
        raise ValueError(
            f"angular_radius_deg={angular_radius_deg:.2f} is at least half the "
            f"protofilament spacing ({spacing:.2f} deg) for N={n_protofilaments}; "
            "neighbouring azimuths would overlap and models become indistinguishable."
        )

    columns = geometry_columns(detections)
    positions = np.column_stack(
        [
            detections[columns["x"]].to_numpy(dtype=np.float64),
            detections[columns["y"]].to_numpy(dtype=np.float64),
        ]
    )
    phi = detections[columns["phi"]].to_numpy(dtype=np.float64)
    theta = detections[columns["theta"]].to_numpy(dtype=np.float64)
    psi = detections[columns["psi"]].to_numpy(dtype=np.float64)
    scores = detections[score_column].to_numpy(dtype=np.float64)

    if anchor_index is None:
        anchor_index = int(np.argmax(scores))

    along_px, _ = filament_coordinates(positions, axis)
    # A filament tilted out of the image plane is projected shorter by sin(theta), so
    # divide it out and work in true three-dimensional Angstroms. The repeat difference
    # under test is 2.7%, and sin(theta) reaches that by theta = 77 degrees.
    foreshortening = float(np.sin(np.radians(float(theta[anchor_index]))))
    if foreshortening < 0.1:
        raise ValueError(
            f"theta = {theta[anchor_index]:.1f} deg is too close to end-on for a "
            "projected lattice measurement."
        )
    along = along_px * pixel_size_angstrom / foreshortening
    anchor_along = float(along[anchor_index])

    # The site range must be taken in the UNWRAPPED coordinate. Protofilaments are
    # offset helically, so the raw along-axis span is wider than the lattice span by up
    # to a turn's worth -- using it would invent sites beyond the filament's ends and
    # depress every model's occupancy by a constant that depends on the model.
    per_degree = (
        geometry.axial_shift_per_turn_angstrom
        * (rise_angstrom / geometry.rise_angstrom)
        / 360.0
    )
    phi_relative = (phi - float(phi[anchor_index])) % 360.0
    unwrapped = along + per_degree * phi_relative
    fractional = (unwrapped - anchor_along) / rise_angstrom
    index_range = (int(np.floor(fractional.min())), int(np.ceil(fractional.max())))

    offsets = [0.0]
    if refine_phase:
        reach = 0.5 * rise_angstrom
        offsets = list(np.arange(-reach, reach + phase_step_angstrom, phase_step_angstrom))

    # Eligibility and the spatial index do not depend on the phase, so build them once.
    # Rebuilding a million-point tree inside the offset scan dominates the runtime.
    # pylint: disable=import-outside-toplevel
    from scipy.spatial import cKDTree

    from leopard_em.analysis.correlation_peaks import _circular_distance_deg

    cache_key = (
        id(detections),
        int(anchor_index),
        float(score_threshold),
        float(theta_tolerance_deg),
        float(psi_tolerance_deg),
        score_column,
    )
    if detection_index_cache is not None and cache_key in detection_index_cache:
        tree, rows = detection_index_cache[cache_key]
    else:
        eligible = (
            (scores > score_threshold)
            & (np.abs(theta - float(theta[anchor_index])) <= theta_tolerance_deg)
            # Excludes the opposite polarity pole, ~180 degrees away in psi. Without
            # this, reversed detections fill sites in every model alike and wash the
            # test out.
            & (
                _circular_distance_deg(psi, float(psi[anchor_index]))
                <= psi_tolerance_deg
            )
        )
        rows = np.flatnonzero(eligible)
        tree = cKDTree(positions[rows]) if rows.size else None
        if detection_index_cache is not None:
            detection_index_cache[cache_key] = (tree, rows)

    best = None
    for offset in offsets:
        predicted_xy, predicted_phi, site_pf, site_m = _predict_lattice_sites(
            geometry,
            axis,
            pixel_size_angstrom,
            n_protofilaments,
            rise_angstrom,
            anchor_along + offset,
            float(phi[anchor_index]) + azimuth_offset_deg,
            float(theta[anchor_index]),
            float(psi[anchor_index]),
            index_range,
            (float(along.min()), float(along.max())),
            foreshortening,
        )
        assigned = _match_sites_to_detections(
            predicted_xy,
            predicted_phi,
            phi,
            scores,
            tree,
            rows,
            xy_radius_px,
            angular_radius_deg,
        )
        n_found = int((assigned >= 0).sum())
        # Tie-break on the residuals. The window is wider than the phase step, so a
        # range of offsets fills the same number of sites; without this the scan keeps
        # whichever edge of that plateau it reached first, leaving a constant axial
        # offset in every deviation and making the anti-circularity check unreadable.
        filled_now = assigned >= 0
        if filled_now.any():
            residual = (
                along[assigned[filled_now]]
                - filament_coordinates(predicted_xy[filled_now], axis)[0]
                * pixel_size_angstrom
                / foreshortening
            )
            penalty = float(np.sum(residual**2))
        else:
            penalty = float("inf")
        key = (n_found, -penalty)
        if best is None or key > best[0]:
            best = (key, assigned, site_pf, site_m, predicted_xy, float(offset))

    assert best is not None
    (n_found, _), assigned, site_pf, site_m, predicted_xy, offset = best

    filled = assigned >= 0
    found_scores = scores[assigned[filled]] if n_found else np.empty(0)
    deviation = np.zeros(len(assigned))
    if n_found:
        deviation[filled] = (
            along[assigned[filled]]
            - filament_coordinates(predicted_xy[filled], axis)[0]
            * pixel_size_angstrom
            / foreshortening
        )

    return LatticeModelScore(
        n_protofilaments=n_protofilaments,
        rise_angstrom=rise_angstrom,
        n_predicted=len(assigned),
        n_found=n_found,
        occupancy=n_found / len(assigned) if len(assigned) else 0.0,
        mean_score=float(found_scores.mean()) if n_found else float("nan"),
        median_score=float(np.median(found_scores)) if n_found else float("nan"),
        score_threshold=score_threshold,
        phase_offset_angstrom=offset,
        axial_deviation_angstrom=deviation[filled],
        site_protofilament=site_pf,
        site_axial_index=site_m,
        detection_index=assigned,
    )


# ------------------------------------------------------------------------------------
# Which filament a detection belongs to -- decided from the detections, not the drawing
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FilamentAssignment:
    """Which filament each detection belongs to, and the curves that decided it.

    Attributes
    ----------
    labels : np.ndarray
        ``(n,)`` filament label per detection, using the seeds' labels; 0 where no
        filament accepted it (clutter between tubes, off-axis false positives).
    axis_points_px : np.ndarray
        ``(n, 2)`` each detection's own predicted axis point, (x, y) pixels, in the
        frame of its position columns.
    transverse_angstrom : np.ndarray
        ``(n,)`` distance from that axis point to the assigned filament's curve. NaN
        where unassigned.
    tangent_mismatch_deg : np.ndarray
        ``(n,)`` angle between the detection's own in-plane direction and the curve's
        local tangent, folded into [0, 90]. NaN where unassigned.
    curves_px : dict[int, np.ndarray]
        Fitted centreline per filament, ``(m, 2)`` (x, y) pixels at about 1 px
        spacing, spanning the filament's own assigned axis points. Fitted to the
        detections alone; a filament left with too few detections has no curve.
    n_iterations : int
        Refit-and-reassign rounds actually run.
    converged : bool
        Whether the labels stopped changing before the iteration limit.
    """

    labels: np.ndarray
    axis_points_px: np.ndarray
    transverse_angstrom: np.ndarray
    tangent_mismatch_deg: np.ndarray
    curves_px: dict[int, np.ndarray]
    n_iterations: int
    converged: bool

    def counts(self) -> dict[int, int]:
        """Detections per filament label, 0 included."""
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}


def _densify_polyline(points_xy: np.ndarray, spacing_px: float = 1.0) -> np.ndarray:
    """Resample a polyline at roughly ``spacing_px`` along its length."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return points
    steps = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] <= 0:
        return points[:1]
    samples = np.linspace(0.0, arc[-1], max(int(np.ceil(arc[-1] / spacing_px)) + 1, 2))
    return np.column_stack(
        [np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1])]
    )


def _curve_tangents(curve_xy: np.ndarray) -> np.ndarray:
    """Unit tangent at every sample of a densely sampled curve."""
    if len(curve_xy) < 2:
        return np.tile([1.0, 0.0], (len(curve_xy), 1))
    tangent = np.gradient(curve_xy, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.where(norm > 0, norm, 1.0)


def _extend_curve(curve_xy: np.ndarray, extend_px: float) -> np.ndarray:
    """Continue a curve straight past both ends along its end tangents."""
    if extend_px <= 0 or len(curve_xy) < 2:
        return curve_xy
    tangents = _curve_tangents(curve_xy)
    steps = np.arange(1.0, extend_px + 1.0)[:, None]
    head = curve_xy[0] - steps[::-1] * tangents[0]
    tail = curve_xy[-1] + steps * tangents[-1]
    return np.vstack([head, curve_xy, tail])


def _binned_along(t, points_xy, weights, bin_px):
    """Score-weighted mean position in bins of the curve parameter ``t``.

    Also returns how many detections fell in each bin, which the knot placement uses:
    a bin's weighted mean is only as good as the number of detections behind it.
    """
    edges = np.arange(t.min(), t.max() + bin_px, bin_px)
    which = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 1)
    total = np.bincount(which, weights=weights, minlength=len(edges))
    count = np.bincount(which, minlength=len(edges))
    keep = total > 0
    mean_t = np.bincount(which, weights=weights * t, minlength=len(edges))[keep]
    mean_x = np.bincount(which, weights=weights * points_xy[:, 0], minlength=len(edges))
    mean_y = np.bincount(which, weights=weights * points_xy[:, 1], minlength=len(edges))
    total = total[keep]
    return (mean_t / total, mean_x[keep] / total, mean_y[keep] / total, total,
            count[keep])


_MIN_DETECTIONS_PER_KNOT = 30


def _spline_xy(t_bins, x_bins, y_bins, w_bins, knot_px, n_bins=None):
    """Least-squares cubic splines x(t), y(t) with knots no closer than ``knot_px``.

    A knot is kept only when the stretch behind it holds enough data: at least
    ``degree + 1`` occupied bins (Schoenberg-Whitney, which a gap in coverage readily
    breaks) and at least ``_MIN_DETECTIONS_PER_KNOT`` detections. The second condition
    is what keeps a sparse stretch stiff -- measured on the bundle synthetic, a stretch
    with a handful of detections let the curve stray 23-29 A where dense stretches sat
    within 4 A. A stubborn fit falls back to fewer knots and finally a polynomial.
    """
    from scipy.interpolate import LSQUnivariateSpline  # pylint: disable=import-outside-toplevel

    degree = int(min(3, len(t_bins) - 1))
    if degree < 1:
        return lambda t: np.full_like(np.asarray(t, float), x_bins[0]), (
            lambda t: np.full_like(np.asarray(t, float), y_bins[0])
        )

    counts = np.ones_like(t_bins) if n_bins is None else np.asarray(n_bins)
    candidates = np.arange(t_bins[0] + knot_px, t_bins[-1] - 0.5 * knot_px, knot_px)
    knots: list[float] = []
    previous = t_bins[0]
    for knot in candidates:
        behind = (t_bins > previous) & (t_bins <= knot)
        ahead = t_bins > knot
        if (
            np.count_nonzero(behind) >= degree + 1
            and counts[behind].sum() >= _MIN_DETECTIONS_PER_KNOT
            and counts[ahead].sum() >= _MIN_DETECTIONS_PER_KNOT
        ):
            knots.append(float(knot))
            previous = knot
    # scipy weights RESIDUALS (it minimises sum((w * r)^2)), and a bin mean over total
    # weight W has standard error ~ 1/sqrt(W) -- so the residual weight is sqrt(W).
    # Passing W itself squares the imbalance: a bin of 3000 detections then outweighs
    # one of 30 by 10^4 instead of 10^2, and on the bundle synthetic that dragged the
    # curve ~10 A off its sparse stretches while those stretches' detections were
    # unbiased to within 3 A.
    residual_weight = np.sqrt(w_bins)
    while True:
        try:
            spline_x = LSQUnivariateSpline(
                t_bins, x_bins, knots, w=residual_weight, k=degree
            )
            spline_y = LSQUnivariateSpline(
                t_bins, y_bins, knots, w=residual_weight, k=degree
            )
            return spline_x, spline_y
        except ValueError:
            if not knots:
                break
            knots = knots[::2] if len(knots) > 1 else []
    coeff_x = np.polyfit(t_bins, x_bins, degree, w=np.sqrt(w_bins))
    coeff_y = np.polyfit(t_bins, y_bins, degree, w=np.sqrt(w_bins))
    return np.poly1d(coeff_x), np.poly1d(coeff_y)


def _fit_filament_curve(
    points_xy: np.ndarray, weights: np.ndarray, knot_px: float
) -> np.ndarray:
    """Smooth centreline through one filament's axis points, sampled at ~1 px.

    Two passes. The first parameterises by position along the points' principal
    axis, which is enough to get the shape; the second re-parameterises by arc length
    along that first curve, so a strongly bent tube is not squashed onto its chord.
    Knot spacing keeps the curve free of structure near the lattice period.
    """
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    points = np.asarray(points_xy, dtype=np.float64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    centre = np.average(points, axis=0, weights=w)
    offset = points - centre
    covariance = (offset * w[:, None]).T @ offset / w.sum()
    principal = np.linalg.eigh(covariance)[1][:, -1]
    t = offset @ principal

    curve = points[np.argsort(t)]
    for _ in range(2):
        t_bins, x_bins, y_bins, w_bins, n_bins = _binned_along(
            t, points, w, max(knot_px / 8.0, 1.0)
        )
        if len(t_bins) < 2:
            return points[:1].copy()
        spline_x, spline_y = _spline_xy(
            t_bins, x_bins, y_bins, w_bins, knot_px, n_bins=n_bins
        )
        fine = np.linspace(t_bins[0], t_bins[-1], max(int(t_bins[-1] - t_bins[0]) * 2, 2))
        curve = _densify_polyline(np.column_stack([spline_x(fine), spline_y(fine)]))
        # Re-parameterise every point by arc length along this curve.
        _, nearest = cKDTree(curve).query(points)
        t = nearest.astype(np.float64)
    return curve


# CPU threads for the nearest-curve search (a shared machine: a few, not all of them).
_QUERY_WORKERS = 8


def _assign_to_curves(axis_xy, direction_deg, curves, max_transverse_px, max_tangent_deg):
    """Nearest acceptable curve per detection, by position AND tangent agreement."""
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    n = len(axis_xy)
    labels = np.zeros(n, dtype=np.int64)
    best = np.full(n, np.inf)
    transverse = np.full(n, np.nan)
    mismatch = np.full(n, np.nan)
    for label, curve in curves.items():
        if len(curve) < 2:
            continue
        # Only detections within max_transverse_px of the curve can join it, so the
        # search stops just past that distance (a little slack, so the <= test below
        # decides at the edge, as before). Same result, and on a 30M-detection
        # table (a bundle micrograph) ~50x faster: ~6.5 min per curve per pass before.
        distance, index = cKDTree(curve).query(
            axis_xy,
            distance_upper_bound=max_transverse_px * (1.0 + 1e-6),
            workers=_QUERY_WORKERS,
        )
        near = np.flatnonzero(np.isfinite(distance))
        distance = distance[near]
        tangent = _curve_tangents(curve)[index[near]]
        tangent_deg = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
        # Polarity-blind: a filament's tangent and a detection's direction agree up
        # to 180 degrees, so fold the difference into [0, 90].
        angle = np.abs(((direction_deg[near] - tangent_deg + 90.0) % 180.0) - 90.0)
        ok = (distance <= max_transverse_px) & (angle <= max_tangent_deg)
        cost = (distance / max_transverse_px) ** 2 + (angle / max_tangent_deg) ** 2
        better = ok & (cost < best[near])
        chosen = near[better]
        best[chosen] = cost[better]
        labels[chosen] = int(label)
        transverse[chosen] = distance[better]
        mismatch[chosen] = angle[better]
    return labels, transverse, mismatch


# pylint: disable=too-many-arguments,too-many-locals
def assign_detections_to_filaments(
    detections,
    geometry: TemplateLatticeGeometry,
    pixel_size_angstrom: float,
    seed_curves_px: dict[int, np.ndarray] | None = None,
    initial_labels: np.ndarray | None = None,
    max_transverse_angstrom: float = 60.0,
    max_tangent_deg: float = 10.0,
    seed_transverse_angstrom: float = 150.0,
    seed_tangent_deg: float = 25.0,
    knot_spacing_angstrom: float = 500.0,
    extend_angstrom: float = 500.0,
    n_iterations: int = 10,
    score_column: str | None = "z_score",
    position_columns: tuple[str, str] | None = None,
    prefer_refined: bool = True,
    min_detections: int = 8,
    settled_fraction: float = 1e-4,
) -> FilamentAssignment:
    """Assign each detection to a filament from its OWN predicted axis point and tangent.

    Deciding membership by which drawn band a detection's box centre falls in has two
    faults. It takes geometry from the drawing rather than the data. And the box
    centre is the wrong point: a patch template's box centre sits 85.8 A off the axis
    and sweeps a circle as ``phi`` changes, so in a tight bundle it can land in the
    neighbouring tube's band. Every detection instead predicts a point ON ITS AXIS
    (:func:`axis_points_from_peaks`, offset undone) and a local in-plane direction
    (from ``theta`` and ``psi``). This uses both:

    1. Seed membership -- from ``initial_labels``, or by matching to
       ``seed_curves_px`` (e.g. the drawn paths) under the loose
       ``seed_transverse_angstrom`` gate. Seeds only decide who starts where.
    2. Fit each filament's centreline to its members' axis points alone.
    3. Reassign every detection to the curve its axis point is nearest to and whose
       local tangent agrees with its direction; reject anything outside both gates.
    4. Repeat 2-3 until the labels stop changing.

    Position separates parallel filaments; the tangent separates a crossing, where
    position alone is ambiguous but the two directions differ by far more than a
    detection's angular error.

    Parameters
    ----------
    detections : pd.DataFrame
        Detections with positions and ``phi``/``theta``/``psi``.
    geometry : TemplateLatticeGeometry
        The template's axis offset (zero for a ring, non-zero for a patch).
    pixel_size_angstrom : float
        Micrograph pixel size.
    seed_curves_px : dict[int, np.ndarray], optional
        ``label -> (m, 2)`` polylines, (x, y) pixels in the SAME frame as the position
        columns (table positions are the correlation-map frame: shift a drawn path by
        minus half the template box). Used only to seed membership.
    initial_labels : np.ndarray, optional
        ``(n,)`` starting labels, 0 = unknown. Takes precedence over the curves.
    max_transverse_angstrom : float
        Largest distance from a detection's axis point to its filament's curve.
    max_tangent_deg : float
        Largest disagreement between a detection's direction and the curve tangent.
    seed_transverse_angstrom : float
        The looser distance gate for matching to seed curves, which carry drawing
        error.
    seed_tangent_deg : float
        The looser tangent gate for the same step. A drawing's tangent is the least
        trustworthy thing about it -- a wobble of a few tens of Angstroms over a
        couple of hundred nanometres tilts it by several degrees -- so seeding must
        not demand the agreement the data-fitted curves are later held to.
    knot_spacing_angstrom : float
        Minimum spacing of spline knots, keeping the curve smooth on the scale of
        the lattice.
    extend_angstrom : float
        How far each curve is continued straight past its ends when assigning, so a
        filament can grow into detections just beyond its current extent.
    n_iterations : int
        Maximum refit-and-reassign rounds.
    score_column : str or None
        Column weighting each detection in the curve fit; None for equal weights.
    position_columns : tuple[str, str], optional
        Override the (x, y) position columns.
    prefer_refined : bool
        Read ``refined_`` columns when present.
    min_detections : int
        A filament with fewer members than this gets no curve, and its detections are
        left unassigned rather than kept on the seed's geometry.
    settled_fraction : float
        Labels count as settled once fewer than this fraction change in a round. A few
        weak detections exactly on a boundary can flip back and forth indefinitely --
        on in situ 823, 105-132 of 866k (0.015%), median z 6.2 -- which is a two-state
        cycle, not a failure to converge.

    Returns
    -------
    FilamentAssignment
    """
    if seed_curves_px is None and initial_labels is None:
        raise ValueError("Give seed_curves_px or initial_labels to seed membership.")

    axis_xy = axis_points_from_peaks(
        detections,
        geometry,
        pixel_size_angstrom,
        position_columns=position_columns,
        prefer_refined=prefer_refined,
    )
    columns = geometry_columns(detections, prefer_refined)
    direction = image_offset_from_template_point(
        np.array([0.0, 0.0, 1.0]),
        detections[columns["phi"]].to_numpy(),
        detections[columns["theta"]].to_numpy(),
        detections[columns["psi"]].to_numpy(),
    )
    direction_deg = np.degrees(np.arctan2(direction[:, 1], direction[:, 0]))
    if score_column is not None and score_column in detections.columns:
        weights = detections[score_column].to_numpy(dtype=np.float64)
    else:
        weights = np.ones(len(detections))

    max_transverse_px = max_transverse_angstrom / pixel_size_angstrom
    knot_px = knot_spacing_angstrom / pixel_size_angstrom
    extend_px = extend_angstrom / pixel_size_angstrom

    if initial_labels is not None:
        labels = np.asarray(initial_labels, dtype=np.int64).copy()
        if labels.shape != (len(detections),):
            raise ValueError("initial_labels must have one entry per detection.")
        filament_ids = sorted(int(v) for v in np.unique(labels) if v > 0)
    else:
        assert seed_curves_px is not None
        filament_ids = sorted(int(k) for k in seed_curves_px)
        seeds = {
            label: _densify_polyline(np.asarray(seed_curves_px[label], dtype=np.float64))
            for label in filament_ids
        }
        labels, _, _ = _assign_to_curves(
            axis_xy,
            direction_deg,
            seeds,
            seed_transverse_angstrom / pixel_size_angstrom,
            seed_tangent_deg,
        )

    curves: dict[int, np.ndarray] = {}
    transverse = np.full(len(detections), np.nan)
    mismatch = np.full(len(detections), np.nan)
    converged = False
    iteration = 0
    for iteration in range(1, n_iterations + 1):
        curves = {}
        for label in filament_ids:
            members = labels == label
            if np.count_nonzero(members) < min_detections:
                continue
            curves[label] = _fit_filament_curve(
                axis_xy[members], weights[members], knot_px
            )
        extended = {label: _extend_curve(c, extend_px) for label, c in curves.items()}
        new_labels, transverse, mismatch = _assign_to_curves(
            axis_xy, direction_deg, extended, max_transverse_px, max_tangent_deg
        )
        changed = np.count_nonzero(new_labels != labels)
        labels = new_labels
        if changed <= settled_fraction * len(labels):
            converged = True
            break

    return FilamentAssignment(
        labels=labels,
        axis_points_px=axis_xy,
        transverse_angstrom=transverse * pixel_size_angstrom,
        tangent_mismatch_deg=mismatch,
        curves_px=curves,
        n_iterations=iteration,
        converged=converged,
    )
