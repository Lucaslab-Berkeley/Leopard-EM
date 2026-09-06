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
    "FilamentAxis",
    "LatticeRise",
    "PolarityEstimate",
    "TemplateLatticeGeometry",
    "axis_points_from_peaks",
    "estimate_lattice_rise",
    "estimate_polarity",
    "estimate_protofilament_number",
    "filament_coordinates",
    "filament_direction_from_angles",
    "fit_filament_axis",
    "geometry_columns",
    "image_offset_from_template_point",
    "lattice_sharpness",
    "rise_from_template_autocorrelation",
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
