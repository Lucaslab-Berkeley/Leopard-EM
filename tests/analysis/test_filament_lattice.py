"""Tests for filament geometry recovery, against synthetic lattices of known truth.

The synthetic generator below builds an ideal helical lattice and places peaks exactly
where the projection geometry says they should land, so every estimator can be checked
against a number that is known rather than merely plausible.
"""

import numpy as np
import pandas as pd
import pytest
import roma
import torch

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    estimate_lattice_rise,
    estimate_polarity,
    estimate_protofilament_number,
    filament_coordinates,
    filament_direction_from_angles,
    fit_filament_axis,
    geometry_columns,
    image_offset_from_template_point,
    rise_from_template_autocorrelation,
    subunit_contrast,
    template_axial_autocorrelation,
    unwrap_helical_axial_coordinate,
)

PIXEL_SIZE = 0.9194


def _rotation(phi: float, theta: float, psi: float) -> np.ndarray:
    return (
        roma.euler_to_rotmat(
            "ZYZ", torch.tensor([[phi, theta, psi]], dtype=torch.float64), degrees=True
        )[0]
        .numpy()
        .astype(np.float64)
    )


def _reference(peaks: pd.DataFrame) -> np.ndarray:
    """Signed filament direction implied by the peaks' own orientations."""
    return filament_direction_from_angles(
        peaks["phi"].to_numpy(), peaks["theta"].to_numpy(), peaks["psi"].to_numpy()
    )


def synthetic_lattice(
    geometry: TemplateLatticeGeometry,
    n_repeats: int = 12,
    psi: float = 279.0,
    theta: float = 90.0,
    origin: tuple[float, float] = (400.0, 300.0),
    pixel_size: float = PIXEL_SIZE,
    azimuth_offset: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build peaks for an ideal helical lattice.

    Returns the peaks and the exact axis point each one came from, so the recovery of
    those points can be checked directly.
    """
    axis_point_template = np.array([*geometry.axis_offset_angstrom, 0.0])
    spacing = geometry.protofilament_angular_spacing_deg

    rows, truth = [], []
    for protofilament in range(geometry.n_protofilaments):
        phi = (protofilament * spacing + azimuth_offset) % 360.0
        rotation = _rotation(phi, theta, psi)

        # phi is a roll about the filament axis, so every protofilament shares one
        # axis direction: the projected template z-axis.
        direction = (rotation.T @ np.array([0.0, 0.0, 1.0]))[:2]
        direction = direction / np.linalg.norm(direction)

        for repeat in range(n_repeats):
            # Rolling the template by phi to reach the next protofilament shifts where
            # its origin must sit along the axis by the helical offset, with the sign
            # set by the roll running opposite to template azimuth.
            height = (
                repeat * geometry.rise_angstrom
                - protofilament * geometry.lateral_axial_offset_angstrom
            )
            axis_point = np.array(origin) + direction * height / pixel_size
            offset = (rotation.T @ axis_point_template)[:2] / pixel_size

            truth.append(axis_point)
            rows.append(
                {
                    "x": axis_point[0] - offset[0],
                    "y": axis_point[1] - offset[1],
                    "phi": phi,
                    "theta": theta,
                    "psi": psi,
                    "z_score": 10.0,
                }
            )

    return pd.DataFrame(rows), np.array(truth)


@pytest.fixture()
def microtubule() -> TemplateLatticeGeometry:
    """A 14_3 microtubule template offset from its box centre, as 6dpu_4_patches is."""
    return TemplateLatticeGeometry(
        axis_offset_angstrom=(52.2, -69.4),
        subunit_radius_angstrom=118.7,
        n_protofilaments=14,
        rise_angstrom=84.0,
        n_start=3,
        lateral_axial_offset_angstrom=-9.0,
    )


class TestGeometryFromCentroids:
    """Reading the lattice back out of a model built to known specification."""

    @staticmethod
    def build_centroids(n_pf, radius, rise, lateral, n_repeats, centre=(0.0, 0.0)):
        points = []
        for protofilament in range(n_pf):
            azimuth = np.radians(protofilament * 360.0 / n_pf)
            for repeat in range(n_repeats):
                points.append(
                    [
                        centre[0] + radius * np.cos(azimuth),
                        centre[1] + radius * np.sin(azimuth),
                        repeat * rise + protofilament * lateral,
                    ]
                )
        return np.array(points)

    @pytest.mark.parametrize("n_pf", [11, 12, 13, 14, 15])
    def test_protofilament_count_recovered(self, n_pf):
        centroids = self.build_centroids(n_pf, 118.0, 84.0, -9.0, 6)
        geometry = TemplateLatticeGeometry.from_chain_centroids(centroids)
        assert geometry.n_protofilaments == n_pf

    def test_radius_rise_and_lateral_offset_recovered(self):
        centroids = self.build_centroids(14, 118.7, 84.0, -9.0, 6)
        geometry = TemplateLatticeGeometry.from_chain_centroids(centroids)
        assert geometry.subunit_radius_angstrom == pytest.approx(118.7, abs=0.5)
        assert geometry.rise_angstrom == pytest.approx(84.0, abs=0.5)
        assert geometry.lateral_axial_offset_angstrom == pytest.approx(-9.0, abs=0.5)

    def test_axis_offset_recovered_for_an_off_centre_template(self):
        # A patch template: only part of the tube, so its centroid is off-axis.
        full = self.build_centroids(14, 118.7, 84.0, -9.0, 6)
        patch = full[full[:, 0] > 40.0]
        patch = patch - np.append(patch[:, :2].mean(axis=0), 0.0)  # re-centre the box

        geometry = TemplateLatticeGeometry.from_chain_centroids(
            patch, n_protofilaments=14
        )
        assert geometry.axis_offset_magnitude_angstrom > 30.0
        assert geometry.subunit_radius_angstrom == pytest.approx(118.7, abs=2.0)

    def test_ring_template_is_axis_centred(self):
        centroids = self.build_centroids(14, 118.7, 84.0, -9.0, 6)
        geometry = TemplateLatticeGeometry.from_chain_centroids(centroids)
        assert geometry.axis_offset_magnitude_angstrom < 1.0


class TestSeamGeometry:
    def test_half_integer_turn_signals_a_seam(self, microtubule):
        # 14 protofilaments x -9 A = -126 A per turn = -1.5 dimer repeats.
        assert microtubule.axial_shift_per_turn_angstrom == pytest.approx(-126.0)
        assert microtubule.turns_are_closed_in_repeats == pytest.approx(-1.5)

    def test_integer_turn_means_no_seam(self):
        closed = TemplateLatticeGeometry(
            axis_offset_angstrom=(0.0, 0.0),
            subunit_radius_angstrom=100.0,
            n_protofilaments=12,
            rise_angstrom=84.0,
            lateral_axial_offset_angstrom=7.0,
        )
        assert closed.turns_are_closed_in_repeats == pytest.approx(1.0)


class TestSuggestedRadii:
    def test_angular_radius_sits_between_lobe_and_protofilament_spacing(
        self, microtubule
    ):
        suggested = microtubule.suggested_angular_radius_deg(lobe_half_width_deg=10.0)
        assert 10.0 < suggested < microtubule.protofilament_angular_spacing_deg

    def test_fewer_protofilaments_allow_a_wider_radius(self, microtubule):
        thirteen = TemplateLatticeGeometry(
            **{**microtubule.__dict__, "n_protofilaments": 13}
        )
        assert (
            thirteen.suggested_angular_radius_deg()
            > microtubule.suggested_angular_radius_deg()
        )

    def test_unresolvable_lattice_raises(self, microtubule):
        with pytest.raises(ValueError, match="protofilament spacing"):
            microtubule.suggested_angular_radius_deg(lobe_half_width_deg=40.0)

    def test_xy_radius_stays_below_half_the_rise(self, microtubule):
        suggested = microtubule.suggested_xy_radius_px(PIXEL_SIZE)
        assert suggested < 0.5 * microtubule.rise_angstrom / PIXEL_SIZE + 1e-9


class TestImageOffset:
    def test_identity_rotation_is_the_identity_projection(self):
        offset = image_offset_from_template_point(
            np.array([12.0, -5.0, 7.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        assert offset[0] == pytest.approx([12.0, -5.0])

    def test_matches_the_transpose_rotation(self):
        point = np.array([18.0, -11.0, 25.0])
        angles = (111.8, 87.5, 280.0)
        offset = image_offset_from_template_point(
            point, *[np.array([a]) for a in angles]
        )
        assert offset[0] == pytest.approx((_rotation(*angles).T @ point)[:2])

    def test_phi_only_rolls_a_point_on_the_axis(self):
        # A point on the filament axis (template z) is unmoved by a roll about it.
        offsets = image_offset_from_template_point(
            np.array([0.0, 0.0, 30.0]),
            np.array([0.0, 90.0, 180.0, 270.0]),
            np.full(4, 90.0),
            np.zeros(4),
        )
        assert np.ptp(offsets, axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)


class TestAxisRecovery:
    """The central claim: every peak independently predicts the same axis."""

    def test_axis_points_recovered_exactly(self, microtubule):
        peaks, truth = synthetic_lattice(microtubule)
        recovered = axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE)
        assert recovered == pytest.approx(truth, abs=1e-6)

    def test_corrected_points_are_collinear(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule)
        axis = fit_filament_axis(
            axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE),
            peaks["z_score"].to_numpy(),
            _reference(peaks),
        )
        assert axis.residual_rms_px < 1e-6

    def test_raw_positions_are_not_collinear(self, microtubule):
        """Without the correction the peaks scatter by the template's axis offset."""
        peaks, _ = synthetic_lattice(microtubule)
        raw = np.stack([peaks["x"].to_numpy(), peaks["y"].to_numpy()], axis=1)
        axis = fit_filament_axis(raw, peaks["z_score"].to_numpy())
        expected = microtubule.axis_offset_magnitude_angstrom / PIXEL_SIZE
        assert axis.residual_rms_px > 0.3 * expected

    def test_ring_template_needs_no_correction(self):
        centred = TemplateLatticeGeometry(
            axis_offset_angstrom=(0.0, 0.0),
            subunit_radius_angstrom=118.7,
            n_protofilaments=14,
            rise_angstrom=84.0,
            lateral_axial_offset_angstrom=-9.0,
        )
        peaks, _ = synthetic_lattice(centred)
        raw = np.stack([peaks["x"].to_numpy(), peaks["y"].to_numpy()], axis=1)
        assert fit_filament_axis(raw).residual_rms_px < 1e-6

    def test_missing_columns_raise(self, microtubule):
        with pytest.raises(ValueError, match="missing required columns"):
            axis_points_from_peaks(
                pd.DataFrame({"x": [1.0], "y": [2.0]}), microtubule, PIXEL_SIZE
            )


class TestFilamentCoordinates:
    def test_along_axis_spacing_matches_the_lattice(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule, n_repeats=10)
        points = axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE)
        axis = fit_filament_axis(points, peaks["z_score"].to_numpy(), _reference(peaks))
        along, across = filament_coordinates(points, axis)

        assert np.abs(across).max() < 1e-6
        one_protofilament = along[:10] * PIXEL_SIZE
        steps = np.diff(np.sort(one_protofilament))
        assert steps == pytest.approx(microtubule.rise_angstrom, abs=1e-6)


class TestHelicalUnwrap:
    def test_unwrapping_sharpens_the_axial_lattice(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule)
        points = axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE)
        axis = fit_filament_axis(points, peaks["z_score"].to_numpy(), _reference(peaks))
        along, _ = filament_coordinates(points, axis)

        def sharpness(values):
            phase = 2j * np.pi * values / microtubule.rise_angstrom
            return float(np.abs(np.exp(phase).mean()))

        pooled = sharpness(along * PIXEL_SIZE)
        unwrapped = sharpness(
            unwrap_helical_axial_coordinate(
                along * PIXEL_SIZE, peaks["phi"].to_numpy(), microtubule
            )
        )
        assert unwrapped > 0.99, "an ideal lattice must be sharp once unwrapped"
        assert pooled < 0.5, "pooling protofilaments must smear it"

    def test_no_helical_offset_is_a_no_op(self, microtubule):
        flat = TemplateLatticeGeometry(
            **{**microtubule.__dict__, "lateral_axial_offset_angstrom": 0.0}
        )
        values = np.array([0.0, 10.0, 20.0])
        unwrapped = unwrap_helical_axial_coordinate(
            values, np.array([0.0, 120.0, 240.0]), flat
        )
        assert unwrapped == pytest.approx(values)


class TestRiseEstimation:
    @pytest.mark.parametrize("true_rise", [81.0, 84.0, 87.0])
    def test_rise_recovered_from_an_ideal_lattice(self, microtubule, true_rise):
        geometry = TemplateLatticeGeometry(
            **{**microtubule.__dict__, "rise_angstrom": true_rise}
        )
        peaks, _ = synthetic_lattice(geometry, n_repeats=12)
        points = axis_points_from_peaks(peaks, geometry, PIXEL_SIZE)
        axis = fit_filament_axis(points, peaks["z_score"].to_numpy(), _reference(peaks))
        along, _ = filament_coordinates(points, axis)
        unwrapped = unwrap_helical_axial_coordinate(
            along * PIXEL_SIZE, peaks["phi"].to_numpy(), geometry
        )

        # Deliberately start from the wrong rise to prove the indexing converges.
        result = estimate_lattice_rise(unwrapped / PIXEL_SIZE, PIXEL_SIZE, 84.0)
        assert result.rise_angstrom == pytest.approx(true_rise, abs=0.05)
        assert result.standard_error_angstrom < 0.05
        # The ratio must agree with the absolute value against the same reference.
        assert result.ratio == pytest.approx(true_rise / 84.0, abs=0.001)
        assert result.rise_pixels == pytest.approx(true_rise / PIXEL_SIZE, abs=0.06)

    def test_expanded_and_compacted_lattices_are_distinguishable(self, microtubule):
        rises = []
        for true_rise in (81.0, 84.0):
            geometry = TemplateLatticeGeometry(
                **{**microtubule.__dict__, "rise_angstrom": true_rise}
            )
            peaks, _ = synthetic_lattice(geometry, n_repeats=12)
            points = axis_points_from_peaks(peaks, geometry, PIXEL_SIZE)
            axis = fit_filament_axis(
                points, peaks["z_score"].to_numpy(), _reference(peaks)
            )
            along, _ = filament_coordinates(points, axis)
            unwrapped = unwrap_helical_axial_coordinate(
                along * PIXEL_SIZE, peaks["phi"].to_numpy(), geometry
            )
            rises.append(
                estimate_lattice_rise(unwrapped / PIXEL_SIZE, PIXEL_SIZE, 84.0)
            )

        compact, expanded = rises
        difference = expanded.rise_angstrom - compact.rise_angstrom
        assert difference == pytest.approx(3.0, abs=0.2)
        assert (
            compact.standard_error_angstrom + expanded.standard_error_angstrom
            < 0.5 * difference
        )
        # Reported as a ratio the same distinction must survive.
        assert expanded.ratio - compact.ratio == pytest.approx(3.0 / 84.0, abs=0.01)

    def test_too_few_peaks_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            estimate_lattice_rise(np.array([0.0, 10.0]), PIXEL_SIZE, 84.0)

    def test_span_shorter_than_one_repeat_raises(self):
        with pytest.raises(ValueError, match="less than one repeat"):
            estimate_lattice_rise(np.array([0.0, 5.0, 10.0]), PIXEL_SIZE, 84.0)


class TestProtofilamentNumber:
    @pytest.mark.parametrize("n_pf", [12, 13, 14, 15])
    def test_recovered_from_evenly_sampled_azimuths(self, n_pf):
        geometry = TemplateLatticeGeometry(
            axis_offset_angstrom=(52.2, -69.4),
            subunit_radius_angstrom=118.7,
            n_protofilaments=n_pf,
            rise_angstrom=84.0,
            lateral_axial_offset_angstrom=-9.0,
        )
        peaks, _ = synthetic_lattice(geometry, n_repeats=8)
        recovered, _ = estimate_protofilament_number(peaks)
        assert recovered == n_pf

    def test_power_spectrum_covers_the_requested_orders(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule, n_repeats=4)
        _, power = estimate_protofilament_number(peaks, max_count=20)
        assert power.shape == (19,)


class TestPolarity:
    def test_dominant_pole_wins(self):
        detections = pd.DataFrame(
            {
                "psi": [279.0] * 50 + [99.0] * 5,
                "z_score": [12.0] * 50 + [8.2] * 5,
            }
        )
        polarity = estimate_polarity(detections)
        assert polarity.psi_deg == pytest.approx(279.0, abs=1.0)
        assert polarity.n_winning == 50
        assert polarity.n_opposite == 5
        assert polarity.score_ratio > 10.0

    def test_balanced_poles_report_a_ratio_near_one(self):
        """A template that cannot tell the two ends apart must not fake an answer."""
        detections = pd.DataFrame(
            {"psi": [279.0] * 20 + [99.0] * 20, "z_score": [10.0] * 40}
        )
        assert estimate_polarity(detections).score_ratio == pytest.approx(1.0, abs=0.01)

    def test_works_when_the_poles_straddle_the_wrap(self):
        detections = pd.DataFrame(
            {"psi": [2.0, 358.0, 1.0] * 5 + [181.0], "z_score": [10.0] * 15 + [6.0]}
        )
        polarity = estimate_polarity(detections)
        assert polarity.n_winning == 15
        assert polarity.n_opposite == 1

    def test_empty_after_threshold_raises(self):
        detections = pd.DataFrame({"psi": [279.0], "z_score": [3.0]})
        with pytest.raises(ValueError, match="No detections"):
            estimate_polarity(detections, score_threshold=8.0)


class TestAxisFitting:
    def test_recovers_a_known_line(self):
        direction = np.array([np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
        offsets = np.outer(np.linspace(-50, 50, 21), direction)
        points = np.array([100.0, 200.0]) + offsets
        axis = fit_filament_axis(points)

        assert axis.residual_rms_px < 1e-9
        assert abs(np.dot(axis.direction, direction)) == pytest.approx(1.0)

    def test_residual_reports_scatter(self):
        rng = np.random.default_rng(0)
        points = np.stack([np.linspace(0, 100, 200), rng.normal(0.0, 3.0, 200)], axis=1)
        assert fit_filament_axis(points).residual_rms_px == pytest.approx(3.0, rel=0.2)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_filament_axis(np.array([[1.0, 2.0]]))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fit_filament_axis(np.array([1.0, 2.0, 3.0]))


class TestSubunitsPerRepeat:
    """The repeat spans however many modelled subunits the caller says it does."""

    def test_two_chains_per_repeat_gives_twice_the_rise(self):
        # One chain per 42 A monomer; the dimer repeat spans two of them.
        centroids = TestGeometryFromCentroids.build_centroids(14, 118.7, 42.0, -4.5, 12)
        monomer = TemplateLatticeGeometry.from_chain_centroids(centroids)
        dimer = TemplateLatticeGeometry.from_chain_centroids(
            centroids, subunits_per_repeat=2
        )
        assert monomer.rise_angstrom == pytest.approx(42.0, abs=0.5)
        assert dimer.rise_angstrom == pytest.approx(84.0, abs=0.5)


class TestAxisSign:
    """A principal-axis fit fixes the axis only up to sign, which the unwrap needs."""

    def test_reference_direction_orients_the_fit(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule)
        points = axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE)
        reference = _reference(peaks)

        oriented = fit_filament_axis(points, None, reference)
        flipped = fit_filament_axis(points, None, -reference)

        assert oriented.direction @ reference > 0
        assert flipped.direction @ reference < 0
        assert oriented.direction == pytest.approx(-flipped.direction)

    def test_unwrap_fails_with_the_wrong_sign(self, microtubule):
        """Guards the trap: a flipped axis makes the unwrap anti-collapse things."""
        peaks, _ = synthetic_lattice(microtubule)
        points = axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE)
        phi = peaks["phi"].to_numpy()

        def sharpness(direction_reference):
            axis = fit_filament_axis(points, None, direction_reference)
            along, _ = filament_coordinates(points, axis)
            values = unwrap_helical_axial_coordinate(
                along * PIXEL_SIZE, phi, microtubule
            )
            phase = 2j * np.pi * values / microtubule.rise_angstrom
            return float(np.abs(np.exp(phase).mean()))

        assert sharpness(_reference(peaks)) > 0.99
        assert sharpness(-_reference(peaks)) < 0.5

    def test_direction_is_independent_of_phi(self, microtubule):
        """phi is a roll, so every protofilament implies the same axis direction."""
        peaks, _ = synthetic_lattice(microtubule)
        directions = [
            filament_direction_from_angles(
                np.array([phi]), np.array([90.0]), np.array([279.0])
            )
            for phi in (0.0, 90.0, 180.0, 270.0)
        ]
        for direction in directions[1:]:
            assert direction == pytest.approx(directions[0], abs=1e-9)


class TestPreferRefinedColumns:
    """Lattice measurements must use refine_template's output when it is available."""

    def test_refined_columns_preferred(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule, n_repeats=3)
        peaks = peaks.rename(
            columns={
                "x": "refined_pos_x",
                "y": "refined_pos_y",
                "phi": "refined_phi",
                "theta": "refined_theta",
                "psi": "refined_psi",
            }
        )
        # Deliberately wrong originals: picking them up would wreck the fit.
        peaks["x"] = 0.0
        peaks["y"] = 0.0
        peaks["phi"] = 0.0
        peaks["theta"] = 0.0
        peaks["psi"] = 0.0

        assert geometry_columns(peaks)["x"] == "refined_pos_x"
        axis = fit_filament_axis(axis_points_from_peaks(peaks, microtubule, PIXEL_SIZE))
        assert axis.residual_rms_px < 1e-6

    def test_plain_columns_used_when_unrefined(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule, n_repeats=3)
        assert geometry_columns(peaks)["x"] == "x"

    def test_preference_can_be_disabled(self, microtubule):
        peaks, _ = synthetic_lattice(microtubule, n_repeats=3)
        peaks["refined_pos_x"] = peaks["x"]
        peaks["refined_pos_y"] = peaks["y"]
        peaks["refined_phi"] = peaks["phi"]
        peaks["refined_theta"] = peaks["theta"]
        peaks["refined_psi"] = peaks["psi"]

        assert geometry_columns(peaks)["x"] == "refined_pos_x"
        assert geometry_columns(peaks, prefer_refined=False)["x"] == "x"

    def test_match_template_naming_is_understood(self, microtubule):
        """A stack straight from match_template uses pos_x / pos_y."""
        peaks, _ = synthetic_lattice(microtubule, n_repeats=3)
        peaks = peaks.rename(columns={"x": "pos_x", "y": "pos_y"})
        assert geometry_columns(peaks)["x"] == "pos_x"


def _synthetic_filament_volume(
    size=96, repeat_px=24.0, n_protofilaments=6, radius_px=20.0, half_contrast=1.0
):
    """A crude helical filament volume: blobs on a lattice, axis along z.

    ``half_contrast`` scales every second subunit, so 1.0 makes the two halves of the
    repeat identical (indistinguishable subunits) and smaller values make them differ.
    """
    volume = np.zeros((size, size, size), dtype=np.float32)
    centre = size // 2
    for pf in range(n_protofilaments):
        azimuth = 2 * np.pi * pf / n_protofilaments
        x = int(centre + radius_px * np.cos(azimuth))
        y = int(centre + radius_px * np.sin(azimuth))
        for step in range(int(size / (repeat_px / 2))):
            z = int((step * repeat_px / 2) % size)
            volume[z, y, x] = 1.0 if step % 2 == 0 else half_contrast
    return volume


class TestTemplateAutocorrelation:
    def test_recovers_the_repeat(self):
        volume = _synthetic_filament_volume(repeat_px=24.0)
        assert rise_from_template_autocorrelation(volume, 24.0) == pytest.approx(
            24.0, abs=0.5
        )

    def test_recovers_the_half_repeat(self):
        volume = _synthetic_filament_volume(repeat_px=24.0)
        assert rise_from_template_autocorrelation(volume, 12.0) == pytest.approx(
            12.0, abs=0.5
        )

    def test_profile_is_normalised_at_zero_lag(self):
        profile = template_axial_autocorrelation(_synthetic_filament_volume())
        assert profile[0] == pytest.approx(1.0)

    def test_identical_subunits_give_contrast_near_one(self):
        volume = _synthetic_filament_volume(half_contrast=1.0)
        assert subunit_contrast(volume, 24.0) == pytest.approx(1.0, abs=0.05)

    def test_differing_subunits_lower_the_contrast(self):
        weak = _synthetic_filament_volume(half_contrast=0.2)
        assert subunit_contrast(weak, 24.0) < 0.9

    def test_rejects_non_3d_input(self):
        with pytest.raises(ValueError, match="3-D"):
            template_axial_autocorrelation(np.zeros((10, 10)))

    def test_empty_volume_raises(self):
        with pytest.raises(ValueError, match="empty"):
            template_axial_autocorrelation(np.zeros((16, 16, 16)))

    def test_absent_peak_raises(self):
        """A non-periodic object has no repeat to find."""
        volume = np.zeros((96, 96, 96), dtype=np.float32)
        volume[40:56, 40:56, 40:56] = 1.0  # a single blob, no lattice
        with pytest.raises(ValueError, match="No autocorrelation peak"):
            rise_from_template_autocorrelation(volume, 30.0)

    def test_window_beyond_the_volume_raises(self):
        volume = _synthetic_filament_volume(size=96)
        with pytest.raises(ValueError, match="Search window"):
            rise_from_template_autocorrelation(volume, 200.0, search_fraction=0.01)


class TestAxisOffsetReference:
    """The offset is measured from the coordinate origin, which is the box centre."""

    def test_offset_is_relative_to_the_origin_not_the_centroid(self):
        full = TestGeometryFromCentroids.build_centroids(14, 118.7, 84.0, -9.0, 6)
        patch = full[full[:, 0] > 40.0]  # an arc, so its centroid is off-axis

        centred = TemplateLatticeGeometry.from_chain_centroids(
            patch, n_protofilaments=14
        )
        assert centred.axis_offset_magnitude_angstrom < 1.0

        # Translating the model moves the axis with it, so the offset follows.
        shifted = patch + np.array([30.0, -20.0, 0.0])
        moved = TemplateLatticeGeometry.from_chain_centroids(
            shifted, n_protofilaments=14
        )
        assert moved.axis_offset_angstrom[0] == pytest.approx(30.0, abs=1.0)
        assert moved.axis_offset_angstrom[1] == pytest.approx(-20.0, abs=1.0)


class TestRiseInferenceIsUnbiased:
    """The two halves of a repeat are unevenly spaced; naive estimators are biased."""

    @staticmethod
    def alternating_centroids(n_pf, radius, first, second, n_repeats, gap_after=None):
        """A lattice with unequal half-steps, optionally split by a non-lattice gap."""
        points = []
        for pf in range(n_pf):
            azimuth = np.radians(pf * 360.0 / n_pf)
            height = 0.0
            for subunit in range(2 * n_repeats):
                points.append(
                    [radius * np.cos(azimuth), radius * np.sin(azimuth), height]
                )
                height += first if subunit % 2 == 0 else second
                if gap_after is not None and subunit == gap_after:
                    height += 6.0  # a gap that is not a whole number of subunits
        return np.array(points)

    def test_alternating_steps_give_their_true_mean(self):
        # Steps of 41 and 43 must average to 42, not to either alternate.
        centroids = self.alternating_centroids(14, 118.7, 41.0, 43.0, 5)
        geometry = TemplateLatticeGeometry.from_chain_centroids(centroids)
        assert geometry.rise_angstrom == pytest.approx(42.0, abs=0.1)

    def test_odd_number_of_steps_is_still_unbiased(self):
        """A plain linear fit is biased when the step count is odd; this must not be."""
        centroids = self.alternating_centroids(14, 118.7, 41.0, 43.0, 5)
        trimmed = centroids[:-1]  # leave an odd number of steps in the last column
        geometry = TemplateLatticeGeometry.from_chain_centroids(trimmed)
        assert geometry.rise_angstrom == pytest.approx(42.0, abs=0.1)

    def test_dimer_repeat_is_twice_the_monomer(self):
        centroids = self.alternating_centroids(14, 118.7, 41.0, 43.0, 5)
        monomer = TemplateLatticeGeometry.from_chain_centroids(centroids)
        dimer = TemplateLatticeGeometry.from_chain_centroids(
            centroids, subunits_per_repeat=2
        )
        assert dimer.rise_angstrom == pytest.approx(2 * monomer.rise_angstrom, abs=0.01)

    def test_a_gap_between_patches_does_not_bias_the_fit(self):
        """A patch template's separate patches are not one continuous lattice."""
        centroids = self.alternating_centroids(6, 118.7, 41.0, 43.0, 4, gap_after=3)
        geometry = TemplateLatticeGeometry.from_chain_centroids(
            centroids, n_protofilaments=6
        )
        assert geometry.rise_angstrom == pytest.approx(42.0, abs=0.2)

    def test_no_contiguous_run_raises(self):
        # Every step a different, non-lattice size.
        points = [[118.7, 0.0, h] for h in (0.0, 13.0, 40.0, 91.0)]
        with pytest.raises(ValueError, match="contiguous|three"):
            TemplateLatticeGeometry.from_chain_centroids(
                np.array(points), n_protofilaments=1
            )


class TestForeshorteningCorrection:
    """A filament tilted out of the image plane projects shorter than it is."""

    def _measure(self, geometry, theta, correct):
        peaks, _ = synthetic_lattice(geometry, n_repeats=12, theta=theta)
        points = axis_points_from_peaks(peaks, geometry, PIXEL_SIZE)
        axis = fit_filament_axis(points, peaks["z_score"].to_numpy(), _reference(peaks))
        along, _ = filament_coordinates(points, axis)
        unwrapped = unwrap_helical_axial_coordinate(
            along * PIXEL_SIZE, peaks["phi"].to_numpy(), geometry
        )
        return estimate_lattice_rise(
            unwrapped / PIXEL_SIZE,
            PIXEL_SIZE,
            84.0,
            theta_deg=peaks["theta"].to_numpy() if correct else None,
        )

    def test_in_plane_needs_no_correction(self, microtubule):
        result = self._measure(microtubule, 90.0, correct=True)
        assert result.foreshortening == pytest.approx(1.0, abs=1e-6)
        assert result.rise_angstrom == pytest.approx(84.0, abs=0.05)

    def test_correction_is_recorded_but_not_applied_twice(self, microtubule):
        uncorrected = self._measure(microtubule, 90.0, correct=False)
        assert uncorrected.foreshortening == 1.0

    def test_summary_mentions_the_correction_only_when_there_is_one(self, microtubule):
        assert (
            "tilt correction"
            not in self._measure(microtubule, 90.0, correct=True).summary()
        )

    def test_foreshortening_matches_sin_theta(self, microtubule):
        # The synthetic generator builds the lattice in the projected frame, so the
        # recorded factor should simply be sin(theta) for the supplied tilt.
        result = self._measure(microtubule, 80.0, correct=True)
        assert result.foreshortening == pytest.approx(
            np.sin(np.radians(80.0)), abs=1e-6
        )
        assert result.rise_angstrom > 84.0, "correcting must lengthen the repeat"

    def test_end_on_filament_raises(self):
        """Near theta = 0 or 180 the repeat is unmeasurable, not merely shortened."""
        with pytest.raises(ValueError, match="viewing direction"):
            estimate_lattice_rise(
                np.arange(10) * 90.0,
                PIXEL_SIZE,
                84.0,
                theta_deg=np.full(10, 180.0),
            )
