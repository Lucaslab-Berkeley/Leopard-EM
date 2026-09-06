"""Tests for orientation-aware peak finding over a correlation table."""

import numpy as np
import pandas as pd
import pytest

from leopard_em.analysis.correlation_peaks import (
    CLUSTER_STATISTIC_COLUMNS,
    constrained_zscore_cutoff,
    find_peaks_orientation_aware,
)


def make_detections(rows: list[dict]) -> pd.DataFrame:
    """Build a detections frame, filling in whatever the caller left out."""
    defaults = {
        "x": 0.0,
        "y": 0.0,
        "phi": 0.0,
        "theta": 90.0,
        "psi": 0.0,
        "relative_defocus": 0.0,
        "z_score": 6.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class TestOverlappingDetections:
    """The behaviour the whole module exists for."""

    def test_same_position_different_angle_both_survive(self):
        # Near and far wall of a tube: identical (x, y), azimuths 180 degrees apart.
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 0.0, "z_score": 10.0},
                {"x": 100, "y": 100, "phi": 180.0, "z_score": 8.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert len(peaks) == 2
        assert sorted(peaks["phi"]) == [0.0, 180.0]

    def test_same_position_similar_angle_collapse(self):
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 0.0, "z_score": 10.0},
                {"x": 100, "y": 100, "phi": 5.0, "z_score": 8.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert len(peaks) == 1
        assert peaks["z_score"].iloc[0] == 10.0, "the stronger detection must win"

    def test_far_apart_positions_survive_despite_identical_angle(self):
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 0.0, "z_score": 10.0},
                {"x": 400, "y": 400, "phi": 0.0, "z_score": 9.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert len(peaks) == 2

    def test_suppression_needs_both_position_and_angle(self):
        """Close in one dimension only is not enough to suppress."""
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 0.0, "z_score": 10.0},
                {"x": 105, "y": 100, "phi": 90.0, "z_score": 9.0},  # near, wrong angle
                {"x": 400, "y": 100, "phi": 2.0, "z_score": 8.0},  # right angle, far
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert len(peaks) == 3


class TestPhiWrapping:
    def test_phi_distance_wraps_at_360(self):
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 359.0, "z_score": 10.0},
                {"x": 100, "y": 100, "phi": 1.0, "z_score": 9.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert len(peaks) == 1, "359 and 1 degrees are 2 degrees apart, not 358"


class TestAngularMetrics:
    @pytest.mark.parametrize("metric", ["phi", "so3"])
    def test_metrics_separate_opposite_azimuths(self, metric):
        """Near and far wall of a tube differ by 180 degrees of roll."""
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 0.0, "theta": 90.0, "z_score": 10.0},
                {"x": 100, "y": 100, "phi": 180.0, "theta": 90.0, "z_score": 8.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections,
            xy_radius_px=20.0,
            angular_radius_deg=25.0,
            angular_metric=metric,
        )
        assert len(peaks) == 2

    def test_axis_metric_ignores_roll_about_the_template_axis(self):
        """phi is a roll, so it leaves the projected axis direction unchanged.

        This is the defining difference between "axis" and the other two metrics: it
        deliberately cannot separate the near and far wall of a filament, because both
        have the filament axis pointing the same way.
        """
        detections = make_detections(
            [
                {"x": 100, "y": 100, "phi": 0.0, "theta": 90.0, "z_score": 10.0},
                {"x": 100, "y": 100, "phi": 180.0, "theta": 90.0, "z_score": 8.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections,
            xy_radius_px=20.0,
            angular_radius_deg=25.0,
            angular_metric="axis",
        )
        assert len(peaks) == 1

    def test_axis_metric_separates_different_axis_directions(self):
        """psi tilts the filament axis in-plane, which "axis" does resolve."""
        detections = make_detections(
            [
                {"x": 100, "y": 100, "theta": 90.0, "psi": 0.0, "z_score": 10.0},
                {"x": 100, "y": 100, "theta": 90.0, "psi": 90.0, "z_score": 8.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections,
            xy_radius_px=20.0,
            angular_radius_deg=25.0,
            angular_metric="axis",
        )
        assert len(peaks) == 2

    def test_unknown_metric_raises(self):
        detections = make_detections([{"x": 1, "y": 1}])
        with pytest.raises(ValueError, match="angular_metric"):
            find_peaks_orientation_aware(
                detections,
                xy_radius_px=1.0,
                angular_radius_deg=1.0,
                angular_metric="nonsense",
            )


class TestOrderingAndLimits:
    def test_peaks_are_returned_best_first(self):
        detections = make_detections(
            [
                {"x": 0, "y": 0, "z_score": 7.0},
                {"x": 200, "y": 0, "z_score": 12.0},
                {"x": 400, "y": 0, "z_score": 9.0},
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert peaks["z_score"].tolist() == [12.0, 9.0, 7.0]

    def test_max_peaks_truncates_to_the_strongest(self):
        detections = make_detections(
            [{"x": 100 * i, "y": 0, "z_score": 6.0 + i} for i in range(5)]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0, max_peaks=2
        )
        assert len(peaks) == 2
        assert peaks["z_score"].tolist() == [10.0, 9.0]

    def test_score_threshold_filters_first(self):
        detections = make_detections(
            [{"x": 100 * i, "y": 0, "z_score": 5.0 + i} for i in range(5)]
        )
        peaks = find_peaks_orientation_aware(
            detections,
            xy_radius_px=20.0,
            angular_radius_deg=25.0,
            score_threshold=7.0,
        )
        assert peaks["z_score"].tolist() == [9.0, 8.0]

    def test_empty_input_returns_empty_frame(self):
        detections = make_detections([{"x": 0, "y": 0, "z_score": 1.0}])
        peaks = find_peaks_orientation_aware(
            detections,
            xy_radius_px=20.0,
            angular_radius_deg=25.0,
            score_threshold=100.0,
        )
        assert len(peaks) == 0


class TestClusterStatistics:
    def test_columns_present_and_counts_are_right(self):
        # One strong peak surrounded by nine near-duplicates, plus a lone detection.
        cluster = [
            {"x": 100 + dx, "y": 100 + dy, "phi": 0.0, "z_score": 6.0}
            for dx in (-2, 0, 2)
            for dy in (-2, 0, 2)
        ]
        cluster[4]["z_score"] = 12.0
        detections = make_detections([*cluster, {"x": 400, "y": 400, "z_score": 7.0}])

        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )

        for column in CLUSTER_STATISTIC_COLUMNS:
            assert column in peaks.columns

        assert peaks["n_detections"].iloc[0] == 9
        assert peaks["n_detections"].iloc[1] == 1
        assert peaks["sum_z_score"].iloc[0] == pytest.approx(12.0 + 8 * 6.0)

    def test_can_be_switched_off(self):
        detections = make_detections([{"x": 0, "y": 0}])
        peaks = find_peaks_orientation_aware(
            detections,
            xy_radius_px=20.0,
            angular_radius_deg=25.0,
            compute_cluster_statistics=False,
        )
        assert not set(CLUSTER_STATISTIC_COLUMNS) & set(peaks.columns)

    def test_defocus_spread_reported(self):
        detections = make_detections(
            [
                {"x": 100, "y": 100, "relative_defocus": d, "z_score": 6.0 + i}
                for i, d in enumerate([-200.0, 0.0, 200.0])
            ]
        )
        peaks = find_peaks_orientation_aware(
            detections, xy_radius_px=20.0, angular_radius_deg=25.0
        )
        assert len(peaks) == 1, "defocus must not separate peaks"
        assert peaks["defocus_spread"].iloc[0] == pytest.approx(400.0)
        assert peaks["n_defocus_planes"].iloc[0] == 3


class TestValidation:
    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing required columns"):
            find_peaks_orientation_aware(
                pd.DataFrame({"x": [1], "y": [1]}),
                xy_radius_px=1.0,
                angular_radius_deg=1.0,
            )

    @pytest.mark.parametrize("xy,ang", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
    def test_non_positive_radii_raise(self, xy, ang):
        detections = make_detections([{"x": 1, "y": 1}])
        with pytest.raises(ValueError, match="positive"):
            find_peaks_orientation_aware(
                detections, xy_radius_px=xy, angular_radius_deg=ang
            )


class TestConstrainedZscoreCutoff:
    def test_smaller_search_gives_a_lower_cutoff(self):
        big = constrained_zscore_cutoff(np.full((100, 100), 1000.0))
        small = constrained_zscore_cutoff(np.full((100, 100), 10.0))
        assert small < big

    def test_defocus_map_increases_the_search_size(self):
        without = constrained_zscore_cutoff(np.full((50, 50), 100.0))
        with_defocus = constrained_zscore_cutoff(
            np.full((50, 50), 100.0), np.full((50, 50), 7.0)
        )
        assert with_defocus > without

    def test_empty_constraint_raises(self):
        with pytest.raises(ValueError, match="no .*combinations"):
            constrained_zscore_cutoff(np.zeros((10, 10)))
