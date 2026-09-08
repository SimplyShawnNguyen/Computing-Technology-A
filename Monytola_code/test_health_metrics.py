#!/usr/bin/env python3
"""
test_health_metrics.py — unit tests for the Sprint-1 health metrics.

These tests exercise ``compute_health_metrics`` (and its two metric helpers)
against SYNTHETIC Result objects, so no camera, no RealSense, and no
perception pipeline are needed to run them. This is exactly the "fake /
synthetic Result values" acceptance criterion from sprint-plan item 6.

Run from the project folder with:

    python -m unittest test_health_metrics -v

or, if pytest is installed:

    pytest -q test_health_metrics.py
"""

import math
import unittest
from types import SimpleNamespace

import health_metrics as hm


def fake_result(**kwargs) -> SimpleNamespace:
    """Build a minimal duck-typed stand-in for ``MaxDivProcessor.Result``.

    Only the fields the metrics layer reads are provided. Anything not passed
    falls back to the "empty / invalid" value a real Result starts with, so
    each test can specify just the inputs it cares about.
    """
    defaults = dict(
        n_vectors=0,
        axis_strength=0.0,
        divmax_raw=float("nan"),
        valid=False,
        tau_s=None,
        arc_vec=(1.0, 0.0),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestMetric1VectorCount(unittest.TestCase):
    """METRIC 1 — valid-vector count."""

    def test_reads_n_vectors(self):
        value, state = hm.metric1_vector_count(fake_result(n_vectors=23))
        self.assertEqual(value, 23)
        self.assertEqual(state, hm.MetricState.AVAILABLE)

    def test_zero_vectors_is_reported_not_omitted(self):
        # A count of zero is a real, meaningful value ("no flow"), not a
        # missing one, so it must be returned as-is.
        value, state = hm.metric1_vector_count(fake_result(n_vectors=0))
        self.assertEqual(value, 0)
        self.assertEqual(state, hm.MetricState.AVAILABLE)

    def test_missing_field_defaults_to_zero(self):
        # Defensive: a partial Result that lacks n_vectors must not crash.
        value, _ = hm.metric1_vector_count(SimpleNamespace())
        self.assertEqual(value, 0)


class TestMetric2ShearDivergence(unittest.TestCase):
    """METRIC 2 — axis_strength / divmax_raw with safe handling."""

    def test_valid_ratio(self):
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=0.05)
        )
        self.assertAlmostEqual(ratio, 0.4)
        self.assertEqual(state, hm.MetricState.AVAILABLE)

    def test_nan_denominator_is_unavailable(self):
        # Default divmax_raw is NaN (field never computed) -> must be None.
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=float("nan"))
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_zero_denominator_is_unavailable(self):
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=0.0)
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_negative_denominator_is_unavailable(self):
        # A negative divergence is not a meaningful approach cue; guard it.
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=-0.05)
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_infinite_denominator_is_unavailable(self):
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=float("inf"))
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_below_floor_denominator_is_unavailable(self):
        # Tiny-but-positive divergence (below the 1e-9 floor) is effectively
        # zero and must be treated as unavailable.
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=1e-12)
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_infinite_ratio_is_unavailable(self):
        # A finite denominator with an overflowing numerator must not leak inf.
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=float("inf"), divmax_raw=0.05)
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)


class TestComputeHealthMetrics(unittest.TestCase):
    """The top-level pure function ties both metrics together."""

    def test_good_frame_populates_both_metrics(self):
        metrics = hm.compute_health_metrics(
            fake_result(
                n_vectors=18, axis_strength=0.02, divmax_raw=0.05,
                valid=True, tau_s=2.0,
            ),
            timestamp=1.5,
            frame_index=7,
        )
        self.assertEqual(metrics.n_vectors, 18)
        self.assertAlmostEqual(metrics.shear_div_ratio, 0.4)
        self.assertEqual(metrics.shear_div_ratio_state, hm.MetricState.AVAILABLE)
        self.assertTrue(metrics.cue_valid)
        self.assertEqual(metrics.tau_s, 2.0)
        self.assertEqual(metrics.frame_index, 7)
        self.assertEqual(metrics.timestamp, 1.5)

    def test_invalid_frame_is_unavailable_not_crash(self):
        # A fully-default fake mirrors a primed-but-not-yet-valid Result.
        metrics = hm.compute_health_metrics(fake_result())
        self.assertEqual(metrics.n_vectors, 0)
        self.assertIsNone(metrics.shear_div_ratio)
        self.assertEqual(metrics.shear_div_ratio_state, hm.MetricState.UNAVAILABLE)
        self.assertFalse(metrics.cue_valid)

    def test_custom_threshold(self):
        # The floor is configurable: 0.005 is below the custom 0.01 floor.
        thresholds = hm.HealthThresholds(min_divergence=0.01)
        ratio, state = hm.metric2_shear_divergence(
            fake_result(axis_strength=0.02, divmax_raw=0.005), thresholds
        )
        self.assertIsNone(ratio)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_missing_fields_do_not_crash(self):
        # Passing an object with none of the expected fields must still yield
        # a valid (all-unavailable) snapshot rather than raising.
        metrics = hm.compute_health_metrics(SimpleNamespace())
        self.assertEqual(metrics.n_vectors, 0)
        self.assertIsNone(metrics.shear_div_ratio)
        self.assertFalse(metrics.cue_valid)
        self.assertIsNone(metrics.tau_s)


class TestMetric3ArcDirectionStability(unittest.TestCase):
    """METRIC 3 — arc-direction stability over a window, with 180° handling."""

    def test_same_line_is_stable_even_across_sign_flips(self):
        # +v and -v are the SAME arc line, so a window that merely flips sense
        # must be reported as stable (this is the 180° direction ambiguity).
        vecs = [(1.0, 0.0), (-1.0, 0.0), (0.9999, 0.01), (-0.9999, -0.01)]
        spread, stability, state = hm.metric3_arc_direction_stability(vecs)
        self.assertEqual(state, hm.MetricState.AVAILABLE)
        self.assertAlmostEqual(stability, 1.0, places=2)
        self.assertLess(spread, 2.0)  # degrees; essentially aligned

    def test_perpendicular_lines_are_maximally_spread(self):
        spread, stability, state = hm.metric3_arc_direction_stability(
            [(1.0, 0.0), (0.0, 1.0)]
        )
        self.assertEqual(state, hm.MetricState.AVAILABLE)
        self.assertAlmostEqual(stability, 0.0, places=3)
        self.assertGreaterEqual(spread, 90.0)

    def test_identical_vectors_are_zero_spread(self):
        spread, stability, state = hm.metric3_arc_direction_stability(
            [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
        )
        self.assertEqual(state, hm.MetricState.AVAILABLE)
        self.assertAlmostEqual(stability, 1.0)
        self.assertAlmostEqual(spread, 0.0, places=2)

    def test_single_vector_is_unavailable(self):
        spread, stability, state = hm.metric3_arc_direction_stability([(1.0, 0.0)])
        self.assertIsNone(spread)
        self.assertIsNone(stability)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_empty_is_unavailable(self):
        spread, stability, state = hm.metric3_arc_direction_stability([])
        self.assertIsNone(spread)
        self.assertIsNone(stability)
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_degenerate_vectors_are_skipped(self):
        # Zero-length entries carry no direction and must be ignored; the two
        # usable vectors align, so it should still be stable.
        spread, stability, state = hm.metric3_arc_direction_stability(
            [(1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        )
        self.assertEqual(state, hm.MetricState.AVAILABLE)
        self.assertAlmostEqual(stability, 1.0)
        self.assertAlmostEqual(spread, 0.0, places=2)


class TestArcDirectionStabilityWindow(unittest.TestCase):
    """Rolling-buffer helper that feeds METRIC 3."""

    def test_window_keeps_last_n(self):
        w = hm.ArcDirectionStabilityWindow(window_size=3)
        for v in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            w.update(v)
        self.assertEqual(len(w), 3)
        self.assertEqual(len(w.arc_vecs()), 3)
        self.assertEqual(w.arc_vecs()[-1], (0.0, -1.0))  # oldest dropped

    def test_update_skips_none_and_garbage(self):
        w = hm.ArcDirectionStabilityWindow(5)
        w.update(None)
        w.update("nope")
        w.update((1.0, 0.0))
        self.assertEqual(len(w), 1)
        self.assertEqual(w.arc_vecs(), [(1.0, 0.0)])

    def test_spread_and_stability(self):
        w = hm.ArcDirectionStabilityWindow(5)
        for v in [(1, 0), (1, 0)]:
            w.update(v)
        spread, stability, state = w.spread_and_stability()
        self.assertEqual(state, hm.MetricState.AVAILABLE)
        self.assertAlmostEqual(stability, 1.0)

    def test_reset(self):
        w = hm.ArcDirectionStabilityWindow(5)
        w.update((1.0, 0.0))
        w.reset()
        self.assertEqual(len(w), 0)
        _, _, state = w.spread_and_stability()
        self.assertEqual(state, hm.MetricState.UNAVAILABLE)

    def test_window_size_clamped_to_two(self):
        w = hm.ArcDirectionStabilityWindow(1)
        self.assertEqual(w.window_size, 2)


class TestComputeArcMetric(unittest.TestCase):
    """compute_health_metrics fills METRIC 3 from the caller-supplied window."""

    def test_arc_history_populates_metric3(self):
        metrics = hm.compute_health_metrics(
            fake_result(n_vectors=10, axis_strength=0.01, divmax_raw=0.02, arc_vec=(1.0, 0.0)),
            arc_history=[(1.0, 0.0), (1.0, 0.0)],
        )
        self.assertEqual(metrics.arc_dir_spread_state, hm.MetricState.AVAILABLE)
        self.assertAlmostEqual(metrics.arc_dir_stability, 1.0)
        self.assertAlmostEqual(metrics.arc_dir_spread_deg, 0.0, places=2)

    def test_no_arc_history_is_unavailable(self):
        # A single frame cannot produce a spread -> unavailable, not a crash.
        metrics = hm.compute_health_metrics(fake_result())
        self.assertEqual(metrics.arc_dir_spread_state, hm.MetricState.UNAVAILABLE)
        self.assertIsNone(metrics.arc_dir_spread_deg)
        self.assertIsNone(metrics.arc_dir_stability)


if __name__ == "__main__":
    unittest.main(verbosity=2)
