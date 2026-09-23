"""
Unit tests for rank_hotspots.py — noise gate + leverage formula + deterministic ranking.

Nyquist-mapped tests (per 113-VALIDATION):
  - test_noise_gate_flags_high_rsd
  - test_leverage_score_deterministic
"""

import importlib
import io
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Minimal test-data helpers
# ---------------------------------------------------------------------------

def _make_row(
    op="fpca",
    library="rust",
    n=100.0,
    m=50.0,
    mean_ns=1_000_000.0,
    rsd_pct=1.0,
    ci_width_pct=1.0,
    noise_gate="PASS",
    source="criterion:test",
    notes="",
):
    """Return a dict with the comparison.csv schema."""
    return dict(
        op=op,
        library=library,
        n=n,
        m=m,
        mean_ns=mean_ns,
        ci_lower_ns=None,
        ci_upper_ns=None,
        std_dev_ns=None,
        rsd_pct=rsd_pct,
        ci_width_pct=ci_width_pct,
        noise_gate=noise_gate,
        source=source,
        notes=notes,
    )


def _make_cross_row(op, library, n, m, mean_ns, notes=""):
    """Return a cross-language (python/r) row with noise_gate='NA'."""
    return dict(
        op=op,
        library=library,
        n=n,
        m=m,
        mean_ns=mean_ns,
        ci_lower_ns=None,
        ci_upper_ns=None,
        std_dev_ns=None,
        rsd_pct=None,
        ci_width_pct=None,
        noise_gate="NA",
        source=f"{library}_runner",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Import under test (late import so we can test errors from the module)
# ---------------------------------------------------------------------------

import rank_hotspots  # noqa: E402  (module in same directory)


class TestNoiseGate(unittest.TestCase):
    """Tests for the noise_gate filter (Nyquist test 1: test_noise_gate_flags_high_rsd)."""

    def test_noise_gate_flags_high_rsd(self):
        """A criterion row with RSD >= 5.0 must be excluded from target selection."""
        rows = [
            _make_row(op="elastic", rsd_pct=6.0, ci_width_pct=2.0, noise_gate="FAIL-NOISY"),
            _make_row(op="fpca", rsd_pct=1.0, ci_width_pct=1.0, noise_gate="PASS"),
        ]
        passed, excluded = rank_hotspots.apply_noise_gate(rows)
        ops_passed = [r["op"] for r in passed]
        ops_excluded = [r["op"] for r in excluded]
        self.assertIn("fpca", ops_passed)
        self.assertNotIn("elastic", ops_passed)
        self.assertIn("elastic", ops_excluded)

    def test_noise_gate_flags_high_ci_width(self):
        """A criterion row with CI-width >= 5.0 (even if RSD < 5) must be excluded."""
        rows = [
            _make_row(op="depth", rsd_pct=2.0, ci_width_pct=8.0, noise_gate="FAIL-NOISY"),
            _make_row(op="fpca", rsd_pct=1.0, ci_width_pct=1.0, noise_gate="PASS"),
        ]
        passed, excluded = rank_hotspots.apply_noise_gate(rows)
        self.assertNotIn("depth", [r["op"] for r in passed])
        self.assertIn("depth", [r["op"] for r in excluded])

    def test_noise_gate_passes_clean_row(self):
        """A row with RSD < 5 AND CI-width < 5 must be in the passed set."""
        rows = [
            _make_row(op="fpca", rsd_pct=0.09, ci_width_pct=0.07, noise_gate="PASS"),
        ]
        passed, excluded = rank_hotspots.apply_noise_gate(rows)
        self.assertEqual(len(passed), 1)
        self.assertEqual(len(excluded), 0)

    def test_noise_gate_boundary_exactly_5_fails(self):
        """The gate is strict: RSD == 5.0 should be excluded (not < 5.0)."""
        rows = [
            _make_row(op="fpcr", rsd_pct=5.0, ci_width_pct=1.0, noise_gate="FAIL-NOISY"),
        ]
        passed, excluded = rank_hotspots.apply_noise_gate(rows)
        self.assertEqual(len(passed), 0)
        self.assertIn("fpcr", [r["op"] for r in excluded])

    def test_cross_language_rows_not_gated(self):
        """Python/R rows (noise_gate='NA') are NOT subjected to the noise gate."""
        rows = [
            _make_cross_row(op="fpca", library="python", n=100, m=50, mean_ns=1_500_000),
        ]
        passed, excluded = rank_hotspots.apply_noise_gate(rows)
        self.assertEqual(len(passed), 1)
        self.assertEqual(len(excluded), 0)

    def test_noise_gate_rejects_na_noise_gate_rust_rows(self):
        """Rust rows with NaN noise_gate field are treated as excluded (no noise_gate = unknown)."""
        rows = [
            _make_row(op="unknown_op", rsd_pct=None, ci_width_pct=None, noise_gate=None),
        ]
        # NaN or None noise_gate is treated conservatively: not passed
        passed, excluded = rank_hotspots.apply_noise_gate(rows)
        self.assertEqual(len(passed), 0)


class TestLeverageScore(unittest.TestCase):
    """Tests for the compute_leverage function."""

    def test_leverage_slower_than_competitor(self):
        """When fdars is slower than competitor: leverage = fdars_ns * (fdars_ns / min_other_ns)."""
        # fdars=10, min_other=2 → factor=5 → leverage=10*5=50
        result = rank_hotspots.compute_leverage(fdars_ns=10.0, min_other_ns=2.0)
        self.assertAlmostEqual(result, 50.0)

    def test_leverage_faster_than_competitor(self):
        """When fdars is fastest (fdars_ns < min_other_ns): leverage = fdars_ns * 1.0."""
        result = rank_hotspots.compute_leverage(fdars_ns=5.0, min_other_ns=10.0)
        self.assertAlmostEqual(result, 5.0)

    def test_leverage_equal_to_competitor(self):
        """When fdars_ns == min_other_ns: max(1.0, 1.0) = 1.0 → leverage = fdars_ns."""
        result = rank_hotspots.compute_leverage(fdars_ns=7.0, min_other_ns=7.0)
        self.assertAlmostEqual(result, 7.0)

    def test_leverage_guard_min_other_zero(self):
        """If min_other_ns <= 0 fall back to fdars_ns (no division by zero)."""
        result = rank_hotspots.compute_leverage(fdars_ns=1_000_000.0, min_other_ns=0.0)
        self.assertAlmostEqual(result, 1_000_000.0)

    def test_leverage_guard_negative_min_other(self):
        """Negative min_other_ns is also guarded: fallback to fdars_ns."""
        result = rank_hotspots.compute_leverage(fdars_ns=1_000_000.0, min_other_ns=-5.0)
        self.assertAlmostEqual(result, 1_000_000.0)

    def test_leverage_guard_fdars_ns_zero(self):
        """If fdars_ns <= 0 fall back to fdars_ns (which is 0 — not a divide-by-zero but still safe)."""
        result = rank_hotspots.compute_leverage(fdars_ns=0.0, min_other_ns=1_000_000.0)
        self.assertAlmostEqual(result, 0.0)

    def test_leverage_score_deterministic(self):
        """Nyquist test 2: same inputs produce identical leverage score on two calls."""
        r1 = rank_hotspots.compute_leverage(fdars_ns=3_437_700_000.0, min_other_ns=2_903_186.0)
        r2 = rank_hotspots.compute_leverage(fdars_ns=3_437_700_000.0, min_other_ns=2_903_186.0)
        self.assertEqual(r1, r2)


class TestRanking(unittest.TestCase):
    """Tests for the full ranking pipeline (noise-gate → leverage → sort)."""

    def _minimal_csv_rows(self):
        """
        Build a tiny synthetic dataset with three rust PASS rows and
        matching cross-language rows.
        """
        rows = [
            # op=fpca, rust PASS, n=100, m=50; small absolute time
            _make_row(op="fpca", n=100.0, m=50.0, mean_ns=200_000.0,
                      rsd_pct=0.1, ci_width_pct=0.1, noise_gate="PASS"),
            # op=elastic, rust PASS, n=100, m=50; large absolute time
            _make_row(op="elastic", n=100.0, m=50.0, mean_ns=300_000_000.0,
                      rsd_pct=0.4, ci_width_pct=1.6, noise_gate="PASS",
                      source="criterion:elastic_align_pair/m/100"),
            # op=noisy_op, rust FAIL-NOISY — should be excluded
            _make_row(op="noisy_op", n=100.0, m=50.0, mean_ns=500_000_000.0,
                      rsd_pct=30.0, ci_width_pct=35.0, noise_gate="FAIL-NOISY"),
            # cross-language rows
            _make_cross_row(op="fpca", library="python", n=100.0, m=50.0, mean_ns=1_300_000.0),
            _make_cross_row(op="fpca", library="r_svd",  n=100.0, m=50.0, mean_ns=950_000.0),
            _make_cross_row(op="elastic", library="python", n=100.0, m=50.0,
                            mean_ns=2_800_000.0,
                            notes="Karcher-mean elastic excluded from grid baseline: "
                                  "noise-gated (RSD 29-46%); anchor uses pairwise elastic_align_pair"),
            _make_cross_row(op="elastic", library="r_svd", n=100.0, m=50.0,
                            mean_ns=1_900_000.0,
                            notes="Karcher-mean elastic excluded from grid baseline: "
                                  "noise-gated (RSD 29-46%); anchor uses pairwise elastic_align_pair"),
        ]
        return rows

    def test_noisy_op_excluded_from_ranked_targets(self):
        """noisy_op must not appear in the ranked output."""
        rows = self._minimal_csv_rows()
        ranked, excluded = rank_hotspots.rank_rows(rows)
        ranked_ops = [r["op"] for r in ranked]
        self.assertNotIn("noisy_op", ranked_ops)

    def test_elastic_ranks_above_fpca_by_leverage(self):
        """
        elastic has a much larger fdars_ns AND is far slower than its competitors
        (fdars=300ms vs python~2.8ms → factor ~107), so its leverage >> fpca's leverage.
        It must rank first.
        """
        rows = self._minimal_csv_rows()
        ranked, _ = rank_hotspots.rank_rows(rows)
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0]["op"], "elastic",
                         f"Expected elastic at rank 1; got {[r['op'] for r in ranked]}")

    def test_ranking_is_deterministic(self):
        """Same rows produce byte-identical (same op-order) ranking on two calls."""
        rows = self._minimal_csv_rows()
        ranked1, _ = rank_hotspots.rank_rows(rows)
        ranked2, _ = rank_hotspots.rank_rows(rows)
        self.assertEqual(
            [r["op"] for r in ranked1],
            [r["op"] for r in ranked2],
        )

    def test_fpca_native_r_excluded_from_ranked_targets(self):
        """r_native_pca_fd library rows must not be used as ranked targets."""
        rows = self._minimal_csv_rows() + [
            _make_cross_row(op="fpca", library="r_native_pca_fd",
                            n=100.0, m=50.0, mean_ns=800_000.0),
        ]
        ranked, _ = rank_hotspots.rank_rows(rows)
        for r in ranked:
            self.assertNotEqual(r.get("library"), "r_native_pca_fd",
                                "r_native_pca_fd row should never be a ranked target")

    def test_elastic_anchor_from_noise_stable_source(self):
        """
        The elastic anchor in the ranked output must come from a noise-STABLE
        (PASS) elastic cell, NOT from a noise-excluded Karcher-mean cell.
        """
        rows = self._minimal_csv_rows()
        ranked, _ = rank_hotspots.rank_rows(rows)
        elastic_ranked = [r for r in ranked if r["op"] == "elastic"]
        self.assertTrue(len(elastic_ranked) > 0, "elastic should appear in ranked output")
        for r in elastic_ranked:
            self.assertEqual(r.get("noise_gate"), "PASS",
                             f"Elastic anchor in ranked output must be PASS, got: {r}")

    def test_rust_only_op_ranks_by_absolute_time(self):
        """
        An op with no cross-language rows (Rust-only) is ranked by absolute time
        (leverage = fdars_ns × 1.0).
        """
        rows = [
            _make_row(op="face_covariance", n=200.0, m=30.0,
                      mean_ns=983_000_000.0,
                      rsd_pct=0.42, ci_width_pct=1.64, noise_gate="PASS"),
        ]
        ranked, _ = rank_hotspots.rank_rows(rows)
        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0]["leverage_score"], 983_000_000.0)

    def test_sort_is_stable_descending(self):
        """Ranking is stable descending by leverage_score (larger = rank 1)."""
        rows = [
            _make_row(op="a_slow", mean_ns=1_000.0, rsd_pct=1.0, ci_width_pct=1.0,
                      noise_gate="PASS", n=100.0, m=50.0),
            _make_row(op="b_slower", mean_ns=2_000.0, rsd_pct=1.0, ci_width_pct=1.0,
                      noise_gate="PASS", n=100.0, m=50.0),
            _make_row(op="c_slowest", mean_ns=5_000.0, rsd_pct=1.0, ci_width_pct=1.0,
                      noise_gate="PASS", n=100.0, m=50.0),
        ]
        ranked, _ = rank_hotspots.rank_rows(rows)
        ops = [r["op"] for r in ranked]
        self.assertEqual(ops[0], "c_slowest")
        self.assertEqual(ops[-1], "a_slow")


if __name__ == "__main__":
    unittest.main()
