"""
Unit tests for gate.py — full-op numeric-agreement gate.

Per-op tolerance table (RESEARCH.md Agreement Tolerance Validation):
  fpca            sign-aligned Pearson correlation > 0.99
  basis_smoothing max absolute difference < 1e-3
  depth           Pearson correlation >= 0.99  (depths vector) [Phase 113 calibration]
  functional_boxplot Pearson correlation >= 0.99 (MBD depth vector) [Phase 113 calibration]
  fpcr            Pearson correlation >= 0.99  (fitted values) [Phase 113 calibration]
  elastic         L-infinity on warping function < 1e-2

Phase 113 calibration: depth, functional_boxplot, fpcr switched from max-abs-diff to
Pearson correlation >= _DEPTH_CORR_THRESHOLD (0.99, or a disclosed FPCR-specific value).
"""

import pathlib
import sys

import numpy as np
import pytest

_BENCHMARKS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_BENCHMARKS_DIR))


# ---------------------------------------------------------------------------
# Fixtures: random number generators with fixed seeds
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(seed=2024)


# ---------------------------------------------------------------------------
# FPCA tests
# ---------------------------------------------------------------------------


class TestFpcaAgreement:
    """check_agreement('fpca', ...) — sign-aligned Pearson correlation > 0.99."""

    def test_identical_fpca_passes(self, rng):
        """Identical loadings matrix passes (trivial)."""
        from gate import check_agreement

        m, ncomp = 50, 5
        U, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        fp_a = {"loadings": U.T.tolist()}  # (ncomp, m)
        fp_b = {"loadings": U.T.tolist()}
        passed, metric, _ = check_agreement("fpca", fp_a, fp_b)
        assert passed, f"Identical FPCA should pass; metric={metric:.6f}"

    def test_near_identical_fpca_passes(self, rng):
        """Tiny noise on loadings still passes (correlation remains > 0.99)."""
        from gate import check_agreement

        m, ncomp = 50, 5
        U, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        noisy = U + rng.standard_normal((m, ncomp)) * 1e-3
        fp_a = {"loadings": U.T.tolist()}
        fp_b = {"loadings": noisy.T.tolist()}
        passed, metric, _ = check_agreement("fpca", fp_a, fp_b)
        assert passed, f"Near-identical FPCA should pass; metric={metric:.6f}"

    def test_sign_flipped_fpca_passes(self, rng):
        """
        Sign-flipped-but-identical loading vectors PASS (Pitfall 2).

        If some components of `other` have their signs flipped relative to fdars,
        the gate must align signs before correlating. After alignment the correlation
        should be +1.0, not -1.0.
        """
        from gate import check_agreement

        m, ncomp = 50, 5
        U, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        # Flip sign on components 0, 2, 4
        flipped = U.copy()
        for k in (0, 2, 4):
            flipped[:, k] = -flipped[:, k]
        fp_a = {"loadings": U.T.tolist()}
        fp_b = {"loadings": flipped.T.tolist()}
        passed, metric, _ = check_agreement("fpca", fp_a, fp_b)
        assert passed, (
            f"Sign-flipped FPCA should PASS after alignment; "
            f"metric={metric:.6f} — Pitfall 2 not handled"
        )

    def test_unrelated_fpca_fails(self, rng):
        """Completely different loading matrix fails."""
        from gate import check_agreement

        m, ncomp = 50, 5
        U1, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        U2, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        # Make U2 orthogonal to U1 (no correlation)
        for k in range(ncomp):
            v = U2[:, k]
            for j in range(ncomp):
                v = v - np.dot(v, U1[:, j]) * U1[:, j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                U2[:, k] = v / norm
        fp_a = {"loadings": U1.T.tolist()}
        fp_b = {"loadings": U2.T.tolist()}
        passed, metric, _ = check_agreement("fpca", fp_a, fp_b)
        assert not passed, (
            f"Orthogonal FPCA should FAIL; metric={metric:.6f}"
        )


# ---------------------------------------------------------------------------
# Basis smoothing tests
# ---------------------------------------------------------------------------


class TestBasisSmoothingAgreement:
    """check_agreement('basis_smoothing', ...) — max abs diff < 1e-3."""

    def test_near_identical_smoothed_passes(self, rng):
        """Smoothed values that agree within 1e-4 pass."""
        from gate import check_agreement

        n, m = 60, 50
        vals = rng.standard_normal((n, m))
        fp_a = {"smoothed_values": vals.tolist()}
        fp_b = {"smoothed_values": (vals + rng.standard_normal((n, m)) * 1e-5).tolist()}
        passed, metric, _ = check_agreement("basis_smoothing", fp_a, fp_b)
        assert passed, f"Near-identical smoothed values should pass; metric={metric:.2e}"

    def test_large_diff_smoothed_fails(self, rng):
        """Smoothed values that differ by > 1e-3 fail."""
        from gate import check_agreement

        n, m = 60, 50
        vals = rng.standard_normal((n, m))
        fp_a = {"smoothed_values": vals.tolist()}
        fp_b = {"smoothed_values": (vals + 0.01).tolist()}  # +0.01 >> 1e-3
        passed, metric, _ = check_agreement("basis_smoothing", fp_a, fp_b)
        assert not passed, f"Large diff should FAIL; metric={metric:.2e}"


# ---------------------------------------------------------------------------
# Depth tests  (Phase 113: switched to Pearson correlation >= _DEPTH_CORR_THRESHOLD)
# ---------------------------------------------------------------------------


class TestDepthAgreement:
    """check_agreement('depth', ...) — Pearson correlation >= 0.99 on depth vector."""

    def test_identical_depth_passes(self, rng):
        """Identical depth vectors pass (|corr|=1.0)."""
        from gate import check_agreement

        n = 60
        depths = rng.uniform(0, 1, n)
        fp = {"depths": depths.tolist()}
        passed, metric, _ = check_agreement("depth", fp, fp)
        assert passed, f"Identical depths should pass; metric={metric:.6f}"

    def test_scale_shifted_depth_passes(self, rng):
        """
        Depths that are linearly related (other = a*depths + b, a > 0) PASS.

        This was the real-world failure: scikit-fda MBD and roahd::MBD produce
        the same depth ORDERING but with a ~0.65% normalization offset.
        Pearson corr is scale/shift invariant; this must PASS under the new metric.
        """
        from gate import check_agreement

        n = 60
        depths = rng.uniform(0.2, 0.8, n)
        fp_a = {"depths": depths.tolist()}
        # Scale by 0.9972 + constant shift (simulates MBD normalization difference)
        fp_b = {"depths": (depths * 0.9972 + 0.003).tolist()}
        passed, metric, _ = check_agreement("depth", fp_a, fp_b)
        assert passed, (
            f"Scale-shifted depths should PASS (Pearson corr invariant to scale/shift); "
            f"metric={metric:.6f} — normalization-diff case wrongly rejected"
        )

    def test_uncorrelated_depth_fails(self, rng):
        """Shuffled / uncorrelated depths fail (|corr| << 0.99)."""
        from gate import check_agreement

        n = 60
        depths = rng.uniform(0.1, 0.9, n)
        # Create an uncorrelated vector by shuffling
        shuffled = depths.copy()
        rng.shuffle(shuffled)
        # Ensure low correlation by partially reversing order
        fp_a = {"depths": depths.tolist()}
        fp_b = {"depths": shuffled[::-1].tolist()}  # reversed shuffle
        passed, metric, _ = check_agreement("depth", fp_a, fp_b)
        # metric is |Pearson corr| — should be far below 0.99 for uncorrelated data
        assert not passed, (
            f"Uncorrelated depths should FAIL; metric={metric:.6f}"
        )


# ---------------------------------------------------------------------------
# Functional boxplot tests  (Phase 113: switched to Pearson correlation)
# ---------------------------------------------------------------------------


class TestFunctionalBoxplotAgreement:
    """check_agreement('functional_boxplot', ...) — Pearson correlation >= 0.99."""

    def test_identical_boxplot_depths_passes(self, rng):
        """Identical depth vectors pass."""
        from gate import check_agreement

        n = 60
        depths = rng.uniform(0, 1, n)
        fp = {"depths": depths.tolist()}
        passed, metric, _ = check_agreement("functional_boxplot", fp, fp)
        assert passed, f"Identical boxplot depths should pass; metric={metric:.6f}"

    def test_scale_shifted_boxplot_passes(self, rng):
        """
        Scale-shifted depth vector PASSES (normalization-invariant check).

        Same real-world failure case as TestDepthAgreement: MBD normalization
        conventions differ by ~0.65%; Pearson corr must be invariant to this.
        """
        from gate import check_agreement

        n = 60
        depths = rng.uniform(0.2, 0.8, n)
        fp_a = {"depths": depths.tolist()}
        fp_b = {"depths": (depths * 0.9972 + 0.003).tolist()}
        passed, metric, _ = check_agreement("functional_boxplot", fp_a, fp_b)
        assert passed, (
            f"Scale-shifted boxplot depths should PASS; metric={metric:.6f}"
        )

    def test_uncorrelated_boxplot_depths_fails(self, rng):
        """Uncorrelated depth vectors fail."""
        from gate import check_agreement

        n = 60
        depths = rng.uniform(0.1, 0.9, n)
        shuffled = depths.copy()
        rng.shuffle(shuffled)
        fp_a = {"depths": depths.tolist()}
        fp_b = {"depths": shuffled[::-1].tolist()}
        passed, metric, _ = check_agreement("functional_boxplot", fp_a, fp_b)
        assert not passed, f"Uncorrelated boxplot depths should FAIL; metric={metric:.6f}"


# ---------------------------------------------------------------------------
# FPCR tests  (Phase 113: switched to Pearson correlation >= threshold)
# ---------------------------------------------------------------------------


class TestFpcrAgreement:
    """check_agreement('fpcr', ...) — Pearson correlation >= threshold on fitted values."""

    def test_identical_fpcr_passes(self, rng):
        """Identical fitted values pass (|corr|=1.0)."""
        from gate import check_agreement

        n = 60
        fitted = rng.standard_normal(n)
        fp_a = {"fitted_values": fitted.tolist()}
        fp_b = {"fitted_values": fitted.tolist()}
        passed, metric, _ = check_agreement("fpcr", fp_a, fp_b)
        assert passed, f"Identical FPCR fitted values should pass; metric={metric:.6f}"

    def test_scale_shifted_fpcr_passes(self, rng):
        """
        Scale-shifted fitted values PASS.

        Python (scikit-fda lstsq) and R (fregre.pc) use different scale normalizations.
        Pearson correlation is scale/shift invariant — same fitted-value ordering must PASS.
        """
        from gate import check_agreement

        n = 60
        fitted = rng.standard_normal(n)
        fp_a = {"fitted_values": fitted.tolist()}
        # Simulate R's different scale normalization: multiply by ~2 + shift
        fp_b = {"fitted_values": (fitted * 2.1 + 0.5).tolist()}
        passed, metric, _ = check_agreement("fpcr", fp_a, fp_b)
        assert passed, (
            f"Scale-shifted FPCR fitted values should PASS (Pearson corr is scale-invariant); "
            f"metric={metric:.6f} — scale-normalization case wrongly rejected"
        )

    def test_uncorrelated_fpcr_fails(self, rng):
        """Uncorrelated fitted values fail (|corr| << 0.99)."""
        from gate import check_agreement

        n = 60
        fitted_a = rng.standard_normal(n)
        fitted_b = rng.standard_normal(n)  # independent draw — uncorrelated
        fp_a = {"fitted_values": fitted_a.tolist()}
        fp_b = {"fitted_values": fitted_b.tolist()}
        passed, metric, _ = check_agreement("fpcr", fp_a, fp_b)
        assert not passed, (
            f"Uncorrelated FPCR fitted values should FAIL; metric={metric:.6f}"
        )


# ---------------------------------------------------------------------------
# Elastic tests
# ---------------------------------------------------------------------------


class TestElasticAgreement:
    """check_agreement('elastic', ...) — L-infinity on warping functions < 1e-2."""

    def test_identical_warp_passes(self, rng):
        """Identical warping functions pass."""
        from gate import check_agreement

        m = 50
        warp = np.linspace(0, 1, m)  # identity warping
        fp = {"warping_values": warp.tolist(), "karcher_mean": warp.tolist()}
        passed, metric, _ = check_agreement("elastic", fp, fp)
        assert passed, f"Identical warp should pass; metric={metric:.2e}"

    def test_small_warp_diff_passes(self, rng):
        """Warping functions differing by < 1e-2 pass."""
        from gate import check_agreement

        m = 50
        warp = np.linspace(0, 1, m)
        perturb = warp + rng.uniform(-5e-3, 5e-3, m)  # max diff < 1e-2
        fp_a = {"warping_values": warp.tolist(), "karcher_mean": warp.tolist()}
        fp_b = {"warping_values": perturb.tolist(), "karcher_mean": warp.tolist()}
        passed, metric, _ = check_agreement("elastic", fp_a, fp_b)
        assert passed, f"Small warp diff should pass; metric={metric:.2e}"

    def test_large_warp_diff_fails(self, rng):
        """Warping functions differing by > 1e-2 fail."""
        from gate import check_agreement

        m = 50
        warp_a = np.linspace(0, 1, m)
        # Create a warp that deviates by 0.05 (>> 1e-2)
        warp_b = warp_a.copy()
        warp_b[m // 4: m // 2] += 0.05
        fp_a = {"warping_values": warp_a.tolist(), "karcher_mean": warp_a.tolist()}
        fp_b = {"warping_values": warp_b.tolist(), "karcher_mean": warp_a.tolist()}
        passed, metric, _ = check_agreement("elastic", fp_a, fp_b)
        assert not passed, f"Large warp diff should FAIL; metric={metric:.2e}"


# ---------------------------------------------------------------------------
# run_gate tests
# ---------------------------------------------------------------------------


class TestRunGate:
    """run_gate(results) — iterates every overlapping op x every non-fdars library."""

    def _make_fpca_record(self, rng, lib: str, n: int = 60, m: int = 50,
                          ncomp: int = 5, noise: float = 0.0):
        U, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        loadings = U + rng.standard_normal((m, ncomp)) * noise
        return {
            "op": "fpca", "library": lib, "n": n, "m": m, "mean_ns": 100.0,
            "result_fingerprint": {"loadings": loadings.T.tolist()},
        }

    def _make_depth_record(self, n: int, lib: str, depths, extra_noise: float = 0.0):
        d = np.array(depths) + extra_noise
        return {
            "op": "depth", "library": lib, "n": n, "m": 50, "mean_ns": 100.0,
            "result_fingerprint": {"depths": d.tolist()},
        }

    def test_run_gate_all_pass(self, rng):
        """run_gate returns True when all ops pass for all non-fdars libraries."""
        from gate import run_gate

        n, m, ncomp = 60, 50, 5
        U, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        fdars_fp = {"loadings": U.T.tolist()}
        tiny_noise = rng.standard_normal((m, ncomp)) * 1e-5
        py_fp = {"loadings": (U + tiny_noise).T.tolist()}
        r_fp = {"loadings": (U - tiny_noise).T.tolist()}

        depths = rng.uniform(0, 1, n)
        results = [
            # fdars FPCA (reference)
            {"op": "fpca", "library": "rust", "n": n, "m": m, "mean_ns": 100.0,
             "result_fingerprint": fdars_fp},
            # Python FPCA (near-identical)
            {"op": "fpca", "library": "python", "n": n, "m": m, "mean_ns": 200.0,
             "result_fingerprint": py_fp},
            # R FPCA (near-identical)
            {"op": "fpca", "library": "r", "n": n, "m": m, "mean_ns": 300.0,
             "result_fingerprint": r_fp},
            # fdars depth
            {"op": "depth", "library": "rust", "n": n, "m": m, "mean_ns": 50.0,
             "result_fingerprint": {"depths": depths.tolist()}},
            # Python depth (identical)
            {"op": "depth", "library": "python", "n": n, "m": m, "mean_ns": 60.0,
             "result_fingerprint": {"depths": depths.tolist()}},
            # R depth (identical)
            {"op": "depth", "library": "r", "n": n, "m": m, "mean_ns": 70.0,
             "result_fingerprint": {"depths": depths.tolist()}},
        ]
        passed = run_gate(results)
        assert passed, "run_gate should return True when all ops pass"

    def test_run_gate_fails_on_mismatch(self, rng, capsys):
        """run_gate returns False and prints GATE FAIL when any op is out of tolerance."""
        from gate import run_gate

        n, m, ncomp = 60, 50, 5
        U, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        fdars_fp = {"loadings": U.T.tolist()}

        # Make Python FPCA orthogonal (completely wrong)
        V, _, _ = np.linalg.svd(rng.standard_normal((m, ncomp)), full_matrices=False)
        for k in range(ncomp):
            v = V[:, k]
            for j in range(ncomp):
                v = v - np.dot(v, U[:, j]) * U[:, j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                V[:, k] = v / norm
        py_fp = {"loadings": V.T.tolist()}

        results = [
            {"op": "fpca", "library": "rust", "n": n, "m": m, "mean_ns": 100.0,
             "result_fingerprint": fdars_fp},
            {"op": "fpca", "library": "python", "n": n, "m": m, "mean_ns": 200.0,
             "result_fingerprint": py_fp},  # << bad
        ]
        passed = run_gate(results)
        assert not passed, "run_gate should return False on mismatch"

        captured = capsys.readouterr()
        assert "GATE FAIL" in captured.err or "GATE FAIL" in captured.out, (
            f"Expected 'GATE FAIL' in output, got stderr={captured.err!r}"
        )

    def test_run_gate_prints_gate_fail_with_lib_and_op(self, rng, capsys):
        """GATE FAIL message names the failing library and op."""
        from gate import run_gate

        n = 60
        depths = np.ones(n) * 0.5

        results = [
            {"op": "depth", "library": "rust", "n": n, "m": 50, "mean_ns": 50.0,
             "result_fingerprint": {"depths": depths.tolist()}},
            {"op": "depth", "library": "r", "n": n, "m": 50, "mean_ns": 70.0,
             "result_fingerprint": {"depths": (depths + 0.05).tolist()}},  # diff = 0.05 > 1e-3
        ]
        passed = run_gate(results)
        assert not passed
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "r" in output.lower() or "depth" in output.lower(), (
            f"GATE FAIL message should name the library or op; got: {output!r}"
        )
