"""
Unit tests for render_report.py — the Phase 116 perf-report generated-table
renderer (analysis core in Task 1, emitters + CLI in Task 2).

Mirrors benchmarks/test_rank_hotspots.py's style: plain dict-row fixtures,
unittest.TestCase grouped by concern, late same-directory import.
"""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Row fixtures matching the comparison.csv schema
# ---------------------------------------------------------------------------


def _make_row(
    op="fpca",
    library="rust",
    n=100.0,
    m=50.0,
    mean_ns=1_000_000.0,
    ci_lower_ns=None,
    ci_upper_ns=None,
    std_dev_ns=None,
    rsd_pct=1.0,
    ci_width_pct=1.0,
    noise_gate="PASS",
    source="criterion:test/n100_m50",
    notes="",
):
    """Return a dict with the comparison.csv schema (rust-shaped defaults)."""
    return dict(
        op=op,
        library=library,
        n=n,
        m=m,
        mean_ns=mean_ns,
        ci_lower_ns=ci_lower_ns,
        ci_upper_ns=ci_upper_ns,
        std_dev_ns=std_dev_ns,
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


def _make_baseline_row(**kwargs):
    """Alias of _make_row for before/after-diff test readability."""
    return _make_row(**kwargs)


def _make_estimate(mean_ns, lower_ns, upper_ns, std_dev_ns):
    """Return a criterion estimates.json-shaped dict (nanoseconds)."""
    return {
        "mean": {
            "point_estimate": mean_ns,
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": lower_ns,
                "upper_bound": upper_ns,
            },
        },
        "std_dev": {"point_estimate": std_dev_ns},
    }


# ---------------------------------------------------------------------------
# Import under test (late import so we can test errors from the module)
# ---------------------------------------------------------------------------

import render_report as r  # noqa: E402  (module in same directory)


class TestLoadRows(unittest.TestCase):
    def test_load_rows_keeps_literal_na_and_nulls_empty_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "cmp.csv"
            path.write_text(
                "op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,"
                "rsd_pct,ci_width_pct,noise_gate,source,notes\n"
                "fpca,python,100.0,50.0,1332059.5,,,,,,NA,python_runner,\n",
                encoding="utf-8",
            )
            rows = r.load_rows(path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["noise_gate"], "NA")
            self.assertIsNone(row["ci_lower_ns"])
            self.assertIsNone(row["ci_upper_ns"])

    def test_load_rows_rust_numeric_fields_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "cmp.csv"
            path.write_text(
                "op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,"
                "rsd_pct,ci_width_pct,noise_gate,source,notes\n"
                "fpca,rust,100.0,50.0,1000000.0,980000.0,1020000.0,20000.0,"
                "2.0,4.0,PASS,criterion:crosslang_fpca/n100_m50,\n",
                encoding="utf-8",
            )
            rows = r.load_rows(path)
            row = rows[0]
            self.assertEqual(row["noise_gate"], "PASS")
            self.assertEqual(float(row["mean_ns"]), 1000000.0)


class TestNoiseStatus(unittest.TestCase):
    def test_rust_pass(self):
        row = _make_row(rsd_pct=4.9, ci_width_pct=4.9, noise_gate="PASS")
        self.assertEqual(r.noise_status(row), "PASS")

    def test_rust_fail_at_boundary(self):
        row = _make_row(rsd_pct=5.0, ci_width_pct=1.0, noise_gate="PASS")
        self.assertEqual(r.noise_status(row), "FAIL-NOISY")

    def test_python_na(self):
        row = _make_cross_row("fpca", "python", 100.0, 50.0, 1_000_000.0)
        self.assertEqual(r.noise_status(row), "NA")

    def test_rust_nan_rsd_fails(self):
        row = _make_row(rsd_pct=float("nan"), ci_width_pct=1.0, noise_gate="PASS")
        self.assertEqual(r.noise_status(row), "FAIL-NOISY")


class TestSignificance(unittest.TestCase):
    def test_disjoint_is_significant(self):
        self.assertTrue(r.is_significant((1, 2), (3, 4)))

    def test_overlapping_not_significant(self):
        self.assertFalse(r.is_significant((1, 3), (2, 4)))

    def test_touching_not_significant(self):
        self.assertFalse(r.is_significant((1, 2), (2, 3)))

    def test_missing_ci_is_none(self):
        self.assertIsNone(r.is_significant(None, (2, 3)))
        self.assertIsNone(r.is_significant((1, 2), None))


class TestBeforeAfterPairing(unittest.TestCase):
    def test_paired_rust_row_computes_delta(self):
        before = [_make_baseline_row(mean_ns=1_000_000.0, source="criterion:audit_fpca/n100_m50")]
        after = [_make_row(mean_ns=900_000.0, source="criterion:audit_fpca/n100_m50")]
        pairs = r.pair_before_after(before, after)
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual(p["status"], "paired")
        self.assertAlmostEqual(p["delta_pct"], -10.0)
        self.assertAlmostEqual(p["speedup"], 1_000_000.0 / 900_000.0)

    def test_stale_identical(self):
        before = [_make_baseline_row(mean_ns=1_000_000.0, source="criterion:audit_fpca/n100_m50")]
        after = [_make_row(mean_ns=1_000_000.0, source="criterion:audit_fpca/n100_m50")]
        pairs = r.pair_before_after(before, after)
        self.assertEqual(pairs[0]["status"], "stale-identical")

    def test_missing_after(self):
        before = [_make_baseline_row(mean_ns=1_000_000.0, source="criterion:orphan_group/n1")]
        after = []
        pairs = r.pair_before_after(before, after)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["status"], "missing-after")

    def test_new_after(self):
        before = []
        after = [_make_row(mean_ns=1_000_000.0, source="criterion:crosslang_fpca/n100_m50")]
        pairs = r.pair_before_after(before, after)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["status"], "new-after")

    def test_python_pairing_by_op_library_n_m(self):
        before = [_make_cross_row("fpca", "python", 100.0, 50.0, 1_000_000.0)]
        after = [_make_cross_row("fpca", "python", 100.0, 50.0, 950_000.0)]
        pairs = r.pair_before_after(before, after)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["status"], "paired")


class TestMissingClassification(unittest.TestCase):
    def _pair(self, source="criterion:functional_boxplot/N/50", library="rust", op="functional_boxplot", n=None, m=None):
        return {
            "source": source, "library": library, "op": op, "n": n, "m": m,
            "status": "missing-after",
        }

    def test_orphan_group(self):
        pair = self._pair(source="criterion:coclustering_co_cluster/n1")
        result = r.classify_missing(pair, defined=set(), rerun_log=None)
        self.assertEqual(result["status"], "orphan")

    def test_timeout(self):
        pair = self._pair()
        rerun_log = {
            "records": [
                {"kind": "rust_chunk", "filter": "^functional_boxplot/", "status": "timeout", "limit_s": 540},
            ]
        }
        result = r.classify_missing(pair, defined={"functional_boxplot"}, rerun_log=rerun_log)
        self.assertEqual(result["status"], "timeout")

    def test_inferred_timeout(self):
        pair = self._pair()
        rerun_log = {
            "records": [
                {"kind": "rust_chunk", "filter": "^functional_boxplot/", "status": "skipped-inferred-timeout"},
            ]
        }
        result = r.classify_missing(pair, defined={"functional_boxplot"}, rerun_log=rerun_log)
        self.assertEqual(result["status"], "inferred-timeout")

    def test_error(self):
        pair = self._pair()
        rerun_log = {"records": [{"kind": "rust_chunk", "filter": "^functional_boxplot/", "status": "error"}]}
        result = r.classify_missing(pair, defined={"functional_boxplot"}, rerun_log=rerun_log)
        self.assertEqual(result["status"], "error")

    def test_id_not_produced(self):
        pair = self._pair()
        rerun_log = {"records": [{"kind": "rust_chunk", "filter": "^functional_boxplot/", "status": "ok"}]}
        result = r.classify_missing(pair, defined={"functional_boxplot"}, rerun_log=rerun_log)
        self.assertEqual(result["status"], "id-not-produced")

    def test_grid_cell_timeout_for_python(self):
        pair = self._pair(source=None, library="python", op="fpca", n=500.0, m=50.0)
        rerun_log = {"records": [{"kind": "grid_cell", "n": 500, "m": 50, "status": "timeout"}]}
        result = r.classify_missing(pair, defined=set(), rerun_log=rerun_log)
        self.assertEqual(result["status"], "timeout")

    def test_not_run_when_no_match(self):
        pair = self._pair()
        result = r.classify_missing(pair, defined={"functional_boxplot"}, rerun_log={"records": []})
        self.assertEqual(result["status"], "not-run")

    def test_not_run_with_no_log(self):
        pair = self._pair(source=None, library="python", op="fpca", n=500.0, m=50.0)
        result = r.classify_missing(pair, defined=set(), rerun_log=None)
        self.assertEqual(result["status"], "not-run")


class TestScalingExponent(unittest.TestCase):
    def test_quadratic_power_law_recovered(self):
        xs = [100, 300, 500]
        c = 3.0
        ys = [c * x**2 for x in xs]
        slope = r.fit_scaling_exponent(xs, ys)
        self.assertAlmostEqual(slope, 2.0, places=9)

    def test_two_points_work(self):
        slope = r.fit_scaling_exponent([100, 500], [3.0 * 100**2, 3.0 * 500**2])
        self.assertAlmostEqual(slope, 2.0, places=9)

    def test_fewer_than_two_distinct_x_is_none(self):
        self.assertIsNone(r.fit_scaling_exponent([100], [300.0]))
        self.assertIsNone(r.fit_scaling_exponent([100, 100], [300.0, 300.0]))

    def test_non_positive_y_is_none(self):
        self.assertIsNone(r.fit_scaling_exponent([100, 200], [0.0, 5.0]))
        self.assertIsNone(r.fit_scaling_exponent([100, 200], [-1.0, 5.0]))


class TestTheory(unittest.TestCase):
    def test_covers_every_crosslang_op(self):
        self.assertEqual(set(r.THEORETICAL_COMPLEXITY), set(r.CROSSLANG_OPS))

    def test_every_entry_has_nonempty_expr_and_real_source(self):
        for op, entry in r.THEORETICAL_COMPLEXITY.items():
            self.assertTrue(entry["expr"], msg=f"{op} has empty expr")
            source_file = entry["source"].split("::")[0]
            self.assertTrue(
                pathlib.Path(source_file).exists(),
                msg=f"{op} source file does not exist: {source_file}",
            )


class TestAbStats(unittest.TestCase):
    def test_speedup_and_significance(self):
        before = _make_estimate(1_000_000.0, 900_000.0, 1_100_000.0, 100_000.0)
        after = _make_estimate(500_000.0, 495_000.0, 505_000.0, 10_000.0)
        stats = r.ab_entry_stats(before, after)
        self.assertAlmostEqual(stats["speedup"], 2.0)
        self.assertAlmostEqual(stats["delta_pct"], -50.0)
        self.assertTrue(stats["significant"])
        self.assertEqual(stats["after"]["noise_gate"], "PASS")

    def test_headline_eligible_requires_pass_and_significant(self):
        # Noisy before (RSD > 5%) -> not eligible even though CIs are disjoint.
        before = _make_estimate(1_000_000.0, 700_000.0, 1_300_000.0, 200_000.0)
        after = _make_estimate(500_000.0, 480_000.0, 520_000.0, 10_000.0)
        stats = r.ab_entry_stats(before, after)
        self.assertFalse(stats["headline_eligible"])

    def test_select_headline_ab_prefers_phase116(self):
        manifest = {
            "entries": [
                {"id": "phase115_face_covariance_n200_m30", "kind": "ab",
                 "criterion_id": "perf_face_covariance/n200_m30"},
                {"id": "phase116_face_covariance_n200_m30", "kind": "ab",
                 "criterion_id": "perf_face_covariance/n200_m30"},
            ]
        }
        entry = r.select_headline_ab(manifest)
        self.assertEqual(entry["id"], "phase116_face_covariance_n200_m30")

    def test_select_headline_ab_falls_back(self):
        manifest = {
            "entries": [
                {"id": "phase115_face_covariance_n200_m30", "kind": "ab",
                 "criterion_id": "perf_face_covariance/n200_m30"},
            ]
        }
        entry = r.select_headline_ab(manifest)
        self.assertEqual(entry["id"], "phase115_face_covariance_n200_m30")

    def test_select_headline_ab_none_when_absent(self):
        self.assertIsNone(r.select_headline_ab({"entries": []}))

    def test_load_estimates_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                r.load_estimates(tmp, "/etc/passwd")

    def test_load_estimates_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                r.load_estimates(tmp, "../../etc/passwd")

    def test_load_estimates_reads_valid_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = pathlib.Path(tmp) / "entry"
            sub.mkdir()
            est_path = sub / "before.json"
            est_path.write_text(json.dumps(_make_estimate(1.0, 0.9, 1.1, 0.1)), encoding="utf-8")
            data = r.load_estimates(tmp, "entry/before.json")
            self.assertEqual(data["mean"]["point_estimate"], 1.0)


class TestCriterionSettings(unittest.TestCase):
    def test_parses_both_duration_forms(self):
        text = """
fn bench_a(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_fpca");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.finish();
}

fn bench_b(c: &mut Criterion) {
    let mut group = c.benchmark_group("functional_boxplot");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    group.finish();
}
"""
        settings = r.parse_criterion_settings(text)
        self.assertEqual(settings["crosslang_fpca"], {
            "sample_size": 30, "measurement_time_s": 10, "warm_up_s": 3,
        })
        self.assertEqual(settings["functional_boxplot"], {
            "sample_size": 50, "measurement_time_s": 20, "warm_up_s": 5,
        })

    def test_group_without_settings_maps_to_none(self):
        text = """
fn bench_c(c: &mut Criterion) {
    let mut group = c.benchmark_group("plain_group");
    group.finish();
}
"""
        settings = r.parse_criterion_settings(text)
        self.assertEqual(settings["plain_group"], {
            "sample_size": None, "measurement_time_s": None, "warm_up_s": None,
        })


class TestMdCell(unittest.TestCase):
    def test_escapes_pipe(self):
        self.assertEqual(r._md_cell("a|b"), "a\\|b")

    def test_strips_newlines(self):
        self.assertEqual(r._md_cell("a\r\nb\nc\rd"), "a b c d")

    def test_none_and_nan_are_em_dash(self):
        self.assertEqual(r._md_cell(None), r._EM_DASH)
        self.assertEqual(r._md_cell(float("nan")), r._EM_DASH)

    def test_plain_value_passthrough(self):
        self.assertEqual(r._md_cell(42), "42")


if __name__ == "__main__":
    unittest.main()
