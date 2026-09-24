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


# ---------------------------------------------------------------------------
# Task 2 fixtures — a full synthetic input tree (baseline/after CSV, ab
# manifest + estimates, env.json pair, rerun_log.json, a fake bench_dir, and
# marker-carrying doc/README files). Never depends on the real re-run having
# happened (116-04-PLAN.md <context>).
# ---------------------------------------------------------------------------

_BASELINE_ROWS_CSV = """op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes
fpca,rust,100.0,50.0,1000000.0,980000.0,1020000.0,20000.0,2.0,4.0,PASS,criterion:audit_fpca/n100_m50,
fake_op,rust,,,500000.0,480000.0,520000.0,10000.0,2.0,4.0,PASS,criterion:orphan_group/n1,
face_covariance,rust,200.0,30.0,2000000.0,1960000.0,2040000.0,40000.0,2.0,4.0,PASS,criterion:perf_face_covariance/n200_m30,
scaling_op,rust,1.0,1.0,3000000.0,2900000.0,3100000.0,60000.0,2.0,6.67,FAIL-NOISY,criterion:perf_parallelism_co_cluster/n1,
fpca,python,100.0,50.0,1200000.0,,,,,,NA,python_runner,
fpca,r_svd,100.0,50.0,1300000.0,,,,,,NA,r_runner,
"""

_AFTER_ROWS_CSV = """op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes
fpca,rust,100.0,50.0,800000.0,785000.0,815000.0,8000.0,1.0,3.75,PASS,criterion:audit_fpca/n100_m50,
fake_op,rust,,,500000.0,480000.0,520000.0,10000.0,2.0,4.0,PASS,criterion:orphan_group/n1,
face_covariance,rust,200.0,30.0,1200000.0,1180000.0,1220000.0,24000.0,2.0,3.33,PASS,criterion:perf_face_covariance/n200_m30,
scaling_op,rust,1.0,1.0,2900000.0,2800000.0,3000000.0,58000.0,2.0,6.9,FAIL-NOISY,criterion:perf_parallelism_co_cluster/n1,
fpca,rust,100.0,50.0,120000.0,118000.0,122000.0,1200.0,1.0,3.33,PASS,criterion:crosslang_fpca/n100_m50,
fpca,rust,500.0,200.0,22000000.0,21500000.0,22500000.0,220000.0,1.0,4.5,PASS,criterion:crosslang_fpca/n500_m200,
fpca,python,100.0,50.0,1150000.0,,,,,,NA,python_runner,
fpca,r_svd,100.0,50.0,1250000.0,,,,,,NA,r_runner,
fpca,python,500.0,200.0,18000000.0,,,,,,NA,python_runner,
fpca,r_svd,500.0,200.0,79000000.0,,,,,,NA,r_runner,
"""

_BENCH_FILE_RS = """
fn bench_fpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_fpca");
    group.finish();
}

fn bench_face_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_face_covariance");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    group.finish();
}

fn bench_co_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_parallelism_co_cluster");
    group.finish();
}

fn bench_crosslang_fpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_fpca");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.finish();
}
"""
# Deliberately no `benchmark_group("orphan_group")` — that baseline row must
# classify as "orphan" in the not-rerun block.


def _write_fixture(root: pathlib.Path) -> dict:
    results = root / "results"
    results.mkdir()
    ab_dir = results / "ab"
    ab_dir.mkdir()
    bench_dir = root / "benches"
    bench_dir.mkdir()

    baseline_path = results / "comparison_baseline.csv"
    baseline_path.write_text(_BASELINE_ROWS_CSV, encoding="utf-8")
    after_path = results / "comparison.csv"
    after_path.write_text(_AFTER_ROWS_CSV, encoding="utf-8")

    env_path = results / "env.json"
    env_path.write_text(json.dumps({
        "cpu": {"model": "Test CPU"},
        "thread_env": {"RAYON_NUM_THREADS": "1"},
    }), encoding="utf-8")
    env_baseline_path = results / "env_baseline.json"
    env_baseline_path.write_text(json.dumps({
        "cpu": {"model": "Test CPU (old)"},
        "thread_env": {"RAYON_NUM_THREADS": "4"},
    }), encoding="utf-8")

    rerun_log_path = results / "rerun_log.json"
    rerun_log_path.write_text(json.dumps({
        "meta": {"started_utc": "2026-09-24T00:00:00Z", "head_sha": "abc1234"},
        "records": [
            {"kind": "rust_chunk", "status": "ok", "wall_s": 100.0,
             "chunk_idx": 0, "filter": "^crosslang_fpca/"},
            {"kind": "footprint", "status": "ok", "wall_s": 0.1,
             "target_bytes": 1073741824, "target_criterion_bytes": 500000000,
             "target_release_bytes": 200000000},
        ],
    }), encoding="utf-8")

    (bench_dir / "fake_benchmarks.rs").write_text(_BENCH_FILE_RS, encoding="utf-8")

    headline_entry_dir = ab_dir / "phase116_face_covariance_n200_m30"
    headline_entry_dir.mkdir()
    (headline_entry_dir / "before.json").write_text(
        json.dumps(_make_estimate(2_000_000.0, 1_960_000.0, 2_040_000.0, 40_000.0)),
        encoding="utf-8",
    )
    (headline_entry_dir / "after.json").write_text(
        json.dumps(_make_estimate(600_000.0, 590_000.0, 610_000.0, 6_000.0)),
        encoding="utf-8",
    )

    manifest_path = ab_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "entries": [
            {
                "id": "phase116_face_covariance_n200_m30", "phase": "116", "kind": "ab",
                "label": "test headline entry", "criterion_id": "perf_face_covariance/n200_m30",
                "before": "phase116_face_covariance_n200_m30/before.json",
                "after": "phase116_face_covariance_n200_m30/after.json",
                "sha_before": "aaa1111", "sha_after": "bbb2222",
                "thread_config": "pinned: all six *_NUM_THREADS=1",
            },
        ],
    }), encoding="utf-8")

    doc_path = root / "PERFORMANCE.md"
    doc_lines = ["# Test Performance Doc", ""]
    for name in r.BLOCKS:
        begin, end = r.block_markers(name)
        doc_lines += [f"## {name}", begin, end, ""]
    doc_path.write_text("\n".join(doc_lines), encoding="utf-8")

    readme_path = root / "README.md"
    begin, end = r.block_markers("headline")
    readme_path.write_text(
        f"# Test Project\n\n## Performance\n\nSome hand-written prose.\n\n{begin}\n{end}\n\n"
        f"## Another Section\n\nUnrelated content.\n",
        encoding="utf-8",
    )

    return {
        "baseline": str(baseline_path),
        "after": str(after_path),
        "env": str(env_path),
        "env_baseline": str(env_baseline_path),
        "rerun_log": str(rerun_log_path),
        "ab_manifest": str(manifest_path),
        "bench_dir": str(bench_dir),
        "doc": str(doc_path),
        "readme": str(readme_path),
    }


def _cli_args(paths: dict, extra: list | None = None) -> list:
    args = [
        "--baseline", paths["baseline"],
        "--after", paths["after"],
        "--env", paths["env"],
        "--env-baseline", paths["env_baseline"],
        "--rerun-log", paths["rerun_log"],
        "--ab-manifest", paths["ab_manifest"],
        "--doc", paths["doc"],
        "--readme", paths["readme"],
        "--bench-dir", paths["bench_dir"],
    ]
    return args + (extra or [])


class TestHeadline(unittest.TestCase):
    def test_starts_with_notice_and_flags_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            rows = r.load_rows(paths["after"])
            content = r.emit_headline(rows, list(r.CROSSLANG_OPS), r.HEADLINE_CELL)
            self.assertTrue(content.startswith("_Generated by"))
            self.assertIn("fpca", content)


class TestEmitters(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = _write_fixture(pathlib.Path(self._tmp.name))
        self.inputs = r._load_inputs_from_paths(self.paths)

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_thirteen_blocks_render_with_notice_and_no_images(self):
        rendered = r.render_blocks(self.inputs, list(r.BLOCKS))
        self.assertEqual(set(rendered), set(r.BLOCKS))
        for name, content in rendered.items():
            self.assertTrue(content.startswith("_Generated by"), msg=name)
            self.assertNotIn("![", content, msg=name)

    def test_internal_table_labels_no_comparator(self):
        content = r._EMITTERS["before-after-internal"](self.inputs)
        self.assertIn("no scikit-fda/R comparator", content)
        self.assertIn("face_covariance", content)

    def test_crosslang_fdars_table_excludes_new_crosslang_rows(self):
        content = r._EMITTERS["before-after-crosslang-fdars"](self.inputs)
        self.assertIn("audit_fpca/n100_m50", content)
        self.assertNotIn("crosslang_fpca/n500_m200", content)

    def test_before_after_tables_exclude_thread_scaling_group(self):
        crosslang = r._EMITTERS["before-after-crosslang-fdars"](self.inputs)
        internal = r._EMITTERS["before-after-internal"](self.inputs)
        self.assertNotIn("perf_parallelism_co_cluster", crosslang)
        self.assertNotIn("perf_parallelism_co_cluster", internal)

    def test_not_rerun_lists_orphan_and_thread_scaling_exclusions(self):
        content = r._EMITTERS["not-rerun"](self.inputs)
        self.assertIn("orphan_group", content)
        self.assertIn("orphan", content)
        self.assertIn("perf_parallelism_co_cluster", content)

    def test_ab_evidence_marks_headline(self):
        content = r._EMITTERS["ab-evidence"](self.inputs)
        self.assertIn("phase116_face_covariance_n200_m30", content)

    def test_gate_tolerances_mentions_fpca_only_agreement(self):
        content = r._EMITTERS["gate-tolerances"](self.inputs)
        self.assertIn("FPCA", content)
        self.assertIn("hard-gated", content)

    def test_criterion_settings_parses_face_covariance(self):
        content = r._EMITTERS["criterion-settings"](self.inputs)
        self.assertIn("perf_face_covariance", content)

    def test_runtime_reports_footprint(self):
        content = r._EMITTERS["runtime"](self.inputs)
        self.assertIn("GiB", content)

    def test_environment_flags_changed_thread_env(self):
        content = r._EMITTERS["environment"](self.inputs)
        self.assertIn("thread_env.RAYON_NUM_THREADS", content)

    def test_key_findings_reports_headline_eligible(self):
        content = r._EMITTERS["key-findings"](self.inputs)
        self.assertIn("faster", content)

    def test_scaling_includes_theory_and_source(self):
        content = r._EMITTERS["scaling"](self.inputs)
        self.assertIn("fdars-core/src/regression.rs::fdata_to_pc", content)


class TestSplice(unittest.TestCase):
    def test_idempotent_resplice(self):
        text = "before\n<!-- BEGIN GENERATED: x -->\nold\n<!-- END GENERATED: x -->\nafter\n"
        once = r.splice_generated(text, "x", "new content")
        twice = r.splice_generated(once, "x", "new content")
        self.assertEqual(once, twice)
        self.assertIn("new content", once)


class TestUpdateDocs(unittest.TestCase):
    def test_splices_into_every_doc_with_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            stale = r.update_docs([paths["doc"], paths["readme"]], {"headline": "NEW HEADLINE"}, write=True)
            self.assertIn("headline", stale)
            doc_text = pathlib.Path(paths["doc"]).read_text(encoding="utf-8")
            readme_text = pathlib.Path(paths["readme"]).read_text(encoding="utf-8")
            self.assertIn("NEW HEADLINE", doc_text)
            self.assertIn("NEW HEADLINE", readme_text)

    def test_missing_markers_in_both_docs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            with self.assertRaises(ValueError):
                r.update_docs([paths["doc"], paths["readme"]], {"not-a-real-block": "x"}, write=True)


class TestCheckMode(unittest.TestCase):
    def test_check_passes_after_fresh_render_and_fails_after_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            rc_write = r.main(_cli_args(paths))
            self.assertEqual(rc_write, 0)
            before_mtime = pathlib.Path(paths["doc"]).stat().st_mtime_ns
            rc_check = r.main(_cli_args(paths, ["--check"]))
            self.assertEqual(rc_check, 0)
            after_mtime = pathlib.Path(paths["doc"]).stat().st_mtime_ns
            self.assertEqual(before_mtime, after_mtime)  # --check never writes

            doc_path = pathlib.Path(paths["doc"])
            doc_path.write_text(
                doc_path.read_text(encoding="utf-8").replace("fpca", "TAMPERED"),
                encoding="utf-8",
            )
            rc_check_stale = r.main(_cli_args(paths, ["--check"]))
            self.assertEqual(rc_check_stale, 1)


class TestNoiseReport(unittest.TestCase):
    def test_reports_pass_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = r.main(_cli_args(paths, ["--noise-report"]))
            output = buf.getvalue()
            self.assertIn("NOISE criterion:crosslang_fpca/n500_m200", output)
            # Only fpca has crosslang data in this fixture -> the other five
            # ops are MISSING, so the noise report must fail closed.
            self.assertIn("MISSING", output)
            self.assertEqual(rc, 1)


class TestAbCheck(unittest.TestCase):
    def test_headline_eligible_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            rc = r.main(_cli_args(paths, ["--ab-check"]))
            self.assertEqual(rc, 0)


class TestCheckProse(unittest.TestCase):
    def test_prose_number_backed_by_generated_block_passes(self):
        doc_text = (
            "Some intro.\n\n"
            "<!-- BEGIN GENERATED: headline -->\n3.33x faster\n<!-- END GENERATED: headline -->\n\n"
            "The headline shows a 3.33x speedup.\n"
        )
        offenders = r.check_prose(doc_text, "")
        self.assertEqual(offenders, [])

    def test_unbacked_prose_number_is_flagged(self):
        doc_text = (
            "<!-- BEGIN GENERATED: headline -->\n3.33x faster\n<!-- END GENERATED: headline -->\n\n"
            "We measured a 10.00x speedup nowhere else recorded.\n"
        )
        offenders = r.check_prose(doc_text, "")
        self.assertIn("10.00", offenders)

    def test_readme_performance_section_scanned(self):
        readme_text = (
            "# Project\n\n## Performance\n\nWe are 99.99x faster somewhere unbacked.\n\n"
            "## Other\n\nIgnored: 42.00x here.\n"
        )
        offenders = r.check_prose("", readme_text)
        self.assertIn("99.99", offenders)
        self.assertNotIn("42.00", offenders)


class TestFailClosed(unittest.TestCase):
    def test_render_blocks_raises_on_missing_required_input(self):
        inputs = {"baseline": None, "baseline_path": "/tmp/does-not-exist.csv", "after": []}
        with self.assertRaises(ValueError):
            r.render_blocks(inputs, ["before-after-crosslang-fdars"])

    def test_main_returns_1_on_missing_required_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(pathlib.Path(tmp))
            paths["baseline"] = str(pathlib.Path(tmp) / "does-not-exist.csv")
            rc = r.main(_cli_args(paths, ["--blocks", "before-after-crosslang-fdars"]))
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
