"""
Unit tests for run_all.py — Phase 116 Plan 03: chunk manifest, rerun log,
budgeted resumable scheduler, grid-cell hardening, footprint, and --full.

Verified behaviors (Task 1):
  - Manifest integrity: every chunk's bench file exists and contains the
    benchmark_group literal; every filter starts with `^`; chunks 0-5 are the
    six crosslang groups in order; the manifest has 67 entries.
  - Coverage (D-09 FULL): every re-runnable baseline criterion id (library
    rust, group has a benchmark_group literal, group not orphaned) matches at
    least one chunk filter; no chunk targets an orphan group.
  - Per-id chunks of one group appear in ascending baseline_mean_ns order.
  - Rerun log round-trips; a malformed log fails closed (SystemExit naming
    the file).
  - Scheduler: first-executed timeout is final; a later timeout is
    interrupted (non-final) and stops the call; a costlier timed-out
    sibling is skipped as inferred without running; error exits 3;
    RUST_CHUNKS_REMAINING is printed.
  - --rust-bench / --rust-requeue remediation semantics.

Verified behaviors (Task 2):
  - _reset_cell_files only touches the two validated per-cell filenames.
  - _run_with_timeout kills the whole process group (grandchild included).
  - _dir_bytes sums file sizes / returns 0 for a missing path.
  - Grid-cell timeout/error is recorded and does not write per-cell files;
    gate failure is recorded then re-raised (fail-closed, D-11).

Uses the real fdars-core/benches/*.rs files and the real committed
benchmarks/results/comparison_baseline.csv as ground truth for the manifest
coverage test (read-only). All destructive/slow behavior (subprocess
invocations, real per-cell result files) is either mocked or redirected to a
tmp_path — no test in this file touches benchmarks/results/*.json or
target/criterion.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

# Import from the benchmarks/ directory (late import, same-directory module).
_BENCHMARKS_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _BENCHMARKS_DIR.parent
sys.path.insert(0, str(_BENCHMARKS_DIR))

import run_all  # noqa: E402  (module in same directory)
import gate  # noqa: E402  (module in same directory)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _defined_groups() -> set[str]:
    """Scan fdars-core/benches/*.rs for every benchmark_group("...") literal."""
    groups: set[str] = set()
    benches_dir = _REPO_ROOT / "fdars-core" / "benches"
    for rs_file in benches_dir.glob("*.rs"):
        text = rs_file.read_text(encoding="utf-8")
        groups.update(re.findall(r'benchmark_group\("([^"]+)"\)', text))
    return groups


def _baseline_rust_rows() -> list[tuple[str, str]]:
    """Return (group, full_criterion_id) for every rust row of the committed baseline."""
    import csv

    rows: list[tuple[str, str]] = []
    baseline_path = _REPO_ROOT / "benchmarks" / "results" / "comparison_baseline.csv"
    with baseline_path.open() as f:
        for row in csv.DictReader(f):
            if row["library"] != "rust":
                continue
            m = re.match(r"^criterion:(.+)$", row["source"])
            if not m:
                continue
            full_id = m.group(1)
            group = full_id.split("/", 1)[0]
            rows.append((group, full_id))
    return rows


def _write_log(path: pathlib.Path, records: list[dict] | None = None) -> None:
    path.write_text(json.dumps({
        "meta": {
            "started_utc": "2026-01-01T00:00:00Z", "head_sha": "deadbeef",
            "features": "linalg,parallel", "thread_env": dict(run_all.THREAD_ENV),
            "criterion_archive": None, "chunk_budget_s": None,
        },
        "records": records or [],
    }))


class _FakeClock:
    """Returns successive absolute timestamps from a fixed queue."""

    def __init__(self, timestamps: list[float]):
        self._timestamps = list(timestamps)

    def __call__(self) -> float:
        return self._timestamps.pop(0)


def _fake_runner(results: list[tuple[str, float]]):
    """Return a run_rust_bench-shaped fake: (bench, filter, env, timeout_s) -> (status, wall_s)."""
    calls: list[tuple[str, str, int]] = []

    def runner(bench, filter_regex, env, timeout_s):
        calls.append((bench, filter_regex, timeout_s))
        return results[len(calls) - 1]

    runner.calls = calls
    return runner


# ---------------------------------------------------------------------------
# TestParseCell (Plan 01 function; first test coverage lands with this file)
# ---------------------------------------------------------------------------


class TestParseCell(unittest.TestCase):
    def test_valid_cell(self):
        self.assertEqual(run_all._parse_cell("500x200"), (500, 200))

    def test_case_insensitive(self):
        self.assertEqual(run_all._parse_cell("500X200"), (500, 200))

    def test_malformed_missing_x(self):
        with self.assertRaises(ValueError):
            run_all._parse_cell("500-200")

    def test_non_integer(self):
        with self.assertRaises(ValueError):
            run_all._parse_cell("abcx200")

    def test_off_grid_cell_rejected(self):
        with self.assertRaises(ValueError):
            run_all._parse_cell("500x75")


# ---------------------------------------------------------------------------
# TestChunkManifest
# ---------------------------------------------------------------------------


class TestChunkManifest(unittest.TestCase):
    def test_manifest_has_67_entries(self):
        self.assertEqual(len(run_all.RUST_RERUN_CHUNKS), 67)

    def test_every_filter_starts_with_caret(self):
        for idx, chunk in enumerate(run_all.RUST_RERUN_CHUNKS):
            self.assertTrue(
                chunk["filter"].startswith("^"),
                f"chunk {idx} filter {chunk['filter']!r} does not start with ^",
            )

    def test_first_six_chunks_are_crosslang_groups_in_order(self):
        expected = [
            "crosslang_fpca", "crosslang_basis_smoothing", "crosslang_depth",
            "crosslang_functional_boxplot", "crosslang_fpcr", "crosslang_elastic",
        ]
        actual = [c["group"] for c in run_all.RUST_RERUN_CHUNKS[:6]]
        self.assertEqual(actual, expected)
        for c in run_all.RUST_RERUN_CHUNKS[:6]:
            self.assertEqual(c["bench"], "crosslang_grid")
            self.assertIsNone(c["baseline_mean_ns"])

    def test_every_bench_exists_and_contains_group_literal(self):
        benches_dir = _REPO_ROOT / "fdars-core" / "benches"
        for chunk in run_all.RUST_RERUN_CHUNKS:
            bench_path = benches_dir / f"{chunk['bench']}.rs"
            self.assertTrue(bench_path.exists(), f"missing bench file: {bench_path}")
            text = bench_path.read_text(encoding="utf-8")
            self.assertIn(
                f'benchmark_group("{chunk["group"]}")', text,
                f"{bench_path} does not define benchmark_group({chunk['group']!r})",
            )

    def test_orphan_groups_never_targeted(self):
        targeted_groups = {c["group"] for c in run_all.RUST_RERUN_CHUNKS}
        self.assertTrue(targeted_groups.isdisjoint(run_all.ORPHAN_GROUPS))

    def test_per_id_chunks_ascending_baseline_order(self):
        by_group: dict[tuple[str, str], list[float]] = {}
        for c in run_all.RUST_RERUN_CHUNKS:
            if c["baseline_mean_ns"] is None:
                continue
            by_group.setdefault((c["bench"], c["group"]), []).append(c["baseline_mean_ns"])
        for key, means in by_group.items():
            if len(means) <= 1:
                continue
            self.assertEqual(means, sorted(means), f"{key} chunks are not ascending: {means}")

    def test_coverage_every_rerunnable_baseline_id_matches_a_chunk(self):
        """D-09 FULL: every baseline rust row whose group is still defined must
        be re-runnable by at least one chunk filter; orphan-group rows are
        exempt (they cannot be re-run — no benchmark_group literal exists)."""
        defined = _defined_groups()
        compiled = [re.compile(c["filter"]) for c in run_all.RUST_RERUN_CHUNKS]
        unmatched: list[str] = []
        for group, full_id in _baseline_rust_rows():
            if group in run_all.ORPHAN_GROUPS:
                continue
            if group not in defined:
                # Not in our expected orphan list AND not defined: manifest
                # or ORPHAN_GROUPS is stale relative to fdars-core/benches.
                unmatched.append(f"{full_id} (group {group!r} has no benchmark_group literal "
                                  "and is not in ORPHAN_GROUPS)")
                continue
            if not any(rx.search(full_id) for rx in compiled):
                unmatched.append(full_id)
        self.assertEqual(unmatched, [], f"baseline ids not covered by any chunk: {unmatched}")

    def test_orphan_groups_have_no_benchmark_group_literal(self):
        defined = _defined_groups()
        for group in run_all.ORPHAN_GROUPS:
            self.assertNotIn(group, defined, f"{group!r} is supposedly orphaned but IS defined")


# ---------------------------------------------------------------------------
# TestListRustChunks
# ---------------------------------------------------------------------------


class TestListRustChunks(unittest.TestCase):
    def test_prints_67_chunk_lines_first_six_crosslang(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_all.list_rust_chunks()
        lines = buf.getvalue().splitlines()
        digit_lines = [l for l in lines if re.match(r"^ *[0-9]+ ", l)]
        self.assertEqual(len(digit_lines), 67)
        self.assertEqual(sum(1 for l in digit_lines[:6] if "crosslang_" in l), 6)
        # Header must not start with a digit.
        self.assertFalse(re.match(r"^ *[0-9]+ ", lines[0]))


# ---------------------------------------------------------------------------
# TestRerunLog
# ---------------------------------------------------------------------------


class TestRerunLog(unittest.TestCase):
    def test_missing_log_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(run_all._load_rerun_log(pathlib.Path(td) / "nope.json"))

    def test_append_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            run_all._append_rerun_log(log_path, {"kind": "rust_chunk", "status": "ok", "chunk_idx": 0})
            log = run_all._load_rerun_log(log_path)
            self.assertEqual(len(log["records"]), 1)
            self.assertEqual(log["records"][0]["status"], "ok")

    def test_append_uses_atomic_replace(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            with mock.patch.object(run_all.os, "replace", wraps=run_all.os.replace) as m:
                run_all._append_rerun_log(log_path, {"kind": "rust_chunk", "status": "ok", "chunk_idx": 0})
            m.assert_called_once()

    def test_malformed_log_raises_systemexit_naming_file(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            log_path.write_text("{not valid json")
            with self.assertRaises(SystemExit) as cm:
                run_all._load_rerun_log(log_path)
            self.assertIn(str(log_path), str(cm.exception))

    def test_append_requires_existing_log(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            with self.assertRaises(SystemExit):
                run_all._append_rerun_log(log_path, {"kind": "rust_chunk"})

    def test_final_status_ignores_non_final_records(self):
        log = {"records": [
            {"kind": "rust_chunk", "chunk_idx": 0, "status": "error"},
            {"kind": "rust_chunk", "chunk_idx": 0, "status": "requeued"},
        ]}
        self.assertEqual(run_all._final_status(log, 0), "error")

    def test_chunk_is_pending_true_when_last_is_interrupted(self):
        log = {"records": [{"kind": "rust_chunk", "chunk_idx": 0, "status": "interrupted"}]}
        self.assertTrue(run_all._chunk_is_pending(log, 0))

    def test_chunk_is_pending_false_when_last_is_ok(self):
        log = {"records": [{"kind": "rust_chunk", "chunk_idx": 0, "status": "ok"}]}
        self.assertFalse(run_all._chunk_is_pending(log, 0))


class TestRerunInit(unittest.TestCase):
    def test_refuses_when_log_exists(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            with self.assertRaises(SystemExit) as cm:
                run_all.rerun_init(log_path=log_path, criterion_dir=pathlib.Path(td) / "no-criterion")
            self.assertEqual(cm.exception.code, 2)

    def test_archives_stale_criterion_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = pathlib.Path(td)
            log_path = td_path / "rerun_log.json"
            criterion_dir = td_path / "criterion"
            criterion_dir.mkdir()
            (criterion_dir / "marker.txt").write_text("stale")
            run_all.rerun_init(log_path=log_path, criterion_dir=criterion_dir)
            self.assertFalse(criterion_dir.exists())
            siblings = list(td_path.glob("criterion.pre116-*"))
            self.assertEqual(len(siblings), 1)
            self.assertTrue((siblings[0] / "marker.txt").exists())
            log = json.loads(log_path.read_text())
            self.assertEqual(log["meta"]["criterion_archive"], str(siblings[0]))
            self.assertEqual(log["meta"]["thread_env"], run_all.THREAD_ENV)
            self.assertEqual(log["records"], [])

    def test_no_criterion_dir_is_a_noop_archive(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = pathlib.Path(td)
            log_path = td_path / "rerun_log.json"
            run_all.rerun_init(log_path=log_path, criterion_dir=td_path / "does-not-exist")
            log = json.loads(log_path.read_text())
            self.assertIsNone(log["meta"]["criterion_archive"])


# ---------------------------------------------------------------------------
# TestScheduler (run_rust_next)
# ---------------------------------------------------------------------------


class TestScheduler(unittest.TestCase):
    def _chunk(self, bench="bench_a", group="group_a", filt=r"^group_a/", baseline=None):
        return {"tier": "P3", "bench": bench, "group": group, "filter": filt, "baseline_mean_ns": baseline}

    def test_first_executed_chunk_timeout_is_final(self):
        chunks = [self._chunk()]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            runner = _fake_runner([("timeout", 540.0)])
            clock = _FakeClock([0.0, 540.0])
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                rc = run_all.run_rust_next(540, {}, runner=runner, clock=clock, log_path=log_path)
            self.assertEqual(rc, 0)
            log = json.loads(log_path.read_text())
            recs = [r for r in log["records"] if r["kind"] == "rust_chunk"]
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["status"], "timeout")
            self.assertEqual(recs[0]["limit_s"], 540)
            self.assertEqual(len(runner.calls), 1)

    def test_later_chunk_timeout_is_interrupted_and_stops_call(self):
        chunks = [self._chunk(group="group_a", filt=r"^group_a/"),
                  self._chunk(group="group_b", filt=r"^group_b/"),
                  self._chunk(group="group_c", filt=r"^group_c/")]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            runner = _fake_runner([("ok", 5.0), ("timeout", 545.0)])
            clock = _FakeClock([0.0, 5.0, 5.0, 550.0])
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                rc = run_all.run_rust_next(540, {}, runner=runner, clock=clock, log_path=log_path)
            self.assertEqual(rc, 0)
            self.assertEqual(len(runner.calls), 2)  # third chunk never attempted
            log = json.loads(log_path.read_text())
            recs = {r["chunk_idx"]: r for r in log["records"] if r["kind"] == "rust_chunk"}
            self.assertEqual(recs[0]["status"], "ok")
            self.assertEqual(recs[1]["status"], "interrupted")
            self.assertNotIn(2, recs)

    def test_costlier_sibling_of_timed_out_chunk_is_skipped_without_running(self):
        chunks = [
            self._chunk(bench="bench_x", group="group_x", filt=r"^group_x/n1$", baseline=10.0),
            self._chunk(bench="bench_x", group="group_x", filt=r"^group_x/n2$", baseline=100.0),
        ]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            runner = _fake_runner([("timeout", 540.0)])
            clock = _FakeClock([0.0, 540.0])
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                run_all.run_rust_next(540, {}, runner=runner, clock=clock, log_path=log_path)
            self.assertEqual(len(runner.calls), 1)  # chunk 1 never ran
            log = json.loads(log_path.read_text())
            recs = {r["chunk_idx"]: r for r in log["records"] if r["kind"] == "rust_chunk"}
            self.assertEqual(recs[0]["status"], "timeout")
            self.assertEqual(recs[1]["status"], "skipped-inferred-timeout")

    def test_error_status_exits_3(self):
        chunks = [self._chunk()]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            runner = _fake_runner([("error", 5.0)])
            clock = _FakeClock([0.0, 5.0])
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                rc = run_all.run_rust_next(540, {}, runner=runner, clock=clock, log_path=log_path)
            self.assertEqual(rc, 3)
            log = json.loads(log_path.read_text())
            self.assertEqual(log["records"][0]["status"], "error")

    def test_prints_rust_chunks_remaining(self):
        chunks = [self._chunk(group="group_a", filt=r"^group_a/"),
                  self._chunk(group="group_b", filt=r"^group_b/")]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            # First chunk ok, consumes almost the whole budget: second chunk
            # never starts (remaining < 60s), so it stays pending.
            runner = _fake_runner([("ok", 530.0)])
            clock = _FakeClock([0.0, 530.0])
            buf = io.StringIO()
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                with contextlib.redirect_stdout(buf):
                    rc = run_all.run_rust_next(540, {}, runner=runner, clock=clock, log_path=log_path)
            self.assertEqual(rc, 0)
            self.assertEqual(len(runner.calls), 1)
            self.assertIn("RUST_CHUNKS_REMAINING=1", buf.getvalue())

    def test_requires_initialized_log(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            with self.assertRaises(SystemExit) as cm:
                run_all.run_rust_next(540, {}, log_path=log_path)
            self.assertEqual(cm.exception.code, 2)


# ---------------------------------------------------------------------------
# TestRustStatus / TestRustRequeue
# ---------------------------------------------------------------------------


class TestRustStatus(unittest.TestCase):
    def test_returns_zero_only_when_nothing_pending(self):
        chunks = [
            {"tier": "P3", "bench": "b", "group": "g0", "filter": "^g0/", "baseline_mean_ns": 1.0},
            {"tier": "P3", "bench": "b", "group": "g1", "filter": "^g1/", "baseline_mean_ns": 2.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path, records=[
                {"kind": "rust_chunk", "chunk_idx": 0, "status": "ok"},
            ])
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                rc = run_all.rust_status(log_path)
            self.assertEqual(rc, 1)  # chunk 1 still pending
            _write_log(log_path, records=[
                {"kind": "rust_chunk", "chunk_idx": 0, "status": "ok"},
                {"kind": "rust_chunk", "chunk_idx": 1, "status": "timeout"},
            ])
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                rc = run_all.rust_status(log_path)
            self.assertEqual(rc, 0)

    def test_requires_initialized_log(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                run_all.rust_status(pathlib.Path(td) / "nope.json")
            self.assertEqual(cm.exception.code, 2)


class TestRustRequeue(unittest.TestCase):
    def _log_with_status(self, log_path, status):
        _write_log(log_path, records=[{"kind": "rust_chunk", "chunk_idx": 0, "status": status}])

    def test_allows_requeue_after_error(self):
        chunks = [{"tier": "P3", "bench": "b", "group": "g", "filter": "^g/", "baseline_mean_ns": 1.0}]
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            self._log_with_status(log_path, "error")
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", chunks):
                rc = run_all.rust_requeue(0, log_path=log_path)
            self.assertEqual(rc, 0)
            log = json.loads(log_path.read_text())
            self.assertEqual(log["records"][-1]["status"], "requeued")

    def test_refuses_for_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            self._log_with_status(log_path, "timeout")
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", run_all.RUST_RERUN_CHUNKS[:1]):
                with self.assertRaises(SystemExit) as cm:
                    run_all.rust_requeue(0, log_path=log_path)
                self.assertEqual(cm.exception.code, 2)

    def test_refuses_for_ok(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            self._log_with_status(log_path, "ok")
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", run_all.RUST_RERUN_CHUNKS[:1]):
                with self.assertRaises(SystemExit) as cm:
                    run_all.rust_requeue(0, log_path=log_path)
                self.assertEqual(cm.exception.code, 2)

    def test_refuses_for_skipped_inferred_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            self._log_with_status(log_path, "skipped-inferred-timeout")
            with mock.patch.object(run_all, "RUST_RERUN_CHUNKS", run_all.RUST_RERUN_CHUNKS[:1]):
                with self.assertRaises(SystemExit) as cm:
                    run_all.rust_requeue(0, log_path=log_path)
                self.assertEqual(cm.exception.code, 2)

    def test_out_of_range_index_refused(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            with self.assertRaises(SystemExit) as cm:
                run_all.rust_requeue(9999, log_path=log_path)
            self.assertEqual(cm.exception.code, 2)


class TestRemediationRecord(unittest.TestCase):
    def test_rust_bench_appends_remediation_when_log_exists(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            argv = ["run_all.py", "--rust-bench", "some_bench", "--rust-filter", "^g/"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(run_all, "RERUN_LOG_PATH", log_path):
                    with mock.patch.object(run_all, "run_rust_bench", return_value=("ok", 1.5)):
                        with self.assertRaises(SystemExit) as cm2:
                            run_all.main()
                        self.assertEqual(cm2.exception.code, 0)
            log = json.loads(log_path.read_text())
            remediation = [r for r in log["records"] if r["kind"] == "remediation"]
            self.assertEqual(len(remediation), 1)
            self.assertEqual(remediation[0]["bench"], "some_bench")

    def test_rust_bench_no_remediation_when_no_log(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"  # never created
            argv = ["run_all.py", "--rust-bench", "some_bench", "--rust-filter", "^g/"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(run_all, "RERUN_LOG_PATH", log_path):
                    with mock.patch.object(run_all, "run_rust_bench", return_value=("ok", 1.5)):
                        with self.assertRaises(SystemExit) as cm:
                            run_all.main()
                        self.assertEqual(cm.exception.code, 0)
            self.assertFalse(log_path.exists())


# ---------------------------------------------------------------------------
# TestMatchRustRec (WR-01 fix: exact n/m match must not substitute wrong size)
# ---------------------------------------------------------------------------


class TestMatchRustRec(unittest.TestCase):
    def test_exact_match_parsed_from_bench_id(self):
        records = [
            {"op": "fpca", "bench": "n100_m50"},
            {"op": "fpca", "bench": "n500_m200"},
        ]
        rec = run_all._match_rust_rec(records, "fpca", 500, 200)
        self.assertEqual(rec["bench"], "n500_m200")

    def test_no_exact_size_match_returns_empty_dict_not_wrong_size(self):
        """WR-01 regression: a functional_boxplot-style hardcoded N sweep
        (unrelated to the caller's n/m) must never be silently substituted
        for the requested cell."""
        records = [
            {"op": "functional_boxplot", "bench": "N/50"},
            {"op": "functional_boxplot", "bench": "N/1000"},
        ]
        rec = run_all._match_rust_rec(records, "functional_boxplot", 500, 200)
        self.assertEqual(rec, {})

    def test_prefixed_bench_id_parses(self):
        records = [{"op": "elastic", "bench": "vert_fpca_n100_m50"}]
        rec = run_all._match_rust_rec(records, "elastic", 100, 50)
        self.assertEqual(rec["bench"], "vert_fpca_n100_m50")

    def test_wrong_op_never_matches(self):
        records = [{"op": "depth", "bench": "n100_m50"}]
        rec = run_all._match_rust_rec(records, "fpca", 100, 50)
        self.assertEqual(rec, {})

    def test_empty_records_returns_empty_dict(self):
        self.assertEqual(run_all._match_rust_rec([], "fpca", 100, 50), {})


class TestParseBenchNM(unittest.TestCase):
    def test_clean_n_m_pair(self):
        self.assertEqual(run_all._parse_bench_n_m("n100_m50"), (100, 50))

    def test_prefixed_n_m_pair(self):
        self.assertEqual(run_all._parse_bench_n_m("vert_fpca_n100_m50"), (100, 50))

    def test_slash_separated_sweep_param_parses_neither(self):
        # functional_boxplot's hardcoded "N/50" BenchmarkId: the digit is not
        # directly adjacent to the "n", so this correctly parses to
        # (None, None) rather than fabricating an (n, m) pair.
        self.assertEqual(run_all._parse_bench_n_m("N/50"), (None, None))

    def test_slash_separated_m_only_parses_neither(self):
        # elastic_align_pair's "m/100" sweep: same reasoning as above.
        self.assertEqual(run_all._parse_bench_n_m("m/100"), (None, None))


# ---------------------------------------------------------------------------
# TestCellFiles (_reset_cell_files)
# ---------------------------------------------------------------------------


class TestCellFiles(unittest.TestCase):
    def test_deletes_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_f = results_dir / "python_n100_m50.json"
            r_f = results_dir / "r_n100_m50.json"
            py_f.write_text("[]")
            r_f.write_text("[]")
            run_all._reset_cell_files(results_dir, 100, 50)
            self.assertFalse(py_f.exists())
            self.assertFalse(r_f.exists())

    def test_noop_when_absent(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            # Should not raise.
            run_all._reset_cell_files(results_dir, 500, 200)

    def test_only_touches_named_files(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            untouched = results_dir / "python_n500_m50.json"
            untouched.write_text("[]")
            run_all._reset_cell_files(results_dir, 100, 50)
            self.assertTrue(untouched.exists())


# ---------------------------------------------------------------------------
# TestRunWithTimeout (grandchild process-group kill)
# ---------------------------------------------------------------------------


class TestRunWithTimeout(unittest.TestCase):
    def test_timeout_kills_grandchild_process_group(self):
        with tempfile.TemporaryDirectory() as td:
            pidfile = pathlib.Path(td) / "gc.pid"
            child_code = textwrap.dedent(f"""
                import subprocess, sys, time
                gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                with open({str(pidfile)!r}, "w") as f:
                    f.write(str(gc.pid))
                time.sleep(60)
            """)
            env = dict(os.environ)
            status, _wall_s, _rc = run_all._run_with_timeout(
                [sys.executable, "-c", child_code], env=env, timeout_s=1,
            )
            self.assertEqual(status, "timeout")

            deadline = time.monotonic() + 2.0
            gc_pid = None
            while time.monotonic() < deadline:
                if pidfile.exists() and pidfile.read_text().strip():
                    gc_pid = int(pidfile.read_text().strip())
                    break
                time.sleep(0.05)
            self.assertIsNotNone(gc_pid, "grandchild never wrote its pid")

            deadline2 = time.monotonic() + 2.0
            gone = False
            while time.monotonic() < deadline2:
                try:
                    os.kill(gc_pid, 0)
                except ProcessLookupError:
                    gone = True
                    break
                time.sleep(0.05)
            self.assertTrue(gone, "grandchild process still alive after timeout kill")


# ---------------------------------------------------------------------------
# TestFootprint (_dir_bytes / record_footprint)
# ---------------------------------------------------------------------------


class TestFootprint(unittest.TestCase):
    def test_dir_bytes_missing_path_is_zero(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(run_all._dir_bytes(pathlib.Path(td) / "nope"), 0)

    def test_dir_bytes_sums_file_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "a.txt").write_bytes(b"x" * 10)
            sub = root / "sub"
            sub.mkdir()
            (sub / "b.txt").write_bytes(b"y" * 25)
            self.assertEqual(run_all._dir_bytes(root), 35)

    def test_record_footprint_appends_when_log_exists(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            fake_target = pathlib.Path(td) / "target"
            fake_target.mkdir()
            (fake_target / "a.bin").write_bytes(b"z" * 5)
            with mock.patch.object(run_all, "_WORKSPACE_ROOT", pathlib.Path(td)):
                rc = run_all.record_footprint(log_path=log_path)
            self.assertEqual(rc, 0)
            log = json.loads(log_path.read_text())
            fp_recs = [r for r in log["records"] if r["kind"] == "footprint"]
            self.assertEqual(len(fp_recs), 1)
            self.assertEqual(fp_recs[0]["target_bytes"], 5)

    def test_record_footprint_no_log_still_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_all, "_WORKSPACE_ROOT", pathlib.Path(td)):
                rc = run_all.record_footprint(log_path=None)
            self.assertEqual(rc, 0)


# ---------------------------------------------------------------------------
# TestGridCellHardening (run_grid: timeout/error recording, gate-fail)
# ---------------------------------------------------------------------------


class TestGridCellHardening(unittest.TestCase):
    def test_cell_timeout_records_and_continues_exits_4(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td) / "results"
            results_dir.mkdir()
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            with mock.patch.object(run_all, "step_generate", side_effect=run_all.CellTimeout("boom")):
                with self.assertRaises(SystemExit) as cm:
                    run_all.run_grid(dry_run=False, seed=1, cells=[(100, 50)],
                                      ops=run_all.ALL_OPS, cell_timeout_s=5,
                                      log_path=log_path, results_dir=results_dir)
                self.assertEqual(cm.exception.code, 4)
            log = json.loads(log_path.read_text())
            cell_recs = [r for r in log["records"] if r["kind"] == "grid_cell"]
            self.assertEqual(len(cell_recs), 1)
            self.assertEqual(cell_recs[0]["status"], "timeout")
            self.assertEqual(cell_recs[0]["n"], 100)
            self.assertEqual(cell_recs[0]["m"], 50)
            # No per-cell files were written for the failed cell.
            self.assertFalse((results_dir / "python_n100_m50.json").exists())

    def test_cell_error_records_and_continues_exits_4(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td) / "results"
            results_dir.mkdir()
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            with mock.patch.object(run_all, "step_generate", side_effect=run_all.CellError("bad exit")):
                with self.assertRaises(SystemExit) as cm:
                    run_all.run_grid(dry_run=False, seed=1, cells=[(100, 50)],
                                      ops=run_all.ALL_OPS, cell_timeout_s=5,
                                      log_path=log_path, results_dir=results_dir)
                self.assertEqual(cm.exception.code, 4)
            log = json.loads(log_path.read_text())
            cell_recs = [r for r in log["records"] if r["kind"] == "grid_cell"]
            self.assertEqual(cell_recs[0]["status"], "error")

    def test_gate_failure_records_gate_fail_then_reraises(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td) / "results"
            results_dir.mkdir()
            (results_dir / "python.json").write_text("[]")
            (results_dir / "r.json").write_text("[]")
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)

            def _raise_gate_fail(**kwargs):
                raise SystemExit(1)

            with mock.patch.object(run_all, "step_generate"), \
                 mock.patch.object(run_all, "_run_all_ops_python"), \
                 mock.patch.object(run_all, "_run_r_with_native_fpca", return_value=[]), \
                 mock.patch.object(run_all, "step_collect_rust"), \
                 mock.patch.object(run_all, "step_run_gate", side_effect=_raise_gate_fail):
                with self.assertRaises(SystemExit) as cm:
                    run_all.run_grid(dry_run=False, seed=1, cells=[(100, 50)],
                                      ops=run_all.ALL_OPS, log_path=log_path,
                                      results_dir=results_dir)
                self.assertEqual(cm.exception.code, 1)
            log = json.loads(log_path.read_text())
            cell_recs = [r for r in log["records"] if r["kind"] == "grid_cell"]
            self.assertEqual(len(cell_recs), 1)
            self.assertEqual(cell_recs[0]["status"], "gate-fail")

    def test_no_log_path_means_no_records(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td) / "results"
            results_dir.mkdir()
            with mock.patch.object(run_all, "step_generate", side_effect=run_all.CellTimeout("boom")):
                with self.assertRaises(SystemExit):
                    run_all.run_grid(dry_run=False, seed=1, cells=[(100, 50)],
                                      ops=run_all.ALL_OPS, cell_timeout_s=5,
                                      log_path=None, results_dir=results_dir)
            # No rerun log was ever created (ad-hoc/tracer-style run stays log-free).
            self.assertEqual(list(pathlib.Path(td).glob("*.json")), [])

    def test_unfiltered_run_resets_cell_files_first(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td) / "results"
            results_dir.mkdir()
            stale = results_dir / "python_n100_m50.json"
            stale.write_text('[{"op": "stale"}]')
            (results_dir / "r_n100_m50.json").write_text("[]")
            with mock.patch.object(run_all, "step_generate", side_effect=run_all.CellTimeout("boom")):
                with self.assertRaises(SystemExit):
                    run_all.run_grid(dry_run=False, seed=1, cells=[(100, 50)],
                                      ops=run_all.ALL_OPS, cell_timeout_s=5,
                                      log_path=None, results_dir=results_dir)
            self.assertFalse(stale.exists())

    def test_dry_run_prints_cell_line_per_requested_cell(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_all.run_grid(dry_run=True, seed=1, cells=[(100, 50), (500, 200)], ops=run_all.ALL_OPS)
        count = buf.getvalue().count("[GRID] Cell n=")
        self.assertEqual(count, 2)


# ---------------------------------------------------------------------------
# TestRunFull (high-level sequencing, everything mocked)
# ---------------------------------------------------------------------------


class TestRunFull(unittest.TestCase):
    def test_full_sequence_runs_all_steps_in_order(self):
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"

            def fake_rerun_init(path=None):
                calls.append("rerun_init")
                _write_log(log_path)

            with mock.patch.object(run_all, "rerun_init", side_effect=fake_rerun_init), \
                 mock.patch.object(run_all, "generate_grid_data", side_effect=lambda seed: calls.append("generate")), \
                 mock.patch.object(run_all, "prebuild", side_effect=lambda env, t, log_path=None: (calls.append("prebuild"), 0)[1]), \
                 mock.patch.object(run_all, "_load_rerun_log", side_effect=lambda p: json.loads(log_path.read_text())), \
                 mock.patch.object(run_all, "_chunk_is_pending", return_value=False), \
                 mock.patch.object(run_all, "run_grid", side_effect=lambda **kw: calls.append("run_grid")), \
                 mock.patch.object(run_all, "step_capture_env", side_effect=lambda env: calls.append("capture_env")), \
                 mock.patch.object(run_all, "step_collect_rust", side_effect=lambda env: calls.append("collect_rust")), \
                 mock.patch.object(run_all, "_run", side_effect=lambda cmd, env: calls.append("assemble")), \
                 mock.patch.object(run_all, "record_footprint", side_effect=lambda log_path=None: (calls.append("footprint"), 0)[1]):
                rc = run_all.run_full(timeout_s=540, cell_timeout_s=540, log_path=log_path)
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [
                "rerun_init", "generate", "prebuild", "run_grid",
                "capture_env", "collect_rust", "assemble", "footprint",
            ])

    def test_prebuild_failure_short_circuits(self):
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as td:
            log_path = pathlib.Path(td) / "rerun_log.json"
            _write_log(log_path)
            with mock.patch.object(run_all, "generate_grid_data", side_effect=lambda seed: calls.append("generate")), \
                 mock.patch.object(run_all, "prebuild", return_value=1), \
                 mock.patch.object(run_all, "run_grid", side_effect=lambda **kw: calls.append("run_grid")):
                rc = run_all.run_full(timeout_s=540, cell_timeout_s=540, log_path=log_path)
            self.assertEqual(rc, 1)
            self.assertNotIn("run_grid", calls)


# ---------------------------------------------------------------------------
# TestStepRunGate (CR-01 fix, WR-04: step_run_gate direct unit coverage)
# ---------------------------------------------------------------------------


def _rec(op: str, library: str) -> dict:
    """Minimal gate-shaped record: op label, library, and a fingerprint that
    fingerprint_to_array() can parse without error (a single-component,
    single-dimension "loadings" row)."""
    return {
        "op": op,
        "library": library,
        "n": 100,
        "m": 50,
        "mean_ns": 1.0,
        "result_fingerprint": {"loadings": [[0.1, 0.2, 0.3]]},
    }


class TestStepRunGate(unittest.TestCase):
    """Direct coverage of step_run_gate — the single function implementing
    the BENCH-05 measure-first invariant. Regression-tests CR-01: a
    requested op with no counterpart record must fail closed, not silently
    report an unconditional pass."""

    def _write_records(self, results_dir: pathlib.Path, py_ops: list[str],
                        r_ops: list[str]) -> tuple[pathlib.Path, pathlib.Path]:
        py_path = results_dir / "python.json"
        r_path = results_dir / "r.json"
        py_path.write_text(json.dumps([_rec(op, "python") for op in py_ops]))
        r_path.write_text(json.dumps([_rec(op, "r") for op in r_ops]))
        return py_path, r_path

    def test_all_ops_present_and_agreeing_passes(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_path, r_path = self._write_records(
                results_dir, run_all.ALL_OPS, run_all.ALL_OPS)
            with mock.patch.object(run_all, "step_fdars_loadings",
                                    return_value=[[0.1, 0.2, 0.3]]), \
                 mock.patch.object(gate, "check_fpca_agreement",
                                    return_value=(True, 1.0, [])), \
                 mock.patch.object(gate, "run_gate", return_value=True):
                passed = run_all.step_run_gate(
                    n=100, m=50, env={}, py_path=py_path, r_path=r_path,
                    ops=run_all.ALL_OPS)
            self.assertTrue(passed)

    def test_fpca_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_path, r_path = self._write_records(
                results_dir, run_all.ALL_OPS, run_all.ALL_OPS)

            def _fake_check(fdars, other, threshold=0.99, label=""):
                # fdars-vs-r is the hard gate; fail it, agree on fdars-vs-python.
                if label == "fdars-vs-r":
                    return (False, 0.5, [])
                return (True, 1.0, [])

            with mock.patch.object(run_all, "step_fdars_loadings",
                                    return_value=[[0.1, 0.2, 0.3]]), \
                 mock.patch.object(gate, "check_fpca_agreement",
                                    side_effect=_fake_check), \
                 mock.patch.object(gate, "run_gate", return_value=True):
                with self.assertRaises(SystemExit) as cm:
                    run_all.step_run_gate(
                        n=100, m=50, env={}, py_path=py_path, r_path=r_path,
                        ops=run_all.ALL_OPS)
                self.assertEqual(cm.exception.code, 1)

    def test_ops_subset_missing_counterpart_record_fails_closed(self):
        """CR-01 regression: `--grid --op fpca` writes a python "fpca" record
        but the r.json record is missing (e.g. the R runner crashed for that
        op only). Before the fix, `fpca_passed` stayed True (never set to
        False) and the gate silently reported an overall pass. It must now
        fail closed."""
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_path, r_path = self._write_records(
                results_dir, py_ops=["fpca"], r_ops=[])
            with mock.patch.object(run_all, "step_fdars_loadings",
                                    return_value=[[0.1, 0.2, 0.3]]), \
                 mock.patch.object(gate, "check_fpca_agreement") as m_check, \
                 mock.patch.object(gate, "run_gate", return_value=True):
                with self.assertRaises(SystemExit) as cm:
                    run_all.step_run_gate(
                        n=100, m=50, env={}, py_path=py_path, r_path=r_path,
                        ops=["fpca"])
                self.assertEqual(cm.exception.code, 1)
                # The hard-gate comparison itself must never have run — there
                # was nothing to compare fdars against.
                m_check.assert_not_called()

    def test_ops_subset_missing_non_fpca_counterpart_fails_closed(self):
        """CR-01 regression, step-6b path: `--grid --op depth` writes a
        python "depth" record but r.json is missing it. Before the fix this
        op was simply never added to gate_records, so run_gate() never saw
        it and gate_all_passed could still be True."""
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_path, r_path = self._write_records(
                results_dir, py_ops=["depth"], r_ops=[])
            with mock.patch.object(gate, "run_gate") as m_run_gate:
                with self.assertRaises(SystemExit) as cm:
                    run_all.step_run_gate(
                        n=100, m=50, env={}, py_path=py_path, r_path=r_path,
                        ops=["depth"])
                self.assertEqual(cm.exception.code, 1)
                # run_gate() must never see a manufactured pass here since
                # the op was never actually gated — gate_records was empty.
                m_run_gate.assert_not_called()

    def test_ops_subset_present_on_both_sides_passes(self):
        """A requested op with records on both sides and agreement still
        passes cleanly (the fix must not over-trigger for the healthy case)."""
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_path, r_path = self._write_records(
                results_dir, py_ops=["depth"], r_ops=["depth"])
            with mock.patch.object(gate, "run_gate", return_value=True):
                passed = run_all.step_run_gate(
                    n=100, m=50, env={}, py_path=py_path, r_path=r_path,
                    ops=["depth"])
            self.assertTrue(passed)

    def test_ops_none_infers_requested_from_present_records(self):
        """Legacy callers that omit `ops` (e.g. pre-existing direct calls)
        keep working: requested-ness is inferred from the union of ops
        present in either record set, so a single-op-only python.json/r.json
        pair (both sides agreeing) still passes without requiring the other
        five ops to be present."""
        with tempfile.TemporaryDirectory() as td:
            results_dir = pathlib.Path(td)
            py_path, r_path = self._write_records(
                results_dir, py_ops=["depth"], r_ops=["depth"])
            with mock.patch.object(gate, "run_gate", return_value=True):
                passed = run_all.step_run_gate(
                    n=100, m=50, env={}, py_path=py_path, r_path=r_path)
            self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
