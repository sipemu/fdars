"""
Unit tests for collect_rust.py — criterion output collector.

Verified behaviors:
  - base/estimates.json is never read (Pitfall 3)
  - Flat path `<group>/<bench>/new/estimates.json` parses correctly
  - BenchmarkId slash path `<group>/<param>/<val>/new/estimates.json` parses correctly
  - mean_ns == data["mean"]["point_estimate"]
  - Out-of-tree paths are rejected (path-traversal guard T-112-06)
  - Every overlapping-op criterion group maps to a shared `op` label

TDD RED: these tests are written before the production implementation.
"""

import json
import pathlib
import sys
import tempfile
import unittest

import pytest

# Import from the benchmarks/ directory
_BENCHMARKS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_BENCHMARKS_DIR))


@pytest.fixture
def fake_criterion_tree(tmp_path):
    """
    Build a minimal fake target/criterion tree with:
      - flat path:         audit_fpca/n60_m50/new/estimates.json
      - BenchmarkId path:  elastic_align_pair/m/50/new/estimates.json
      - decoy base path:   audit_fpca/n60_m50/base/estimates.json  (must be ignored)
    """
    def _make_estimates(mean_ns: float) -> dict:
        return {
            "mean": {
                "point_estimate": mean_ns,
                "confidence_interval": {"lower_bound": mean_ns * 0.95, "upper_bound": mean_ns * 1.05},
                "standard_error": mean_ns * 0.01,
            },
            "median": {"point_estimate": mean_ns, "confidence_interval": {}, "standard_error": 0.0},
            "std_dev": {"point_estimate": 0.0, "confidence_interval": {}, "standard_error": 0.0},
            "median_abs_dev": {"point_estimate": 0.0, "confidence_interval": {}, "standard_error": 0.0},
        }

    criterion_dir = tmp_path / "criterion"

    # Flat path (plain bench_function)
    flat_new = criterion_dir / "audit_fpca" / "n60_m50" / "new"
    flat_new.mkdir(parents=True)
    (flat_new / "estimates.json").write_text(json.dumps(_make_estimates(16_000_000.0)))

    # Decoy base/ — must be ignored
    flat_base = criterion_dir / "audit_fpca" / "n60_m50" / "base"
    flat_base.mkdir(parents=True)
    (flat_base / "estimates.json").write_text(json.dumps(_make_estimates(9_999_999.0)))

    # BenchmarkId slash path
    bench_id_new = criterion_dir / "elastic_align_pair" / "m" / "50" / "new"
    bench_id_new.mkdir(parents=True)
    (bench_id_new / "estimates.json").write_text(json.dumps(_make_estimates(5_000_000.0)))

    return criterion_dir


def test_flat_path_parsed(fake_criterion_tree):
    """Flat `<group>/<bench>/new/estimates.json` → group and bench extracted correctly."""
    from collect_rust import iter_estimates

    records = list(iter_estimates(fake_criterion_tree))
    flat = [r for r in records if r["group"] == "audit_fpca"]
    assert len(flat) == 1, f"Expected 1 audit_fpca record, got {len(flat)}"
    assert flat[0]["bench"] == "n60_m50"
    assert flat[0]["mean_ns"] == pytest.approx(16_000_000.0)


def test_benchmarkid_slash_path_parsed(fake_criterion_tree):
    """BenchmarkId path `elastic_align_pair/m/50/new/estimates.json` parses correctly."""
    from collect_rust import iter_estimates

    records = list(iter_estimates(fake_criterion_tree))
    elastic = [r for r in records if r["group"] == "elastic_align_pair"]
    assert len(elastic) == 1, f"Expected 1 elastic record, got {len(elastic)}"
    assert elastic[0]["bench"] == "m/50", f"Expected bench 'm/50', got {elastic[0]['bench']!r}"
    assert elastic[0]["mean_ns"] == pytest.approx(5_000_000.0)


def test_base_dir_ignored(fake_criterion_tree):
    """base/estimates.json must never be read (Pitfall 3)."""
    from collect_rust import iter_estimates

    records = list(iter_estimates(fake_criterion_tree))
    # The decoy base has mean_ns=9_999_999; must not appear in results
    mean_values = [r["mean_ns"] for r in records]
    assert 9_999_999.0 not in mean_values, (
        "base/estimates.json was read — Pitfall 3 guard failed!"
    )
    # Total: exactly 2 records (flat + BenchmarkId)
    assert len(records) == 2, f"Expected 2 records, got {len(records)}: {records}"


def test_mean_ns_matches_point_estimate(fake_criterion_tree):
    """mean_ns must equal data['mean']['point_estimate']."""
    from collect_rust import iter_estimates

    records = list(iter_estimates(fake_criterion_tree))
    for r in records:
        assert r["mean_ns"] > 0, f"mean_ns should be positive, got {r['mean_ns']}"
        # Verify the value came from the specific fixture we created
        if r["group"] == "audit_fpca":
            assert r["mean_ns"] == pytest.approx(16_000_000.0)
        elif r["group"] == "elastic_align_pair":
            assert r["mean_ns"] == pytest.approx(5_000_000.0)


def test_out_of_tree_path_rejected(tmp_path):
    """Out-of-tree paths are rejected (T-112-06 path-traversal guard)."""
    from collect_rust import iter_estimates

    # Create a symlink that points outside the criterion_dir
    criterion_dir = tmp_path / "criterion"
    criterion_dir.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    # Create a real estimates.json outside the criterion tree
    outside_new = outside_dir / "malicious_group" / "bench" / "new"
    outside_new.mkdir(parents=True)
    (outside_new / "estimates.json").write_text(json.dumps({
        "mean": {
            "point_estimate": 1234567.0,
            "confidence_interval": {"lower_bound": 0.0, "upper_bound": 0.0},
            "standard_error": 0.0,
        },
    }))

    # Create a symlink inside criterion_dir pointing outside
    symlink_path = criterion_dir / "escaped_group"
    try:
        symlink_path.symlink_to(outside_dir / "malicious_group")
    except OSError:
        pytest.skip("Cannot create symlink in this environment")

    records = list(iter_estimates(criterion_dir))
    # The symlinked path resolves outside criterion_dir → must be rejected
    mean_values = [r["mean_ns"] for r in records]
    assert 1234567.0 not in mean_values, (
        "Out-of-tree path was NOT rejected — path-traversal guard failed!"
    )


def test_collect_maps_op_labels(fake_criterion_tree):
    """collect() maps every criterion group to a shared `op` label and sets library='rust'."""
    from collect_rust import collect, OP_MAP

    records = collect(criterion_dir=fake_criterion_tree)
    for r in records:
        assert r["library"] == "rust"
        # Each record must have an 'op' field that is one of the known shared labels
        assert "op" in r, f"Missing 'op' in record: {r}"
        assert r["op"] in OP_MAP.values(), (
            f"op={r['op']!r} is not in OP_MAP — group {r['group']!r} has no mapping?"
        )


def test_op_map_covers_all_overlapping_groups():
    """Every overlapping-op criterion group is mapped in OP_MAP."""
    from collect_rust import OP_MAP

    # These are the criterion groups documented in RESEARCH.md Table 3
    # that correspond to the 6 overlapping ops
    expected_ops = {
        "fpca",
        "basis_smoothing",
        "depth",
        "functional_boxplot",
        "fpcr",
        "elastic",
    }
    mapped_ops = set(OP_MAP.values())
    # All 6 shared op labels must appear in the mapping
    for op in expected_ops:
        assert op in mapped_ops, (
            f"OP_MAP does not map any criterion group to op={op!r}. "
            f"Current mapped ops: {mapped_ops}"
        )


class TestCrosslangOpMap(unittest.TestCase):
    """
    Phase 116 (RPT-01, D-13): the six `crosslang_<op>` criterion groups added
    in `fdars-core/benches/crosslang_grid.rs` (Plan 116-02) must be labeled
    with the shared op names in OP_MAP, so `rust.json` / `comparison.csv` rows
    for the grid-aligned bench carry the same `op` value as the Python/R rows.
    """

    def test_op_map_has_all_six_crosslang_entries(self):
        """OP_MAP maps every crosslang_<op> group to its shared op label."""
        from collect_rust import OP_MAP

        expected = {
            "crosslang_fpca": "fpca",
            "crosslang_basis_smoothing": "basis_smoothing",
            "crosslang_depth": "depth",
            "crosslang_functional_boxplot": "functional_boxplot",
            "crosslang_fpcr": "fpcr",
            "crosslang_elastic": "elastic",
        }
        for group, op in expected.items():
            self.assertEqual(
                OP_MAP.get(group),
                op,
                f"OP_MAP[{group!r}] should map to {op!r}, got {OP_MAP.get(group)!r}",
            )

    def test_collect_labels_crosslang_depth_row(self):
        """collect() on a crosslang_depth/n300_m50 estimates.json yields op='depth'."""
        from collect_rust import collect

        with tempfile.TemporaryDirectory() as tmp:
            criterion_dir = pathlib.Path(tmp) / "criterion"
            new_dir = criterion_dir / "crosslang_depth" / "n300_m50" / "new"
            new_dir.mkdir(parents=True)
            estimates = {
                "mean": {
                    "point_estimate": 7_000_000.0,
                    "confidence_interval": {
                        "lower_bound": 6_800_000.0,
                        "upper_bound": 7_200_000.0,
                    },
                    "standard_error": 70_000.0,
                },
                "std_dev": {
                    "point_estimate": 350_000.0,
                    "confidence_interval": {},
                    "standard_error": 0.0,
                },
            }
            (new_dir / "estimates.json").write_text(json.dumps(estimates))

            records = collect(criterion_dir=criterion_dir)
            matches = [r for r in records if r["group"] == "crosslang_depth"]
            self.assertEqual(
                len(matches), 1, f"Expected 1 crosslang_depth record, got {len(matches)}"
            )
            self.assertEqual(matches[0]["op"], "depth")
            self.assertEqual(matches[0]["bench"], "n300_m50")


if __name__ == "__main__":
    unittest.main()
