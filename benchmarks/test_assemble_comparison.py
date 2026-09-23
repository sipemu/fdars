"""
Unit tests for assemble_comparison.py.

Tests run against small synthetic inputs — no heavy grid run needed.

Schema columns (RESEARCH Section 5):
    op, library, n, m, mean_ns, ci_lower_ns, ci_upper_ns, std_dev_ns,
    rsd_pct, ci_width_pct, noise_gate, source, notes
"""

import io
import json
import pathlib
import tempfile

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — build synthetic per-cell JSON records
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = [
    "op", "library", "n", "m",
    "mean_ns", "ci_lower_ns", "ci_upper_ns", "std_dev_ns",
    "rsd_pct", "ci_width_pct", "noise_gate", "source", "notes",
]


def _rust_record(
    group: str,
    bench: str,
    mean_ns: float,
    ci_lower_ns: float,
    ci_upper_ns: float,
    std_dev_ns: float,
    op: str = "fpca",
    library: str = "rust",
) -> dict:
    """Minimal criterion-style rust record (as produced by collect_rust.py)."""
    return {
        "op": op,
        "library": library,
        "group": group,
        "bench": bench,
        "mean_ns": mean_ns,
        "ci_lower_ns": ci_lower_ns,
        "ci_upper_ns": ci_upper_ns,
        "std_dev_ns": std_dev_ns,
    }


def _python_record(op: str, n: int, m: int, mean_ns: float,
                   result_fingerprint: dict | None = None) -> dict:
    rec = {
        "op": op,
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
    }
    if result_fingerprint is not None:
        rec["result_fingerprint"] = result_fingerprint
    return rec


def _r_svd_record(op: str, n: int, m: int, mean_ns: float,
                  result_fingerprint: dict | None = None) -> dict:
    rec = {
        "op": op,
        "library": "r",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
    }
    if result_fingerprint is not None:
        rec["result_fingerprint"] = result_fingerprint
    return rec


def _r_native_record(n: int, m: int, mean_ns: float) -> dict:
    """fpca_native_r record (no fingerprint, library=r)."""
    return {
        "op": "fpca_native_r",
        "library": "r",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
    }


def _write_cell_files(tmp: pathlib.Path, n: int, m: int,
                      py_records: list, r_records: list) -> None:
    """Write python_n{n}_m{m}.json and r_n{n}_m{m}.json to tmp directory."""
    (tmp / f"python_n{n}_m{m}.json").write_text(json.dumps(py_records))
    (tmp / f"r_n{n}_m{m}.json").write_text(json.dumps(r_records))


def _write_rust_json(tmp: pathlib.Path, records: list) -> None:
    (tmp / "rust.json").write_text(json.dumps(records))


# ---------------------------------------------------------------------------
# Import helper — defer import so RED tests fail at test time, not collection
# ---------------------------------------------------------------------------

def _import_assembler():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "assemble_comparison",
        pathlib.Path(__file__).parent / "assemble_comparison.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test: rsd_pct and ci_width_pct computed from criterion fields
# ---------------------------------------------------------------------------

class TestRustNoiseFields:
    def test_rsd_pct_computed_from_criterion_fields(self, tmp_path):
        """rsd_pct = std_dev_ns / mean_ns * 100 for a Rust criterion row."""
        assembler = _import_assembler()
        mean = 1_000_000.0
        std_dev = 30_000.0  # RSD = 3%
        ci_lower = 990_000.0
        ci_upper = 1_010_000.0  # CI-width = 2%

        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  mean, ci_lower, ci_upper, std_dev,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 100, 50, 4_000_000.0)]

        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        rust_row = df[(df["library"] == "rust") & (df["op"] == "fpca")].iloc[0]

        expected_rsd = std_dev / mean * 100
        expected_ci_width = (ci_upper - ci_lower) / mean * 100
        assert abs(rust_row["rsd_pct"] - expected_rsd) < 1e-9, rust_row["rsd_pct"]
        assert abs(rust_row["ci_width_pct"] - expected_ci_width) < 1e-9, rust_row["ci_width_pct"]

    def test_ci_width_pct_computed_from_criterion_fields(self, tmp_path):
        """ci_width_pct = (ci_upper - ci_lower) / mean * 100."""
        assembler = _import_assembler()
        mean = 2_000_000.0
        ci_lower = 1_980_000.0
        ci_upper = 2_020_000.0  # ci_width = 40_000 → 2.0%
        std_dev = 10_000.0      # rsd = 0.5%

        rust_recs = [_rust_record("fregre_lm", "ncomp/5",
                                  mean, ci_lower, ci_upper, std_dev,
                                  op="fpcr")]
        py_recs = [_python_record("fpcr", 100, 50, 8_000_000.0)]
        r_recs = [_r_svd_record("fpcr", 100, 50, 7_000_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        rust_row = df[(df["library"] == "rust") & (df["op"] == "fpcr")].iloc[0]

        expected_ci = (ci_upper - ci_lower) / mean * 100
        assert abs(rust_row["ci_width_pct"] - expected_ci) < 1e-9


# ---------------------------------------------------------------------------
# Test: noise_gate labels
# ---------------------------------------------------------------------------

class TestNoiseGateLabels:
    def test_high_rsd_row_gets_fail_noisy(self, tmp_path):
        """A Rust row with RSD >= 5.0 must get noise_gate = 'FAIL-NOISY'."""
        assembler = _import_assembler()
        mean = 1_000_000.0
        std_dev = 60_000.0   # RSD = 6% → FAIL
        ci_lower = 995_000.0
        ci_upper = 1_005_000.0  # CI-width = 1% → PASS alone, but RSD fails

        rust_recs = [_rust_record("audit_p4_fpca", "n500_m200",
                                  mean, ci_lower, ci_upper, std_dev,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 500, 200, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 500, 200, 4_500_000.0)]
        _write_cell_files(tmp_path, 500, 200, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        rust_row = df[(df["library"] == "rust") & (df["op"] == "fpca")].iloc[0]
        assert rust_row["noise_gate"] == "FAIL-NOISY", rust_row["noise_gate"]

    def test_low_rsd_and_ci_gets_pass(self, tmp_path):
        """A Rust row with RSD < 5 AND CI-width < 5 must get noise_gate = 'PASS'."""
        assembler = _import_assembler()
        mean = 1_000_000.0
        std_dev = 20_000.0   # RSD = 2%
        ci_lower = 990_000.0
        ci_upper = 1_010_000.0  # CI-width = 2%

        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  mean, ci_lower, ci_upper, std_dev,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 100, 50, 4_000_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        rust_row = df[(df["library"] == "rust") & (df["op"] == "fpca")].iloc[0]
        assert rust_row["noise_gate"] == "PASS", rust_row["noise_gate"]

    def test_python_r_rows_get_na_noise_gate(self, tmp_path):
        """Python and R (r_svd) rows must have noise_gate = 'NA' (no CI data)."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 100, 50, 4_000_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        py_row = df[df["library"] == "python"].iloc[0]
        r_row = df[df["library"] == "r_svd"].iloc[0]
        assert py_row["noise_gate"] == "NA", py_row["noise_gate"]
        assert r_row["noise_gate"] == "NA", r_row["noise_gate"]


# ---------------------------------------------------------------------------
# Test: r_native_pca_fd notes
# ---------------------------------------------------------------------------

class TestNativePcaFdNotes:
    def test_r_native_pca_fd_has_notes_disclosure(self, tmp_path):
        """r_native_pca_fd rows must carry a non-empty notes disclosure."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [
            _r_svd_record("fpca", 100, 50, 4_000_000.0),
            _r_native_record(100, 50, 18_000_000.0),
        ]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        native_rows = df[df["library"] == "r_native_pca_fd"]
        assert len(native_rows) > 0, "No r_native_pca_fd row in assembled CSV"
        for _, row in native_rows.iterrows():
            assert row["notes"] and "native pca.fd" in str(row["notes"]).lower(), \
                f"notes cell missing disclosure: {row['notes']!r}"

    def test_r_native_pca_fd_noise_gate_is_na(self, tmp_path):
        """r_native_pca_fd rows must have noise_gate = 'NA' (not gated for agreement)."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [
            _r_svd_record("fpca", 100, 50, 4_000_000.0),
            _r_native_record(100, 50, 18_000_000.0),
        ]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        native_rows = df[df["library"] == "r_native_pca_fd"]
        for _, row in native_rows.iterrows():
            assert row["noise_gate"] == "NA", row["noise_gate"]


# ---------------------------------------------------------------------------
# Test: CSV header column order
# ---------------------------------------------------------------------------

class TestCsvHeaderOrder:
    def test_header_matches_research_schema(self, tmp_path):
        """The output CSV header must be exactly the RESEARCH schema column order."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 100, 50, 4_000_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        out_csv = tmp_path / "comparison.csv"
        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        assembler.write_comparison_csv(df, out_csv)

        written = pd.read_csv(out_csv)
        assert list(written.columns) == SCHEMA_COLUMNS, \
            f"Column mismatch:\n  got: {list(written.columns)}\n  expected: {SCHEMA_COLUMNS}"

    def test_assemble_returns_dataframe_with_schema_columns(self, tmp_path):
        """assemble() must return a DataFrame with schema columns in order."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 100, 50, 4_000_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        assert list(df.columns) == SCHEMA_COLUMNS, \
            f"Column mismatch: {list(df.columns)}"


# ---------------------------------------------------------------------------
# Test: gate_passed guard — no write without gate
# ---------------------------------------------------------------------------

class TestGateGuard:
    def test_write_csv_blocked_without_gate_pass(self, tmp_path):
        """write_comparison_csv must raise if gate_passed is not True (BENCH-05)."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("audit_p4_fpca", "n100_m50",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpca")]
        py_recs = [_python_record("fpca", 100, 50, 5_000_000.0)]
        r_recs = [_r_svd_record("fpca", 100, 50, 4_000_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        out_csv = tmp_path / "comparison.csv"

        # Should raise if gate_passed=False
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            assembler.write_comparison_csv(df, out_csv, gate_passed=False)

        # File must NOT have been written
        assert not out_csv.exists(), "comparison.csv was written despite gate_passed=False"


# ---------------------------------------------------------------------------
# Test: FPCR disclosure note (Phase 113 Plan 02 — user-approved)
# ---------------------------------------------------------------------------

class TestFpcrDisclosureNote:
    def _fpcr_fingerprint(self, values):
        return {"fitted_values": list(values)}

    def test_fpcr_rows_carry_measured_corr_in_notes(self, tmp_path):
        """FPCR python + r rows must carry the measured cross-library corr in notes."""
        import numpy as np
        assembler = _import_assembler()

        # Build two fitted-value vectors with a KNOWN correlation
        rng = np.random.default_rng(7)
        base = rng.standard_normal(50)
        py_fv = base
        r_fv = base * 2.0 + 0.3  # scale+shift → corr ≈ 1.0
        expected_corr = abs(np.corrcoef(py_fv, r_fv)[0, 1])

        rust_recs = [_rust_record("fregre_lm", "ncomp/5",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpcr")]
        py_recs = [_python_record("fpcr", 100, 50, 5_000_000.0,
                                  result_fingerprint=self._fpcr_fingerprint(py_fv))]
        r_recs = [_r_svd_record("fpcr", 100, 50, 4_000_000.0,
                                result_fingerprint=self._fpcr_fingerprint(r_fv))]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        # The cross-library disclosure note lives on the python + r_svd fpcr rows
        # (the rust row is criterion timing only — no cross-library comparison).
        fpcr_xlib = df[(df["op"] == "fpcr") & (df["library"].isin(["python", "r_svd"]))]
        assert set(fpcr_xlib["library"]) >= {"python", "r_svd"}, set(fpcr_xlib["library"])
        for _, row in fpcr_xlib.iterrows():
            note = str(row["notes"])
            assert "FPCR cross-library agreement" in note, note
            assert "NOT hard-gated" in note, note
            # measured corr present with 4-decimal formatting
            assert f"{expected_corr:.4f}" in note, f"expected {expected_corr:.4f} in {note!r}"

    def test_fpcr_disclosure_note_when_fingerprint_missing(self, tmp_path):
        """If fingerprints are absent, FPCR rows still get a disclosure note (corr=n/a)."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("fregre_lm", "ncomp/5",
                                  1_000_000.0, 990_000.0, 1_010_000.0, 20_000.0,
                                  op="fpcr")]
        py_recs = [_python_record("fpcr", 100, 50, 5_000_000.0)]  # no fingerprint
        r_recs = [_r_svd_record("fpcr", 100, 50, 4_000_000.0)]    # no fingerprint
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        fpcr_rows = df[(df["op"] == "fpcr") & (df["library"] == "python")]
        assert len(fpcr_rows) == 1
        note = str(fpcr_rows.iloc[0]["notes"])
        assert "FPCR cross-library agreement" in note
        assert "corr=n/a" in note, note


# ---------------------------------------------------------------------------
# Test: Karcher-mean exclusion note on elastic rows (Phase 113 Plan 02)
# ---------------------------------------------------------------------------

class TestKarcherExclusionNote:
    def test_elastic_rows_carry_karcher_exclusion_note(self, tmp_path):
        """Elastic python + r_svd rows must disclose the Karcher-mean exclusion."""
        assembler = _import_assembler()
        rust_recs = [_rust_record("elastic_align_pair", "m/50",
                                  3_000_000.0, 2_970_000.0, 3_030_000.0, 40_000.0,
                                  op="elastic")]
        py_recs = [_python_record("elastic", 100, 50, 2_900_000.0)]
        r_recs = [_r_svd_record("elastic", 100, 50, 3_100_000.0)]
        _write_cell_files(tmp_path, 100, 50, py_recs, r_recs)
        _write_rust_json(tmp_path, rust_recs)

        df = assembler.assemble(results_dir=tmp_path, gate_passed=True)
        elastic_xlib = df[(df["op"] == "elastic") & (df["library"].isin(["python", "r_svd"]))]
        assert set(elastic_xlib["library"]) >= {"python", "r_svd"}, set(elastic_xlib["library"])
        for _, row in elastic_xlib.iterrows():
            note = str(row["notes"])
            assert "Karcher-mean elastic excluded" in note, note
            assert "pairwise elastic_align_pair" in note, note
