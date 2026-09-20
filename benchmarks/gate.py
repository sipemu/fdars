"""
Numeric agreement gate for the fdars cross-language harness.

Implements sign-aligned Pearson correlation check for FPCA loadings.
The sign convention mirrors fdars `fix_svd_signs` (regression.rs:288-301):
for each loading vector, the element with the largest absolute value is
made strictly positive.

Usage:
    python gate.py --self-test         # prove pass-on-agreement AND fail-on-mismatch
    python gate.py --check-csv-identity --n 60 --m 50
    python gate.py --dry-run --n 60 --m 50    # generate + check (no cargo bench)
"""

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np
from scipy.stats import pearsonr

# -------------------------------------------------------------------
# Core gate function
# -------------------------------------------------------------------


def check_fpca_agreement(
    fdars_loadings: np.ndarray,
    other_loadings: np.ndarray,
    threshold: float = 0.99,
    label: str = "",
) -> tuple[bool, float, list[dict]]:
    """
    Sign-align and check Pearson correlation between FPCA loading matrices.

    Mirrors the fdars `fix_svd_signs` convention (regression.rs:288-301):
    for each component k, if dot(fdars_col_k, other_col_k) < 0, flip other_col_k.
    Then compute Pearson correlation and check against threshold.

    Args:
        fdars_loadings: shape (m, ncomp) — rotation matrix from fdata_to_pc.
                        Each column k is the k-th loading vector.
        other_loadings: shape (m, ncomp) — from scikit-fda or R (same shape).
        threshold:      minimum |corr| required for each component (default 0.99).
        label:          description string for error messages.

    Returns:
        (passed, min_corr, details)
        - passed:   True if all |corr| >= threshold.
        - min_corr: minimum |corr| across all components.
        - details:  list of {component, correlation, sign_flipped} per component.
    """
    assert fdars_loadings.shape == other_loadings.shape, (
        f"Shape mismatch: {fdars_loadings.shape} vs {other_loadings.shape}"
    )
    m, ncomp = fdars_loadings.shape
    details = []
    min_corr = 1.0

    for k in range(ncomp):
        l1 = fdars_loadings[:, k]
        l2 = other_loadings[:, k].copy()

        # Sign-align l2 to l1 (mirrors fdars fix_svd_signs: dot < 0 → flip)
        sign_flipped = False
        if np.dot(l1, l2) < 0:
            l2 = -l2
            sign_flipped = True

        corr, _ = pearsonr(l1, l2)
        abs_corr = abs(corr)
        details.append({
            "component": k,
            "correlation": float(corr),
            "abs_correlation": float(abs_corr),
            "sign_flipped": sign_flipped,
        })
        min_corr = min(min_corr, abs_corr)

    passed = min_corr >= threshold

    if not passed:
        tag = f" [{label}]" if label else ""
        print(
            f"GATE FAIL{tag} FPCA: min |corr| = {min_corr:.6f} < {threshold}",
            file=sys.stderr,
        )
        for d in details:
            print(
                f"  PC{d['component']+1}: corr={d['correlation']:.6f} "
                f"(flip={d['sign_flipped']})",
                file=sys.stderr,
            )

    return passed, min_corr, details


# -------------------------------------------------------------------
# Helper: convert result_fingerprint loadings to (m, ncomp) array
# -------------------------------------------------------------------


def fingerprint_to_array(fingerprint: dict) -> np.ndarray:
    """
    Convert a result_fingerprint dict to a (m, ncomp) numpy array.

    The fingerprint has key 'loadings' with shape (ncomp, m) — transposed
    from the internal (m, ncomp) convention for JSON serialisation.
    """
    rows = fingerprint["loadings"]  # list of ncomp rows, each of length m
    arr = np.array(rows, dtype=np.float64)  # shape (ncomp, m)
    return arr.T  # shape (m, ncomp)


# -------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------


def self_test() -> bool:
    """
    Prove the gate discriminates: pass on agreement, fail on mismatch.

    Returns True if both sub-tests pass.
    """
    rng = np.random.default_rng(0)
    m, ncomp = 50, 5

    # Construct a clean loading matrix (columns are orthonormal-ish)
    raw = rng.standard_normal((m, ncomp))
    U, _, Vt = np.linalg.svd(raw, full_matrices=False)
    fdars_load = U  # shape (m, ncomp)

    # ---- Sub-test 1: PASS on near-identical loadings ----
    # Add tiny noise; the sign may be flipped on some components — gate should handle that
    other_agree = fdars_load.copy()
    # Flip sign of component 2 (simulating sign ambiguity that gate must tolerate)
    other_agree[:, 2] = -other_agree[:, 2]
    # Add tiny numerical noise
    other_agree += rng.standard_normal((m, ncomp)) * 1e-4

    passed_agree, min_corr_agree, _ = check_fpca_agreement(
        fdars_load, other_agree, threshold=0.99, label="self-test-agree"
    )
    if not passed_agree:
        print(f"SELF-TEST FAIL: agreement case should PASS (min_corr={min_corr_agree:.4f})",
              file=sys.stderr)
        return False
    print(f"Self-test agree: PASS  (min |corr| = {min_corr_agree:.6f} >= 0.99)")

    # ---- Sub-test 2: FAIL on scrambled loadings ----
    # Shuffle the loading vectors so they are genuinely different
    other_disagree = fdars_load.copy()
    # Replace components with independent random vectors (orthogonal to fdars)
    random_load = rng.standard_normal((m, ncomp))
    # Make them orthogonal to fdars_load columns (Gram-Schmidt)
    for k in range(ncomp):
        v = random_load[:, k]
        for j in range(ncomp):
            v -= np.dot(v, fdars_load[:, j]) * fdars_load[:, j]
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            other_disagree[:, k] = v / norm
        else:
            other_disagree[:, k] = rng.standard_normal(m)

    passed_disagree, min_corr_disagree, _ = check_fpca_agreement(
        fdars_load, other_disagree, threshold=0.99, label="self-test-disagree"
    )
    if passed_disagree:
        print(
            f"SELF-TEST FAIL: disagree case should FAIL "
            f"(min_corr={min_corr_disagree:.4f} — gate did not catch it)",
            file=sys.stderr,
        )
        return False
    print(f"Self-test disagree: PASS  (min |corr| = {min_corr_disagree:.6f} < 0.99 — correctly REJECTED)")

    print("\nGATE SELF-TEST: PASSED (both pass-on-agree and fail-on-mismatch verified)")
    return True


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FPCA numeric agreement gate")
    parser.add_argument("--self-test", action="store_true",
                        help="Run gate self-test (agree+disagree)")
    parser.add_argument("--check-csv-identity", action="store_true",
                        help="Re-run CSV bit-identity check")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check gate on existing results/python.json + results/r.json")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    args = parser.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    script_dir = pathlib.Path(__file__).parent

    if args.check_csv_identity:
        # Delegate to verify_inputs.py
        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_dir / "verify_inputs.py"),
             "--n", str(args.n), "--m", str(args.m)],
            check=False,
        )
        sys.exit(result.returncode)

    if args.dry_run:
        # Check FPCA agreement using existing python.json and r.json
        py_path = script_dir / "results" / "python.json"
        r_path = script_dir / "results" / "r.json"

        if not py_path.exists() or not r_path.exists():
            print(f"ERROR: Missing results files. Run python_runner.py and r_runner.R first.",
                  file=sys.stderr)
            sys.exit(1)

        with py_path.open() as f:
            py_records = json.load(f)
        with r_path.open() as f:
            r_records = json.load(f)

        py_fpca = next((r for r in py_records if r["op"] == "fpca"), None)
        r_fpca = next((r for r in r_records if r["op"] == "fpca"), None)

        if py_fpca is None or r_fpca is None:
            print("ERROR: No FPCA records found in results files.", file=sys.stderr)
            sys.exit(1)

        py_load = fingerprint_to_array(py_fpca["result_fingerprint"])
        r_load = fingerprint_to_array(r_fpca["result_fingerprint"])

        passed, min_corr, details = check_fpca_agreement(
            py_load, r_load, threshold=0.99, label="python-vs-r"
        )
        print(f"Gate [python-vs-r]: {'PASS' if passed else 'FAIL'} "
              f"(min |corr| = {min_corr:.6f})")
        for d in details:
            print(f"  PC{d['component']+1}: corr={d['correlation']:.6f}")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
