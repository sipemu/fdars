"""
Numeric agreement gate for the fdars cross-language harness.

Per-op tolerance table (RESEARCH.md Agreement Tolerance Validation):
  fpca            sign-aligned Pearson correlation > 0.99   [HARD-GATED]
  basis_smoothing max absolute difference < 1e-3            [HARD-GATED]
  depth           Pearson correlation >= 0.99  (depths)     [HARD-GATED] [Phase 113]
  functional_boxplot Pearson correlation >= 0.99 (MBD)      [HARD-GATED] [Phase 113]
  fpcr            Pearson correlation vs _FPCR_CORR_THRESHOLD (fitted values) [DISCLOSED — NOT hard-gated] [Phase 113 Plan 02]
  elastic         L-infinity on warping functions < 1e-2    [HARD-GATED]

FPCR disclosure policy (Phase 113 Plan 02, user-approved 2026-09-22):
  scikit-fda (FPCA->OLS) and R fda.usc (fregre.pc) implement FPCR with related
  but not-identical procedures. Cross-library Pearson correlation is ~0.95 at
  n=60 but drops below the 0.95 disclosure threshold at n=100 (measured 0.9495).
  This is a genuine, disclosed cross-library divergence — NOT a numerical bug.
  FPCR is therefore COMPUTED (its correlation is recorded per cell) but does NOT
  hard-abort the grid run. The five other ops (fpca, basis_smoothing, depth,
  functional_boxplot, elastic) remain HARD-GATED and DO abort on failure.
  _FPCR_CORR_THRESHOLD is used only to LABEL agreement, never to abort.

FPCA sign convention mirrors fdars `fix_svd_signs` (regression.rs:288-301):
for each loading vector, the element with the largest absolute value is
made strictly positive.

Excludes random-projection depth from the gated path — use deterministic
MBD/Fraiman-Muniz instead (RESEARCH Pitfall 5).

Usage:
    python gate.py --self-test         # prove pass-on-agreement AND fail-on-mismatch
    python gate.py --check-csv-identity --n 60 --m 50
    python gate.py --dry-run --n 60 --m 50    # check gate on existing runner results
"""

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np
from scipy.stats import pearsonr


# -------------------------------------------------------------------
# Per-op tolerance constants (RESEARCH.md §5)
# -------------------------------------------------------------------

_FPCA_CORR_THRESHOLD = 0.99          # minimum |Pearson correlation| per loading component [HARD-GATED]
_DEPTH_CORR_THRESHOLD = 0.99         # depth, functional_boxplot (Phase 113 calibration) [HARD-GATED]
# FPCR DISCLOSURE THRESHOLD (Phase 113 Plan 02, user-approved 2026-09-22):
# This is NO LONGER a hard-abort threshold — FPCR is not in _GATED_OPS. It is used only
# to LABEL whether the FPCR cross-library correlation cleared the disclosure bar in the
# recorded notes. scikit-fda (FPCA->OLS) and R fda.usc (fregre.pc) implement FPCR with
# related-but-not-identical procedures: Phase-112 measured Python-vs-R FPCR corr = 0.96
# at n=60; at n=100 it drops to 0.9495 (genuine cross-library divergence, disclosed for
# the Phase-116 report, NOT a bug). Recording the measured per-cell corr in comparison.csv
# notes is how this divergence is surfaced honestly.
_FPCR_CORR_THRESHOLD = 0.95          # fpcr DISCLOSURE label only — NOT a gate/abort threshold
_MAX_ABS_DIFF_THRESHOLD = 1e-3       # basis_smoothing only
_ELASTIC_LINF_THRESHOLD = 1e-2       # L-infinity on warping functions


# -------------------------------------------------------------------
# Core FPCA gate (kept for backward compat + check_agreement dispatch)
# -------------------------------------------------------------------


def check_fpca_agreement(
    fdars_loadings: np.ndarray,
    other_loadings: np.ndarray,
    threshold: float = _FPCA_CORR_THRESHOLD,
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
# Per-op dispatch: check_agreement
# -------------------------------------------------------------------


def check_agreement(
    op: str,
    fdars_fp: dict,
    other_fp: dict,
    label: str = "",
) -> tuple[bool, float, list[dict]]:
    """
    Dispatch to the op-specific agreement check.

    Args:
        op:       shared op label (fpca, basis_smoothing, depth,
                  functional_boxplot, fpcr, elastic)
        fdars_fp: result_fingerprint dict from the fdars (rust) runner
        other_fp: result_fingerprint dict from python or R runner
        label:    description for error messages (e.g. "[python] elastic")

    Returns:
        (passed, metric, details)
        - passed: True if within tolerance
        - metric: the primary numeric metric (min |corr| for fpca; max abs diff for others;
                  L-inf for elastic)
        - details: list of per-component or summary dicts
    """
    if op == "fpca":
        fdars_arr = fingerprint_to_array(fdars_fp)
        other_arr = fingerprint_to_array(other_fp)
        return check_fpca_agreement(fdars_arr, other_arr,
                                    threshold=_FPCA_CORR_THRESHOLD, label=label)

    elif op in ("basis_smoothing",):
        # Compare smoothed_values: shape (n, m) — max absolute difference
        fdars_vals = np.array(fdars_fp["smoothed_values"], dtype=np.float64)
        other_vals = np.array(other_fp["smoothed_values"], dtype=np.float64)
        max_diff = float(np.max(np.abs(fdars_vals - other_vals)))
        passed = max_diff < _MAX_ABS_DIFF_THRESHOLD
        if not passed:
            tag = f" [{label}]" if label else ""
            print(
                f"GATE FAIL{tag} basis_smoothing: max|diff|={max_diff:.2e} "
                f">= {_MAX_ABS_DIFF_THRESHOLD:.0e}",
                file=sys.stderr,
            )
        return passed, max_diff, [{"max_abs_diff": max_diff}]

    elif op in ("depth", "functional_boxplot"):
        # Phase 113 calibration: Pearson correlation on depths vector.
        # MBD normalization conventions differ ~0.65% between scikit-fda and roahd
        # (scale/shift difference) — abs-diff wrongly rejects valid agreement.
        # Pearson corr is scale/shift invariant; threshold = _DEPTH_CORR_THRESHOLD.
        fdars_depths = np.array(fdars_fp["depths"], dtype=np.float64)
        other_depths = np.array(other_fp["depths"], dtype=np.float64)
        corr, _ = pearsonr(fdars_depths, other_depths)
        metric = float(abs(corr))
        passed = metric >= _DEPTH_CORR_THRESHOLD
        if not passed:
            tag = f" [{label}]" if label else ""
            print(
                f"GATE FAIL{tag} {op}: |Pearson corr|={metric:.4f} "
                f"< {_DEPTH_CORR_THRESHOLD}",
                file=sys.stderr,
            )
        return passed, metric, [{"pearson_corr": float(corr)}]

    elif op == "fpcr":
        # Phase 113 calibration: Pearson correlation on fitted_values vector.
        # Python (scikit-fda/lstsq) and R (fregre.pc) use different scale normalizations;
        # abs-diff wrongly rejects high-correlation results differing only in scale.
        # Threshold = _FPCR_CORR_THRESHOLD (0.95 — disclosed relaxation; see constant above).
        fdars_fitted = np.array(fdars_fp["fitted_values"], dtype=np.float64)
        other_fitted = np.array(other_fp["fitted_values"], dtype=np.float64)
        corr, _ = pearsonr(fdars_fitted, other_fitted)
        metric = float(abs(corr))
        passed = metric >= _FPCR_CORR_THRESHOLD
        if not passed:
            tag = f" [{label}]" if label else ""
            print(
                f"GATE FAIL{tag} fpcr: |Pearson corr|={metric:.4f} "
                f"< {_FPCR_CORR_THRESHOLD}",
                file=sys.stderr,
            )
        return passed, metric, [{"pearson_corr": float(corr)}]

    elif op == "elastic":
        # Compare warping functions via L-infinity norm (RESEARCH Practical recommendation)
        # Primary check: pairwise warping function
        fdars_warp = np.array(fdars_fp["warping_values"], dtype=np.float64)
        other_warp = np.array(other_fp["warping_values"], dtype=np.float64)
        linf_warp = float(np.max(np.abs(fdars_warp - other_warp)))

        # Secondary check: Karcher mean (average L2 norm, per RESEARCH)
        karcher_metric = 0.0
        karcher_details: dict = {}
        if "karcher_mean" in fdars_fp and "karcher_mean" in other_fp:
            fdars_karcher = np.array(fdars_fp["karcher_mean"], dtype=np.float64)
            other_karcher = np.array(other_fp["karcher_mean"], dtype=np.float64)
            karcher_metric = float(np.mean(np.abs(fdars_karcher - other_karcher)))
            karcher_details = {"karcher_avg_l2": karcher_metric}

        # Gate on pairwise warping L-inf (primary metric)
        passed = linf_warp < _ELASTIC_LINF_THRESHOLD
        if not passed:
            tag = f" [{label}]" if label else ""
            print(
                f"GATE FAIL{tag} elastic: warp L-inf={linf_warp:.2e} "
                f">= {_ELASTIC_LINF_THRESHOLD:.0e}",
                file=sys.stderr,
            )
        details = [{"linf_warp": linf_warp, **karcher_details}]
        return passed, linf_warp, details

    else:
        # Unknown op — fail closed. Silently returning True for an unrecognised op
        # would mask typos and allow future callers to bypass the gate entirely.
        raise ValueError(
            f"check_agreement: unknown op={op!r}. "
            f"Valid ops: fpca, basis_smoothing, depth, functional_boxplot, fpcr, elastic"
        )


# -------------------------------------------------------------------
# Top-level gate runner
# -------------------------------------------------------------------


# Overlapping ops that HARD-GATE the run (random-projection depth excluded, Pitfall 5).
# FPCR is deliberately EXCLUDED (Phase 113 Plan 02): it is a disclosed cross-library
# divergence — computed and recorded, but must NOT abort the grid run.
_GATED_OPS = frozenset(["fpca", "basis_smoothing", "depth", "functional_boxplot",
                        "elastic"])

# Ops that are COMPUTED for disclosure but do NOT hard-gate the run.
# Their cross-library correlation is measured and recorded (comparison.csv notes),
# but a sub-threshold value never aborts the run.
_DISCLOSED_OPS = frozenset(["fpcr"])


def compute_fpcr_disclosure_corr(fdars_fp: dict, other_fp: dict) -> float:
    """
    Compute the FPCR cross-library Pearson correlation for DISCLOSURE only.

    This never aborts and never prints a GATE FAIL — it simply returns the
    measured |Pearson correlation| between the two fitted-values vectors so the
    caller can record it (comparison.csv notes) and label whether it cleared the
    _FPCR_CORR_THRESHOLD disclosure bar. See the FPCR disclosure policy in the
    module docstring (Phase 113 Plan 02).

    Returns:
        |Pearson correlation| in [0, 1].  Returns float('nan') if the fitted
        values are missing or degenerate.
    """
    try:
        fdars_fitted = np.array(fdars_fp["fitted_values"], dtype=np.float64)
        other_fitted = np.array(other_fp["fitted_values"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if fdars_fitted.size == 0 or other_fitted.size != fdars_fitted.size:
        return float("nan")
    corr, _ = pearsonr(fdars_fitted, other_fitted)
    return float(abs(corr))


def run_gate(results: list[dict]) -> bool:
    """
    Check every HARD-GATED overlapping op against every non-fdars library.

    Iterates all (op, lib) pairs in `results` where lib != 'rust'. For each
    HARD-GATED op (_GATED_OPS), finds the fdars (rust) record for the same op
    and runs check_agreement. Returns False (fail-closed) if ANY hard-gated op
    is out of tolerance, printing a "GATE FAIL [<lib>] <op>" line naming the
    offending library and op.

    FPCR (and any op in _DISCLOSED_OPS) is COMPUTED for disclosure but does NOT
    affect the return value — a sub-threshold FPCR correlation is recorded and
    labeled (via a DISCLOSURE line to stdout) but never aborts the run.
    Phase 113 Plan 02 (user-approved 2026-09-22).

    Args:
        results: list of records, each with keys:
                   op, library, n, m, mean_ns, result_fingerprint (optional)

    Returns:
        True if ALL HARD-GATED ops pass for ALL non-fdars libraries.
        False if ANY hard-gated op/library pair is out of tolerance.
        (FPCR / disclosed ops never change this value.)
    """
    # Index fdars records by op
    fdars_by_op: dict[str, dict] = {}
    for rec in results:
        if rec.get("library") == "rust" and "result_fingerprint" in rec:
            fdars_by_op[rec["op"]] = rec

    all_passed = True
    for rec in results:
        lib = rec.get("library", "")
        op = rec.get("op", "")
        if lib == "rust":
            continue  # skip fdars records — they are the reference
        if "result_fingerprint" not in rec:
            continue  # no fingerprint to check

        fdars_rec = fdars_by_op.get(op)
        if fdars_rec is None:
            # No fdars reference for this op — skip (fdars may not have this bench yet)
            continue

        # Disclosed (non-gating) ops: compute + report, but NEVER abort.
        if op in _DISCLOSED_OPS:
            disc_corr = compute_fpcr_disclosure_corr(
                fdars_rec["result_fingerprint"], rec["result_fingerprint"]
            )
            cleared = (not np.isnan(disc_corr)) and disc_corr >= _FPCR_CORR_THRESHOLD
            label_status = "CLEARED" if cleared else "DISCLOSED-BELOW"
            print(
                f"[DISCLOSURE] [{lib}] {op}: |corr|={disc_corr:.4f} "
                f"(disclosure bar {_FPCR_CORR_THRESHOLD}; {label_status}; NOT hard-gated)"
            )
            continue  # never affects all_passed

        if op not in _GATED_OPS:
            continue  # op not hard-gated and not disclosed — skip

        label = f"[{lib}] {op}"
        passed, metric, details = check_agreement(
            op,
            fdars_rec["result_fingerprint"],
            rec["result_fingerprint"],
            label=label,
        )
        if not passed:
            # Already printed GATE FAIL inside check_agreement
            all_passed = False

    return all_passed


# -------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------


def self_test() -> bool:
    """
    Prove the gate discriminates for all ops: pass on agreement, fail on mismatch.

    Returns True if all sub-tests pass.
    """
    rng = np.random.default_rng(0)
    m, ncomp, n = 50, 5, 60
    passed_all = True

    # --- FPCA sub-test ---
    raw = rng.standard_normal((m, ncomp))
    U, _, _ = np.linalg.svd(raw, full_matrices=False)
    fdars_load = U  # shape (m, ncomp)

    other_agree = fdars_load.copy()
    other_agree[:, 2] = -other_agree[:, 2]  # sign flip
    other_agree += rng.standard_normal((m, ncomp)) * 1e-4

    passed_agree, min_corr_agree, _ = check_fpca_agreement(
        fdars_load, other_agree, threshold=0.99, label="self-test-agree"
    )
    if not passed_agree:
        print(f"SELF-TEST FAIL [FPCA agree]: should PASS (min_corr={min_corr_agree:.4f})",
              file=sys.stderr)
        passed_all = False
    else:
        print(f"Self-test FPCA agree: PASS  (min |corr| = {min_corr_agree:.6f} >= 0.99)")

    random_load = rng.standard_normal((m, ncomp))
    other_disagree = fdars_load.copy()
    for k in range(ncomp):
        v = random_load[:, k]
        for j in range(ncomp):
            v -= np.dot(v, fdars_load[:, j]) * fdars_load[:, j]
        norm = np.linalg.norm(v)
        other_disagree[:, k] = v / norm if norm > 1e-10 else rng.standard_normal(m)

    passed_disagree, min_corr_disagree, _ = check_fpca_agreement(
        fdars_load, other_disagree, threshold=0.99, label="self-test-disagree"
    )
    if passed_disagree:
        print(f"SELF-TEST FAIL [FPCA disagree]: should FAIL (corr={min_corr_disagree:.4f})",
              file=sys.stderr)
        passed_all = False
    else:
        print(f"Self-test FPCA disagree: PASS  (correctly REJECTED, min corr={min_corr_disagree:.6f})")

    # --- basis_smoothing sub-test (unchanged: abs-diff) ---
    for op, key in [("basis_smoothing", "smoothed_values")]:
        vals = rng.uniform(0, 1, (n, m))
        fp_a = {key: vals.tolist()}
        fp_b_pass = {key: (vals + rng.uniform(-5e-4, 5e-4, vals.shape)).tolist()}
        fp_b_fail = {key: (vals + 0.01).tolist()}

        ok_pass, m_pass, _ = check_agreement(op, fp_a, fp_b_pass, label=f"self-test-{op}-pass")
        ok_fail, m_fail, _ = check_agreement(op, fp_a, fp_b_fail, label=f"self-test-{op}-fail")
        if not ok_pass:
            print(f"SELF-TEST FAIL [{op} agree]: should PASS (metric={m_pass:.2e})",
                  file=sys.stderr)
            passed_all = False
        else:
            print(f"Self-test {op} agree: PASS (metric={m_pass:.2e} < 1e-3)")
        if ok_fail:
            print(f"SELF-TEST FAIL [{op} disagree]: should FAIL (metric={m_fail:.2e})",
                  file=sys.stderr)
            passed_all = False
        else:
            print(f"Self-test {op} disagree: PASS (correctly REJECTED, metric={m_fail:.2e})")

    # --- depth / fpcr sub-tests (Phase 113: Pearson correlation, scale/shift-invariant) ---
    for op, key in [("depth", "depths"), ("fpcr", "fitted_values")]:
        vals = rng.uniform(0.1, 0.9, n)
        fp_a = {key: vals.tolist()}
        # Pass: scale+shift (simulates MBD normalization diff / FPCR scale-normalization diff)
        fp_b_pass = {key: (vals * 0.9972 + 0.003).tolist()}
        # Fail: shuffled+reversed to break correlation
        shuffled = vals.copy()
        rng.shuffle(shuffled)
        fp_b_fail = {key: shuffled[::-1].tolist()}

        ok_pass, m_pass, _ = check_agreement(op, fp_a, fp_b_pass, label=f"self-test-{op}-pass")
        ok_fail, m_fail, _ = check_agreement(op, fp_a, fp_b_fail, label=f"self-test-{op}-fail")
        if not ok_pass:
            print(f"SELF-TEST FAIL [{op} agree]: should PASS (|corr|={m_pass:.4f})",
                  file=sys.stderr)
            passed_all = False
        else:
            print(f"Self-test {op} agree: PASS (|corr|={m_pass:.6f} >= threshold)")
        if ok_fail:
            print(f"SELF-TEST FAIL [{op} disagree]: should FAIL (|corr|={m_fail:.4f})",
                  file=sys.stderr)
            passed_all = False
        else:
            print(f"Self-test {op} disagree: PASS (correctly REJECTED, |corr|={m_fail:.6f})")

    # --- elastic sub-test ---
    warp = np.linspace(0, 1, m)
    fp_warp_a = {"warping_values": warp.tolist(), "karcher_mean": warp.tolist()}
    small_perturb = warp + rng.uniform(-3e-3, 3e-3, m)
    fp_warp_pass = {"warping_values": small_perturb.tolist(), "karcher_mean": warp.tolist()}
    large_perturb = warp.copy()
    large_perturb[m // 4: m // 2] += 0.05
    fp_warp_fail = {"warping_values": large_perturb.tolist(), "karcher_mean": warp.tolist()}

    ok_e_pass, me_pass, _ = check_agreement("elastic", fp_warp_a, fp_warp_pass,
                                             label="self-test-elastic-pass")
    ok_e_fail, me_fail, _ = check_agreement("elastic", fp_warp_a, fp_warp_fail,
                                             label="self-test-elastic-fail")
    if not ok_e_pass:
        print(f"SELF-TEST FAIL [elastic agree]: should PASS (L-inf={me_pass:.2e})",
              file=sys.stderr)
        passed_all = False
    else:
        print(f"Self-test elastic agree: PASS (warp L-inf={me_pass:.2e} < 1e-2)")
    if ok_e_fail:
        print(f"SELF-TEST FAIL [elastic disagree]: should FAIL (L-inf={me_fail:.2e})",
              file=sys.stderr)
        passed_all = False
    else:
        print(f"Self-test elastic disagree: PASS (correctly REJECTED, L-inf={me_fail:.2e})")

    if passed_all:
        print("\nGATE SELF-TEST: PASSED (all ops: pass-on-agree and fail-on-mismatch verified)")
    else:
        print("\nGATE SELF-TEST: FAILED", file=sys.stderr)
    return passed_all


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-op numeric agreement gate")
    parser.add_argument("--self-test", action="store_true",
                        help="Run gate self-test (agree+disagree for all ops)")
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
        # Run full-op gate on existing runner results
        py_path = script_dir / "results" / "python.json"
        r_path = script_dir / "results" / "r.json"
        rust_path = script_dir / "results" / "rust.json"

        if not py_path.exists() or not r_path.exists():
            print("ERROR: Missing results/python.json or results/r.json. "
                  "Run python_runner.py and r_runner.R first.",
                  file=sys.stderr)
            sys.exit(1)

        with py_path.open() as f:
            py_records = json.load(f)
        with r_path.open() as f:
            r_records = json.load(f)

        # Merge all records; rust records have no fingerprint but are used for reference
        # during FPCA: the fdars runner provides fingerprints via fpca_dump, not criterion.
        # For dry-run, we compare python vs r only (fdars fingerprints need fpca_dump run).
        all_records = py_records + r_records

        # Check python vs r agreement (both have fingerprints)
        print("Running full-op gate: python vs r ...")
        py_by_op = {r["op"]: r for r in py_records if "result_fingerprint" in r}
        r_by_op = {r["op"]: r for r in r_records if "result_fingerprint" in r}

        gate_passed = True
        for op in _GATED_OPS:
            py_rec = py_by_op.get(op)
            r_rec = r_by_op.get(op)
            if py_rec is None or r_rec is None:
                print(f"  {op}: SKIPPED (missing python or r record)")
                continue
            passed, metric, details = check_agreement(
                op, py_rec["result_fingerprint"], r_rec["result_fingerprint"],
                label=f"[r] {op}"
            )
            status = "PASS" if passed else "FAIL"
            print(f"  {op}: {status} (metric={metric:.2e})")
            if not passed:
                gate_passed = False

        # Disclosed (non-gating) ops: compute + report, never affect gate_passed.
        for op in _DISCLOSED_OPS:
            py_rec = py_by_op.get(op)
            r_rec = r_by_op.get(op)
            if py_rec is None or r_rec is None:
                print(f"  {op}: SKIPPED (missing python or r record)")
                continue
            disc_corr = compute_fpcr_disclosure_corr(
                py_rec["result_fingerprint"], r_rec["result_fingerprint"]
            )
            cleared = (not np.isnan(disc_corr)) and disc_corr >= _FPCR_CORR_THRESHOLD
            label_status = "CLEARED" if cleared else "DISCLOSED-BELOW"
            print(f"  {op}: DISCLOSURE |corr|={disc_corr:.4f} "
                  f"({label_status}; NOT hard-gated)")

        if gate_passed:
            print("\nGATE: PASS — all HARD-GATED overlapping ops agree between python and R "
                  "(FPCR disclosed separately)")
        else:
            print("\nGATE: FAIL — see GATE FAIL lines above", file=sys.stderr)
        sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()
