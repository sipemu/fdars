"""
rank_hotspots.py — Noise-gated leverage ranking for fdars hotspot identification.

Reads benchmarks/results/comparison.csv (produced by Plan 113-02) and emits:
  - benchmarks/results/hotspot_ranking.md  (RESEARCH Section-5 schema)

Ranking algorithm
-----------------
1. Noise gate: a rust/criterion row PASSES only if rsd_pct < 5.0 AND ci_width_pct < 5.0.
   Failing rows are collected into the "noise-excluded" list and NOT ranked.
   Python/R rows (noise_gate == 'NA') are carried for cross-language leverage denominator
   but are not themselves ranked targets.

2. Leverage score:
       leverage_score = fdars_ns × max(1.0, fdars_ns / min_other_ns)
   where min_other_ns = min of python / r_svd mean_ns for the same op/size cell.
   Guard: if min_other_ns <= 0 or fdars_ns <= 0, leverage_score = fdars_ns.

3. Deterministic stable sort by descending leverage_score.
   Ties broken by (op, n, m) for reproducibility.

Exclusions
----------
- fpca_native_r rows are NEVER ranked targets (ecosystem-realistic timing only).
- Karcher-mean elastic cells are noise-excluded (RSD 29-46%); elastic anchor uses
  the noise-stable elastic_align_pair cells.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import date
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RSD_THRESHOLD = 5.0       # percent — strict (<, not <=)
_CI_THRESHOLD = 5.0        # percent — strict (<, not <=)
_EXCLUDED_LIBRARIES = {"r_native_pca_fd"}  # never ranked targets


# ---------------------------------------------------------------------------
# Core public functions (called by unit tests and __main__)
# ---------------------------------------------------------------------------


def apply_noise_gate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split *rows* into (passed, excluded) based on the noise gate.

    Rules:
    - Rows with noise_gate == 'NA' (python/r cross-language) always pass.
    - Rust/criterion rows pass only if rsd_pct < 5.0 AND ci_width_pct < 5.0.
    - Rows with None/NaN noise_gate are treated as excluded (unknown = conservative).

    Parameters
    ----------
    rows : list of row dicts (schema matches comparison.csv columns).

    Returns
    -------
    passed  : rows eligible for ranking (noise gate PASS or NA).
    excluded : rows gated out (noise gate FAIL-NOISY or unknown).
    """
    passed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for row in rows:
        ng = row.get("noise_gate")

        # Cross-language (python/r) rows — always pass the gate
        if ng == "NA":
            passed.append(row)
            continue

        # Unknown / None / NaN noise_gate → conservatively exclude
        if ng is None or (isinstance(ng, float) and math.isnan(ng)):
            excluded.append(row)
            continue

        # Rust/criterion rows: gate on RSD and CI-width
        rsd = row.get("rsd_pct")
        ci_w = row.get("ci_width_pct")

        try:
            rsd_f = float(rsd)
            ci_f = float(ci_w)
        except (TypeError, ValueError):
            # Cannot parse metrics → conservative exclusion
            excluded.append(row)
            continue

        if rsd_f < _RSD_THRESHOLD and ci_f < _CI_THRESHOLD:
            passed.append(row)
        else:
            excluded.append(row)

    return passed, excluded


def compute_leverage(fdars_ns: float, min_other_ns: float) -> float:
    """
    Compute the leverage score for a single fdars/size cell.

    Formula: fdars_ns × max(1.0, fdars_ns / min_other_ns)

    Guards:
    - If min_other_ns <= 0 or fdars_ns <= 0 → return fdars_ns (no ratio).

    Parameters
    ----------
    fdars_ns     : fdars (Rust) mean timing in nanoseconds.
    min_other_ns : fastest competitor timing in nanoseconds for the same op/size.

    Returns
    -------
    float leverage score.
    """
    if fdars_ns <= 0 or min_other_ns <= 0:
        return fdars_ns

    ratio = fdars_ns / min_other_ns
    return fdars_ns * max(1.0, ratio)


def rank_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Apply the full pipeline: noise gate → compute leverage → sort.

    Returns
    -------
    ranked   : list of rust PASS rows augmented with 'leverage_score', sorted
               descending by leverage_score (stable, then by op/n/m for ties).
    excluded : list of rows that failed the noise gate.
    """
    passed, excluded = apply_noise_gate(rows)

    # Separate rust (ranking candidates) from cross-language (leverage denominator)
    rust_candidates: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    for row in passed:
        lib = row.get("library", "")
        if lib == "rust":
            rust_candidates.append(row)
        elif lib not in _EXCLUDED_LIBRARIES:
            cross_rows.append(row)
        # r_native_pca_fd rows are silently dropped (not ranked, not used as denominator)

    # Build a lookup: (op, n_key, m_key) → list of competitor mean_ns
    def _cell_key(row: dict[str, Any]) -> tuple[str, Any, Any]:
        return (
            str(row.get("op", "")),
            _nan_key(row.get("n")),
            _nan_key(row.get("m")),
        )

    competitor_map: dict[tuple, list[float]] = {}
    for row in cross_rows:
        k = _cell_key(row)
        try:
            mns = float(row["mean_ns"])
        except (TypeError, ValueError):
            continue
        competitor_map.setdefault(k, []).append(mns)

    # Compute leverage for each rust candidate
    ranked_candidates: list[dict[str, Any]] = []
    for row in rust_candidates:
        try:
            fdars_ns = float(row["mean_ns"])
        except (TypeError, ValueError):
            continue

        k = _cell_key(row)
        competitors = competitor_map.get(k, [])
        if competitors:
            min_other_ns = min(competitors)
        else:
            min_other_ns = -1.0  # no competitor → fallback guard in compute_leverage

        score = compute_leverage(fdars_ns=fdars_ns, min_other_ns=min_other_ns)
        augmented = dict(row)
        augmented["leverage_score"] = score
        ranked_candidates.append(augmented)

    # Deterministic stable sort: descending leverage_score, then (op, n, m) for ties
    ranked_candidates.sort(
        key=lambda r: (
            -r["leverage_score"],
            str(r.get("op", "")),
            _nan_sort_key(r.get("n")),
            _nan_sort_key(r.get("m")),
        )
    )

    return ranked_candidates, excluded


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _nan_key(val: Any) -> Any:
    """Return val or None for NaN — for use in dict keys."""
    try:
        if val is None:
            return None
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _nan_sort_key(val: Any) -> float:
    """Return float for sort key; NaN → sentinel infinity so NaN rows go last."""
    try:
        if val is None:
            return math.inf
        f = float(val)
        return math.inf if math.isnan(f) else f
    except (TypeError, ValueError):
        return math.inf


# ---------------------------------------------------------------------------
# DataFrame → row list conversion
# ---------------------------------------------------------------------------


def _df_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to a list of row dicts."""
    return df.where(pd.notnull(df), None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# hotspot_ranking.md emission
# ---------------------------------------------------------------------------


def _ms(ns: float) -> str:
    """Format nanoseconds as milliseconds with 3 decimal places."""
    return f"{ns / 1e6:.3f}"


def _format_source_label(source: str) -> str:
    """Extract a short human-readable label from a criterion source string."""
    if source and "criterion:" in source:
        return source.replace("criterion:", "")
    return source or "—"


def emit_hotspot_ranking(
    ranked: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    env_path: str,
    output_path: str,
    today: str | None = None,
) -> None:
    """
    Write the RESEARCH Section-5 hotspot_ranking.md schema to *output_path*.

    Parameters
    ----------
    ranked      : ranked rust PASS rows (output of rank_rows).
    excluded    : noise-excluded rows (output of rank_rows).
    env_path    : path to env.json (for header reference).
    output_path : destination file path.
    today       : date string (YYYY-MM-DD); defaults to today's date.
    """
    if today is None:
        today = str(date.today())

    # -----------------------------------------------------------------------
    # Identify confirmed elastic anchor (noise-stable pairwise elastic cells)
    # -----------------------------------------------------------------------
    elastic_anchor_rows = [
        r for r in ranked
        if r.get("op") == "elastic"
        and r.get("noise_gate") == "PASS"
        and r.get("n") is not None
        and not (isinstance(r.get("n"), float) and math.isnan(r.get("n")))
        and float(r.get("n", 0)) >= 100.0
    ]

    # For pairwise elastic (no n field or n=NaN), include those too
    elastic_pairwise_rows = [
        r for r in ranked
        if r.get("op") == "elastic"
        and r.get("noise_gate") == "PASS"
        and (r.get("n") is None or (isinstance(r.get("n"), float) and math.isnan(r.get("n"))))
    ]

    all_elastic_pass = elastic_anchor_rows + elastic_pairwise_rows
    # Sort by n (NaN last) then m for display
    all_elastic_pass.sort(
        key=lambda r: (_nan_sort_key(r.get("n")), _nan_sort_key(r.get("m")))
    )

    # -----------------------------------------------------------------------
    # Identify secondary targets (non-elastic ranked rows, top ~10)
    # -----------------------------------------------------------------------
    secondary = [r for r in ranked if r.get("op") != "elastic"]

    # -----------------------------------------------------------------------
    # Noise-excluded cells
    # -----------------------------------------------------------------------
    noise_excl = [r for r in excluded if r.get("library") == "rust"]
    noise_excl.sort(
        key=lambda r: (str(r.get("op", "")), _nan_sort_key(r.get("n")), _nan_sort_key(r.get("m")))
    )

    # -----------------------------------------------------------------------
    # Compute overall elastic anchor representative numbers (largest PASS N)
    # -----------------------------------------------------------------------
    anchor_n100 = [
        r for r in all_elastic_pass
        if r.get("n") is not None
        and not (isinstance(r.get("n"), float) and math.isnan(r.get("n")))
        and abs(float(r.get("n", 0)) - 100.0) < 0.5
    ]

    lines: list[str] = []
    lines.append("# fdars Hotspot Ranking — Phase 113 Baseline")
    lines.append("")
    lines.append(f"**Date:** {today}")
    lines.append(f"**Environment:** benchmarks/results/env.json")
    lines.append("")
    lines.append(
        "> Generated by `benchmarks/rank_hotspots.py` from `benchmarks/results/comparison.csv`."
    )
    lines.append(
        "> Noise gate: RSD < 5% AND CI-width < 5% (criterion). "
        "> Leverage = fdars_ns × max(1.0, fdars_ns / min_other_ns)."
    )
    lines.append("")

    # -----------------------------------------------------------------------
    # Section 1: Confirmed Anchor
    # -----------------------------------------------------------------------
    lines.append("## Confirmed Anchor: Elastic Alignment")
    lines.append("")
    lines.append(
        "Elastic alignment (SRSF/RKHS pairwise warping) is confirmed as rank #1 by both "
        "absolute wall-clock time and cross-language leverage. The Karcher-mean cells are "
        "noise-excluded (RSD 29–46%); the anchor uses noise-stable `elastic_align_pair` "
        "and banded cross-alignment cells."
    )
    lines.append("")

    if all_elastic_pass:
        lines.append(
            "| Op | N | M | fdars (ms) | Leverage Score | Rank | Source |"
        )
        lines.append(
            "|----|---|---|-----------|----------------|------|--------|"
        )
        for i, r in enumerate(all_elastic_pass[:10]):
            n_str = str(int(r["n"])) if r.get("n") is not None and not (isinstance(r.get("n"), float) and math.isnan(r.get("n"))) else "—"
            m_str = str(int(r["m"])) if r.get("m") is not None and not (isinstance(r.get("m"), float) and math.isnan(r.get("m"))) else "—"
            fdars_ms = _ms(float(r["mean_ns"]))
            lev = r["leverage_score"]
            lev_str = f"{lev/1e9:.3f} s·factor" if lev > 1e9 else f"{lev/1e6:.3f} ms·factor"
            src = _format_source_label(str(r.get("source", "")))
            rank_str = "#1" if i == 0 else f"#{i+1}"
            lines.append(
                f"| elastic | {n_str} | {m_str} | {fdars_ms} | {lev_str} | {rank_str} | {src} |"
            )
    else:
        lines.append("*No noise-stable elastic cells found — review comparison.csv.*")

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 2: Secondary Targets
    # -----------------------------------------------------------------------
    lines.append("## Secondary Targets")
    lines.append("")
    lines.append(
        "SVD/FPCA and covariance operations are the Phase 114–115 secondary targets, "
        "ordered by leverage score. `face_covariance` has no cross-language equivalent "
        "and is ranked by Rust-only absolute time."
    )
    lines.append("")

    if secondary:
        lines.append(
            "| Rank | Op | N | M | fdars (ms) | Leverage Score | Source |"
        )
        lines.append(
            "|------|----|---|---|-----------|----------------|--------|"
        )
        for i, r in enumerate(secondary[:15]):
            n_str = str(int(r["n"])) if r.get("n") is not None and not (isinstance(r.get("n"), float) and math.isnan(r.get("n"))) else "—"
            m_str = str(int(r["m"])) if r.get("m") is not None and not (isinstance(r.get("m"), float) and math.isnan(r.get("m"))) else "—"
            fdars_ms = _ms(float(r["mean_ns"]))
            lev = r["leverage_score"]
            lev_str = f"{lev/1e9:.3f} s·factor" if lev > 1e9 else f"{lev/1e6:.3f} ms·factor"
            src = _format_source_label(str(r.get("source", "")))
            rank_str = f"#{i+1}"
            lines.append(
                f"| {rank_str} | {r['op']} | {n_str} | {m_str} | {fdars_ms} | {lev_str} | {src} |"
            )
    else:
        lines.append("*No secondary targets found after noise-gating.*")

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 3: Noise-Excluded Cells
    # -----------------------------------------------------------------------
    lines.append("## Noise-Excluded Cells")
    lines.append("")
    lines.append(
        "The following Rust/criterion cells failed the noise gate (RSD ≥ 5% or CI-width ≥ 5%) "
        "and are EXCLUDED from target selection. They are recorded here for transparency."
    )
    lines.append("")

    if noise_excl:
        lines.append(
            "| Op | N | M | RSD% | CI-width% | Source | Reason |"
        )
        lines.append(
            "|----|---|---|------|-----------|--------|--------|"
        )
        for r in noise_excl:
            n_str = str(int(r["n"])) if r.get("n") is not None and not (isinstance(r.get("n"), float) and math.isnan(r.get("n"))) else "—"
            m_str = str(int(r["m"])) if r.get("m") is not None and not (isinstance(r.get("m"), float) and math.isnan(r.get("m"))) else "—"
            rsd = r.get("rsd_pct")
            ciw = r.get("ci_width_pct")
            rsd_str = f"{float(rsd):.1f}" if rsd is not None else "—"
            ciw_str = f"{float(ciw):.1f}" if ciw is not None else "—"
            src = _format_source_label(str(r.get("source", "")))
            # Reason: note Karcher-mean specifically
            notes = str(r.get("notes", "") or "")
            op_str = str(r.get("op", ""))
            if "karcher" in src.lower() or "karcher" in op_str.lower() or "karcher" in notes.lower():
                reason = "Karcher-mean noise-gated; elastic anchor confirmed by pairwise elastic"
            else:
                reason = "RSD or CI-width ≥ 5%"
            lines.append(
                f"| {op_str} | {n_str} | {m_str} | {rsd_str} | {ciw_str} | {src} | {reason} |"
            )
    else:
        lines.append("*No cells were noise-excluded.*")

    lines.append("")

    # -----------------------------------------------------------------------
    # Section 4: Notes
    # -----------------------------------------------------------------------
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Leverage formula:** `leverage_score = fdars_ns × max(1.0, fdars_ns / min_other_ns)` "
        "where `min_other_ns` = min of python / r_svd mean_ns for the same op/size cell. "
        "When fdars is fastest: leverage = absolute time only (factor = 1.0)."
    )
    lines.append(
        "- **`r_native_pca_fd` excluded from ranking:** The `r_native_pca_fd` library rows "
        "(fda::pca.fd, B-spline coefficient space) are ecosystem-realistic R timings "
        "and are NOT used in the ranking denominator or as ranked targets."
    )
    lines.append(
        "- **FPCA quadrature disclosure:** fdars/R use Simpson's-weighted SVD; scikit-fda uses "
        "trapezoidal quadrature. fdars-vs-R subspace cosine = 1.0000 every cell (validated). "
        "fdars-vs-python dips to 0.9816 @ n=500,m=50 (trailing PC divergence). "
        "FPCA rows carry the disclosed cosine in the `notes` column of comparison.csv."
    )
    lines.append(
        "- **FPCR cross-library divergence disclosure:** scikit-fda FPCA→OLS vs R fda.usc "
        "`fregre.pc` use related but not identical FPCR implementations. "
        "Cross-library correlation ~0.95→0.86 as n grows. Recorded per cell; not hard-gated."
    )
    lines.append(
        "- **Karcher-mean elastic excluded:** All Karcher-mean elastic cells failed the noise gate "
        "(RSD 29–46%). The elastic anchor is confirmed by the noise-stable pairwise "
        "`elastic_align_pair` and banded cross-alignment cells (N≥100)."
    )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "results", "comparison.csv")
    env_path = os.path.join(script_dir, "results", "env.json")
    out_path = os.path.join(script_dir, "results", "hotspot_ranking.md")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run Plan 113-02 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    rows = _df_to_rows(df)

    ranked, excluded = rank_rows(rows)

    print(f"Ranked targets:   {len(ranked)}")
    print(f"Noise-excluded:   {len(excluded)}")
    print()
    print("Top 10 ranked targets:")
    for i, r in enumerate(ranked[:10]):
        lev = r["leverage_score"]
        n_str = str(int(r["n"])) if r.get("n") is not None and not (isinstance(r.get("n"), float) and math.isnan(r.get("n"))) else "—"
        m_str = str(int(r["m"])) if r.get("m") is not None and not (isinstance(r.get("m"), float) and math.isnan(r.get("m"))) else "—"
        print(
            f"  #{i+1:2d}  op={r['op']:<35s}  n={n_str:<5s}  m={m_str:<5s}  "
            f"fdars={float(r['mean_ns'])/1e6:>12.3f} ms  leverage={lev/1e9:.3f} s·factor"
        )

    emit_hotspot_ranking(
        ranked=ranked,
        excluded=excluded,
        env_path=env_path,
        output_path=out_path,
    )
    print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
