"""
Criterion output collector for the fdars cross-language harness.

Parses `target/criterion/*/new/estimates.json` files and emits structured
timing records to benchmarks/results/rust.json.

The fdars side does NOT emit FPCA loadings via criterion — only timing.
The agreement gate compares Python vs R loadings (both extracted by their
runners). The fdars timing is collected here from the existing criterion
output files, with no new dev-dependency.

Usage:
    python collect_rust.py            # collect all groups, write rust.json
    python collect_rust.py --check    # print available groups and exit
    python collect_rust.py --group fdata_to_pc   # filter to a specific group

Group-to-op mapping (RESEARCH.md Table 3 — criterion groups for overlapping ops):
  audit_fpca, audit_p4_fpca, fdata_to_pc -> fpca
  smooth_basis, bspline_basis              -> basis_smoothing
  fraiman_muniz, audit_depth, audit_streaming_depth -> depth
  modified_band                            -> depth        (MBD variant)
  functional_boxplot                       -> functional_boxplot
  fregre_lm                                -> fpcr
  elastic_align_pair, karcher_mean, audit_karcher,
    audit_p3_karcher, audit_p4_elastic_fpca -> elastic
"""

import argparse
import json
import pathlib
import sys
from typing import Iterator

# Resolve CRITERION_DIR relative to the workspace root (parent of benchmarks/)
_SCRIPT_DIR = pathlib.Path(__file__).parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parent
CRITERION_DIR = _WORKSPACE_ROOT / "target" / "criterion"

# -------------------------------------------------------------------
# Group-to-shared-op mapping (BENCH-03)
# Maps criterion group names to the shared `op` labels used by the runners.
# Groups not in this map are collected without an op label (timing only).
# -------------------------------------------------------------------

OP_MAP: dict[str, str] = {
    # FPCA
    "audit_fpca": "fpca",
    "audit_p4_fpca": "fpca",
    "fdata_to_pc": "fpca",
    # Basis smoothing
    "smooth_basis": "basis_smoothing",
    "bspline_basis": "basis_smoothing",
    # Functional depth (Fraiman-Muniz + Modified Band Depth)
    "fraiman_muniz": "depth",
    "modified_band": "depth",
    "audit_depth": "depth",
    "audit_streaming_depth": "depth",
    # Functional boxplot
    "functional_boxplot": "functional_boxplot",
    # FPCR regression
    "fregre_lm": "fpcr",
    # Elastic alignment / Karcher mean
    "elastic_align_pair": "elastic",
    "karcher_mean": "elastic",
    "audit_karcher": "elastic",
    "audit_p3_karcher": "elastic",
    "audit_p3_karcher_banded": "elastic",
    "audit_p4_elastic_fpca": "elastic",
    "audit_p3_elastic_cross": "elastic",
    "audit_p3_elastic_cross_banded": "elastic",
    "audit_p3_elastic_self": "elastic",
    "audit_p3_elastic_self_banded": "elastic",
    "audit_elastic": "elastic",
    "audit_p5_karcher_paybackN": "elastic",
    "audit_p5_karcher_threads": "elastic",
}


def iter_estimates(criterion_dir: pathlib.Path = CRITERION_DIR) -> Iterator[dict]:
    """
    Yield one record per new/estimates.json found under criterion_dir.

    Record fields:
        group:         first path component (benchmark group name)
        bench:         remaining path components joined with '/' (bench ID)
        mean_ns:       mean.point_estimate in nanoseconds
        ci_lower_ns:   mean confidence interval lower bound (ns)
        ci_upper_ns:   mean confidence interval upper bound (ns)
        std_err_ns:    mean standard error (ns)
        source_path:   string path to the estimates.json (for debugging)
    """
    # Security: assert criterion_dir is a resolved absolute path
    criterion_dir = criterion_dir.resolve()

    for est_path in criterion_dir.rglob("new/estimates.json"):
        # Threat mitigation T-112-01: assert each path is under CRITERION_DIR
        try:
            est_path.relative_to(criterion_dir)
        except ValueError:
            # Path traversal: skip silently
            continue

        rel = est_path.relative_to(criterion_dir)
        parts = rel.parts
        # parts: ("group", [bench_segment...], "new", "estimates.json")
        # We strip the last two components ("new", "estimates.json")
        if len(parts) < 3:
            continue  # unexpected layout; skip
        group = parts[0]
        bench = "/".join(parts[1:-2])  # everything between group and "new"

        try:
            with est_path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        mean = data.get("mean", {})
        ci = mean.get("confidence_interval", {})
        yield {
            "group": group,
            "bench": bench,
            "mean_ns": mean.get("point_estimate", 0.0),
            "ci_lower_ns": ci.get("lower_bound", 0.0),
            "ci_upper_ns": ci.get("upper_bound", 0.0),
            "std_err_ns": mean.get("standard_error", 0.0),
            "source_path": str(est_path),
        }


def collect(group_filter: str | None = None,
            criterion_dir: pathlib.Path = CRITERION_DIR) -> list[dict]:
    """
    Collect all estimates, optionally filtered to a single group.

    Returns a list of records in the shared comparison schema:
      - 'op': shared op label from OP_MAP (e.g. 'fpca', 'elastic') — falls back
              to the raw group name for groups not in the map.
      - 'library': 'rust'
      - 'group': original criterion group name
      - 'bench', 'mean_ns', 'ci_lower_ns', 'ci_upper_ns', 'std_err_ns'

    No result_fingerprint — Rust side emits timing only from criterion.
    """
    records = []
    for rec in iter_estimates(criterion_dir):
        if group_filter and rec["group"] != group_filter:
            continue
        # Map criterion group name to shared op label
        op = OP_MAP.get(rec["group"], rec["group"])
        # Emit in the shared comparison schema
        records.append({
            "op": op,
            "library": "rust",
            "group": rec["group"],
            "bench": rec["bench"],
            "mean_ns": rec["mean_ns"],
            "ci_lower_ns": rec["ci_lower_ns"],
            "ci_upper_ns": rec["ci_upper_ns"],
            "std_err_ns": rec["std_err_ns"],
            # No result_fingerprint on the Rust side — timing only from criterion
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect criterion full-op timing output")
    parser.add_argument("--check", action="store_true",
                        help="Print available criterion groups and exit")
    parser.add_argument("--group", type=str, default=None,
                        help="Filter to a specific criterion group name")
    args = parser.parse_args()

    if not CRITERION_DIR.exists():
        print(
            f"WARN: criterion output directory not found: {CRITERION_DIR}\n"
            "Run: cargo bench -p fdars-core --features linalg,parallel "
            "--bench regression_benchmarks --bench depth_benchmarks --bench alignment_benchmarks",
            file=sys.stderr,
        )
        if args.check:
            sys.exit(1)

    if args.check:
        # Print available groups and sample mean_ns values
        groups: dict[str, list[float]] = {}
        for rec in iter_estimates():
            groups.setdefault(rec["group"], []).append(rec["mean_ns"])
        if not groups:
            print("No criterion output found.", file=sys.stderr)
            sys.exit(1)
        print(f"Available criterion groups ({len(groups)}):")
        for g, timings in sorted(groups.items()):
            print(f"  {g}: {len(timings)} bench(es), "
                  f"example mean_ns={timings[0]:.0f}")
        return

    records = collect(group_filter=args.group)

    script_dir = pathlib.Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "rust.json"

    with out_path.open("w") as f:
        json.dump(records, f, indent=2)

    print(f"Collected {len(records)} criterion record(s)")
    if records:
        # Group by shared op label for summary
        from collections import defaultdict
        by_op: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_op[r["op"]].append(r)
        for op, op_records in sorted(by_op.items()):
            print(f"  op={op!r}: {len(op_records)} bench(es), "
                  f"e.g. group={op_records[0]['group']!r} mean_ns={op_records[0]['mean_ns']:.0f}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
