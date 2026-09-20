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

    Returns a list of records with 'op' set to the group name and 'library'='rust'.
    """
    records = []
    for rec in iter_estimates(criterion_dir):
        if group_filter and rec["group"] != group_filter:
            continue
        # Emit in the shared comparison schema
        records.append({
            "op": rec["group"],
            "library": "rust",
            "bench": rec["bench"],
            "mean_ns": rec["mean_ns"],
            "ci_lower_ns": rec["ci_lower_ns"],
            "ci_upper_ns": rec["ci_upper_ns"],
            "std_err_ns": rec["std_err_ns"],
            # No result_fingerprint on the Rust side — timing only from criterion
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect criterion FPCA timing output")
    parser.add_argument("--check", action="store_true",
                        help="Print available criterion groups and exit")
    parser.add_argument("--group", type=str, default=None,
                        help="Filter to a specific criterion group name")
    args = parser.parse_args()

    if not CRITERION_DIR.exists():
        print(
            f"WARN: criterion output directory not found: {CRITERION_DIR}\n"
            "Run: cargo bench -p fdars-core --features linalg,parallel "
            "--bench regression_benchmarks",
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
        fpca_records = [r for r in records
                        if "fpca" in r["op"].lower() or "fdata_to_pc" in r["op"].lower()]
        print(f"  FPCA-related records: {len(fpca_records)}")
        for r in fpca_records[:3]:
            print(f"    {r['op']}/{r['bench']}: mean_ns={r['mean_ns']:.0f}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
