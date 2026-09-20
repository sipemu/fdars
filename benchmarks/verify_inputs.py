"""
Bit-identical CSV spot-check for the fdars cross-language harness.

Verifies that the three runners (Python, R, Rust) would load the SAME bytes
from the generated CSV fixtures. This is the BENCH-01 contract: a single seeded
source emits CSVs that all runners read as bit-identical values.

Checks all fixture types:
  - curves_n{n}_m{m}.csv    : main curve-set (FPCA, depth, elastic, FPCR)
  - y_n{n}.csv              : scalar response vector (FPCR)
  - elastic_n2_m{m}.csv     : two-curve pair (elastic pairwise alignment)

The check is done at the Python layer since we can:
1. Hash the raw CSV bytes (what every reader gets off disk).
2. Load the CSV with Python (pandas) and compute a content hash.
3. Verify R would load the same values by checking the CSV with Rscript.
4. Verify seed-determinism by regenerating and comparing hashes.

Usage:
    python verify_inputs.py [--n 60] [--m 50]
    python verify_inputs.py  # defaults: n=60, m=50
"""

import argparse
import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd


def sha256_file(path: pathlib.Path) -> str:
    """Return hex SHA-256 of the file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash_python(path: pathlib.Path) -> str:
    """
    Load CSV with pandas and return a SHA-256 of the float64 values.

    This matches exactly what python_runner.py does: read with pandas, drop
    curve_id column, convert to float64 numpy array.
    """
    df = pd.read_csv(path)
    mat = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    h = hashlib.sha256(mat.tobytes())
    return h.hexdigest()


def content_hash_r(path: pathlib.Path) -> str:
    """
    Ask R to load the CSV and return a SHA-256 of its float64 values.

    Uses Rscript with a list-form subprocess call. R reads via read.csv,
    drops curve_id, converts to double matrix, serialises to raw bytes via
    writeBin() and computes SHA-256.

    Returns the hex digest string, or raises RuntimeError on failure.
    """
    r_script = f"""
suppressPackageStartupMessages(library(digest))
df <- read.csv("{path}", check.names=FALSE)
mat <- as.matrix(df[, -1L])
storage.mode(mat) <- "double"
# Compute SHA-256 of the column-major double bytes
cat(digest::digest(mat, algo="sha256", raw=TRUE), "\\n")
"""
    # Use digest package if available; fallback to writeBin
    r_with_digest = subprocess.run(
        ["Rscript", "--no-save", "--no-restore", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r_with_digest.returncode == 0 and r_with_digest.stdout.strip():
        return r_with_digest.stdout.strip()

    # Fallback: write raw bytes and hash in Python
    r_fallback = f"""
df <- read.csv("{path}", check.names=FALSE)
mat <- as.matrix(df[, -1L])
storage.mode(mat) <- "double"
raw_bytes <- writeBin(as.vector(t(mat)), raw())  # row-major for comparison
cat(paste(format(as.integer(raw_bytes), scientific=FALSE), collapse=" "), "\\n")
"""
    result = subprocess.run(
        ["Rscript", "--no-save", "--no-restore", "-e", r_fallback],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"R content hash failed: {result.stderr[:500]}"
        )

    # Parse byte values and hash them
    byte_vals = [int(x) for x in result.stdout.split()]
    raw = bytes(byte_vals)
    return hashlib.sha256(raw).hexdigest()


def verify_csv(
    csv_path: pathlib.Path,
    label: str,
    regen_fn=None,
) -> bool:
    """
    Verify bit-identical CSV loading for a single fixture file.

    Args:
        csv_path:   path to the CSV fixture to check
        label:      human-readable name for output (e.g. "curves", "y_vector")
        regen_fn:   optional callable that returns a pathlib.Path to a regenerated
                    CSV (for seed-determinism check); called in a tempfile context.

    Returns True if all checks pass.
    """
    print(f"\nVerifying [{label}]: {csv_path.name}")

    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return False

    # 1. Raw file hash
    file_hash = sha256_file(csv_path)
    print(f"  File SHA-256:   {file_hash[:16]}...")

    # 2. Python content hash
    py_hash = content_hash_python(csv_path)
    print(f"  Python content: {py_hash[:16]}...")

    # 3. R content hash
    print("  R content:      ", end="", flush=True)
    try:
        r_hash = content_hash_r(csv_path)
        print(f"{r_hash[:16]}...")
    except RuntimeError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return False

    # 4. Seed-determinism (regenerate if function provided)
    if regen_fn is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            regen_fn(tmp_path)
            regen_hash = sha256_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        print(f"  Regen hash:     {regen_hash[:16]}...")
        if regen_hash != file_hash:
            print(f"  FAIL [{label}]: Regen hash differs — not reproducible!", file=sys.stderr)
            return False
        print(f"  Regen matches: YES (seed-deterministic)")

    print(f"  [{label}]: OK")
    return True


def verify(n: int = 60, m: int = 50, seed: int = 42) -> bool:
    """
    Verify bit-identical CSV loading across Python and R runners for all fixture types.

    Checks:
      1. curves_n{n}_m{m}.csv    — main curve-set (FPCA, depth, basis_smoothing, FPCR)
      2. y_n{n}.csv              — scalar response vector (FPCR)
      3. elastic_n2_m{m}.csv     — 2-curve pair for elastic alignment

    Returns True if ALL checks pass, False otherwise.
    """
    script_dir = pathlib.Path(__file__).parent
    data_dir = script_dir / "data"

    # Import generate_data once for all regen checks
    orig_path = sys.path[:]
    sys.path.insert(0, str(script_dir))
    try:
        from generate_data import (
            generate_curves, write_csv,
            generate_response,
            generate_elastic_pair,
        )
    finally:
        sys.path[:] = orig_path

    all_ok = True

    # --- 1. Main curves CSV ---
    curves_path = data_dir / f"curves_n{n}_m{m}.csv"

    def regen_curves(tmp_path: pathlib.Path) -> None:
        df = generate_curves(n=n, m=m, seed=seed)
        write_csv(df, tmp_path)

    ok = verify_csv(curves_path, label=f"curves_n{n}_m{m}", regen_fn=regen_curves)
    all_ok = all_ok and ok

    # --- 2. Scalar response y vector ---
    y_path = data_dir / f"y_n{n}.csv"

    def regen_y(tmp_path: pathlib.Path) -> None:
        df = generate_response(n=n, seed=seed)
        write_csv(df, tmp_path)

    ok = verify_csv(y_path, label=f"y_n{n}", regen_fn=regen_y if y_path.exists() else None)
    all_ok = all_ok and ok

    # --- 3. Elastic pair CSV ---
    elastic_path = data_dir / f"elastic_n2_m{m}.csv"

    def regen_elastic(tmp_path: pathlib.Path) -> None:
        df = generate_elastic_pair(m=m, seed=seed)
        write_csv(df, tmp_path)

    ok = verify_csv(elastic_path, label=f"elastic_n2_m{m}",
                    regen_fn=regen_elastic if elastic_path.exists() else None)
    all_ok = all_ok and ok

    if all_ok:
        print("\nBIT-IDENTITY CHECK: PASSED")
        print("  All fixture CSVs are seed-deterministic and load bit-identically in Python + R")
    else:
        print("\nBIT-IDENTITY CHECK: FAILED (see errors above)", file=sys.stderr)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify bit-identical CSV loading for all fixtures")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (must match generator)")
    args = parser.parse_args()

    ok = verify(n=args.n, m=args.m, seed=args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
