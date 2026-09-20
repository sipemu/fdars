"""
Environment metadata capture for the fdars cross-language harness.

Writes benchmarks/results/env.json with:
  - thread_env: all six thread-count vars (must all be "1" during benchmark runs)
  - cpu: model name and logical core count
  - python: version, numpy version, numpy BLAS backend
  - scikit_fda: version
  - rust: version string, features used
  - r: version, BLAS path, LAPACK version
  - r_packages: fda, fda.usc, roahd, fdasrvf, bench, refund, fdapace, jsonlite
  - criterion_version: "0.5"
  - timestamp: ISO-8601 UTC

Usage:
    python capture_env.py
    python capture_env.py --out-dir benchmarks/results
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command safely (list-form only — no shell=True)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kwargs)


def _capture_cpu() -> dict:
    """Read CPU model and logical core count from /proc/cpuinfo."""
    model = ""
    cores = 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                line = line.strip()
                if line.startswith("model name") and not model:
                    model = line.split(":", 1)[1].strip()
                if line.startswith("processor"):
                    cores += 1
    except OSError:
        model = "unknown"
        cores = 0
    return {"model": model, "cores_logical": cores}


def _capture_python() -> dict:
    """Capture Python version and numpy BLAS backend."""
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        import numpy as np
        np_version = np.__version__
        # Capture numpy BLAS info via show_config() stdout
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        config_text = buf.getvalue()
        # Extract blas library names from the config output
        blas_line = ""
        for line in config_text.splitlines():
            low = line.lower()
            if "blas" in low and ("lib" in low or "openblas" in low or "mkl" in low
                                  or "scipy" in low or "version" in low):
                blas_line = line.strip()
                break
        # Try numpy 2.x API
        try:
            info = np.__config__.blas_opt_info  # type: ignore[attr-defined]
            blas_str = str(info.get("libraries", blas_line))
        except AttributeError:
            blas_str = blas_line or "unknown"
    except ImportError:
        np_version = "not installed"
        blas_str = "unknown"

    return {
        "version": py_version,
        "numpy_version": np_version,
        "numpy_blas": blas_str,
    }


def _capture_scikit_fda() -> dict:
    """Capture scikit-fda version."""
    try:
        import skfda
        version = getattr(skfda, "__version__", "unknown")
    except ImportError:
        version = "not installed"
    return {"version": version}


def _capture_rust() -> dict:
    """Capture Rust toolchain version."""
    result = _run(["rustc", "--version"])
    version = result.stdout.strip() if result.returncode == 0 else "unknown"
    return {
        "version": version,
        "features": "linalg,parallel",
    }


def _capture_r_env() -> dict:
    """Capture R version, BLAS path, LAPACK version, and package versions."""
    r_script = r"""
cat(R.version$major, ".", R.version$minor, "\n", sep="")
cat(sessionInfo()$BLAS, "\n")
cat(sessionInfo()$LAPACK, "\n")
pkgs <- c("fda", "fda.usc", "roahd", "fdasrvf", "bench", "refund", "fdapace", "jsonlite")
for (p in pkgs) {
  v <- tryCatch(as.character(packageVersion(p)), error=function(e) "NA")
  cat(p, ":", v, "\n")
}
"""
    result = _run(["Rscript", "--no-save", "--no-restore", "-e", r_script])

    r_env = {
        "version": "unknown",
        "blas": "unknown",
        "lapack_version": "unknown",
    }
    pkg_versions: dict[str, str] = {}

    if result.returncode == 0:
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) >= 1:
            r_env["version"] = lines[0]
        if len(lines) >= 2:
            r_env["blas"] = lines[1]
        if len(lines) >= 3:
            r_env["lapack_version"] = lines[2]
        for line in lines[3:]:
            if " : " in line:
                pkg, ver = line.split(" : ", 1)
                pkg_versions[pkg.strip()] = ver.strip()

    return r_env, pkg_versions


def _capture_thread_env() -> dict:
    """Read the thread-count environment variables from os.environ."""
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    return {k: os.environ.get(k, "not set") for k in keys}


def capture_all() -> dict:
    """Capture complete environment metadata."""
    cpu = _capture_cpu()
    py = _capture_python()
    skfda = _capture_scikit_fda()
    rust = _capture_rust()
    r_env, r_packages = _capture_r_env()
    thread_env = _capture_thread_env()

    return {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cpu": cpu,
        "rust": rust,
        "python": py,
        "scikit_fda": skfda,
        "r": r_env,
        "r_packages": r_packages,
        "thread_env": thread_env,
        "criterion_version": "0.5",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture benchmark environment metadata")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: benchmarks/results/)")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    results_dir = pathlib.Path(args.out_dir) if args.out_dir else script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "env.json"

    env_data = capture_all()

    with out_path.open("w") as f:
        json.dump(env_data, f, indent=2)

    print(f"Captured environment metadata: {out_path}")
    print(f"  CPU: {env_data['cpu']['model']} ({env_data['cpu']['cores_logical']} logical cores)")
    print(f"  Python: {env_data['python']['version']}")
    print(f"  numpy BLAS: {env_data['python']['numpy_blas']}")
    print(f"  R: {env_data['r']['version']}")
    print(f"  R BLAS: {env_data['r']['blas']}")
    print(f"  RAYON_NUM_THREADS: {env_data['thread_env']['RAYON_NUM_THREADS']}")
    print(f"  OMP_NUM_THREADS: {env_data['thread_env']['OMP_NUM_THREADS']}")


if __name__ == "__main__":
    main()
