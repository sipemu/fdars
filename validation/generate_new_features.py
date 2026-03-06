#!/usr/bin/env python3
"""Generate validation data for Soft-DTW, Landmark Registration, and TSRVF.

Uses:
- tslearn: Soft-DTW reference implementation (Cuturi & Blondel, ICML 2017)
- fdasrsf: TSRVF reference implementation (Tucker et al.)
- scipy: Fritsch-Carlson monotone interpolation for landmark registration

Outputs JSON fixtures in validation/expected/ matching the project's convention.
"""

import json
import numpy as np
from pathlib import Path

VALIDATION_DIR = Path(__file__).parent


def load_standard_data():
    """Load the standard 50x101 test dataset."""
    with open(VALIDATION_DIR / "data" / "standard_50x101.json") as f:
        d = json.load(f)
    n, m = d["n"], d["m"]
    # Column-major flat array → (n, m) row-major numpy array
    data = np.array(d["data"]).reshape(m, n).T  # (m, n) col-major → (n, m)
    argvals = np.array(d["argvals"])
    return data, argvals, n, m


def generate_soft_dtw_expected():
    """Generate Soft-DTW reference values using tslearn."""
    from tslearn.metrics import soft_dtw as tslearn_soft_dtw

    data, argvals, n, m = load_standard_data()

    # Use first 5 curves for pairwise tests
    n_sub = 5
    sub = data[:n_sub]

    gamma = 1.0

    # Pairwise distance: curves 0 and 1
    d01 = tslearn_soft_dtw(sub[0].reshape(-1, 1), sub[1].reshape(-1, 1), gamma=gamma)

    # Self-distance (sdtw(x,x)) for divergence
    d00 = tslearn_soft_dtw(sub[0].reshape(-1, 1), sub[0].reshape(-1, 1), gamma=gamma)
    d11 = tslearn_soft_dtw(sub[1].reshape(-1, 1), sub[1].reshape(-1, 1), gamma=gamma)

    # Divergence: sdtw(x,y) - 0.5*(sdtw(x,x) + sdtw(y,y))
    div_01 = d01 - 0.5 * (d00 + d11)

    # 5x5 self-distance matrix (off-diagonal)
    dist_matrix = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            d = tslearn_soft_dtw(
                sub[i].reshape(-1, 1), sub[j].reshape(-1, 1), gamma=gamma
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # 5x5 divergence matrix (off-diagonal)
    self_dists = []
    for i in range(n_sub):
        sd = tslearn_soft_dtw(
            sub[i].reshape(-1, 1), sub[i].reshape(-1, 1), gamma=gamma
        )
        self_dists.append(sd)
    div_matrix = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            dv = dist_matrix[i, j] - 0.5 * (self_dists[i] + self_dists[j])
            div_matrix[i, j] = dv
            div_matrix[j, i] = dv

    # Small gamma convergence test (should approach hard DTW)
    d01_small = tslearn_soft_dtw(
        sub[0].reshape(-1, 1), sub[1].reshape(-1, 1), gamma=0.001
    )

    # Single-point test
    x1 = np.array([[3.0]])
    y1 = np.array([[5.0]])
    d_single = tslearn_soft_dtw(x1, y1, gamma=1.0)

    # Multi-gamma sweep for curves 0 vs 1
    gamma_sweep = {}
    for g in [0.01, 0.1, 1.0, 10.0]:
        d = tslearn_soft_dtw(sub[0].reshape(-1, 1), sub[1].reshape(-1, 1), gamma=g)
        gamma_sweep[str(g)] = float(d)

    return {
        "gamma": gamma,
        "n_sub": n_sub,
        "distance_01": float(d01),
        "self_distance_00": float(d00),
        "self_distance_11": float(d11),
        "divergence_01": float(div_01),
        "distance_matrix": dist_matrix.flatten().tolist(),  # row-major
        "divergence_matrix": div_matrix.flatten().tolist(),  # row-major
        "distance_01_small_gamma": float(d01_small),
        "single_point_distance": float(d_single),
        "gamma_sweep": gamma_sweep,
    }


def generate_landmark_expected():
    """Generate landmark registration reference values using scipy."""
    from scipy.interpolate import PchipInterpolator

    # Test monotone interpolation (Fritsch-Carlson / PCHIP)
    # PCHIP is scipy's implementation of monotone cubic Hermite interpolation
    source = np.array([0.0, 0.3, 0.6, 1.0])
    target = np.array([0.0, 0.4, 0.7, 1.0])

    # Build warp: gamma(target[k]) = source[k]
    # So x=target, y=source for the interpolant
    interp = PchipInterpolator(target, source)

    eval_points = np.linspace(0, 1, 101)
    warp_values = interp(eval_points).tolist()

    # Verify monotonicity
    is_monotone = all(
        warp_values[i + 1] >= warp_values[i] for i in range(len(warp_values) - 1)
    )

    # Test with sinusoidal data and known peak positions
    m = 201
    t = np.linspace(0, 1, m)
    shifts = [0.0, 0.03, -0.02, 0.04, -0.03]
    n = len(shifts)

    curves = np.array([np.sin(2 * np.pi * (t - s)) for s in shifts])

    # Peak positions (analytical: peak of sin(2π(t-s)) is at t = s + 0.25)
    peak_positions = [s + 0.25 for s in shifts]

    # Second test case with more extreme warp
    source2 = np.array([0.0, 0.2, 0.8, 1.0])
    target2 = np.array([0.0, 0.5, 0.6, 1.0])
    interp2 = PchipInterpolator(target2, source2)
    warp_values2 = interp2(eval_points).tolist()
    is_monotone2 = all(
        warp_values2[i + 1] >= warp_values2[i] for i in range(len(warp_values2) - 1)
    )

    return {
        "pchip_source": source.tolist(),
        "pchip_target": target.tolist(),
        "pchip_eval_points": eval_points.tolist(),
        "pchip_warp_values": warp_values,
        "pchip_is_monotone": is_monotone,
        "pchip_source2": source2.tolist(),
        "pchip_target2": target2.tolist(),
        "pchip_warp_values2": warp_values2,
        "pchip_is_monotone2": is_monotone2,
        "peak_positions": peak_positions,
        "shifts": shifts,
        "n": n,
        "m": m,
    }


def generate_tsrvf_expected():
    """Generate TSRVF reference values using fdasrsf.

    Strategy: validate TSRVF transport independently of Karcher mean by
    (1) exporting fdasrsf's aligned SRSFs + mean, and
    (2) computing tangent vectors from those, so Rust can replicate the
        transport step using the same pre-aligned input.
    """
    import fdasrsf as fs

    data, argvals, n, m = load_standard_data()

    # Use a subset for speed
    n_sub = 10
    sub = data[:n_sub]

    # fdasrsf expects (m, n) shape — columns are curves
    f = sub.T  # (m, n_sub)
    time = argvals.copy()

    # Compute alignment using fdasrsf
    obj = fs.fdawarp(f, time)
    obj.srsf_align(MaxItr=30)

    # ── Component 1: sphere geometry ──
    # Test inv_exp_map and exp_map with known vectors
    # Create two unit vectors on the Hilbert sphere for testing
    psi1 = np.sqrt(2.0) * np.ones(m)  # constant, normalized
    psi1_norm = np.sqrt(np.trapezoid(psi1**2, time))
    psi1 = psi1 / psi1_norm

    # Another vector: slightly different
    psi2 = np.sqrt(2.0) * (1.0 + 0.3 * np.sin(2 * np.pi * time))
    psi2_norm = np.sqrt(np.trapezoid(psi2**2, time))
    psi2 = psi2 / psi2_norm

    # inv_exp_map: tangent vector from psi1 toward psi2
    ip = np.trapezoid(psi1 * psi2, time)
    ip = np.clip(ip, -1, 1)
    sphere_theta = np.arccos(ip)
    if sphere_theta > 1e-10:
        coeff = sphere_theta / np.sin(sphere_theta)
        v12 = coeff * (psi2 - np.cos(sphere_theta) * psi1)
    else:
        v12 = np.zeros(m)

    # exp_map: psi1 + v12 should recover psi2
    v_norm = np.sqrt(np.trapezoid(v12**2, time))
    if v_norm > 1e-10:
        psi2_recovered = np.cos(v_norm) * psi1 + np.sin(v_norm) * v12 / v_norm
    else:
        psi2_recovered = psi1.copy()

    sphere_round_trip_error = np.sqrt(np.trapezoid((psi2 - psi2_recovered)**2, time))

    # ── Component 2: TSRVF from fdasrsf's alignment ──
    mean_srsf = obj.mqn
    mean_srsf_norm = np.sqrt(np.trapezoid(mean_srsf**2, time))
    mu_unit = mean_srsf / mean_srsf_norm if mean_srsf_norm > 1e-10 else np.zeros(m)

    # Aligned SRSFs and their norms
    aligned_srsf_norms = []
    tangent_vectors = []
    for i in range(n_sub):
        qi = obj.qn[:, i]
        qi_norm = np.sqrt(np.trapezoid(qi**2, time))
        aligned_srsf_norms.append(float(qi_norm))

        if qi_norm > 1e-10 and mean_srsf_norm > 1e-10:
            qi_unit = qi / qi_norm
            ip = np.trapezoid(mu_unit * qi_unit, time)
            ip = np.clip(ip, -1, 1)
            theta = np.arccos(ip)
            if theta > 1e-10:
                coeff = theta / np.sin(theta)
                vi = coeff * (qi_unit - np.cos(theta) * mu_unit)
            else:
                vi = np.zeros(m)
        else:
            vi = np.zeros(m)
        tangent_vectors.append(vi.tolist())

    # Mean tangent vector (should be ≈ 0)
    mean_tv = np.mean(tangent_vectors, axis=0)
    mean_tv_norm = np.sqrt(np.trapezoid(mean_tv**2, time))

    # Per-curve tangent vector norms
    tv_norms = []
    for vi in tangent_vectors:
        vi = np.array(vi)
        tv_norms.append(float(np.sqrt(np.trapezoid(vi**2, time))))

    # ── Component 3: export pre-aligned data for Rust ──
    # This allows Rust to skip Karcher mean and test TSRVF transport directly
    # Export as column-major flat arrays matching FdMatrix convention
    aligned_srsfs_flat = []  # n_sub × m, column-major
    for j in range(m):
        for i in range(n_sub):
            aligned_srsfs_flat.append(float(obj.qn[j, i]))

    mean_srsf_flat = mean_srsf.tolist()

    # Also export the aligned curves for SRSF consistency check
    aligned_curves_flat = []  # n_sub × m, column-major
    for j in range(m):
        for i in range(n_sub):
            aligned_curves_flat.append(float(obj.fn[j, i]))

    mean_curve = obj.fmean.tolist()

    # Gammas from fdasrsf alignment (n_sub × m, column-major)
    gammas_flat = []
    for j in range(m):
        for i in range(n_sub):
            gammas_flat.append(float(obj.gam[j, i]))

    return {
        "n_sub": n_sub,
        "m": m,
        # Sphere geometry validation
        "sphere_psi1": psi1.tolist(),
        "sphere_psi2": psi2.tolist(),
        "sphere_v12": v12.tolist(),
        "sphere_theta": float(sphere_theta),
        "sphere_round_trip_error": float(sphere_round_trip_error),
        # Pre-aligned data from fdasrsf (column-major flat)
        "aligned_srsfs_flat": aligned_srsfs_flat,
        "aligned_curves_flat": aligned_curves_flat,
        "mean_srsf": mean_srsf_flat,
        "mean_curve": mean_curve,
        "gammas_flat": gammas_flat,
        # TSRVF results from fdasrsf's alignment
        "mean_srsf_norm": float(mean_srsf_norm),
        "aligned_srsf_norms": aligned_srsf_norms,
        "tangent_vectors_flat": np.array(tangent_vectors).flatten().tolist(),
        "tangent_vector_norms": tv_norms,
        "mean_tangent_norm": float(mean_tv_norm),
    }


def nadaraya_watson_gaussian(x, y, x_new, bandwidth):
    """Nadaraya-Watson kernel smoother with Gaussian kernel.

    Matches the Rust implementation in smoothing.rs exactly:
      K(u) = exp(-0.5 * u²) / sqrt(2π)
      ŷ(x₀) = Σ K((xᵢ - x₀)/h) yᵢ / Σ K((xᵢ - x₀)/h)
    """
    result = np.zeros(len(x_new))
    for idx, x0 in enumerate(x_new):
        u = (x - x0) / bandwidth
        w = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
        denom = w.sum()
        if denom > 1e-10:
            result[idx] = (w * y).sum() / denom
        else:
            result[idx] = 0.0
    return result


def generate_tsrvf_smoothed_expected():
    """Generate smoothed TSRVF reference values.

    Applies the same Nadaraya-Watson smoothing to aligned SRSFs as Rust's
    tsrvf_from_alignment does (bandwidth = 2/(m-1), Gaussian kernel), then
    computes tangent vectors from the smoothed data.

    This validates that Rust's smoothing + tangent vector computation matches
    Python applying the identical smoothing procedure to the same input data.

    Background:
    - R fdasrvf does NOT smooth — it suffers from DP kink spike artifacts
    - Python fdasrsf uses spline smoothing (s=1e-4) in SqrtMean(smooth=True)
      on warping functions, which is a different smoothing target
    - Our Rust code smooths aligned SRSFs with Nadaraya-Watson before computing
      tangent vectors — this test validates that exact pipeline
    """
    import fdasrsf as fs

    data, argvals, n, m = load_standard_data()

    # Use same subset as raw TSRVF test
    n_sub = 10
    sub = data[:n_sub]
    f = sub.T  # (m, n_sub)
    time = argvals.copy()

    # Compute alignment using fdasrsf (same as raw test)
    obj = fs.fdawarp(f, time)
    obj.srsf_align(MaxItr=30)

    # Internal [0,1] time grid and bandwidth matching Rust
    time_01 = np.linspace(0, 1, m)
    bandwidth = 2.0 / (m - 1)

    # Compute SRSFs of aligned curves (gradient-based, matching Rust's srsf_transform)
    h = (argvals[-1] - argvals[0]) / (m - 1)
    aligned_srsfs = np.zeros((m, n_sub))
    for i in range(n_sub):
        fi = obj.fn[:, i]
        grad = np.gradient(fi, h)
        aligned_srsfs[:, i] = np.sign(grad) * np.sqrt(np.abs(grad))

    # Use obj.mqn directly — this is the Karcher mean SRSF that Rust receives
    # as karcher.mean_srsf in tsrvf_from_alignment.
    # (NOT the SRSF of obj.fmean, which is a different quantity.)
    mean_srsf = obj.mqn

    # Smooth aligned SRSFs with NW (matching Rust's smooth_aligned_srsfs)
    smoothed_srsfs = np.zeros((m, n_sub))
    for i in range(n_sub):
        smoothed_srsfs[:, i] = nadaraya_watson_gaussian(
            time_01, aligned_srsfs[:, i], time_01, bandwidth
        )

    # Smooth mean SRSF with same NW (matching Rust's tsrvf_from_alignment)
    mean_srsf_smooth = nadaraya_watson_gaussian(time_01, mean_srsf, time_01, bandwidth)

    # Compute tangent vectors from smoothed data
    mean_srsf_norm = np.sqrt(np.trapezoid(mean_srsf_smooth**2, time_01))
    mu_unit = mean_srsf_smooth / mean_srsf_norm if mean_srsf_norm > 1e-10 else np.zeros(m)

    smoothed_srsf_norms = []
    tangent_vectors = []
    for i in range(n_sub):
        qi = smoothed_srsfs[:, i]
        qi_norm = np.sqrt(np.trapezoid(qi**2, time_01))
        smoothed_srsf_norms.append(float(qi_norm))

        if qi_norm > 1e-10 and mean_srsf_norm > 1e-10:
            qi_unit = qi / qi_norm
            ip = np.trapezoid(mu_unit * qi_unit, time_01)
            ip = np.clip(ip, -1, 1)
            theta = np.arccos(ip)
            if theta > 1e-10:
                coeff = theta / np.sin(theta)
                vi = coeff * (qi_unit - np.cos(theta) * mu_unit)
            else:
                vi = np.zeros(m)
        else:
            vi = np.zeros(m)
        tangent_vectors.append(vi.tolist())

    # Mean tangent vector norm (should be small)
    mean_tv = np.mean(tangent_vectors, axis=0)
    mean_tv_norm = np.sqrt(np.trapezoid(mean_tv**2, time_01))

    tv_norms = []
    for vi in tangent_vectors:
        vi = np.array(vi)
        tv_norms.append(float(np.sqrt(np.trapezoid(vi**2, time_01))))

    # Export smoothed aligned SRSFs (column-major for FdMatrix)
    smoothed_srsfs_flat = []
    for j in range(m):
        for i in range(n_sub):
            smoothed_srsfs_flat.append(float(smoothed_srsfs[j, i]))

    return {
        "n_sub": n_sub,
        "m": m,
        # Smoothed mean SRSF
        "smoothed_mean_srsf": mean_srsf_smooth.tolist(),
        "smoothed_mean_srsf_norm": float(mean_srsf_norm),
        # Smoothed aligned SRSFs (column-major flat)
        "smoothed_srsfs_flat": smoothed_srsfs_flat,
        "smoothed_srsf_norms": smoothed_srsf_norms,
        # Tangent vectors from smoothed data (row-major flat)
        "tangent_vectors_flat": np.array(tangent_vectors).flatten().tolist(),
        "tangent_vector_norms": tv_norms,
        "mean_tangent_norm": float(mean_tv_norm),
    }


def generate_soft_dtw_barycenter_expected():
    """Generate Soft-DTW barycenter reference values using tslearn."""
    from tslearn.barycenters import softdtw_barycenter

    # 3 shifted sinusoids on m=50 grid
    m = 50
    t = np.linspace(0, 1, m)
    shifts = [0.0, 0.05, -0.05]
    curves = np.array(
        [np.sin(2 * np.pi * (t - s)).reshape(-1, 1) for s in shifts]
    )

    bary = softdtw_barycenter(curves, gamma=1.0, max_iter=100, tol=1e-6)
    bary_flat = bary.flatten().tolist()

    return {
        "m": m,
        "shifts": shifts,
        "gamma": 1.0,
        "barycenter": bary_flat,
    }


def generate_landmark_sinusoid_expected():
    """Generate landmark sinusoid alignment reference data."""
    m = 201
    t = np.linspace(0, 1, m)
    shifts = [0.0, 0.03, -0.02, 0.04, -0.03]
    n = len(shifts)

    curves = np.array([np.sin(2 * np.pi * (t - s)) for s in shifts])
    peak_positions = [s + 0.25 for s in shifts]

    # Target: mean of peak positions
    target_peak = float(np.mean(peak_positions))

    return {
        "m": m,
        "n": n,
        "shifts": shifts,
        "peak_positions": peak_positions,
        "target_peak": target_peak,
    }


def main():
    print("Generating Soft-DTW validation data...")
    soft_dtw = generate_soft_dtw_expected()
    print(f"  distance(0,1) = {soft_dtw['distance_01']:.6f}")
    print(f"  divergence(0,1) = {soft_dtw['divergence_01']:.6f}")
    print(f"  single_point = {soft_dtw['single_point_distance']:.6f}")
    print(f"  gamma_sweep = {soft_dtw['gamma_sweep']}")

    print("\nGenerating Landmark Registration validation data...")
    landmark = generate_landmark_expected()
    print(f"  PCHIP monotone: {landmark['pchip_is_monotone']}")
    print(f"  PCHIP2 monotone: {landmark['pchip_is_monotone2']}")
    print(f"  Peak positions: {landmark['peak_positions']}")

    print("\nGenerating TSRVF validation data...")
    tsrvf = generate_tsrvf_expected()
    print(f"  sphere round-trip error = {tsrvf['sphere_round_trip_error']:.2e}")
    print(f"  mean_srsf_norm = {tsrvf['mean_srsf_norm']:.6f}")
    print(f"  mean_tangent_norm = {tsrvf['mean_tangent_norm']:.6f}")
    print(f"  aligned_srsf_norms = {tsrvf['aligned_srsf_norms'][:5]}")
    print(f"  tangent_vector_norms = {tsrvf['tangent_vector_norms'][:5]}")

    print("\nGenerating smoothed TSRVF validation data...")
    tsrvf_smoothed = generate_tsrvf_smoothed_expected()
    print(f"  smoothed mean_srsf_norm = {tsrvf_smoothed['smoothed_mean_srsf_norm']:.6f}")
    print(f"  smoothed mean_tangent_norm = {tsrvf_smoothed['mean_tangent_norm']:.6f}")
    print(f"  smoothed srsf_norms = {tsrvf_smoothed['smoothed_srsf_norms'][:5]}")
    print(f"  smoothed tv_norms = {tsrvf_smoothed['tangent_vector_norms'][:5]}")

    print("\nGenerating Soft-DTW barycenter validation data...")
    soft_dtw_bary = generate_soft_dtw_barycenter_expected()
    bary_mean = np.mean(soft_dtw_bary["barycenter"])
    print(f"  barycenter mean = {bary_mean:.6f}")

    print("\nGenerating Landmark sinusoid alignment data...")
    landmark_sinusoid = generate_landmark_sinusoid_expected()
    print(f"  target peak = {landmark_sinusoid['target_peak']:.6f}")

    # Save combined output
    output = {
        "soft_dtw": soft_dtw,
        "landmark": landmark,
        "tsrvf": tsrvf,
        "tsrvf_smoothed": tsrvf_smoothed,
        "soft_dtw_barycenter": soft_dtw_bary,
        "landmark_sinusoid": landmark_sinusoid,
    }

    out_path = VALIDATION_DIR / "expected" / "new_features_expected.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
