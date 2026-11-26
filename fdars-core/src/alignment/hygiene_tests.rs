//! Bit-identity and correctness tests for the ALN-HYGIENE changes (GH #56):
//! closed-curve SRSF/weight hoisting, outlier threshold options, SRSF
//! pre-smoothing, and the parallelised elastic-FPCA loops.

#![allow(clippy::field_reassign_with_default)]

use crate::alignment::{
    elastic_align_pair_closed, elastic_outlier_detection, karcher_mean, karcher_mean_closed,
    ElasticOutlierConfig,
};
use crate::elastic_fpca::{horiz_fpca, joint_fpca, vert_fpca};
use crate::matrix::FdMatrix;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn fnv(h: &mut u64, xs: &[f64]) {
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// Full-rank, non-degenerate probe data: two bumps at curve-specific centres
/// plus a curve-specific low-frequency sinusoid.
fn probe_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let c = 0.3 + 0.05 * i as f64;
        for j in 0..m {
            let s = t[j];
            d[(i, j)] = (-(s - c).powi(2) / 0.01).exp()
                + 0.5 * (-(s - c - 0.3).powi(2) / 0.02).exp()
                + 0.1 * (3.0 * s + i as f64).sin();
        }
    }
    (d, t)
}

/// Periodic probe curves (closed-curve tests).
fn periodic_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let ph = 0.07 * i as f64;
        for j in 0..m {
            let s = t[j] + ph;
            d[(i, j)] = (tau * s).sin() + 0.4 * (2.0 * tau * s).cos() + 0.05 * i as f64;
        }
    }
    (d, t)
}

// ─── 1. Closed-curve rotation search: hoisting is bit-identical ─────────────

/// FNV-1a hashes recorded on the pre-change code (commit ce810d5f) with the
/// exact same probes.
#[test]
fn closed_curve_hoist_is_bit_identical_to_pre_change() {
    let (d, t) = periodic_data(5, 48);
    let expected: [(f64, u64, u64); 2] = [
        (0.0, 0xe28e_d3b6_1a64_dd70, 0x334d_75f8_81d2_0205),
        (0.2, 0x1956_4737_9dd0_d6cb, 0x4fcf_4d97_18e9_19a7),
    ];
    for (lam, h_pair, h_k) in expected {
        let r = elastic_align_pair_closed(&d.row(0), &d.row(3), &t, lam).unwrap();
        let mut h = FNV0;
        fnv(&mut h, &r.gamma);
        fnv(&mut h, &r.f_aligned);
        fnv(&mut h, &[r.distance, r.optimal_rotation as f64]);
        assert_eq!(h, h_pair, "closed pair lam={lam}: got {h:#x}");

        let k = karcher_mean_closed(&d, &t, 4, 1e-6, lam).unwrap();
        let mut h = FNV0;
        fnv(&mut h, &k.mean);
        fnv(&mut h, &k.mean_srsf);
        fnv(&mut h, k.gammas.as_slice());
        fnv(&mut h, k.aligned_data.as_slice());
        let rots: Vec<f64> = k.rotations.iter().map(|&r| r as f64).collect();
        fnv(&mut h, &rots);
        fnv(&mut h, &[k.n_iter as f64, f64::from(u8::from(k.converged))]);
        assert_eq!(h, h_k, "closed karcher lam={lam}: got {h:#x}");
    }
}

// ─── 2. Outlier detection: default threshold unchanged ──────────────────────

fn outlier_probe() -> (FdMatrix, Vec<f64>) {
    let (mut d, t) = probe_data(9, 40);
    // Curve 8 is a gross amplitude outlier.
    for j in 0..40 {
        d[(8, j)] = 6.0 * (12.0 * t[j]).cos();
    }
    (d, t)
}

fn hash_outlier(r: &crate::alignment::ElasticOutlierResult) -> u64 {
    let mut h = FNV0;
    let idx: Vec<f64> = r.outlier_indices.iter().map(|&i| i as f64).collect();
    fnv(&mut h, &idx);
    fnv(&mut h, &r.distances);
    fnv(&mut h, &[r.threshold]);
    fnv(&mut h, &r.amplitude_distances);
    fnv(&mut h, &r.phase_distances);
    h
}

#[test]
fn outlier_default_is_bit_identical_to_pre_change() {
    let (d, t) = outlier_probe();
    let expected: [(bool, u64); 2] = [
        (true, 0x0495_1ccf_b5c3_d8d6),
        (false, 0xf090_a3df_024f_7ab5),
    ];
    for (use_median, hv) in expected {
        let mut cfg = ElasticOutlierConfig::default();
        cfg.use_median = use_median;
        let r = elastic_outlier_detection(&d, &t, &cfg).unwrap();
        let h = hash_outlier(&r);
        assert_eq!(h, hv, "outlier use_median={use_median}: got {h:#x}");
    }
}

// ─── 4. Elastic FPCA: parallel loops are bit-identical ──────────────────────

#[test]
fn elastic_fpca_is_bit_identical_to_pre_change() {
    let (d, t) = probe_data(12, 41);
    let km = karcher_mean(&d, &t, 8, 1e-4, 0.0);

    let v = vert_fpca(&km, &t, 3).unwrap();
    let mut h = FNV0;
    fnv(&mut h, v.scores.as_slice());
    fnv(&mut h, v.eigenfunctions_q.as_slice());
    fnv(&mut h, v.eigenfunctions_f.as_slice());
    fnv(&mut h, &v.eigenvalues);
    fnv(&mut h, &v.mean_q);
    assert_eq!(h, 0x66f6_1c11_5b81_cb13, "vert_fpca: got {h:#x}");

    let hz = horiz_fpca(&km, &t, 3).unwrap();
    let mut h = FNV0;
    fnv(&mut h, hz.scores.as_slice());
    fnv(&mut h, hz.eigenfunctions_psi.as_slice());
    fnv(&mut h, hz.eigenfunctions_gam.as_slice());
    fnv(&mut h, &hz.eigenvalues);
    fnv(&mut h, &hz.mean_psi);
    fnv(&mut h, hz.shooting_vectors.as_slice());
    assert_eq!(h, 0x9407_b602_44f4_ac0f, "horiz_fpca: got {h:#x}");

    for (bc, hj) in [
        (None, 0xbadb_3986_99a8_94e0_u64),
        (Some(0.7), 0xa4b6_7860_272c_5e74),
    ] {
        let j = joint_fpca(&km, &t, 3, bc).unwrap();
        let mut h = FNV0;
        fnv(&mut h, j.scores.as_slice());
        fnv(&mut h, &j.eigenvalues);
        fnv(&mut h, &[j.balance_c]);
        fnv(&mut h, j.vert_component.as_slice());
        fnv(&mut h, j.horiz_component.as_slice());
        assert_eq!(h, hj, "joint_fpca bc={bc:?}: got {h:#x}");
    }
}
