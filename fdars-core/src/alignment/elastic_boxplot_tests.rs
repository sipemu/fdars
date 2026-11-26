//! Tests for the elastic amplitude / phase box plots.
//!
//! Golden values: fdasrsf 2.7.5 `boxplots.ampbox` / `boxplots.phbox`
//! `construct_boxplot(alpha=0.05, k_a=1)` + `classify`, on a mocked fdawarp
//! holding the analytic inputs below (no DP involved). Script:
//! `.planning/quick/261126-3sq-aln-box-55-amplitude-boxplot-boxplot-cla/golden_boxplots.py`.

use super::*;
use std::f64::consts::PI;

const M: usize = 41;
const A: [f64; 15] = [
    1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0, 1.3, 0.7, 1.02, 2.5, 1.08,
];
const B: [f64; 15] = [
    0.5, 0.4, 0.6, 0.55, 0.45, 0.3, 0.7, 0.5, 0.52, 0.8, 0.2, 0.5, 0.48, 0.5, 0.35,
];
const C: [f64; 15] = [
    0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.3, 0.15, -0.05, 0.25, -0.15, 0.85, 0.12, -0.22, 0.02,
];
const E: [f64; 15] = [
    0.05, 0.1, -0.05, 0.02, -0.1, 0.0, 0.05, -0.08, 0.12, 0.03, -0.02, 0.1, -0.04, 0.06, 0.15,
];

fn grid() -> Vec<f64> {
    (0..M).map(|j| j as f64 / (M - 1) as f64).collect()
}

fn fq(a: f64, b: f64, t: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let f = t
        .iter()
        .map(|&x| a * (2.0 * PI * x).sin() + b * x * x)
        .collect();
    let q = t
        .iter()
        .map(|&x| {
            let d = a * 2.0 * PI * (2.0 * PI * x).cos() + 2.0 * b * x;
            d.signum() * d.abs().sqrt()
        })
        .collect();
    (f, q)
}

fn gm(c: f64, e: f64, t: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&x| x + c * x * (1.0 - x) + e * (2.0 * PI * x).sin() / (2.0 * PI))
        .collect()
}

fn mat(rows: &[Vec<f64>]) -> FdMatrix {
    let n = rows.len();
    let m = rows[0].len();
    let mut out = FdMatrix::zeros(n, m);
    for (i, r) in rows.iter().enumerate() {
        for j in 0..m {
            out[(i, j)] = r[j];
        }
    }
    out
}

fn amp_inputs(a: &[f64], b: &[f64]) -> (FdMatrix, FdMatrix) {
    let t = grid();
    let (fs, qs): (Vec<_>, Vec<_>) = a.iter().zip(b).map(|(&a, &b)| fq(a, b, &t)).unzip();
    (mat(&fs), mat(&qs))
}

fn amp_fit() -> AmplitudeBoxplot {
    let t = grid();
    let (f, q) = amp_inputs(&A, &B);
    let (fm, qm) = fq(1.01, 0.49, &t);
    amplitude_boxplot_from_aligned(&f, &q, &fm, &qm, &t, &ElasticBoxplotConfig::default()).unwrap()
}

fn phase_fit() -> ElasticPhaseBoxplot {
    let t = grid();
    let g: Vec<Vec<f64>> = C.iter().zip(&E).map(|(&c, &e)| gm(c, e, &t)).collect();
    elastic_phase_boxplot(&mat(&g), &t, &ElasticBoxplotConfig::default()).unwrap()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
    }
}

fn levels(r: &[BoxplotRegion]) -> Vec<u8> {
    r.iter().map(|x| x.level()).collect()
}

const NEW_A: [f64; 4] = [1.0, 1.25, 3.0, 0.75];
const NEW_C: [f64; 4] = [0.0001, 0.2, 0.95, -0.4];
const NEW_E: [f64; 4] = [0.01, 0.1, 0.0, 0.1];

// ─── Amplitude golden ───────────────────────────────────────────────────────

#[test]
fn amplitude_matches_fdasrsf() {
    let bp = amp_fit();
    let golden = [
        0.01144425346538391,
        0.09871331740610122,
        0.13919301738736647,
        0.18124499487068105,
        0.22271597164505122,
        0.08608765165622943,
        0.16171066481333088,
        0.13591244349466763,
        0.17287010893074586,
        0.18146934187607672,
        0.30001647076606325,
        0.3584642401740125,
        0.01133200156351945,
        1.1573610310017948,
        0.091808442212991,
    ];
    assert_close(&bp.distances, &golden, 1e-12);
    assert!((bp.r50 - 0.16171066481333088).abs() < 1e-12);
    assert!((bp.ralpha - 0.3584642401740125).abs() < 1e-12);
    assert!((bp.whisker_distance - 0.4142984782694006).abs() < 1e-12);
    assert_eq!(
        (bp.q1_index, bp.q3_index, bp.q1a_index, bp.q3a_index),
        (7, 2, 10, 11)
    );
    assert_eq!(bp.outlier_indices, vec![13]);
    assert_eq!((bp.min_index, bp.max_index), (10, 11));
}

#[test]
fn amplitude_classify_matches_fdasrsf() {
    let bp = amp_fit();
    let t = grid();
    let (_, q) = amp_inputs(&NEW_A, &[0.5; 4]);
    let cls = bp.classify_srsfs(&q, &t).unwrap();
    assert_close(
        &cls.distances,
        &[
            0.01144425346538391,
            0.22804499587929294,
            1.459334655917108,
            0.2880384630851068,
        ],
        1e-12,
    );
    assert_eq!(levels(&cls.regions), vec![1, 2, 4, 2]);
    use BoxplotSide::{Q1, Q3};
    assert_eq!(cls.sides, vec![Q3, Q1, Q1, Q3]);
}

// ─── Phase golden ───────────────────────────────────────────────────────────

#[test]
fn phase_matches_fdasrsf() {
    let bp = phase_fit();
    let golden = [
        0.01630788971674848,
        0.07952193304489291,
        0.07551631761752522,
        0.04651098168913535,
        0.04039915490053979,
        0.04058543625006432,
        0.1041221029055542,
        0.04325411640466574,
        0.0489741477578637,
        0.05414883678258828,
        0.06272033832500357,
        0.23714262976197165,
        0.02611860668270282,
        0.08168608216379958,
        0.04954029313638383,
    ];
    assert_close(&bp.distances, &golden, 1e-10);
    assert!((bp.r50 - 0.049540293136383834).abs() < 1e-10);
    assert!((bp.ralpha - 0.1041221029055542).abs() < 1e-10);
    assert!((bp.whisker_distance - 0.14120241192039312).abs() < 1e-10);
    assert_eq!(
        (bp.q1_index, bp.q3_index, bp.q1a_index, bp.q3a_index),
        (7, 8, 2, 6)
    );
    assert_eq!(bp.outlier_indices, vec![11]);
    let med: Vec<f64> = (0..5).map(|k| bp.median[10 * k]).collect();
    assert_close(
        &med,
        &[
            0.0,
            0.26391968258004944,
            0.5153674944714289,
            0.7593271929748349,
            1.0,
        ],
        1e-10,
    );
}

#[test]
fn phase_classify_matches_fdasrsf() {
    let bp = phase_fit();
    let t = grid();
    let g: Vec<Vec<f64>> = NEW_C
        .iter()
        .zip(&NEW_E)
        .map(|(&c, &e)| gm(c, e, &t))
        .collect();
    let cls = bp.classify_warps(&mat(&g), &t).unwrap();
    assert_close(
        &cls.distances,
        &[
            0.0178287088786044,
            0.04820909357972063,
            0.29486481250671465,
            0.13373420111365333,
        ],
        1e-10,
    );
    assert_eq!(levels(&cls.regions), vec![1, 1, 4, 3]);
    use BoxplotSide::{Q1, Q3};
    assert_eq!(cls.sides, vec![Q3, Q1, Q1, Q3]);
}

// ─── Invariants ─────────────────────────────────────────────────────────────

/// Classifying the training curves reproduces the fit: same distances,
/// outliers exactly the fitted outliers, `ceil(n/2)` curves in the 50% region.
fn check_training_consistency(
    dist: &[f64],
    cls: &BoxplotClassification,
    outliers: &[usize],
    r50: f64,
    ralpha: f64,
    whisker: f64,
) {
    assert_close(&cls.distances, dist, 1e-12);
    let n = dist.len();
    for i in 0..n {
        assert_eq!(
            cls.regions[i] == BoxplotRegion::Outlier,
            outliers.contains(&i),
            "curve {i}"
        );
    }
    let n50 = cls
        .regions
        .iter()
        .filter(|r| **r == BoxplotRegion::Central50)
        .count();
    assert_eq!(n50, n.div_ceil(2));
    assert!(r50 <= ralpha, "central regions must be nested");
    assert!(r50 <= whisker);
}

#[test]
fn amplitude_training_consistency_and_nesting() {
    let bp = amp_fit();
    let (_, q) = amp_inputs(&A, &B);
    let cls = bp.classify_srsfs(&q, &grid()).unwrap();
    check_training_consistency(
        &bp.distances,
        &cls,
        &bp.outlier_indices,
        bp.r50,
        bp.ralpha,
        bp.whisker_distance,
    );
    // quartile distances bounded by r50, quantiles by ralpha
    assert!(bp.distances[bp.q1_index] <= bp.r50 && bp.distances[bp.q3_index] <= bp.r50);
    assert!(bp.distances[bp.q1a_index] <= bp.ralpha && bp.distances[bp.q3a_index] <= bp.ralpha);
    // Q1 and Q3 point in opposite directions from the median
    assert_eq!(cls.sides[bp.q3_index], BoxplotSide::Q3);
    assert_eq!(cls.sides[bp.q1_index], BoxplotSide::Q1);
}

#[test]
fn phase_training_consistency_and_nesting() {
    let bp = phase_fit();
    let t = grid();
    let g: Vec<Vec<f64>> = C.iter().zip(&E).map(|(&c, &e)| gm(c, e, &t)).collect();
    let cls = bp.classify_warps(&mat(&g), &t).unwrap();
    check_training_consistency(
        &bp.distances,
        &cls,
        &bp.outlier_indices,
        bp.r50,
        bp.ralpha,
        bp.whisker_distance,
    );
}

#[test]
fn median_curve_classifies_as_deepest() {
    // The median itself is at distance 0: deepest possible, inside the 50% region.
    let bp = amp_fit();
    let t = grid();
    let q = FdMatrix::from_slice(&bp.median_srsf, 1, M).unwrap();
    let cls = bp.classify_srsfs(&q, &t).unwrap();
    assert!(cls.distances[0].abs() < 1e-15);
    assert_eq!(cls.regions[0], BoxplotRegion::Central50);

    let pb = phase_fit();
    let g = FdMatrix::from_slice(&pb.median, 1, M).unwrap();
    let cls = pb.classify_warps(&g, &t).unwrap();
    assert!(
        cls.distances[0] < 1e-2,
        "median warp distance {}",
        cls.distances[0]
    );
    assert!(cls.distances[0] <= pb.distances.iter().copied().fold(f64::INFINITY, f64::min) + 1e-2);
    assert_eq!(cls.regions[0], BoxplotRegion::Central50);
}

#[test]
fn phase_median_is_weiszfeld_stationary() {
    // At the Karcher median the weighted mean of unit shooting vectors vanishes.
    let pb = phase_fit();
    let t = unit_grid(M);
    let mut s = vec![0.0; M];
    for i in 0..15 {
        let v = pb.shooting_vectors.row(i);
        let d = pb.distances[i];
        for j in 0..M {
            s[j] += v[j] / d;
        }
    }
    assert!(
        l2_norm(&s, &t) < 1e-5,
        "Weiszfeld residual {}",
        l2_norm(&s, &t)
    );
}

#[test]
fn whisker_scale_monotone() {
    let t = grid();
    let (f, q) = amp_inputs(&A, &B);
    let (fm, qm) = fq(1.01, 0.49, &t);
    let mut prev = 0.0;
    for k in [0.5, 1.0, 1.5, 3.0] {
        let cfg = ElasticBoxplotConfig {
            k_a: k,
            ..ElasticBoxplotConfig::default()
        };
        let bp = amplitude_boxplot_from_aligned(&f, &q, &fm, &qm, &t, &cfg).unwrap();
        assert!(bp.whisker_distance >= prev);
        prev = bp.whisker_distance;
    }
}

#[test]
fn invalid_inputs_rejected() {
    let t = grid();
    let (f, q) = amp_inputs(&A[..2], &B[..2]);
    let (fm, qm) = fq(1.0, 0.5, &t);
    let cfg = ElasticBoxplotConfig::default();
    assert!(amplitude_boxplot_from_aligned(&f, &q, &fm, &qm, &t, &cfg).is_err());
    let (f, q) = amp_inputs(&A, &B);
    let mut bad = cfg.clone();
    bad.alpha = 1.0;
    assert!(amplitude_boxplot_from_aligned(&f, &q, &fm, &qm, &t, &bad).is_err());
    bad = cfg.clone();
    bad.k_a = 0.0;
    assert!(amplitude_boxplot_from_aligned(&f, &q, &fm, &qm, &t, &bad).is_err());
    assert!(amplitude_boxplot_from_aligned(&f, &q, &fm[..10], &qm, &t, &cfg).is_err());
    let bp = amp_fit();
    assert!(bp.classify_srsfs(&FdMatrix::zeros(2, 10), &t).is_err());
    assert!(elastic_phase_boxplot(&FdMatrix::zeros(2, M), &t, &cfg).is_err());
}

#[test]
fn elastic_boxplots_end_to_end() {
    // Amplitude- and phase-varying sines plus one amplitude outlier.
    let t = grid();
    let n = 12;
    let mut rows = Vec::new();
    for i in 0..n {
        let a = if i == n - 1 {
            4.0
        } else {
            1.0 + 0.04 * (i as f64 - 5.0)
        };
        let c = 0.3 * ((i * 7 % 11) as f64 / 10.0 - 0.5);
        let w = gm(c, 0.0, &t);
        rows.push(
            w.iter()
                .map(|&x| a * (2.0 * PI * x).sin())
                .collect::<Vec<f64>>(),
        );
    }
    let data = mat(&rows);
    let bp = elastic_boxplots(&data, &t, 0.0, &ElasticBoxplotConfig::default()).unwrap();
    assert!(bp.amplitude.outlier_indices.contains(&(n - 1)));
    assert!(bp.amplitude.r50 <= bp.amplitude.ralpha);
    assert!(bp.phase.r50 <= bp.phase.ralpha);
    // Training curves aligned by the fit classify consistently with the fit.
    let cls = bp.amplitude.classify_srsfs(&bp.aligned_srsfs, &t).unwrap();
    check_training_consistency(
        &bp.amplitude.distances,
        &cls,
        &bp.amplitude.outlier_indices,
        bp.amplitude.r50,
        bp.amplitude.ralpha,
        bp.amplitude.whisker_distance,
    );
    // New functions: the template itself is central; a 5x scaled curve is an outlier.
    let template = bp.amplitude.median.clone();
    let big: Vec<f64> = template.iter().map(|x| 5.0 * x).collect();
    let new = mat(&[template, big]);
    let out = bp.classify(&new, &t).unwrap();
    assert_eq!(out.amplitude.regions[0], BoxplotRegion::Central50);
    assert_eq!(out.amplitude.regions[1], BoxplotRegion::Outlier);
    assert_eq!(out.gammas.shape(), (2, M));
    assert_eq!(out.phase.regions.len(), 2);
}
