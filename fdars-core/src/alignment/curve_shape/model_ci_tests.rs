//! Tests for the fdasrvf-compatible model-based shape CI (GH #71).
#![allow(clippy::field_reassign_with_default, clippy::excessive_precision)]

use super::geometry::{self, Grid};
use super::karcher::log_map;
use super::*;
use std::f64::consts::PI;

fn linspace(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Open planar curve `(t, a sin(πt) + c sin(2π t^p))`.
fn bump(m: usize, a: f64, c: f64, p: f64) -> FdMatrix {
    let t = linspace(m);
    let mut b = FdMatrix::zeros(2, m);
    for (j, &tj) in t.iter().enumerate() {
        b[(0, j)] = tj;
        b[(1, j)] = a * (PI * tj).sin() + c * (2.0 * PI * tj.powf(p)).sin();
    }
    b
}

fn golden_curves(m: usize) -> Vec<FdMatrix> {
    let a = [0.20, 0.25, 0.30, 0.35, 0.40, 0.60];
    let c = [0.05, -0.03, 0.02, 0.04, -0.05, 0.30];
    let p = [1.0, 1.2, 0.9, 1.1, 1.0, 1.5];
    (0..6).map(|i| bump(m, a[i], c[i], p[i])).collect()
}

fn set(curves: &[FdMatrix]) -> FdCurveSet {
    curves_to_set(curves).unwrap()
}

fn model_cfg(n_samples: usize, seed: u64) -> CurveShapeCiConfig {
    let mut ci = CurveShapeCiConfig::default();
    ci.method = CurveShapeCiMethod::ModelBased;
    ci.n_samples = n_samples;
    ci.seed = seed;
    ci
}

/// A two-parameter family of open planar curves (full-rank tangent variation).
fn family(m: usize, n: usize) -> Vec<FdMatrix> {
    (0..n)
        .map(|i| {
            let u = (i as f64 * 0.7).sin();
            let w = (i as f64 * 1.9 + 0.3).cos();
            bump(m, 0.3 + 0.08 * u, 0.06 * w, 1.0 + 0.1 * u * w)
        })
        .collect()
}

fn rigid_cfg() -> CurveShapeConfig {
    let mut c = CurveShapeConfig::default();
    c.rotation = false;
    c.reparam = false;
    c
}

// ─── Default / API ──────────────────────────────────────────────────────────

#[test]
fn default_method_is_the_unchanged_bootstrap() {
    let m = 25;
    let curves = set(&family(m, 6));
    let shape = CurveShapeConfig::default();
    let mut ci = CurveShapeCiConfig::default();
    assert_eq!(ci.method, CurveShapeCiMethod::Bootstrap);
    ci.n_bootstrap = 6;
    let base = curve_shape_confidence_interval(&curves, &shape, &ci).unwrap();
    assert!(base.model.is_none());
    // model-only knobs do not touch the bootstrap path
    let mut ci2 = ci.clone();
    ci2.n_components = 2;
    ci2.n_samples = 7;
    ci2.n_warp_components = 1;
    let again = curve_shape_confidence_interval(&curves, &shape, &ci2).unwrap();
    assert_eq!(base, again);
    // and it is the Karcher mean, not the median
    let km = curve_shape_karcher_mean(&curves, &shape, ci.max_iter, ci.tol).unwrap();
    assert_eq!(base.mean, km);
}

#[test]
fn model_based_rejects_unsupported_settings() {
    let m = 20;
    let curves = set(&family(m, 5));
    let shape = CurveShapeConfig::default();
    let run = |shape: &CurveShapeConfig, ci: &CurveShapeCiConfig| {
        curve_shape_confidence_interval(&curves, shape, ci)
    };
    let mut ci = model_cfg(10, 1);
    ci.n_samples = 2;
    assert!(run(&shape, &ci).is_err());
    let mut ci = model_cfg(10, 1);
    ci.n_components = 0;
    assert!(run(&shape, &ci).is_err());
    let mut ci = model_cfg(10, 1);
    ci.n_warp_components = 0;
    assert!(run(&shape, &ci).is_err());
    let mut flat = shape.clone();
    flat.scale = false; // fdasrvf sample_shapes: "Not implemented for scale = FALSE"
    assert!(run(&flat, &model_cfg(10, 1)).is_err());
    // rotation sampling is planar only (fdasrvf aborts for n != 2)
    let curves3: Vec<FdMatrix> = family(m, 5)
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut c = FdMatrix::zeros(3, m);
            for j in 0..m {
                c[(0, j)] = b[(0, j)];
                c[(1, j)] = b[(1, j)];
                c[(2, j)] = 0.1 * (i as f64 + 1.0) * b[(0, j)] * b[(0, j)];
            }
            c
        })
        .collect();
    let s3 = set(&curves3);
    assert!(curve_shape_confidence_interval(&s3, &shape, &model_cfg(10, 1)).is_err());
    let mut norot = shape.clone();
    norot.rotation = false;
    let ok = curve_shape_confidence_interval(&s3, &norot, &model_cfg(5, 1)).unwrap();
    assert_eq!(ok.model.unwrap().q1a.shape(), (3, m));
}

// ─── Structure, determinism ─────────────────────────────────────────────────

#[test]
fn model_based_structure_and_determinism() {
    let m = 30;
    let curves = set(&family(m, 8));
    let shape = CurveShapeConfig::default(); // rotation + warps sampled
    let ci = model_cfg(12, 3);
    let res = curve_shape_confidence_interval(&curves, &shape, &ci).unwrap();
    let med = curve_shape_karcher_median(&curves, &shape, ci.max_iter, ci.tol).unwrap();
    assert_eq!(res.mean, med, "centre is the Karcher median");
    let model = res.model.as_ref().unwrap();
    let grid = Grid::new(m);
    assert_eq!(model.samples.ncurves(), 12);
    assert_eq!(model.shape_samples.ncurves(), 12);
    assert_eq!(model.scores.shape(), (12, 5));
    assert_eq!(model.warps.shape(), (12, m));
    assert_eq!(res.bootstrap_means.len(), 12);
    assert_eq!(model.q1a, model.boxplot.q1a);
    assert_eq!(
        model.q3a,
        curve_from_set(&model.samples, model.boxplot.q3a_index).unwrap()
    );
    // box plot alpha = 1 − a: (1 − alpha) region of round(0.95·12) = 11 samples
    assert!(model.boxplot.r50 <= model.boxplot.ralpha);
    for i in 0..12 {
        // warps are diffeomorphisms of [0, 1]
        assert_eq!(model.warps[(i, 0)], 0.0);
        assert_eq!(model.warps[(i, m - 1)], 1.0);
        for j in 1..m {
            assert!(model.warps[(i, j)] >= model.warps[(i, j - 1)]);
        }
        // rotations are proper
        let r = &model.rotations[i];
        let det = r[(0, 0)] * r[(1, 1)] - r[(0, 1)] * r[(1, 0)];
        assert!((det - 1.0).abs() < 1e-12);
        // samples are centred, unit-length shapes
        let b = curve_from_set(&model.samples, i).unwrap();
        assert!((geometry::curve_length(&b, &grid) - 1.0).abs() < 1e-9);
        assert!(geometry::centroid(&b, &grid).iter().all(|c| c.abs() < 1e-9));
    }
    // fixed seed → identical; other seed → different samples
    let again = curve_shape_confidence_interval(&curves, &shape, &ci).unwrap();
    assert_eq!(res, again);
    let other = curve_shape_confidence_interval(&curves, &shape, &model_cfg(12, 4)).unwrap();
    assert_ne!(res.model.unwrap().scores, other.model.unwrap().scores);
}

#[cfg(feature = "parallel")]
#[test]
fn model_based_is_thread_count_independent() {
    let m = 20;
    let curves = set(&family(m, 6));
    let shape = CurveShapeConfig::default();
    let ci = model_cfg(9, 11);
    let par = curve_shape_confidence_interval(&curves, &shape, &ci).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let seq = pool.install(|| curve_shape_confidence_interval(&curves, &shape, &ci).unwrap());
    assert_eq!(par, seq);
}

#[test]
fn posed_samples_have_the_sampled_shapes() {
    // the random warp and rotation only change the pose: aligning a posed
    // sample back recovers its unposed shape
    let m = 40;
    let curves = set(&family(m, 8));
    let shape = CurveShapeConfig::default();
    let res = curve_shape_confidence_interval(&curves, &shape, &model_cfg(4, 5)).unwrap();
    let model = res.model.unwrap();
    for i in 0..4 {
        let base = curve_from_set(&model.shape_samples, i).unwrap();
        let posed = curve_from_set(&model.samples, i).unwrap();
        let d = curve_shape_distance(&base, &posed, &shape).unwrap();
        let spread = res.bootstrap_distances[i];
        assert!(
            d < 0.05,
            "sample {i}: shape changed by posing ({d}, spread {spread})"
        );
    }
}

// ─── Distributional checks ──────────────────────────────────────────────────

#[test]
fn sampled_tangent_covariance_matches_the_model() {
    let m = 30;
    let shape = rigid_cfg(); // no warps / rotations: isolate the shape model
    let curves = set(&family(m, 10));
    let n_s = 2000;
    let mut ci = model_cfg(n_s, 2024);
    ci.n_components = 3;
    let res = curve_shape_confidence_interval(&curves, &shape, &ci).unwrap();
    let model = res.model.as_ref().unwrap();
    let med = &res.mean;
    let grid = Grid::new(m);
    let comps = &model.pca.components;
    let lams = &model.pca.eigenvalues;
    assert_eq!(lams.len(), 3);
    // projections of the log maps of the sampled SRVFs onto the components
    let mut proj = vec![vec![0.0; n_s]; 3];
    for i in 0..n_s {
        let q = curve_from_set(&model.shape_samples_srvf, i).unwrap();
        let v = log_map(&med.mean_srvf, &q, &med.config, &grid);
        let mut resid = v.clone();
        for k in 0..3 {
            let s = geometry::inner(&v, &comps[k], &grid);
            proj[k][i] = s;
            resid = geometry::axpby(1.0, &resid, -s, &comps[k]);
            // log ∘ exp round trip through the curve recovers the drawn score
            let drawn = model.scores[(i, k)];
            assert!(
                (s - drawn).abs() < 2e-3 + 2e-2 * drawn.abs(),
                "sample {i} comp {k}: {s} vs {drawn}"
            );
        }
        // the sampled tangent vector lies in the span of the model components
        assert!(geometry::norm(&resid, &grid) < 0.05 * geometry::norm(&v, &grid) + 1e-3);
    }
    let nf = n_s as f64;
    for k in 0..3 {
        let mean = proj[k].iter().sum::<f64>() / nf;
        let var = proj[k].iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (nf - 1.0);
        // sampling SE of a variance ≈ λ√(2/N) (3.2% at N = 2000); allow 4 SE
        // plus the curve round-trip bias.
        let rel = (var / lams[k] - 1.0).abs();
        assert!(
            rel < 4.0 * (2.0 / nf).sqrt() + 0.02,
            "comp {k}: var {var} vs λ {}",
            lams[k]
        );
        assert!(
            mean.abs() < 4.0 * (lams[k] / nf).sqrt() + 1e-4,
            "comp {k}: mean {mean}"
        );
        for l in 0..k {
            let cov = proj[k]
                .iter()
                .zip(&proj[l])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / nf;
            let corr = cov / (lams[k] * lams[l]).sqrt();
            assert!(
                corr.abs() < 4.0 / nf.sqrt() + 0.02,
                "corr({k},{l}) = {corr}"
            );
        }
    }
    // geodesic distance of a sample to the median = ‖v‖ (single exact step)
    for i in 0..50 {
        let nv = (0..3)
            .map(|k| model.scores[(i, k)].powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((res.bootstrap_distances[i] - nv).abs() < 2e-3 + 2e-2 * nv);
    }
}

#[test]
fn closed_model_samples_are_closed_and_near_the_model() {
    let m = 41;
    let blob = |e: f64, f: f64| {
        let th: Vec<f64> = linspace(m).iter().map(|&t| 2.0 * PI * t).collect();
        let mut b = FdMatrix::zeros(2, m);
        for (j, &a) in th.iter().enumerate() {
            let r = 1.0 + e * (3.0 * a).cos() + f * (2.0 * a).sin();
            b[(0, j)] = r * a.cos();
            b[(1, j)] = r * a.sin();
        }
        b
    };
    let curves: Vec<FdMatrix> = (0..7)
        .map(|i| blob(0.1 + 0.03 * (i as f64).sin(), 0.05 * (i as f64 * 1.3).cos()))
        .collect();
    let mut shape = rigid_cfg();
    shape.closed = true;
    let mut ci = model_cfg(30, 9);
    ci.n_components = 2;
    let res = curve_shape_confidence_interval(&set(&curves), &shape, &ci).unwrap();
    let model = res.model.unwrap();
    let grid = Grid::new(m);
    let mu = &res.mean.mean_srvf;
    let rt_q = prepare(&transform_to_curve(mu, &shape, &grid), &shape, &grid)
        .unwrap()
        .q;
    let rt = curve_shape_distance(
        &srvf_to_curve(mu).unwrap(),
        &srvf_to_curve(&rt_q).unwrap(),
        &rigid_cfg(),
    )
    .unwrap();
    assert!(rt < 0.03, "round-trip error {rt}");
    for i in 0..30 {
        let b = curve_from_set(&model.shape_samples, i).unwrap();
        let gap =
            ((b[(0, m - 1)] - b[(0, 0)]).powi(2) + (b[(1, m - 1)] - b[(1, 0)]).powi(2)).sqrt();
        assert!(gap < 1e-6, "sample {i} not closed: gap {gap}");
        // stepped geodesic shooting lands at distance ‖v‖ from the median; the
        // reported distance uses the SRVF recomputed from the sampled curve
        // (fdasrvf `qns`), which moves it by at most the SRVF → curve → SRVF
        // round-trip error (triangle inequality; measured on the median, with
        // a factor 2 for the nearby sampled shapes)
        let nv = (0..2)
            .map(|k| model.scores[(i, k)].powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (res.bootstrap_distances[i] - nv).abs() < 2.0 * rt + 0.02 * nv + 2e-3,
            "sample {i}: d = {} vs ‖v‖ = {nv}",
            res.bootstrap_distances[i]
        );
    }
}

// ─── fdasrvf golden ─────────────────────────────────────────────────────────

/// fdasrvf 2.6.0 `shape_CI(beta, a = 0.95, no = 5, Nsamp = 100, mode = "O",
/// rotated = FALSE, scale = TRUE)` on `golden_curves` (script
/// `.planning/quick/261126-mbw-*/golden_shape_ci.R`, output `.out`):
/// `out.med$betamean` (centred) at T = 60.
const R_MEDIAN_FULL_X: [f64; 60] = [
    -4.061988989e-01,
    -3.921987300e-01,
    -3.782002796e-01,
    -3.642178381e-01,
    -3.502660455e-01,
    -3.363197499e-01,
    -3.223533897e-01,
    -3.083619253e-01,
    -2.943404700e-01,
    -2.803134023e-01,
    -2.663108514e-01,
    -2.523132936e-01,
    -2.383014239e-01,
    -2.242999578e-01,
    -2.103141985e-01,
    -1.963307074e-01,
    -1.823919025e-01,
    -1.684625567e-01,
    -1.544222267e-01,
    -1.403038893e-01,
    -1.261468811e-01,
    -1.119216168e-01,
    -9.767914651e-02,
    -8.350784268e-02,
    -6.947065803e-02,
    -5.553779303e-02,
    -4.159935634e-02,
    -2.759683336e-02,
    -1.356597017e-02,
    4.591256147e-04,
    1.438301812e-02,
    2.811286482e-02,
    4.175207391e-02,
    5.539801916e-02,
    6.904811069e-02,
    8.255098673e-02,
    9.574605662e-02,
    1.087832824e-01,
    1.217963562e-01,
    1.347611547e-01,
    1.477026816e-01,
    1.605679267e-01,
    1.732416893e-01,
    1.857652398e-01,
    1.982536784e-01,
    2.107770942e-01,
    2.233540039e-01,
    2.359362814e-01,
    2.484834817e-01,
    2.609880011e-01,
    2.734684161e-01,
    2.859297257e-01,
    2.983500568e-01,
    3.107190267e-01,
    3.230919656e-01,
    3.355430204e-01,
    3.480423935e-01,
    3.605911248e-01,
    3.734029045e-01,
    3.862480741e-01,
];
const R_MEDIAN_FULL_Y: [f64; 60] = [
    -1.623166532e-01,
    -1.465031842e-01,
    -1.306927916e-01,
    -1.149645528e-01,
    -9.941138484e-02,
    -8.405960093e-02,
    -6.893078869e-02,
    -5.407932736e-02,
    -3.955946272e-02,
    -2.545143323e-02,
    -1.184125169e-02,
    1.239219731e-03,
    1.376089530e-02,
    2.565489819e-02,
    3.687579780e-02,
    4.739414405e-02,
    5.713437396e-02,
    6.607243552e-02,
    7.424036030e-02,
    8.158682702e-02,
    8.805877489e-02,
    9.363199326e-02,
    9.825767668e-02,
    1.018995003e-01,
    1.045543151e-01,
    1.062504397e-01,
    1.070146429e-01,
    1.068505514e-01,
    1.057622906e-01,
    1.037672023e-01,
    1.009043330e-01,
    9.723942194e-02,
    9.280217739e-02,
    8.760358805e-02,
    8.167839225e-02,
    7.512623368e-02,
    6.807820699e-02,
    6.053389551e-02,
    5.247918102e-02,
    4.397383254e-02,
    3.504645339e-02,
    2.577276546e-02,
    1.627806267e-02,
    6.585891294e-03,
    -3.342774380e-03,
    -1.352419814e-02,
    -2.393602114e-02,
    -3.450884562e-02,
    -4.517716707e-02,
    -5.590002532e-02,
    -6.667035045e-02,
    -7.747851765e-02,
    -8.828753363e-02,
    -9.905826217e-02,
    -1.098163868e-01,
    -1.205942265e-01,
    -1.313165181e-01,
    -1.419429140e-01,
    -1.524580956e-01,
    -1.622865064e-01,
];
/// Eigenvalues of `multivariate_karcher_cov(out.med)` in the L² scale (`svd(K)$d / T`).
const R_EIG_60: [f64; 5] = [
    6.4446808520e-02,
    1.1313617406e-02,
    2.4525027419e-03,
    3.1353537764e-04,
    2.6160805846e-05,
];
const R_EIG_240: [f64; 5] = [
    6.3010628305e-02,
    1.0886600749e-02,
    2.0767062649e-03,
    3.0226178693e-04,
    2.1239307695e-05,
];
/// Moments of the geodesic distances of 2000 model samples to the median (script
/// `golden_shape_ci_moments.R`, `set.seed(1)`; RNG-dependent — R's Mersenne-Twister
/// / inversion normals vs `rand` `StdRng` here, so only the distributions agree):
/// unposed samples (`qns`) and posed samples (`qs`, the box-plot distances), and
/// the sum of the 5 model eigenvalues.
const R_UNPOSED_MEAN_SD: (f64, f64) = (0.240146, 0.139115);
const R_POSED_MEAN_SD: (f64, f64) = (0.297443, 0.126026);
const R_EIG_SUM_60: f64 = 0.078553;

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    (mean, var.sqrt())
}

#[test]
fn golden_median_shape_and_covariance_fdasrvf() {
    let m = 60;
    let mut shape = CurveShapeConfig::default();
    shape.rotation = false;
    let curves = golden_curves(m);
    let n_s = 2000;
    let res = curve_shape_confidence_interval(&set(&curves), &shape, &model_cfg(n_s, 1)).unwrap();
    let med = &res.mean;
    let model = res.model.as_ref().unwrap();
    // fdasrvf shape_CI output structure: Q1a is d × T, Nsamp samples
    assert_eq!(model.q1a.shape(), (2, m));
    assert_eq!(model.samples.ncurves(), n_s);
    let mut r_med = FdMatrix::zeros(2, m);
    for j in 0..m {
        r_med[(0, j)] = R_MEDIAN_FULL_X[j];
        r_med[(1, j)] = R_MEDIAN_FULL_Y[j];
    }
    // Median shape. fdasrvf's median is an approximate Weiszfeld minimiser of a
    // nearly flat objective (its distances to the data move by up to 0.09
    // between T = 60 and T = 240, where it collapses onto curve 4), so the
    // parametrisations differ and only shapes are comparable: the two medians
    // are within 10% of the mean spread (observed 0.019 vs 0.198), and ours
    // attains the smaller Σ dᵢ under our metric.
    let d_med = curve_shape_distance(&med.mean, &r_med, &shape).unwrap();
    let spread = med.distances.iter().sum::<f64>() / 6.0;
    assert!(
        d_med < 0.1 * spread,
        "medians {d_med} apart (spread {spread})"
    );
    let r_sum: f64 = curves
        .iter()
        .map(|b| curve_shape_distance(&r_med, b, &shape).unwrap())
        .sum();
    let ours: f64 = med.distances.iter().sum();
    assert!(ours <= r_sum, "Σd: fdars {ours} vs fdasrvf median {r_sum}");
    // Tangent covariance eigenvalues (shape PCA at the median, L² scale) agree
    // within twice fdasrvf's own T = 60 → 240 drift + 2% (observed relative
    // errors 4.8%, 5.8%, 21%, 2.4% vs drifts 2.2%, 3.8%, 15%, 3.6%).
    let eig = &model.pca.eigenvalues;
    for k in 0..4 {
        let rel = (eig[k] / R_EIG_60[k] - 1.0).abs();
        let drift = (R_EIG_240[k] / R_EIG_60[k] - 1.0).abs();
        assert!(
            rel < 2.0 * drift + 0.02,
            "eigenvalue {k}: {} vs {}",
            eig[k],
            R_EIG_60[k]
        );
    }
    // Sample distributions (RNG streams differ): after rescaling fdasrvf's
    // distance moments by √(Σλ_fdars / Σλ_fdasrvf) for the slightly different
    // model variance, the means and sds agree within 4 combined standard
    // errors (observed ≤ 1.7 SE).
    let scale = (eig.iter().sum::<f64>() / R_EIG_SUM_60).sqrt();
    for (ours, (r_mean, r_sd), what) in [
        (
            mean_sd(&res.bootstrap_distances),
            R_UNPOSED_MEAN_SD,
            "unposed",
        ),
        (mean_sd(&model.boxplot.distances), R_POSED_MEAN_SD, "posed"),
    ] {
        let se = (ours.1.powi(2) + r_sd.powi(2)).sqrt() / (n_s as f64).sqrt();
        assert!(
            (ours.0 - scale * r_mean).abs() < 4.0 * se,
            "{what} mean {} vs fdasrvf {r_mean} (scaled {})",
            ours.0,
            scale * r_mean
        );
        // SE of a sd ≈ sd/√(2N)
        let se_sd = (ours.1.powi(2) + r_sd.powi(2)).sqrt() / (2.0 * n_s as f64).sqrt();
        assert!(
            (ours.1 - scale * r_sd).abs() < 4.0 * se_sd,
            "{what} sd {} vs fdasrvf {r_sd}",
            ours.1
        );
    }
}

#[test]
fn von_mises_sampler_matches_its_moments() {
    // density ∝ exp(κ cos(φ − μ)), κ = 1/σ²: E cos(φ − μ) = I₁(κ)/I₀(κ),
    // E sin(φ − μ) = 0. For κ = 25 the asymptotic series gives
    // 1 − 1/(2κ) − 1/(8κ²) − 1/(8κ³) = 0.97979.
    let (mu, sigma) = (0.5, 0.2);
    let mut rng = crate::helpers::seed_for_thread(3, 0);
    let n = 20000;
    let draws: Vec<f64> = (0..n)
        .map(|_| super::model_ci::sample_von_mises(mu, sigma, &mut rng))
        .collect();
    assert!(draws.iter().all(|&p| (0.0..2.0 * PI).contains(&p)));
    let c = draws.iter().map(|p| (p - mu).cos()).sum::<f64>() / n as f64;
    let s = draws.iter().map(|p| (p - mu).sin()).sum::<f64>() / n as f64;
    let k: f64 = 25.0;
    let a = 1.0 - 1.0 / (2.0 * k) - 1.0 / (8.0 * k * k) - 1.0 / (8.0 * k.powi(3));
    // sd of cos(φ − μ) ≈ 1/κ·√(1/2), of sin ≈ 1/√κ
    assert!(
        (c - a).abs() < 4.0 * 0.03 / (n as f64).sqrt() + 1e-4,
        "{c} vs {a}"
    );
    assert!(s.abs() < 4.0 * 0.2 / (n as f64).sqrt(), "{s}");
    let mut r2 = crate::helpers::seed_for_thread(3, 0);
    assert_eq!(super::model_ci::sample_von_mises(mu, 0.0, &mut r2), mu);
}
