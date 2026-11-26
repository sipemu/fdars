//! Elastic DP-means clustering with alignment (learned number of clusters).
//!
//! Port of fdasrsf 2.7.5 `kmeans.dpmeans_align` (fdasrvf 2.6.0
//! `dpmeans_align`): the DP-means algorithm of Kulis & Jordan (2012), the
//! small-variance limit of a Dirichlet-process mixture, in the elastic SRSF
//! metric. Each pass aligns every function to every cluster template; a
//! sequential sweep then opens a new cluster whenever a function's elastic
//! distance to all templates exceeds the penalty `λ`. Empty clusters are
//! dropped, the warps of each cluster are centred by their inverse Karcher
//! mean warp (`SqrtMeanInverse`), and each template becomes the mean aligned
//! SRSF of its cluster. The objective is `Σ dᵢ² + λ²·K`.
//!
//! # References
//!
//! - Kulis, B. & Jordan, M. I. (2012). Revisiting k-means: new algorithms via
//!   Bayesian nonparametrics. *ICML*.
//! - Tucker, J. D. fdasrsf / fdasrvf `dpmeans_align`.

use super::karcher::sqrt_mean_inverse;
use super::optim::{optimal_warp, WarpOptimConfig};
use super::srsf::{compose_warps, reparameterize_curve, srsf_inverse, srsf_single, srsf_transform};
use crate::error::FdarError;
use crate::helpers::trapz;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Configuration for [`elastic_dpmeans`].
///
/// Construct via `DpMeansConfig::default()`, then assign the fields you
/// need. This struct is `#[non_exhaustive]`, so external crates cannot build
/// it with a struct literal.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DpMeansConfig {
    /// Cluster penalty `λ`, as an elastic distance: a function farther than
    /// this from every template opens a new cluster. `None` (default) picks
    /// it with fdasrsf's k-means++ heuristic: the largest distance from any
    /// function to its nearest of 4 k-means++ centres (seeded by `seed`).
    pub penalty: Option<f64>,
    /// Alignment roughness penalty passed to the DP (default 0).
    pub lambda: f64,
    /// Maximum number of passes (default 20).
    pub max_iter: usize,
    /// Relative objective change for convergence (default 0.01).
    pub tol: f64,
    /// Whether to align (default `true`; `false` uses identity warps, i.e.
    /// plain DP-means on SRSFs).
    pub align: bool,
    /// Seed for the k-means++ penalty heuristic (default 42).
    pub seed: u64,
}

impl Default for DpMeansConfig {
    fn default() -> Self {
        Self {
            penalty: None,
            lambda: 0.0,
            max_iter: 20,
            tol: 0.01,
            align: true,
            seed: 42,
        }
    }
}

/// Result of [`elastic_dpmeans`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DpMeansResult {
    /// Cluster label of each function (0-indexed, length n).
    pub labels: Vec<usize>,
    /// Number of clusters found.
    pub k: usize,
    /// Cluster template functions (k × m).
    pub templates: FdMatrix,
    /// Cluster template SRSFs (k × m).
    pub templates_srsf: FdMatrix,
    /// Each function aligned to its own cluster template (n × m).
    pub aligned_data: FdMatrix,
    /// SRSFs of the aligned functions (n × m).
    pub aligned_srsfs: FdMatrix,
    /// Warping functions aligning each function to its template (n × m).
    pub gammas: FdMatrix,
    /// Inverse Karcher mean warp used to centre each cluster (k × m).
    pub gamma_inverse: FdMatrix,
    /// DP-means objective `Σ dᵢ² + λ²K` after each pass.
    pub objective: Vec<f64>,
    /// Cluster penalty `λ` used.
    pub penalty: f64,
    /// Number of passes performed.
    pub n_iter: usize,
    /// Whether the relative objective change fell below `tol`.
    pub converged: bool,
}

/// Every function aligned to one template SRSF.
struct TemplateAlignment {
    gam: Vec<Vec<f64>>,
    fw: Vec<Vec<f64>>,
    qw: Vec<Vec<f64>>,
    dist: Vec<f64>,
}

/// Shared inputs of the clustering passes.
struct Problem<'a> {
    f: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    t: &'a [f64],
    warp_cfg: WarpOptimConfig,
    align: bool,
}

impl Problem<'_> {
    /// fdasrsf `_align_to_template`.
    fn align_to(&self, tq: &[f64]) -> TemplateAlignment {
        let t = self.t;
        let out: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..self.f.len())
            .map(|i| {
                let gam = if self.align {
                    optimal_warp(tq, &self.q[i], t, &self.warp_cfg, None)
                } else {
                    t.to_vec()
                };
                let fw = reparameterize_curve(&self.f[i], t, &gam);
                let qw = srsf_single(&fw, t);
                let d = l2_dist(&qw, tq, t);
                (gam, fw, qw, d)
            })
            .collect();
        let mut res = TemplateAlignment {
            gam: Vec::with_capacity(out.len()),
            fw: Vec::with_capacity(out.len()),
            qw: Vec::with_capacity(out.len()),
            dist: Vec::with_capacity(out.len()),
        };
        for (g, fw, qw, d) in out {
            res.gam.push(g);
            res.fw.push(fw);
            res.qw.push(qw);
            res.dist.push(d);
        }
        res
    }
}

fn l2_dist(a: &[f64], b: &[f64], t: &[f64]) -> f64 {
    let sq: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).collect();
    trapz(&sq, t).max(0.0).sqrt()
}

/// Cluster state carried across passes.
struct State {
    templates: Vec<Vec<f64>>,
    templates_q: Vec<Vec<f64>>,
    labels: Vec<usize>,
    dmin: Vec<f64>,
}

/// fdasrsf `_kpp_init`: k-means++ seeding of 4 centres; returns the largest
/// distance of any function to its nearest centre.
fn kpp_penalty(p: &Problem<'_>, seed: u64) -> f64 {
    const K: usize = 4;
    let n = p.q.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx = rng.gen_range(0..n);
    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..K {
        let d = p.align_to(&p.q[idx]).dist;
        for (a, b) in nearest.iter_mut().zip(&d) {
            *a = a.min(*b);
        }
        if i + 1 < K {
            let w: Vec<f64> = nearest.iter().map(|d| d * d).collect();
            let total: f64 = w.iter().sum();
            if total == 0.0 {
                break;
            }
            idx = searchsorted_right(&w, rng.gen::<f64>() * total);
        }
    }
    nearest.iter().copied().fold(0.0, f64::max)
}

/// `numpy.searchsorted(cumsum(w), target, "right")`, clamped to `len - 1`.
fn searchsorted_right(w: &[f64], target: f64) -> usize {
    let mut cum = 0.0;
    for (j, &wj) in w.iter().enumerate() {
        cum += wj;
        if cum > target {
            return j;
        }
    }
    w.len() - 1
}

/// Sequential DP-means sweep: assign each function to its nearest template,
/// opening a new cluster (template = that function) when every distance
/// exceeds `lam`. Returns the per-template alignments, extended for any new
/// clusters.
fn dp_sweep(p: &Problem<'_>, st: &mut State, lam: f64) -> Vec<TemplateAlignment> {
    let mut al: Vec<TemplateAlignment> = st.templates_q.iter().map(|tq| p.align_to(tq)).collect();
    for i in 0..p.f.len() {
        let (mut ik, mut dk) = (0, f64::INFINITY);
        for (k, a) in al.iter().enumerate() {
            if a.dist[i] < dk {
                (ik, dk) = (k, a.dist[i]);
            }
        }
        if dk > lam {
            st.templates.push(p.f[i].clone());
            st.templates_q.push(p.q[i].clone());
            let a = p.align_to(&p.q[i]);
            ik = al.len();
            dk = a.dist[i];
            al.push(a);
        }
        st.labels[i] = ik;
        st.dmin[i] = dk;
    }
    al
}

/// Drop clusters without members and relabel `0..K` in order of the old labels.
fn prune(st: &mut State, al: Vec<TemplateAlignment>) -> Vec<TemplateAlignment> {
    let kmax = al.len();
    let mut used = vec![false; kmax];
    st.labels.iter().for_each(|&l| used[l] = true);
    let mut remap = vec![usize::MAX; kmax];
    let mut next = 0;
    for k in 0..kmax {
        if used[k] {
            remap[k] = next;
            next += 1;
        }
    }
    st.labels.iter_mut().for_each(|l| *l = remap[*l]);
    let keep = |v: &mut Vec<Vec<f64>>| {
        let old = std::mem::take(v);
        *v = old
            .into_iter()
            .enumerate()
            .filter(|(k, _)| used[*k])
            .map(|(_, x)| x)
            .collect();
    };
    keep(&mut st.templates);
    keep(&mut st.templates_q);
    al.into_iter()
        .enumerate()
        .filter(|(k, _)| used[*k])
        .map(|(_, a)| a)
        .collect()
}

/// Per-function alignment to its own template.
struct Assigned {
    gam: Vec<Vec<f64>>,
    fw: Vec<Vec<f64>>,
    qw: Vec<Vec<f64>>,
}

fn gather(st: &State, al: &[TemplateAlignment]) -> Assigned {
    let pick = |sel: fn(&TemplateAlignment) -> &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        st.labels
            .iter()
            .enumerate()
            .map(|(i, &k)| sel(&al[k])[i].clone())
            .collect()
    };
    Assigned {
        gam: pick(|a| &a.gam),
        fw: pick(|a| &a.fw),
        qw: pick(|a| &a.qw),
    }
}

/// Centre each cluster by its inverse Karcher mean warp (fdasrsf `_normalize`)
/// and set the templates to the mean aligned SRSF. Returns the `gamI` per cluster.
fn normalize_and_update(st: &mut State, asg: &mut Assigned, t: &[f64]) -> Vec<Vec<f64>> {
    let m = t.len();
    let k = st.templates.len();
    let mut gam_inv = Vec::with_capacity(k);
    for c in 0..k {
        let idx: Vec<usize> = (0..st.labels.len())
            .filter(|&i| st.labels[i] == c)
            .collect();
        let mut g = FdMatrix::zeros(idx.len(), m);
        for (r, &i) in idx.iter().enumerate() {
            for j in 0..m {
                g[(r, j)] = asg.gam[i][j];
            }
        }
        let gi = sqrt_mean_inverse(&g, t);
        let mut tq = vec![0.0; m];
        let mut f0 = 0.0;
        for &i in &idx {
            asg.fw[i] = reparameterize_curve(&asg.fw[i], t, &gi);
            asg.qw[i] = srsf_single(&asg.fw[i], t);
            asg.gam[i] = compose_warps(&asg.gam[i], &gi, t);
            for j in 0..m {
                tq[j] += asg.qw[i][j];
            }
            f0 += asg.fw[i][0];
        }
        let cnt = idx.len() as f64;
        tq.iter_mut().for_each(|x| *x /= cnt);
        st.templates[c] = srsf_inverse(&tq, t, f0 / cnt);
        st.templates_q[c] = tq;
        gam_inv.push(gi);
    }
    gam_inv
}

fn to_matrix(rows: &[Vec<f64>], m: usize) -> FdMatrix {
    let mut out = FdMatrix::zeros(rows.len(), m);
    for (i, r) in rows.iter().enumerate() {
        for j in 0..m {
            out[(i, j)] = r[j];
        }
    }
    out
}

fn validate(data: &FdMatrix, argvals: &[f64], cfg: &DpMeansConfig) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m < 3 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (m >= 3)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    validate_params(cfg)
}

fn non_negative_finite(x: f64) -> bool {
    x >= 0.0 && x.is_finite()
}

fn validate_params(cfg: &DpMeansConfig) -> Result<(), FdarError> {
    if let Some(p) = cfg.penalty {
        if !non_negative_finite(p) {
            return Err(FdarError::InvalidParameter {
                parameter: "penalty",
                message: format!("must be non-negative and finite, got {p}"),
            });
        }
    }
    if !non_negative_finite(cfg.lambda) {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("must be non-negative and finite, got {}", cfg.lambda),
        });
    }
    if cfg.max_iter == 0 || cfg.tol.is_nan() || cfg.tol < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter/tol",
            message: format!(
                "need max_iter >= 1 and tol >= 0, got {} / {}",
                cfg.max_iter, cfg.tol
            ),
        });
    }
    Ok(())
}

/// Initial single cluster: the function whose SRSF is closest to the
/// cross-sectional mean SRSF.
fn initial_state(p: &Problem<'_>) -> State {
    let n = p.q.len();
    let m = p.t.len();
    let mut mnq = vec![0.0; m];
    for q in &p.q {
        for j in 0..m {
            mnq[j] += q[j] / n as f64;
        }
    }
    let d: Vec<f64> = p.q.iter().map(|q| l2_dist(q, &mnq, p.t)).collect();
    let mut best = 0;
    for i in 1..n {
        if d[i] < d[best] {
            best = i;
        }
    }
    State {
        templates: vec![p.f[best].clone()],
        templates_q: vec![p.q[best].clone()],
        labels: vec![0; n],
        dmin: vec![0.0; n],
    }
}

/// Elastic DP-means clustering with alignment (fdasrsf/fdasrvf `dpmeans_align`).
///
/// Clusters functions in the elastic (Fisher–Rao / SRSF) metric without
/// fixing the number of clusters: the cluster penalty `λ` controls how far a
/// function may be from every template before it opens a new cluster.
/// Large `λ` gives one cluster; `λ → 0` gives one cluster per distinct
/// function.
///
/// # Arguments
/// * `data` — Functional data (n × m).
/// * `argvals` — Evaluation grid (length m).
/// * `config` — Penalty, alignment penalty, iterations, tolerance, seed.
///
/// # Errors
/// [`FdarError::InvalidDimension`] if n < 2, m < 3 or `argvals` length
/// differs; [`FdarError::InvalidParameter`] for a negative/non-finite
/// penalty or lambda, `max_iter == 0` or negative `tol`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{elastic_dpmeans, DpMeansConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 30;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 8;
/// let mut f = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     let a = if i < 4 { 1.0 } else { 4.0 };
///     for j in 0..m {
///         f[(i, j)] = a * (std::f64::consts::PI * 2.0 * t[j]).sin() + 0.01 * i as f64 * t[j];
///     }
/// }
/// let mut cfg = DpMeansConfig::default();
/// cfg.penalty = Some(1.5);
/// let res = elastic_dpmeans(&f, &t, &cfg).unwrap();
/// assert_eq!(res.k, 2);
/// assert_eq!(res.labels[0], res.labels[3]);
/// assert_ne!(res.labels[0], res.labels[4]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_dpmeans(
    data: &FdMatrix,
    argvals: &[f64],
    config: &DpMeansConfig,
) -> Result<DpMeansResult, FdarError> {
    validate(data, argvals, config)?;
    let (n, m) = data.shape();
    let q = srsf_transform(data, argvals);
    let p = Problem {
        f: (0..n).map(|i| data.row(i)).collect(),
        q: (0..n).map(|i| q.row(i)).collect(),
        t: argvals,
        warp_cfg: WarpOptimConfig::classic_dp(config.lambda),
        align: config.align,
    };
    let mut st = initial_state(&p);
    let lam = config
        .penalty
        .unwrap_or_else(|| kpp_penalty(&p, config.seed));

    let mut objective = Vec::with_capacity(config.max_iter);
    let mut converged = false;
    let mut asg = Assigned {
        gam: Vec::new(),
        fw: Vec::new(),
        qw: Vec::new(),
    };
    let mut gam_inv = Vec::new();
    for itr in 0..config.max_iter {
        let al = dp_sweep(&p, &mut st, lam);
        let al = prune(&mut st, al);
        let k = st.templates.len();
        objective.push(st.dmin.iter().map(|d| d * d).sum::<f64>() + lam * lam * k as f64);
        asg = gather(&st, &al);
        gam_inv = normalize_and_update(&mut st, &mut asg, argvals);
        if itr > 0 && (objective[itr] - objective[itr - 1]).abs() < config.tol * objective[itr - 1]
        {
            converged = true;
            break;
        }
    }

    Ok(DpMeansResult {
        k: st.templates.len(),
        labels: st.labels,
        templates: to_matrix(&st.templates, m),
        templates_srsf: to_matrix(&st.templates_q, m),
        aligned_data: to_matrix(&asg.fw, m),
        aligned_srsfs: to_matrix(&asg.qw, m),
        gammas: to_matrix(&asg.gam, m),
        gamma_inverse: to_matrix(&gam_inv, m),
        n_iter: objective.len(),
        objective,
        penalty: lam,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// Three well-separated shape groups (one bump, two bumps, ramp), each with
    /// distinct amplitude and phase perturbations so no two curves coincide.
    fn three_groups(per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
        let t = grid(m);
        let n = 3 * per;
        let mut f = FdMatrix::zeros(n, m);
        let mut truth = Vec::with_capacity(n);
        for i in 0..n {
            let g = i % 3;
            let r = (i / 3) as f64;
            let a = 1.0 + 0.07 * r;
            let c = 0.15 * (r / per as f64 - 0.5);
            for j in 0..m {
                let x = t[j] + c * t[j] * (1.0 - t[j]);
                f[(i, j)] = match g {
                    0 => a * (PI * x).sin(),
                    1 => a * (4.0 * PI * x).sin(),
                    _ => 3.0 * a * x * x,
                };
            }
            truth.push(g);
        }
        (f, t, truth)
    }

    fn same_partition(a: &[usize], b: &[usize]) -> bool {
        (0..a.len()).all(|i| (0..a.len()).all(|j| (a[i] == a[j]) == (b[i] == b[j])))
    }

    fn run(f: &FdMatrix, t: &[f64], pen: f64) -> DpMeansResult {
        let cfg = DpMeansConfig {
            penalty: Some(pen),
            ..DpMeansConfig::default()
        };
        elastic_dpmeans(f, t, &cfg).unwrap()
    }

    #[test]
    fn recovers_three_groups() {
        let (f, t, truth) = three_groups(5, 41);
        let res = run(&f, &t, 1.0);
        assert_eq!(res.k, 3);
        assert!(
            same_partition(&res.labels, &truth),
            "labels {:?}",
            res.labels
        );
        assert_eq!(res.templates.shape(), (3, 41));
        assert_eq!(res.gammas.shape(), (15, 41));
        // warps are monotone and keep the endpoints
        for i in 0..15 {
            let g = res.gammas.row(i);
            assert!((g[0] - t[0]).abs() < 1e-9 && (g[40] - t[40]).abs() < 1e-9);
            assert!(g.windows(2).all(|w| w[1] >= w[0] - 1e-12));
        }
    }

    #[test]
    fn huge_penalty_gives_one_cluster() {
        let (f, t, _) = three_groups(4, 31);
        let res = run(&f, &t, 1e6);
        assert_eq!(res.k, 1);
        assert!(res.labels.iter().all(|&l| l == 0));
        // objective = Σd² + λ²: dominated by λ²
        let sumsq: f64 = res.objective.last().unwrap() - 1e12;
        assert!(sumsq >= 0.0);
    }

    #[test]
    fn zero_penalty_gives_n_clusters() {
        let (f, t, _) = three_groups(3, 31);
        let res = run(&f, &t, 0.0);
        assert_eq!(res.k, 9);
        let mut l = res.labels.clone();
        l.sort_unstable();
        assert_eq!(l, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn cluster_count_non_increasing_in_penalty() {
        let (f, t, _) = three_groups(4, 31);
        let mut prev = usize::MAX;
        for pen in [0.0, 0.05, 0.2, 1.0, 5.0, 100.0] {
            let k = run(&f, &t, pen).k;
            assert!(k <= prev, "K={k} at penalty {pen} exceeds {prev}");
            prev = k;
        }
        assert_eq!(prev, 1);
    }

    #[test]
    fn objective_matches_definition() {
        // One pass, no alignment, one cluster: the objective must equal
        // Σ‖qᵢ − q*‖² + λ² with q* the SRSF closest to the mean SRSF,
        // recomputed here independently.
        let (f, t, _) = three_groups(3, 31);
        let pen = 10.0;
        let cfg = DpMeansConfig {
            penalty: Some(pen),
            align: false,
            max_iter: 1,
            ..DpMeansConfig::default()
        };
        let res = elastic_dpmeans(&f, &t, &cfg).unwrap();
        assert_eq!(res.k, 1);
        let q = srsf_transform(&f, &t);
        let (n, m) = q.shape();
        let mean: Vec<f64> = (0..m)
            .map(|j| (0..n).map(|i| q[(i, j)]).sum::<f64>() / n as f64)
            .collect();
        let dist = |a: &[f64], b: &[f64]| {
            let sq: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).collect();
            trapz(&sq, &t).sqrt()
        };
        let best = (0..n)
            .min_by(|&a, &b| {
                dist(&q.row(a), &mean)
                    .partial_cmp(&dist(&q.row(b), &mean))
                    .unwrap()
            })
            .unwrap();
        let expected: f64 = (0..n)
            .map(|i| dist(&q.row(i), &q.row(best)).powi(2))
            .sum::<f64>()
            + pen * pen;
        assert!(
            (res.objective[0] - expected).abs() < 1e-9 * expected,
            "{} vs {expected}",
            res.objective[0]
        );
        for i in 0..n {
            assert!(res
                .gammas
                .row(i)
                .iter()
                .zip(&t)
                .all(|(a, b)| (a - b).abs() < 1e-8));
        }
    }

    #[test]
    fn kpp_penalty_is_deterministic_and_positive() {
        let (f, t, _) = three_groups(3, 31);
        let cfg = DpMeansConfig::default();
        let a = elastic_dpmeans(&f, &t, &cfg).unwrap();
        let b = elastic_dpmeans(&f, &t, &cfg).unwrap();
        assert_eq!(a.penalty, b.penalty);
        assert_eq!(a.labels, b.labels);
        assert!(a.penalty > 0.0);
    }

    /// Reference: fdasrsf 2.7.5 `kmeans.dpmeans_align(f, t, lam=λ, omethod="DP2")`
    /// on the same `three_groups(4, 31)` data
    /// (`.planning/quick/261126-3sq-…/golden_dpmeans.py`). K and labels must
    /// match exactly. Final objectives agree to a few percent only: fdasrsf
    /// computes SRSFs from a spline derivative, fdars from finite differences,
    /// so the residual distances differ slightly; the bound below (10%)
    /// documents that numerics gap, it is not a fitted tolerance.
    #[test]
    fn matches_fdasrsf_dpmeans_align() {
        let (f, t, _) = three_groups(4, 31);
        let golden: [(f64, usize, [usize; 12], f64); 6] = [
            (0.0, 12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 0.0),
            (
                0.05,
                12,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                0.030000000000000006,
            ),
            (
                0.2,
                6,
                [0, 1, 2, 0, 3, 2, 0, 4, 2, 0, 5, 2],
                0.28280866145784034,
            ),
            (
                1.0,
                3,
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                3.356412004689473,
            ),
            (5.0, 1, [0; 12], 46.928463707816746),
            (100.0, 1, [0; 12], 10027.092295792547),
        ];
        for (pen, k, labels, obj) in golden {
            let r = run(&f, &t, pen);
            assert_eq!(r.k, k, "K at penalty {pen}");
            assert_eq!(r.labels, labels.to_vec(), "labels at penalty {pen}");
            let o = *r.objective.last().unwrap();
            assert!(
                (o - obj).abs() <= 0.1 * obj + 1e-12,
                "objective {o} vs fdasrsf {obj} at {pen}"
            );
        }
    }

    #[test]
    fn invalid_inputs_rejected() {
        let t = grid(10);
        let cfg = DpMeansConfig::default();
        assert!(elastic_dpmeans(&FdMatrix::zeros(1, 10), &t, &cfg).is_err());
        assert!(elastic_dpmeans(&FdMatrix::zeros(3, 9), &t, &cfg).is_err());
        let mut bad = cfg.clone();
        bad.penalty = Some(-1.0);
        assert!(elastic_dpmeans(&FdMatrix::zeros(3, 10), &t, &bad).is_err());
        bad = cfg.clone();
        bad.max_iter = 0;
        assert!(elastic_dpmeans(&FdMatrix::zeros(3, 10), &t, &bad).is_err());
    }
}
