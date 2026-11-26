//! Landmark-constrained elastic alignment.

use super::optim::{check_grid, refined_euclid, WarpOptimConfig, WarpOptimMethod};
use super::pairwise::{align_pair_cfg, elastic_align_pair};
use super::rbfgs::{rbfgs_warp_multi, RbfgsParams};
use super::srsf::{reparameterize_curve, srsf_transform};
use super::{
    coprime_nbhd, dp_edge_rslope_span, dp_edge_weight, dp_edge_weight_core,
    dp_edge_weight_linear_core, dp_grid_solve, dp_grid_solve_moves, dp_l2psi_penalty,
    dp_lambda_penalty, dp_prune_is_exact, refine_srsf_linear, DpEdgeCost, DpMoves,
    WarpObjectivePenalty,
};
use crate::error::FdarError;
use crate::helpers::{l2_distance, linear_interp, simpsons_weights};
use crate::matrix::FdMatrix;
use crate::warping::normalize_warp;

/// Result of landmark-constrained elastic alignment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConstrainedAlignmentResult {
    /// Optimal warping function (length m).
    pub gamma: Vec<f64>,
    /// Aligned curve f2∘γ (length m).
    pub f_aligned: Vec<f64>,
    /// Elastic distance after alignment.
    pub distance: f64,
    /// Enforced landmark pairs (snapped to grid): `(target_t, source_t)`.
    pub enforced_landmarks: Vec<(f64, f64)>,
}

/// Snap a time value to the nearest grid point index.
fn snap_to_grid(t_val: f64, argvals: &[f64]) -> usize {
    let mut best = 0;
    let mut best_dist = (t_val - argvals[0]).abs();
    for (i, &a) in argvals.iter().enumerate().skip(1) {
        let d = (t_val - a).abs();
        if d < best_dist {
            best = i;
            best_dist = d;
        }
    }
    best
}

/// Run DP on a rectangular sub-grid `[sc..=ec] × [sr..=er]`.
///
/// Uses global indices for `dp_edge_weight`. Returns the path segment
/// as a list of `(tc_idx, tr_idx)` pairs from start to end.
fn dp_segment(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    sc: usize,
    ec: usize,
    sr: usize,
    er: usize,
    lambda: f64,
) -> Vec<(usize, usize)> {
    let nc = ec - sc + 1;
    let nr = er - sr + 1;

    if nc <= 1 || nr <= 1 {
        return vec![(sc, sr), (ec, er)];
    }

    // Exact pruning (non-negative edge costs on a strictly increasing grid).
    let prune = dp_prune_is_exact(argvals, lambda);
    let path = dp_grid_solve(nr, nc, prune, |local_sr, local_sc, local_tr, local_tc| {
        let gsr = sr + local_sr;
        let gsc = sc + local_sc;
        let gtr = sr + local_tr;
        let gtc = sc + local_tc;
        dp_edge_weight(q1, q2, argvals, gsc, gtc, gsr, gtr)
            + dp_lambda_penalty(argvals, gsc, gtc, gsr, gtr, lambda)
    });

    // Convert local indices to global
    path.iter().map(|&(lr, lc)| (sc + lc, sr + lr)).collect()
}

/// Build DP waypoints from landmark pairs: snap to grid, deduplicate, add endpoints.
fn build_constrained_waypoints(
    landmark_pairs: &[(f64, f64)],
    argvals: &[f64],
    m: usize,
) -> Vec<(usize, usize)> {
    let mut waypoints: Vec<(usize, usize)> = Vec::with_capacity(landmark_pairs.len() + 2);
    waypoints.push((0, 0));
    for &(tt, st) in landmark_pairs {
        let tc = snap_to_grid(tt, argvals);
        let tr = snap_to_grid(st, argvals);
        if let Some(&(prev_c, prev_r)) = waypoints.last() {
            if tc > prev_c && tr > prev_r {
                waypoints.push((tc, tr));
            }
        }
    }
    let last = m - 1;
    if let Some(&(prev_c, prev_r)) = waypoints.last() {
        if prev_c != last || prev_r != last {
            waypoints.push((last, last));
        }
    }
    waypoints
}

/// Run DP segments between consecutive waypoints and assemble into a gamma warp.
fn segmented_dp_gamma(
    q1n: &[f64],
    q2n: &[f64],
    argvals: &[f64],
    waypoints: &[(usize, usize)],
    lambda: f64,
) -> Vec<f64> {
    let paths = segment_paths(waypoints, |sc, ec, sr, er| {
        dp_segment(q1n, q2n, argvals, sc, ec, sr, er, lambda)
    });
    paths_to_gamma(&paths, argvals)
}

/// The path segment `seg(sc, ec, sr, er)` between every pair of consecutive
/// waypoints.
fn segment_paths<S>(waypoints: &[(usize, usize)], seg: S) -> Vec<Vec<(usize, usize)>>
where
    S: Fn(usize, usize, usize, usize) -> Vec<(usize, usize)>,
{
    waypoints
        .windows(2)
        .map(|w| seg(w[0].0, w[1].0, w[0].1, w[1].1))
        .collect()
}

/// Assemble segment paths into a gamma warp (linear interpolation + warp
/// normalisation).
fn paths_to_gamma(paths: &[Vec<(usize, usize)>], argvals: &[f64]) -> Vec<f64> {
    let mut full_path_tc: Vec<f64> = Vec::new();
    let mut full_path_tr: Vec<f64> = Vec::new();

    for (seg, segment_path) in paths.iter().enumerate() {
        let start = usize::from(seg > 0);
        for &(tc, tr) in &segment_path[start..] {
            full_path_tc.push(argvals[tc]);
            full_path_tr.push(argvals[tr]);
        }
    }

    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&full_path_tc, &full_path_tr, t))
        .collect();
    normalize_warp(&mut gamma, argvals);
    gamma
}

/// [`dp_segment`] with the DP options of `cfg` (`nbhd_dim`, `dp_scl`,
/// `dp_cost`, `dp_prune`, `L2Gam` / `L2Psi`), penalty weight `lambda`, on the
/// normalised SRSFs `q1e`, `q2e` (already refined `cfg.dp_scl` times).
#[allow(clippy::too_many_arguments)]
fn dp_segment_opts(
    q1e: &[f64],
    q2e: &[f64],
    argvals: &[f64],
    (sc, ec, sr, er): (usize, usize, usize, usize),
    cfg: &WarpOptimConfig,
    lambda: f64,
) -> Vec<(usize, usize)> {
    let nc = ec - sc + 1;
    let nr = er - sr + 1;
    if nc <= 1 || nr <= 1 {
        return vec![(sc, sr), (ec, er)];
    }
    let s = cfg.dp_scl.max(1);
    let nbhd = coprime_nbhd(cfg.nbhd_dim);
    let moves = DpMoves {
        nbhd: &nbhd,
        prune: cfg.dp_prune && dp_prune_is_exact(argvals, lambda),
    };
    let weight = match cfg.dp_cost {
        DpEdgeCost::PiecewiseLinear => dp_edge_weight_linear_core,
        _ => dp_edge_weight_core,
    };
    let l2psi = lambda > 0.0 && cfg.penalty != WarpObjectivePenalty::L2Gam;
    let path = dp_grid_solve_moves(nr, nc, None, moves, |lsr, lsc, ltr, ltc| {
        let (gsr, gsc, gtr, gtc) = (sr + lsr, sc + lsc, sr + ltr, sc + ltc);
        let (rs, span) = dp_edge_rslope_span(argvals, gsc, gtc, gsr, gtr);
        let pen = if l2psi {
            dp_l2psi_penalty(argvals, gsc, gtc, gsr, gtr, lambda)
        } else {
            dp_lambda_penalty(argvals, gsc, gtc, gsr, gtr, lambda)
        };
        weight(q1e, q2e, gsc * s, gtc * s, gsr * s, gtr * s, rs, span) + pen
    });
    path.iter().map(|&(lr, lc)| (sc + lc, sr + lr)).collect()
}

/// Segment paths of the landmark-constrained DP under `cfg` with weight
/// `lambda`; the classic options run the unchanged [`dp_segment`].
fn dp_segment_paths_cfg(
    q1n: &[f64],
    q2n: &[f64],
    argvals: &[f64],
    waypoints: &[(usize, usize)],
    cfg: &WarpOptimConfig,
    lambda: f64,
) -> Vec<Vec<(usize, usize)>> {
    let classic =
        cfg.dp_seed().is_classic() && (lambda == 0.0 || cfg.penalty == WarpObjectivePenalty::L2Gam);
    if classic {
        return segment_paths(waypoints, |sc, ec, sr, er| {
            dp_segment(q1n, q2n, argvals, sc, ec, sr, er, lambda)
        });
    }
    let s = cfg.dp_scl.max(1);
    let refine = |q: &[f64]| {
        if s > 1 {
            refine_srsf_linear(q, s)
        } else {
            q.to_vec()
        }
    };
    let (q1e, q2e) = (refine(q1n), refine(q2n));
    segment_paths(waypoints, |sc, ec, sr, er| {
        dp_segment_opts(&q1e, &q2e, argvals, (sc, ec, sr, er), cfg, lambda)
    })
}

/// Inputs of the per-segment RBFGS refinement shared by all segments.
struct SegmentRbfgs<'a> {
    q1: &'a [f64],
    q2: &'a [f64],
    argvals: &'a [f64],
    cfg: &'a WarpOptimConfig,
    /// Penalty weight in the RBFGS convention.
    lambda: f64,
    /// Penalty weight of the DP seed (DP convention).
    seed_lambda: f64,
    /// Euclidean norms of the `scl`-refined whole-curve SRSFs (fdasrsf
    /// `lamScale` of the unconstrained problem).
    euclid: (f64, f64),
}

impl SegmentRbfgs<'_> {
    /// RBFGS warp of one landmark segment `[a, b] → [c, d]` (grid indices
    /// `sc..=ec` → `sr..=er`), started from its DP path. The segment is mapped
    /// affinely onto `[0, 1]²`: `q̃1(s) = q1(a + (b − a)s)`,
    /// `q̃2(u) = √κ·q2(c + (d − c)u)` with `κ = (d − c)/(b − a)`, so the
    /// segment's share of the unconstrained objective is (up to a positive
    /// factor and a constant) the standard problem with the penalty slope `κ`
    /// (`L2Gam`, `L2Psi`), the roughness weight scaled by `κ²/L²`
    /// (`L = (b − a)/span`), and fdasrsf's `lamScale` taken from the whole
    /// curves. Returns γ at `argvals[sc..=ec]` with `γ(a) = c`, `γ(b) = d`
    /// exactly.
    fn warp(
        &self,
        (sc, ec, sr, er): (usize, usize, usize, usize),
        path: &[(usize, usize)],
    ) -> Vec<f64> {
        let t = self.argvals;
        let (a, b, c, d) = (t[sc], t[ec], t[sr], t[er]);
        let path_c: Vec<f64> = path.iter().map(|&(tc, _)| t[tc]).collect();
        let path_r: Vec<f64> = path.iter().map(|&(_, tr)| t[tr]).collect();
        let dp_at = |x: f64| linear_interp(&path_c, &path_r, x);
        let msub = (ec - sc).max(er - sr) + 1;
        let mut out: Vec<f64> = t[sc..=ec].iter().map(|&x| dp_at(x).clamp(c, d)).collect();
        if msub >= 3 {
            let gs = self.solve(msub, (a, b, c, d), &dp_at);
            let u: Vec<f64> = (0..msub).map(|j| j as f64 / (msub - 1) as f64).collect();
            for (o, &x) in out.iter_mut().zip(&t[sc..=ec]) {
                let v = c + (d - c) * linear_interp(&u, &gs, (x - a) / (b - a));
                *o = v.clamp(c, d);
            }
        }
        out[0] = c;
        let last = out.len() - 1;
        out[last] = d;
        out
    }

    /// The segment's standard RBFGS problem on a uniform grid of `msub`
    /// points of `[0, 1]`; returns γ̃ on that grid.
    fn solve(
        &self,
        msub: usize,
        (a, b, c, d): (f64, f64, f64, f64),
        dp_at: &dyn Fn(f64) -> f64,
    ) -> Vec<f64> {
        let t = self.argvals;
        let span = t[t.len() - 1] - t[0];
        let kappa = (d - c) / (b - a);
        let rk = kappa.sqrt();
        let u: Vec<f64> = (0..msub).map(|j| j as f64 / (msub - 1) as f64).collect();
        let q1s: Vec<f64> = u
            .iter()
            .map(|&x| linear_interp(t, self.q1, a + (b - a) * x))
            .collect();
        let q2s: Vec<f64> = u
            .iter()
            .map(|&x| rk * linear_interp(t, self.q2, c + (d - c) * x))
            .collect();
        let mut p = RbfgsParams::from_cfg(self.cfg, self.lambda, None);
        p.seed_lambda = self.seed_lambda;
        p.lam_euclid = Some(self.euclid);
        match self.cfg.penalty {
            WarpObjectivePenalty::Roughness => {
                let len = (b - a) / span;
                p.lambda *= kappa * kappa / (len * len);
            }
            _ => p.pen_slope = kappa,
        }
        rbfgs_warp_multi(&[&q1s], &[&q2s], &u, &p, |_, _, tc, _| {
            let mut g: Vec<f64> = tc
                .iter()
                .map(|&x| ((dp_at(a + (b - a) * x) - c) / (d - c)).clamp(0.0, 1.0))
                .collect();
            let n = g.len();
            g[0] = 0.0;
            g[n - 1] = 1.0;
            g
        })
    }
}

/// Align f2 to f1 with landmark constraints.
///
/// Landmark pairs define waypoints on the DP grid. Between consecutive waypoints,
/// an independent smaller DP is run. The resulting warp passes through all landmarks.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `landmark_pairs` — `(target_t, source_t)` pairs in increasing order
/// * `lambda` — Penalty weight
///
/// # Returns
/// [`ConstrainedAlignmentResult`] with warp, aligned curve, and enforced landmarks.
pub fn elastic_align_pair_constrained(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    landmark_pairs: &[(f64, f64)],
    lambda: f64,
) -> ConstrainedAlignmentResult {
    let m = f1.len();

    if landmark_pairs.is_empty() {
        let r = elastic_align_pair(f1, f2, argvals, lambda);
        return ConstrainedAlignmentResult {
            gamma: r.gamma,
            f_aligned: r.f_aligned,
            distance: r.distance,
            enforced_landmarks: Vec::new(),
        };
    }

    let (q1, _, q1n, q2n) = constrained_srsfs(f1, f2, argvals);

    let waypoints = build_constrained_waypoints(landmark_pairs, argvals, m);
    let gamma = segmented_dp_gamma(&q1n, &q2n, argvals, &waypoints, lambda);
    finish_constrained(f2, argvals, &q1, gamma, &waypoints)
}

/// Warped curve, distance and enforced landmarks of a constrained warp.
fn finish_constrained(
    f2: &[f64],
    argvals: &[f64],
    q1: &[f64],
    gamma: Vec<f64>,
    waypoints: &[(usize, usize)],
) -> ConstrainedAlignmentResult {
    let m = f2.len();
    let f_aligned = reparameterize_curve(f2, argvals, &gamma);
    let f_aligned_mat =
        FdMatrix::from_slice(&f_aligned, 1, m).expect("dimension invariant: data.len() == n * m");
    let q_aligned_mat = srsf_transform(&f_aligned_mat, argvals);
    let q_aligned: Vec<f64> = q_aligned_mat.row(0);
    let weights = simpsons_weights(argvals);
    let distance = l2_distance(q1, &q_aligned, &weights);

    let enforced: Vec<(f64, f64)> = waypoints[1..waypoints.len() - 1]
        .iter()
        .map(|&(tc, tr)| (argvals[tc], argvals[tr]))
        .collect();

    ConstrainedAlignmentResult {
        gamma,
        f_aligned,
        distance,
        enforced_landmarks: enforced,
    }
}

/// SRSFs of `f1`, `f2` and their unit-Euclidean-norm versions.
fn constrained_srsfs(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = f1.len();
    let f1_mat = FdMatrix::from_slice(f1, 1, m).expect("dimension invariant: data.len() == n * m");
    let f2_mat = FdMatrix::from_slice(f2, 1, m).expect("dimension invariant: data.len() == n * m");
    let q1_mat = srsf_transform(&f1_mat, argvals);
    let q2_mat = srsf_transform(&f2_mat, argvals);
    let q1: Vec<f64> = q1_mat.row(0);
    let q2: Vec<f64> = q2_mat.row(0);
    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();
    (q1, q2, q1n, q2n)
}

/// [`elastic_align_pair_constrained`] with a configurable warp optimiser.
///
/// * **DP** options (`nbhd_dim`, `dp_scl`, `dp_cost`, `dp_prune`, `L2Psi`)
///   apply to every landmark segment's DP.
/// * **RBFGS**: the segmented DP warp (with the DP options) is computed first;
///   each landmark segment `[a, b] → [c, d]` is then mapped affinely onto
///   `[0, 1]²` and refined by Riemannian L-BFGS started from its DP path, so
///   the warp passes through every (grid-snapped) landmark **exactly**. The
///   segment objective is the unconstrained objective restricted to the
///   segment: the pointwise penalties act on the original slope
///   `γ̇ = κ·γ̃'` (`κ = (d − c)/(b − a)`), the roughness penalty is rescaled
///   accordingly (kinks at the landmarks are not penalised), and `λ` is
///   weighed with the whole curves' norms, so `λ` means the same as in
///   [`elastic_align_pair_with_config`](super::elastic_align_pair_with_config).
///   `Geodesic` (a non-additive function of the whole warp) with `λ > 0` is
///   rejected.
///
/// [`WarpOptimConfig::default`] with `lambda` set gives exactly
/// [`elastic_align_pair_constrained`]; without landmarks this is
/// [`elastic_align_pair_with_config`](super::elastic_align_pair_with_config).
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `f1`, `f2`, `argvals` differ in length
/// or have < 2 points; [`FdarError::InvalidParameter`] for a non-increasing
/// `argvals`, non-finite landmarks, an invalid `cfg`, `cfg.band_frac > 0`
/// with landmarks (no band for segmented alignment) or RBFGS + `Geodesic`
/// with `λ > 0`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{
///     elastic_align_pair_constrained_with_config, WarpOptimConfig, WarpOptimMethod,
/// };
///
/// let t: Vec<f64> = (0..81).map(|i| i as f64 / 80.0).collect();
/// let f1: Vec<f64> = t.iter().map(|&s| (-(s - 0.3).powi(2) / 0.005).exp()
///     + (-(s - 0.7).powi(2) / 0.005).exp()).collect();
/// let f2: Vec<f64> = t.iter().map(|&s| (-(s - 0.4).powi(2) / 0.005).exp()
///     + (-(s - 0.75).powi(2) / 0.005).exp()).collect();
/// let mut cfg = WarpOptimConfig::default();
/// cfg.method = WarpOptimMethod::Rbfgs;
/// let res = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &[(0.3, 0.4)], &cfg)
///     .unwrap();
/// assert_eq!(res.gamma[24], t[32]); // γ(0.3) = 0.4 exactly
/// assert!(res.gamma.windows(2).all(|w| w[1] >= w[0]));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_align_pair_constrained_with_config(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    landmark_pairs: &[(f64, f64)],
    cfg: &WarpOptimConfig,
) -> Result<ConstrainedAlignmentResult, FdarError> {
    check_grid(argvals, f1.len(), "f1")?;
    check_grid(argvals, f2.len(), "f2")?;
    cfg.validate()?;
    if landmark_pairs.is_empty() {
        let r = align_pair_cfg(f1, f2, argvals, cfg);
        return Ok(ConstrainedAlignmentResult {
            gamma: r.gamma,
            f_aligned: r.f_aligned,
            distance: r.distance,
            enforced_landmarks: Vec::new(),
        });
    }
    check_constrained_cfg(landmark_pairs, cfg)?;
    let m = f1.len();
    let (q1, q2, q1n, q2n) = constrained_srsfs(f1, f2, argvals);
    let waypoints = build_constrained_waypoints(landmark_pairs, argvals, m);
    let phi = || cfg.rbfgs_lambda_factor(&q1, &q2);
    let gamma = match cfg.method {
        WarpOptimMethod::Dp => {
            let lam = cfg.native_lambda(WarpOptimMethod::Dp, phi);
            paths_to_gamma(
                &dp_segment_paths_cfg(&q1n, &q2n, argvals, &waypoints, cfg, lam),
                argvals,
            )
        }
        WarpOptimMethod::Rbfgs => {
            let lam = cfg.native_lambda(WarpOptimMethod::Rbfgs, phi);
            // fdasrsf pen_start: the seed DP uses the same weight unless the
            // λ conversion is active (then the DP-convention weight).
            let seed_lambda = match cfg.lambda_scale {
                super::WarpLambdaScale::Native => lam,
                _ => cfg.native_lambda(WarpOptimMethod::Dp, phi),
            };
            let paths = dp_segment_paths_cfg(&q1n, &q2n, argvals, &waypoints, cfg, seed_lambda);
            let scl = cfg.scl.max(1);
            let seg = SegmentRbfgs {
                q1: &q1,
                q2: &q2,
                argvals,
                cfg,
                lambda: lam,
                seed_lambda,
                euclid: (refined_euclid(&[&q1], scl), refined_euclid(&[&q2], scl)),
            };
            let mut gamma = vec![0.0; m];
            for (w, path) in waypoints.windows(2).zip(&paths) {
                let g = seg.warp((w[0].0, w[1].0, w[0].1, w[1].1), path);
                gamma[w[0].0..=w[1].0].copy_from_slice(&g);
            }
            normalize_warp(&mut gamma, argvals);
            gamma
        }
    };
    Ok(finish_constrained(f2, argvals, &q1, gamma, &waypoints))
}

/// Option checks of [`elastic_align_pair_constrained_with_config`] with
/// landmarks.
fn check_constrained_cfg(
    landmark_pairs: &[(f64, f64)],
    cfg: &WarpOptimConfig,
) -> Result<(), FdarError> {
    if landmark_pairs
        .iter()
        .any(|&(a, b)| !a.is_finite() || !b.is_finite())
    {
        return Err(FdarError::InvalidParameter {
            parameter: "landmark_pairs",
            message: "landmark positions must be finite".to_string(),
        });
    }
    if cfg.band_frac != 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "band_frac",
            message: "the Sakoe–Chiba band is not supported with landmark constraints".to_string(),
        });
    }
    if cfg.method == WarpOptimMethod::Rbfgs
        && cfg.penalty == WarpObjectivePenalty::Geodesic
        && cfg.lambda > 0.0
    {
        return Err(FdarError::InvalidParameter {
            parameter: "penalty",
            message: "the Geodesic penalty is not additive over landmark segments; use L2Gam, \
                      L2Psi or Roughness"
                .to_string(),
        });
    }
    Ok(())
}

/// Align f2 to f1 with automatic landmark detection and elastic constraints.
///
/// Detects landmarks in both curves, matches them, and uses the matches
/// as constraints for segmented DP alignment.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `kind` — Type of landmarks to detect
/// * `min_prominence` — Minimum prominence for landmark detection
/// * `expected_count` — Expected number of landmarks (0 = all detected)
/// * `lambda` — Penalty weight
pub fn elastic_align_pair_with_landmarks(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    kind: crate::landmark::LandmarkKind,
    min_prominence: f64,
    expected_count: usize,
    lambda: f64,
) -> ConstrainedAlignmentResult {
    let lm1 = crate::landmark::detect_landmarks(f1, argvals, kind, min_prominence);
    let lm2 = crate::landmark::detect_landmarks(f2, argvals, kind, min_prominence);

    // Match landmarks by order (take min count)
    let n_match = if expected_count > 0 {
        expected_count.min(lm1.len()).min(lm2.len())
    } else {
        lm1.len().min(lm2.len())
    };

    let pairs: Vec<(f64, f64)> = (0..n_match)
        .map(|i| (lm1[i].position, lm2[i].position))
        .collect();

    elastic_align_pair_constrained(f1, f2, argvals, &pairs, lambda)
}
