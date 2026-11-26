//! Coordinate-wise analytic alignment of a piecewise-constant SRV `q` (breaks `s`)
//! to a piecewise-constant SRV `p` (breaks `r`) — elasdics
//! `find_optimal_t_discrete`, `find_optimal_t_discrete_closed`,
//! `optimise_one_coord_analytic`, `get_loss_discrete`, `compute_distance`.
//!
//! The warped inner product is `Σ_k sqrt(Δs_k ∫_{t_k}^{t_{k+1}} ⟨q_k, p(u)⟩_+² du)`;
//! maximising it over non-decreasing `t` (`t_0 = 0`, `t_M = 1`) gives the elastic
//! distance `‖p‖² + ‖q‖² − 2·max`.

use super::polygon::{dot, SrvPoly};

/// Piecewise-constant reference SRV on breaks `r` (`r.len() = p.len()/d + 1`).
pub(super) struct Reference<'a> {
    pub d: usize,
    pub r: &'a [f64],
    pub p: &'a [f64],
}

impl Reference<'_> {
    fn pk(&self, k: usize) -> &[f64] {
        &self.p[k * self.d..(k + 1) * self.d]
    }

    /// Index of the last break `r[k] <= x`, clamped to a valid SRV segment.
    fn seg_of(&self, x: f64) -> usize {
        let k = self.r.partition_point(|&v| v <= x).saturating_sub(1);
        k.min(self.r.len() - 2)
    }
}

#[inline]
fn cross_plus(a: &[f64], b: &[f64]) -> f64 {
    dot(a, b).max(0.0)
}

/// Maximiser of `√(a x + b) + √(c x + dd)` on `[lo, hi]` (elasdics' closed form
/// with its special cases), returned as `(value, x)`.
fn piece_optimum(a: f64, b: f64, c: f64, dd: f64, lo: f64, hi: f64) -> (f64, f64) {
    let mut x = (c * c * b - a * a * dd) / (a * c * (a - c));
    if a == 0.0 {
        x = if c <= 0.0 { lo } else { hi };
    }
    if c == 0.0 {
        x = if a <= 0.0 { lo } else { hi };
    }
    if x < lo {
        x = lo;
    }
    if x > hi {
        x = hi;
    }
    ((a * x + b).sqrt() + (c * x + dd).sqrt(), x)
}

/// Optimal `t_i` for interior index `i` with all other coordinates fixed
/// (elasdics `optimise_one_coord_analytic`); `s`/`q` describe the warped SRV.
fn optimise_one_coord(t: &[f64], i: usize, p: &Reference<'_>, s: &[f64], q: &SrvSlices<'_>) -> f64 {
    let (lo, hi) = (t[i - 1], t[i + 1]);
    if lo == hi {
        return t[i];
    }
    // Restricted break points and the reference SRV on each restricted piece.
    let mut rr = vec![lo];
    let mut pidx = vec![p.r.partition_point(|&v| v <= lo).saturating_sub(1)];
    for (k, &rk) in p.r.iter().enumerate() {
        if rk > lo && rk < hi {
            rr.push(rk);
            pidx.push(k);
        }
    }
    rr.push(hi);
    let npiece = rr.len() - 1;
    let (q_left, q_right) = (q.qk(i - 1), q.qk(i));
    let (ds_left, ds_right) = (s[i] - s[i - 1], s[i + 1] - s[i]);
    let h_left: Vec<f64> = (0..npiece)
        .map(|k| cross_plus(p.pk(pidx[k]), q_left).powi(2) * (rr[k + 1] - rr[k]))
        .collect();
    let h_right: Vec<f64> = (0..npiece)
        .map(|k| cross_plus(p.pk(pidx[k]), q_right).powi(2) * (rr[k + 1] - rr[k]))
        .collect();
    let mut best_val = f64::NEG_INFINITY;
    let mut best_t = t[i];
    let mut sum_left = 0.0;
    let mut sum_right: f64 = h_right.iter().sum();
    for j in 0..npiece {
        sum_right -= h_right[j];
        let a = ds_left * cross_plus(p.pk(pidx[j]), q_left).powi(2);
        let b = ds_left * sum_left - a * rr[j];
        let c = -ds_right * cross_plus(p.pk(pidx[j]), q_right).powi(2);
        let dd = ds_right * sum_right - c * rr[j + 1];
        let (val, x) = piece_optimum(a, b, c, dd, rr[j], rr[j + 1]);
        // R `which.max`: first maximum, NaN ignored.
        if val > best_val {
            best_val = val;
            best_t = x;
        }
        sum_left += h_left[j];
    }
    best_t
}

/// Borrowed view of the warped SRV vectors.
pub(super) struct SrvSlices<'a> {
    pub d: usize,
    pub q: &'a [f64],
}

impl SrvSlices<'_> {
    fn qk(&self, k: usize) -> &[f64] {
        &self.q[k * self.d..(k + 1) * self.d]
    }
}

/// One elasdics sweep: even interior indices (1-based `2, 4, …`), then odd ones.
fn sweep(
    t: &mut [f64],
    p: &Reference<'_>,
    s: &[f64],
    q: &SrvSlices<'_>,
    after_even: impl Fn(&mut [f64]),
) {
    let len = t.len();
    // 1-based even indices 2..=len-1 are 0-based 1, 3, 5, …
    for i in (1..len - 1).step_by(2) {
        t[i] = optimise_one_coord(t, i, p, s, q);
    }
    after_even(t);
    for i in (2..len - 1).step_by(2) {
        t[i] = optimise_one_coord(t, i, p, s, q);
    }
}

/// elasdics `find_optimal_t_discrete`: open curves, `initial_t.len() = q.nseg() + 1`.
pub(super) fn find_optimal_t_discrete(
    p: &Reference<'_>,
    q: &SrvPoly,
    initial_t: &[f64],
    eps: f64,
) -> Vec<f64> {
    let mut t = initial_t.to_vec();
    if t.len() < 3 {
        return t;
    }
    let eps = eps / t.len() as f64;
    let qs = SrvSlices { d: q.d, q: &q.q };
    loop {
        let t_old = t.clone();
        sweep(&mut t, p, &q.t, &qs, |_| {});
        let delta = max_abs_diff(&t_old, &t);
        if delta.is_nan() || delta < eps {
            break;
        }
    }
    t
}

/// Periodic extension of a closed SRV by one period on either side.
pub(super) struct Extended {
    pub r: Vec<f64>,
    pub p: Vec<f64>,
}

pub(super) fn extend_periodic(srv: &SrvPoly) -> Extended {
    let n = srv.nseg();
    let mut r = Vec::with_capacity(3 * n + 1);
    r.extend(srv.t[..n].iter().map(|v| v - 1.0));
    r.extend_from_slice(&srv.t);
    r.extend(srv.t[1..].iter().map(|v| v + 1.0));
    let mut p = Vec::with_capacity(3 * srv.q.len());
    for _ in 0..3 {
        p.extend_from_slice(&srv.q);
    }
    Extended { r, p }
}

/// elasdics `find_optimal_t_discrete_closed`: returns `t` with `t[0] = t[M] − 1`
/// (not yet wrapped into `[0, 1]`).
pub(super) fn find_optimal_t_discrete_closed(
    p_srv: &SrvPoly,
    q: &SrvPoly,
    initial_t: &[f64],
    eps: f64,
) -> Vec<f64> {
    let ext = extend_periodic(p_srv);
    let p = Reference {
        d: p_srv.d,
        r: &ext.r,
        p: &ext.p,
    };
    let d = q.d;
    let mut q_ext = q.q.clone();
    q_ext.extend_from_slice(&q.q[..d]);
    let mut s_ext = q.t.clone();
    s_ext.push(q.t[1] + 1.0);
    let qs = SrvSlices { d, q: &q_ext };
    let mut t = initial_t.to_vec();
    t.push(initial_t[1] + 1.0);
    let len = t.len();
    let eps = eps / len as f64;
    loop {
        let t_old = t.clone();
        sweep(&mut t, &p, &s_ext, &qs, |t| t[len - 1] = t[1] + 1.0);
        t[0] = t[len - 2] - 1.0;
        let delta = max_abs_diff(&t_old, &t);
        if delta.is_nan() || delta < eps {
            break;
        }
    }
    t.truncate(len - 1);
    t
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |m, v| {
            if v.is_nan() || m.is_nan() {
                f64::NAN
            } else {
                m.max(v)
            }
        })
}

/// elasdics `get_loss_discrete`: `Σ_k sqrt(Δs_k Σ_breaks Δ ⟨q_k, p⟩_+²)` for the
/// warp `t` of `q` against the reference with breaks `r` (`r` covering `t`).
pub(super) fn loss_discrete(t: &[f64], p: &Reference<'_>, q: &SrvPoly) -> f64 {
    // Breaks: sorted unique union of t and r without its final break.
    let mut breaks: Vec<f64> = t.iter().chain(&p.r[..p.r.len() - 1]).copied().collect();
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let mut total = 0.0;
    for k in 0..t.len() - 1 {
        let (a, b) = (t[k], t[k + 1]);
        let lo = breaks.partition_point(|&v| v < a);
        let hi = breaks.partition_point(|&v| v <= b);
        let mut val = 0.0;
        if hi > lo + 1 {
            for w in breaks[lo..hi].windows(2) {
                let ip = dot(q.qk(k), p.pk(p.seg_of(w[0]))).max(0.0);
                val += (w[1] - w[0]) * ip * ip;
            }
        }
        total += (val * (q.t[k + 1] - q.t[k])).sqrt();
    }
    total
}

/// elasdics `compute_distance`: elastic distance between `p` (fixed) and `q`
/// warped by `t` (open, or closed with periodic extension of `p`).
pub(super) fn compute_distance(p_srv: &SrvPoly, q: &SrvPoly, t: &[f64], closed: bool) -> f64 {
    let cross = if closed {
        let ext = extend_periodic(p_srv);
        let p = Reference {
            d: p_srv.d,
            r: &ext.r,
            p: &ext.p,
        };
        loss_discrete(t, &p, q)
    } else {
        let p = Reference {
            d: p_srv.d,
            r: &p_srv.t,
            p: &p_srv.q,
        };
        loss_discrete(t, &p, q)
    };
    let d2 = p_srv.norm_sq() + q.norm_sq() - 2.0 * cross;
    if d2 <= 0.0 {
        0.0
    } else {
        d2.sqrt()
    }
}
