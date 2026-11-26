//! Polygon / piecewise-constant SRV representation of a sparsely observed curve
//! (elasdics `get_srv_from_points`, `get_arc_length_param`, `remove_duplicate`,
//! `check_closed`, `get_points_from_srv`).

use super::SparseCurve;
use crate::error::FdarError;

/// A curve observed at `m` points in R^d, stored point-major (`pts[j*d..(j+1)*d]`
/// is point `j`), with its parametrisation `t` (length `m`, `t[0] = 0`).
#[derive(Debug, Clone)]
pub(super) struct Polygon {
    pub d: usize,
    pub pts: Vec<f64>,
    pub t: Vec<f64>,
    /// Whether duplicated consecutive points were removed.
    pub removed: bool,
}

impl Polygon {
    pub fn m(&self) -> usize {
        self.t.len()
    }
}

/// Piecewise-constant SRV: `q[k*d..(k+1)*d]` is the SRV on `[t[k], t[k+1])`;
/// `t` has one more entry than there are SRV vectors and ends at 1.
#[derive(Debug, Clone)]
pub(super) struct SrvPoly {
    pub d: usize,
    pub t: Vec<f64>,
    pub q: Vec<f64>,
}

impl SrvPoly {
    pub fn nseg(&self) -> usize {
        self.t.len() - 1
    }

    pub fn qk(&self, k: usize) -> &[f64] {
        &self.q[k * self.d..(k + 1) * self.d]
    }

    /// `∫ |q|²` of the piecewise-constant SRV.
    pub fn norm_sq(&self) -> f64 {
        (0..self.nseg())
            .map(|k| sq(self.qk(k)) * (self.t[k + 1] - self.t[k]))
            .sum()
    }
}

#[inline]
pub(super) fn sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

#[inline]
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Add `f · d(v|v|)/dv = f · (|v| I + v vᵀ/|v|)` (with `nrm = |v| > 0`) to the
/// row-major `d × nc` Jacobian `jac`, columns `col0..col0 + d`.
pub(super) fn add_norm_jacobian(
    jac: &mut [f64],
    nc: usize,
    col0: usize,
    f: f64,
    v: &[f64],
    nrm: f64,
) {
    for (a, va) in v.iter().enumerate() {
        let row = &mut jac[a * nc + col0..a * nc + col0 + v.len()];
        for (b, (cell, vb)) in row.iter_mut().zip(v).enumerate() {
            let delta = if a == b { nrm } else { 0.0 };
            *cell += f * (delta + va * vb / nrm);
        }
    }
}

/// Validate a user-supplied parametrisation (elasdics `check_param`).
fn check_param(t: &[f64], m: usize, closed: bool) -> Result<(), FdarError> {
    if t.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "param",
            expected: format!("{m} values (one per point)"),
            actual: t.len().to_string(),
        });
    }
    let ok_range = t.iter().all(|v| v.is_finite()) && t[0] >= 0.0 && t[m - 1] <= 1.0;
    let ok_mono = t.windows(2).all(|w| w[1] >= w[0]);
    if !(ok_range && ok_mono) {
        return Err(FdarError::InvalidParameter {
            parameter: "param",
            message: "parametrisation must lie in [0, 1] and be non-decreasing".to_string(),
        });
    }
    if t[0] != 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "param",
            message: "parametrisation must start at 0".to_string(),
        });
    }
    if !closed && t[m - 1] != 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "param",
            message: "parametrisation of an open curve must end at 1".to_string(),
        });
    }
    Ok(())
}

/// Remove consecutive points whose squared step is at most `max|x| · ε`
/// (elasdics `remove_duplicate`; elasdics uses `max(x)`, which is the same for
/// curves with a positive coordinate and avoids a non-positive threshold otherwise).
fn remove_duplicates(d: usize, pts: &[f64], t: Option<&[f64]>, closed: bool) -> Polygon {
    let m = pts.len() / d;
    let thr = pts.iter().fold(0.0_f64, |a, v| a.max(v.abs())) * f64::EPSILON;
    let mut keep = vec![true; m];
    for j in 1..m {
        let step: f64 = (0..d)
            .map(|c| (pts[j * d + c] - pts[(j - 1) * d + c]).powi(2))
            .sum();
        keep[j] = step > thr;
    }
    let removed = keep.iter().any(|k| !k);
    let mut out_pts = Vec::with_capacity(pts.len());
    let mut out_t = Vec::with_capacity(m);
    for j in (0..m).filter(|&j| keep[j]) {
        out_pts.extend_from_slice(&pts[j * d..(j + 1) * d]);
        if let Some(t) = t {
            out_t.push(t[j]);
        }
    }
    if t.is_some() && !closed && removed {
        if let Some(last) = out_t.last_mut() {
            *last = 1.0;
        }
    }
    Polygon {
        d,
        pts: out_pts,
        t: out_t,
        removed,
    }
}

/// Close a curve by appending its first point when needed (elasdics `check_closed`).
fn close_polygon(poly: &mut Polygon, has_param: bool) -> Result<(), FdarError> {
    let d = poly.d;
    let m = poly.m_pts();
    let first: Vec<f64> = poly.pts[..d].to_vec();
    let last_idx = (m - 1) * d;
    if has_param {
        if *poly.t.last().unwrap_or(&0.0) == 1.0 {
            if poly.pts[last_idx..] != first[..] {
                return Err(FdarError::InvalidParameter {
                    parameter: "curve",
                    message: "parametrisation ends at 1 but the curve is not closed".to_string(),
                });
            }
        } else {
            poly.pts.extend_from_slice(&first);
            poly.t.push(1.0);
        }
    } else {
        let gap: f64 = (0..d)
            .map(|c| (first[c] - poly.pts[last_idx + c]).powi(2))
            .sum();
        let err = gap / sq(&first);
        if err > f64::EPSILON.sqrt() {
            poly.pts.extend_from_slice(&first);
        } else {
            poly.pts[last_idx..].copy_from_slice(&first);
        }
    }
    Ok(())
}

impl Polygon {
    fn m_pts(&self) -> usize {
        self.pts.len() / self.d
    }
}

/// Arc-length parametrisation of point-major points (elasdics `get_arc_length_param`).
pub(super) fn arc_length_param(d: usize, pts: &[f64]) -> Vec<f64> {
    let m = pts.len() / d;
    let mut t = Vec::with_capacity(m);
    t.push(0.0);
    for j in 1..m {
        let step: f64 = (0..d)
            .map(|c| (pts[j * d + c] - pts[(j - 1) * d + c]).powi(2))
            .sum::<f64>()
            .sqrt();
        t.push(t[j - 1] + step);
    }
    let total = t[m - 1];
    for v in &mut t {
        *v /= total;
    }
    t
}

/// Validate and normalise a [`SparseCurve`]: duplicate removal, closure, and
/// parametrisation (given or arc length).
pub(super) fn prepare_curve(
    curve: &SparseCurve,
    closed: bool,
    name: &'static str,
) -> Result<Polygon, FdarError> {
    let (d, m) = curve.points.shape();
    if d == 0 || m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: name,
            expected: "d >= 1 coordinates and at least 2 points".to_string(),
            actual: format!("{d} x {m}"),
        });
    }
    let pts = curve.points.as_slice();
    if pts.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: name,
            message: "curve points must be finite".to_string(),
        });
    }
    if let Some(t) = &curve.param {
        check_param(t, m, closed)?;
    }
    let mut poly = remove_duplicates(d, pts, curve.param.as_deref(), closed);
    if poly.m_pts() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: name,
            message: "curve has fewer than 2 distinct points".to_string(),
        });
    }
    if closed {
        close_polygon(&mut poly, curve.param.is_some())?;
    }
    if curve.param.is_none() {
        poly.t = arc_length_param(d, &poly.pts);
    }
    Ok(poly)
}

/// Piecewise-constant SRV of a polygon (elasdics `get_srv_from_points`).
pub(super) fn srv_from_polygon(poly: &Polygon, name: &'static str) -> Result<SrvPoly, FdarError> {
    let (d, m) = (poly.d, poly.m());
    let t = &poly.t;
    if t[0] != 0.0 || t[m - 1] != 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: name,
            message: "parametrisation must start at 0 and end at 1".to_string(),
        });
    }
    if t.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: name,
            message: "parametrisation must be strictly increasing".to_string(),
        });
    }
    Ok(srv_from_points_unchecked(d, &poly.pts, t))
}

/// SRV of points on an increasing grid `t`, without validation; a segment whose
/// factor `1/(√Δt·|Δβ|^{1/2})` is non-finite gets factor 0 (non-finite steps stay
/// non-finite, as in elasdics).
pub(super) fn srv_from_points_unchecked(d: usize, pts: &[f64], t: &[f64]) -> SrvPoly {
    let m = t.len();
    let mut q = Vec::with_capacity((m - 1) * d);
    for k in 0..m - 1 {
        let diff: Vec<f64> = (0..d)
            .map(|c| pts[(k + 1) * d + c] - pts[k * d + c])
            .collect();
        let factor = 1.0 / ((t[k + 1] - t[k]).sqrt() * sq(&diff).powf(0.25));
        let factor = if factor.is_finite() { factor } else { 0.0 };
        q.extend(diff.iter().map(|v| factor * v));
    }
    SrvPoly {
        d,
        t: t.to_vec(),
        q,
    }
}

/// Points of the polygon with SRV `srv` (first point at the origin)
/// (elasdics `get_points_from_srv`); point-major, length `(nseg+1)·d`.
pub(super) fn points_from_srv(srv: &SrvPoly) -> Vec<f64> {
    let d = srv.d;
    let mut pts = vec![0.0; (srv.nseg() + 1) * d];
    for k in 0..srv.nseg() {
        let qk = srv.qk(k);
        let w = sq(qk).sqrt() * (srv.t[k + 1] - srv.t[k]);
        for c in 0..d {
            pts[(k + 1) * d + c] = pts[k * d + c] + qk[c] * w;
        }
    }
    pts
}

/// Linear interpolation of point-major `pts` on parametrisation `t` (possibly with
/// ties) at `x` (elasdics `get_evals.data.frame`: `findInterval(rightmost.closed)`).
/// A zero-length bracketing interval yields NaN, as in elasdics.
pub(super) fn eval_polygon(d: usize, pts: &[f64], t: &[f64], x: f64, out: &mut [f64]) {
    let n = t.len();
    // findInterval: largest idx with t[idx] <= x; rightmost.closed maps x == t[n-1]
    // to the last interval.
    let mut idx = t.partition_point(|&v| v <= x).saturating_sub(1);
    if idx >= n - 1 {
        idx = n - 2;
    }
    let (a, b) = (t[idx], t[idx + 1]);
    let w0 = (b - x) / (b - a);
    let w1 = (x - a) / (b - a);
    for c in 0..d {
        out[c] = w0 * pts[idx * d + c] + w1 * pts[(idx + 1) * d + c];
    }
}
