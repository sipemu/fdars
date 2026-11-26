//! Geometry primitives for R^d curve shape analysis.
//!
//! All curves and SRVFs are `FdMatrix` values of shape `d × m` (row = coordinate,
//! column = sample) on the implicit uniform grid `t_j = j/(m−1)` of `[0, 1]`.
//! L² inner products use trapezoidal weights on that grid.

use crate::alignment::{dp_edge_weight_core, dp_grid_solve, dp_lambda_penalty, dp_path_to_gamma};
use crate::error::FdarError;
use crate::helpers::linear_interp;
use crate::matrix::FdMatrix;

/// Uniform parameter grid on `[0, 1]` with trapezoidal quadrature weights.
#[derive(Debug, Clone)]
pub(super) struct Grid {
    /// Sample positions `t_j = j/(m−1)`.
    pub t: Vec<f64>,
    /// Trapezoidal weights (sum to 1).
    pub w: Vec<f64>,
    /// Grid spacing `1/(m−1)`.
    pub h: f64,
}

impl Grid {
    /// Build the grid for `m ≥ 2` samples.
    pub fn new(m: usize) -> Self {
        let h = 1.0 / (m - 1) as f64;
        let t: Vec<f64> = (0..m).map(|j| j as f64 * h).collect();
        let mut w = vec![h; m];
        w[0] = h / 2.0;
        w[m - 1] = h / 2.0;
        Self { t, w, h }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.t.len()
    }
}

/// `numpy.gradient`-style derivative along the sample axis (second-order central
/// differences inside, first-order one-sided differences at the two ends).
pub(super) fn gradient(beta: &FdMatrix, h: f64) -> FdMatrix {
    let (d, m) = beta.shape();
    let mut out = FdMatrix::zeros(d, m);
    for k in 0..d {
        out[(k, 0)] = (beta[(k, 1)] - beta[(k, 0)]) / h;
        out[(k, m - 1)] = (beta[(k, m - 1)] - beta[(k, m - 2)]) / h;
        for j in 1..m - 1 {
            out[(k, j)] = (beta[(k, j + 1)] - beta[(k, j - 1)]) / (2.0 * h);
        }
    }
    out
}

/// Euclidean norm of column `j` of a `d × m` matrix.
#[inline]
pub(super) fn col_norm(x: &FdMatrix, j: usize) -> f64 {
    x.column(j).iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Raw SRVF `q = β'/√‖β'‖` (no scaling, no closure projection).
pub(super) fn srvf_raw(beta: &FdMatrix, grid: &Grid) -> FdMatrix {
    let mut q = gradient(beta, grid.h);
    for j in 0..q.ncols() {
        let nv = col_norm(&q, j);
        let s = if nv > 1e-300 { 1.0 / nv.sqrt() } else { 0.0 };
        for v in q.column_mut(j) {
            *v *= s;
        }
    }
    q
}

/// Continuous lift of the polar angle of a planar (`2 × m`) vector field:
/// `θ_0 = atan2(v_0) ∈ (−π, π]` and `θ_j = θ_{j−1} + ∠(v_{j−1}, v_j)` with the
/// signed increment `∠ ∈ (−π, π]` (geomstats `FTransform._cartesian_to_polar`,
/// computed here with `atan2(cross, dot)` instead of `sign(det)·arccos(cos)`).
/// Zero vectors carry the previous angle forward.
pub(super) fn unwrapped_angle(v: &FdMatrix) -> Vec<f64> {
    let m = v.ncols();
    let mut theta = Vec::with_capacity(m);
    theta.push(v[(1, 0)].atan2(v[(0, 0)]));
    for j in 1..m {
        let (x0, y0, x1, y1) = (v[(0, j - 1)], v[(1, j - 1)], v[(0, j)], v[(1, j)]);
        let cross = x0 * y1 - y0 * x1;
        let dot = x0 * x1 + y0 * y1;
        let inc = if cross == 0.0 && dot == 0.0 {
            0.0
        } else {
            cross.atan2(dot)
        };
        theta.push(theta[j - 1] + inc);
    }
    theta
}

/// Kurtek–Needham F-transform of a planar velocity field:
/// `F = 2b |v|^{1/2} (cos λθ, sin λθ)`, `λ = a/(2b)`, `θ` the continuous lift of
/// [`unwrapped_angle`] (the `k = 0` branch of geomstats `FTransform`).
pub(super) fn f_from_velocity(v: &FdMatrix, a: f64, b: f64) -> FdMatrix {
    let m = v.ncols();
    let lam = a / (2.0 * b);
    let theta = unwrapped_angle(v);
    let mut f = FdMatrix::zeros(2, m);
    for (j, th) in theta.iter().enumerate() {
        let r = 2.0 * b * col_norm(v, j).sqrt();
        let (s, c) = (lam * th).sin_cos();
        f[(0, j)] = r * c;
        f[(1, j)] = r * s;
    }
    f
}

/// Velocity field of the inverse F-transform: speed `|F|²/(4b²)` along the
/// angle `φ/λ`, `φ` the continuous lift of `arg F` (geomstats `FTransform.inverse`).
pub(super) fn velocity_from_f(f: &FdMatrix, a: f64, b: f64) -> FdMatrix {
    let m = f.ncols();
    let inv_lam = 2.0 * b / a;
    let phi = unwrapped_angle(f);
    let mut v = FdMatrix::zeros(2, m);
    for (j, ph) in phi.iter().enumerate() {
        let nf = col_norm(f, j);
        let speed = nf * nf / (4.0 * b * b);
        let (s, c) = (inv_lam * ph).sin_cos();
        v[(0, j)] = speed * c;
        v[(1, j)] = speed * s;
    }
    v
}

/// F-transform of a planar `2 × m` curve (velocity by [`gradient`], as for the SRVF).
pub(super) fn f_transform(beta: &FdMatrix, grid: &Grid, a: f64, b: f64) -> FdMatrix {
    f_from_velocity(&gradient(beta, grid.h), a, b)
}

/// Integrate a sampled velocity field from the origin (cumulative trapezoid).
fn integrate_velocity(v: &FdMatrix, grid: &Grid) -> FdMatrix {
    let (d, m) = v.shape();
    let mut beta = FdMatrix::zeros(d, m);
    for j in 1..m {
        for k in 0..d {
            beta[(k, j)] = beta[(k, j - 1)] + 0.5 * grid.h * (v[(k, j - 1)] + v[(k, j)]);
        }
    }
    beta
}

/// Reconstruct a planar curve from its F-transform, starting at the origin.
pub(super) fn f_to_curve(f: &FdMatrix, grid: &Grid, a: f64, b: f64) -> FdMatrix {
    integrate_velocity(&velocity_from_f(f, a, b), grid)
}

/// L² inner product `∫ ⟨a(t), b(t)⟩ dt` (trapezoidal).
pub(super) fn inner(a: &FdMatrix, b: &FdMatrix, grid: &Grid) -> f64 {
    let d = a.nrows();
    let (sa, sb) = (a.as_slice(), b.as_slice());
    grid.w
        .iter()
        .enumerate()
        .map(|(j, &wj)| {
            let off = j * d;
            wj * (0..d).map(|k| sa[off + k] * sb[off + k]).sum::<f64>()
        })
        .sum()
}

/// L² norm.
pub(super) fn norm(a: &FdMatrix, grid: &Grid) -> f64 {
    inner(a, a, grid).max(0.0).sqrt()
}

/// `a·x + b·y` element-wise.
pub(super) fn axpby(a: f64, x: &FdMatrix, b: f64, y: &FdMatrix) -> FdMatrix {
    let data: Vec<f64> = x
        .as_slice()
        .iter()
        .zip(y.as_slice())
        .map(|(&xv, &yv)| a * xv + b * yv)
        .collect();
    FdMatrix::from_column_major(data, x.nrows(), x.ncols())
        .expect("dimension invariant: same shape as inputs")
}

/// `s·x`.
pub(super) fn scaled(x: &FdMatrix, s: f64) -> FdMatrix {
    let data: Vec<f64> = x.as_slice().iter().map(|&v| v * s).collect();
    FdMatrix::from_column_major(data, x.nrows(), x.ncols())
        .expect("dimension invariant: same shape as input")
}

/// Curve length `∫ ‖β'(t)‖ dt`.
pub(super) fn curve_length(beta: &FdMatrix, grid: &Grid) -> f64 {
    let v = gradient(beta, grid.h);
    (0..v.ncols()).map(|j| grid.w[j] * col_norm(&v, j)).sum()
}

/// Arc-length weighted centroid `∫ β ‖β'‖ / ∫ ‖β'‖` (fdasrsf `calculatecentroid`);
/// falls back to the plain sample mean for a degenerate (zero-length) curve.
pub(super) fn centroid(beta: &FdMatrix, grid: &Grid) -> Vec<f64> {
    let (d, m) = beta.shape();
    let v = gradient(beta, grid.h);
    let mut c = vec![0.0; d];
    let mut len = 0.0;
    for j in 0..m {
        let wj = grid.w[j] * col_norm(&v, j);
        len += wj;
        for (k, ck) in c.iter_mut().enumerate() {
            *ck += wj * beta[(k, j)];
        }
    }
    if len > 1e-300 {
        c.iter_mut().for_each(|ck| *ck /= len);
    } else {
        for (k, ck) in c.iter_mut().enumerate() {
            *ck = (0..m).map(|j| beta[(k, j)]).sum::<f64>() / m as f64;
        }
    }
    c
}

/// Translate the curve so that its arc-length centroid is at the origin.
pub(super) fn center(beta: &FdMatrix, grid: &Grid) -> FdMatrix {
    let c = centroid(beta, grid);
    let mut out = beta.clone();
    for j in 0..out.ncols() {
        for (v, ck) in out.column_mut(j).iter_mut().zip(&c) {
            *v -= ck;
        }
    }
    out
}

/// Apply a `d × d` matrix to every sample: `(R x)(t) = R x(t)`.
pub(super) fn rotate(r: &FdMatrix, x: &FdMatrix) -> FdMatrix {
    let (d, m) = x.shape();
    let mut out = FdMatrix::zeros(d, m);
    for j in 0..m {
        let xj = x.column(j);
        let oj = out.column_mut(j);
        for (a, o) in oj.iter_mut().enumerate() {
            *o = (0..d).map(|b| r[(a, b)] * xj[b]).sum();
        }
    }
    out
}

/// Identity `d × d` matrix.
pub(super) fn identity(d: usize) -> FdMatrix {
    let mut r = FdMatrix::zeros(d, d);
    for k in 0..d {
        r[(k, k)] = 1.0;
    }
    r
}

/// Optimal rotation `R = argmax_{R ∈ SO(d)} ⟨q1, R q2⟩` (Kabsch / orthogonal
/// Procrustes via the SVD of `A = ∫ q1 q2ᵀ`). With `allow_reflection` the
/// maximiser is taken over O(d) instead.
pub(super) fn procrustes(
    q1: &FdMatrix,
    q2: &FdMatrix,
    grid: &Grid,
    allow_reflection: bool,
) -> Result<FdMatrix, FdarError> {
    let (d, m) = q1.shape();
    let mut a = nalgebra::DMatrix::<f64>::zeros(d, d);
    for j in 0..m {
        let (c1, c2) = (q1.column(j), q2.column(j));
        for r in 0..d {
            for c in 0..d {
                a[(r, c)] += grid.w[j] * c1[r] * c2[c];
            }
        }
    }
    let svd = nalgebra::SVD::new(a, true, true);
    let (u, vt) = match (svd.u, svd.v_t) {
        (Some(u), Some(vt)) => (u, vt),
        _ => {
            return Err(FdarError::ComputationFailed {
                operation: "Procrustes SVD",
                detail: "SVD of the cross-covariance failed".to_string(),
            })
        }
    };
    let mut s = nalgebra::DMatrix::<f64>::identity(d, d);
    if !allow_reflection && (&u * &vt).determinant() < 0.0 {
        s[(d - 1, d - 1)] = -1.0;
    }
    Ok(FdMatrix::from_dmatrix(&(u * s * vt)))
}

/// Closure functional `G(q) = ∫ q(t) ‖q(t)‖ dt ∈ R^d` (zero iff the curve
/// reconstructed from `q` is closed).
pub(super) fn closure_vec(q: &FdMatrix, grid: &Grid) -> Vec<f64> {
    let (d, m) = q.shape();
    let mut g = vec![0.0; d];
    for j in 0..m {
        let wn = grid.w[j] * col_norm(q, j);
        for (k, gk) in g.iter_mut().enumerate() {
            *gk += wn * q[(k, j)];
        }
    }
    g
}

/// Normal fields `b_k = ‖q‖ e_k + q_k q/‖q‖` (L² gradients of the closure
/// constraints `G_k`), the d-dimensional form of fdasrsf `Basis_Normal_A`.
pub(super) fn closure_normals(q: &FdMatrix) -> Vec<FdMatrix> {
    let (d, m) = q.shape();
    (0..d)
        .map(|k| {
            let mut b = FdMatrix::zeros(d, m);
            for j in 0..m {
                let qn = col_norm(q, j);
                let qk_over = if qn > 1e-300 { q[(k, j)] / qn } else { 0.0 };
                for i in 0..d {
                    b[(i, j)] = qk_over * q[(i, j)];
                }
                b[(k, j)] += qn;
            }
            b
        })
        .collect()
}

/// Maximum number of damped Newton steps in [`project_closed`].
const CLOSURE_MAX_ITER: usize = 500;
/// Closure residual tolerance (on the unit-norm working copy).
pub(super) const CLOSURE_TOL: f64 = 1e-10;

/// Solve the small dense `d × d` system `J x = r` (Gaussian elimination with
/// partial pivoting); `None` if `J` is numerically singular.
fn solve_small(mut j: Vec<Vec<f64>>, mut r: Vec<f64>) -> Option<Vec<f64>> {
    let d = r.len();
    for col in 0..d {
        let piv = (col..d).max_by(|&a, &b| j[a][col].abs().total_cmp(&j[b][col].abs()))?;
        if j[piv][col].abs() < 1e-300 {
            return None;
        }
        j.swap(col, piv);
        r.swap(col, piv);
        for row in col + 1..d {
            let f = j[row][col] / j[col][col];
            for c in col..d {
                j[row][c] -= f * j[col][c];
            }
            r[row] -= f * r[col];
        }
    }
    let mut x = vec![0.0; d];
    for row in (0..d).rev() {
        let s: f64 = (row + 1..d).map(|c| j[row][c] * x[c]).sum();
        x[row] = (r[row] - s) / j[row][row];
    }
    Some(x)
}

/// One damped Newton step of the closure projection; returns the residual norm
/// *before* the step (the step is skipped once it is below tolerance).
fn closure_newton_step(q: &mut FdMatrix, grid: &Grid, dt: f64) -> f64 {
    let (d, m) = q.shape();
    let g = closure_vec(q, grid);
    let res = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    if res < CLOSURE_TOL {
        return res;
    }
    // Exact Jacobian of G along the normal fields: J_ik = ⟨b_i, b_k⟩ = ‖q‖² δ_ik + 3∫ q_i q_k.
    let qq = inner(q, q, grid);
    let mut jac = vec![vec![0.0; d]; d];
    for (i, row) in jac.iter_mut().enumerate() {
        for (k, cell) in row.iter_mut().enumerate() {
            let s: f64 = (0..m).map(|t| grid.w[t] * q[(i, t)] * q[(k, t)]).sum();
            *cell = 3.0 * s + if i == k { qq } else { 0.0 };
        }
    }
    let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
    let Some(x) = solve_small(jac, rhs) else {
        return res;
    };
    let normals = closure_normals(q);
    for (xk, bk) in x.iter().zip(&normals) {
        for (qv, bv) in q.as_mut_slice().iter_mut().zip(bk.as_slice()) {
            *qv += dt * xk * bv;
        }
    }
    res
}

/// Project an SRVF onto the closed-curve manifold `{q : ∫ q‖q‖ = 0}`
/// (Srivastava–Klassen; fdasrsf `project_curve`): damped Newton iterations along
/// the closure normals, step 0.35 for d = 2 and 0.2 otherwise (fdasrsf values),
/// with the exact Jacobian. The L² norm of the input is preserved.
///
/// Returns the projected SRVF and whether the residual reached [`CLOSURE_TOL`].
pub(super) fn project_closed(q: &FdMatrix, grid: &Grid) -> (FdMatrix, bool) {
    let n0 = norm(q, grid);
    if n0 < 1e-300 {
        return (q.clone(), true);
    }
    let dt = if q.nrows() == 2 { 0.35 } else { 0.2 };
    let mut qn = scaled(q, 1.0 / n0);
    let mut converged = false;
    for _ in 0..CLOSURE_MAX_ITER {
        if closure_newton_step(&mut qn, grid, dt) < CLOSURE_TOL {
            converged = true;
            break;
        }
    }
    let n1 = norm(&qn, grid);
    (scaled(&qn, n0 / n1), converged)
}

/// Project a vector field `v` onto the tangent space at `q`.
///
/// * `sphere` — remove the radial component `⟨v, q⟩ q/‖q‖²` (pre-shape sphere).
/// * `closed` — remove the closure normals (Gram–Schmidt orthonormalised).
pub(super) fn project_tangent(
    v: &FdMatrix,
    q: &FdMatrix,
    grid: &Grid,
    sphere: bool,
    closed: bool,
) -> FdMatrix {
    let mut basis: Vec<FdMatrix> = Vec::new();
    if sphere {
        basis.push(q.clone());
    }
    if closed {
        basis.extend(closure_normals(q));
    }
    // Gram–Schmidt (modified) on the normal directions, then remove them from v.
    let mut ortho: Vec<FdMatrix> = Vec::with_capacity(basis.len());
    for b in basis {
        let mut e = b;
        for o in &ortho {
            let c = inner(&e, o, grid);
            e = axpby(1.0, &e, -c, o);
        }
        let ne = norm(&e, grid);
        if ne > 1e-12 {
            ortho.push(scaled(&e, 1.0 / ne));
        }
    }
    let mut w = v.clone();
    for o in &ortho {
        let c = inner(&w, o, grid);
        w = axpby(1.0, &w, -c, o);
    }
    w
}

/// Reconstruct a curve from its SRVF: `β(t) = ∫₀ᵗ q(s)‖q(s)‖ ds` (cumulative
/// trapezoid, `β(0) = 0`).
pub(super) fn q_to_curve(q: &FdMatrix, grid: &Grid) -> FdMatrix {
    let (d, m) = q.shape();
    let mut beta = FdMatrix::zeros(d, m);
    let norms: Vec<f64> = (0..m).map(|j| col_norm(q, j)).collect();
    for j in 1..m {
        for k in 0..d {
            let a = q[(k, j - 1)] * norms[j - 1];
            let b = q[(k, j)] * norms[j];
            beta[(k, j)] = beta[(k, j - 1)] + 0.5 * grid.h * (a + b);
        }
    }
    beta
}

/// Change the starting point of a closed curve sampled with `β(m−1) = β(0)`:
/// circularly shift the first `m − 1` samples by `tau` and re-close
/// (fdasrsf `shift_f`, sign convention: new sample `j` is old sample `j − tau`).
pub(super) fn shift_closed(beta: &FdMatrix, tau: usize) -> FdMatrix {
    let (d, m) = beta.shape();
    let p = m - 1;
    let mut out = FdMatrix::zeros(d, m);
    for j in 0..p {
        let src = (j + p - tau % p) % p;
        for k in 0..d {
            out[(k, j)] = beta[(k, src)];
        }
    }
    for k in 0..d {
        out[(k, p)] = out[(k, 0)];
    }
    out
}

/// Reparametrise a curve, `(β∘γ)(t)`, by linear interpolation per coordinate.
pub(super) fn warp_curve(beta: &FdMatrix, grid: &Grid, gamma: &[f64]) -> FdMatrix {
    let (d, m) = beta.shape();
    let mut out = FdMatrix::zeros(d, m);
    for k in 0..d {
        let row = beta.row(k);
        for (j, &g) in gamma.iter().enumerate() {
            out[(k, j)] = linear_interp(&grid.t, &row, g);
        }
    }
    out
}

/// R^d dynamic-programming reparametrisation: the warp `γ` minimising
/// `‖q1 − (q2∘γ)√γ'‖²` (the shared fdasrvf coprime-neighbourhood DP engine with
/// the edge weight summed over coordinates — no per-coordinate renormalisation,
/// so the R^d Euclidean geometry of the SRVF is preserved).
pub(super) fn dp_warp(q1: &FdMatrix, q2: &FdMatrix, grid: &Grid, lambda: f64) -> Vec<f64> {
    let (d, m) = q1.shape();
    let r1: Vec<Vec<f64>> = (0..d).map(|k| q1.row(k)).collect();
    let r2: Vec<Vec<f64>> = (0..d).map(|k| q2.row(k)).collect();
    let mut rslope = [[0.0f64; 8]; 8];
    for (dr, row) in rslope.iter_mut().enumerate().skip(1) {
        for (dc, cell) in row.iter_mut().enumerate().skip(1) {
            *cell = (dr as f64 / dc as f64).sqrt();
        }
    }
    let h = grid.h;
    let t = &grid.t;
    let path = dp_grid_solve(m, m, |sr, sc, tr, tc| {
        let (dr, dc) = (tr - sr, tc - sc);
        let rs = rslope[dr][dc];
        let span = dc as f64 * h;
        let w: f64 = (0..d)
            .map(|k| dp_edge_weight_core(&r1[k], &r2[k], sc, tc, sr, tr, rs, span))
            .sum();
        w + dp_lambda_penalty(t, sc, tc, sr, tr, lambda)
    });
    dp_path_to_gamma(&path, t)
}
