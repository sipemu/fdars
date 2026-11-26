//! Exact `(a, b)` elastic shape distance between piecewise-linear curves.
//!
//! Bauer, Charon, Klassen, Kurtek, Needham & Pierron, "Elastic metrics on
//! spaces of Euclidean curves: theory and algorithms", J. Nonlinear Sci. 34, 56
//! (2024), arXiv:2209.09862 — Corollary 2.1 / Theorem 3.1 (distance formula and
//! existence of optimal generalised reparametrisations) and §4.1 (exact
//! algorithm for PL curves), which adapts the precise matching of Lahiri,
//! Robinson & Klassen, "Precise matching of PL curves in R^N in the square root
//! velocity framework", arXiv:1501.00577 (2015), Theorem 6 and §8.
//!
//! In arc-length parametrisation the quotient distance is
//! `d² = 4b² (L1 + L2 − 2 S)`, `S = sup ∫ κ(γ1, γ2) √(γ1'γ2') du` over monotone
//! matchings `(γ1, γ2)` of `[0, L1] × [0, L2]`, with the *rectangular* weight
//! `κ_ij = cos(λ θ_ij)` if `λ θ_ij < π/2` and `0` otherwise on the cell
//! (segment `i` of `c1`) × (segment `j` of `c2`), `θ_ij ∈ [0, π]` the angle
//! between the two segments and `λ = a/(2b)`.
//!
//! An optimal matching is a chain of *N-segments* (horizontal / vertical moves
//! along grid lines, value 0) and *P-segments*: vertex-to-vertex paths that are
//! straight inside positive cells, refract at cell edges (`H'/H = (κ'/κ)²`
//! across a vertical edge, `(κ/κ')²` across a horizontal one) and cross zero
//! cells horizontally / vertically. A P-segment is determined by its start
//! vertex and its initial slope `h`. Rescaling the visited columns by `α` and
//! rows by `β` with `α β = (κ/κ0)²` ("path straightening") turns the ray into
//! the straight line `Y = hX` (zero cells collapse) with value `κ0 √(XY)`; the
//! P-segments from a vertex are therefore found by sweeping `h` upwards from 0,
//! jumping each time to the smallest corner ratio `Y/X` above the current
//! slope. A longest-path dynamic program over the grid vertices then gives the
//! exact supremum in `O(N1² N2² (N1 + N2))` operations.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Exact `(a, b)` elastic matching of two piecewise-linear curves (see
/// [`ab_elastic_match_pl`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AbPlMatching {
    /// Quotient distance `2b √(L1 + L2 − 2S)` (modulo translation and
    /// reparametrisation; no rotation or scale quotient).
    pub distance: f64,
    /// Optimal matching value `S = sup ∫ κ √(γ1'γ2')` (arc-length units).
    pub similarity: f64,
    /// Length `L1` of `c1`.
    pub length1: f64,
    /// Length `L2` of `c2`.
    pub length2: f64,
    /// Optimal (generalised) matching as a monotone polyline from `(0, 0)` to
    /// `(1, 1)` in normalised arc length: point `(s1, s2)` matches
    /// `c1` at arc length `s1·L1` with `c2` at arc length `s2·L2`. Horizontal /
    /// vertical pieces collapse a part of one curve onto a point of the other.
    pub path: Vec<(f64, f64)>,
}

/// Matching grid of two polygons in arc-length coordinates.
struct PlGrid {
    /// Segment lengths of `c1` (columns).
    l1: Vec<f64>,
    /// Segment lengths of `c2` (rows).
    l2: Vec<f64>,
    /// Cumulative arc length of `c1` at its vertices (`N1 + 1`).
    s: Vec<f64>,
    /// Cumulative arc length of `c2` at its vertices (`N2 + 1`).
    t: Vec<f64>,
    /// Cell weights `κ_ij` at `i + N1·j`.
    kappa: Vec<f64>,
}

impl PlGrid {
    fn n1(&self) -> usize {
        self.l1.len()
    }
    fn n2(&self) -> usize {
        self.l2.len()
    }
    fn kap(&self, i: usize, j: usize) -> f64 {
        self.kappa[i + self.n1() * j]
    }
    /// Vertex index of `(i, j)`, `i ≤ N1`, `j ≤ N2`.
    fn vid(&self, i: usize, j: usize) -> usize {
        i + (self.n1() + 1) * j
    }
}

/// Non-degenerate segment vectors of a `d × n` polygon (zero-length segments,
/// i.e. repeated vertices, are dropped: they do not change the curve).
fn segments(c: &FdMatrix, parameter: &'static str) -> Result<Vec<Vec<f64>>, FdarError> {
    let (d, n) = c.shape();
    if d == 0 || n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: "d x n polygon with d >= 1 and n >= 2 vertices".to_string(),
            actual: format!("{d} x {n}"),
        });
    }
    if c.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter,
            message: "contains non-finite values".to_string(),
        });
    }
    let segs: Vec<Vec<f64>> = (1..n)
        .map(|j| {
            (0..d)
                .map(|k| c[(k, j)] - c[(k, j - 1)])
                .collect::<Vec<f64>>()
        })
        .filter(|u| u.iter().any(|&v| v != 0.0))
        .collect();
    if segs.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "ab_elastic_match_pl",
            detail: format!("{parameter} has zero length"),
        });
    }
    Ok(segs)
}

fn norm(u: &[f64]) -> f64 {
    u.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Cell weight `cos(λθ)` if `λθ < π/2`, else 0 (`θ ∈ [0, π]` the angle of `u, w`).
fn cell_weight(u: &[f64], w: &[f64], lam: f64) -> f64 {
    let (nu, nw) = (norm(u), norm(w));
    let dot: f64 = u.iter().zip(w).map(|(a, b)| a * b).sum();
    let cross2 = (nu * nu * nw * nw - dot * dot).max(0.0);
    let theta = cross2.sqrt().atan2(dot);
    let x = lam * theta;
    if x < std::f64::consts::FRAC_PI_2 {
        x.cos()
    } else {
        0.0
    }
}

fn cumulative(l: &[f64]) -> Vec<f64> {
    let mut s = Vec::with_capacity(l.len() + 1);
    s.push(0.0);
    for &v in l {
        s.push(s[s.len() - 1] + v);
    }
    s
}

impl PlGrid {
    fn new(c1: &FdMatrix, c2: &FdMatrix, lam: f64) -> Result<Self, FdarError> {
        let s1 = segments(c1, "c1")?;
        let s2 = segments(c2, "c2")?;
        let l1: Vec<f64> = s1.iter().map(|u| norm(u)).collect();
        let l2: Vec<f64> = s2.iter().map(|u| norm(u)).collect();
        let mut kappa = Vec::with_capacity(l1.len() * l2.len());
        for w in &s2 {
            for u in &s1 {
                kappa.push(cell_weight(u, w, lam));
            }
        }
        Ok(Self {
            s: cumulative(&l1),
            t: cumulative(&l2),
            l1,
            l2,
            kappa,
        })
    }
}

/// A grid vertex reached by a P-segment ray.
#[derive(Debug, Clone, Copy)]
struct Corner {
    /// Vertex column / row index.
    i: usize,
    j: usize,
    /// Initial slope (straightened) at which the ray hits the vertex.
    ratio: f64,
    /// P-segment value `κ0 √(XY)`.
    value: f64,
}

/// What a ray trace is asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TraceMode {
    /// Pass every corner with ratio `≤ h` above (slope just above `h`) and
    /// report the next event: the first corner with the smallest ratio `> h`.
    Sweep,
    /// Stop at the first corner with ratio exactly `h` and report the path.
    Hit,
}

/// Result of [`trace`].
struct TraceOut {
    corner: Option<Corner>,
    /// Physical path points (Hit mode only), starting at the start vertex.
    path: Vec<(f64, f64)>,
}

/// Straightened state of a ray: current cell, straightened origin of the
/// current column / row and their scales.
struct Ray {
    c: usize,
    r: usize,
    xc: f64,
    yr: f64,
    al: f64,
    be: f64,
    from_left: bool,
    k0: f64,
}

impl Ray {
    /// Move into the next column (`c + 1`); `false` at the right boundary.
    fn step_right(&mut self, g: &PlGrid, xend: f64) -> bool {
        self.c += 1;
        if self.c == g.n1() {
            return false;
        }
        self.xc = xend;
        let k = g.kap(self.c, self.r);
        self.al = if k > 0.0 {
            (k / self.k0).powi(2) / self.be
        } else {
            0.0
        };
        self.from_left = true;
        true
    }

    /// Move into the next row (`r + 1`); `false` at the top boundary.
    fn step_up(&mut self, g: &PlGrid, yend: f64) -> bool {
        self.r += 1;
        if self.r == g.n2() {
            return false;
        }
        self.yr = yend;
        let k = g.kap(self.c, self.r);
        self.be = if k > 0.0 {
            (k / self.k0).powi(2) / self.al
        } else {
            0.0
        };
        self.from_left = false;
        true
    }
}

/// Physical exit point of the current cell for a ray of straightened slope `h`.
fn exit_point(g: &PlGrid, ray: &Ray, h: f64, right: bool, cur: (f64, f64)) -> (f64, f64) {
    let (c, r) = (ray.c, ray.r);
    let positive = g.kap(c, r) > 0.0;
    if right {
        let y = if positive {
            (g.t[r] + (h * (ray.xc + ray.al * g.l1[c]) - ray.yr) / ray.be).clamp(g.t[r], g.t[r + 1])
        } else {
            cur.1
        };
        (g.s[c + 1], y)
    } else {
        let x = if positive {
            let yend = ray.yr + ray.be * g.l2[r];
            (g.s[c] + (yend / h - ray.xc) / ray.al).clamp(g.s[c], g.s[c + 1])
        } else {
            cur.0
        };
        (x, g.t[r + 1])
    }
}

/// Decision of a ray in its current cell.
enum CellStep {
    Right,
    Up,
    /// Hit-mode stop at this corner.
    Hit(Corner),
}

/// Decide how the ray leaves its current cell (`xend`, `yend` the
/// straightened far edges); in Sweep mode record the corner as an event
/// candidate in `best`.
fn cell_step(
    g: &PlGrid,
    ray: &Ray,
    (xend, yend): (f64, f64),
    h: f64,
    mode: TraceMode,
    best: &mut Option<Corner>,
) -> CellStep {
    if g.kap(ray.c, ray.r) <= 0.0 {
        return if ray.from_left {
            CellStep::Right
        } else {
            CellStep::Up
        };
    }
    let ratio = yend / xend;
    let corner = Corner {
        i: ray.c + 1,
        j: ray.r + 1,
        ratio,
        value: ray.k0 * (xend * yend).sqrt(),
    };
    if mode == TraceMode::Hit && ratio == h {
        return CellStep::Hit(corner);
    }
    if mode == TraceMode::Sweep && ratio > h && best.map_or(true, |b| ratio < b.ratio) {
        *best = Some(corner);
    }
    if ratio > h {
        CellStep::Right
    } else {
        CellStep::Up
    }
}

/// Trace the P-segment ray of straightened slope `h` from vertex `(i0, j0)`
/// into the positive cell `(i0, j0)`.
fn trace(g: &PlGrid, i0: usize, j0: usize, h: f64, mode: TraceMode) -> TraceOut {
    let mut ray = Ray {
        c: i0,
        r: j0,
        xc: 0.0,
        yr: 0.0,
        al: 1.0,
        be: 1.0,
        from_left: true,
        k0: g.kap(i0, j0),
    };
    let mut best: Option<Corner> = None;
    let record = mode == TraceMode::Hit;
    let mut cur = (g.s[i0], g.t[j0]);
    let mut path = if record { vec![cur] } else { Vec::new() };
    loop {
        let xend = ray.xc + ray.al * g.l1[ray.c];
        let yend = ray.yr + ray.be * g.l2[ray.r];
        let right = match cell_step(g, &ray, (xend, yend), h, mode, &mut best) {
            CellStep::Hit(corner) => {
                path.push((g.s[corner.i], g.t[corner.j]));
                return TraceOut {
                    corner: Some(corner),
                    path,
                };
            }
            CellStep::Right => true,
            CellStep::Up => false,
        };
        if record {
            cur = exit_point(g, &ray, h, right, cur);
            path.push(cur);
        }
        let inside = if right {
            ray.step_right(g, xend)
        } else {
            ray.step_up(g, yend)
        };
        if !inside {
            break;
        }
    }
    TraceOut {
        corner: if record { None } else { best },
        path,
    }
}

/// How the best path reaches a vertex.
#[derive(Debug, Clone, Copy)]
struct Pred {
    from: (usize, usize),
    /// Initial slope of the P-segment (`None` for an edge move).
    h: Option<f64>,
}

/// Longest-path dynamic program over the grid vertices.
struct VertexDp {
    best: Vec<f64>,
    pred: Vec<Option<Pred>>,
}

impl VertexDp {
    fn relax(&mut self, v: usize, val: f64, p: Pred) {
        if val > self.best[v] {
            self.best[v] = val;
            self.pred[v] = Some(p);
        }
    }
}

/// Relax every P-segment leaving vertex `(i, j)` (whose NE cell is positive).
fn sweep_p_segments(g: &PlGrid, dp: &mut VertexDp, i: usize, j: usize) {
    let base = dp.best[g.vid(i, j)];
    let cap = 2 * (g.n1() + 1) * (g.n2() + 1) + 8;
    let mut h = 0.0;
    for _ in 0..cap {
        let Some(cn) = trace(g, i, j, h, TraceMode::Sweep).corner else {
            break;
        };
        let pred = Pred {
            from: (i, j),
            h: Some(cn.ratio),
        };
        dp.relax(g.vid(cn.i, cn.j), base + cn.value, pred);
        h = cn.ratio;
    }
}

/// Relax the edge moves and P-segments leaving vertex `(i, j)`.
fn relax_from(g: &PlGrid, dp: &mut VertexDp, i: usize, j: usize) {
    let val = dp.best[g.vid(i, j)];
    if val == f64::NEG_INFINITY {
        return;
    }
    let (n1, n2) = (g.n1(), g.n2());
    let edge = Pred {
        from: (i, j),
        h: None,
    };
    if i < n1 {
        dp.relax(g.vid(i + 1, j), val, edge);
    }
    if j < n2 {
        dp.relax(g.vid(i, j + 1), val, edge);
    }
    if i < n1 && j < n2 && g.kap(i, j) > 0.0 {
        sweep_p_segments(g, dp, i, j);
    }
}

fn solve(g: &PlGrid) -> VertexDp {
    let nv = (g.n1() + 1) * (g.n2() + 1);
    let mut dp = VertexDp {
        best: vec![f64::NEG_INFINITY; nv],
        pred: vec![None; nv],
    };
    dp.best[0] = 0.0;
    // row-major order: every predecessor of a vertex is processed before it
    for j in 0..=g.n2() {
        for i in 0..=g.n1() {
            relax_from(g, &mut dp, i, j);
        }
    }
    dp
}

/// Backtrack the optimal matching polyline (normalised arc length).
fn backtrack(g: &PlGrid, dp: &VertexDp) -> Vec<(f64, f64)> {
    let (n1, n2) = (g.n1(), g.n2());
    let mut pieces: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut v = (n1, n2);
    while v != (0, 0) {
        let Some(p) = dp.pred[g.vid(v.0, v.1)] else {
            break;
        };
        let piece = match p.h {
            Some(h) => trace(g, p.from.0, p.from.1, h, TraceMode::Hit).path,
            None => vec![(g.s[p.from.0], g.t[p.from.1]), (g.s[v.0], g.t[v.1])],
        };
        pieces.push(piece);
        v = p.from;
    }
    let (l1, l2) = (g.s[n1], g.t[n2]);
    let mut path: Vec<(f64, f64)> = vec![(0.0, 0.0)];
    for piece in pieces.iter().rev() {
        for &(x, y) in piece {
            let pt = ((x / l1).clamp(0.0, 1.0), (y / l2).clamp(0.0, 1.0));
            if path.last() != Some(&pt) {
                path.push(pt);
            }
        }
    }
    path
}

/// Exact `(a, b)` elastic matching of two piecewise-linear curves in R^d
/// (Bauer et al. 2024, §4.1): the optimal generalised reparametrisation pair
/// and the quotient distance
/// `d = 2b √(L1 + L2 − 2 sup ∫ κ(γ1, γ2) √(γ1'γ2'))`, computed exactly (no
/// discretisation of the matching) by the Lahiri–Robinson–Klassen structure
/// theorem — see the module notes of `pl_exact` in the source.
///
/// `c1`, `c2` are `d × n1`, `d × n2` vertex matrices (column = vertex, any
/// `d ≥ 1`, the polygons need not have the same number of vertices;
/// repeated vertices are ignored). The distance is the elastic shape distance
/// modulo translation and reparametrisation — **not** rotation or scale
/// (`CurveShapeConfig { rotation: false, scale: false, .. }` of
/// [`curve_shape_distance`](super::curve_shape_distance), whose DP value it
/// lower-bounds when the tangent-angle differences stay below `π/(2λ)`; see
/// below). The pointwise angle is `θ ∈ [0, π]` between the two tangents: for
/// `d ≥ 3` this is the global geodesic distance of Bauer et al. Theorem 2.1;
/// for planar curves it is the distance of the geodesic completion
/// (Corollary 2.1), which coincides with the F-transform distance as long as
/// the lifted tangent-angle difference stays in `[−π/(2λ), π/(2λ)]` along the
/// matching.
///
/// Cost: `O(N1² N2² (N1 + N2))` for `N1`, `N2` segments (polynomial, but
/// meant for polygons with tens of vertices, not densely sampled curves —
/// use [`curve_shape_distance`](super::curve_shape_distance) for those).
///
/// # Errors
/// [`FdarError::InvalidDimension`] if a polygon has fewer than 2 vertices or
/// the dimensions differ; [`FdarError::InvalidParameter`] for non-finite
/// vertices or `a, b` not finite and positive;
/// [`FdarError::ComputationFailed`] for a zero-length polygon.
///
/// # Example
/// ```
/// use fdars_core::alignment::ab_elastic_match_pl;
/// use fdars_core::matrix::FdMatrix;
/// // c1: one unit segment; c2: two unit segments at ±0.3 rad around it
/// let c1 = FdMatrix::from_column_major(vec![0.0, 0.0, 1.0, 0.0], 2, 2).unwrap();
/// let (s, c) = 0.3f64.sin_cos();
/// let c2 = FdMatrix::from_column_major(vec![0.0, 0.0, c, s, 2.0 * c, 0.0], 2, 3).unwrap();
/// let (a, b) = (1.0, 1.0); // λ = 1/2
/// let res = ab_elastic_match_pl(&c1, &c2, a, b).unwrap();
/// // closed form for a one-segment target: S = √(L1 Σ κ_j² ℓ_j), κ_j = cos(λθ_j)
/// let k = (0.5f64 * 0.3).cos();
/// let s_exact = (1.0 * 2.0 * k * k).sqrt();
/// let d_exact = 2.0 * b * (1.0 + 2.0 - 2.0 * s_exact).sqrt();
/// assert!((res.distance - d_exact).abs() < 1e-12);
/// assert_eq!(res.path.first(), Some(&(0.0, 0.0)));
/// assert_eq!(res.path.last(), Some(&(1.0, 1.0)));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn ab_elastic_match_pl(
    c1: &FdMatrix,
    c2: &FdMatrix,
    a: f64,
    b: f64,
) -> Result<AbPlMatching, FdarError> {
    for (parameter, v) in [("a", a), ("b", b)] {
        if !(v.is_finite() && v > 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter,
                message: format!("must be finite and > 0, got {v}"),
            });
        }
    }
    if c1.nrows() != c2.nrows() {
        return Err(FdarError::InvalidDimension {
            parameter: "c2",
            expected: format!("{} rows (dimension of c1)", c1.nrows()),
            actual: format!("{} rows", c2.nrows()),
        });
    }
    let g = PlGrid::new(c1, c2, a / (2.0 * b))?;
    let dp = solve(&g);
    let (l1, l2) = (g.s[g.n1()], g.t[g.n2()]);
    let similarity = dp.best[g.vid(g.n1(), g.n2())];
    let distance = 2.0 * b * (l1 + l2 - 2.0 * similarity).max(0.0).sqrt();
    Ok(AbPlMatching {
        distance,
        similarity,
        length1: l1,
        length2: l2,
        path: backtrack(&g, &dp),
    })
}

/// Exact `(a, b)` elastic distance between two piecewise-linear curves
/// (modulo translation and reparametrisation); the distance of
/// [`ab_elastic_match_pl`].
///
/// # Errors
/// As [`ab_elastic_match_pl`].
///
/// # Example
/// ```
/// use fdars_core::alignment::ab_elastic_distance_pl;
/// use fdars_core::matrix::FdMatrix;
/// // the same L-shaped polygon, once with an extra collinear vertex
/// let c1 = FdMatrix::from_column_major(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0], 2, 3).unwrap();
/// let c2 = FdMatrix::from_column_major(
///     vec![0.0, 0.0, 0.3, 0.0, 1.0, 0.0, 1.0, 1.0], 2, 4).unwrap();
/// assert!(ab_elastic_distance_pl(&c1, &c2, 1.0, 0.5).unwrap() < 1e-7);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn ab_elastic_distance_pl(
    c1: &FdMatrix,
    c2: &FdMatrix,
    a: f64,
    b: f64,
) -> Result<f64, FdarError> {
    ab_elastic_match_pl(c1, c2, a, b).map(|r| r.distance)
}
