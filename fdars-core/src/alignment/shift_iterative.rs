//! Iterative least-squares shift registration with template re-estimation.
//!
//! [`iterative_shift_registration`] is the Ramsay & Silverman (2005, ch. 7)
//! modified Newton–Raphson shift registration, as implemented by scikit-fda's
//! `LeastSquaresShiftRegistration`: every iteration re-evaluates the shifted
//! curves, re-estimates the template (sample mean, a fixed curve, or a user
//! closure — see [`ShiftTemplate`]) and takes one damped Newton step per curve.
//!
//! [`shift_curves`] applies a given set of shifts with an explicit
//! [`ShiftExtrapolation`] policy and optional domain restriction; it is the
//! building block shared with landmark shift registration
//! ([`crate::alignment::landmark_shift_registration`]).
//!
//! # Sign convention
//!
//! fdars uses the convention of [`crate::alignment::least_squares_shift_registration`]:
//! the registered curve is `fᵢ(t − δᵢ)`, so a positive δᵢ moves the graph to the
//! right. scikit-fda registers `xᵢ(t + dᵢ)`, hence `δᵢ(fdars) = −dᵢ(scikit-fda)`.
//!
//! # Deviations from scikit-fda 0.10.1
//!
//! * Shifts carry the opposite sign (see above); `initial` uses the fdars sign.
//! * A single [`ShiftExtrapolation`] is used both inside the iterations and for
//!   the final re-evaluation (scikit-fda uses the data's own extrapolation inside
//!   the iterations and the transformer's `extrapolation` only at the end; they
//!   coincide when the transformer's `extrapolation=None`). The default is
//!   [`ShiftExtrapolation::Boundary`] (clamping, like the existing fdars shift
//!   registration); scikit-fda's `FDataGrid` default is linear extrapolation —
//!   select [`ShiftExtrapolation::Linear`] to reproduce it.
//! * With `restrict_domain`, the integration interval is recomputed from the
//!   current shifts on every iteration. scikit-fda restricts the derivative
//!   cumulatively, which raises an `AssertionError` as soon as the admissible
//!   interval widens again; whenever scikit-fda succeeds the two coincide.
//! * Integrals use [`crate::helpers::simpsons_weights`]; for an even number of
//!   (restricted) grid points this closes with a trapezoid panel, whereas SciPy
//!   ≥ 1.11 `simpson` uses Cartwright's correction (difference `O(h³)`).
//! * A curve with zero derivative energy (`∫ Dfᵢ² = 0`, e.g. a constant curve)
//!   gets a zero Newton step instead of a division by zero.
//! * The returned template is evaluated on the full `argvals` grid.

use crate::error::FdarError;
use crate::helpers::{gradient_nonuniform, linear_interp, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ---------------------------------------------------------------------------
// Extrapolation
// ---------------------------------------------------------------------------

/// How a shifted curve is evaluated outside the observed domain `[t₀, t_{m−1}]`.
///
/// Mirrors scikit-fda's extrapolation modes (`"bounds"`, the `FDataGrid`
/// default linear extension, `"periodic"`, `"zeros"`). Inside the domain all
/// modes use linear interpolation between grid points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShiftExtrapolation {
    /// Clamp to the boundary value (scikit-fda `"bounds"`; fdars default, same
    /// policy as [`crate::alignment::least_squares_shift_registration`]).
    #[default]
    Boundary,
    /// Extend the first / last linear segment (scikit-fda `FDataGrid` default).
    Linear,
    /// Wrap modulo the domain length `t_{m−1} − t₀` (scikit-fda `"periodic"`).
    Periodic,
    /// Evaluate to `0` outside the domain (scikit-fda `"zeros"`).
    Zeros,
}

/// Evaluate the piecewise-linear curve `(argvals, row)` at `x` with `ext`.
pub(crate) fn eval_extrapolated(
    argvals: &[f64],
    row: &[f64],
    x: f64,
    ext: ShiftExtrapolation,
) -> f64 {
    let m = argvals.len();
    let (lo, hi) = (argvals[0], argvals[m - 1]);
    if (lo..=hi).contains(&x) {
        return linear_interp(argvals, row, x);
    }
    match ext {
        ShiftExtrapolation::Boundary => linear_interp(argvals, row, x),
        ShiftExtrapolation::Zeros => 0.0,
        ShiftExtrapolation::Periodic => {
            let period = hi - lo;
            linear_interp(argvals, row, lo + (x - lo).rem_euclid(period))
        }
        ShiftExtrapolation::Linear => {
            if x < lo {
                let slope = (row[1] - row[0]) / (argvals[1] - argvals[0]);
                row[0] + (x - lo) * slope
            } else {
                let slope = (row[m - 1] - row[m - 2]) / (argvals[m - 1] - argvals[m - 2]);
                row[m - 1] + (x - hi) * slope
            }
        }
    }
}

/// Second-order finite-difference derivative, as NumPy `gradient(y, t,
/// edge_order=2)` (scikit-fda's default `FDataGrid.derivative`): central
/// differences inside, one-sided 3-point formulas at the ends. Uniform grids
/// use NumPy's uniform-spacing formulas (exactly zero for constant curves);
/// non-uniform grids use the equivalent 3-point Lagrange formulas of
/// [`gradient_nonuniform`].
fn gradient_edge_order2(y: &[f64], t: &[f64]) -> Vec<f64> {
    let m = y.len();
    let h = t[1] - t[0];
    let uniform = t
        .windows(2)
        .all(|w| ((w[1] - w[0]) - h).abs() < 1e-12 * h.abs().max(1.0));
    if m < 3 || !uniform {
        return gradient_nonuniform(y, t);
    }
    let mut g = vec![0.0; m];
    g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
    for i in 1..m - 1 {
        g[i] = (y[i + 1] - y[i - 1]) / (2.0 * h);
    }
    g[m - 1] = (3.0 * y[m - 1] - 4.0 * y[m - 2] + y[m - 3]) / (2.0 * h);
    g
}

/// Re-evaluate every curve at `t − δᵢ` on the full grid.
fn shift_matrix(
    data: &FdMatrix,
    argvals: &[f64],
    shifts: &[f64],
    ext: ShiftExtrapolation,
) -> FdMatrix {
    let (n, m) = data.shape();
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let row = data.row(i);
            argvals
                .iter()
                .map(|&t| eval_extrapolated(argvals, &row, t - shifts[i], ext))
                .collect()
        })
        .collect();
    let mut out = FdMatrix::zeros(n, m);
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    out
}

/// Grid index range `[start, end)` of points where every shifted curve is
/// observed: `t ∈ [t₀ + max(maxδ, 0), t_{m−1} + min(minδ, 0)]`.
fn restricted_range(argvals: &[f64], shifts: &[f64]) -> (usize, usize) {
    let m = argvals.len();
    let max_d = shifts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_d = shifts.iter().copied().fold(f64::INFINITY, f64::min);
    let a = argvals[0] + max_d.max(0.0);
    let b = argvals[m - 1] + min_d.min(0.0);
    let start = argvals.partition_point(|&t| t < a);
    let end = argvals.partition_point(|&t| t <= b);
    (start, end.max(start))
}

// ---------------------------------------------------------------------------
// shift_curves
// ---------------------------------------------------------------------------

/// Curves re-evaluated after a rigid horizontal shift (see [`shift_curves`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShiftedCurves {
    /// Shifted curves, `data[(i, j)] = fᵢ(argvals[j] − δᵢ)` (n × m', m' = `argvals.len()`).
    pub data: FdMatrix,
    /// Grid of the shifted curves: the input grid, or its restriction to the
    /// sub-interval observed by every shifted curve when `restrict_domain` is set.
    pub argvals: Vec<f64>,
}

fn validate_data_grid(data: &FdMatrix, argvals: &[f64]) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: m.to_string(),
            actual: argvals.len().to_string(),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must have at least 2 evaluation points".to_string(),
        });
    }
    if argvals
        .windows(2)
        .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
        || argvals.iter().any(|t| !t.is_finite())
    {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must be finite and strictly increasing".to_string(),
        });
    }
    Ok(())
}

fn validate_shifts(shifts: &[f64], n: usize, parameter: &'static str) -> Result<(), FdarError> {
    if shifts.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: n.to_string(),
            actual: shifts.len().to_string(),
        });
    }
    if shifts.iter().any(|d| !d.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter,
            message: "all shifts must be finite".to_string(),
        });
    }
    Ok(())
}

/// Restrict the columns of `x` to `[start, end)`.
fn restrict_columns(x: &FdMatrix, start: usize, end: usize) -> FdMatrix {
    let n = x.nrows();
    let mut out = FdMatrix::zeros(n, end - start);
    for j in start..end {
        for i in 0..n {
            out[(i, j - start)] = x[(i, j)];
        }
    }
    out
}

/// Assemble the (optionally domain-restricted) shifted curves.
fn finalize_shift(
    data: &FdMatrix,
    argvals: &[f64],
    shifts: &[f64],
    ext: ShiftExtrapolation,
    restrict_domain: bool,
) -> Result<ShiftedCurves, FdarError> {
    let full = shift_matrix(data, argvals, shifts, ext);
    if !restrict_domain {
        return Ok(ShiftedCurves {
            data: full,
            argvals: argvals.to_vec(),
        });
    }
    let (start, end) = restricted_range(argvals, shifts);
    if end == start {
        return Err(FdarError::ComputationFailed {
            operation: "shift_curves",
            detail: "restricted domain contains no grid point (shift range exceeds the domain)"
                .to_string(),
        });
    }
    Ok(ShiftedCurves {
        data: restrict_columns(&full, start, end),
        argvals: argvals[start..end].to_vec(),
    })
}

/// Shift every curve rigidly in time: `gᵢ(t) = fᵢ(t − δᵢ)`.
///
/// The counterpart of scikit-fda's `FData.shift(-δ, restrict_domain, extrapolation)`
/// (opposite sign convention). Values are obtained by linear interpolation on
/// `argvals`; points that fall outside the domain follow `extrapolation`. With
/// `restrict_domain = true` the output keeps only the grid points
/// `t ∈ [t₀ + max(maxᵢδᵢ, 0), t_{m−1} + min(minᵢδᵢ, 0)]`, i.e. where no curve
/// needs extrapolation.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `data` is empty, `argvals.len() != m`
///   or `shifts.len() != n`.
/// * [`FdarError::InvalidParameter`] if `argvals` has fewer than 2 points, is
///   not strictly increasing, or a shift is not finite.
/// * [`FdarError::ComputationFailed`] if the restricted domain is empty.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{shift_curves, ShiftExtrapolation};
/// use fdars_core::matrix::FdMatrix;
///
/// let argvals: Vec<f64> = (0..11).map(|j| j as f64 / 10.0).collect();
/// let data = FdMatrix::from_column_major(argvals.clone(), 1, 11).unwrap(); // f(t) = t
/// let out = shift_curves(&data, &argvals, &[0.2], ShiftExtrapolation::Linear, false).unwrap();
/// assert!((out.data[(0, 5)] - 0.3).abs() < 1e-12); // f(0.5 - 0.2)
/// assert!((out.data[(0, 0)] + 0.2).abs() < 1e-12); // linear extrapolation
/// let restricted = shift_curves(&data, &argvals, &[0.2], ShiftExtrapolation::Boundary, true).unwrap();
/// assert_eq!(restricted.argvals.len(), 9); // t in [0.2, 1.0]
/// ```
#[must_use = "shift_curves returns the shifted curves without modifying the input"]
pub fn shift_curves(
    data: &FdMatrix,
    argvals: &[f64],
    shifts: &[f64],
    extrapolation: ShiftExtrapolation,
    restrict_domain: bool,
) -> Result<ShiftedCurves, FdarError> {
    validate_data_grid(data, argvals)?;
    validate_shifts(shifts, data.nrows(), "shifts")?;
    finalize_shift(data, argvals, shifts, extrapolation, restrict_domain)
}

// ---------------------------------------------------------------------------
// Template, config, result
// ---------------------------------------------------------------------------

/// Template used by [`iterative_shift_registration`].
///
/// With [`ShiftTemplate::Mean`] or [`ShiftTemplate::Custom`] the template is
/// re-estimated from the currently shifted curves on every iteration
/// (scikit-fda `template="mean"` / callable); [`ShiftTemplate::Fixed`] is used
/// as-is (scikit-fda `template=FData`).
#[derive(Clone, Copy, Default)]
#[non_exhaustive]
pub enum ShiftTemplate<'a> {
    /// Cross-sectional mean of the shifted curves (default).
    #[default]
    Mean,
    /// A fixed template evaluated on `argvals` (length m).
    Fixed(&'a [f64]),
    /// A closure mapping the currently shifted curves (n × m on `argvals`) to a
    /// template of length m — e.g. a pointwise median or a depth-based median.
    Custom(&'a dyn Fn(&FdMatrix) -> Vec<f64>),
}

impl std::fmt::Debug for ShiftTemplate<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShiftTemplate::Mean => f.write_str("Mean"),
            ShiftTemplate::Fixed(t) => f.debug_tuple("Fixed").field(t).finish(),
            ShiftTemplate::Custom(_) => f.write_str("Custom(<closure>)"),
        }
    }
}

/// Configuration of [`iterative_shift_registration`] (scikit-fda
/// `LeastSquaresShiftRegistration` parameters).
///
/// Construct with `IterativeShiftConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IterativeShiftConfig {
    /// Maximum number of Newton iterations (default 5, as scikit-fda).
    pub max_iter: usize,
    /// Stop when `maxᵢ |δᵢ⁽ᵛ⁾ − δᵢ⁽ᵛ⁻¹⁾| ≤ tol` (default `1e-2`, as scikit-fda).
    pub tol: f64,
    /// Newton step multiplier (default 1).
    pub step_size: f64,
    /// Integrate only over the sub-interval observed by every shifted curve and
    /// return the registered curves on that sub-grid (default `false`).
    pub restrict_domain: bool,
    /// Out-of-domain evaluation policy (default [`ShiftExtrapolation::Boundary`]).
    pub extrapolation: ShiftExtrapolation,
    /// Initial shifts in the fdars sign convention (default: all zeros).
    pub initial: Option<Vec<f64>>,
}

impl Default for IterativeShiftConfig {
    fn default() -> Self {
        Self {
            max_iter: 5,
            tol: 1e-2,
            step_size: 1.0,
            restrict_domain: false,
            extrapolation: ShiftExtrapolation::Boundary,
            initial: None,
        }
    }
}

impl IterativeShiftConfig {
    /// Validate the scalar parameters.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `tol` is negative or not finite, or
    /// `step_size` is not finite and positive.
    pub fn validate(&self) -> Result<(), FdarError> {
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter: "tol",
                message: format!("must be finite and >= 0, got {}", self.tol),
            });
        }
        if !(self.step_size.is_finite() && self.step_size > 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter: "step_size",
                message: format!("must be finite and > 0, got {}", self.step_size),
            });
        }
        Ok(())
    }
}

/// Result of [`iterative_shift_registration`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IterativeShiftResult {
    /// Registered curves `fᵢ(t − δᵢ)` on `registered_argvals`.
    pub registered_data: FdMatrix,
    /// Grid of `registered_data` (equal to `argvals` unless `restrict_domain`).
    pub registered_argvals: Vec<f64>,
    /// Per-curve shifts δᵢ (fdars sign: positive moves the curve right).
    pub shifts: Vec<f64>,
    /// Template of the last iteration, on the full `argvals` grid.
    pub template: Vec<f64>,
    /// Number of Newton iterations performed.
    pub n_iter: usize,
    /// Whether the last update satisfied `max |Δδ| ≤ tol`.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Iterative registration
// ---------------------------------------------------------------------------

fn compute_template(
    template: &ShiftTemplate<'_>,
    shifted: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let m = shifted.ncols();
    let tmpl = match template {
        ShiftTemplate::Mean => crate::fdata::mean_1d(shifted),
        ShiftTemplate::Fixed(t) => t.to_vec(),
        ShiftTemplate::Custom(f) => f(shifted),
    };
    if tmpl.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "template",
            expected: m.to_string(),
            actual: tmpl.len().to_string(),
        });
    }
    if tmpl.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::ComputationFailed {
            operation: "iterative_shift_registration",
            detail: "template contains non-finite values".to_string(),
        });
    }
    Ok(tmpl)
}

/// One modified Newton step for curve `i` on columns `[start, end)`:
/// `step · ∫(xᵢ − μ)·Dfᵢ / ∫(Dfᵢ)²`, with the derivative of the *unshifted*
/// curve and the second-derivative term of the Hessian dropped (as scikit-fda).
fn newton_step(
    shifted: &FdMatrix,
    deriv: &[f64],
    template: &[f64],
    weights: &[f64],
    i: usize,
    start: usize,
    step_size: f64,
) -> f64 {
    let mut d1 = 0.0;
    let mut d2 = 0.0;
    for (k, &w) in weights.iter().enumerate() {
        let j = start + k;
        let dj = deriv[j];
        d1 += w * (shifted[(i, j)] - template[j]) * dj;
        d2 += w * dj * dj;
    }
    if d2 > 0.0 && d2.is_finite() {
        step_size * d1 / d2
    } else {
        0.0
    }
}

fn validate_iterative_inputs(
    data: &FdMatrix,
    argvals: &[f64],
    template: &ShiftTemplate<'_>,
    config: &IterativeShiftConfig,
) -> Result<Vec<f64>, FdarError> {
    validate_data_grid(data, argvals)?;
    config.validate()?;
    if let ShiftTemplate::Fixed(t) = template {
        if t.len() != argvals.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "template",
                expected: argvals.len().to_string(),
                actual: t.len().to_string(),
            });
        }
    }
    match &config.initial {
        Some(init) => {
            validate_shifts(init, data.nrows(), "initial")?;
            Ok(init.clone())
        }
        None => Ok(vec![0.0; data.nrows()]),
    }
}

/// One Newton update of all shifts; returns `(template, max |Δδ|)`.
fn newton_iteration(
    data: &FdMatrix,
    argvals: &[f64],
    derivs: &[Vec<f64>],
    template: &ShiftTemplate<'_>,
    config: &IterativeShiftConfig,
    shifts: &mut [f64],
) -> Result<(Vec<f64>, f64), FdarError> {
    let shifted = shift_matrix(data, argvals, shifts, config.extrapolation);
    let tmpl = compute_template(template, &shifted)?;
    let (start, end) = if config.restrict_domain {
        restricted_range(argvals, shifts)
    } else {
        (0, argvals.len())
    };
    if end - start < 2 {
        return Err(FdarError::ComputationFailed {
            operation: "iterative_shift_registration",
            detail: "restricted domain has fewer than 2 grid points".to_string(),
        });
    }
    let weights = simpsons_weights(&argvals[start..end]);
    let steps: Vec<f64> = iter_maybe_parallel!(0..data.nrows())
        .map(|i| {
            newton_step(
                &shifted,
                &derivs[i],
                &tmpl,
                &weights,
                i,
                start,
                config.step_size,
            )
        })
        .collect();
    let mut max_diff = 0.0_f64;
    for (d, s) in shifts.iter_mut().zip(&steps) {
        *d += s;
        max_diff = max_diff.max(s.abs());
    }
    Ok((tmpl, max_diff))
}

/// Iterative least-squares shift registration with template re-estimation.
///
/// Minimises `REGSSE = Σᵢ ∫ [fᵢ(t − δᵢ) − μ(t)]² dt` by the modified
/// Newton–Raphson scheme of Ramsay & Silverman (2005, ch. 7), exactly as
/// scikit-fda 0.10.1 `LeastSquaresShiftRegistration`:
///
/// 1. start from `config.initial` (default zeros);
/// 2. evaluate `xᵢ(t) = fᵢ(t − δᵢ)` on the grid (with `config.extrapolation`);
/// 3. form the template μ from the shifted curves ([`ShiftTemplate`]);
/// 4. update `δᵢ ← δᵢ + step_size · ∫(xᵢ − μ)·Dfᵢ / ∫(Dfᵢ)²`, where `Dfᵢ` is the
///    second-order finite-difference derivative of the *unshifted* curve
///    (NumPy `gradient(edge_order=2)`, as scikit-fda's `FDataGrid.derivative`);
/// 5. repeat until `max |Δδ| ≤ tol` or `max_iter` iterations.
///
/// With `restrict_domain` the integrals run over the grid points observed by
/// every shifted curve and the result is returned on that sub-grid. Because a
/// mean template does not pin the common shift, `Σδᵢ` may drift slowly over many
/// iterations; centre the shifts afterwards if a zero-mean convention is wanted.
/// Newton steps assume reasonably smooth curves and moderate shifts; for large
/// misalignment seed `config.initial` with e.g.
/// [`crate::alignment::least_squares_shift_registration`].
///
/// See the [module documentation](self) for the sign convention (opposite to
/// scikit-fda) and the documented deviations.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] for empty `data`, `argvals.len() != m`, a
///   fixed/custom template of length ≠ m, or `initial.len() != n`.
/// * [`FdarError::InvalidParameter`] for fewer than 2 / non-increasing
///   `argvals`, invalid `tol` / `step_size`, or non-finite initial shifts.
/// * [`FdarError::ComputationFailed`] if the template is non-finite or the
///   restricted domain has fewer than 2 grid points.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{iterative_shift_registration, IterativeShiftConfig,
///     ShiftExtrapolation, ShiftTemplate};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 101;
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let true_shifts = [0.04, -0.03, 0.0, 0.02];
/// let mut data = FdMatrix::zeros(4, m);
/// for (i, d) in true_shifts.iter().enumerate() {
///     for (j, &t) in argvals.iter().enumerate() {
///         data[(i, j)] = (2.0 * std::f64::consts::PI * (t - d)).sin();
///     }
/// }
/// let mut config = IterativeShiftConfig::default();
/// config.extrapolation = ShiftExtrapolation::Periodic;
/// config.max_iter = 50;
/// config.tol = 1e-10;
/// // Align to the known template sin(2πt): shifts are recovered.
/// let template: Vec<f64> =
///     argvals.iter().map(|&t| (2.0 * std::f64::consts::PI * t).sin()).collect();
/// let res = iterative_shift_registration(&data, &argvals, ShiftTemplate::Fixed(&template), &config)
///     .unwrap();
/// for (est, truth) in res.shifts.iter().zip(true_shifts) {
///     assert!((est + truth).abs() < 1e-3); // registered f(t - δ) = sin(2π t) ⇒ δ = -d
/// }
/// assert!(res.converged);
/// ```
#[must_use = "iterative_shift_registration is expensive; use the returned registration"]
pub fn iterative_shift_registration(
    data: &FdMatrix,
    argvals: &[f64],
    template: ShiftTemplate<'_>,
    config: &IterativeShiftConfig,
) -> Result<IterativeShiftResult, FdarError> {
    let mut shifts = validate_iterative_inputs(data, argvals, &template, config)?;
    let derivs: Vec<Vec<f64>> = iter_maybe_parallel!(0..data.nrows())
        .map(|i| gradient_edge_order2(&data.row(i), argvals))
        .collect();

    let mut n_iter = 0;
    let mut max_diff = f64::INFINITY;
    let mut last_template = None;
    // Same loop condition as scikit-fda: iterate while max|Δδ| > tol.
    while max_diff > config.tol && n_iter < config.max_iter {
        let (tmpl, diff) =
            newton_iteration(data, argvals, &derivs, &template, config, &mut shifts)?;
        last_template = Some(tmpl);
        max_diff = diff;
        n_iter += 1;
    }
    let template_vec = match last_template {
        Some(t) => t,
        None => {
            let shifted = shift_matrix(data, argvals, &shifts, config.extrapolation);
            compute_template(&template, &shifted)?
        }
    };
    let out = finalize_shift(
        data,
        argvals,
        &shifts,
        config.extrapolation,
        config.restrict_domain,
    )?;
    Ok(IterativeShiftResult {
        registered_data: out.data,
        registered_argvals: out.argvals,
        shifts,
        template: template_vec,
        n_iter,
        converged: max_diff <= config.tol,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const C: [f64; 5] = [0.03, -0.05, 0.08, -0.02, 0.06];
    const A: [f64; 5] = [0.3, 0.25, 0.35, 0.2, 0.4];
    const B: [f64; 5] = [0.1, -0.1, 0.05, 0.0, -0.05];

    fn grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// Golden data of `golden_shift.py`: m = 51 on [0, 1],
    /// fᵢ(t) = sin(2π(t − cᵢ)) + aᵢ cos(4π(t − cᵢ)) + bᵢ t.
    fn golden_data() -> (FdMatrix, Vec<f64>) {
        let t = grid(51);
        let mut x = FdMatrix::zeros(5, 51);
        for i in 0..5 {
            for (j, &tj) in t.iter().enumerate() {
                x[(i, j)] = (2.0 * PI * (tj - C[i])).sin()
                    + A[i] * (4.0 * PI * (tj - C[i])).cos()
                    + B[i] * tj;
            }
        }
        (x, t)
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < tol, "{what}[{k}]: {a} vs {e}");
        }
    }

    fn neg(v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| -x).collect()
    }

    fn row_slice(x: &FdMatrix, i: usize, r: std::ops::Range<usize>) -> Vec<f64> {
        r.map(|j| x[(i, j)]).collect()
    }

    fn cfg(extrapolation: ShiftExtrapolation, max_iter: usize, tol: f64) -> IterativeShiftConfig {
        IterativeShiftConfig {
            extrapolation,
            max_iter,
            tol,
            ..Default::default()
        }
    }

    // Golden: scikit-fda 0.10.1 `LeastSquaresShiftRegistration()` (defaults:
    // mean template, max_iter=5, tol=1e-2) on an FDataGrid with its default
    // linear extrapolation. skfda deltas are negated (fdars sign convention).
    #[test]
    fn golden_skfda_default_linear() {
        let (x, t) = golden_data();
        let r = iterative_shift_registration(
            &x,
            &t,
            ShiftTemplate::Mean,
            &cfg(ShiftExtrapolation::Linear, 5, 1e-2),
        )
        .unwrap();
        let skfda = [
            0.005346551638920673,
            -0.07353639852695774,
            0.05309146969086503,
            -0.04681662427182223,
            0.03194501868706887,
        ];
        assert_close(&r.shifts, &neg(&skfda), 1e-10, "shifts");
        assert_eq!(r.n_iter, 2);
        assert!(r.converged);
        assert_close(
            &row_slice(&r.registered_data, 0, 0..3),
            &[0.13039225963762435, 0.27094982153873515, 0.3932667237448959],
            1e-10,
            "row0",
        );
        assert_close(
            &row_slice(&r.registered_data, 4, 48..51),
            &[
                -0.2030134521527976,
                -0.013127597758793963,
                0.17675825663520972,
            ],
            1e-10,
            "row4",
        );
        let tmpl: Vec<f64> = r.template.iter().step_by(10).copied().collect();
        assert_close(
            &tmpl,
            &[
                0.16572255411001394,
                0.7193180632623323,
                0.7020057618215242,
                -0.2812402868103908,
                -1.2658642144022765,
                0.14643944785548638,
            ],
            1e-10,
            "template",
        );
    }

    // Golden: scikit-fda 0.10.1 `LeastSquaresShiftRegistration(max_iter=10,
    // tol=1e-12)` on FDataGrid(extrapolation="bounds").
    #[test]
    fn golden_skfda_bounds_10_iterations() {
        let (x, t) = golden_data();
        let r = iterative_shift_registration(
            &x,
            &t,
            ShiftTemplate::Mean,
            &cfg(ShiftExtrapolation::Boundary, 10, 1e-12),
        )
        .unwrap();
        let skfda = [
            -0.004305282544714397,
            -0.0879737560117377,
            0.04901723232680556,
            -0.05992044221630647,
            0.024588068168219537,
        ];
        assert_close(&r.shifts, &neg(&skfda), 1e-10, "shifts");
        assert_eq!(r.n_iter, 10);
        assert!(!r.converged);
        assert_close(
            &row_slice(&r.registered_data, 0, 0..3),
            &[0.09155163118075083, 0.2055676793899881, 0.3369613088988751],
            1e-10,
            "row0",
        );
        assert_close(
            &row_slice(&r.registered_data, 4, 48..51),
            &[
                -0.27286249390310313,
                -0.1265371017161145,
                -0.1265371017161145,
            ],
            1e-10,
            "row4",
        );
    }

    // Golden: scikit-fda 0.10.1 `LeastSquaresShiftRegistration(template=
    // FDataGrid(sin(2πt)), max_iter=20, tol=1e-8)` on FDataGrid(extrapolation="periodic").
    #[test]
    fn golden_skfda_fixed_template_periodic() {
        let (x, t) = golden_data();
        let tmpl: Vec<f64> = t.iter().map(|&s| (2.0 * PI * s).sin()).collect();
        let r = iterative_shift_registration(
            &x,
            &t,
            ShiftTemplate::Fixed(&tmpl),
            &cfg(ShiftExtrapolation::Periodic, 20, 1e-8),
        )
        .unwrap();
        let skfda = [
            0.022627304282585848,
            -0.03767034777642996,
            0.05504617539815348,
            -0.017282570128384914,
            0.03590365792784716,
        ];
        assert_close(&r.shifts, &neg(&skfda), 1e-10, "shifts");
        assert_eq!(r.n_iter, 8);
        assert!(r.converged);
        assert_eq!(r.template, tmpl);
        assert_close(
            &row_slice(&r.registered_data, 0, 0..3),
            &[0.2536036013798041, 0.3785978349176683, 0.4837966089634951],
            1e-10,
            "row0",
        );
        assert_close(
            &row_slice(&r.registered_data, 4, 48..51),
            &[
                -0.16542897242915755,
                0.06450439841845168,
                0.2274793748548045,
            ],
            1e-10,
            "row4",
        );
    }

    /// Pointwise median (same as `np.median(axis=0)`).
    fn pointwise_median(x: &FdMatrix) -> Vec<f64> {
        (0..x.ncols())
            .map(|j| {
                let mut col = x.column(j).to_vec();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = col.len();
                if n % 2 == 1 {
                    col[n / 2]
                } else {
                    0.5 * (col[n / 2 - 1] + col[n / 2])
                }
            })
            .collect()
    }

    // Golden: scikit-fda 0.10.1 `LeastSquaresShiftRegistration(template=<pointwise
    // median callable>, max_iter=7, tol=1e-6, step_size=0.8,
    // initial=[0.01, -0.02, 0, 0.02, -0.01])` on FDataGrid(extrapolation="zeros").
    #[test]
    fn golden_skfda_custom_median_zeros_step_initial() {
        let (x, t) = golden_data();
        let mut c = cfg(ShiftExtrapolation::Zeros, 7, 1e-6);
        c.step_size = 0.8;
        c.initial = Some(neg(&[0.01, -0.02, 0.0, 0.02, -0.01]));
        let med = pointwise_median;
        let r = iterative_shift_registration(&x, &t, ShiftTemplate::Custom(&med), &c).unwrap();
        let skfda = [
            -0.005985209467996968,
            -0.08456091926246823,
            0.04314946863495152,
            -0.05875303807896869,
            0.021746393187812456,
        ];
        assert_close(&r.shifts, &neg(&skfda), 1e-10, "shifts");
        assert_eq!(r.n_iter, 7);
        assert_close(
            &row_slice(&r.registered_data, 0, 0..3),
            &[0.0, 0.1933636604505789, 0.3262449677781275],
            1e-10,
            "row0",
        );
        assert_close(
            &row_slice(&r.registered_data, 4, 48..51),
            &[-0.29984218798133644, 0.0, 0.0],
            1e-10,
            "row4",
        );
        let tmpl: Vec<f64> = r.template.iter().step_by(10).copied().collect();
        assert_close(
            &tmpl,
            &[
                0.0,
                0.7293955758362706,
                0.7264187275990622,
                -0.14430900837932395,
                -1.2927908125680487,
                0.0,
            ],
            1e-10,
            "template",
        );
    }

    // Golden: scikit-fda 0.10.1 `LeastSquaresShiftRegistration(restrict_domain=True)`
    // (linear extrapolation). The restricted grids have even length (44), where
    // SciPy's `simpson` (Cartwright end correction) and fdars' Simpson + trapezoid
    // panel differ by O(h³) per integral. Two references:
    // * `quad_matched`: the same scikit-fda run with `scipy.integrate.simpson`
    //   patched to fdars' even-N rule (golden_shift.py case 7) — must agree to
    //   floating-point precision, proving the algorithms are identical;
    // * `raw`: unpatched scikit-fda — differs only through the quadrature rule
    //   (observed max |Δδ| = 6.1e-6).
    #[test]
    fn golden_skfda_restrict_domain() {
        let (x, t) = golden_data();
        let mut c = cfg(ShiftExtrapolation::Linear, 5, 1e-2);
        c.restrict_domain = true;
        let r = iterative_shift_registration(&x, &t, ShiftTemplate::Mean, &c).unwrap();
        let quad_matched = [
            0.0062455179835034635,
            -0.07831562791641501,
            0.05582720119212679,
            -0.05305402549121352,
            0.03246605374556555,
        ];
        assert_close(
            &r.shifts,
            &neg(&quad_matched),
            1e-12,
            "shifts (quadrature-matched)",
        );
        let raw = [
            0.006244100390870666,
            -0.07831107960542696,
            0.05582859892215349,
            -0.053047899138305564,
            0.03247093954319142,
        ];
        assert_close(&r.shifts, &neg(&raw), 1e-5, "shifts (raw skfda)");
        assert_eq!(r.n_iter, 3);
        assert_eq!(r.registered_argvals.len(), 44);
        assert!((r.registered_argvals[0] - 0.08).abs() < 1e-12);
        assert!((r.registered_argvals[43] - 0.94).abs() < 1e-12);
        assert_close(
            &row_slice(&r.registered_data, 0, 0..3),
            &[0.5807333878870369, 0.6421687618393118, 0.6855777068104194],
            1e-12,
            "row0",
        );
        assert_close(
            &row_slice(&r.registered_data, 4, 41..44),
            &[
                -0.7761330365915317,
                -0.5853528280266251,
                -0.3901635263200954,
            ],
            1e-12,
            "row4",
        );
        // skfda returns the template restricted to the sub-grid (indices 4..48).
        let tmpl: Vec<f64> = r.template[4..48].iter().step_by(10).copied().collect();
        assert_close(
            &tmpl,
            &[
                0.5666095644786538,
                0.7023013421242166,
                0.5350158283324313,
                -0.9300290262943932,
                -0.8645266144317354,
            ],
            1e-12,
            "template",
        );
    }

    // Invariant: mean-template registration of exactly shifted periodic curves
    // recovers the relative shifts (the common shift is not identified).
    #[test]
    fn recovers_relative_shifts_with_mean_template() {
        let m = 201;
        let t = grid(m);
        let d = [0.05, -0.04, 0.02, -0.01, 0.0, 0.03];
        let mut x = FdMatrix::zeros(d.len(), m);
        for (i, &di) in d.iter().enumerate() {
            for (j, &tj) in t.iter().enumerate() {
                x[(i, j)] = (2.0 * PI * (tj - di)).sin() + 0.4 * (4.0 * PI * (tj - di)).cos();
            }
        }
        let r = iterative_shift_registration(
            &x,
            &t,
            ShiftTemplate::Mean,
            &cfg(ShiftExtrapolation::Periodic, 100, 1e-12),
        )
        .unwrap();
        assert!(r.converged);
        let mean_est = r.shifts.iter().sum::<f64>() / d.len() as f64;
        let mean_true = d.iter().sum::<f64>() / d.len() as f64;
        for (est, &truth) in r.shifts.iter().zip(&d) {
            // registered fᵢ(t − δᵢ) aligned ⇔ δᵢ = −dᵢ + const
            let err = (est - mean_est) + (truth - mean_true);
            assert!(err.abs() < 1e-3, "relative shift error {err}");
        }
    }

    // Invariant: a curve identical to the fixed template gets exactly zero shift.
    #[test]
    fn template_curve_has_zero_shift() {
        let (x, t) = golden_data();
        let tmpl = x.row(2);
        let r = iterative_shift_registration(
            &x,
            &t,
            ShiftTemplate::Fixed(&tmpl),
            &cfg(ShiftExtrapolation::Boundary, 5, 1e-2),
        )
        .unwrap();
        assert_eq!(r.shifts[2], 0.0);
        assert_eq!(r.registered_data.row(2), tmpl);
    }

    // max_iter = 0 returns the initial shifts and the matching re-evaluation.
    #[test]
    fn zero_iterations_returns_initial() {
        let (x, t) = golden_data();
        let mut c = cfg(ShiftExtrapolation::Linear, 0, 1e-2);
        let init = vec![0.01, -0.02, 0.03, 0.0, -0.01];
        c.initial = Some(init.clone());
        let r = iterative_shift_registration(&x, &t, ShiftTemplate::Mean, &c).unwrap();
        assert_eq!(r.n_iter, 0);
        assert!(!r.converged);
        assert_eq!(r.shifts, init);
        let s = shift_curves(&x, &t, &init, ShiftExtrapolation::Linear, false).unwrap();
        assert_eq!(r.registered_data, s.data);
        assert_eq!(r.template, crate::fdata::mean_1d(&s.data));
    }

    // A constant curve has zero derivative energy: its Newton step is 0.
    #[test]
    fn constant_curve_gets_zero_step() {
        let t = grid(21);
        let mut x = FdMatrix::zeros(2, 21);
        for (j, &tj) in t.iter().enumerate() {
            x[(0, j)] = 3.0;
            x[(1, j)] = (2.0 * PI * tj).sin();
        }
        let r = iterative_shift_registration(
            &x,
            &t,
            ShiftTemplate::Mean,
            &cfg(ShiftExtrapolation::Boundary, 5, 1e-2),
        )
        .unwrap();
        assert_eq!(r.shifts[0], 0.0);
        assert!(r.shifts.iter().all(|d| d.is_finite()));
    }

    // Reference: NumPy 2.x `np.gradient(y, t, edge_order=2)`.
    #[test]
    fn gradient_matches_numpy_edge_order2() {
        let t = [0.0, 0.1, 0.3, 0.6, 1.0];
        let y: Vec<f64> = t.iter().map(|v: &f64| v.powi(3)).collect();
        assert_close(
            &gradient_edge_order2(&y, &t),
            &[
                -0.029999999999999995,
                0.049999999999999996,
                0.3299999999999999,
                1.2,
                2.7200000000000006,
            ],
            1e-12,
            "nonuniform",
        );
        let t = grid(6);
        let y: Vec<f64> = t.iter().map(|v| v.exp()).collect();
        assert_close(
            &gradient_edge_order2(&y, &t),
            &[
                0.9844658374985209,
                1.2295617441031759,
                1.5017901055758474,
                1.834290577127994,
                2.2404075701713406,
                2.6870014294944333,
            ],
            1e-12,
            "uniform",
        );
        assert!(gradient_edge_order2(&[2.0; 6], &t)
            .iter()
            .all(|&g| g == 0.0));
    }

    #[test]
    fn shift_curves_extrapolation_modes() {
        let t = grid(11);
        // f(t) = t² on the grid
        let x = FdMatrix::from_column_major(t.iter().map(|v| v * v).collect(), 1, 11).unwrap();
        let lin = shift_curves(&x, &t, &[0.2], ShiftExtrapolation::Linear, false).unwrap();
        // t = 0 → f(−0.2) = f(0) − 0.2·(f(0.1) − f(0))/0.1 = −0.02
        assert!((lin.data[(0, 0)] + 0.02).abs() < 1e-12);
        assert!((lin.data[(0, 5)] - 0.09).abs() < 1e-12);
        let bnd = shift_curves(&x, &t, &[0.2], ShiftExtrapolation::Boundary, false).unwrap();
        assert_eq!(bnd.data[(0, 0)], 0.0);
        assert_eq!(bnd.data[(0, 1)], 0.0);
        let zer = shift_curves(&x, &t, &[-0.25], ShiftExtrapolation::Zeros, false).unwrap();
        assert_eq!(zer.data[(0, 10)], 0.0); // f(1.25) → 0
        assert!((zer.data[(0, 7)] - 0.905).abs() < 1e-12); // f(0.95) by linear interp
        let per = shift_curves(&x, &t, &[-0.25], ShiftExtrapolation::Periodic, false).unwrap();
        assert!((per.data[(0, 10)] - 0.065).abs() < 1e-12); // f(0.25) interp = (0.04+0.09)/2
                                                            // restriction: δ = −0.25 ⇒ t ∈ [0, 0.75]
        let r = shift_curves(&x, &t, &[-0.25], ShiftExtrapolation::Linear, true).unwrap();
        assert_eq!(r.argvals.len(), 8);
        assert!((r.argvals[7] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn input_validation() {
        let (x, t) = golden_data();
        let base = IterativeShiftConfig::default();
        let short = vec![0.0; 50];
        assert!(matches!(
            iterative_shift_registration(&x, &t, ShiftTemplate::Fixed(&short), &base),
            Err(FdarError::InvalidDimension {
                parameter: "template",
                ..
            })
        ));
        let bad_len = |_: &FdMatrix| vec![0.0; 3];
        assert!(matches!(
            iterative_shift_registration(&x, &t, ShiftTemplate::Custom(&bad_len), &base),
            Err(FdarError::InvalidDimension {
                parameter: "template",
                ..
            })
        ));
        let mut c = base.clone();
        c.initial = Some(vec![0.0; 4]);
        assert!(matches!(
            iterative_shift_registration(&x, &t, ShiftTemplate::Mean, &c),
            Err(FdarError::InvalidDimension {
                parameter: "initial",
                ..
            })
        ));
        let mut c = base.clone();
        c.step_size = 0.0;
        assert!(matches!(
            iterative_shift_registration(&x, &t, ShiftTemplate::Mean, &c),
            Err(FdarError::InvalidParameter {
                parameter: "step_size",
                ..
            })
        ));
        let mut c = base.clone();
        c.tol = f64::NAN;
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.restrict_domain = true;
        c.initial = Some(vec![0.6, -0.6, 0.0, 0.0, 0.0]);
        assert!(matches!(
            iterative_shift_registration(&x, &t, ShiftTemplate::Mean, &c),
            Err(FdarError::ComputationFailed { .. })
        ));
        assert!(matches!(
            iterative_shift_registration(&x, &t[..50], ShiftTemplate::Mean, &base),
            Err(FdarError::InvalidDimension {
                parameter: "argvals",
                ..
            })
        ));
        assert!(matches!(
            iterative_shift_registration(&FdMatrix::zeros(0, 51), &t, ShiftTemplate::Mean, &base),
            Err(FdarError::InvalidDimension {
                parameter: "data",
                ..
            })
        ));
        assert!(matches!(
            shift_curves(&x, &t, &[0.0; 4], ShiftExtrapolation::Boundary, false),
            Err(FdarError::InvalidDimension {
                parameter: "shifts",
                ..
            })
        ));
    }
}
