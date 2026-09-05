//! In-crate forward-mode automatic differentiation (AD) substrate.
//!
//! This module provides the numeric substrate for forward-mode automatic
//! differentiation used by the differentiable FDA subset. It defines a
//! [`Scalar`] trait bounding the arithmetic and transcendental operations a
//! differentiable computation needs, a forward-mode [`Dual`] number carrying a
//! value (primal) and a tangent (directional derivative), and a
//! zero-cost [`Scalar`] implementation for `f64`.
//!
//! # Design invariants
//!
//! - **No external dependency.** The [`Scalar`] trait is defined entirely
//!   in-crate. The `num-traits` crate is a *transitive-only* dependency of
//!   `fdars-core` and cannot be `use`d without a `Cargo.toml` change, which
//!   would violate the no-new-dependency constraint. This module therefore
//!   never imports it.
//! - **Additive / non-breaking.** `Scalar` is implemented for `f64` as a
//!   zero-cost passthrough to the inherent `f64` methods, so generic code
//!   instantiated at `f64` is numerically identical to plain `f64` code.
//! - **Value-only ordering for `Dual`.** [`Dual`]'s [`PartialOrd`] compares the
//!   `value` (primal) field ONLY. This is the correct forward-mode branching
//!   semantics: control-flow decisions are made on the primal while tangents
//!   propagate through the selected branch. Deriving `PartialOrd` would use the
//!   tangent as a lexicographic tiebreaker and corrupt those semantics.
//!
//! # Forward-mode in one line
//!
//! Seed an input's tangent to `1.0` (via [`Dual::seed`]), run a computation
//! written against [`Scalar`], and [`extract`](Dual::extract) the
//! `(value, derivative)` pair. The [`diff`] helper wraps this pattern.
//!
//! ```
//! use fdars_core::autodiff::{diff, Dual, Scalar};
//!
//! // f(x) = x^2, f'(x) = 2x. At x = 3: f = 9, f' = 6.
//! let (value, deriv) = diff(|x| x * x, 3.0);
//! assert!((value - 9.0).abs() < 1e-12);
//! assert!((deriv - 6.0).abs() < 1e-12);
//! ```
//!
//! # Composing differentiable ops
//!
//! [`grad`] flows a gradient through a composition of the crate's
//! `Scalar`-generic differentiable ops. Here one scalar objective composes a
//! soft-DTW distance and an FPCA-score projection, then `grad` returns the
//! objective value and its full gradient w.r.t. the input curve's samples.
//!
//! ```
//! use fdars_core::prelude::*;
//! use fdars_core::regression::fdata_to_pc_1d;
//!
//! // Small trained FPCA model (mirrors regression::fdata_to_pc_1d usage).
//! let m = 10usize;
//! let n = 12usize;
//! let argvals: Vec<f64> = (0..m).map(|j| 0.1 + 0.8 * j as f64 / (m - 1) as f64).collect();
//! let mut raw = vec![0.0f64; n * m];
//! for i in 0..n {
//!     for (j, &t) in argvals.iter().enumerate() {
//!         let phase = i as f64 * 0.3;
//!         raw[i + j * n] = (std::f64::consts::PI * t + phase).sin()
//!             + 0.5 * (2.0 * std::f64::consts::PI * t).cos();
//!     }
//! }
//! let data = FdMatrix::from_column_major(raw, n, m).unwrap();
//! let fpca = fdata_to_pc_1d(&data, 2, &argvals).unwrap();
//!
//! // Objective: soft-DTW(curve, reference) + sum of squared FPCA scores.
//! let reference: Vec<Dual> = argvals
//!     .iter()
//!     .map(|&t| Dual::constant((std::f64::consts::PI * t).sin()))
//!     .collect();
//! let curve: Vec<f64> = argvals
//!     .iter()
//!     .map(|&t| (std::f64::consts::PI * t).cos())
//!     .collect();
//!
//! let objective = |c: &[Dual]| -> Dual {
//!     let sdtw = soft_dtw_distance_generic(c, &reference, 0.1);
//!     let scores = project_scores_generic(c, &fpca.mean, &fpca.rotation, &fpca.weights, 2);
//!     let mut acc = Dual::constant(0.0);
//!     for s in &scores {
//!         acc += *s * *s;
//!     }
//!     sdtw + acc
//! };
//!
//! let (value, gradient) = grad(objective, &curve);
//! assert_eq!(gradient.len(), m);
//! assert!(value.is_finite());
//! ```

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Numeric substrate for forward-mode automatic differentiation.
///
/// A `Scalar` provides the arithmetic and transcendental operations a
/// differentiable computation is written against. It is implemented for `f64`
/// (a zero-cost passthrough) and for [`Dual`] (which propagates tangents via
/// the chain rule).
///
/// # Domain restrictions
///
/// The transcendental methods inherit the domain restrictions of the
/// underlying `f64` operations. In particular [`sqrt`](Scalar::sqrt) and
/// [`ln`](Scalar::ln) require an in-domain (non-negative / positive) value, and
/// [`powf`](Scalar::powf) can produce `NaN`/`Inf` tangents at `value == 0.0`
/// with `p < 1.0`. Out-of-domain inputs propagate `NaN`/`Inf` exactly as they
/// do for plain `f64`; callers own range checking.
pub trait Scalar:
    Copy
    + Clone
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The additive identity (`0`).
    fn zero() -> Self;
    /// The multiplicative identity (`1`).
    fn one() -> Self;
    /// Construct a constant from an `f64` (tangent `0` for [`Dual`]).
    fn from_f64(v: f64) -> Self;
    /// Positive infinity sentinel (used e.g. for DP recurrence initialization).
    fn infinity() -> Self;

    /// Square root. Requires a non-negative value; the derivative diverges at 0.
    fn sqrt(self) -> Self;
    /// Natural exponential.
    fn exp(self) -> Self;
    /// Natural logarithm. Requires a strictly positive value.
    fn ln(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Raise to a concrete `f64` power. The tangent uses `p * v^(p-1)`; at
    /// `value == 0.0` with `p < 1.0` this may be `NaN`/`Inf`.
    fn powf(self, p: f64) -> Self;
    /// Absolute value. The tangent uses the subdifferential convention
    /// `d/dx |v| = signum(v)`, with the honest at-zero selection
    /// `signum(0) = 0`: at exactly `value == 0.0` the tangent is `0.0` (the
    /// midpoint of the subdifferential `[-1, 1]`). The *value* is `v.abs()`
    /// (bit-for-bit `f64` parity).
    fn abs(self) -> Self;
    /// Sign. Piecewise-constant, so the tangent is `0.0` everywhere. The
    /// *value* is `f64::signum(v)` (`+1.0` at `+0.0`, `-1.0` at `-0.0`) to
    /// preserve `f64` parity.
    fn signum(self) -> Self;
}

impl Scalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn infinity() -> Self {
        f64::INFINITY
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f64::signum(self)
    }
}

/// A forward-mode dual number: a value (primal) paired with a tangent
/// (directional derivative).
///
/// Running a computation written against [`Scalar`] on `Dual` propagates the
/// derivative through every operation via the chain rule. Seed an input's
/// tangent to `1.0` with [`Dual::seed`], then [`extract`](Dual::extract) the
/// `(value, derivative)` pair.
///
/// Both `PartialEq` and `PartialOrd` compare the `value` (primal) field only,
/// so equality and ordering agree on the "primal decides control flow"
/// semantics (see the module-level docs). Two `Dual`s with equal value but
/// different tangents therefore compare *equal* and *unordered-as-Equal*; this
/// keeps `a == b ⟺ a.partial_cmp(&b) == Some(Equal)` — the std contract that a
/// derived (both-field) `PartialEq` would violate against the value-only
/// `PartialOrd`.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    /// The primal value of the computation.
    pub value: f64,
    /// The tangent (directional derivative) accumulated by the chain rule.
    pub tangent: f64,
}

impl Dual {
    /// Seed an independent variable: `value = x`, `tangent = 1.0`.
    ///
    /// Use this on the input you are differentiating with respect to.
    #[inline]
    #[must_use]
    pub fn seed(value: f64) -> Self {
        Dual {
            value,
            tangent: 1.0,
        }
    }

    /// A constant: `value = x`, `tangent = 0.0` (no gradient flows through it).
    #[inline]
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Dual {
            value,
            tangent: 0.0,
        }
    }

    /// Extract the `(value, derivative)` pair after a computation.
    #[inline]
    #[must_use]
    pub fn extract(self) -> (f64, f64) {
        (self.value, self.tangent)
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Dual {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: d(u*v) = u'*v + u*v'
        Dual {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: d(u/v) = (u'*v - u*v') / v^2
        let v2 = rhs.value * rhs.value;
        Dual {
            value: self.value / rhs.value,
            tangent: (self.tangent * rhs.value - self.value * rhs.tangent) / v2,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Dual {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Dual {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// Hand-written value-only equality, matching the value-only `PartialOrd` below.
// Deriving `PartialEq` would compare the tangent too, breaking the std contract
// `a == b ⟺ a.partial_cmp(&b) == Some(Equal)` for equal-value/different-tangent
// Duals. Equality is primal-value-based (branch semantics).
impl PartialEq for Dual {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// Hand-written value-only ordering. Do NOT `#[derive(PartialOrd)]`: derive would
// use the tangent as a lexicographic tiebreaker, corrupting forward-mode branch
// semantics (control flow must be decided by the primal value alone).
impl PartialOrd for Dual {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Scalar for Dual {
    #[inline]
    fn zero() -> Self {
        Dual {
            value: 0.0,
            tangent: 0.0,
        }
    }
    #[inline]
    fn one() -> Self {
        Dual {
            value: 1.0,
            tangent: 0.0,
        }
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        Dual {
            value: v,
            tangent: 0.0,
        }
    }
    #[inline]
    fn infinity() -> Self {
        Dual {
            value: f64::INFINITY,
            tangent: 0.0,
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        // d/dx sqrt(v) = 1 / (2*sqrt(v))
        let s = self.value.sqrt();
        Dual {
            value: s,
            tangent: self.tangent / (2.0 * s),
        }
    }
    #[inline]
    fn exp(self) -> Self {
        // d/dx exp(v) = exp(v)
        let e = self.value.exp();
        Dual {
            value: e,
            tangent: self.tangent * e,
        }
    }
    #[inline]
    fn ln(self) -> Self {
        // d/dx ln(v) = 1/v
        Dual {
            value: self.value.ln(),
            tangent: self.tangent / self.value,
        }
    }
    #[inline]
    fn sin(self) -> Self {
        // d/dx sin(v) = cos(v)
        Dual {
            value: self.value.sin(),
            tangent: self.tangent * self.value.cos(),
        }
    }
    #[inline]
    fn cos(self) -> Self {
        // d/dx cos(v) = -sin(v)
        Dual {
            value: self.value.cos(),
            tangent: -self.tangent * self.value.sin(),
        }
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        // d/dx v^p = p * v^(p-1)
        Dual {
            value: self.value.powf(p),
            tangent: self.tangent * p * self.value.powf(p - 1.0),
        }
    }
    #[inline]
    fn abs(self) -> Self {
        // Subdifferential convention: d/dx |v| = signum(v), with the honest
        // at-zero selection signum(0) = 0 (f64::signum returns ±1 at zero, so
        // special-case exact zero). Value stays v.abs() for f64 parity.
        let sub = if self.value == 0.0 {
            0.0
        } else {
            self.value.signum()
        };
        Dual {
            value: self.value.abs(),
            tangent: self.tangent * sub,
        }
    }
    #[inline]
    fn signum(self) -> Self {
        // Piecewise-constant: derivative is 0 everywhere. Value uses
        // f64::signum for parity (returns ±1 at zero, only NaN yields NaN).
        Dual {
            value: self.value.signum(),
            tangent: 0.0,
        }
    }
}

/// Compute `f(x)` and `f'(x)` in one forward-mode pass.
///
/// Seeds `x` (tangent `1.0`), runs `f`, and returns the extracted
/// `(value, derivative)` pair.
///
/// ```
/// use fdars_core::autodiff::{diff, Scalar};
///
/// // d/dx exp(x) at x = 0 is 1.
/// let (v, d) = diff(|x| Scalar::exp(x), 0.0);
/// assert!((v - 1.0).abs() < 1e-12);
/// assert!((d - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn diff<F: Fn(Dual) -> Dual>(f: F, x: f64) -> (f64, f64) {
    f(Dual::seed(x)).extract()
}

/// Compute a scalar objective's value and its full gradient over an `m`-vector
/// input, in `m` forward-mode passes (one per input).
///
/// This is the multi-input generalization of [`diff`]: for each input index
/// `k`, it builds the argument vector where element `k` is
/// [`Dual::seed`]ed (tangent `1.0`, the variable being differentiated) and every
/// other element `j` is a [`Dual::constant`] (tangent `0.0`), runs `f`, and
/// records the tangent of the result as `gradient[k]`. The primal `value` is
/// identical across passes (only tangents differ), so it is captured once.
///
/// Returns `(value, gradient)` where `gradient.len() == x.len()`. An empty
/// input yields `(f(&[]).value, Vec::new())`.
///
/// ```
/// use fdars_core::autodiff::{grad, Dual, Scalar};
///
/// // f(x) = x0^2 + x1^2. Gradient = [2*x0, 2*x1]. At [3, 4]: value 25, grad [6, 8].
/// let (value, gradient) = grad(|x| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0]);
/// assert!((value - 25.0).abs() < 1e-12);
/// assert!((gradient[0] - 6.0).abs() < 1e-12);
/// assert!((gradient[1] - 8.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
    let m = x.len();
    if m == 0 {
        return (f(&[]).value, Vec::new());
    }
    let mut gradient = vec![0.0; m];
    let mut value = 0.0;
    for k in 0..m {
        let duals: Vec<Dual> = (0..m)
            .map(|j| {
                if j == k {
                    Dual::seed(x[j])
                } else {
                    Dual::constant(x[j])
                }
            })
            .collect();
        let (v, t) = f(&duals).extract();
        if k == 0 {
            value = v;
        }
        gradient[k] = t;
    }
    (value, gradient)
}

/// Compute a vector-valued map's values and its full Jacobian over an `m`-vector
/// input, in `m` forward-mode passes (one per input).
///
/// `f` returns a length-`n` `Vec<Dual>` (the outputs). Seeding input `k` in turn
/// (as in [`grad`]) fills column `k` of the returned `n × m` Jacobian, where
/// `jacobian[i][k] = d(output_i)/d(x[k])`. Output primal values are captured on
/// the first pass.
///
/// Returns `(values, jacobian)` with `values.len() == n`, `jacobian.len() == n`,
/// and each row of length `m`. An empty input yields `(values, empty rows)`.
///
/// ```
/// use fdars_core::autodiff::{jacobian, Dual};
///
/// // f(x) = [x0*x1, x0 + x1]. J = [[x1, x0], [1, 1]]. At [2, 3]: [[3, 2], [1, 1]].
/// let (values, j) = jacobian(|x| vec![x[0] * x[1], x[0] + x[1]], &[2.0, 3.0]);
/// assert!((values[0] - 6.0).abs() < 1e-12);
/// assert!((values[1] - 5.0).abs() < 1e-12);
/// assert!((j[0][0] - 3.0).abs() < 1e-12 && (j[0][1] - 2.0).abs() < 1e-12);
/// assert!((j[1][0] - 1.0).abs() < 1e-12 && (j[1][1] - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn jacobian<F: Fn(&[Dual]) -> Vec<Dual>>(f: F, x: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let m = x.len();
    if m == 0 {
        let outputs = f(&[]);
        let values: Vec<f64> = outputs.iter().map(|d| d.value).collect();
        let rows = values.len();
        return (values, vec![Vec::new(); rows]);
    }
    let mut values: Vec<f64> = Vec::new();
    let mut jac: Vec<Vec<f64>> = Vec::new();
    for k in 0..m {
        let duals: Vec<Dual> = (0..m)
            .map(|j| {
                if j == k {
                    Dual::seed(x[j])
                } else {
                    Dual::constant(x[j])
                }
            })
            .collect();
        let outputs = f(&duals);
        if k == 0 {
            values = outputs.iter().map(|d| d.value).collect();
            jac = vec![vec![0.0; m]; outputs.len()];
        }
        for (i, out) in outputs.iter().enumerate() {
            jac[i][k] = out.tangent;
        }
    }
    (values, jac)
}

/// Compute a scalar objective's value and its directional derivative along a
/// supplied `direction`, in a SINGLE forward-mode pass.
///
/// Each input `j` is lifted to `Dual { value: x[j], tangent: direction[j] }`, so
/// the returned tangent is `∇f(x) · direction`. Requires
/// `direction.len() == x.len()` (checked with `debug_assert`).
///
/// ```
/// use fdars_core::autodiff::{directional_derivative, Dual};
///
/// // f(x) = x0^2 + x1^2, ∇f = [2*x0, 2*x1]. At [1, 2] along [1, 0]: dir-deriv = 2.
/// let (value, dd) = directional_derivative(|x| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0], &[1.0, 0.0]);
/// assert!((value - 5.0).abs() < 1e-12);
/// assert!((dd - 2.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn directional_derivative<F: Fn(&[Dual]) -> Dual>(
    f: F,
    x: &[f64],
    direction: &[f64],
) -> (f64, f64) {
    debug_assert_eq!(
        direction.len(),
        x.len(),
        "direction length must match input length"
    );
    let duals: Vec<Dual> = x
        .iter()
        .zip(direction.iter())
        .map(|(&value, &tangent)| Dual { value, tangent })
        .collect();
    f(&duals).extract()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    // ---------------------------------------------------------------------
    // Tier 1: known-answer derivatives, tolerance 1e-10 (one per op).
    // ---------------------------------------------------------------------

    #[test]
    fn dual_mul_known_answer() {
        // f(x) = x^2, f'(x) = 2x. At x = 3: f = 9, f' = 6.
        let d = Dual::seed(3.0);
        let r = d * d;
        assert!((r.value - 9.0).abs() < TOL, "primal {} != 9.0", r.value);
        assert!(
            (r.tangent - 6.0).abs() < TOL,
            "tangent {} != 6.0",
            r.tangent
        );
    }

    #[test]
    fn dual_sqrt_known_answer() {
        // f(x) = sqrt(x), f'(x) = 1/(2 sqrt(x)). At x = 4: f = 2, f' = 0.25.
        let r = Scalar::sqrt(Dual::seed(4.0));
        assert!((r.value - 2.0).abs() < TOL);
        assert!((r.tangent - 0.25).abs() < TOL);
    }

    #[test]
    fn dual_exp_known_answer() {
        // f(x) = exp(x), f'(x) = exp(x). At x = 1: both = e.
        let e = std::f64::consts::E;
        let r = Scalar::exp(Dual::seed(1.0));
        assert!((r.value - e).abs() < TOL);
        assert!((r.tangent - e).abs() < TOL);
    }

    #[test]
    fn dual_ln_known_answer() {
        // f(x) = ln(x), f'(x) = 1/x. At x = 2: f = ln 2, f' = 0.5.
        let r = Scalar::ln(Dual::seed(2.0));
        assert!((r.value - 2.0_f64.ln()).abs() < TOL);
        assert!((r.tangent - 0.5).abs() < TOL);
    }

    #[test]
    fn dual_sin_known_answer() {
        // f(x) = sin(x), f'(x) = cos(x). At x = PI/4: both = sqrt(2)/2.
        let expected = 2.0_f64.sqrt() / 2.0;
        let r = Scalar::sin(Dual::seed(PI / 4.0));
        assert!((r.value - expected).abs() < TOL);
        assert!((r.tangent - expected).abs() < TOL);
    }

    #[test]
    fn dual_cos_known_answer() {
        // f(x) = cos(x), f'(x) = -sin(x). At x = PI/4: value sqrt(2)/2, deriv -sqrt(2)/2.
        let expected = 2.0_f64.sqrt() / 2.0;
        let r = Scalar::cos(Dual::seed(PI / 4.0));
        assert!((r.value - expected).abs() < TOL);
        assert!((r.tangent + expected).abs() < TOL);
    }

    #[test]
    fn dual_powf_known_answer() {
        // f(x) = x^1.5, f'(x) = 1.5 x^0.5. At x = 4: f = 8, f' = 1.5*2 = 3.
        let r = Scalar::powf(Dual::seed(4.0), 1.5);
        assert!((r.value - 8.0).abs() < TOL);
        assert!((r.tangent - 3.0).abs() < TOL);
    }

    #[test]
    fn dual_abs_known_answer() {
        // f(x) = |x|, f'(x) = signum(x). At x = 2: f = 2, f' = 1. (Not probed at 0.)
        let r = Scalar::abs(Dual::seed(2.0));
        assert!((r.value - 2.0).abs() < TOL);
        assert!((r.tangent - 1.0).abs() < TOL);
        // And on the negative branch.
        let rn = Scalar::abs(Dual::seed(-3.0));
        assert!((rn.value - 3.0).abs() < TOL);
        assert!((rn.tangent + 1.0).abs() < TOL);
    }

    #[test]
    fn dual_sub_div_neg_known_answer() {
        // f(x) = (x - 1) / 2, f'(x) = 0.5.
        let d = Dual::seed(5.0);
        let r = (d - Dual::constant(1.0)) / Dual::constant(2.0);
        assert!((r.value - 2.0).abs() < TOL);
        assert!((r.tangent - 0.5).abs() < TOL);
        // f(x) = -x, f'(x) = -1.
        let n = -Dual::seed(7.0);
        assert!((n.value + 7.0).abs() < TOL);
        assert!((n.tangent + 1.0).abs() < TOL);
    }

    #[test]
    fn dual_assign_ops() {
        // += , -=, *= must compose the same chain rules as their binary forms.
        let mut acc = Dual::constant(0.0);
        let x = Dual::seed(2.0);
        acc += x; // acc = x           -> value 2, tangent 1
        acc *= x; // acc = x^2         -> value 4, tangent 4  (2*x*x')
        acc -= Dual::constant(1.0); // acc = x^2 - 1 -> value 3, tangent 4
        assert!((acc.value - 3.0).abs() < TOL);
        assert!((acc.tangent - 4.0).abs() < TOL);
    }

    #[test]
    fn dual_composed_chain_known_answer() {
        // f(x) = sqrt(exp(x)*sin(x) + x^2). Make-or-break composed chain.
        // f'(x) = (1/(2 f(x))) * (e^x*(sin x + cos x) + 2x).
        let x0 = 1.0_f64;
        let (value, deriv) = diff(
            |x| {
                let e = Scalar::exp(x);
                let s = Scalar::sin(x);
                let x2 = x * x;
                Scalar::sqrt(e * s + x2)
            },
            x0,
        );
        let f = (x0.exp() * x0.sin() + x0 * x0).sqrt();
        let expected_deriv = (1.0 / (2.0 * f)) * (x0.exp() * (x0.sin() + x0.cos()) + 2.0 * x0);
        assert!((value - f).abs() < TOL, "value {value} != {f}");
        assert!(
            (deriv - expected_deriv).abs() < TOL,
            "deriv {deriv} != {expected_deriv}"
        );
    }

    #[test]
    fn dual_partial_cmp_value_only() {
        use std::cmp::Ordering;
        // Equal value, different tangent -> compares Equal (value-only).
        let a = Dual {
            value: 1.0,
            tangent: 0.5,
        };
        let b = Dual {
            value: 1.0,
            tangent: 0.3,
        };
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        // Smaller value is Less regardless of tangent.
        let small = Dual {
            value: 0.5,
            tangent: 99.0,
        };
        let big = Dual {
            value: 2.0,
            tangent: -99.0,
        };
        assert_eq!(small.partial_cmp(&big), Some(Ordering::Less));
        assert!(small < big);
    }

    #[test]
    fn dual_eq_is_value_only_and_consistent_with_ord() {
        use std::cmp::Ordering;
        // Equal value, different tangent: PartialEq (value-only) says equal,
        // and this must agree with PartialOrd == Some(Equal) (std contract).
        let a = Dual {
            value: 1.0,
            tangent: 0.5,
        };
        let b = Dual {
            value: 1.0,
            tangent: -7.0,
        };
        assert_eq!(a, b);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        // Different value: not equal.
        let c = Dual {
            value: 2.0,
            tangent: 0.5,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn dual_abs_at_zero_tangent_is_zero() {
        // HI-01: honest subdifferential selection signum(0) = 0, so abs of a
        // Dual seeded at exactly 0.0 yields tangent 0.0 (not ±1 from f64::signum).
        let r = Scalar::abs(Dual::seed(0.0));
        assert_eq!(r.value, 0.0);
        assert_eq!(r.tangent, 0.0);
        // Negative-zero primal also selects the 0 subgradient.
        let rn = Scalar::abs(Dual::seed(-0.0));
        assert_eq!(rn.value, 0.0);
        assert_eq!(rn.tangent, 0.0);
    }

    #[test]
    fn dual_signum_tangent_is_zero_value_is_f64_signum() {
        // Tangent is 0 everywhere; value preserves f64::signum parity (±1 at 0).
        let rp = Scalar::signum(Dual::seed(3.0));
        assert_eq!(rp.value, 1.0);
        assert_eq!(rp.tangent, 0.0);
        let rn = Scalar::signum(Dual::seed(-3.0));
        assert_eq!(rn.value, -1.0);
        assert_eq!(rn.tangent, 0.0);
        // f64::signum returns +1.0 at +0.0 (value parity), tangent still 0.
        let rz = Scalar::signum(Dual::seed(0.0));
        assert_eq!(rz.value, 1.0);
        assert_eq!(rz.tangent, 0.0);
    }

    // ---------------------------------------------------------------------
    // Guard tests: lock documented singular-point behavior (LO-02, LO-03).
    // These pin the "callers own range checking" contract so a future refactor
    // (e.g. clamping a denominator) cannot silently change the singular result.
    // ---------------------------------------------------------------------

    #[test]
    fn dual_sqrt_at_zero_tangent_is_nonfinite() {
        // LO-02: sqrt(0) tangent = 1/(2*0) diverges (non-finite).
        let r = Scalar::sqrt(Dual::seed(0.0));
        assert_eq!(r.value, 0.0);
        assert!(
            !r.tangent.is_finite(),
            "tangent {} should be non-finite",
            r.tangent
        );
    }

    #[test]
    fn dual_ln_at_zero_tangent_is_nonfinite() {
        // LO-02: ln(0) value = -inf, tangent = 1/0 diverges (non-finite).
        let r = Scalar::ln(Dual::seed(0.0));
        assert!(r.value.is_infinite() && r.value < 0.0);
        assert!(
            !r.tangent.is_finite(),
            "tangent {} should be non-finite",
            r.tangent
        );
    }

    #[test]
    fn dual_powf_singular_and_linear_edges() {
        // LO-03: powf(0, 0.5) tangent = 0.5 * 0^(-0.5) = Inf (documented divergence).
        let r = Scalar::powf(Dual::seed(0.0), 0.5);
        assert_eq!(r.value, 0.0);
        assert!(
            r.tangent.is_infinite(),
            "tangent {} should be infinite",
            r.tangent
        );
        // powf(0, 1.0) relies on 0^0 == 1.0, giving tangent 1*1*1 = 1 (d/dx x = 1).
        let lin = Scalar::powf(Dual::seed(0.0), 1.0);
        assert_eq!(lin.value, 0.0);
        assert_eq!(lin.tangent, 1.0);
        // Negative base with non-integer power: value and tangent both NaN.
        let nan = Scalar::powf(Dual::seed(-2.0), 0.5);
        assert!(nan.value.is_nan());
        assert!(nan.tangent.is_nan());
    }

    // ---------------------------------------------------------------------
    // Tier 2: central finite-difference cross-check, tolerance 1e-6.
    // ---------------------------------------------------------------------

    fn central_fd(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-8_f64;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn finite_diff_cross_check_composed() {
        let x0 = 1.0_f64;
        let (_, ad) = diff(
            |x| {
                let e = Scalar::exp(x);
                let s = Scalar::sin(x);
                Scalar::sqrt(e * s + x * x)
            },
            x0,
        );
        let fd = central_fd(|x| (x.exp() * x.sin() + x * x).sqrt(), x0);
        assert!((ad - fd).abs() < 1e-6, "AD {ad} FD {fd}");
    }

    #[test]
    fn finite_diff_cross_check_log_trig() {
        // f(x) = ln(x) * cos(x) at x = 2.
        let x0 = 2.0_f64;
        let (_, ad) = diff(|x| Scalar::ln(x) * Scalar::cos(x), x0);
        let fd = central_fd(|x| x.ln() * x.cos(), x0);
        assert!((ad - fd).abs() < 1e-6, "AD {ad} FD {fd}");
    }

    #[test]
    fn finite_diff_cross_check_powf_exp() {
        // f(x) = x^1.5 / exp(x) at x = 1.5.
        let x0 = 1.5_f64;
        let (_, ad) = diff(|x| Scalar::powf(x, 1.5) / Scalar::exp(x), x0);
        let fd = central_fd(|x| x.powf(1.5) / x.exp(), x0);
        assert!((ad - fd).abs() < 1e-6, "AD {ad} FD {fd}");
    }

    // ---------------------------------------------------------------------
    // Tier 3: f64 parity — bit-for-bit vs direct f64 methods.
    // ---------------------------------------------------------------------

    #[test]
    fn f64_parity_transcendentals() {
        let x = 2.5_f64;
        assert_eq!(<f64 as Scalar>::sqrt(x), x.sqrt());
        assert_eq!(<f64 as Scalar>::exp(x), x.exp());
        assert_eq!(<f64 as Scalar>::ln(x), x.ln());
        assert_eq!(<f64 as Scalar>::sin(x), x.sin());
        assert_eq!(<f64 as Scalar>::cos(x), x.cos());
        assert_eq!(<f64 as Scalar>::powf(x, 1.5), x.powf(1.5));
        assert_eq!(<f64 as Scalar>::abs(-x), (-x).abs());
        assert_eq!(<f64 as Scalar>::signum(-x), (-x).signum());
    }

    #[test]
    fn f64_parity_constants() {
        assert_eq!(<f64 as Scalar>::zero(), 0.0);
        assert_eq!(<f64 as Scalar>::one(), 1.0);
        assert_eq!(<f64 as Scalar>::from_f64(3.25), 3.25);
        assert_eq!(<f64 as Scalar>::infinity(), f64::INFINITY);
    }

    // ---------------------------------------------------------------------
    // grad / jacobian / directional_derivative (DIF-04 SC #1).
    // ---------------------------------------------------------------------

    #[test]
    fn grad_sum_of_squares_closed_form() {
        // f(x) = sum(x_i^2), grad = [2*x_i]. At [1,2,3]: value 14, grad [2,4,6].
        let (value, gradient) = grad(
            |x| {
                let mut acc = Dual::constant(0.0);
                for &xi in x {
                    acc += xi * xi;
                }
                acc
            },
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(gradient.len(), 3);
        assert!((value - 14.0).abs() <= 1e-12, "value {value} != 14.0");
        for (g, expected) in gradient.iter().zip([2.0, 4.0, 6.0]) {
            assert!((g - expected).abs() <= 1e-12, "grad {g} != {expected}");
        }
    }

    #[test]
    fn grad_single_input_agrees_with_diff() {
        // Single-element input [3.0], f = x0^2 -> (9.0, [6.0]) matches diff.
        let (value, gradient) = grad(|x| x[0] * x[0], &[3.0]);
        assert_eq!(gradient.len(), 1);
        assert!((value - 9.0).abs() <= 1e-12);
        assert!((gradient[0] - 6.0).abs() <= 1e-12);
        let (dv, dd) = diff(|x| x * x, 3.0);
        assert!((value - dv).abs() <= 1e-12);
        assert!((gradient[0] - dd).abs() <= 1e-12);
    }

    #[test]
    fn grad_empty_input_returns_constant() {
        // m == 0: evaluate f once, empty gradient, never index x.
        let (value, gradient) = grad(|_x| Dual::constant(7.0), &[]);
        assert!((value - 7.0).abs() <= 1e-12);
        assert!(gradient.is_empty());
    }

    #[test]
    fn jacobian_known_answer() {
        // f(x) = [x0*x1, x0 + x1]; J = [[x1, x0], [1, 1]]; at [2,3] -> [[3,2],[1,1]].
        let (values, j) = jacobian(|x| vec![x[0] * x[1], x[0] + x[1]], &[2.0, 3.0]);
        assert_eq!(values.len(), 2);
        assert!((values[0] - 6.0).abs() <= 1e-12);
        assert!((values[1] - 5.0).abs() <= 1e-12);
        assert_eq!(j.len(), 2);
        assert!((j[0][0] - 3.0).abs() <= 1e-12 && (j[0][1] - 2.0).abs() <= 1e-12);
        assert!((j[1][0] - 1.0).abs() <= 1e-12 && (j[1][1] - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn directional_derivative_projects_gradient() {
        // f(x) = x0^2 + x1^2, grad [2*x0, 2*x1]. At [1,2] along [1,0]: dd = 2.
        let (value, dd) =
            directional_derivative(|x| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0], &[1.0, 0.0]);
        assert!((value - 5.0).abs() <= 1e-12);
        assert!((dd - 2.0).abs() <= 1e-12);
    }

    // ---------------------------------------------------------------------
    // Composed-objective demo + central finite-difference cross-check
    // (DIF-04 SC #2): grad flows through soft_dtw + FPCA-score projection.
    // ---------------------------------------------------------------------

    #[test]
    fn grad_composed_objective_matches_finite_diff() {
        use crate::matrix::FdMatrix;
        use crate::metric::soft_dtw_distance_generic;
        use crate::regression::{fdata_to_pc_1d, project_scores_generic};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let m = 24usize;
        let n = 40usize;
        let ncomp = 3usize;
        let gamma = 0.1_f64;
        let lambda = 1.0_f64;

        // Grid on [0.1, 0.9] avoids SRSF derivative zeros / degenerate points.
        let argvals: Vec<f64> = (0..m)
            .map(|j| 0.1 + 0.8 * j as f64 / (m - 1) as f64)
            .collect();

        // Spanning full-rank training set: seeded random combos of three basis
        // functions with distinct coefficients (n >> m). Column-major FdMatrix.
        let mut rng = StdRng::seed_from_u64(20260906);
        let mut data = vec![0.0f64; n * m];
        for i in 0..n {
            let a: f64 = rng.gen_range(-1.0..1.0);
            let b: f64 = rng.gen_range(-1.0..1.0);
            let c: f64 = rng.gen_range(-1.0..1.0);
            for (j, &t) in argvals.iter().enumerate() {
                let v = a * (PI * t).sin() + b * (2.0 * PI * t).cos() + c * (3.0 * PI * t).sin();
                data[i + j * n] = v;
            }
        }
        let data = FdMatrix::from_column_major(data, n, m).unwrap();
        let fpca = fdata_to_pc_1d(&data, ncomp, &argvals).unwrap();
        let mean = fpca.mean.clone();
        let rotation = fpca.rotation.clone();
        let weights = fpca.weights.clone();

        // Input curve + reference curve: further spanning combos, nonzero deriv.
        let curve: Vec<f64> = argvals
            .iter()
            .map(|&t| {
                0.7 * (PI * t).sin() - 0.4 * (2.0 * PI * t).cos() + 0.3 * (3.0 * PI * t).sin()
            })
            .collect();
        let reference: Vec<f64> = argvals
            .iter()
            .map(|&t| {
                0.2 * (PI * t).sin() + 0.5 * (2.0 * PI * t).cos() - 0.6 * (3.0 * PI * t).sin()
            })
            .collect();
        let reference_duals: Vec<Dual> = reference.iter().map(|&r| Dual::constant(r)).collect();

        // Composed scalar objective: soft-DTW (op 1) + lambda * sum(scores^2) (op 2).
        let objective = |c: &[Dual]| -> Dual {
            let sdtw = soft_dtw_distance_generic(c, &reference_duals, gamma);
            let scores = project_scores_generic(c, &mean, &rotation, &weights, ncomp);
            let mut acc = Dual::constant(0.0);
            for s in &scores {
                acc += *s * *s;
            }
            sdtw + Dual::constant(lambda) * acc
        };

        let (value, gradient) = grad(objective, &curve);
        assert_eq!(gradient.len(), m);
        assert!(value.is_finite(), "objective value not finite: {value}");
        assert!(value > 0.0, "objective value not positive: {value}");

        // f64 reference for composition parity + central finite differences.
        let f64_obj = |c: &[f64]| -> f64 {
            let d = soft_dtw_distance_generic::<f64>(c, &reference, gamma);
            let sc = project_scores_generic::<f64>(c, &mean, &rotation, &weights, ncomp);
            d + lambda * sc.iter().map(|s| s * s).sum::<f64>()
        };

        // Composition parity at f64.
        assert!(
            (value - f64_obj(&curve)).abs() < 1e-12,
            "composition parity broke: {value}"
        );

        // Central FD (h = 1e-6) cross-check on every gradient component.
        let h = 1e-6_f64;
        for j in 0..m {
            let mut plus = curve.clone();
            let mut minus = curve.clone();
            plus[j] += h;
            minus[j] -= h;
            let fd = (f64_obj(&plus) - f64_obj(&minus)) / (2.0 * h);
            assert!(
                (gradient[j] - fd).abs() < 1e-6,
                "component {j}: AD {} vs FD {fd}",
                gradient[j]
            );
        }
    }

    #[test]
    fn dual_constants() {
        // `Dual`'s `PartialEq` is value-only, so assert both fields explicitly
        // to genuinely verify the tangent is 0.0 for these constants.
        let z = <Dual as Scalar>::zero();
        assert_eq!(z.value, 0.0);
        assert_eq!(z.tangent, 0.0);
        let o = <Dual as Scalar>::one();
        assert_eq!(o.value, 1.0);
        assert_eq!(o.tangent, 0.0);
        let c = <Dual as Scalar>::from_f64(4.5);
        assert_eq!(c.value, 4.5);
        assert_eq!(c.tangent, 0.0);
        let inf = <Dual as Scalar>::infinity();
        assert!(inf.value.is_infinite() && inf.value > 0.0);
        assert_eq!(inf.tangent, 0.0);
    }
}
