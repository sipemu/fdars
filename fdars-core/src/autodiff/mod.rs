//! Automatic differentiation substrate — forward-mode ([`Dual`]) and
//! reverse-mode ([`Var`]).
//!
//! # Forward mode
//!
//! Seed an input's tangent to `1.0` (via [`Dual::seed`]), run a computation
//! written against [`Scalar`], and [`extract`](Dual::extract) the
//! `(value, derivative)` pair. The [`diff`] helper wraps this pattern; [`grad`]
//! and [`jacobian`] compute full gradients / Jacobians in `m` forward passes.
//!
//! # Reverse mode
//!
//! Build a Wengert-list tape by running a [`Scalar`]-generic closure on
//! [`Var`] inputs, then call [`vjp`] to seed the output adjoint and sweep
//! backward in one pass, collecting all input gradients simultaneously. This
//! is `O(cost(f))` regardless of the number of inputs, versus `O(m · cost(f))`
//! for forward-mode [`grad`].
//!
//! # Example — one objective, two modes
//!
//! A [`Scalar`]-generic objective can be differentiated by forward-mode [`grad`]
//! (seed each input) or reverse-mode [`vjp`] (one backward sweep). Both agree:
//!
//! ```
//! use fdars_core::autodiff::{grad, vjp, Dual, Scalar, Var};
//!
//! // f(x) = Σ_i sin(x_i) · exp(x_i) — written once, generic over the scalar type.
//! fn objective<T: Scalar>(x: &[T]) -> T {
//!     let mut acc = T::zero();
//!     for &xi in x {
//!         acc += xi.sin() * xi.exp();
//!     }
//!     acc
//! }
//!
//! let x = [0.3_f64, -0.7, 1.1];
//! let (fwd_val, fwd_grad) = grad(|d: &[Dual]| objective(d), &x);
//! let (rev_val, rev_grad) = vjp(|v: &[Var]| objective(v), &x);
//!
//! assert!((fwd_val - rev_val).abs() < 1e-12);
//! for i in 0..x.len() {
//!     // analytic: d/dx [sin x · e^x] = e^x (sin x + cos x)
//!     let analytic = x[i].exp() * (x[i].sin() + x[i].cos());
//!     assert!((fwd_grad[i] - analytic).abs() < 1e-9);
//!     assert!((rev_grad[i] - analytic).abs() < 1e-9);
//! }
//! ```

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Numeric substrate for automatic differentiation (forward and reverse mode).
///
/// A `Scalar` provides the arithmetic and transcendental operations a
/// differentiable computation is written against. It is implemented for `f64`
/// (a zero-cost passthrough), for [`Dual`] (which propagates tangents via the
/// chain rule), and for [`Var`] (which records operations on the
/// thread-local Wengert-list tape).
///
/// # Domain restrictions
///
/// The transcendental methods inherit the domain restrictions of the
/// underlying `f64` operations. In particular [`sqrt`](Scalar::sqrt) and
/// [`ln`](Scalar::ln) require an in-domain (non-negative / positive) value, and
/// [`powf`](Scalar::powf) can produce `NaN`/`Inf` tangents/adjoints at
/// `value == 0.0` with `p < 1.0`. Out-of-domain inputs propagate `NaN`/`Inf`
/// exactly as they do for plain `f64`; callers own range checking.
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
    /// Construct a constant from an `f64` (tangent `0` for [`Dual`]; off-tape
    /// sentinel for [`Var`]).
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
    /// Raise to a concrete `f64` power. The tangent/adjoint uses `p * v^(p-1)`;
    /// at `value == 0.0` with `p < 1.0` this may be `NaN`/`Inf`.
    fn powf(self, p: f64) -> Self;
    /// Absolute value. The tangent/adjoint uses the subdifferential convention
    /// `d/dx |v| = signum(v)`, with the honest at-zero selection
    /// `signum(0) = 0`: at exactly `value == 0.0` the tangent is `0.0` (the
    /// midpoint of the subdifferential `[-1, 1]`). The *value* is `v.abs()`
    /// (bit-for-bit `f64` parity).
    fn abs(self) -> Self;
    /// Sign. Piecewise-constant, so the tangent/adjoint is `0.0` everywhere.
    /// The *value* is `f64::signum(v)` (`+1.0` at `+0.0`, `-1.0` at `-0.0`) to
    /// preserve `f64` parity.
    fn signum(self) -> Self;
}

mod forward;
mod reverse;

// ---------------------------------------------------------------------------
// Re-exports — preserve all existing public paths:
//   autodiff::Dual, autodiff::Scalar, autodiff::Var,
//   autodiff::diff, autodiff::grad, autodiff::jacobian,
//   autodiff::directional_derivative, autodiff::vjp
// ---------------------------------------------------------------------------
pub use forward::{diff, directional_derivative, grad, jacobian, Dual};
pub use reverse::{vjp, Var};
