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
    /// `signum(value)`, so the derivative at exactly `0.0` is `0.0`.
    fn abs(self) -> Self;
    /// Sign (`-1`, `0`, or `1`). Constant almost everywhere, so tangent `0`.
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
/// `PartialOrd` compares the `value` field only (see the module-level docs).
#[derive(Debug, Clone, Copy, PartialEq)]
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
        // Subdifferential convention: d/dx |v| = signum(v), with signum(0) = 0.
        Dual {
            value: self.value.abs(),
            tangent: self.tangent * self.value.signum(),
        }
    }
    #[inline]
    fn signum(self) -> Self {
        // Step function: derivative is 0 almost everywhere.
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

    #[test]
    fn dual_constants() {
        assert_eq!(<Dual as Scalar>::zero(), Dual::constant(0.0));
        assert_eq!(<Dual as Scalar>::one(), Dual::constant(1.0));
        assert_eq!(<Dual as Scalar>::from_f64(4.5), Dual::constant(4.5));
        let inf = <Dual as Scalar>::infinity();
        assert!(inf.value.is_infinite() && inf.value > 0.0);
        assert_eq!(inf.tangent, 0.0);
    }
}
