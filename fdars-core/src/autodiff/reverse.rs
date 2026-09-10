//! Reverse-mode (VJP) automatic differentiation substrate.
//!
//! Implements a Wengert-list tape approach to reverse-mode AD:
//! - [`Var`] — a Copy scalar handle carrying a primal value and a tape node index.
//! - [`vjp`] — run a [`Scalar`]-generic closure forward (building the tape),
//!   seed the output adjoint = 1.0, and sweep backward in one pass to collect
//!   all input gradients.
//!
//! The tape is thread-local, so `vjp` calls on different rayon worker threads
//! never interfere — the same pattern used by `alignment/mod.rs` for DP scratch.

use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Scalar;

// ---------------------------------------------------------------------------
// Sentinel: a node index meaning "off-tape / constant, no adjoint contribution".
// ---------------------------------------------------------------------------

/// Node index indicating a constant `Var` (not on tape).
///
/// Constants produced by `Scalar::from_f64`, `zero`, `one`, and `infinity`
/// carry `node = SENTINEL`. The backward pass skips sentinel parents, so
/// constant values correctly contribute zero adjoint.
const SENTINEL: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Node record
// ---------------------------------------------------------------------------

/// A single record on the Wengert-list tape.
///
/// Covers both binary ops (both slots populated) and unary ops (slot 1 =
/// `SENTINEL`, weight 1 = `0.0`). Constants produce no node — they use
/// `SENTINEL` as their `node` index in the owning [`Var`].
#[derive(Clone, Copy)]
struct Node {
    /// Parent node indices. `SENTINEL` = no contribution for that slot.
    deps: [usize; 2],
    /// Local partial derivatives w.r.t. each parent.
    ///
    /// `deps[k]` contributes `adjoints[deps[k]] += adjoints[self] * weights[k]`
    /// during the backward pass.
    weights: [f64; 2],
}

// ---------------------------------------------------------------------------
// Thread-local tape
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread Wengert-list tape.
    ///
    /// Each rayon worker thread has its own tape, so [`vjp`] calls on different
    /// threads never interfere. Cleared at the start and end of every `vjp` call
    /// (double-clear: safe re-entry after a panic mid-forward-pass).
    ///
    /// Follows the identical idiom used by `DP_SCRATCH` in `alignment/mod.rs:473`.
    static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// Push helpers
// ---------------------------------------------------------------------------

/// Record a binary operation on the tape and return its new node index.
///
/// The new node has parents `dep0` (with local partial `w0`) and `dep1`
/// (with local partial `w1`). Either parent may be `SENTINEL` for ops where
/// one operand is a constant.
fn push_binary(dep0: usize, w0: f64, dep1: usize, w1: f64) -> usize {
    TAPE.with(|cell| {
        let mut tape = cell.borrow_mut();
        let idx = tape.len();
        tape.push(Node {
            deps: [dep0, dep1],
            weights: [w0, w1],
        });
        idx
    })
}

/// Record a unary operation on the tape and return its new node index.
///
/// The second slot is set to `SENTINEL` / `0.0` (no second parent).
fn push_unary(dep0: usize, w0: f64) -> usize {
    TAPE.with(|cell| {
        let mut tape = cell.borrow_mut();
        let idx = tape.len();
        tape.push(Node {
            deps: [dep0, SENTINEL],
            weights: [w0, 0.0],
        });
        idx
    })
}

// ---------------------------------------------------------------------------
// Var type
// ---------------------------------------------------------------------------

/// A reverse-mode scalar: a primal value paired with a tape node index.
///
/// `Var` is a `Copy` handle, like [`super::Dual`] for forward mode. The tape
/// index lets [`vjp`] locate this variable's row in the adjoint table during the
/// backward sweep. A `node == SENTINEL` means the `Var` is a constant (off-tape)
/// and contributes zero gradient.
///
/// Both `PartialEq` and `PartialOrd` compare the `value` field only — tape
/// indices must not influence control-flow decisions in `Scalar`-generic code
/// (e.g. `softmin3_generic` uses `<=` and `>=` for primal branching).
#[derive(Debug, Clone, Copy)]
pub struct Var {
    /// The primal (forward) value.
    pub(crate) value: f64,
    /// Index into the thread-local tape. `SENTINEL` = constant (off-tape).
    pub(crate) node: usize,
}

// Hand-written value-only equality — do NOT derive (derive would compare node).
impl PartialEq for Var {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// Hand-written value-only ordering — do NOT derive (derive would use node as
// a lexicographic tiebreaker, corrupting branch semantics in generic code).
impl PartialOrd for Var {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

// ---------------------------------------------------------------------------
// Arithmetic op impls
// ---------------------------------------------------------------------------

/// Add two `Var`s, recording unit partials on the tape.
///
/// `d(u+v)/du = 1`,  `d(u+v)/dv = 1`.
///
/// Short-circuits to a constant when both inputs are constants.
impl Add for Var {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let value = self.value + rhs.value;
        if self.node == SENTINEL && rhs.node == SENTINEL {
            return Var {
                value,
                node: SENTINEL,
            };
        }
        // ∂(u+v)/∂u = 1,  ∂(u+v)/∂v = 1
        let node = push_binary(self.node, 1.0, rhs.node, 1.0);
        Var { value, node }
    }
}

/// Subtract two `Var`s, recording +1 and -1 partials on the tape.
///
/// `d(u-v)/du = 1`,  `d(u-v)/dv = -1`.
///
/// Short-circuits to a constant when both inputs are constants.
impl Sub for Var {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let value = self.value - rhs.value;
        if self.node == SENTINEL && rhs.node == SENTINEL {
            return Var {
                value,
                node: SENTINEL,
            };
        }
        // ∂(u-v)/∂u = 1,  ∂(u-v)/∂v = -1
        let node = push_binary(self.node, 1.0, rhs.node, -1.0);
        Var { value, node }
    }
}

/// Multiply two `Var`s, recording the product-rule partial derivatives.
///
/// `d(u*v)/du = v`,  `d(u*v)/dv = u`.
///
/// Short-circuits to a constant (no tape push) when both inputs are constants
/// (`node == SENTINEL`).
impl Mul for Var {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let value = self.value * rhs.value;
        if self.node == SENTINEL && rhs.node == SENTINEL {
            // const * const: no tape node, result is a constant.
            return Var {
                value,
                node: SENTINEL,
            };
        }
        // General: push a binary node with product-rule partials.
        // ∂(u*v)/∂u = v,  ∂(u*v)/∂v = u
        let node = push_binary(self.node, rhs.value, rhs.node, self.value);
        Var { value, node }
    }
}

/// Divide two `Var`s using the quotient rule.
///
/// `d(u/v)/du = 1/v`,  `d(u/v)/dv = -u/v²`.
///
/// Handles three constant-folding cases:
/// - Both sentinel → constant result (no tape push).
/// - RHS sentinel only → `push_unary(self.node, 1/v)` (common `x/gamma` case).
/// - Otherwise → `push_binary` with full quotient-rule partials.
impl Div for Var {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        if self.node == SENTINEL && rhs.node == SENTINEL {
            // const / const: constant result.
            return Var {
                value,
                node: SENTINEL,
            };
        }
        if rhs.node == SENTINEL {
            // x / constant: only self contributes; partial = 1/v.
            let node = push_unary(self.node, 1.0 / rhs.value);
            return Var { value, node };
        }
        // General quotient rule: ∂(u/v)/∂u = 1/v,  ∂(u/v)/∂v = -u/v²
        let node = push_binary(
            self.node,
            1.0 / rhs.value,
            rhs.node,
            -self.value / (rhs.value * rhs.value),
        );
        Var { value, node }
    }
}

/// Negate a `Var`, recording the `-1` partial on the tape.
///
/// `d(-u)/du = -1`.
///
/// Short-circuits to a constant when the input is a constant.
impl Neg for Var {
    type Output = Self;
    fn neg(self) -> Self {
        let value = -self.value;
        if self.node == SENTINEL {
            return Var {
                value,
                node: SENTINEL,
            };
        }
        // ∂(-u)/∂u = -1
        let node = push_unary(self.node, -1.0);
        Var { value, node }
    }
}

// Assign-ops delegate to binary ops so push_binary/push_unary is called and
// self.node is updated — never hand-write value-only mutation (that would
// skip the tape push and silently drop gradient).
impl AddAssign for Var {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Var {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Var {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ---------------------------------------------------------------------------
// Scalar impl for Var
// ---------------------------------------------------------------------------

impl Scalar for Var {
    // Constant constructors — set node = SENTINEL, push nothing to tape.
    // Called extensively by Scalar-generic code (from_f64 for scaling factors,
    // zero/one for accumulators, infinity for DP initialization).

    #[inline]
    fn zero() -> Self {
        Var {
            value: 0.0,
            node: SENTINEL,
        }
    }

    #[inline]
    fn one() -> Self {
        Var {
            value: 1.0,
            node: SENTINEL,
        }
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Var {
            value: v,
            node: SENTINEL,
        }
    }

    #[inline]
    fn infinity() -> Self {
        Var {
            value: f64::INFINITY,
            node: SENTINEL,
        }
    }

    /// Square root with reverse-mode partial `1 / (2 * sqrt(v))`.
    ///
    /// Singular point: `sqrt(0)` pushes partial `1/(2*0) = Inf` onto the tape,
    /// propagating a non-finite adjoint without panicking — matching `Dual`.
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        if self.node == SENTINEL {
            return Var {
                value: s,
                node: SENTINEL,
            };
        }
        // d/dx sqrt(v) = 1 / (2 * sqrt(v))
        let node = push_unary(self.node, 1.0 / (2.0 * s));
        Var { value: s, node }
    }

    /// Exponential with reverse-mode partial `exp(v)`.
    #[inline]
    fn exp(self) -> Self {
        let e = self.value.exp();
        if self.node == SENTINEL {
            return Var {
                value: e,
                node: SENTINEL,
            };
        }
        // d/dx exp(v) = exp(v)
        let node = push_unary(self.node, e);
        Var { value: e, node }
    }

    /// Natural logarithm with reverse-mode partial `1 / v`.
    ///
    /// Singular point: `ln(0)` pushes partial `1/0 = Inf` — non-finite adjoint,
    /// no panic.
    #[inline]
    fn ln(self) -> Self {
        let l = self.value.ln();
        if self.node == SENTINEL {
            return Var {
                value: l,
                node: SENTINEL,
            };
        }
        // d/dx ln(v) = 1 / v
        let node = push_unary(self.node, 1.0 / self.value);
        Var { value: l, node }
    }

    /// Sine with reverse-mode partial `cos(v)`.
    #[inline]
    fn sin(self) -> Self {
        let s = self.value.sin();
        if self.node == SENTINEL {
            return Var {
                value: s,
                node: SENTINEL,
            };
        }
        // d/dx sin(v) = cos(v)
        let node = push_unary(self.node, self.value.cos());
        Var { value: s, node }
    }

    /// Cosine with reverse-mode partial `-sin(v)`.
    #[inline]
    fn cos(self) -> Self {
        let c = self.value.cos();
        if self.node == SENTINEL {
            return Var {
                value: c,
                node: SENTINEL,
            };
        }
        // d/dx cos(v) = -sin(v)
        let node = push_unary(self.node, -self.value.sin());
        Var { value: c, node }
    }

    /// Power function with reverse-mode partial `p * v^(p - 1)` w.r.t. the base.
    ///
    /// The exponent `p` is a constant `f64`; only the base contributes gradient.
    ///
    /// Singular point: `powf(0, 0.5)` pushes partial `0.5 * 0^(-0.5) = Inf` —
    /// non-finite adjoint, no panic.
    #[inline]
    fn powf(self, p: f64) -> Self {
        let v = self.value.powf(p);
        if self.node == SENTINEL {
            return Var {
                value: v,
                node: SENTINEL,
            };
        }
        // d/dx v^p = p * v^(p - 1)
        let node = push_unary(self.node, p * self.value.powf(p - 1.0));
        Var { value: v, node }
    }

    /// Absolute value with subdifferential partial: `signum(v)`, selecting `0`
    /// at exactly `v == 0.0` (matching the `Dual` subdifferential convention).
    #[inline]
    fn abs(self) -> Self {
        let a = self.value.abs();
        if self.node == SENTINEL {
            return Var {
                value: a,
                node: SENTINEL,
            };
        }
        // Subdifferential: d|v|/dv = signum(v), honest-zero selection at v == 0.
        let sub = if self.value == 0.0 {
            0.0
        } else {
            self.value.signum()
        };
        let node = push_unary(self.node, sub);
        Var { value: a, node }
    }

    /// Signum: piecewise-constant → derivative is **0 everywhere**.
    ///
    /// Returns a `SENTINEL` constant (no tape node) regardless of whether the
    /// input is on-tape. The primal value is `f64::signum(v)` for parity.
    /// This matches the `Dual` convention (zero tangent, `Dual::signum@459`).
    #[inline]
    fn signum(self) -> Self {
        // Piecewise-constant: gradient is zero everywhere — return a constant.
        Var {
            value: self.value.signum(),
            node: SENTINEL,
        }
    }
}

// ---------------------------------------------------------------------------
// vjp entry point
// ---------------------------------------------------------------------------

/// Compute a scalar objective's value and its full gradient over an `m`-vector
/// input via one forward pass (building the tape) and one reverse sweep.
///
/// This is `O(cost(f))` regardless of the number of inputs `m`, versus the
/// `O(m · cost(f))` cost of forward-mode [`super::grad`].
///
/// Returns `(value, gradient)` where `gradient.len() == x.len()`. An empty
/// input yields `(f(&[]).value, Vec::new())`.
///
/// # Lifecycle
///
/// The thread-local tape is cleared before the forward pass (safe re-entry
/// after a prior panic) and again after the backward pass (prevents leakage
/// into the next `vjp` call). Never call `vjp` recursively or concurrently
/// on the same thread.
///
/// # Example
///
/// ```
/// use fdars_core::autodiff::vjp;
///
/// // f(x) = x^2, f'(x) = 2x. At x = 3: f = 9, f' = 6.
/// let (value, gradient) = vjp(|x| x[0] * x[0], &[3.0]);
/// assert!((value - 9.0).abs() < 1e-10);
/// assert!((gradient[0] - 6.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
    // Step 1: Clear tape from any prior call (panic-safe double-clear pattern).
    TAPE.with(|cell| cell.borrow_mut().clear());

    let m = x.len();
    if m == 0 {
        // Empty input: run forward to get primal, no gradients.
        let out = f(&[]);
        TAPE.with(|cell| cell.borrow_mut().clear());
        return (out.value, Vec::new());
    }

    // Step 2: Seed one REAL leaf node per input.
    //
    // Each input leaf is pushed as push_binary(SENTINEL, 0.0, SENTINEL, 0.0):
    // a real node with a valid tape index so its adjoint accumulates, but with
    // both deps = SENTINEL and both weights = 0.0 so it contributes nothing
    // to the backward propagation of further-upstream nodes.
    //
    // NEVER use from_f64 or Scalar constants for inputs: those carry
    // node=SENTINEL and would yield all-zero gradients (Pitfall 2).
    let vars: Vec<Var> = x
        .iter()
        .map(|&v| {
            let node = push_binary(SENTINEL, 0.0, SENTINEL, 0.0);
            Var { value: v, node }
        })
        .collect();

    // Step 3: Forward pass — runs f, recording every op on the tape.
    let output = f(&vars);
    let primal = output.value;

    // Steps 4–6: Backward pass inside a single TAPE borrow.
    let gradient = TAPE.with(|cell| {
        let tape = cell.borrow();
        let n = tape.len();

        // Step 4: Allocate adjoint table, seed output adjoint = 1.0.
        let mut adjoints = vec![0.0_f64; n];
        if output.node != SENTINEL {
            adjoints[output.node] = 1.0;
        }

        // Step 5: Reverse sweep — evaluation order equals topological order,
        // so plain reverse iteration is sufficient (no graph analysis needed).
        for i in (0..n).rev() {
            let node = tape[i];
            let adj = adjoints[i];
            for slot in 0..2 {
                if node.deps[slot] != SENTINEL {
                    adjoints[node.deps[slot]] += adj * node.weights[slot];
                }
            }
        }

        // Step 6: Collect gradient — adjoint at each input leaf's tape index.
        vars.iter()
            .map(|v| {
                if v.node == SENTINEL {
                    0.0
                } else {
                    adjoints[v.node]
                }
            })
            .collect()
    });

    // Step 7: Clear tape after reading (prevent leakage into next call).
    TAPE.with(|cell| cell.borrow_mut().clear());

    (primal, gradient)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    /// End-to-end tracer test: f(x) = x^2, df/dx = 2x.
    /// At x = 3: f = 9, df/dx = 6.
    /// Proves the full record→seed→backward→read-adjoint loop through Mul.
    #[test]
    fn var_mul_known_answer() {
        let (value, grad) = vjp(|x| x[0] * x[0], &[3.0]);
        assert!((value - 9.0).abs() < TOL, "primal {} != 9.0", value);
        assert!(
            (grad[0] - 6.0).abs() < TOL,
            "gradient[0] {} != 6.0",
            grad[0]
        );
    }

    // -------------------------------------------------------------------------
    // Tier 1 — arithmetic ops known-answer tests (added in Task 1, Plan 02)
    // -------------------------------------------------------------------------

    /// Add: f(x) = x + x = 2x, df/dx = 2. At x = 5: f = 10, df/dx = 2.
    #[test]
    fn var_add_known_answer() {
        let (value, grad) = vjp(|x| x[0] + x[0], &[5.0]);
        assert!((value - 10.0).abs() < TOL, "primal {} != 10.0", value);
        assert!(
            (grad[0] - 2.0).abs() < TOL,
            "gradient[0] {} != 2.0",
            grad[0]
        );
    }

    /// Sub: f(x, y) = x - y. At (4, 1): f = 3, df/dx = 1, df/dy = -1.
    #[test]
    fn var_sub_known_answer() {
        let (value, grad) = vjp(|x| x[0] - x[1], &[4.0, 1.0]);
        assert!((value - 3.0).abs() < TOL, "primal {} != 3.0", value);
        assert!((grad[0] - 1.0).abs() < TOL, "grad[0] {} != 1.0", grad[0]);
        assert!((grad[1] + 1.0).abs() < TOL, "grad[1] {} != -1.0", grad[1]);
    }

    /// Div: f(x, y) = x / y. Quotient rule: df/dx = 1/y, df/dy = -x/y^2.
    /// At (6, 3): f = 2, df/dx = 1/3, df/dy = -6/9 = -2/3.
    #[test]
    fn var_div_known_answer() {
        let (value, grad) = vjp(|x| x[0] / x[1], &[6.0, 3.0]);
        assert!((value - 2.0).abs() < TOL, "primal {} != 2.0", value);
        assert!(
            (grad[0] - 1.0 / 3.0).abs() < TOL,
            "grad[0] {} != 1/3",
            grad[0]
        );
        assert!(
            (grad[1] - (-2.0 / 3.0)).abs() < TOL,
            "grad[1] {} != -2/3",
            grad[1]
        );
    }

    /// Div with RHS constant: f(x) = x / 4.0, df/dx = 0.25. At x = 8: f = 2, df/dx = 0.25.
    #[test]
    fn var_div_rhs_const_known_answer() {
        let (value, grad) = vjp(|x| x[0] / Scalar::from_f64(4.0), &[8.0]);
        assert!((value - 2.0).abs() < TOL, "primal {} != 2.0", value);
        assert!((grad[0] - 0.25).abs() < TOL, "grad[0] {} != 0.25", grad[0]);
    }

    /// Neg: f(x) = -(x * x), df/dx = -2x. At x = 3: f = -9, df/dx = -6.
    #[test]
    fn var_neg_known_answer() {
        let (value, grad) = vjp(|x| -(x[0] * x[0]), &[3.0]);
        assert!((value + 9.0).abs() < TOL, "primal {} != -9.0", value);
        assert!((grad[0] + 6.0).abs() < TOL, "grad[0] {} != -6.0", grad[0]);
    }

    /// AddAssign: acc += x*x propagates gradient.
    /// f(x) = 0 + x^2 = x^2, df/dx = 2x. At x = 4: f = 16, df/dx = 8.
    #[test]
    fn var_add_assign_propagates_gradient() {
        let (value, grad) = vjp(
            |x| {
                let mut acc = Scalar::zero();
                acc += x[0] * x[0];
                acc
            },
            &[4.0],
        );
        assert!((value - 16.0).abs() < TOL, "primal {} != 16.0", value);
        assert!((grad[0] - 8.0).abs() < TOL, "grad[0] {} != 8.0", grad[0]);
    }

    /// SubAssign: acc -= x delegates to Sub, gradient flows.
    /// f(x) = 1 - x, df/dx = -1. At x = 3: f = -2, df/dx = -1.
    #[test]
    fn var_sub_assign_propagates_gradient() {
        let (value, grad) = vjp(
            |x| {
                let mut acc = Scalar::one();
                acc -= x[0];
                acc
            },
            &[3.0],
        );
        assert!((value + 2.0).abs() < TOL, "primal {} != -2.0", value);
        assert!((grad[0] + 1.0).abs() < TOL, "grad[0] {} != -1.0", grad[0]);
    }

    /// MulAssign: acc *= x delegates to Mul, gradient flows.
    /// f(x) = 2 * x, df/dx = 2. At x = 5: f = 10, df/dx = 2.
    #[test]
    fn var_mul_assign_propagates_gradient() {
        let (value, grad) = vjp(
            |x| {
                let mut acc = Scalar::from_f64(2.0);
                acc *= x[0];
                acc
            },
            &[5.0],
        );
        assert!((value - 10.0).abs() < TOL, "primal {} != 10.0", value);
        assert!((grad[0] - 2.0).abs() < TOL, "grad[0] {} != 2.0", grad[0]);
    }

    // -------------------------------------------------------------------------
    // Tier 1 — transcendental known-answer tests (added in Task 2, Plan 02)
    // -------------------------------------------------------------------------

    /// sqrt: f(x) = sqrt(x), df/dx = 1/(2*sqrt(x)).
    /// At x = 4: f = 2, df/dx = 0.25.
    #[test]
    fn var_sqrt_known_answer() {
        let (value, grad) = vjp(|x| Scalar::sqrt(x[0]), &[4.0]);
        assert!((value - 2.0).abs() < TOL, "primal {} != 2.0", value);
        assert!((grad[0] - 0.25).abs() < TOL, "grad[0] {} != 0.25", grad[0]);
    }

    /// exp: f(x) = exp(x), df/dx = exp(x).
    /// At x = 1: f = e, df/dx = e.
    #[test]
    fn var_exp_known_answer() {
        let e = std::f64::consts::E;
        let (value, grad) = vjp(|x| Scalar::exp(x[0]), &[1.0]);
        assert!((value - e).abs() < TOL, "primal {} != e", value);
        assert!((grad[0] - e).abs() < TOL, "grad[0] {} != e", grad[0]);
    }

    /// ln: f(x) = ln(x), df/dx = 1/x.
    /// At x = 2: f = ln(2), df/dx = 0.5.
    #[test]
    fn var_ln_known_answer() {
        let (value, grad) = vjp(|x| Scalar::ln(x[0]), &[2.0]);
        assert!(
            (value - 2.0_f64.ln()).abs() < TOL,
            "primal {} != ln(2)",
            value
        );
        assert!((grad[0] - 0.5).abs() < TOL, "grad[0] {} != 0.5", grad[0]);
    }

    /// sin: f(x) = sin(x), df/dx = cos(x).
    /// At x = PI/4: f = sqrt(2)/2, df/dx = sqrt(2)/2.
    #[test]
    fn var_sin_known_answer() {
        use std::f64::consts::PI;
        let expected = 2.0_f64.sqrt() / 2.0;
        let (value, grad) = vjp(|x| Scalar::sin(x[0]), &[PI / 4.0]);
        assert!(
            (value - expected).abs() < TOL,
            "primal {} != sqrt(2)/2",
            value
        );
        assert!(
            (grad[0] - expected).abs() < TOL,
            "grad[0] {} != sqrt(2)/2",
            grad[0]
        );
    }

    /// cos: f(x) = cos(x), df/dx = -sin(x).
    /// At x = PI/4: f = sqrt(2)/2, df/dx = -sqrt(2)/2.
    #[test]
    fn var_cos_known_answer() {
        use std::f64::consts::PI;
        let expected = 2.0_f64.sqrt() / 2.0;
        let (value, grad) = vjp(|x| Scalar::cos(x[0]), &[PI / 4.0]);
        assert!(
            (value - expected).abs() < TOL,
            "primal {} != sqrt(2)/2",
            value
        );
        assert!(
            (grad[0] + expected).abs() < TOL,
            "grad[0] {} != -sqrt(2)/2",
            grad[0]
        );
    }

    /// powf: f(x) = x^3, df/dx = 3*x^2.
    /// At x = 2: f = 8, df/dx = 12.
    #[test]
    fn var_powf_known_answer() {
        let (value, grad) = vjp(|x| Scalar::powf(x[0], 3.0), &[2.0]);
        assert!((value - 8.0).abs() < TOL, "primal {} != 8.0", value);
        assert!((grad[0] - 12.0).abs() < TOL, "grad[0] {} != 12.0", grad[0]);
    }

    /// abs: f(x) = |x|, df/dx = sign(x) (subdifferential: 0 at 0).
    /// At x = -3: f = 3, df/dx = -1.
    #[test]
    fn var_abs_known_answer() {
        let (value, grad) = vjp(|x| Scalar::abs(x[0]), &[-3.0]);
        assert!((value - 3.0).abs() < TOL, "primal {} != 3.0", value);
        assert!((grad[0] + 1.0).abs() < TOL, "grad[0] {} != -1.0", grad[0]);
        // Positive branch
        let (vp, gp) = vjp(|x| Scalar::abs(x[0]), &[4.0]);
        assert!((vp - 4.0).abs() < TOL);
        assert!((gp[0] - 1.0).abs() < TOL);
    }

    /// abs at zero: subdifferential selects 0 (not ±1).
    #[test]
    fn var_abs_at_zero_gradient_is_zero() {
        let (value, grad) = vjp(|x| Scalar::abs(x[0]), &[0.0]);
        assert_eq!(value, 0.0);
        assert_eq!(grad[0], 0.0, "abs'(0) must be 0 (subdifferential)");
    }

    /// signum: gradient is 0 everywhere (piecewise constant).
    #[test]
    fn var_signum_gradient_is_zero() {
        let (vp, gp) = vjp(|x| Scalar::signum(x[0]), &[3.0]);
        assert_eq!(vp, 1.0);
        assert_eq!(gp[0], 0.0, "signum gradient must be 0");
        let (vn, gn) = vjp(|x| Scalar::signum(x[0]), &[-3.0]);
        assert_eq!(vn, -1.0);
        assert_eq!(gn[0], 0.0, "signum gradient must be 0");
    }

    // -------------------------------------------------------------------------
    // Tier 2 — singular-point guard tests (Task 2, Plan 02)
    // -------------------------------------------------------------------------

    /// sqrt(0): local partial 1/(2*sqrt(0)) = Inf. Adjoint must be non-finite, not a panic.
    #[test]
    fn var_sqrt_at_zero_adjoint_is_nonfinite() {
        let (value, grad) = vjp(|x| Scalar::sqrt(x[0]), &[0.0]);
        assert_eq!(value, 0.0);
        assert!(
            !grad[0].is_finite(),
            "sqrt'(0) adjoint {} should be non-finite (Inf)",
            grad[0]
        );
    }

    /// ln(0): local partial 1/0 = Inf (or NaN). Adjoint must be non-finite, not a panic.
    #[test]
    fn var_ln_at_zero_adjoint_is_nonfinite() {
        let (value, grad) = vjp(|x| Scalar::ln(x[0]), &[0.0]);
        assert!(
            value.is_infinite() && value < 0.0,
            "ln(0) value should be -Inf"
        );
        assert!(
            !grad[0].is_finite(),
            "ln'(0) adjoint {} should be non-finite",
            grad[0]
        );
    }

    /// powf(0, 0.5): partial = 0.5 * 0^(-0.5) = Inf. Adjoint must be non-finite, not a panic.
    #[test]
    fn var_powf_at_zero_adjoint_is_nonfinite() {
        let (value, grad) = vjp(|x| Scalar::powf(x[0], 0.5), &[0.0]);
        assert_eq!(value, 0.0);
        assert!(
            !grad[0].is_finite(),
            "powf'(0, 0.5) adjoint {} should be non-finite (Inf)",
            grad[0]
        );
    }

    // -------------------------------------------------------------------------
    // Tier 5 — lifecycle / edge-case tests (Task 1, Plan 03)
    // -------------------------------------------------------------------------

    /// Empty input: vjp(|_| constant, &[]) returns (constant_value, empty vec),
    /// no panic and no tape residue.
    #[test]
    fn vjp_empty_input_no_panic() {
        let (value, grad) = vjp(|_x| Scalar::from_f64(5.0), &[]);
        assert!((value - 5.0).abs() < TOL, "primal {} != 5.0", value);
        assert!(grad.is_empty(), "gradient must be empty for empty input");
    }

    /// Constant-only closure: f always returns 2.0 regardless of inputs.
    /// The output node is SENTINEL, so every input's adjoint stays 0.0.
    #[test]
    fn vjp_constant_only_closure_gradient_is_zero() {
        let (value, grad) = vjp(|_x| Scalar::from_f64(2.0), &[1.0, 2.0]);
        assert!((value - 2.0).abs() < TOL, "primal {} != 2.0", value);
        assert_eq!(grad.len(), 2, "gradient length must match input length");
        assert_eq!(grad[0], 0.0, "grad[0] must be 0.0 for constant closure");
        assert_eq!(grad[1], 0.0, "grad[1] must be 0.0 for constant closure");
    }

    /// Single input cube: f(x) = x^3, f'(x) = 3x^2.
    /// At x = 2: f = 8, f' = 12.
    #[test]
    fn vjp_single_input_cube() {
        let (value, grad) = vjp(|x| x[0] * x[0] * x[0], &[2.0]);
        assert!((value - 8.0).abs() < TOL, "primal {} != 8.0", value);
        assert!((grad[0] - 12.0).abs() < TOL, "grad[0] {} != 12.0", grad[0]);
    }

    /// Repeated-call stability: calling vjp 3 times in a row on the same closure
    /// yields bit-identical gradients (proves no tape leakage between calls).
    #[test]
    fn vjp_repeated_calls_no_gradient_drift() {
        let f = |x: &[Var]| x[0] * x[0] + x[1];
        let x = &[3.0_f64, 1.0_f64];
        let (v0, g0) = vjp(f, x);
        let (v1, g1) = vjp(f, x);
        let (v2, g2) = vjp(f, x);
        // All three calls must agree exactly (no drift from tape residue).
        assert_eq!(v0, v1, "primal drift on call 2");
        assert_eq!(v0, v2, "primal drift on call 3");
        assert_eq!(g0[0], g1[0], "grad[0] drift on call 2");
        assert_eq!(g0[0], g2[0], "grad[0] drift on call 3");
        assert_eq!(g0[1], g1[1], "grad[1] drift on call 2");
        assert_eq!(g0[1], g2[1], "grad[1] drift on call 3");
        // Also assert the values are correct: f = x0^2 + x1 = 10, df/dx0 = 2*3 = 6, df/dx1 = 1.
        assert!((v0 - 10.0).abs() < TOL, "primal {} != 10.0", v0);
        assert!((g0[0] - 6.0).abs() < TOL, "grad[0] {} != 6.0", g0[0]);
        assert!((g0[1] - 1.0).abs() < TOL, "grad[1] {} != 1.0", g0[1]);
    }

    /// Many-input→scalar single backward sweep: f(x0, x1, x2) = x0*x1 + x2.
    /// At (2, 3, 4): f = 10, df/dx0 = 3, df/dx1 = 2, df/dx2 = 1.
    /// All three gradients are accumulated in one reverse sweep — the RAD-02
    /// efficiency claim over grad's 3 forward passes.
    #[test]
    fn vjp_many_input_scalar_single_sweep() {
        let (value, grad) = vjp(|x| x[0] * x[1] + x[2], &[2.0, 3.0, 4.0]);
        assert!((value - 10.0).abs() < TOL, "primal {} != 10.0", value);
        assert!((grad[0] - 3.0).abs() < TOL, "grad[0] {} != 3.0", grad[0]);
        assert!((grad[1] - 2.0).abs() < TOL, "grad[1] {} != 2.0", grad[1]);
        assert!((grad[2] - 1.0).abs() < TOL, "grad[2] {} != 1.0", grad[2]);
    }

    // -------------------------------------------------------------------------
    // Tier 3 — reverse-vs-Dual agreement tests (Task 2, Plan 03)
    //
    // Each test runs the SAME mathematical function as both:
    //   grad(|d: &[Dual]| ..., x)   — forward-mode (m passes)
    //   vjp( |v: &[Var]|  ..., x)   — reverse-mode (one sweep)
    // and asserts value + each gradient component agree within TOL (1e-10).
    // Points are chosen in the interior of each op's domain to avoid NaN/Inf.
    // -------------------------------------------------------------------------

    /// Agreement: add. f(x0, x1) = x0 + x1.
    #[test]
    fn agreement_add() {
        use crate::autodiff::{grad, Dual};
        let x = &[3.0_f64, 7.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| d[0] + d[1], x);
        let (rv, rg) = vjp(|v: &[Var]| v[0] + v[1], x);
        assert!((fv - rv).abs() < TOL, "value mismatch: fwd={fv} rev={rv}");
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
        assert!(
            (fg[1] - rg[1]).abs() < TOL,
            "grad[1]: fwd={} rev={}",
            fg[1],
            rg[1]
        );
    }

    /// Agreement: sub. f(x0, x1) = x0 - x1.
    #[test]
    fn agreement_sub() {
        use crate::autodiff::{grad, Dual};
        let x = &[5.0_f64, 2.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| d[0] - d[1], x);
        let (rv, rg) = vjp(|v: &[Var]| v[0] - v[1], x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
        assert!(
            (fg[1] - rg[1]).abs() < TOL,
            "grad[1]: fwd={} rev={}",
            fg[1],
            rg[1]
        );
    }

    /// Agreement: mul. f(x0, x1) = x0 * x1.
    #[test]
    fn agreement_mul() {
        use crate::autodiff::{grad, Dual};
        let x = &[3.0_f64, 4.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| d[0] * d[1], x);
        let (rv, rg) = vjp(|v: &[Var]| v[0] * v[1], x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
        assert!(
            (fg[1] - rg[1]).abs() < TOL,
            "grad[1]: fwd={} rev={}",
            fg[1],
            rg[1]
        );
    }

    /// Agreement: div. f(x0, x1) = x0 / x1.
    #[test]
    fn agreement_div() {
        use crate::autodiff::{grad, Dual};
        let x = &[6.0_f64, 3.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| d[0] / d[1], x);
        let (rv, rg) = vjp(|v: &[Var]| v[0] / v[1], x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
        assert!(
            (fg[1] - rg[1]).abs() < TOL,
            "grad[1]: fwd={} rev={}",
            fg[1],
            rg[1]
        );
    }

    /// Agreement: neg. f(x) = -x.
    #[test]
    fn agreement_neg() {
        use crate::autodiff::{grad, Dual};
        let x = &[4.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| -d[0], x);
        let (rv, rg) = vjp(|v: &[Var]| -v[0], x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: sqrt. f(x) = sqrt(x). At x = 9.
    #[test]
    fn agreement_sqrt() {
        use crate::autodiff::{grad, Dual};
        let x = &[9.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::sqrt(d[0]), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::sqrt(v[0]), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: exp. f(x) = exp(x). At x = 1.
    #[test]
    fn agreement_exp() {
        use crate::autodiff::{grad, Dual};
        let x = &[1.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::exp(d[0]), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::exp(v[0]), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: ln. f(x) = ln(x). At x = 2.
    #[test]
    fn agreement_ln() {
        use crate::autodiff::{grad, Dual};
        let x = &[2.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::ln(d[0]), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::ln(v[0]), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: sin. f(x) = sin(x). At x = PI/3.
    #[test]
    fn agreement_sin() {
        use crate::autodiff::{grad, Dual};
        let x = &[std::f64::consts::PI / 3.0];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::sin(d[0]), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::sin(v[0]), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: cos. f(x) = cos(x). At x = PI/6.
    #[test]
    fn agreement_cos() {
        use crate::autodiff::{grad, Dual};
        let x = &[std::f64::consts::PI / 6.0];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::cos(d[0]), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::cos(v[0]), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: powf. f(x) = x^3. At x = 2.
    #[test]
    fn agreement_powf() {
        use crate::autodiff::{grad, Dual};
        let x = &[2.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::powf(d[0], 3.0), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::powf(v[0], 3.0), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: abs. f(x) = |x|. At x = -3 (negative branch; subdifferential = -1).
    #[test]
    fn agreement_abs() {
        use crate::autodiff::{grad, Dual};
        let x = &[-3.0_f64];
        let (fv, fg) = grad(|d: &[Dual]| Scalar::abs(d[0]), x);
        let (rv, rg) = vjp(|v: &[Var]| Scalar::abs(v[0]), x);
        assert!((fv - rv).abs() < TOL);
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
    }

    /// Agreement: multi-input composed chain. f(x0, x1, x2) = sin(x0) * exp(x1) + sqrt(x2).
    /// Exercises cross-op gradient accumulation through mul, add, and unary ops.
    /// At (PI/4, 0.5, 4.0).
    #[test]
    fn agreement_composed_chain() {
        use crate::autodiff::{grad, Dual};
        let x = &[std::f64::consts::PI / 4.0, 0.5_f64, 4.0_f64];
        let (fv, fg) = grad(
            |d: &[Dual]| Scalar::sin(d[0]) * Scalar::exp(d[1]) + Scalar::sqrt(d[2]),
            x,
        );
        let (rv, rg) = vjp(
            |v: &[Var]| Scalar::sin(v[0]) * Scalar::exp(v[1]) + Scalar::sqrt(v[2]),
            x,
        );
        assert!((fv - rv).abs() < TOL, "value: fwd={fv} rev={rv}");
        assert!(
            (fg[0] - rg[0]).abs() < TOL,
            "grad[0]: fwd={} rev={}",
            fg[0],
            rg[0]
        );
        assert!(
            (fg[1] - rg[1]).abs() < TOL,
            "grad[1]: fwd={} rev={}",
            fg[1],
            rg[1]
        );
        assert!(
            (fg[2] - rg[2]).abs() < TOL,
            "grad[2]: fwd={} rev={}",
            fg[2],
            rg[2]
        );
    }
}
