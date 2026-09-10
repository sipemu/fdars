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
// Used by transcendental op impls added in Plan 02.
#[allow(dead_code)]
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

impl Add for Var {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        unimplemented!("Add for Var — implemented in Plan 02")
    }
}

impl Sub for Var {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        unimplemented!("Sub for Var — implemented in Plan 02")
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

impl Div for Var {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        unimplemented!("Div for Var — implemented in Plan 02")
    }
}

impl Neg for Var {
    type Output = Self;
    fn neg(self) -> Self {
        unimplemented!("Neg for Var — implemented in Plan 02")
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

    // Transcendental stubs — filled in Plan 02 with real push_unary calls.
    // These are the only allowed tracer stubs: the trait shape is proven here,
    // the bodies come next.

    fn sqrt(self) -> Self {
        unimplemented!("Scalar::sqrt for Var — implemented in Plan 02")
    }
    fn exp(self) -> Self {
        unimplemented!("Scalar::exp for Var — implemented in Plan 02")
    }
    fn ln(self) -> Self {
        unimplemented!("Scalar::ln for Var — implemented in Plan 02")
    }
    fn sin(self) -> Self {
        unimplemented!("Scalar::sin for Var — implemented in Plan 02")
    }
    fn cos(self) -> Self {
        unimplemented!("Scalar::cos for Var — implemented in Plan 02")
    }
    fn powf(self, _p: f64) -> Self {
        unimplemented!("Scalar::powf for Var — implemented in Plan 02")
    }
    fn abs(self) -> Self {
        unimplemented!("Scalar::abs for Var — implemented in Plan 02")
    }
    fn signum(self) -> Self {
        unimplemented!("Scalar::signum for Var — implemented in Plan 02")
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
}
