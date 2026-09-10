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

impl Add for Var {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        unimplemented!("Add for Var — implemented in Plan 02")
    }
}

impl Sub for Var {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
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
    fn div(self, rhs: Self) -> Self {
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
// vjp entry point — stub, filled in Task 3
// ---------------------------------------------------------------------------

/// Compute a scalar objective's value and its full gradient over an `m`-vector
/// input via one forward pass (building the tape) and one reverse sweep.
///
/// This is `O(cost(f))` regardless of the number of inputs `m`, versus the
/// `O(m · cost(f))` cost of forward-mode [`super::grad`].
///
/// Returns `(value, gradient)` where `gradient.len() == x.len()`.
///
/// See `autodiff::grad` for the equivalent forward-mode entry point.
#[must_use]
pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
    // Full implementation added in Task 3.
    let _ = f;
    let _ = x;
    unimplemented!("vjp — full implementation added in Task 3")
}
