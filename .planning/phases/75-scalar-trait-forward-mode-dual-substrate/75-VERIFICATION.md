---
phase: 75-scalar-trait-forward-mode-dual-substrate
verified: 2026-09-06T00:00:00Z
status: passed
score: 7/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 75: Scalar Trait & Forward-Mode Dual Substrate Verification Report

**Phase Goal:** The numeric substrate exists — a `Scalar` trait plus a forward-mode `Dual` number carrying value + tangent — that the differentiable subset is written against, with all arithmetic/transcendental ops the subset needs and gradient seed/extract helpers, verified against analytical derivatives.
**Requirement:** DIF-01
**Verified:** 2026-09-06
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | Construct a `Dual`, seed tangent=1, run composed expression over ±, ×, ÷, sqrt, exp, ln, sin, cos, powf, abs, partial comparisons, extract (value, derivative) (D-op-set) | ✓ VERIFIED | All ops present in `autodiff.rs`: Add/Sub/Mul/Div/Neg (L213-269), AddAssign/SubAssign/MulAssign (L271-290), transcendentals sqrt/exp/ln/sin/cos/powf/abs/signum (L343-416), value-only PartialOrd (L306-311), seed/constant/extract (L182-211), `diff` helper (L433). Exercised by `dual_composed_chain_known_answer` (composes exp·sin+x², sqrt) and `dual_partial_cmp_value_only`. 24 lib tests pass. |
| 2 | Dual arithmetic reproduces analytical derivative of composed elementary functions to ≤1e-10 (DIF-01 known-answer gate) | ✓ VERIFIED | `const TOL = 1e-10`; per-op known-answer tests (mul, sqrt, exp, ln, sin, cos, powf, abs, sub/div/neg, assign-ops) + `dual_composed_chain_known_answer` asserting closed-form `(1/(2f))·(e^x(sin+cos)+2x)` within 1e-10. `cargo test ... --lib autodiff` = 24 passed, 0 failed. |
| 3 | Dual gradients cross-check against central finite differences to ≤1e-6 for multi-op compositions | ✓ VERIFIED | Tier 2: `finite_diff_cross_check_composed`, `_log_trig`, `_powf_exp` (h=1e-8, `<1e-6`), all green. |
| 4 | `Scalar` implemented for f64 as zero-cost passthrough; f64 transcendentals match direct f64 bit-for-bit (assert_eq!) | ✓ VERIFIED | `impl Scalar for f64` (L107-157), every method `#[inline]` delegating to inherent f64 method. `f64_parity_transcendentals` + `f64_parity_constants` use exact `assert_eq!` for sqrt/exp/ln/sin/cos/powf/abs/signum + zero/one/from_f64/infinity. Green. |
| 5 | No new crate dependency: Cargo.toml [dependencies] byte-identical, no `use num_traits` in autodiff.rs | ✓ VERIFIED | `grep -c num_traits autodiff.rs` = 0; `grep -c num[-_]traits Cargo.toml` = 0; `git diff 41ff88e0 HEAD -- fdars-core/Cargo.toml` empty. |
| 6 | Additive/non-breaking: no existing public signature changed; only `pub mod autodiff;` added to lib.rs | ✓ VERIFIED | `git diff 41ff88e0 HEAD -- fdars-core/` = only 2 files: new `autodiff.rs` (+783) and `lib.rs` (+1, the `pub mod autodiff;` line at L76). No existing signature altered. Whole-crate suite 2845 lib + all integration/doc green — no regression. |
| 7 | `Scalar::infinity()` exists on the trait (Phase 76 soft-DTW DP sentinel), avoiding a later trait-breaking change | ✓ VERIFIED | Trait method declared (L80); `f64` impl returns `f64::INFINITY` (L121); `Dual` impl returns `{value: INFINITY, tangent: 0.0}` (L336). Tested in `f64_parity_constants` + `dual_constants`. |

**Score:** 7/7 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `fdars-core/src/autodiff.rs` | Scalar trait + Dual struct + Scalar impls for f64 & Dual + seed/extract/diff helpers + inline three-tier tests | ✓ VERIFIED | 783 lines. Full op set, both impls, hand-written value-only PartialOrd + PartialEq, `diff` helper, 24 lib tests + 2 doctests across three tiers + singular-point guard tests. |
| `fdars-core/src/lib.rs` | contains `pub mod autodiff;` | ✓ VERIFIED | Line 76 (alphabetical, after `pub mod andrews;`). Sole change to lib.rs. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| Dual PartialOrd | value field only | hand-written `partial_cmp` compares `self.value.partial_cmp(&other.value)` | ✓ WIRED | L306-311, not derived. Tested by `dual_partial_cmp_value_only` (equal value/diff tangent → Equal). Bonus: PartialEq also hand-written value-only (L296-301) keeping `a==b ⟺ partial_cmp==Equal` std contract (review commit e8e8c3b6). |
| Scalar trait bounds | AddAssign + SubAssign + MulAssign | supertrait bounds so Phase 76 `sum += ...` compiles | ✓ WIRED | L69-71. Impls L271-290, tested by `dual_assign_ops`. |
| powf | concrete f64 exponent | `fn powf(self, p: f64)` (not Self) | ✓ WIRED | L94 (trait), L386 (Dual impl), L146 (f64 impl). |

### Behavioral Spot-Checks / Gate Results

| Gate | Command | Result | Status |
| ---- | ------- | ------ | ------ |
| autodiff lib tests | `cargo test -p fdars-core --features linalg,parallel --lib autodiff` | 24 passed, 0 failed | ✓ PASS |
| autodiff doctests | `cargo test -p fdars-core --features linalg,parallel --doc autodiff` | 2 passed, 0 failed | ✓ PASS |
| whole-crate tests | `cargo test -p fdars-core --features linalg,parallel` | 2845 lib passed + all integration/doc suites passed; 0 failed (4 pre-existing ignored) | ✓ PASS |
| clippy | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | exit 0, zero warnings | ✓ PASS |
| no new dependency | `git diff Cargo.toml` empty; `grep -c num[-_]traits Cargo.toml` = 0 | confirmed | ✓ PASS |
| no num_traits in module | `grep -c num_traits fdars-core/src/autodiff.rs` | 0 | ✓ PASS |
| module registered | `grep -c 'pub mod autodiff;' fdars-core/src/lib.rs` | 1 | ✓ PASS |
| additive-only diff | `git diff 41ff88e0 HEAD -- fdars-core/` | 2 files, +784, no existing signature changed | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| DIF-01 | 75-01 | In-crate forward-mode AD substrate (Scalar trait + Dual number) | ✓ SATISFIED | All 7 truths + 2 artifacts + 3 key links verified; all gates green. |

### Anti-Patterns Found

None. No TBD/FIXME/XXX/TODO/placeholder markers in `autodiff.rs`. `return null`/empty-impl patterns not applicable (numeric substrate). No stub data flow. The `return`-less transcendental bodies are real chain-rule implementations, each covered by known-answer + finite-diff tests.

### Gaps Summary

None. Every ROADMAP Success Criterion and PLAN must-have is backed by present, wired, and test-exercised code. The known-answer (≤1e-10), finite-difference (≤1e-6), and f64-parity (exact `assert_eq!`) tiers all pass; the no-new-dependency and additive/non-breaking invariants are confirmed by git diff and grep, not merely by SUMMARY claim. Note: SUMMARY stated "18 passed" for the autodiff module; the actual verified count is 24 lib tests (SUMMARY undercounted — additional singular-point guard tests were included), which strengthens rather than weakens the result.

---

_Verified: 2026-09-06_
_Verifier: Claude (gsd-verifier)_
