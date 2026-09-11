---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: "01"
subsystem: autodiff
tags: [reverse-mode, autodiff, wengert-list, tape, vjp, Var, Dual, Scalar, refactor]

# Dependency graph
requires: []
provides:
  - "autodiff/ directory module (mod.rs + forward.rs + reverse.rs)"
  - "Scalar trait moved to mod.rs (shared by forward and reverse)"
  - "Var struct (Copy, thread-local tape handle) with value-only PartialEq/PartialOrd"
  - "Node record {deps:[usize;2], weights:[f64;2]} — uniform 2-slot Wengert-list entry"
  - "TAPE thread-local RefCell<Vec<Node>> (rayon-compatible, const-init idiom)"
  - "push_binary / push_unary helpers"
  - "Mul for Var with product-rule partials and const*const short-circuit"
  - "Scalar impl for Var: constant constructors + transcendental stubs"
  - "vjp entry point: one forward pass + one reverse sweep"
  - "var_mul_known_answer end-to-end tracer test"
  - "prelude.rs extended with vjp and Var"
affects: [94-02, 94-03, 94-04, 95, 96, 97, 98, 99]

# Actuals (#2632)
actuals:
  tokens: 14290
  tasks: 3
  commits: 4

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Thread-local tape via thread_local! { static: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) }; } — identical to alignment/mod.rs:473"
    - "Sentinel constant SENTINEL = usize::MAX for off-tape Var handles"
    - "Uniform 2-slot Node record covers both unary and binary ops — no per-op dispatch in backward loop"
    - "Double-clear vjp lifecycle: clear before forward (panic safety) + after backward (no leakage)"
    - "Leaf node via push_binary(SENTINEL,0,SENTINEL,0) — real tape index, zero parent contribution"
    - "git mv autodiff.rs → autodiff/forward.rs to preserve file history"

key-files:
  created:
    - fdars-core/src/autodiff/mod.rs
    - fdars-core/src/autodiff/reverse.rs
  modified:
    - fdars-core/src/autodiff/forward.rs
    - fdars-core/src/prelude.rs

key-decisions:
  - "Scalar trait moved to mod.rs (shared between forward and reverse) rather than staying in forward.rs"
  - "Tape fully opaque — not publicly constructible, not re-exported from prelude; only vjp is exposed"
  - "push_unary marked #[allow(dead_code)] rather than removed — used by Plan 02 transcendental impls"
  - "Prelude extended with vjp+Var immediately (not deferred to Plan 04) since it costs nothing and removes a drift risk"

patterns-established:
  - "Tracer-first: architecture validated on one op (Mul) before expanding to full op set (Plan 02)"
  - "Module refactor via git mv preserves history; Cargo resolves pub mod autodiff to directory automatically"
  - "TDD RED-GREEN per Plan task: failing test committed first, then implementation"

requirements-completed: [RAD-01, RAD-02]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "autodiff refactored into directory module with Scalar trait in mod.rs; all public paths preserved"
    requirement: RAD-01
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/forward.rs::tests (30 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Var/Node/TAPE/push_binary/push_unary/SENTINEL skeleton with Mul and sentinel constants"
    requirement: RAD-01
    verification:
      - kind: unit
        ref: "cargo build -p fdars-core --features linalg,parallel (clean)"
        status: pass
    human_judgment: false
  - id: D3
    description: "vjp entry point: one forward pass + one backward sweep returning (f64, Vec<f64>)"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_mul_known_answer"
        status: pass
    human_judgment: false

# Metrics
duration: 7min
completed: 2026-09-10
status: complete
---

# Phase 94 Plan 01: Tracer Tape Skeleton Summary

**Wengert-list reverse-mode tape proven end-to-end on Mul: autodiff/ directory refactor (Scalar in mod.rs, Dual in forward.rs, Var/Node/TAPE/vjp in reverse.rs) with var_mul_known_answer passing (9.0, [6.0]) within 1e-10**

## Performance

- **Duration:** 7 min
- **Started:** 2026-09-10T20:25:15Z
- **Completed:** 2026-09-10T20:32:54Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Refactored monolithic `autodiff.rs` (1143 lines) into `autodiff/{mod.rs,forward.rs,reverse.rs}` via `git mv` preserving history; all 30 forward-mode `Dual` tests pass unchanged
- Scalar trait moved to `mod.rs` (shared between forward and reverse); `lib.rs:76 pub mod autodiff;` unchanged; Cargo resolves to directory automatically
- Full Var/Node/TAPE skeleton: Copy handle `{value:f64, node:usize}`, `SENTINEL=usize::MAX`, `Node {deps:[usize;2], weights:[f64;2]}`, thread-local tape (const-init idiom from alignment/mod.rs:473), `push_binary`/`push_unary`
- `Mul for Var` with product-rule partials (`∂(u*v)/∂u=v`, `∂(u*v)/∂v=u`) and const*const short-circuit; all other ops stub as `unimplemented!()` for Plan 02
- `vjp` 7-step lifecycle: double-clear (panic-safe), leaf-seeding via `push_binary(SENTINEL,0,SENTINEL,0)` (real tape index, avoids Pitfall 2), one forward pass, one reverse sweep with SENTINEL guard, gradient collection
- End-to-end tracer test `var_mul_known_answer`: `vjp(|x| x[0]*x[0], &[3.0]) == (9.0, [6.0])` within 1e-10 — record→seed→backward→read proven
- `prelude.rs` extended with `vjp` and `Var` alongside existing forward-mode exports
- All gates green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, 31 autodiff tests (30 forward + 1 reverse), no new dependency

## Task Commits

1. **Task 1: Refactor autodiff.rs into autodiff/{mod,forward,reverse}** — `eaeb1381` (feat)
2. **Task 2: Var/Node/TAPE skeleton with Mul and sentinel constants** — `a552b7ae` (feat)
3. **Task 3: Implement vjp + var_mul_known_answer tracer test** — `b0556e4e` (feat)
4. **Prelude update: add vjp+Var re-exports** — `892ae6cf` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `fdars-core/src/autodiff/mod.rs` — new: Scalar trait definition + pub mod forward/reverse + re-exports
- `fdars-core/src/autodiff/forward.rs` — moved from autodiff.rs (git mv), Scalar trait removed, `use super::Scalar` + std::ops imports added
- `fdars-core/src/autodiff/reverse.rs` — new: Var, Node, TAPE, push_binary, push_unary, SENTINEL, Mul, partial Scalar impl (constants real, transcendentals stub), vjp, var_mul_known_answer test
- `fdars-core/src/prelude.rs` — line 21 extended with vjp, Var

## Decisions Made

- **Scalar trait in mod.rs**: plan specified this; both forward.rs and reverse.rs add `use super::Scalar;`. This is the shared contract across both AD modes.
- **Tape fully opaque**: per CONTEXT.md discretion guidance, `Tape` is not exported from prelude. Only `vjp` is the public entry point. If Phase 99 needs inspection, add it then.
- **push_unary #[allow(dead_code)]**: the function is defined in Task 2 and used by Plan 02 transcendental impls. Rather than remove it (would require Plan 02 to add it) or export it (unnecessary), it stays with the `#[allow]` annotation.
- **Prelude extended immediately**: PATTERNS.md called this out as a Phase 94 deliverable and it costs nothing; deferred to Plan 04 was the original intent but doing it here removes drift.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added std::ops imports to forward.rs**
- **Found during:** Task 1 (first build attempt)
- **Issue:** After removing the Scalar trait definition from forward.rs, the `use std::fmt::Debug; use std::ops::{...}` imports were removed with it, but `impl Add for Dual`, `impl Mul for Dual`, etc. still need them.
- **Fix:** Added `use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};` back to forward.rs under `use super::Scalar;`.
- **Files modified:** `fdars-core/src/autodiff/forward.rs`
- **Verification:** `cargo build` succeeded after the fix.
- **Committed in:** eaeb1381 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed clippy: unused variable names in stub impls**
- **Found during:** Task 3 (clippy gate)
- **Issue:** `Add`, `Sub`, `Div` stub impls had `rhs: Self` (named) for an unimplemented body; clippy -D warnings rejected this.
- **Fix:** Renamed to `_rhs: Self` in Add/Sub/Div stubs; added `#[allow(dead_code)]` to `push_unary`.
- **Files modified:** `fdars-core/src/autodiff/reverse.rs`
- **Verification:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` passes.
- **Committed in:** b0556e4e (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 missing import, 1 clippy naming)
**Impact on plan:** Both auto-fixes were minor; they did not affect architecture or correctness. No scope creep.

## Issues Encountered

None — the architectural choices from 94-RESEARCH.md and 94-PATTERNS.md were exact; the implementation matched the plan on the first attempt.

## Known Stubs

| File | Symbol | Reason |
|------|--------|--------|
| `fdars-core/src/autodiff/reverse.rs` | `Scalar::sqrt/exp/ln/sin/cos/powf/abs/signum` for `Var` | Transcendentals deferred to Plan 02 per plan spec; `unimplemented!()` bodies |
| `fdars-core/src/autodiff/reverse.rs` | `Add/Sub/Div/Neg for Var` | Binary/unary ops deferred to Plan 02; `unimplemented!()` bodies |

These stubs are the ONE allowed tracer stub (the plan explicitly permits them): the `Scalar` trait shape is proven by the compilation, the transcendental bodies come in Plan 02. They do not prevent the plan's goal (end-to-end Mul gradient proven by `var_mul_known_answer`).

## Threat Flags

None — pure numeric computation on caller-provided f64 slices; no I/O, network, deserialization, auth, or privilege boundary. Thread-local tape is process-local with no exfiltration vector.

## Next Phase Readiness

- Architecture proven: thread-local tape + Copy Var handle + sentinel constants + one backward sweep all work.
- Plan 02 expands the op set: Add, Sub, Div, Neg, all transcendentals (`sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, `signum`).
- Plan 03 hardens `vjp` and adds the full test tier (validation vs `Dual` and central FD on `soft_dtw_distance_generic`/`project_scores_generic`).
- Plan 04 adds crate-root doctest and any remaining prelude updates.
- No blockers. All gates green.

---
*Phase: 94-reverse-mode-autodiff-core-vjp-tape*
*Completed: 2026-09-10*

## Self-Check: PASSED

- `fdars-core/src/autodiff/mod.rs`: FOUND
- `fdars-core/src/autodiff/forward.rs`: FOUND
- `fdars-core/src/autodiff/reverse.rs`: FOUND
- `fdars-core/src/autodiff.rs` removed: CONFIRMED
- `lib.rs:76 pub mod autodiff;`: CONFIRMED
- Commits eaeb1381, a552b7ae, b0556e4e, 892ae6cf: PRESENT in git log
- 31 autodiff tests: PASS
- cargo fmt --check: PASS
- cargo clippy --all-targets --features linalg,parallel -- -D warnings: PASS
- git diff Cargo.toml fdars-core/Cargo.toml: EMPTY (no new dep)
