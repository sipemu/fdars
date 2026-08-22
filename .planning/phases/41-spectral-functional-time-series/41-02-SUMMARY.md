---
phase: 41-spectral-functional-time-series
plan: 02
subsystem: simulation
tags: [functional-var, vma, farma, simulation, rand, seeded, functional-time-series]

requires:
  - phase: 41-spectral-functional-time-series (plan 41-01)
    provides: fts/spectral.rs spectral pipeline that these simulators produce fixtures for
provides:
  - sim_fvarma — functional VAR/VMA simulator from operator kernels (FTS-03-04)
  - sim_farma — functional ARMA (combined AR+MA) simulator (FTS-03-05)
  - FvarmaResult / FarmaResult structs + crate-root re-exports
affects: [fts, spectral, simulation]

actuals:
  tokens: 15000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Shared private fvarma_core recurrence so sim_fvarma and sim_farma are bit-identical for identical inputs"
    - "Column-major m×m operator kernels applied by matrix-vector product a_k[j1 + j2*m] * x[j2]"
    - "NaN/Inf finiteness guard each step → ComputationFailed instead of emitting divergent curves"

key-files:
  created: []
  modified:
    - fdars-core/src/simulation.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "seed: u64 (mandatory, StdRng::seed_from_u64, no entropy fallback) per crate FTS convention — bit-identical output"
  - "i.i.d. N(0,1) innovations per grid point (identity covariance) — documented freqdom divergence (freqdom accepts a user σ)"
  - "sim_farma is a thin named entry point over the shared recurrence; FARMA = combined AR+MA"
  - "Stationarity (spectral radius < 1) is the caller's responsibility; only numeric divergence is guarded"

patterns-established:
  - "Operator-kernel simulators: validate every kernel is flat m×m at entry, ring-buffer histories of last p curves / q innovations, discard burn_in prefix"

requirements-completed: [FTS-03-04, FTS-03-05]

coverage:
  - id: D1
    description: "sim_fvarma runs the seeded operator recurrence; bit-identical for a fixed seed; zero AR op ⇒ white noise; rank-1 op ⇒ serial dependence; dim + NaN/Inf guards"
    requirement: FTS-03-04
    verification:
      - kind: unit
        ref: "fdars-core/src/simulation.rs#fvarma_deterministic, fvarma_zero_op_white_noise, fvarma_rank1_dependence, fvarma_dimension_errors, fvarma_divergence_guard"
        status: pass
  - id: D2
    description: "sim_farma combines AR+MA, is deterministic and shape-correct, and equals sim_fvarma on identical inputs"
    requirement: FTS-03-05
    verification:
      - kind: unit
        ref: "fdars-core/src/simulation.rs#farma_shape_and_order, farma_deterministic, farma_equals_fvarma"
        status: pass
---

# Plan 41-02 Summary: Functional VAR/VMA + FARMA simulators

## Accomplishments

- **`sim_fvarma`** (FTS-03-04): simulates `X_t = Σ_k A_k·X_{t-k} + ε_t + Σ_k B_k·ε_{t-k}` from user-supplied flat column-major m×m AR/MA operator kernels, with i.i.d. N(0,1) innovations per grid point, a configurable burn-in prefix, and a per-step NaN/Inf finiteness guard that returns `ComputationFailed` for non-stationary operators. Deterministic under `seed: u64` (`StdRng::seed_from_u64`, no entropy fallback).
- **`sim_farma`** (FTS-03-05): the named combined-AR+MA entry point, delegating to the shared private `fvarma_core` so it is bit-identical to `sim_fvarma` for identical inputs.
- `FvarmaResult` / `FarmaResult` (`curves`, `ar_order`, `ma_order`, `burn_in`) added; both simulators + structs re-exported at the crate root (existing `EFunType`/`EValType` re-exports preserved).
- Additive-only: no existing signature changed, no new dependency (reuses `rand`/`rand_distr`).

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib simulation::tests` — 29/29 pass (8 new + 21 existing).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo fmt` — clean.

## Oracles covered

- Oracle 4: zero AR operator ⇒ ‖C_1‖ < 0.15·‖C_0‖ (white noise).
- Oracle 6: rank-1 AR operator (coeff 0.8) ⇒ ‖C_1‖ > 0.1·‖C_0‖ (serial dependence).
- Oracle 5: bit-identical output for a fixed seed (both simulators).
- Divergence guard: 2×identity AR ⇒ `ComputationFailed` (not a panic, not an Inf-filled result).
- Dimension guards: wrong kernel length ⇒ `InvalidDimension{parameter: "ar_ops"/"ma_ops"}`; empty grid ⇒ `InvalidDimension{parameter: "argvals"}`.

## Divergence from freqdom/ftsa (documented in rustdoc)

- Identity innovation covariance (i.i.d. N(0,1) per grid point) vs freqdom's user-supplied σ.
