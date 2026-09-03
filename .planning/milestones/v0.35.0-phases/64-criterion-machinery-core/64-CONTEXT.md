# Phase 64: Criterion Machinery Core - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the pure, isolated, known-answer-testable math core of FOptDes in a new
`src/optimal_design.rs`: a single public `#[must_use]` `design_criterion` evaluator
that scores any caller-supplied set of design points against a fitted
`pace_fpca::PaceFpcaResult`, computing either the integrated trajectory-reconstruction
BLUP-MSE (FOD-01) or the A-/D-optimal posterior FPC-score covariance summary (FOD-02),
dispatched through a public `DesignCriterion` / `OptimalityKind` enum pair (FOD-03).

Includes the shared private `build_sigma_design` helper (p×p
`Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p`, mirroring `pace_fpca.rs:461–474`) plus both criterion
branches. **NO greedy loop** — that is Phase 65. This phase front-loads every numerical
make-or-break gate; every downstream selection result inherits any bug here.

Out of scope: greedy `optimal_design` wrapper, `OptDesConfig`/`OptDesResult` types,
crate-root/`prelude` full re-export surface, and the benchmark — all deferred to Phase 65.
Only the enums + `design_criterion` are additively re-exported from `lib.rs` this phase.

</domain>

<decisions>
## Implementation Decisions

### Public API Surface
- **Criterion selection via a single nested `DesignCriterion` enum**: `DesignCriterion::Trajectory`
  and `DesignCriterion::Score(OptimalityKind)`. One public dispatch point; mirrors the
  `pace_fpca` config-enum style. (Not two separate functions, not a flat 3-variant enum.)
- **`design_criterion` signature**:
  `design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>`.
  `selected` are **indices into `model.argvals`** (exact index arithmetic, no interpolation —
  matches the grid-constrained-candidates MVP decision). Read-only borrow of the model.
- **`OptimalityKind` variants**: `A` (trace of posterior covariance) and `D` (log-det of
  posterior covariance) only — the two locked criteria. No `E`/`G` now.
- **Empty-set `selected == &[]` returns the prior baseline** — `MSE(∅) ≈ Σ_k λ_k`,
  `A(∅) = Σ_k λ_k`, `D(∅) = Σ_k log λ_k`. This powers the Phase 65 greedy loop's start and the
  monotonicity gate. (Not rejected as `InvalidParameter`.)

### Numerical Contracts (locked in STATE.md at HIGH research confidence)
- **Shared `build_sigma_design`**: assembles `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p` where p = |selected|,
  row-major, mirroring `pace_fpca.rs:461–474`. Shape is `|S|×|S|`, **not** K×K. Both criterion
  branches call it, then differ only in post-solve usage. Solve via `linalg::cholesky_solve`
  (row-major, always available — NOT behind the `linalg` feature). Ridge-retry (`1e-8`) on
  near-singular Σ_d (`pace_fpca.rs:480–490` pattern); never panic.
- **Trajectory criterion (FOD-01)**: integrated Simpson-weighted conditional BLUP-MSE
  `Σ_j w_j (Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j))`. MUST use
  `helpers::simpsons_weights(&model.argvals)` (never uniform 1.0 — else grid-scale-wrong).
  Quadratic form includes Ω off-diagonals. Known-answer: `MSE(∅) ≈ Σ_k λ_k`, grid-invariant.
- **Score criterion (FOD-02)**: K×K posterior `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ` via
  `cholesky_solve` (the `pace_fpca.rs:547–558` A_mat/Ω_i pattern). A-opt = `trace(Cov)`,
  D-opt = `log det(Cov)` (NEGATIVE — posterior eigenvalues ≤ prior λ_k). Known-answer:
  `Cov(ξ|∅) = diag(λ)` → `A(∅) = Σλ_k`, `D(∅) = Σ log λ_k`. Do not drop the σ²I term.
- **Optimality-sign / monotonicity gate**: Trajectory, A-opt, and D-opt are all monotone
  NON-increasing as points are added: `criterion(S∪{t}) ≤ criterion(S) + 1e-12`. Assert in tests —
  guarantees the Phase 65 greedy loop minimizes (never maximizes). Catches sign flips.

### Validation
- Grid-constrained: every index in `selected` must be `< model.argvals.len()`; out-of-range →
  `FdarError::InvalidParameter`. Duplicate indices tolerated at criterion level (greedy excludes
  them upstream) but documented.
- `ncomp == 0` / `sigma2 <= 0` guards inherited from the supplied model; validate defensively and
  return `InvalidParameter` with contextual messages.

### Claude's Discretion
- Exact internal helper factoring (~3 private helpers), variable naming, and test-module layout
  are at Claude's discretion, following existing `optimal_design`-peer conventions
  (`kshape.rs`, `kernel_kmeans.rs`).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pace_fpca::PaceFpcaResult` — read-only source of `eigenvalues` (Vec<f64>, len ncomp),
  `eigenfunctions` (FdMatrix, m×ncomp column-major), `argvals` (Vec<f64>, len m), `sigma2` (f64),
  `ncomp` (usize). Σ_yi assembly at `pace_fpca.rs:461–474`; A_mat/Ω_i posterior-covariance pattern
  at `547–558`; ridge-retry at `480–490`.
- `linalg::cholesky_solve` — p×p linear solves, row-major, always available (not `linalg`-gated).
- `helpers::simpsons_weights(&argvals)` — quadrature weights for the trajectory integral.
- `helpers::linear_interp` — available but NOT needed this phase (candidates are on-grid indices).
- `iter_maybe_parallel!` — available; not required in Phase 64 (single-set evaluation), used by the
  Phase 65 greedy sweep.
- `error::FdarError` — `InvalidParameter` / `InvalidDimension` / `ComputationFailed` variants.

### Established Patterns
- New algorithm as a top-level `src/*.rs` peer (like `kshape.rs`, `kernel_kmeans.rs`), not a
  submodule directory — self-contained (config + result + enums + fns + private helpers).
- Public types derive `Debug, Clone, PartialEq`; `#[must_use]` on expensive evaluators.
- All public fns return `Result<T, FdarError>`; dimension/param checks at entry.
- Column-major `FdMatrix` indexing: element (row, col) at `row + col * nrows`.
- Inline `#[cfg(test)] mod tests` with known-answer numerical gates.

### Integration Points
- Additive `lib.rs` re-export of the enums + `design_criterion` only (partial surface — full
  `pub mod optimal_design` + prelude + greedy fn land in Phase 65).
- No `Cargo.toml` change; MSRV stays 1.81; `linalg` feature NOT required.

</code_context>

<specifics>
## Specific Ideas

- Reference: Ji & Müller (2017) optimal-design formulation; `fdapace` conventions.
- Both criteria share `build_sigma_design`; keep FOD-02 in this core phase so the Phase 65 greedy
  wrapper stays pure orchestration with no new math.

</specifics>

<deferred>
## Deferred Ideas

- Greedy `optimal_design` wrapper, `OptDesConfig`/`OptDesResult`, full re-export surface, benchmark
  → Phase 65.
- FOD-BREADTH (SR-criterion, exhaustive/branch-and-bound, CV-ridge selection, rank-1 Cholesky
  update, off-grid interpolated candidates) → future milestone.

</deferred>
