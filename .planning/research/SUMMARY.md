# Project Research Summary: FOptDes v0.35.0

**Project:** fdars-core — Optimal Experimental Design for Sparse FDA (FOptDes)
**Domain:** Rust functional-data-analysis library; PACE-based optimal-design module
**Milestone:** v0.35.0 (promotes GAP-05 from backlog)
**Researched:** 2026-09-02
**Confidence:** HIGH (mathematical formulas verified against Ji & Müller 2017 + fdapace R source; codebase read confirms reuse targets; no new dependency required)

---

## Executive Summary

FOptDes is a deterministic experimental-design module that selects optimal sparse-measurement locations for trajectory recovery or FPC-score prediction, given an already-fitted PACE model. All mathematics builds on the Ji–Müller (2017) framework; implementation reuses three existing fdars primitives (`cholesky_solve`, `simpsons_weights`, `linear_interp`) — **zero new crate dependencies required**. The recommended approach is a clean two-stage workflow: (1) fit PACE model via `pace_fpca()`, (2) select design points via `foptdes_trajectory()` or `foptdes_scores()`. No re-estimation of covariance surfaces occurs in the design step. The key risk is numerical correctness of the Σ_yi assembly (posterior covariance matrix); this is mitigated by copying the exact pattern from `pace_fpca.rs` lines 461–474 and validating against known-answer tests in Phase 64. A second risk is optimality-sign convention in the greedy loop; this is caught by asserting monotone criterion decrease across steps. Estimated two-phase build (Phases 64–65) addresses all table-stakes features and differentiators.

---

## Key Findings

### Recommended Stack

**All existing dependencies — NO new Cargo.toml entries required.**

The STACK.md research confirms that every required operation can be built on nalgebra 0.33, rayon 1.10 (optional, feature-gated), and existing fdars infrastructure:

| Technology | Role in FOptDes | Rationale |
|------------|-----------------|-----------|
| **nalgebra 0.33** | Provides eigendecomposition via `pace_fpca.rs`; not directly called by FOptDes | FOptDes consumes eigenvalues and eigenfunctions from `PaceFpcaResult`; no new nalgebra calls needed |
| **rayon 1.10** (optional, via `parallel` feature) | Parallelizes candidate-point evaluation in greedy inner loop | Each candidate evaluation is independent; `iter_maybe_parallel!` gates parallelism; determinism preserved via sequential argmin |
| **linalg::cholesky_solve** (always available, no feature gate) | Solves p×p posterior-covariance systems at each criterion evaluation | Identical call signature as in `pace_fpca.rs` line 493; small p ≤ 20 means O(p³) is negligible |
| **helpers::simpsons_weights** (always available) | Quadrature weights for integrating pointwise prediction variance | Consistent with project-wide Simpson's rule; used in `pace_fpca.rs` |
| **helpers::linear_interp** (always available) | Evaluate eigenfunctions at candidate design points | Same pattern as `pace_fpca.rs` line 457 |
| **parallel::iter_maybe_parallel!** macro (optional, via `parallel` feature) | Parallelize inner greedy candidate sweep | No RNG needed (criterion is deterministic); tie-breaking via sequential argmin ensures determinism |

**MSRV: 1.81 preserved.** No post-1.81 stabilizations required. `linalg` feature (which gates faer 0.23, requiring Rust 1.84) is not needed — `cholesky_solve` is always available without the feature gate.

### Expected Features

**Table Stakes (P1 — must ship for v0.35.0):**

1. **Trajectory-reconstruction greedy design** (`foptdes_trajectory`) — minimizes integrated BLUP prediction MSE of x̂(t) given noisy observations at selected design points. Canonical PACE FOptDes criterion from Ji & Müller 2017. Users expect this; every reference implementation exposes it first.

2. **FPC-score-prediction A-optimal greedy design** (`foptdes_scores(AOptimal)`) — minimizes trace of posterior score covariance, reducing total posterior variance across all K components. Natural complement to trajectory recovery when downstream use is FPC-score prediction.

3. **Design-criterion evaluator** (`design_criterion()` public function) — users must evaluate hand-crafted or historical designs against the model without re-running greedy. Independent of the greedy loop; reusable.

4. **Per-step R² output** (`criterion_curve`, `r2_curve` in result struct) — essential for budget-selection workflow; researchers need to plot R² vs. budget to choose minimum sufficient p. Both MATLAB PACE and R fdapace return this.

5. **Deterministic reproducibility** — no randomness; bit-for-bit identical results across runs and platforms. Determinism test (identical inputs → identical outputs) is mandatory.

**Differentiators (P2 — competitive advantage, not blocking v0.35.0 but planned):**

- **D-optimal score design** (`foptdes_scores(DOptimal)`) — log-det criterion has stronger theoretical properties than trace; fdapace does not expose this. Straightforward given A-optimal already implemented.

- **Multi-criterion evaluation** (`FOptDesEval` struct) — compute all three criteria (trajectory, A-optimal, D-optimal) in one call sharing a single Cholesky factorization. No reference does this; pure efficiency gain.

- **Parallel candidate evaluation** (via `--features parallel` flag) — embarrassingly parallel inner loop over G\S candidates at each step. Negligible for typical m ≤ 200, p ≤ 10; beneficial for dense grids.

**Anti-Features (explicitly out of scope for v0.35.0, documented as future work):**

- **Global exhaustive search** over all C(m,p) combinations — combinatorially infeasible for m > 20, p > 4. Greedy is proven near-optimal (Ji & Müller 2017). Document bounded suboptimality; exhaustive available only in fdapace for toy examples.

- **Scalar-response prediction design (SR-criterion)** — fdapace supports this but requires response vector as input to design step, coupling design to a specific outcome. Score-prediction criterion is response-agnostic and generalizes. Defer SR-design to v0.36.0+ if user demand confirmed.

- **Bayesian / MCMC D-optimal design** (Huang et al. 2025 extension) — MCMC sampling is non-deterministic and massively complex. Beyond scope of pure design on a fitted model.

### Architecture Approach

FOptDes is a single top-level module (`src/optimal_design.rs`, peer of `kshape.rs` and `kernel_kmeans.rs`) with two public functions, two public enums, two public config/result structs, and ~3 private helpers. No submodule directory needed — the algorithm is self-contained in ~300 lines.

**Module structure follows established fdars convention:**
- `OptDesConfig` struct (no `#[non_exhaustive]`) for caller-supplied settings (candidate grid, budget p, criterion type, optimality kind).
- `OptDesResult` struct (`#[non_exhaustive]`) for selected indices, selected time values, and criterion curve at each greedy step.
- `DesignCriterion` enum: `Trajectory` or `Score`.
- `OptimalityKind` enum: `AOptimal` or `DOptimal`.
- `foptdes_trajectory()` entry point (or `optimal_design()` unified entry point accepting criterion enum).
- `design_criterion()` public function — reusable criterion evaluator, independent of greedy loop.
- Private helpers: `build_sigma_design()`, `trajectory_criterion()`, `score_criterion()`.

**Critical pattern: criterion evaluation → greedy wrapper.** Phase 64 implements the reusable `design_criterion()` public function and shared `build_sigma_design()` private helper. Phase 65 wraps this in the greedy forward-selection loop. This ordering matches the `sbd.rs` → `kshape.rs` precedent (Phase 61 → 62): build the composable primitive first, then the algorithm that wraps it.

### Critical Pitfalls

1. **P1: Wrong Σ_yi Assembly (Missing σ² Ridge)** — Omitting `+ σ²I_p` causes Cholesky failure when |S| > K. Mitigation: Copy exact assembly from `pace_fpca.rs` lines 461–474.

2. **P2: Score Posterior Covariance (Wrong Matrix Inverse)** — Forgetting `diag(1/λ)` prior term causes singularity. A-optimality trace must *decrease* as points are added. Mitigation: Verify known-answer `Cov(ξ|∅) = diag(λ)`.

3. **P3: Integrated Prediction Variance (Missing Quadrature Weights)** — Omitting Simpson weights produces dimensionally incorrect values. Mitigation: Verify `MSE(∅) ≈ Σ_k λ_k`.

4. **P4: Optimality-Sign Convention** — Criterion must non-increase as points are added; greedy selects *minimizing* candidate. Sign flip causes worst-point selection. Mitigation: Monotone-decrease assertion in tests.

5. **P5: Near-Duplicate Candidates and σ²→0** — Fine grids or small σ² make Σ_yi nearly singular. Mitigation: Exclude already-selected indices; enforce σ² ≥ 1e-8 floor.

---

## Implications for Roadmap

**Two sequential phases (Phases 64–65) building on each other.**

### Phase 64: Core Criterion Machinery

**Rationale:** Greedy loop (Phase 65) depends entirely on `design_criterion()` public function. Implementing criterion evaluator first validates mathematical correctness with focused known-answer tests.

**Deliverables:** `optimal_design.rs` module with `DesignCriterion` enum, `OptimalityKind` enum, private helpers (`build_sigma_design`, `trajectory_criterion`, `score_criterion`), public `design_criterion()` function, comprehensive tests (Σ_yi assembly, criterion monotonicity, prior recovery, error paths).

**Research flag:** None — criterion formula well-documented.

### Phase 65: Greedy Selection Loop + Full Integration

**Rationale:** With Phase 64 criterion validated, greedy wrapper is thin O(budget × |candidates|) loop. This phase completes the public API with config/result types and entry-point functions.

**Deliverables:** `OptDesConfig` struct, `OptDesResult` struct, `foptdes_trajectory()`, `foptdes_scores()`, greedy forward-selection loop with parallelization and deterministic argmin, determinism test, benchmark.

**Research flag:** None — greedy is standard forward-selection algorithm.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | **HIGH** | No new dependencies; all reuse targets exist and well-used. Verified by direct codebase read. |
| **Features** | **HIGH** | Table-stakes from Ji & Müller 2017 and fdapace. Differentiators clearly incremental. Anti-features explicitly scoped. |
| **Architecture** | **HIGH** | Matches established fdars convention. Single-file module (peer of kshape.rs). Reuse pattern standard. Integration purely additive. |
| **Pitfalls** | **HIGH** | 12 pitfalls identified with concrete prevention strategies. Critical pitfalls grounded in mathematical invariants. |

**Overall: HIGH** — Mathematical foundation solid, implementation reuses tested primitives, phase decomposition follows patterns, pitfall strategies concrete and testable.

---

## Gaps to Address

None identified. Research covers all essentials: stack dependency inventoried, features scoped, architecture pattern established, pitfalls with phase-specific gates, numerical correctness roadmap clear.

---

## Sources

**Primary (HIGH):**
- Ji & Müller (2017) JRSSB 79(3):859–876 — normative math for design criteria
- fdapace R package v0.6.0 source — reference implementation, greedy algorithm
- fdars-core codebase (direct read) — `pace_fpca.rs` lines 461–474 (Σ_yi), 547–558 (Ω_i), 480–490 (ridge-retry), 493 (cholesky_solve); linalg.rs, helpers.rs, parallel.rs, matrix.rs
- PROJECT.md (fdars v0.35.0) — scope, two-stage workflow, GAP-05 promotion
- Yao, Müller & Wang (2005) JASA 100(470):577–590 — PACE BLUP formulas

**Secondary (MEDIUM):**
- Huang et al. (2025) WIREs — extended framework (A/D/E-optimality, Bayesian)
- MATLAB PACE@2.17 documentation — API baseline

---

**Research completed:** 2026-09-02
**Ready for requirements definition:** YES
