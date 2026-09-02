# Research Summary: k-Shape Clustering & Shape-Based Distance (v0.34.0)

**Milestone:** v0.34.0 Implementation  
**Domain:** Functional Data Analysis — clustering via shape-based time-series distance  
**Researched:** 2026-09-02  
**Status:** READY FOR REQUIREMENTS → ROADMAP

---

## Executive Summary

k-Shape clustering is a proven shape-centric alternative to distance-based clustering for functional data. It solves the core gap (GAP-03) for shape-invariant similarity metrics in fdars: two curves with identical shape but different offset or scale now correctly measure distance ≈ 0. The implementation requires **no new crate dependencies** — all four deliverables (SBD distance core, k-Shape fit, out-of-sample predict, SBD-k-medoids integration) build entirely on existing infrastructure (rustfft for FFT, nalgebra for eigendecomposition, shapelet z-normalization, kernel_kmeans architecture patterns).

The milestone is **strictly sequential in three phases:** Phase 61 (SBD distance primitive via FFT cross-correlation) → Phase 62 (k-Shape fit with shape-extraction centroid refinement + predict) → Phase 63 (SBD-k-medoids adapter + integration). This ordering is mandatory because each phase depends on the previous one's deliverables. Confidence is HIGH across stack, features, and architecture. Pitfalls research identified 14 silent-correctness killers with clear prevention strategies — all addressable in Phase 61 (core numerical primitives) and Phase 62 (centroid refinement).

**Critical success factors:** Correct FFT zero-padding (`2m-1` minimum, rounded to power-of-two) and coefficient-normalized cross-correlation (NCC) divided by `‖x‖·‖y‖`, not by lag count. Shape extraction must use the top eigenvector of `M = Q^T S Q` (not arithmetic mean or bottom eigenvector), with sign disambiguation via correlation to the member mean. Deterministic seeding via `seed + restart_idx` ensures reproducibility across `n_init` restarts.

---

## Key Findings Across Research Files

### From STACK.md

| Finding | Implication |
|---------|------------|
| **NO new crate dependencies.** | Milestone is locked to existing Cargo.toml; no feature-flag or version changes needed. Simplifies CI, release process, CRAN compliance. |
| **Reuse: rustfft (6.2), nalgebra (0.33), rand (0.8), rayon (1.10).** | All already direct dependencies with proven usage in `fts/spectral.rs`, `seasonal/mod.rs`, `kernel_kmeans.rs`. Copy idioms verbatim. |
| **Reuse: shapelet::z_normalize_window + z_normalize_into (v0.33.0).** | Series z-normalization mandatory for scale/offset invariance. Population-std convention with `STD_EPS = 1e-12` guard already present. |
| **Reuse: kernel_kmeans.rs architecture (n_init, seeding, empty-cluster recovery, predict).** | k-Shape mirrors kernel_kmeans structurally. Reduces novel code risk. |
| **FFT-size minimum: `(2 * sz - 1).next_power_of_two()`.** | Power-of-two optimization already established in `seasonal/mod.rs:350`. |
| **MSRV stays 1.81; no faer gate needed.** | nalgebra 0.33 `SymmetricEigen` is core (not behind `linalg` feature). Shape-extraction m×m eigenproblem is small (m ≤ 500). |
| **Serde support via `#[cfg_attr(feature = "serde", ...)]`.** | Follow convention on `KShapeConfig` and `KShapeResult`. Users save/reload fitted centroids for production predict. |

### From FEATURES.md

| Category | Finding | Roadmap Impact |
|----------|---------|-----------------|
| **P1 (Table Stakes)** | `sbd(x, y) -> (dist, shift)` must return both distance AND optimal shift. Shift is mandatory for shape-extraction centroid alignment. | Phase 61 deliverable is `SbdResult { dist: f64, shift: i64 }`. |
| **P1 (Table Stakes)** | `sbd_distance_matrix(data) -> FdMatrix` n×n symmetric pairwise matrix. Needed for k-medoids consumer and O(n·k) assignment path. | Phase 61 extension; reuse `metric::self_distance_matrix` for parallel upper-triangle loop. |
| **P1 (Table Stakes)** | Shape extraction is **not arithmetic mean**; it is the top eigenvector of `M = S^T (I - 11^T/n) S`. This distinguishes k-Shape from k-means. | Phase 62 centroid step. Wrong algorithm = silent failure (converges to k-means, not k-Shape). |
| **P1 (Table Stakes)** | `kshape_fd(data, config) -> KShapeResult` with full n_init loop, deterministic seeding, empty-cluster recovery, convergence check, inertia tracking. | Phase 62 headline deliverable; follows `kernel_kmeans.rs` structurally. |
| **P1 (Table Stakes)** | All `Result<T, FdarError>` error handling; dimension checks (n > 0, k > 0, k ≤ n, n_init ≥ 1). | Standard fdars convention. |
| **P2 (Differentiator)** | `n_init = 10` default (not tslearn's 1). k-Shape is sensitive to initialization; 10 restarts dramatically improves quality. | `KShapeConfig::default()` sets `n_init: 10`. Document as "exceeds tslearn for robustness." |
| **P2 (Differentiator)** | Rayon parallelism over assignment step via `iter_maybe_parallel!`. SBD assignment (n×k calls per iteration) is embarrassingly parallel. | Phase 62 implementation detail; modest win on large n. |
| **P2 (Differentiator)** | Criterion benchmark: measure fit time vs (n, m, k); quantify rayon benefit. | Phase 63 (optional, follows convention). |

**MVP definition:** Phases 61 + 62 complete all table-stakes. Phase 63 is a thin adapter (two lines). All P2+ features are post-shipping polish.

### From ARCHITECTURE.md

| Layer | Finding | Roadmap Impact |
|-------|---------|-----------------|
| **Public API** | New re-exports in `lib.rs`: `pub mod kshape; pub use kshape::{kshape_fd, KShapeConfig, KShapeResult, sbd_kmedoids}; pub use metric::{sbd, sbd_matrix_fd, SbdResult}`. | Phase 63 only; Phases 61–62 are internal until crate-root re-export. |
| **Module structure** | SBD in `src/metric/sbd.rs` (peer of gak.rs, soft_dtw.rs). k-Shape in `src/kshape.rs` (top-level, peer of kernel_kmeans.rs). Mirrors metric-first layering. | Phase 61: new `metric/sbd.rs`. Phase 62: new `kshape.rs`. Phase 63: modify `lib.rs`, `metric/mod.rs`, `prelude.rs`. |
| **Component boundaries** | `metric/sbd.rs` has zero imports from k-Shape; `kshape.rs` imports `metric::sbd`. Enables reuse (sbd_kmedoids, future hierarchical). | Strict one-way dependency. |
| **Data flow for SBD** | `&[f64] x, y` → z-normalize → zero-pad → `FftPlanner` → FFT both → multiply + conj → IFFT → rearrange lags → max(NCC) → `1 - max_NCC`. Shift = argmax (sign-adjusted). | Phase 61 implementation order. Each step is a pitfall risk. |
| **Data flow for k-Shape** | `FdMatrix` → random partition init → loop: (assignment + recovery) + (centroid update + sign fix + re-z-norm) → convergence. Best restart by inertia. | Phase 62 implementation order. Centroid update is bottleneck (k sequential eigh calls). |
| **Integration points** | `metric/sbd.rs` → `shapelet/distance.rs`, `rustfft`, `metric/mod.rs`. `kshape.rs` → `metric/sbd`, `nalgebra`, `alignment/clustering`. No new deps; all imports existing. | No API changes to existing modules; pure additive. |
| **Reuse map** | 10 existing primitives directly reused unchanged: z_normalize_window, FftPlanner, SymmetricEigen, seed pattern, iter_maybe_parallel!, FdMatrix methods, KMedoidsConfig, empty-cluster recovery, self_distance_matrix. | Reduces test surface (tested infrastructure reused). |

### From PITFALLS.md

**14 pitfalls identified; 11 are Phase 61–62 (core numerical primitives), 3 are Phase 62–63 (integration).** All have clear verification hooks.

| Pitfall | Severity | When It Breaks | Prevention |
|---------|----------|-----------------|------------|
| P1: FFT length < 2m−1 (circular wrap) | CRITICAL | Every pair with nonzero shift | `fft_len = (2 * sz - 1).next_power_of_two()` |
| P2: Count-normalized NCC (not ‖x‖·‖y‖) | CRITICAL | All SBD distances silently wrong | `ncc[s] = cc_raw[s] / (norm_x * norm_y)` |
| P3: Z-normalization missing in SBD | CRITICAL | Scale/offset-sensitive distances | `z_normalize_window(x); z_normalize_window(y)` at top |
| P4: Wrong lag sign from argmax (fftshift) | HIGH | Negative shifts return wrap-around index | `shift = if idx <= n-1 { idx as i64 } else { (idx as i64) - fft_len as i64 }` |
| P5: Arithmetic mean instead of eigenvector | CRITICAL | Silent algorithm degradation (k-means, not k-Shape) | Use `SymmetricEigen` on `M = S^T(I - 11^T/n)S`; take last eigenvector |
| P6: Eigenvector sign ambiguous | HIGH | Inverted centroid; oscillation | `if dot_sum < 0 { v = -v }` (correlation with member mean) |
| P7: Members not shift-aligned before shape extraction | CRITICAL | Centroid update useless; no convergence | Store `shifts[i]` from SBD; apply circular shift before building S |
| P8: Centroid not re-z-normalized post-extraction | MEDIUM | Miscalibrated NCC in next iteration | `centroid = z_normalize_window(&eigenvector)` after each extraction |
| P9: Empty cluster unhandled | HIGH | Panic or NaN centroid | Use `recover_empty_clusters` pattern from `kernel_kmeans.rs` |
| P10: Inertia not monotone-decreasing | HIGH | Oscillation; convergence false on clean data | Track per iteration; convergence on label stability OR `Δinertia/inertia < tol` |
| P11: n_init seeding wrong | HIGH | Identical restarts or non-determinism | `seed.wrapping_add(restart as u64)` per restart |
| P12: IFFT scale factor not divided out | CRITICAL | NCC off by fft_len; SBD wrong | Divide raw IFFT by `fft_len` before NCC normalization |
| P13: k-Medoids fed wrong distance | HIGH | Silent algorithm swap (L2 instead of SBD) | Explicit `sbd_distance_matrix` call in doctest |
| P14: Predict re-estimates centroid | MEDIUM | Training-set predict returns different labels | Use stored centroids as-is; no re-z-norm in predict |

All pitfalls have 1–2 line verification hooks (e.g., `sbd(x,x) ≈ 0`, shifted-copy test, two-group recovery, determinism check).

---

## Implications for Roadmap

### Recommended Three-Phase Sequential Structure

**Phase 61: SBD Distance Core** (2–3 days)
- **Deliverables:** `sbd()` returning `(dist, shift)`, `sbd_distance_matrix()` n×n symmetric matrix, `SbdResult` struct.
- **Verification gates:** Self-distance ≈ 0, symmetry, shifted-copy (all shifts), NCC in [-1,1], offset/scale invariance, constant-series guard.
- **Risk:** Pitfalls P1, P2, P3, P4, P12 (all numerical). Mitigation: verification hooks mandatory before Phase 62 starts.
- **Test coverage requirement:** ≥90%.
- **Code review gate:** Must pass before Phase 62.

**Phase 62: k-Shape Fit + Predict** (3–4 days)
- **Dependencies:** Phase 61 complete (sbd, sbd_distance_matrix, SbdResult available).
- **Deliverables:** `kshape_fd()`, `KShapeConfig`, `KShapeResult`, `predict()` method, thin `sbd_kmedoids()` adapter.
- **Verification gates:** Two-group recovery (centroid correlation >0.99), determinism (seed reproducibility), n_init benefit, empty-cluster (no panic, all sizes ≥1), inertia monotone-decreasing, predict round-trip (`predict(train) == res.cluster`), convergence on synthetic 2-group/3-group/overpartitioned data.
- **Risk:** Pitfalls P5–P11, P14 (shape extraction, seeding, predict consistency). Mitigation: synthetic-data known-answer tests must pass.
- **Code review gate:** Must pass before Phase 63.

**Phase 63: Integration & Delivery** (1 day)
- **Dependencies:** Phase 61 + Phase 62 complete.
- **Deliverables:** Re-exports from `lib.rs`, `prelude.rs` updates, Criterion benchmark, example file, integration test.
- **Verification gates:** P13 integration test (sbd_kmedoids ≡ explicit sbd_matrix_fd + kmedoids_from_distances).
- **Risk:** P13 only (low). Mitigation: doctest shows explicit flow.

### Research Phase Flags

| Phase | Research Needed | Justification |
|-------|---|---|
| **Phase 61** | **MEDIUM** — Run `/gsd-plan-phase --research-phase 61` | FFT zero-padding, NCC normalization, lag indexing are numerical fundamentals. Recommend pair-program validation: two independent FFT implementations cross-checked against aeon/tslearn reference. Known-answer tests mandatory. High correctness risk justifies research depth. |
| **Phase 62** | **MEDIUM** — Run `/gsd-plan-phase --research-phase 62` | Shape-extraction eigenvector (M = Q^T S Q, sign fix, re-z-norm) is novel algorithmic step. Validate matrix formulation against Paparrizos/tslearn; confirm nalgebra eigenvalue order (ascending); test sign-fix criterion. Two-group shifted-motif known-answer test required before implementation. No domain gaps. |
| **Phase 63** | **NONE** — Standard patterns | Wrapper + re-exports + benchmarks. No research needed. |

### Confidence Assessment

| Area | Level | Evidence | Gaps |
|------|-------|----------|------|
| **Stack** | HIGH | Direct codebase read (rustfft/nalgebra/rand/rayon usage in fts/, seasonal/, kernel_kmeans). tslearn API verified via GitHub (20+ code patterns cross-checked). MSRV 1.81 sufficient (SymmetricEigen stable since nalgebra 0.30+). No new deps. | None. Dependency audit complete. |
| **Features** | MEDIUM | SBD NCCc formula HIGH from aeon docs (authoritative). k-Shape algorithm MEDIUM from tslearn source + Paparrizos secondary sources (PDF binary not extractable; formula verified via 5 independent sources: aeon, dtwclust R, kshape-python, cybergarage, tslearn). MVP unambiguous. | Minor: Paparrizos PDF not directly readable; formula reconstructed via secondary sources. Confidence sufficient for implementation. |
| **Architecture** | HIGH | Codebase module structure fully audited. Reuse map verified (10 existing primitives, zero new public APIs). Data flow traced end-to-end (two complete paths: SBD pairwise and k-Shape fit). | None. Structure sound and follows established patterns. |
| **Pitfalls** | HIGH | 14 pitfalls identified (Paparrizos + tslearn source + fdars kernel_kmeans patterns + rustfft/nalgebra conventions). All have verification hooks (1–2 lines each). All addressable in Phase 61–62 with zero design changes. Recovery cost mapped (14/14 have clear fixes). | None. Well-understood based on k-means failure modes + SBD-specific numerical hazards. |

### Known Unknowns (No Blocking Risk)

| Item | Relevance | Plan |
|------|-----------|------|
| Approximate eigendecomposition for large m (>500) | MEDIUM | Shape-extraction eigh is O(m³). For typical m ≤ 200, negligible (<100ms per cluster). Deferred to v0.34.1 if profiling shows bottleneck. No design impact now. |
| Multivariate SBD (per-channel, then average) | LOW | Requires multivariate FdMatrix representation decision (TBD in architecture). Gate as v0.35+ feature. |
| GPU-accelerated FFT cross-correlation | LOW | Contradicts WASM deployment model (OOS-01). Deferred indefinitely. |
| Streaming/online k-Shape | LOW | Batch MVP sufficient. Gate as v0.35+. |

---

## Success Criteria

**Phase 61 is done when:**
- `sbd(x: &[f64], y: &[f64]) -> Result<SbdResult, FdarError>` with `dist: f64, shift: i64` implemented.
- `sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>` (n×n symmetric, parallelized via `iter_maybe_parallel!`).
- All verification hooks pass: self-distance ≈ 0, symmetry, shifted-copy (all shifts), NCC bounded in [-1,1], offset/scale invariance, constant-series guard.
- Test coverage ≥90%.
- Code review passed.

**Phase 62 is done when:**
- `kshape_fd()`, `KShapeConfig`, `KShapeResult`, `predict()` all public and fully tested.
- All verification hooks pass: two-group recovery (centroid correlation >0.99), determinism (same seed = identical output), n_init benefit validated, empty-cluster (no panic, all sizes ≥1), inertia monotone-decreasing per iteration, predict round-trip (`predict(train) == res.cluster`).
- Convergence validated on synthetic 2-group, 3-group, and overpartitioned data (k > natural clusters).
- Code review passed.

**Phase 63 is done when:**
- All symbols re-exported from `lib.rs`, `prelude.rs` updated with `KShapeConfig`, `KShapeResult`.
- Criterion benchmark compiles and runs; example file compiles and runs.
- Integration test: `sbd_kmedoids(data, cfg)` output ≡ `(sbd_distance_matrix(data) + kmedoids_from_distances)`.
- Doctest demonstrates explicit sbd_distance_matrix → kmedoids_from_distances flow.
- Docs pass clippy and `cargo test --doc`.

---

## Dependency & Feature-Flag Summary

**No Cargo.toml changes needed.** All dependencies already present with proven usage:
- rustfft 6.2 — FFT cross-correlation (fts/spectral.rs, seasonal/mod.rs)
- nalgebra 0.33 — SymmetricEigen for shape extraction (regression.rs, fts/spectral.rs)
- rand 0.8 — StdRng per-restart seeding (kernel_kmeans.rs, clustering.rs)
- rayon 1.10 — Optional parallelism via `parallel` feature and `iter_maybe_parallel!` macro
- shapelet (internal module, v0.33.0) — z_normalize_window, z_normalize_into for SBD + centroid

**MSRV:** Stays at 1.81 (SymmetricEigen is in nalgebra core, not behind `linalg` feature).

**Feature flags:** `serde` support on `KShapeConfig` and `KShapeResult` via `#[cfg_attr(feature = "serde", derive(...))]` (standard pattern matching 20+ other config/result types in codebase).

---

## Integration Points (All Additive)

- **`metric/sbd.rs` imports:** `shapelet::distance::{z_normalize_window, z_normalize_into}`, `rustfft`, `metric::self_distance_matrix` helper (existing).
- **`kshape.rs` imports:** `metric::sbd::{sbd, sbd_distance_matrix, SbdResult}`, `nalgebra::{DMatrix, SymmetricEigen}`, `alignment::clustering::{kmedoids_from_distances, KMedoidsConfig, KMedoidsResult}`.
- **`lib.rs` updates:** Add `pub mod kshape;` and re-exports. No existing symbol changes.
- **`metric/mod.rs` updates:** Add `pub mod sbd;` and re-exports to barrel.
- **`prelude.rs` updates:** Add `KShapeConfig`, `KShapeResult` to convenience exports.

**Result:** Zero breaking changes. Pure additive integration.

---

## Sources

- **STACK.md** — Dependency verification, API signatures, existing usage patterns, reuse map
- **FEATURES.md** — Mathematical specification (NCC formula, SBD distance, shape extraction), MVP definition, feature prioritization, complexity analysis
- **ARCHITECTURE.md** — Module structure, data flow diagrams, reuse boundaries, phase deliverables, integration points
- **PITFALLS.md** — 14 correctness pitfalls with prevention strategies and verification hooks, recovery cost mapping, phase-to-pitfall assignments
- **Primary sources:** Paparrizos & Gravano (2015) "k-Shape: Efficient and Accurate Clustering of Time Series" SIGMOD; tslearn 0.9.0 source (GitHub); aeon sbd_distance docs; dtwclust R package; fdars-core kernel_kmeans.rs, fts/spectral.rs, seasonal/mod.rs, shapelet/distance.rs, alignment/clustering.rs

---

*Synthesis complete: 2026-09-02*  
*Ready for orchestrator → requirements definition → roadmap creation*
