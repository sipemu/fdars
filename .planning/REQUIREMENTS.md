# Requirements: fdars v0.47.0 — Performance: Cross-Language Benchmarks & Optimization Pass

**Defined:** 2026-09-19
**Core Value:** A comprehensive, *fast* Rust functional-data-analysis library — this milestone produces evidence for the "fast" half (head-to-head vs scikit-fda + R) and improves the worst bottlenecks, behavior-preservingly.

## v0.47.0 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase (phases continue from v0.46.0: …111 → **112+**).

### Benchmark Harness (BENCH)

- [x] **BENCH-01**: An external `benchmarks/` harness (workspace root, **not** a Cargo member — zero `fdars-core` dependency) with a seeded single-source-of-truth generator emitting matched curve-set files (shared format, CSV-default) consumed identically by the Rust, Python, and R runners.
- [x] **BENCH-02**: A Python runner (scikit-fda 0.10.1 + pytest-benchmark, pinned via `requirements.lock`) and an R runner (`bench` + `fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`fdasrvf`, pinned via `renv.lock`) covering the overlapping FDA operation set (FPCA, basis smoothing, functional depth, functional boxplot, FPCR regression, elastic alignment/Karcher mean).
- [x] **BENCH-03**: fdars timings collected from the **existing** criterion 0.5 harness (parse `target/criterion/*/estimates.json`) into the shared comparison format — no new dev-dependency, no `cargo-criterion` binary.
- [x] **BENCH-04**: Fairness controls — pin `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1` for the baseline run and capture full version + BLAS-backend + CPU + thread-env metadata into `benchmarks/results/env.json`.
- [x] **BENCH-05**: A numeric-agreement gate that verifies each library computes the same result (sign-aligned FPCA correlation > 0.99; per-operation tolerances — 1e-3 depth/regression, 1e-2 elastic warping) **before** any timing is recorded; abort/flag on mismatch.

### Baseline & Ranking (BASE)

- [x] **BASE-01**: A cross-language baseline established across all three ecosystems over an N×M input grid (≥3 sizes; elastic target at N≥100), merged into `comparison.csv`.
- [x] **BASE-02**: An evidence-ranked hotspot list combining the cross-language results with the criterion suite (absolute + relative-to-baseline leverage), gated on measurement noise (RSD < 5% / CI < ±5%).

### Optimization (OPT)

- [ ] **OPT-01**: Optimize the elastic-alignment hot path (anchor target, O(N²·M²)) in `src/alignment/`, strictly non-breaking + numeric-identical, with committed before/after criterion evidence.
- [ ] **OPT-02**: Optimize the next-ranked hot paths (SVD/FPCA + covariance surface, conditional on the BASE-02 ranking), same non-breaking + numeric-identity proof pipeline.
- [ ] **OPT-03**: Every optimization commit preserves numeric identity and all build variants — `cargo test --release --features linalg` green, determinism guardrail green, all five builds (default / `linalg` / `serde` / `linalg,serde,parallel` / WASM) compile, and the pmat quality-gate stays green.

### Report (RPT)

- [ ] **RPT-01**: Re-run the full harness post-optimization and publish a perf report with head-to-head before/after tables, N×M scaling analysis, methodology disclosure, and reproduction instructions (likely a README speed story).

### Release (REL)

- [ ] **REL-01**: Bump 0.46.0 → 0.47.0, write the CHANGELOG (root + crate), and land all whole-crate gates green — `cargo fmt --check`, `clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build, 28 examples + doctests, `cargo package`, **plus** the pmat quality-gate. The `git tag v0.47.0` → crates.io publish is the deferred operator step.

## Future Requirements

Deferred beyond this milestone.

### Continuous Performance Tracking

- **PERF-CI-01**: asv/airspeed-velocity-style continuous perf tracking + CI perf-regression gating across machines. Deferred — a single-milestone head-to-head does not need it.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New `fdars-core` crate dependency (SIMD crate, dhat, etc.) | Standing project convention — no new Cargo dep; the harness is external, optimizations reuse existing infra. |
| Public API / signature / behavioral changes | Milestone is strictly non-breaking + numeric-preserving (protects R + WASM bindings + 28 examples). |
| New FDA algorithms / capabilities | Performance milestone, not a feature milestone. |
| Updating the external `pyfda` / `fdars-r` bindings | Separate external packages, out of `fdars-core` scope (`fdars-j75` tracks the R migration). |
| `fdars-j75` R migration and the terminal `1.0-CUT` | Remaining 1.0-checklist items; not touched this milestone. |
| Multi-machine / GPU / cross-platform perf | Out of scope; single controlled-environment comparison. |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BENCH-01 | Phase 112 | Complete |
| BENCH-02 | Phase 112 | Complete |
| BENCH-03 | Phase 112 | Complete |
| BENCH-04 | Phase 112 | Complete |
| BENCH-05 | Phase 112 | Complete |
| BASE-01 | Phase 113 | Complete |
| BASE-02 | Phase 113 | Complete |
| OPT-01 | Phase 114 | Pending |
| OPT-02 | Phase 115 | Pending |
| OPT-03 | Phase 115 | Pending |
| RPT-01 | Phase 116 | Pending |
| REL-01 | Phase 117 | Pending |

**Coverage:**

- v0.47.0 requirements: 12 total
- Mapped to phases: 12 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-19*
*Last updated: 2026-09-19 after roadmap creation (traceability mapped)*
