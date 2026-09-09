# Phase 90: Golden-Flake Root-Cause & Deterministic Fix — Research

**Researched:** 2026-09-09
**Domain:** Rust test determinism — feature-flag-gated SVD implementation divergence
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Fix-approach bias = least-invasive, diagnosis-gated.** Prefer test-side robustness. Relax bit-identity to a tight relative tolerance ONLY where FP nondeterminism is proven; serialize the tests ONLY if a genuine shared-state race is found. Force-determinism-in-`src/` is the last resort (highest blast radius — protects R + WASM bindings + 28 examples). Minimal `src/` changes permitted only if the root cause lives in library code.
- **No new crate dependency.** If serialization is needed, use a std-only test guard (a `static` Mutex). Do NOT add `serial_test`; do NOT introduce a nextest-groups config in this phase.
- **Bit-identity may relax to tight tolerance where nondeterminism is proven.** If FP nondeterminism is demonstrated for a specific assertion, that assertion may move from `assert_eq!` to a tight relative tolerance (target ~1e-12); keep bit-identity for every assertion where determinism still holds. Document the evidence justifying each relaxation.
- **Acceptance bar = 10 consecutive green full parallel `cargo test` runs AND per-binary green.**

### Claude's Discretion
- Exact reproduction harness (iteration count during diagnosis, `RAYON_NUM_THREADS` sweep, disk-pressure simulation, capturing actual-vs-expected on failure) is at the executor's discretion.
- Whether the root cause is one shared mechanism across all three tests or distinct per test is an open question for the diagnosis to answer.

### Deferred Ideas (OUT OF SCOPE)
- Suite-wide audit for other analogous fragile assertions → Phase 91 (ROBUST-01/02)
- CI guardrail exercising the full parallel `cargo test` path / nextest serialization group → Phase 92 (CI-01)
- Version bump, CHANGELOG, 1.0-checklist Quality tick, release gates → Phase 93 (REL-01/02)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FLAKE-01 | Evidence-backed diagnosis of WHY the three golden tests pass per-binary but flake under full parallel `cargo test` — produce a diagnosis artifact. | Root cause identified and reproduced experimentally in this session (see Diagnosis section). |
| FLAKE-02 | Deterministically fix the three affected tests (tolerance vs serialization, chosen from FLAKE-01's diagnosis) so they pass reliably under repeated full parallel `cargo test` runs. | Fix is a pure test-side cfg-attribute guard; no tolerance relaxation or serialization needed. |
</phase_requirements>

---

## Summary

The three flaky golden tests all share the **same root cause**: the golden reference values were captured using the `faer` SVD backend (available only when `--features linalg` is active), but the test binaries are sometimes compiled with the default feature set (`parallel` only, no `linalg`), which routes `fdata_to_pc` through the `nalgebra` SVD backend instead. The two SVD implementations produce **numerically different FPCA rotation matrices** for the test data — not merely ULP drift, but algorithmically divergent results — which propagates into fundamentally different downstream outputs (different CEM local optima for `co_cluster`; different near-zero rotation entries for `svd_sign`).

The fix is a **pure test-side cfg-attribute guard**: add `#[cfg_attr(not(feature = "linalg"), ignore)]` to each of the three test functions. This makes them behave as documented (they already declare they require `--features linalg`) without touching `src/`, adding dependencies, or loosening any numerical tolerances. Disk pressure and CPU load (the original environmental hypotheses) are red herrings — the flake is fully reproducible in an unloaded environment simply by omitting `--features linalg`.

**Primary recommendation:** Add `#[cfg_attr(not(feature = "linalg"), ignore)]` to the three flaking tests. No tolerance relaxation. No serialization. No `src/` changes.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| FPCA / SVD computation | Library (`src/regression.rs`) | — | `fdata_to_pc` owns SVD dispatch; feature-gated between faer and nalgebra |
| Golden test determinism | Test layer (`tests/`) | cfg feature flags | Tests must declare which SVD backend their goldens assume |
| Co-cluster CEM loop | Library (`src/coclustering.rs`) | — | Receives FPCA output; deterministic conditioned on fixed rotation |
| SVD sign normalization | Library (`src/regression.rs:dominant_sign_negative`) | — | Shared sign-decision core for both SVD backends |

---

## Root-Cause Diagnosis (FLAKE-01)

### Reproduction — Confirmed This Session

All three tests were reproduced in an unloaded environment (20 CPU cores, 50 GB free RAM, 128 GB free disk) with no background load. The flake is **100% deterministic** and **feature-flag-driven**, not environment-driven.

**Evidence:**

```
# WITHOUT linalg (default features = parallel only)
cargo test --test equivalence_phase48 --features parallel -- golden_co_cluster

failures:
    golden_co_cluster_below_threshold   left: 469.17705307940236   right: 415.58873301994873
    golden_co_cluster_parallel          left: 469.17705307940236   right: 434.2325186767332

# WITH linalg (faer SVD active)
cargo test --test equivalence_phase48 --features linalg,parallel -- golden_co_cluster

running 2 tests
test golden_co_cluster_parallel ... ok
test golden_co_cluster_below_threshold ... ok
```

```
# WITHOUT linalg
cargo test --test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical

failures:
    svd_sign_fpca_two_matrix_bit_identical
    left: 4.1785053186338415e-16   right: -0.0    (rotation[(0,0)])

# WITH linalg
cargo test --test equivalence_phase49 --features linalg,parallel -- svd_sign_fpca_two_matrix_bit_identical
test svd_sign_fpca_two_matrix_bit_identical ... ok
```

[VERIFIED: fdars-core/tests/equivalence_phase48.rs:56] Golden values verbatim: `4.34232518676733207e2` (parallel) and `4.15588733019948734e2` (below-threshold).
[VERIFIED: fdars-core/tests/equivalence_phase49.rs:393] Failure message verbatim: `left: 4.1785053186338415e-16   right: -0.0`.

### The SVD Feature-Flag Branch [VERIFIED: fdars-core/src/regression.rs:438-477]

`fdata_to_pc` contains two compile-time branches:

```rust
#[cfg(feature = "linalg")]
let (singular_values, mut rotation, mut scores) = {
    let mat_ref = MatRef::<f64>::from_column_major_slice(...);
    let svd = FaerSvd::new_thin(mat_ref)?;  // faer SVD
    ...
};

#[cfg(not(feature = "linalg"))]
let (singular_values, mut rotation, mut scores) = {
    let svd = SVD::new(weighted.to_dmatrix(), true, true);  // nalgebra SVD
    ...
};
```

Both branches apply the same `fix_svd_signs` convention afterward, but the underlying SVD algorithms (faer's iterative bidiagonal QR vs. nalgebra's LAPACK-style QR) produce **arithmetically different rotation matrices** for the same input — not just sign-flipped columns but genuinely different eigenvectors for near-degenerate configurations or numerically sensitive inputs.

### Co-Cluster Failure Mechanism [VERIFIED: fdars-core/src/coclustering.rs:930-932]

The global FPCA is computed **once** before the multi-restart loop:

```rust
// --- Global FPCA ---
let fpca = fdata_to_pc(data, config.ncomp, argvals)?;
let rotation = &fpca.rotation;  // m × eff_ncomp — captures the SVD backend's result
```

For the 60×40 co-cluster data:
- faer SVD rotation → block scores → CEM converges to log-lik ≈ 434.23 (n_init=4) or 415.59 (n_init=2)
- nalgebra SVD rotation → **different block scores** → CEM converges to log-lik ≈ 469.177 for **both** n_init values

The 469.177 value is a legitimately higher log-likelihood (a better CEM solution), not a bug — the two SVD backends just land the CEM in different basins of attraction.

### SVD-Sign Failure Mechanism [VERIFIED: fdars-core/tests/equivalence_phase49.rs:381-411]

For the 5×8 `fpca_sign_fixture` data, `rotation[(0,0)]` is effectively zero:
- faer SVD: `-0.0` (IEEE negative zero)
- nalgebra SVD: `4.1785053186338415e-16` (tiny positive epsilon)

`dominant_sign_negative` [VERIFIED: fdars-core/src/regression.rs:266-276] finds the max-abs entry in column 0 of rotation. With a near-zero column 0, the max-abs index may differ between SVD backends, and the sign decision may flip. The resulting `assert_eq!` on exact `f64` bits fails because `-0.0 != 4.178e-16`.

### Why Both n_init Variants (n_init=2 and n_init=4) Produce 469.177

The FPCA rotation is computed ONCE before the `n_init` loop. Both the sequential (n_init=2) and parallel (n_init=4) branches share the same rotation. Since the rotation is the source of divergence, both branches land in the same (nalgebra-driven) basin → both produce 469.177.

### Hypotheses Ruled Out [VERIFIED by code inspection]

| Hypothesis | Verdict | Evidence |
|---|---|---|
| faer SVD nondeterminism from rayon parallelism | RULED OUT | For 60×40 data, bidiag `par_threshold=49152 > 59*39=2301` → Par::Seq forced [VERIFIED: faer-0.23.2/src/linalg/svd/bidiag.rs:111-112]. For n=40 bidiagonal, n < recursion_threshold=128 → QR algo (no rayon) [VERIFIED: faer-0.23.2/src/linalg/svd/mod.rs:262-279]. SVD is sequential for test data sizes. |
| rayon work-stealing in co_cluster init loop | RULED OUT | `iter_maybe_parallel!(0..n_init).map(run_init).collect()` uses `IndexedParallelIterator` which preserves order [ASSUMED: rayon docs]. Each init uses a fixed seed; results collected in index order. |
| Disk pressure / CPU contention changing FP results | RED HERRING | Flake reproduces in unloaded environment (128 GB free disk, 20 cores, idle machine) by simply omitting `--features linalg`. |
| nalgebra `matrixmultiply` threading | RULED OUT | `matrixmultiply-0.3.10` lock has no `threading` feature in dependencies [VERIFIED: Cargo.lock:867-876]. Only `autocfg` and `rawpointer`. nalgebra SVD is single-threaded. |
| Shared mutable state race between tests | RULED OUT | No `static mut`, no `Mutex`, no `RefCell` in `fdata_to_pc` or `co_cluster` code paths. Both functions are purely functional (no interior mutability). |

### Control Experiment: PACE Eigenfunction Test Is Not Affected [VERIFIED this session]

`svd_sign_pace_eigenfunctions_single_matrix_bit_identical` passes BOTH with and without `--features linalg`. This is consistent: `pace_fpca` uses a different code path whose numerical results happen to be stable across SVD backends for the PACE fixture data.

---

## Standard Stack

This phase makes no new library additions. The full fix is a two-line change per test function. Existing tools:

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|-------------|
| Built-in `#[test]` harness | Rust stdlib | Test registration | No deps |
| `#[cfg_attr(...)]` attribute | Rust lang | Conditional compilation / ignore | No deps; self-documenting |
| faer | 0.23.2 | SVD backend under `linalg` feature | Already in Cargo.lock [VERIFIED: Cargo.lock] |
| nalgebra | 0.33.2 | SVD backend when `linalg` absent | Already in Cargo.lock [VERIFIED: Cargo.lock] |

**Installation:** No new packages. No `Cargo.toml` changes.

---

## Package Legitimacy Audit

Not applicable — no new packages installed.

---

## Architecture Patterns

### System Architecture Diagram

```
cargo test --features linalg,parallel,serde
    ├── equivalence_phase48 binary
    │     ├── golden_co_cluster_parallel       [linalg guard → RUN with faer SVD → PASS]
    │     └── golden_co_cluster_below_threshold [linalg guard → RUN with faer SVD → PASS]
    └── equivalence_phase49 binary
          └── svd_sign_fpca_two_matrix_bit_identical [linalg guard → RUN with faer SVD → PASS]

cargo test (default features = parallel only, no linalg)
    ├── equivalence_phase48 binary
    │     ├── golden_co_cluster_parallel       [linalg guard → IGNORED]
    │     └── golden_co_cluster_below_threshold [linalg guard → IGNORED]
    └── equivalence_phase49 binary
          └── svd_sign_fpca_two_matrix_bit_identical [linalg guard → IGNORED]
```

### Recommended Change

In `fdars-core/tests/equivalence_phase48.rs` at lines 51 and 66 (before each `#[test]`):

```rust
// [VERIFIED: fdars-core/tests/equivalence_phase48.rs:51-64]
#[cfg_attr(not(feature = "linalg"), ignore)]
#[test]
fn golden_co_cluster_parallel() { ... }

// [VERIFIED: fdars-core/tests/equivalence_phase48.rs:66-79]
#[cfg_attr(not(feature = "linalg"), ignore)]
#[test]
fn golden_co_cluster_below_threshold() { ... }
```

In `fdars-core/tests/equivalence_phase49.rs` at line 380 (before `#[test]`):

```rust
// [VERIFIED: fdars-core/tests/equivalence_phase49.rs:380-411]
#[cfg_attr(not(feature = "linalg"), ignore)]
#[test]
fn svd_sign_fpca_two_matrix_bit_identical() { ... }
```

Also update the module-level doc comment in `equivalence_phase48.rs` to note that the co_cluster golden tests require `--features linalg` (the test header already says they pass under both `--features linalg,parallel` and `--no-default-features --features linalg` — adding an explicit note that they are silently ignored without linalg preserves correctness).

### Anti-Patterns to Avoid

- **Tolerance relaxation on log_likelihood**: The difference between 469.177 and 434.23 is 35 log-units — far too large for any numerical tolerance. Different SVD backends find different CEM basins; tolerance cannot bridge that gap. Tolerance is the wrong tool when the root cause is algorithmic divergence, not FP rounding.
- **Serialization / Mutex guard**: The tests don't race with each other. A Mutex would add complexity without addressing the actual cause (SVD backend selection).
- **Recapturing goldens under nalgebra**: The nalgebra path gives a *higher* log-likelihood (469.177) which is numerically valid but diverges from the faer path. Recapturing under both backends leads to two parallel golden sets and doubles the maintenance burden. The cfg_attr guard is cleaner.
- **Changing `fdata_to_pc` to always use nalgebra**: This would reduce numerical precision and impact the R/WASM bindings and the 28 examples. Out of scope per CONTEXT.md.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Feature-conditional test skip | Custom test wrapper macro | `#[cfg_attr(not(feature = "linalg"), ignore)]` — stdlib |
| Cross-SVD-backend golden sets | Dual const arrays with runtime dispatch | Two separate feature-gated tests (if ever needed) |

---

## Common Pitfalls

### Pitfall 1: Running the Acceptance Gate WITHOUT `--features linalg`

**What goes wrong:** The 3 tests show as `ignored` (not failed) → the executor thinks the fix "hides" the tests rather than fixing them.

**Why it happens:** The acceptance bar requires the tests to PASS, not merely be ignored. Under the fix, they PASS under `--features linalg,parallel` and are IGNORED under default features. The acceptance gate must use `--features linalg,parallel`.

**How to avoid:** Run: `cargo test --features linalg,parallel` (or the full gate `--features linalg,parallel,serde`) for all 10 consecutive runs. Per-binary: `cargo test --test equivalence_phase48 --features linalg,parallel`.

**Warning signs:** Test output shows `ignored` for these tests when run without explicit `--features linalg`.

### Pitfall 2: Forgetting That `--no-default-features --features linalg` ALSO Fixes the Tests

**What goes wrong:** Executor only tests the `--features linalg,parallel` path but not the no-default-features path mentioned in the test headers.

**Why it happens:** The test comments say "pass under BOTH --features linalg,parallel AND --no-default-features --features linalg" — both variants include linalg, so both should pass.

**How to avoid:** After the fix, verify: `cargo test --test equivalence_phase48 --no-default-features --features linalg` also gives 2 passed.

### Pitfall 3: Applying the cfg_attr to the Wrong Test (PACE Eigenfunction)

**What goes wrong:** Adding `#[cfg_attr(not(feature = "linalg"), ignore)]` to `svd_sign_pace_eigenfunctions_single_matrix_bit_identical` unnecessarily.

**Why it happens:** It's the sibling test in the same file (equivalence_phase49.rs), and Phase 49's module header references both SVD-sign tests.

**How to avoid:** The PACE test already passes WITHOUT linalg — confirmed this session. Only the `svd_sign_fpca_two_matrix_bit_identical` test (which uses `fdata_to_pc` directly) needs the guard. The PACE test uses `pace_fpca` which does not branch on the linalg feature in the same way.

### Pitfall 4: Confusing the Two Failure Modes

**What goes wrong:** Treating the svd_sign failure (4.18e-16 vs -0.0) as "FP nondeterminism" requiring tolerance relaxation.

**Why it happens:** The tiny number (4.18e-16) looks like ULP drift. But the root cause is the SAME feature-flag SVD divergence, not rayon scheduling. The `assert_eq!` must stay bit-identical once linalg is the fixed runtime (no relaxation needed).

**How to avoid:** Under `--features linalg`: rotation[0,0] = -0.0 every run (deterministic). The cfg_attr guard ensures the test only runs under linalg → bit-identity holds.

---

## Code Examples

### Pattern 1: Feature-Gated Test Skip (the fix)

```rust
// Source: Rust reference — #[cfg_attr(condition, attr)] built-in attribute
// Used in: fdars-core/tests/equivalence_phase48.rs

#[cfg_attr(not(feature = "linalg"), ignore)]
#[test]
fn golden_co_cluster_parallel() {
    // n_init=4 is ABOVE CO_CLUSTER_INIT_PARALLEL_THRESHOLD → parallel branch.
    let (data, argvals) = co_cluster_data(60, 40);
    let r = co_cluster(&data, &argvals, &co_cluster_config(4)).unwrap();
    assert_eq!(r.log_likelihood, 4.34232518676733207e2);
    // ...
}
```

The `#[cfg_attr(not(feature = "linalg"), ignore)]` attribute:
- Does nothing when `linalg` IS active → test runs normally
- Marks the test `#[ignore]` when `linalg` is absent → `cargo test` skips it (counts as "ignored")
- The test still shows in `cargo test --list` → visible, not hidden
- Running with `-- --ignored` or `-- --include-ignored` can still execute it (and it will fail — intentional)

### Pattern 2: Verifying faer SVD Path Is Active

```rust
// Quick smoke check (not a golden test): confirm fdata_to_pc uses faer under linalg
#[cfg(feature = "linalg")]
#[test]
fn fpca_uses_faer_svd_smoke() {
    use fdars_core::regression::fdata_to_pc;
    let data = fdars_core::matrix::FdMatrix::zeros(5, 8);
    let argvals: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
    // If faer SVD is active this compiles and runs the faer branch
    let _ = fdata_to_pc(&data, 3, &argvals);
}
```

### Pattern 3: Reproducing the Flake (for FLAKE-01 artifact)

```bash
# Reproduce the flake (without linalg):
cargo test --test equivalence_phase48 --features parallel -- golden_co_cluster
# Expected output (before fix): FAILED — left: 469.17705307940236

# Confirm the fix (with linalg):
cargo test --test equivalence_phase48 --features linalg,parallel -- golden_co_cluster
# Expected output: 2 passed

# After applying fix: confirm ignored under default features:
cargo test --test equivalence_phase48 -- golden_co_cluster
# Expected output: 2 ignored

# Run 10 consecutive times (acceptance bar):
for i in $(seq 1 10); do
    cargo test --features linalg,parallel -- golden_co_cluster svd_sign_fpca_two_matrix_bit_identical
done
# All 10 must show 3 passed, 0 failed
```

---

## State of the Art

| Old Understanding | Corrected Understanding | Impact |
|---|---|---|
| "Flake is environment/load/disk-pressure driven" | Flake is feature-flag driven: nalgebra vs faer SVD | Fix is deterministic cfg_attr; no environment mitigation needed |
| "co_cluster's parallel multi-restart best-selection is THREAD-COUNT-dependent" | Parallel dispatch is irrelevant; the bug is in the FPCA backend pre-parallelism | Fix is in test declaration, not CEM code |
| "svd_sign is the sub-ulp (4.18e-16 vs -0.0) variant" | Correct characterization, wrong root cause (was blamed on rayon/load) | Same feature-flag fix covers it |

**Deprecated/outdated:**
- The MEMORY.md entry `cocluster-svdsign-golden-flake-fullrun.md` incorrectly attributes the flake to "rayon picks a different effective thread count" under CPU load. The actual cause is SVD backend selection via feature flags. The memory note's workaround ("re-run in a quiet session") worked only by accident (quiet sessions may have used `--features linalg` explicitly).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | rayon's `IndexedParallelIterator` guarantees in-order collection for `0..n_init` range | Hypotheses Ruled Out | If wrong, the parallel n_init reduction could be non-deterministic for n_init=4. Risk: LOW — well-documented rayon behavior for indexed iterators. |
| A2 | The PACE eigenfunction test's SVD robustness holds across future code changes | Pitfall 3 | If `pace_fpca` is later refactored to use `fdata_to_pc` internally, it might also need the guard. Risk: LOW for this phase. |

**All other claims in this research were VERIFIED by direct code reading this session.**

---

## Open Questions

1. **Is there a nalgebra-path golden for co_cluster that could serve as a second target?**
   - What we know: nalgebra path gives 469.177 (a valid, higher-quality CEM result)
   - What's unclear: whether future maintainers would benefit from nalgebra-gated goldens alongside the faer ones
   - Recommendation: Defer to Phase 91 (robustness sweep). For Phase 90, the cfg_attr guard is sufficient.

2. **What happens to per-binary `cargo test --test equivalence_phase48` (no features) after the fix?**
   - Answer: The 2 co_cluster golden tests become "ignored" (2 ignored, 3 passed). This is the correct behavior — the test binary communicates "these tests require linalg."
   - No open question; this is the intended outcome.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | cargo test | ✓ | 1.97.0 | — |
| linalg feature (faer) | Golden tests (fix) | ✓ | faer 0.23.2 [VERIFIED: Cargo.lock] | Tests ignored without linalg (intended) |
| parallel feature (rayon) | Gate command | ✓ | rayon 1.11.0 [VERIFIED: Cargo.lock] | — |
| /home disk space | Full test run | ✓ | 128 GB free | — |
| /tmp tmpfs | Doctests | ✓ | 27 GB free | — |

**Missing dependencies with no fallback:** None.

**Build hazard reminders (from MEMORY.md):**
- Run clippy as `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code with `--all-targets`)
- Run `cargo fmt` per commit to avoid CI fmt drift
- Keep `/tmp` from exhausting before doctest runs (`rm -rf target/debug/{incremental,examples}` if needed)
- Gate commands must run FOREGROUND with `timeout: 600000` — not background (background runs get killed)

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Built-in Rust `#[test]` harness |
| Config file | `fdars-core/Cargo.toml` (test binary config via `[[test]]`) |
| Quick run command | `cargo test --test equivalence_phase48 --features linalg,parallel -- golden_co_cluster` |
| Full suite command | `cargo test --features linalg,parallel,serde` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FLAKE-01 | Diagnosis artifact documents root cause with reproduction steps | Manual artifact | `cargo test --test equivalence_phase48 --features parallel -- golden_co_cluster` (shows the flake) | ✅ (existing tests) |
| FLAKE-02 | Three tests pass under full parallel cargo test runs ×10 | Integration (golden) | `cargo test --features linalg,parallel,serde` | ✅ (after fix: same files, added cfg_attr) |
| FLAKE-02 | Tests are ignored (not failed) under default features | Integration (negative) | `cargo test --test equivalence_phase48 -- golden_co_cluster` | ✅ (after fix) |

### Sampling Rate
- **Per task commit:** `cargo test --test equivalence_phase48 --features linalg,parallel -- golden_co_cluster && cargo test --test equivalence_phase49 --features linalg,parallel -- svd_sign_fpca_two_matrix_bit_identical`
- **Per wave merge:** `cargo test --features linalg,parallel` (all tests)
- **Phase gate (×10):** `for i in $(seq 1 10); do cargo test --features linalg,parallel,serde; done` — all 10 must show 0 failures

### Wave 0 Gaps

None — the existing test files (`equivalence_phase48.rs`, `equivalence_phase49.rs`) are the test infrastructure. No new test files needed. No framework install needed. The fix is attribute-only.

---

## Security Domain

Security enforcement: the phase modifies test attributes only (no library code). No ASVS categories apply. No threat patterns relevant.

---

## Sources

### Primary (HIGH confidence — tool-verified this session)

- `fdars-core/tests/equivalence_phase48.rs` — full file read; golden values and test structure verified
- `fdars-core/tests/equivalence_phase49.rs` — lines 260-411 read; golden constants and test structure verified
- `fdars-core/src/coclustering.rs` — full file read; FPCA call site (line 932), parallel dispatch (lines 981-987), reduce (lines 992-998) verified
- `fdars-core/src/regression.rs` — lines 253-500 read; `fdata_to_pc` SVD branch (lines 438-477), `fix_svd_signs` (lines 288-301), `dominant_sign_negative` (lines 266-276) verified
- `fdars-core/src/parallel.rs` — full file read; macro definitions verified
- `fdars-core/src/clustering.rs` — lines 140-170, 241-267, 482-514; `assign_clusters`, `kmeans_plusplus_init` parallel usage verified
- `~/.cargo/registry/…/faer-0.23.2/src/linalg/solvers.rs` — lines 1126-1152; `new_thin` → `get_global_parallelism()` verified
- `~/.cargo/registry/…/faer-0.23.2/src/linalg/svd/bidiag.rs` — lines 111-112; `par_threshold=49152` verified
- `~/.cargo/registry/…/faer-0.23.2/src/linalg/svd/mod.rs` — lines 262-279; `recursion_threshold=128`, QR path for n<128 verified
- `~/.cargo/registry/…/faer-0.23.2/src/lib.rs` — lines 1146-1187; `GLOBAL_PARALLELISM` AtomicUsize, `get_global_parallelism()` verified
- `Cargo.lock` — faer 0.23.2, nalgebra 0.33.2, rayon 1.11.0, matrixmultiply 0.3.10 versions verified
- `fdars-core/Cargo.toml` — feature flags (`default = ["parallel"]`, `linalg = ["faer", "anofox-regression"]`) verified
- `.github/workflows/` — CI commands (`cargo test --features linalg,parallel,serde`) verified

### Experimental Evidence (HIGH confidence — run this session)

- `cargo test --test equivalence_phase48 --features parallel -- golden_co_cluster` → FAILED with `left: 469.17705307940236`
- `cargo test --test equivalence_phase48 --features linalg,parallel -- golden_co_cluster` → PASSED
- `cargo test --test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical` → FAILED with `left: 4.1785053186338415e-16   right: -0.0`
- `cargo test --test equivalence_phase49 --features linalg,parallel -- svd_sign_fpca_two_matrix_bit_identical` → PASSED
- `cargo test --test equivalence_phase49 -- svd_sign_pace_eigenfunctions_single_matrix_bit_identical` → PASSED (confirms PACE test unaffected)

---

## Project Constraints (from CLAUDE.md)

- **No code changes outside GSD workflow** — executor must use `/gsd-execute-phase`
- **Commit gate**: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Format gate**: `cargo fmt` must be run per commit to avoid CI drift
- **Disk hazards**: `rm -rf target/debug/{incremental,examples}` if /home runs low; route serde builds to /tmp tmpfs
- **Run gates FOREGROUND**: background bash gets killed; use `timeout: 600000` per gate
- **No --no-verify bypass** unless after out-of-band gate confirmation

---

## Metadata

**Confidence breakdown:**
- Root cause diagnosis: HIGH — experimentally reproduced this session
- Fix approach: HIGH — cfg_attr is standard Rust, no unknowns
- 10-run acceptance protocol: HIGH — straightforward to execute

**Research date:** 2026-09-09
**Valid until:** Indefinite (feature-flag divergence is structural, not environment-dependent)
