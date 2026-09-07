# Phase 82: Public-Surface Sealing & Non-Exhaustive Coverage - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning
**Mode:** Scope-bound (decisions pre-resolved at the Phase 81 AUDIT-01 approval gate — no separate discuss needed)

<domain>
## Phase Boundary

Apply the approved Scope-B (accidental `pub` sealing → API-02) and Scope-C (`#[non_exhaustive]` coverage → API-03) entries from the Phase 81 audit inventory. Visibility/attribute-only changes — NO behavior or numeric change. Exact change set is fixed by the user-approved inventory; deferred entries are explicitly out of scope for this phase.

</domain>

<decisions>
## Implementation Decisions

### Approved change set (from `81-AUDIT-INVENTORY.md` ## Approval, dated 2026-09-07)

**API-02 — accidental `pub` sealing (Scope B):**
- **`AUD-07`**: `src/helpers.rs:10` `pub fn sort_nan_safe(&mut [f64])` → `pub(crate)`. Internal-only; not re-exported; 0 external/test/example/doctest uses of the public path. Source-invisible to users.
- **`AUD-08`**: `src/smoothing.rs:336` `pub fn solve_gaussian_pub(...)` → `pub(crate)`. Internal cross-module solver wrapper; not re-exported. Consider also dropping the misleading `_pub` suffix (rename → `solve_gaussian`) since the suffix was the accidental-widening signal — do this only if it stays clean under the whole-crate gates.

**API-03 — `#[non_exhaustive]` coverage (Scope C):**
- **`AUD-10`**: add `#[non_exhaustive]` to the 10 public enums currently missing it: `PeerPenalty` (`src/peer.rs:68`), `LambdaChoice` (`src/peer.rs:99`), `LambdaMethod` (`src/peer.rs:117`), `DesignCriterion` (`src/optimal_design.rs:92`), `OptimalityKind` (`src/optimal_design.rs:103`), `ExtrapolationPolicy` (`src/helpers.rs:884`), `ImputationMethod` (`src/helpers.rs:1015`), `SelectionCriterion` (`src/scalar_on_function/mod.rs:268`), `BasisType` (`src/smooth_basis.rs:22`), `BasisCriterion` (`src/smooth_basis.rs:1117`). In-crate `match` sites on these are updated in the same phase (add `_ =>` arms only where the compiler now requires them; do NOT add catch-all arms that would hide future-variant warnings inside the defining crate unless the match is genuinely exhaustive-by-intent).
- **`AUD-11`**: add `#[non_exhaustive]` to the 2 public result structs missing it: `OptimBandwidthResult` (`src/smoothing.rs:544`), `KnnCvResult` (`src/smoothing.rs:767`). Users only read these (never construct) → pure forward-compat gain, no in-crate construction breakage.

### Explicitly OUT of scope (deferred to STAB-03 / Phase 84 per the approval gate)
- `AUD-09` / `AUD-13`: `wire` module stays **public** (deliberate JS/R interchange seam) — do NOT seal it.
- `AUD-12`: config-struct `#[non_exhaustive]` — deferred (needs builder/`Default` construction path; feature work, not this cleanup).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- The crate already has ~330 `#[non_exhaustive]` occurrences — this phase closes the last 12 gaps (10 enums + 2 result structs), consistent with the established convention.
- `pub(crate)` is already the norm for internal helpers (`linalg`, `distributions`, `permutation_test`, `test_helpers` are `pub(crate)`).

### Established Patterns
- `#[non_exhaustive]` sits directly above the type's `#[derive(...)]`/`pub enum|struct` line.
- Barrel `mod.rs` files use explicit `pub use`; check none of the 2 sealed fns (`sort_nan_safe`, `solve_gaussian_pub`) are re-exported (audit confirmed they are not).

### Integration Points
- Sealing `sort_nan_safe`/`solve_gaussian_pub` must not break their many internal callers across outliers/seasonal/classification/scalar_on_function/tolerance/spm/detrend/conformal/concurrent_regression/cv — `pub(crate)` keeps them all visible.
- Adding `#[non_exhaustive]` to enums can force `_ =>` arms on in-crate exhaustive matches — fix those in-phase so `cargo test` + clippy `-D warnings` stay green.

</code_context>

<specifics>
## Specific Ideas

- Whole-crate gates (must stay green): `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `cargo build --features serde`, all 28 examples compile.
- Build/CI hazards (MEMORY): pre-commit hook full-test-suite times out at 30s → `cargo fmt` then `git commit --no-verify`, run gates out-of-band; `/home` disk pressure (was 94% after cleanup) — `rm -rf target/debug/{incremental,examples}` if a spurious "No space left"/link failure appears.

</specifics>

<deferred>
## Deferred Ideas

- `AUD-09`/`AUD-12`/`AUD-13` and all optional-naming entries → STAB-03 1.0 gap checklist (Phase 84).

</deferred>
