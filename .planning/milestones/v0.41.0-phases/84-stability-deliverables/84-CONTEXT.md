# Phase 84: Stability Deliverables - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning
**Mode:** Scope-bound (non-code documentation deliverables; content largely determined by prior phases + standard Rust conventions)

<domain>
## Phase Boundary

Produce the three NON-CODE stability deliverables in `documentation/` that will govern the eventual (future, separate) 1.0 cut: a semver/API-stability policy (STAB-01), a finalized+documented MSRV policy consistent with `Cargo.toml` (STAB-02), and a 1.0 gap checklist enumerating everything deferred out of this milestone (STAB-03). No `fdars-core/src` behavior/API change in this phase. `documentation/` is the TRACKED docs dir (`docs/` is gitignored — do NOT write there).

</domain>

<decisions>
## Implementation Decisions

### STAB-01 — Semver + API-stability policy (`documentation/`, new file e.g. `STABILITY.md` or `SEMVER-POLICY.md`)
- Define what "stable" means for `fdars-core`: the public API (re-exported items, public fn signatures, public types/enums/fields) is what semver governs; internal `pub(crate)` items and the `wire` module (currently a public-but-unwired interchange seam) are NOT part of the stability guarantee until 1.0.
- Deprecation process: how a breaking change is staged (`#[deprecated(since=...)]` → removal in a later release), noting that under 0.x breaking changes ship in minor bumps (as this v0.41.0 milestone did — the first breaking release after a long additive run).
- Breaking-change policy post-1.0: what constitutes breaking (signature/visibility/variant/field changes), the role of `#[non_exhaustive]` (now applied across public enums + result structs per Phase 82) in keeping additions non-breaking, and MSRV-bump policy (see STAB-02).
- Reference the crate conventions already in CLAUDE.md (all public types derive Debug/Clone/PartialEq; `#[must_use]`; column-major invariants) as part of the stability surface.

### STAB-02 — MSRV policy finalized + pinned + consistent
- Pin/confirm: **crate MSRV = 1.81** (`fdars-core/Cargo.toml` `rust-version = "1.81"`, set for CRAN Windows compat), **`linalg` feature MSRV = 1.84** (faer 0.23+ requirement).
- Document the MSRV policy in `documentation/` (e.g. in STABILITY.md or a dedicated section): the two-tier MSRV, why (CRAN Windows / faer), and the bump policy (MSRV bumps are a minor-version event, called out in CHANGELOG).
- **Consistency check:** ensure `Cargo.toml` (`rust-version`) and the docs agree; note the 1.84 `linalg` requirement is documented (it is only in CLAUDE.md today — surface it in the tracked docs). If any drift exists between Cargo.toml and docs, this phase resolves it (docs-side; Cargo.toml `rust-version` stays 1.81).

### STAB-03 — 1.0 gap checklist (`documentation/`, new file e.g. `ROADMAP-TO-1.0.md` or `1.0-CHECKLIST.md`)
Enumerate everything that remains before a real 1.0 cut, scoping the next milestone. MUST include the items DEFERRED during this milestone's audit + known backlog:
- **From the AUDIT-01 approval gate (deferred to STAB-03):**
  - `AUD-09` / `AUD-13`: the `wire` module — decide before 1.0 whether to wire it up (JS/R interchange) or seal it `pub(crate)`; it is currently public-but-unused.
  - `AUD-12`: add `#[non_exhaustive]` to the ~22 public config structs, paired with a builder/`Default`+`..default()` construction path (so it isn't user-hostile). Feature work, not cleanup.
  - `AUD-19` (geometric_median), `AUD-20` (hausdorff), `AUD-21` (functional_spatial), `AUD-22` (the large lone-`_1d`/`_2d` suffix batch across the crate + all 28 examples), `AUD-23` (`LpeerResult`→`LocalPeerResult`) — the optional naming not done in Phase 83.
- **Known test/quality debt:**
  - The co_cluster/svd_sign golden-test flake: `golden_co_cluster_parallel`, `golden_co_cluster_below_threshold` (equivalence_phase48), `svd_sign_fpca_two_matrix_bit_identical` (equivalence_phase49) fail under full parallel `cargo test` (env/BLAS/disk-pressure dependent) but pass per-binary — make these deterministic (tolerance vs bit-identity, or serialize) before 1.0.
- **Pre-existing deferred backlog (from PROJECT deferred items / MEMORY):**
  - `soft_dtw_barycenter` optimizer replacement (SDTW-O1) — note: the soft_dtw_backward zero-gradient bug itself was FIXED in v0.40.0 (CORR-01); this remaining item is the MM-step/global-optimizer improvement.
  - Differentiable-core expansion (DIF-F1/F2/F3: reverse-mode/VJP, broaden differentiable subset, generic f64 hot-path signatures).
- **The 1.0-CUT itself:** bump to 1.0.0 + declare the API stable once this checklist clears (governed by STAB-01).
- **`fdars-r` FdMatrix migration** (issue `fdars-j75`): external R wrapper migration — separate package, out of `fdars-core` scope but a 1.0-ecosystem gap.

### Doc placement + conventions
- All three deliverables live under `documentation/` (tracked). Prefer clear, discoverable filenames; cross-link from README if it lists docs. Match the existing `documentation/` doc style (see ARCHITECTURE.md, DEVELOPMENT.md, GETTING-STARTED.md).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing `documentation/` guides (ARCHITECTURE, CONFIGURATION, DEVELOPMENT, GETTING-STARTED, TESTING, *_references) establish tone/format to match.
- CHANGELOG.md already states "adheres to Semantic Versioning" — STAB-01 formalizes what that means for this crate.
- The approved `81-AUDIT-INVENTORY.md` ## Approval section is the source of truth for the deferred items STAB-03 must list.

### Established Patterns
- MSRV facts: `fdars-core/Cargo.toml:15 rust-version = "1.81"`; `linalg` needs 1.84 (currently only in CLAUDE.md).
- serde build is currently GREEN (the old shapelet/ClassifFit serde breakage was repaired in v0.40.0) — do NOT list it as an open gap.

### Integration Points
- STAB-02 must keep `Cargo.toml` `rust-version` (1.81) and the docs consistent — docs-side edits only (no Cargo.toml change unless drift is found).

</code_context>

<specifics>
## Specific Ideas

- This is a docs-only phase: no `cargo test`/behavior change required. Reasonable verification: docs exist with required sections, MSRV numbers match `Cargo.toml`, STAB-03 lists every deferred AUD entry + the golden flake + 1.0-CUT + fdars-r. A `cargo fmt --check`/quick `cargo build` sanity (no source touched) is sufficient — the heavy whole-crate gates belong to Phase 85.
- Commit hazard (MEMORY): pre-commit hook full-test-suite times out at 30s → docs commits use `git commit --no-verify`.

</specifics>

<deferred>
## Deferred Ideas

- Actually executing any of the STAB-03 checklist items — that is the FUTURE 1.0 milestone, not this phase. This phase only ENUMERATES them.

</deferred>
