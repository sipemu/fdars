# Phase 80: Release Hardening & Ship v0.40.0 - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning

<domain>
## Phase Boundary

The crate is validated and release-ready: outstanding v0.39.0 phases (75/76/77) formally signed off, version/CHANGELOG/docs updated to v0.40.0, and all whole-crate gates green — so the operator's `git tag v0.40.0` → crates.io publish (via `release.yml`) is the ONLY remaining step. This phase PREPARES + VERIFIES release-readiness; it does NOT itself tag or publish. Must land last (validates + folds in Phase 78 + Phase 79).

</domain>

<decisions>
## Implementation Decisions

### REL-01 — Nyquist sign-off of phases 75/76/77 (lightweight)
- **Lightweight audit-and-sign-off**, NOT the full validate-phase test-generation flow. Phases 75/76/77 are all `verification: passed` and archived under `.planning/milestones/v0.39.0-phases/`; their `*-VALIDATION.md` are `status: draft, nyquist_compliant: false`.
- For each: confirm the existing tests cover the phase's requirements (read the VALIDATION.md's per-task map + the phase's tests), then flip frontmatter `status: draft → validated` and set `nyquist_compliant: true` where coverage holds. If a genuine coverage gap exists, RECORD it in the VALIDATION.md (and as a backlog item) rather than synthesizing new tests into an archived phase — no new tests into archived phases (scope guard).
- Editing archived-milestone artifacts is expected and fine for this sign-off.

### REL-02 — Version, CHANGELOG, docs
- **Version bump:** `fdars-core/Cargo.toml` `0.38.0 → 0.40.0` (skip the never-published 0.39.0). Root `Cargo.toml` is a virtual workspace (no version). Bump version strings in `README.md` (line ~83 `fdars-core = { version = "0.38" ... }` → `"0.40"`) and any other tracked version references. Keep MSRV/feature docs accurate.
- **CHANGELOG:** two separate entries above `[0.38.0]`:
  - `[0.39.0]` — forward-mode `Dual` AD core (Scalar trait + `soft_dtw_distance_generic`, differentiable elastic distance/FPCA scores, gradient-API composition demo — phases 75/76/77). This work was code-complete but never published; fold it in honestly.
  - `[0.40.0]` — CORR-01 soft_dtw endpoint-seed gradient fix + barycenter optimizer stabilization (inverse-curvature step), CORR-02 gradient-pass audit, BUILD-01 serde feature repair. Note the soft_dtw barycenter behavior change (it now actually converges).
  - Follow the existing keepachangelog-style format already in CHANGELOG.md.
- **Docs refresh (targeted, not comprehensive):** bump version strings; add forward-mode AD + the two fixes to README highlights / feature list where natural; refresh only stale spots in `documentation/` (tracked — NOT `docs/`, which is gitignored). No new guide pages (the AD features shipped in the folded-in 0.39.0 work; a mention suffices).

### Ship boundary (locked)
- The phase ends at "release-ready + all gates green + SUMMARY documents the tag/publish steps." The actual `git tag v0.40.0` push → crates.io publish via `release.yml` is the **operator's** final manual step, documented in the SUMMARY, gated on all prior phases green. Do NOT create the git tag or publish.

### Claude's Discretion
- Exact CHANGELOG wording, which `documentation/` files are stale, and the precise README highlight phrasing.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CHANGELOG.md` (keepachangelog style; latest `[0.38.0] - 2026-09-05`). `documentation/` holds tracked guides (ARCHITECTURE, CONFIGURATION, DEVELOPMENT, GETTING-STARTED, references). `README.md`.
- 75/76/77 archived at `.planning/milestones/v0.39.0-phases/{75,76,77}-*/` with `-VALIDATION.md` (draft) and `-VERIFICATION.md` (passed).

### Established Patterns
- Release flow (MEMORY.md): bump `fdars-core/Cargo.toml` → commit → `git tag vX.Y.Z` → push tag → `release.yml` runs `cargo publish`. This milestone gets a real tag (real code changes) — but tag/publish is operator-driven (this phase stops at readiness).
- `docs/` is gitignored; tracked docs live in `documentation/`.

### Integration Points
- CI gates: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel[,serde] -- -D warnings` (+ CI `-A` allows), `cargo test` (+ serde matrix). All must be green at phase end.

</code_context>

<specifics>
## Specific Ideas

- Phase 78 already changed `soft_dtw_barycenter` behavior (now converges via inverse-curvature step; SDTW-O1 backlogged for a proper L-BFGS optimizer) — the CHANGELOG must mention this behavior change.
- Phase 79 fixed the serde build (BUILD-01); the CI serde matrix is now green.
- Build hazards carry over: `--no-verify` commits (30s hook timeout), `cargo fmt` per commit, clippy `--all-targets`, watch `/tmp` + `target/` disk pressure.

</specifics>

<deferred>
## Deferred Ideas

- Actual `git tag v0.40.0` + crates.io publish — operator's manual step, out of phase scope.
- SDTW-O1 (proper soft-DTW barycenter optimizer) — already backlogged in Phase 78.
- `fdars-r` R-wrapper `FdMatrix` migration (issue fdars-j75) — not this milestone.

</deferred>
