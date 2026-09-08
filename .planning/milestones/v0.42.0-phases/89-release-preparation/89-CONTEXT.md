# Phase 89: Release Preparation & Verification - Context

**Gathered:** 2026-09-08
**Status:** Ready for planning
**Mode:** Auto-generated (release-prep phase — smart discuss skipped)

<domain>
## Phase Boundary

Non-code release preparation for v0.42.0 (REL-01). Deliverables: version bump, breaking-framed CHANGELOG, docs refresh, 1.0-checklist checkoff, all gates green + `cargo package`. The `git tag v0.42.0` push → crates.io publish is the DEFERRED operator step (NOT done here).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
- Bump `fdars-core/Cargo.toml` `version = "0.41.0"` → `"0.42.0"`.
- Add a `[0.42.0]` entry to BOTH `CHANGELOG.md` (root) and `fdars-core/CHANGELOG.md`, breaking-framed, explicitly calling out: sealed `wire` `pub(crate)` (SEAL-01); `#[non_exhaustive]` on 38 config structs + the changed construction idiom — external callers now use `Config::default()` + field assignment, NOT struct literals / `..Default::default()` (SEAL-02/03); Phase-87 consolidations (`geometric_median`, `hausdorff_self`/`_cross`, `functional_spatial`/`kernel_functional_spatial` onto `Dim`) + `LpeerResult`→`LocalPeerResult` (NAME-01/02/03/05); Phase-88 41 `Dim`-dispatch consolidations + 5 plain-renames (`fdata_to_pc`/`fdata_to_pls`/`fourier_fit`/`pspline_fit`/`select_basis_auto` drop `_1d`) (NAME-04).
- Check off `documentation/ROADMAP-TO-1.0.md` API items: AUD-09, AUD-12, AUD-13, AUD-19, AUD-20, AUD-21, AUD-22, AUD-23 (`[ ]` → `[x]`).
- Gate set: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `cargo build --features serde`, `cargo build --examples`, `cargo test --doc`, `cargo package -p fdars-core`.
- Do NOT `git tag` or publish.

Constraints: `fdars-core` only, no code/behavioral change (this phase is docs + version metadata).

</decisions>

<code_context>
## Existing Code Insights

- `fdars-core/Cargo.toml` version line (currently `0.41.0`).
- `CHANGELOG.md` (root) + `fdars-core/CHANGELOG.md` — both carry a `[0.41.0]` entry; the crate CHANGELOG is the shipped one (`documentation/` is where tracked guides live; `docs/` is gitignored per project memory).
- `documentation/ROADMAP-TO-1.0.md` — the 1.0 gap checklist; API section items AUD-09/12/13/19-23.
- Phase 86/87/88 SUMMARYs record the exact breaking changes to document.

</code_context>

<specifics>
## Specific Ideas

The CHANGELOG must be honest about the **construction-idiom change** for the 38 sealed configs (a subtle but real breaking change beyond "sealed configs"). `cargo package` must succeed (a stronger gate than `cargo build` — it checks the packaged tarball builds).

</specifics>

<deferred>
## Deferred Ideas

- `git tag v0.42.0` + crates.io publish — the operator's step, deliberately deferred.
- Remaining 1.0-checklist items (golden flake, SDTW-O1, differentiable core, fdars-r) — stay open, not this milestone.

</deferred>
