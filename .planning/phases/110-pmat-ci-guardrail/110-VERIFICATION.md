---
phase: 110-pmat-ci-guardrail
verified: 2026-09-15T21:10:00Z
status: passed
score: 6/6 must-haves verified
covered_files:
  - .github/workflows/rust-ci.yml
  - .planning/REQUIREMENTS.md
  - .planning/phases/110-pmat-ci-guardrail/110-01-SUMMARY.md
  - .planning/phases/110-pmat-ci-guardrail/110-CONTEXT.md
covered_digest: "v1:sha256:ab9d64d5c3137cd5d1fe0759372decb0b6071c027685dbabe212b2ab201dee0c"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 110: pmat CI Guardrail — Verification Report

**Phase Goal:** Add a GitHub Actions CI job that installs pmat and runs the CURATED quality-gate against fdars-core, failing the build if violations return — mirroring the v0.43.0 determinism-guardrail — so quality can't silently regress.
**Verified:** 2026-09-15T21:10:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pmat-quality-gate` job exists in `.github/workflows/rust-ci.yml` | VERIFIED | Job present at lines 122–161; `name: pmat Quality Gate` |
| 2 | Job uses curated `--checks complexity,satd,dead-code,duplicates` (entropy excluded) | VERIFIED | Line 160: `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation`; entropy intentionally excluded per inline comment |
| 3 | Job installs pmat 3.18.0 (pinned version) | VERIFIED | Line 150: `cargo install pmat --version 3.18.0 --locked` with cache-hit guard |
| 4 | Job is `runs-on: ubuntu-latest` with `working-directory: fdars-core` | VERIFIED | YAML parsed: `runs-on: ubuntu-latest`, `defaults.run.working-directory: fdars-core` |
| 5 | `pmat-quality-gate` is in `publish.needs` (release-gated) | VERIFIED | `publish.needs: ['test', 'clippy', 'fmt', 'docs', 'wasm', 'determinism-guardrail', 'pmat-quality-gate']` — confirmed via Python YAML parse |
| 6 | Gate command exits 0 on the current tree (0 violations, guardrail green on introduction) | VERIFIED | `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` from `fdars-core/` → exit 0, "Quality Gate: PASSED", 0 violations |

**Score: 6/6 truths verified**

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/rust-ci.yml` | CI file containing `pmat-quality-gate` job | VERIFIED | Modified in commit `1cad28bb`; only file changed (plus 110-CONTEXT.md) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pmat-quality-gate` job | `publish` job | `publish.needs` array | WIRED | Confirmed in parsed YAML |
| Gate command | pmat binary | `cargo install pmat --version 3.18.0 --locked` | WIRED | Cache-guarded install step precedes gate run step |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Gate passes on current tree | `cd fdars-core && pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` | exit 0, "Quality Gate: PASSED", Total violations: 0 | PASS |
| YAML is syntactically valid | `python3 -c "import yaml; yaml.safe_load(open('.../rust-ci.yml'))"` | No exception, printed "YAML_VALID" | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| CI-01 | 110-01 | pmat quality-gate CI job added, failing build on violations, local repro documented | SATISFIED | Job exists, curated command present, `--fail-on-violation` flag, local repro documented in inline comment on line 158 |

### Anti-Patterns Found

None. No TBD / FIXME / XXX / TODO / PLACEHOLDER markers in `.github/workflows/rust-ci.yml`. pmat is confirmed absent from `fdars-core/Cargo.toml` and workspace root `Cargo.toml` — correctly installed as a CI-only tool.

### Human Verification Required

None. All six success criteria are deterministically verifiable from the CI YAML and local gate run.

---

## Verification Detail

### Check 1 — Job exists and is correctly structured

`pmat-quality-gate` job found at lines 122–161 of `.github/workflows/rust-ci.yml`:
- `runs-on: ubuntu-latest` (line 124)
- `working-directory: fdars-core` (via `defaults.run`)
- Steps: checkout → `dtolnay/rust-toolchain@stable` → `actions/cache@v4` (cache key includes `pmat-3.18.0` + `Cargo.lock` hash) → install pmat → run gate

### Check 2 — CURATED command (critical)

Gate step (line 159–160):
```
name: Run pmat quality gate (complexity, satd, dead-code, duplicates)
run: pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation
```
The `entropy` analyzer is NOT included. The inline comment at line 152 explicitly documents the rationale: "pmat 3.18.0's entropy sampler is non-deterministic (different violation count each run) and would make this gate flaky."

### Check 3 — Release-gated

`publish.needs` (line 270): `[test, clippy, fmt, docs, wasm, determinism-guardrail, pmat-quality-gate]`

### Check 4 — YAML valid

`python3 -c "import yaml; yaml.safe_load(open('.../rust-ci.yml'))"` completed without exception.

### Check 5 — Gate passes on current tree

```
Quality Gate: PASSED
Total violations: 0
EXIT_CODE=0
```
The guardrail is green on introduction — it will first fail on a push that reintroduces a complexity/SATD/dead-code/duplicate violation.

### Check 6 — No crate dependency added

`grep -i pmat fdars-core/Cargo.toml` → NOT_IN_CARGO
`grep -i pmat Cargo.toml` → NOT_IN_ROOT_CARGO

Commit `1cad28bb` shows only `.github/workflows/rust-ci.yml` and `.planning/phases/110-pmat-ci-guardrail/110-CONTEXT.md` were modified — no `Cargo.toml` touched.

---

_Verified: 2026-09-15T21:10:00Z_
_Verifier: Claude (gsd-verifier)_
