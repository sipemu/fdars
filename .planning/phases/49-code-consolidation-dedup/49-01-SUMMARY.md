---
phase: 49-code-consolidation-dedup
plan: 01
subsystem: distributions
tags: [refactor, dedup, consolidation, chi-squared, gamma, special-functions, golden-equivalence, tracer, CONS-01]
requires:
  - phase: 46
    provides: PROF-02 dedup inventory (χ²/gamma ranked #1, HIGH leverage — 2 hand-rolled kernels)
  - phase: 47
    provides: golden-equivalence-then-migrate harness pattern
  - phase: 48
    provides: equivalence_phaseNN.rs bit-identical golden pattern (assert_eq!, both feature configs)
provides:
  - src/distributions.rs — the single crate-internal home for χ²/gamma special functions (pub(crate) ln_gamma, reg_gamma_p, reg_gamma_q, chi2_cdf, chi2_quantile, chi2_sf)
  - tests/equivalence_phase49.rs — shared Wave-0 golden harness for ALL Phase-49 plans (02-04 append to it)
  - "#[doc(hidden)] __equivalence_test_support forwarding module in lib.rs (test-only reach into pub(crate) surface; no public signature change)"
  - CONS-01 χ²/gamma tracer: share-primitives-split-tail-policy proven bit-identical (incl. far tail) under both feature configs
affects: [49-02, 49-03, 49-04]
actuals:
  tokens: 78000
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - "share-primitives-split-tail-policy: one ln_gamma + CDF-family P-direct reg_gamma_p, SF-family keeps its OWN private Q continued fraction (tiny=1e-300) to avoid the 1-P far-tail cliff"
    - "golden-equivalence capture-then-assert (assert_eq! bit-identical, NOT tolerance — code-motion)"
    - "#[doc(hidden)] forwarding fns (not pub use) to expose pub(crate) internals to integration tests without widening any real signature"
key-files:
  created:
    - fdars-core/src/distributions.rs
    - fdars-core/tests/equivalence_phase49.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/inference/dist.rs
    - fdars-core/src/spm/chi_squared.rs
    - fdars-core/src/spm/mod.rs
    - fdars-core/src/spm/bootstrap.rs
key-decisions:
  - "One kernel CANNOT serve both families bit-identically (RESEARCH-measured far-tail cliff, χ²-SF x=70.59 k=1 → 4.3974044505938783e-17 vs the 1-P route's 0.0). Shared ONLY the primitives; kept two tail-specialized wrappers."
  - "SF family (chi2_sf) uses its OWN private gamma_p_series_sf/gamma_q_cf_sf (tiny=1e-300, inline 1e-15, no underflow guard) extracted verbatim from inference/dist.rs — NOT routed through reg_gamma_p."
  - "CDF family (reg_gamma_p → chi2_cdf/chi2_quantile) extracted verbatim from spm/chi_squared.rs (tiny=1e-30, eps=1e-14, -700 underflow guard)."
  - "ln_gamma adopted the spm GUARDED reflection form (sin().abs().ln() + <1e-30→INFINITY). Dead-code for χ² args (a=k/2>0 never hits x<0.5); golden confirmed zero χ² value change. The two coefficient literals (...809_93 vs ...809_9) round to the same f64."
  - "Single chi2_sf(x, df: f64) entry serves BOTH chi_square_sf (k as f64) and chi_square_sf_df (real df) — both derive a=<df>/2.0 → identical f64 at integer df (RESEARCH A1 confirmed by golden)."
  - "Test reach: added #[doc(hidden)] __equivalence_test_support with forwarding fns (not pub use, which cannot escape pub(crate)) so no existing public signature is widened. Also widened spm::chi_squared module+fns pub(super)→pub(crate) (superset; internal callers unchanged)."
  - "spm/bootstrap.rs:378 erf_via_gamma migrated to distributions::reg_gamma_p (the third primitive consumer)."
patterns-established:
  - "Bit-identical golden (assert_eq!) captured from pre-refactor code; asserted against BOTH the current public tail path (survives migration) AND the new distributions::* surface directly; must hold under --features linalg,parallel AND --no-default-features --features linalg."
requirements-completed: [CONS-01 (χ²/gamma target)]
coverage:
  - id: D1
    description: "χ² SF/CDF/quantile + reg_gamma_p bit-identical before and after the refactor (incl. far tail x=70.59 k=1), both feature configs"
    requirement: CONS-01
    verification:
      - kind: integration
        ref: "cargo test --test equivalence_phase49 gamma under --features linalg,parallel AND --no-default-features --features linalg => 4/4 pass both configs (current + new SF/CDF families)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Two hand-rolled gamma kernels now share the ln_gamma primitive from distributions.rs; SF keeps Q-direct tail, CDF keeps P-direct tail"
    requirement: CONS-01
    verification:
      - kind: integration
        ref: "inference/dist.rs + spm/chi_squared.rs delegate to distributions::*; local gamma_p_series/gamma_q_cf/ln_gamma/gamma_series/gamma_cf/normal_quantile_approx removed (-358 LOC); full suite 2583 lib + integration + doctests green both configs"
        status: pass
    human_judgment: false
  - id: D3
    description: "spm/bootstrap.rs:378 regularized_gamma_p(0.5, x*x) migrated to distributions::reg_gamma_p (third primitive consumer)"
    requirement: CONS-01
    verification:
      - kind: integration
        ref: "erf_via_gamma now calls distributions::reg_gamma_p; spm bootstrap tests + doctests green both configs"
        status: pass
    human_judgment: false
  - id: D4
    description: "Wave-1 gate: full suite green both configs + clippy --all-targets clean; no public signature change; no new dependency"
    requirement: CONS-01
    verification:
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean"
        status: pass
      - kind: integration
        ref: "cargo test both feature configs => all green (2583 lib each)"
        status: pass
    human_judgment: false
status: complete
---

# Phase 49 Plan 01: χ²/Gamma Consolidation (CONS-01 Tracer) Summary

Consolidated the two independent hand-rolled χ²/regularized-incomplete-gamma kernels
(`inference/dist.rs` SF-oriented + `spm/chi_squared.rs` CDF-oriented) into a new
`src/distributions.rs` that **shares only the primitives** (`ln_gamma` + series + continued
fraction) and **keeps two tail-specialized wrappers**, so every existing call site stays
BIT-IDENTICAL — including the far-tail linchpin χ²-SF at x=70.59, k=1 → `4.3974044505938783e-17`.
This is the CONS-01 numerical target and the Phase-49 tracer: it proves the
share-primitives-split-tail-policy pattern end-to-end (scaffold golden harness → capture goldens
from current code → build shared module → migrate call sites → assert bit-identical under both
feature configs) before the expansion plans build out from it.

## What shipped

- **`src/distributions.rs`** (new, `pub(crate)`): `ln_gamma` (shared Lanczos g=7,n=9, guarded
  reflection form), `reg_gamma_p`/`reg_gamma_q`/`chi2_cdf`/`chi2_quantile` (CDF family, P-direct,
  verbatim from spm), and `chi2_sf` (SF family, with its OWN private `gamma_p_series_sf`/
  `gamma_q_cf_sf` Q-direct continued fraction, verbatim from inference).
- **`tests/equivalence_phase49.rs`** (new): the shared Wave-0 golden harness. χ²/gamma
  SF/CDF/quantile + `reg_gamma_p` goldens captured BIT-IDENTICALLY (`assert_eq!`) from the CURRENT
  pre-refactor code, asserted against both the current public tail path and the new
  `distributions::*` surface. Later Phase-49 plans (02-04) append their goldens here.
- **Migrated three consumers**: `inference/dist.rs` (`chi_square_sf`/`chi_square_sf_df` →
  `distributions::chi2_sf`; `f_sf`'s betai/betacf repointed to `distributions::ln_gamma`);
  `spm/chi_squared.rs` (`regularized_gamma_p`/`chi2_cdf`/`chi2_quantile` delegate to
  `distributions::*`, `pub(crate)` names preserved); `spm/bootstrap.rs:378` (`erf_via_gamma` →
  `distributions::reg_gamma_p`). Removed the duplicated local kernels (net **-358 LOC** of numerics).
- **`lib.rs`**: `pub(crate) mod distributions;` + a `#[doc(hidden)] __equivalence_test_support`
  forwarding module (test-only reach into `pub(crate)` internals; no real signature widened).

## Commit count

3 atomic commits (one per task):
- `0aeb45a8` test(49-01): scaffold equivalence_phase49 + capture χ²/gamma goldens (tracer RED)
- `25d8a291` feat(49-01): add src/distributions.rs — shared ln_gamma + 2 tail wrappers (GREEN)
- `3090e6d5` refactor(49-01): migrate χ²/gamma call sites onto distributions.rs

## Golden results (both feature configs)

| Config | `equivalence_phase49 gamma` | Full suite (lib) | clippy --all-targets |
|--------|-----------------------------|------------------|----------------------|
| `--features linalg,parallel` | 4/4 pass | 2583 pass, 0 fail | clean |
| `--no-default-features --features linalg` | 4/4 pass | 2583 pass, 0 fail | clean |

Far-tail linchpin held: `chi2_sf(70.59, 1.0)` and the current `chi_square_sf(70.59, 1)` both equal
`4.3974044505938783e-17` (the SF-private Q continued fraction was preserved — NOT routed through the
CDF family's `1 - P` path, which floors to `0.0`).

## Deviations from Plan

None material — the plan's split-tail-policy design was followed exactly (RESEARCH had already
proven one kernel diverges, so the wrapper split was the planned path, not a deviation).

Minor enabling adjustments, all within the plan's "Claude's Discretion on exact mechanics":
- **[Rule 3 - Blocking] Test-reach mechanism.** `pub use crate::distributions;` cannot re-export
  `pub(crate)` items outside the crate (E0365/E0364), so integration tests could not reach the
  primitives. Resolved with `#[doc(hidden)]` forwarding **fns** (thin `pub fn` wrappers that call
  the `pub(crate)` items) rather than `pub use`. No existing public signature is widened; the module
  is `#[doc(hidden)]` and carries no stability guarantee.
- **[Rule 3 - Blocking] Visibility.** Widened `spm::chi_squared` module + its three fns from
  `pub(super)` to `pub(crate)` (a superset — spm-internal callers and `spm/tests.rs` unchanged) so
  the shared home and the test-support forwarders can reference them. The plan (Task 3) anticipated
  keeping the `pub(super)` names/signatures; `pub(crate)` preserves those callers while enabling the
  crate-level reach.
- `chi_squared::ln_gamma` was fully removed (not left as a delegator) after migration made it unused
  in non-test builds; the one spm test that exercised it now calls `crate::distributions::ln_gamma`.

## Known Stubs

None. All migrated paths call real consolidated implementations; no placeholders or empty returns.

## Threat Flags

None. Internal numerical refactor — no new network endpoint, auth path, file access, or schema at a
trust boundary. Threat T-49-01 (numerical output tampering) was mitigated exactly as the register
prescribed: `assert_eq!` bit-identical goldens (incl. far tail) under both feature configs,
before and after migration.

## Self-Check: PASSED

- `fdars-core/src/distributions.rs` — FOUND
- `fdars-core/tests/equivalence_phase49.rs` — FOUND
- Commit `0aeb45a8` — FOUND
- Commit `25d8a291` — FOUND
- Commit `3090e6d5` — FOUND
