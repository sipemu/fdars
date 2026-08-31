//! Permanent golden-equivalence tests for Phase 49 (Code Consolidation / Dedup).
//!
//! BEHAVIOR-PRESERVING consolidation phase: this is a CODE-MOTION refactor. Each golden reference is
//! the CURRENT (pre-refactor) f64 output of the target χ²/gamma function, captured as a `const` from
//! the code that shipped before `src/distributions.rs` existed. The consolidated code (shared
//! `ln_gamma` + primitives, two tail-specialized wrappers) must reproduce every value
//! **BIT-IDENTICALLY** — `assert_eq!`, NOT tolerance — because moving code without re-deriving the
//! arithmetic preserves exact bits. The suite must pass under BOTH `--features linalg,parallel` AND
//! `--no-default-features --features linalg`.
//!
//! Per RESEARCH: a single kernel CANNOT serve both χ² families bit-identically (measured far-tail
//! divergence, χ²-SF at x=70.59 k=1 → 4.397e-17 vs the 1−P route's 0.0). The consolidation therefore
//! shares only the PRIMITIVES and keeps two tail wrappers — the SF family keeps its Q-direct
//! continued fraction (tiny=1e-300), the CDF family keeps its P-direct path (tiny=1e-30, eps=1e-14,
//! −700 underflow guard). The far-tail golden below is the linchpin that proves the split was kept.
//!
//! This file is the shared Wave-0 golden harness for ALL Phase-49 plans (CONS-01/02); later plans
//! append their own goldens (SVD signs, frechet_anova, RNG stream) to it.
//!
//! Access: the `pub(crate)` primitives and the current `inference`/`spm` tail functions are reached
//! via the crate's `#[doc(hidden)] __equivalence_test_support` re-export surface (test-only; not part
//! of the public API).

#![allow(clippy::excessive_precision)]

use fdars_core::__equivalence_test_support::current;
// NOTE: `distributions::*` `_new` goldens are added in Plan 49-01 Task 2 (once src/distributions.rs
// exists). This Task-1 harness pins the pre-refactor bits via the CURRENT tail path only.

// ─── χ² survival function (SF family, inference/dist.rs) ────────────────────────────────────────
// Q-direct upper-tail path (tiny=1e-300). The x=70.59,k=1 point is the far-tail linchpin: the SF
// family MUST NOT be routed through the CDF family's 1−P path (which floors to 0.0 here).
const SF_K1_X0_1: f64 = 0.7518296340458491;
const SF_K1_X3_84: f64 = 0.05004352124870506;
const SF_K1_X70_59: f64 = 4.3974044505938783e-17; // FAR TAIL — split-tail-policy linchpin
const SF_K2_X4: f64 = 0.1353352832366128; // near-`a+1` boundary (a=1, xx=2=a+1)
const SF_K2_X5_99: f64 = 0.05003662708658632;
const SF_K3_X0_1: f64 = 0.9918374237318764;
const SF_K3_X3_84: f64 = 0.27926761711861037;
const SF_K10_X5_99: f64 = 0.816102700005434;
const SF_K10_X21: f64 = 0.021093565587437982;
const SF_K20_X21: f64 = 0.3971325993508147;
const SF_K20_X31_41: f64 = 0.05000523920231491;

// Real-df SF variant (Satterthwaite/Box non-integer df, chi_square_sf_df).
const SFDF_DF2_X5_99: f64 = 0.05003662708658632; // == SF_K2_X5_99 (integer-df equivalence, A1)
const SFDF_DF3_7_X8: f64 = 0.07590002055098717;
const SFDF_DF3_7_X2: f64 = 0.6920172053452117;

// F survival function (shares only ln_gamma; keeps its own incomplete-beta path).
const FSF_F4_9646_D1_D10: f64 = 0.05000005219291389;
const FSF_F2_7109_D5_D20: f64 = 0.049999375760946194;
const FSF_F4_5097_D3_D30: f64 = 0.010000383882289389;
const FSF_F3_4_D5_D20: f64 = 0.021974102900931207;
const FSF_F2_5_D4_D15: f64 = 0.08673499728031378;

// ─── χ² CDF + quantile (CDF family, spm/chi_squared.rs) ─────────────────────────────────────────
// P-direct path (tiny=1e-30, eps=1e-14, −700 underflow guard). chi2_quantile's Newton loop must land
// on the SAME x bits after consolidation.
const CDF_X1_386_K2: f64 = 0.5000000000000003;
const CDF_X5_991_K2: f64 = 0.9499999999999998;
const CDF_X0_1_K1: f64 = 0.24817036595415087;
const CDF_X3_84_K1: f64 = 0.9499564787512955;
const CDF_X10_K5: f64 = 0.9247647538534881;
const CDF_X21_K10: f64 = 0.978906434412562;

const QUANT_P0_5_K1: f64 = 0.4549364231195719;
const QUANT_P0_95_K1: f64 = 3.841458820692417;
const QUANT_P0_99_K1: f64 = 6.634896601021193;
const QUANT_P0_5_K5: f64 = 4.351460191088459;
const QUANT_P0_95_K5: f64 = 11.070497693516337;
const QUANT_P0_99_K5: f64 = 15.086272469388966;
const QUANT_P0_5_K10: f64 = 9.341817765591898;
const QUANT_P0_95_K10: f64 = 18.30703805327515;
const QUANT_P0_99_K10: f64 = 23.20925115895436;
const QUANT_P0_5_K20: f64 = 19.337429229428313;
const QUANT_P0_95_K20: f64 = 31.41043284422962;
const QUANT_P0_99_K20: f64 = 37.566234786625024;

// regularized_gamma_p(0.5, x*x) — the spm/bootstrap.rs half-normal erf-via-gamma consumer.
const RGP_HALFNORM_X0_5: f64 = 0.5204998778130469;
const RGP_HALFNORM_X1: f64 = 0.8427007929497153;
const RGP_HALFNORM_X1_5: f64 = 0.9661051464753111;
const RGP_HALFNORM_X2: f64 = 0.9953222650189527;
const RGP_HALFNORM_X3: f64 = 0.9999779095030014;

// Direct regularized_gamma_p at both-branch representative points (series x<a+1 and CF x>=a+1).
const RGP_A2_X1: f64 = 0.2642411176571152; // series branch
const RGP_A50_X48_15: f64 = 0.4138633369587336; // CF branch (worst-divergence point per RESEARCH)

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Goldens asserted against the CURRENT public tail path (chi_square_sf / chi2_cdf / …). Before Task 3
// these call the original kernels; after migration they call the consolidated ones — the bits must
// not move.
// ════════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_chi2_sf_family_current_bit_identical() {
    use current::{chi_square_sf, chi_square_sf_df, f_sf};
    assert_eq!(chi_square_sf(0.1, 1), SF_K1_X0_1);
    assert_eq!(chi_square_sf(3.84, 1), SF_K1_X3_84);
    assert_eq!(chi_square_sf(70.59, 1), SF_K1_X70_59);
    assert_eq!(chi_square_sf(4.0, 2), SF_K2_X4);
    assert_eq!(chi_square_sf(5.99, 2), SF_K2_X5_99);
    assert_eq!(chi_square_sf(0.1, 3), SF_K3_X0_1);
    assert_eq!(chi_square_sf(3.84, 3), SF_K3_X3_84);
    assert_eq!(chi_square_sf(5.99, 10), SF_K10_X5_99);
    assert_eq!(chi_square_sf(21.0, 10), SF_K10_X21);
    assert_eq!(chi_square_sf(21.0, 20), SF_K20_X21);
    assert_eq!(chi_square_sf(31.41, 20), SF_K20_X31_41);

    assert_eq!(chi_square_sf_df(5.99, 2.0), SFDF_DF2_X5_99);
    assert_eq!(chi_square_sf_df(8.0, 3.7), SFDF_DF3_7_X8);
    assert_eq!(chi_square_sf_df(2.0, 3.7), SFDF_DF3_7_X2);
    // Integer-df equivalence (RESEARCH assumption A1): chi_square_sf_df at integer df == chi_square_sf.
    assert_eq!(chi_square_sf_df(5.99, 2.0), chi_square_sf(5.99, 2));

    assert_eq!(f_sf(4.9646, 1.0, 10.0), FSF_F4_9646_D1_D10);
    assert_eq!(f_sf(2.7109, 5.0, 20.0), FSF_F2_7109_D5_D20);
    assert_eq!(f_sf(4.5097, 3.0, 30.0), FSF_F4_5097_D3_D30);
    assert_eq!(f_sf(3.4, 5.0, 20.0), FSF_F3_4_D5_D20);
    assert_eq!(f_sf(2.5, 4.0, 15.0), FSF_F2_5_D4_D15);
}

#[test]
fn gamma_chi2_cdf_family_current_bit_identical() {
    use current::{chi2_cdf, chi2_quantile, regularized_gamma_p};
    assert_eq!(chi2_cdf(1.3862943611198906, 2), CDF_X1_386_K2);
    assert_eq!(chi2_cdf(5.991464547107979, 2), CDF_X5_991_K2);
    assert_eq!(chi2_cdf(0.1, 1), CDF_X0_1_K1);
    assert_eq!(chi2_cdf(3.84, 1), CDF_X3_84_K1);
    assert_eq!(chi2_cdf(10.0, 5), CDF_X10_K5);
    assert_eq!(chi2_cdf(21.0, 10), CDF_X21_K10);
    assert_eq!(chi2_cdf(0.0, 1), 0.0);

    assert_eq!(chi2_quantile(0.5, 1), QUANT_P0_5_K1);
    assert_eq!(chi2_quantile(0.95, 1), QUANT_P0_95_K1);
    assert_eq!(chi2_quantile(0.99, 1), QUANT_P0_99_K1);
    assert_eq!(chi2_quantile(0.5, 5), QUANT_P0_5_K5);
    assert_eq!(chi2_quantile(0.95, 5), QUANT_P0_95_K5);
    assert_eq!(chi2_quantile(0.99, 5), QUANT_P0_99_K5);
    assert_eq!(chi2_quantile(0.5, 10), QUANT_P0_5_K10);
    assert_eq!(chi2_quantile(0.95, 10), QUANT_P0_95_K10);
    assert_eq!(chi2_quantile(0.99, 10), QUANT_P0_99_K10);
    assert_eq!(chi2_quantile(0.5, 20), QUANT_P0_5_K20);
    assert_eq!(chi2_quantile(0.95, 20), QUANT_P0_95_K20);
    assert_eq!(chi2_quantile(0.99, 20), QUANT_P0_99_K20);

    // bootstrap.rs half-normal erf-via-gamma: regularized_gamma_p(0.5, x*x).
    assert_eq!(regularized_gamma_p(0.5, 0.5 * 0.5), RGP_HALFNORM_X0_5);
    assert_eq!(regularized_gamma_p(0.5, 1.0 * 1.0), RGP_HALFNORM_X1);
    assert_eq!(regularized_gamma_p(0.5, 1.5 * 1.5), RGP_HALFNORM_X1_5);
    assert_eq!(regularized_gamma_p(0.5, 2.0 * 2.0), RGP_HALFNORM_X2);
    assert_eq!(regularized_gamma_p(0.5, 3.0 * 3.0), RGP_HALFNORM_X3);

    assert_eq!(regularized_gamma_p(2.0, 1.0), RGP_A2_X1);
    assert_eq!(regularized_gamma_p(50.0, 48.15), RGP_A50_X48_15);
}

// The NEW consolidated `distributions::*` `_new` goldens (asserting the shared module reproduces
// these same bits directly) are appended in Plan 49-01 Task 2, once `src/distributions.rs` exists.
