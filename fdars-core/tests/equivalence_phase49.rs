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

use fdars_core::__equivalence_test_support::{current, distributions};
use fdars_core::matrix::FdMatrix;

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

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Goldens asserted against the NEW consolidated `distributions::*` surface. These prove the shared
// module reproduces the exact pre-refactor bits directly (independent of the call-site migration).
// ════════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_chi2_sf_family_new_bit_identical() {
    use distributions::chi2_sf;
    assert_eq!(chi2_sf(0.1, 1.0), SF_K1_X0_1);
    assert_eq!(chi2_sf(3.84, 1.0), SF_K1_X3_84);
    assert_eq!(chi2_sf(70.59, 1.0), SF_K1_X70_59); // FAR TAIL via the SF-private Q continued fraction
    assert_eq!(chi2_sf(4.0, 2.0), SF_K2_X4);
    assert_eq!(chi2_sf(5.99, 2.0), SF_K2_X5_99);
    assert_eq!(chi2_sf(0.1, 3.0), SF_K3_X0_1);
    assert_eq!(chi2_sf(3.84, 3.0), SF_K3_X3_84);
    assert_eq!(chi2_sf(5.99, 10.0), SF_K10_X5_99);
    assert_eq!(chi2_sf(21.0, 10.0), SF_K10_X21);
    assert_eq!(chi2_sf(21.0, 20.0), SF_K20_X21);
    assert_eq!(chi2_sf(31.41, 20.0), SF_K20_X31_41);
    // Real-df path (single df:f64 entry serves both usize-k and real-df SF sites).
    assert_eq!(chi2_sf(5.99, 2.0), SFDF_DF2_X5_99);
    assert_eq!(chi2_sf(8.0, 3.7), SFDF_DF3_7_X8);
    assert_eq!(chi2_sf(2.0, 3.7), SFDF_DF3_7_X2);
}

#[test]
fn gamma_chi2_cdf_family_new_bit_identical() {
    use distributions::{chi2_cdf, chi2_quantile, reg_gamma_p, reg_gamma_q};
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

    assert_eq!(reg_gamma_p(0.5, 0.5 * 0.5), RGP_HALFNORM_X0_5);
    assert_eq!(reg_gamma_p(0.5, 1.0 * 1.0), RGP_HALFNORM_X1);
    assert_eq!(reg_gamma_p(0.5, 1.5 * 1.5), RGP_HALFNORM_X1_5);
    assert_eq!(reg_gamma_p(0.5, 2.0 * 2.0), RGP_HALFNORM_X2);
    assert_eq!(reg_gamma_p(0.5, 3.0 * 3.0), RGP_HALFNORM_X3);
    assert_eq!(reg_gamma_p(2.0, 1.0), RGP_A2_X1);
    assert_eq!(reg_gamma_p(50.0, 48.15), RGP_A50_X48_15);

    // reg_gamma_q is the exact complement (CDF family only).
    assert_eq!(reg_gamma_q(2.0, 1.0), 1.0 - RGP_A2_X1);
    assert_eq!(reg_gamma_q(50.0, 48.15), 1.0 - RGP_A50_X48_15);
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// SVD SIGN-FIX goldens (CONS-01, plan 49-02). The sign convention — "for each component k, make the
// largest-|·| entry positive" — is consolidated into ONE pub(crate) decision core in regression.rs.
// Two call sites gate their flips from it: fdata_to_pc_1d's fix_svd_signs (flips rotation AND scores
// in lockstep) and pace_fpca's eigendecompose_cov (flips eigenfunctions ONLY — no scores matrix at
// that point). These goldens are the exact f64 bits produced by the CURRENT (pre-refactor) code and
// MUST reproduce BIT-IDENTICALLY (assert_eq!) after the sign-decision core is extracted, under BOTH
// --features linalg,parallel AND --no-default-features --features linalg.
// ════════════════════════════════════════════════════════════════════════════════════════════════

// FPCA rotation (m=8 rows × ncomp=3 cols, column-major flatten). Column signs are set by the
// sign-decision core; the dominant |·| entry of each rotation column is positive.
const FPCA_ROTATION: [f64; 24] = [
    -0.0,
    0.3269209249205384,
    -0.8875273506317356,
    -1.0745506015092254,
    0.5552174854021699,
    1.5313934507848797,
    1.2032114837353158,
    0.7127196995738764,
    -9.857382031715978e-16,
    0.019189741069902917,
    -1.1219371034802588,
    0.9004762592858471,
    1.0748442149135518,
    -0.9025616139944942,
    1.44587539280986,
    1.7208858461357057,
    1.5263043145882805e-15,
    -0.4065526952335004,
    -0.5590584746937156,
    0.8832675032755642,
    2.194203356838404,
    0.8410604393825144,
    -1.146644256221298,
    -0.7516873993254113,
];

// FPCA scores (n=5 rows × ncomp=3 cols, column-major flatten). Flipped in lockstep with the rotation
// column whenever that column's dominant entry was negative.
const FPCA_SCORES: [f64; 15] = [
    -0.07764966330218807,
    -0.7044830972210234,
    0.2657050422841701,
    0.4371545850379199,
    0.07927313320112106,
    -0.021329399651274443,
    -0.1724470348473898,
    -0.0874128096622758,
    -0.3414733270182405,
    0.6226625711791806,
    0.3209396681056325,
    -0.08201044634433237,
    -0.5003813845693599,
    0.2217874336136012,
    0.0396647291944584,
];

// pace_fpca eigenfunctions (m=21 rows × ncomp=2 cols, column-major flatten). SINGLE-matrix flip: the
// eigenfunction column is negated when its dominant |·| entry is negative (there is NO scores matrix
// at that point — BLUP scores are computed later).
const PACE_NCOMP: usize = 2;
const PACE_EIGENFUNCTIONS: [f64; 42] = [
    0.1456965165550319,
    0.20291892412883983,
    0.27209662942964286,
    0.35325981177313526,
    0.4447048932415157,
    0.5428831831187857,
    0.6430762415817446,
    0.7407656766542221,
    0.8330558990416346,
    0.9193296326533805,
    1.0007255054019575,
    1.0787077298729817,
    1.1534914960321347,
    1.2231680791769577,
    1.2840052825067514,
    1.331720239207729,
    1.3629590663302416,
    1.3762165573842906,
    1.3719326674434202,
    1.3520102182490779,
    1.3191578863667774,
    1.7848574660932566,
    1.6816640706565742,
    1.5511755776397556,
    1.3961821636579554,
    1.2246684007487443,
    1.0500986882459402,
    0.8893993975774812,
    0.7583406233155535,
    0.6655385347790131,
    0.6074498547579806,
    0.5667367657626416,
    0.5155000266506697,
    0.4234074667452139,
    0.26839144107858015,
    0.045104428663088655,
    -0.23357400916887633,
    -0.5426177631772527,
    -0.8543766569626076,
    -1.146329153249015,
    -1.404555187230801,
    -1.6234975584071192,
];

/// Deterministic FPCA fixture (5 curves × 8 points) driving `fdata_to_pc_1d` — the two-matrix
/// (rotation + scores) sign-flip site.
fn fpca_sign_fixture() -> (fdars_core::matrix::FdMatrix, Vec<f64>) {
    use fdars_core::matrix::FdMatrix;
    let n = 5usize;
    let m = 8usize;
    let mut cm = Vec::with_capacity(n * m);
    for j in 0..m {
        for i in 0..n {
            let t = j as f64 / (m - 1) as f64;
            cm.push(((i as f64 + 1.0) * (t * 3.0)).sin() + 0.3 * (i as f64) * t);
        }
    }
    let data = FdMatrix::from_column_major(cm, n, m).unwrap();
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    (data, argvals)
}

/// Deterministic irregular fixture (mirrors src `small_irreg_data`) driving `pace_fpca` — the
/// eigenfunction-only (single-matrix) sign-flip site.
fn pace_sign_fixture() -> (fdars_core::IrregFdata, fdars_core::PaceFpcaConfig) {
    use fdars_core::{IrregFdata, PaceFpcaConfig};
    let argvals_list = vec![
        vec![0.1, 0.4, 0.7],
        vec![0.0, 0.3, 0.6, 0.9],
        vec![0.2, 0.5, 0.8],
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0.1, 0.5, 0.9],
        vec![0.0, 0.4, 0.8],
    ];
    let values_list: Vec<Vec<f64>> = argvals_list
        .iter()
        .enumerate()
        .map(|(i, ts)| {
            ts.iter()
                .map(|&t: &f64| (i as f64 + 1.0) * t.sin())
                .collect()
        })
        .collect();
    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);
    let pm = 21usize;
    let config = PaceFpcaConfig {
        ncomp: 2,
        bandwidth: 0.2,
        sigma2: 0.01,
        work_grid: (0..pm).map(|i| i as f64 / (pm - 1) as f64).collect(),
        alpha: 0.05,
    };
    (ifd, config)
}

#[test]
fn svd_sign_fpca_two_matrix_bit_identical() {
    use fdars_core::regression::fdata_to_pc_1d;
    let (data, argvals) = fpca_sign_fixture();
    let fpca = fdata_to_pc_1d(&data, 3, &argvals).unwrap();

    assert_eq!(fpca.rotation.shape(), (8, 3));
    assert_eq!(fpca.scores.shape(), (5, 3));

    // Rotation: every entry bit-identical (column signs fixed by the sign-decision core).
    let (rm, rk) = fpca.rotation.shape();
    for k in 0..rk {
        for j in 0..rm {
            assert_eq!(
                fpca.rotation[(j, k)],
                FPCA_ROTATION[j + k * rm],
                "rotation[({j},{k})] drifted"
            );
        }
    }
    // Scores: flipped in lockstep with the rotation column — bit-identical.
    let (sn, sk) = fpca.scores.shape();
    for k in 0..sk {
        for i in 0..sn {
            assert_eq!(
                fpca.scores[(i, k)],
                FPCA_SCORES[i + k * sn],
                "scores[({i},{k})] drifted"
            );
        }
    }
}

#[test]
fn svd_sign_pace_eigenfunctions_single_matrix_bit_identical() {
    use fdars_core::pace_fpca;
    let (ifd, config) = pace_sign_fixture();
    let res = pace_fpca(&ifd, &config).unwrap();

    assert_eq!(res.ncomp, PACE_NCOMP);
    assert_eq!(res.eigenfunctions.shape(), (21, PACE_NCOMP));

    // Eigenfunctions: single-matrix sign flip (no scores at that point) — bit-identical.
    let (em, ek) = res.eigenfunctions.shape();
    for k in 0..ek {
        for j in 0..em {
            assert_eq!(
                res.eigenfunctions[(j, k)],
                PACE_EIGENFUNCTIONS[j + k * em],
                "eigenfunctions[({j},{k})] drifted"
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// SEEDED-RNG CONSOLIDATION goldens (CONS-02, plan 49-03). The per-thread determinism contract
// `seed + k` is consolidated into ONE pub(crate) helper `helpers::seed_for_thread(seed, k)` whose body
// is EXACTLY `StdRng::seed_from_u64(seed.wrapping_add(k as u64))`. `wrapping_add` is bit-identical to
// the plain `seed + k as u64` form for all non-overflowing inputs, so every migrated thread-offset
// site keeps its EXACT RNG stream. The consts below are the first 8 u64 draws produced by the
// PRE-REFACTOR formula `StdRng::seed_from_u64(42 + k)` (captured verbatim; for seed=42, k∈{0,1,3} the
// wrapping and plain forms are trivially equal — 42+3 ≪ u64::MAX). This golden pins the contract on the
// helper ALONE (independent of the per-site downstream goldens), under BOTH feature configs.
// ════════════════════════════════════════════════════════════════════════════════════════════════

// seed_from_u64(42 + 0) — first 8 u64 draws.
const RNG_SEED42_K0: [u64; 8] = [
    9713269763989775522,
    10011513049433592189,
    11740708795755607249,
    7487565853151867058,
    633513173585076202,
    7654602743214997928,
    13603079691933612283,
    15665927001465599799,
];
// seed_from_u64(42 + 1) — first 8 u64 draws.
const RNG_SEED42_K1: [u64; 8] = [
    13888656601109899510,
    7748424868751293521,
    7281802212127063514,
    6945627241540686980,
    1527362893385383704,
    18389352529620154249,
    2885470914457776372,
    564464919976309943,
];
// seed_from_u64(42 + 3) — first 8 u64 draws.
const RNG_SEED42_K3: [u64; 8] = [
    5701288609795542878,
    963872696619426818,
    16485337332698311621,
    5460190459464948039,
    9785743222799402055,
    8506977482393017863,
    3039877466836775605,
    9172263138704748396,
];

#[test]
fn rng_stream_seed_for_thread_bit_identical() {
    use fdars_core::__equivalence_test_support::helpers::seed_for_thread_draws;

    // The helper's stream must equal the pre-refactor `seed_from_u64(seed + k)` draws EXACTLY.
    assert_eq!(seed_for_thread_draws(42, 0, 8), RNG_SEED42_K0.to_vec());
    assert_eq!(seed_for_thread_draws(42, 1, 8), RNG_SEED42_K1.to_vec());
    assert_eq!(seed_for_thread_draws(42, 3, 8), RNG_SEED42_K3.to_vec());
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// PERMUTATION-SCAFFOLD CONSOLIDATION golden (CONS-02, plan 49-04). `frechet_anova`'s PRIMARY per-perm
// loop is migrated onto the one authoritative `permutation_test::permutation_pvalue` helper (per-perm
// reseed, threshold-gated parallel, (1+n_ge)/(1+n_perm)). This golden is captured against the CURRENT
// (pre-migration) frechet_anova and duplicates the PERMANENT Phase-48 goldens
// (equivalence_phase48.rs::golden_frechet_anova_parallel + _below_threshold) as cheap in-phase
// insurance (RESEARCH A3): the migrated helper MUST reproduce these EXACT bits under BOTH
// --features linalg,parallel AND --no-default-features --features linalg. The draw-application
// contract (helper shuffles 0..n; closure gathers group_labels[perm_idx[i]]) makes the bit-identity
// PROVABLE — one Fisher–Yates on a length-n slice under the same per-perm seed ⇒ identical position-
// permutation ⇒ identical perm_labels. The n_perm=999 case exercises the FRECHET_ANOVA_PERM_PARALLEL_
// THRESHOLD=200 parallel path; the n_perm=50 case exercises the below-threshold sequential path.
// ════════════════════════════════════════════════════════════════════════════════════════════════

// Captured from CURRENT frechet_anova (identical to the Phase-48 goldens — same fixture, seed=42).
const FRECHET_STATISTIC: f64 = 1.17320834419366224e3;
const FRECHET_P_PERM_999: f64 = 1.00000000000000002e-3; // n_perm=999 → parallel branch
const FRECHET_P_PERM_50: f64 = 1.96078431372549017e-2; // n_perm=50 → sequential branch
const FRECHET_P_ASYMPTOTIC: f64 = 4.05441402554881990e-257;

/// Deterministic two-group **density** data (rows strictly positive — WassersteinDensitySpace),
/// mirroring the Phase-48 `two_group_densities` fixture EXACTLY so the goldens coincide.
fn frechet_two_group_densities(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
    let argvals: Vec<f64> = (0..m)
        .map(|j| -3.0 + 6.0 * j as f64 / (m - 1) as f64)
        .collect();
    let n = 2 * n_per;
    let mut data = vec![0.0; n * m];
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let g = if i < n_per { 0 } else { 1 };
        labels[i] = g;
        let mu = if g == 0 { -0.5 } else { 0.5 } + 0.2 * ((i as f64 * 1.7).sin());
        let sigma = 0.7 + 0.2 * ((i as f64 * 1.3).sin().abs());
        for j in 0..m {
            let z = (argvals[j] - mu) / sigma;
            data[i + j * n] = (-0.5 * z * z).exp() / sigma + 1e-6;
        }
    }
    (
        FdMatrix::from_column_major(data, n, m).unwrap(),
        argvals,
        labels,
    )
}

#[test]
fn frechet_anova_permutation_scaffold_bit_identical() {
    use fdars_core::frechet::frechet_anova;
    let (data, argvals, labels) = frechet_two_group_densities(12, 20);

    // Parallel branch (n_perm=999 >= FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200).
    let r_par = frechet_anova(&data, &argvals, &labels, 999, 42).unwrap();
    assert_eq!(r_par.statistic, FRECHET_STATISTIC);
    assert_eq!(r_par.p_value_permutation, FRECHET_P_PERM_999);
    assert_eq!(r_par.p_value_asymptotic, FRECHET_P_ASYMPTOTIC);

    // Sequential branch (n_perm=50 < threshold) — same statistic, below-threshold p-value.
    let r_seq = frechet_anova(&data, &argvals, &labels, 50, 42).unwrap();
    assert_eq!(r_seq.statistic, FRECHET_STATISTIC);
    assert_eq!(r_seq.p_value_permutation, FRECHET_P_PERM_50);
}
