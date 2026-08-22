//! Functional two-sample inference tests.
//!
//! This module provides fdars' first standalone functional-inference surface:
//! two-sample functional hypothesis tests built by reusing existing
//! permutation, Hotelling-T², and bootstrap-band machinery elsewhere in the
//! crate. It is additive and non-breaking — no existing public signature is
//! altered by this module.
//!
//! # R baselines
//!
//! The functions here mirror the table-stakes two-sample tests from the R FDA
//! ecosystem:
//!
//! * [`t_perm_test`] / [`f_perm_test`] — `fda::tperm.fd` / `fda::Fperm.fd`
//!   (permutation two-sample mean tests).
//! * [`two_sample_mean_test`] — the FPC-basis Hotelling-T² mean-equality test
//!   in the spirit of `fda.usc` mean/covariance equality tests.
//! * [`mean_scb`] / [`scb_two_sample_test`] — `SCBmeanfd`-style simultaneous
//!   confidence bands for the mean and the mean difference.
//!
//! # Conventions
//!
//! Permutation tests take an explicit deterministic `seed`
//! (`StdRng::seed_from_u64(seed)`) and default to `n_perm = 999` at the call
//! site. All public functions return `Result<_, FdarError>` and validate their
//! inputs at entry. Result structs derive `Debug, Clone, PartialEq` and are
//! serde-gated.

mod anova;
pub(crate) mod dist;
mod flm;
mod hotelling;
mod itp;
mod permutation;
mod scb;

pub use anova::oneway_anova_vstat;
pub use flm::{flm_f_test, flm_gof_test};
pub use hotelling::two_sample_mean_test;
pub use itp::{itp_flm, itp_one_pop, itp_two_pop, ItpResult};
pub use permutation::{f_perm_test, t_perm_test, DEFAULT_N_PERM};
pub use scb::{mean_scb, scb_two_sample_test};

/// Result of a functional two-sample hypothesis test.
///
/// Carries the observed test statistic and the associated p-value. For
/// permutation-based tests, `n_perm` records the number of permutations used;
/// for the non-permutation (asymptotic / SCB) paths it is `0`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TestResult {
    /// Observed test statistic.
    pub statistic: f64,
    /// P-value for the null hypothesis of equal group means.
    pub p_value: f64,
    /// Number of permutations used (0 for non-permutation paths).
    pub n_perm: usize,
}
