//! FDboost-style stability selection over boosting base-learners (REG-06-05).
//!
//! Wraps `boost_fosr` with a subsampling loop: B resamples of ⌊n/2⌋ rows (without
//! replacement), aggregating per-base-learner selection frequencies. Base-learners
//! with frequency ≥ π_thr are declared "stable". The PFER bound is reported as an
//! informational diagnostic.
//!
//! **Status:** Skeleton — implementation delivered in Plan 05.
//!
//! # References
//!
//! Meinshausen & Bühlmann (2010). Stability Selection. *JRSS-B*, 72(4).
//! Hofner et al. (2015). Controlling false discoveries in high-dimensional situations:
//! Boosting with stability selection. *The R Journal*, 7(1).
//!
//! # Divergences from stabs (R package)
//!
//! Uses subsampling ⌊n/2⌋ without replacement (Meinshausen-Bühlmann default).
//! Selection criterion: base-learner appears in `selected_learners` at any iteration.
//! Seeded per replicate for full reproducibility.

use super::{BoostingConfig, StabilityConfig, StabilityResult};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// FDboost-style stability selection.
///
/// Runs `config.n_resamples` subsamples of size ⌊n/2⌋ (without replacement),
/// fitting `boost_fosr` on each subsample, and aggregates selection frequencies
/// per base-learner across resamples.
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t).
/// * `predictors` — Scalar predictor matrix (n × p). One base-learner per column.
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `boost_config` — [`BoostingConfig`] for the inner `boost_fosr` calls.
/// * `stab_config` — [`StabilityConfig`] controlling resamples, threshold, and seed.
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] with `detail: "not yet implemented (Plan 05)"`.
/// Full implementation arrives in Plan 05.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn stability_selection(
    _data: &FdMatrix,
    _predictors: &FdMatrix,
    _argvals: &[f64],
    _boost_config: &BoostingConfig,
    _stab_config: &StabilityConfig,
) -> Result<StabilityResult, FdarError> {
    // filled in Plan 05
    Err(FdarError::ComputationFailed {
        operation: "stability_selection",
        detail: "not yet implemented (Plan 05)".to_string(),
    })
}
