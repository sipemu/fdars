//! FDboost-style stability selection over boosting base-learners (REG-06-05).
//!
//! Wraps `boost_fosr` with a subsampling loop: B resamples of ⌊n/2⌋ rows (without
//! replacement), aggregating per-base-learner selection frequencies. Base-learners
//! with frequency ≥ π_thr are declared "stable". The PFER bound is reported as an
//! informational diagnostic.
//!
//! # Algorithm
//!
//! For each resample `b = 0..B`:
//! 1. Draw ⌊n/2⌋ distinct row indices without replacement, seeded by
//!    `seed.wrapping_add(b)` (per-replicate isolation → deterministic + parallel-safe).
//! 2. Fit `boost_fosr` on the subsample.
//! 3. Mark every base-learner that appears in the boosting path (`selected_learners`).
//!
//! Selection frequency `π̂[j] = (# resamples selecting j) / B`. The stable set is
//! `{ j : π̂[j] ≥ π_thr }`. The Meinshausen–Bühlmann per-family-error-rate bound is
//! `E[V] ≤ q² / ((2·π_thr − 1)·p)` where `q` is the mean per-subsample selection count.
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

use super::boost_fosr::boost_fosr;
use super::{BoostingConfig, StabilityConfig, StabilityResult};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Copy the given rows of `src` into a new (|indices| × ncols) matrix.
fn subsample_rows(src: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let ncols = src.ncols();
    let mut out = FdMatrix::zeros(indices.len(), ncols);
    for (dst_i, &src_i) in indices.iter().enumerate() {
        for j in 0..ncols {
            out[(dst_i, j)] = src[(src_i, j)];
        }
    }
    out
}

/// FDboost-style stability selection.
///
/// Runs `stab_config.n_resamples` subsamples of size ⌊n/2⌋ (without replacement),
/// fits `boost_fosr` on each subsample, and aggregates per-base-learner selection
/// frequencies across resamples.
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t).
/// * `predictors` — Scalar predictor matrix (n × p). One base-learner per column.
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `boost_config` — [`BoostingConfig`] for the inner `boost_fosr` calls.
/// * `stab_config` — [`StabilityConfig`] controlling resamples, threshold, and seed.
///
/// # Returns
///
/// [`StabilityResult`] with per-learner selection frequencies, the stable set at
/// `pi_thr`, the PFER bound, and the number of resamples used.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] on shape problems (including too-small subsamples);
/// [`FdarError::InvalidParameter`] on out-of-range config; propagates any
/// [`FdarError`] from the inner `boost_fosr` fits.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn stability_selection(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    boost_config: &BoostingConfig,
    stab_config: &StabilityConfig,
) -> Result<StabilityResult, FdarError> {
    let (n, m_t) = data.shape();
    let p = predictors.ncols();

    // ---- Validation --------------------------------------------------------
    if m_t == 0 || predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/predictors",
            expected: format!("m_t > 0 and predictors.nrows() == n (n={n})"),
            actual: format!("m_t={m_t}, predictors.nrows()={}", predictors.nrows()),
        });
    }
    if p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: "at least 1 predictor column".to_string(),
            actual: "0 columns".to_string(),
        });
    }
    if argvals.len() != m_t {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length == data.ncols() = {m_t}"),
            actual: format!("length = {}", argvals.len()),
        });
    }
    if stab_config.n_resamples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_resamples",
            message: "must be >= 1".to_string(),
        });
    }
    if !(stab_config.pi_thr > 0.5 && stab_config.pi_thr <= 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "pi_thr",
            message: format!("must be in (0.5, 1.0], got {}", stab_config.pi_thr),
        });
    }
    let half = n / 2;
    if half < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 6 so that ⌊n/2⌋ >= 3 (minimum for boost_fosr)".to_string(),
            actual: format!("n={n} → ⌊n/2⌋={half}"),
        });
    }

    let b_count = stab_config.n_resamples;

    // ---- Resample loop (deterministic per replicate, parallel-safe) --------
    let per_resample: Vec<Vec<bool>> = iter_maybe_parallel!(0..b_count)
        .map(|b| -> Result<Vec<bool>, FdarError> {
            let mut rng = StdRng::seed_from_u64(stab_config.seed.wrapping_add(b as u64));
            // Sample `half` distinct row indices without replacement via a partial
            // Fisher–Yates shuffle (first `half` slots hold the sample).
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..half {
                let j = rng.gen_range(i..n);
                idx.swap(i, j);
            }
            let sub = &idx[..half];
            let sub_data = subsample_rows(data, sub);
            let sub_pred = subsample_rows(predictors, sub);
            let fit = boost_fosr(&sub_data, &sub_pred, argvals, boost_config)?;
            let mut selected = vec![false; p];
            for &j in &fit.selected_learners {
                if j < p {
                    selected[j] = true;
                }
            }
            Ok(selected)
        })
        .collect::<Result<Vec<Vec<bool>>, FdarError>>()?;

    // ---- Aggregate ---------------------------------------------------------
    let mut counts = vec![0usize; p];
    let mut total_selected = 0usize; // Σ_b (#unique learners selected in resample b)
    for sel in &per_resample {
        for (j, &s) in sel.iter().enumerate() {
            if s {
                counts[j] += 1;
                total_selected += 1;
            }
        }
    }
    let selection_freq: Vec<f64> = counts.iter().map(|&c| c as f64 / b_count as f64).collect();
    let stable_set: Vec<usize> = (0..p)
        .filter(|&j| selection_freq[j] >= stab_config.pi_thr)
        .collect();

    // q = mean per-subsample selection count; PFER = q² / ((2·π_thr − 1)·p)
    let q = total_selected as f64 / b_count as f64;
    let pfer_bound = (q * q) / ((2.0 * stab_config.pi_thr - 1.0) * p as f64);

    Ok(StabilityResult {
        selection_freq,
        stable_set,
        pi_thr: stab_config.pi_thr,
        pfer_bound,
        n_resamples: b_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;

    fn default_boost() -> BoostingConfig {
        BoostingConfig {
            mstop: 5,
            nu: 0.3,
            nbasis: 8,
            order: 4,
            lfd_order: 2,
            lambda: 1.0,
            ncomp_x: 3,
            seed: 0,
        }
    }

    fn default_stab() -> StabilityConfig {
        StabilityConfig {
            n_resamples: 30,
            pi_thr: 0.6,
            seed: 20260824,
        }
    }

    /// Predictor 0 strongly drives Y(t) = x0·sin(π t); predictors 1..p are unrelated.
    fn make_signal_dataset(n: usize, m: usize, p: usize) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let mut pred = vec![0.0f64; n * p];
        for i in 0..n {
            // strong predictor
            pred[i] = -1.0 + 2.0 * i as f64 / (n - 1).max(1) as f64;
            // unrelated predictors (deterministic, uncorrelated with the signal)
            for j in 1..p {
                pred[i + j * n] = ((i as f64 * (1.7 + j as f64) + j as f64 * 0.9).sin()) * 0.8;
            }
        }
        let predictors = FdMatrix::from_column_major(pred.clone(), n, p).unwrap();

        let mut y = vec![0.0f64; n * m];
        for (t_idx, &tv) in argvals.iter().enumerate() {
            let beta = (PI * tv).sin();
            for i in 0..n {
                let x0 = pred[i];
                let noise = 0.01 * ((i as f64 * 1.23 + t_idx as f64 * 0.71).sin());
                y[i + t_idx * n] = x0 * beta + noise;
            }
        }
        (
            FdMatrix::from_column_major(y, n, m).unwrap(),
            predictors,
            argvals,
        )
    }

    #[test]
    fn stability_selects_strong_signal() {
        let (data, predictors, argvals) = make_signal_dataset(50, 15, 4);
        let result = stability_selection(
            &data,
            &predictors,
            &argvals,
            &default_boost(),
            &default_stab(),
        )
        .unwrap();
        assert_eq!(result.selection_freq.len(), 4);
        // The strong predictor is selected far more often than the unrelated ones.
        for j in 1..4 {
            assert!(
                result.selection_freq[0] > result.selection_freq[j],
                "strong predictor freq {} must exceed noise predictor {j} freq {}",
                result.selection_freq[0],
                result.selection_freq[j]
            );
        }
        assert!(
            result.stable_set.contains(&0),
            "strong predictor must be in the stable set (freq={})",
            result.selection_freq[0]
        );
    }

    #[test]
    fn stability_freqs_in_range() {
        let (data, predictors, argvals) = make_signal_dataset(40, 12, 3);
        let result = stability_selection(
            &data,
            &predictors,
            &argvals,
            &default_boost(),
            &default_stab(),
        )
        .unwrap();
        assert!(result
            .selection_freq
            .iter()
            .all(|&f| (0.0..=1.0).contains(&f)));
        assert!(result.pfer_bound.is_finite() && result.pfer_bound >= 0.0);
    }

    #[test]
    fn stability_is_deterministic_under_seed() {
        let (data, predictors, argvals) = make_signal_dataset(44, 10, 4);
        let r1 = stability_selection(
            &data,
            &predictors,
            &argvals,
            &default_boost(),
            &default_stab(),
        )
        .unwrap();
        let r2 = stability_selection(
            &data,
            &predictors,
            &argvals,
            &default_boost(),
            &default_stab(),
        )
        .unwrap();
        assert_eq!(r1.selection_freq, r2.selection_freq);
        assert_eq!(r1.stable_set, r2.stable_set);
        assert_eq!(r1.pfer_bound, r2.pfer_bound);
    }

    #[test]
    fn stability_errors_on_invalid_params() {
        let (data, predictors, argvals) = make_signal_dataset(30, 10, 3);
        let mut bad_pi = default_stab();
        bad_pi.pi_thr = 0.4; // must be > 0.5
        assert!(
            stability_selection(&data, &predictors, &argvals, &default_boost(), &bad_pi).is_err()
        );
        let mut bad_b = default_stab();
        bad_b.n_resamples = 0;
        assert!(
            stability_selection(&data, &predictors, &argvals, &default_boost(), &bad_b).is_err()
        );
    }

    #[test]
    fn stability_errors_on_tiny_n() {
        let (data, predictors, argvals) = make_signal_dataset(4, 8, 2);
        // ⌊4/2⌋ = 2 < 3 → error
        assert!(stability_selection(
            &data,
            &predictors,
            &argvals,
            &default_boost(),
            &default_stab()
        )
        .is_err());
    }
}
