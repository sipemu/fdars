//! Joint registration and split-conformal amplitude / phase prediction sets for
//! partially observed functional data.
//!
//! Implements the SFCP (amplitude) and SFCPP (phase) procedures of Wang, Kurtek &
//! Zhang, *"Joint Registration and Conformal Prediction for Partially Observed
//! Functional Data"* (J. Comput. Graph. Statist., 2026; PMC13160368).
//!
//! # Setting
//! Fully observed curves `f_1, …, f_n` are available; a new curve `f_{n+1}` is only
//! observed on a prefix `[t_0, s]` of the domain. The data are split into a training
//! part (used only to estimate the elastic template `f̄`, a Karcher mean) and a
//! calibration part. Each calibration curve is registered to the template,
//! `γ_i* = argmin_γ ‖q̄ − (q_i ∘ γ)√γ̇‖`, giving
//!
//! * the **amplitude** response `Y_i(t) = (f_i ∘ γ_i*)(t)` (the aligned curve), and
//! * the **phase** response `γ_i*` (the relative warp).
//!
//! The predictor is the observed prefix `X_i = f_i|_{[t_0, s]}`. Responses of the new
//! curve are predicted by Nadaraya–Watson neighbourhood smoothing over the predictors
//! (paper eq. 4, Gaussian kernel `K(u) = exp(−u²/2)`, bandwidth `h`), fitted
//! *leave-self-out* on the augmented set of the `n_2` calibration points plus the new
//! point (full conformal within the calibration set). Because the template comes from
//! the independent training part, the `n_2 + 1` (predictor, response) pairs are
//! exchangeable and the sets have finite-sample coverage
//!
//! * amplitude: `P((f_{n+1} ∘ γ_{n+1}*)(t) ∈ I_t) ≥ 1 − α` for every grid point `t`;
//! * phase: `P(γ_{n+1}* ∈ I_γ) ≥ 1 − α`,
//!
//! where `γ_{n+1}*` is the (unobservable) registration of the *complete* new curve.
//!
//! # What is implemented, and simplifications
//! * Split + Karcher template + DP registration exactly as in Algorithms 2/3.
//! * Predictor distances: L² on the observed prefix, L² between SRSFs on the prefix
//!   (no alignment), or the elastic amplitude distance on the prefix
//!   ([`PartialPredictorDistance`]).
//! * Amplitude scores are the **absolute** residuals `|Y_i(t) − Ŷ_i(t)|` (two-sided
//!   intervals). Since every leave-self-out NW fit is affine in the trial value `y`, the
//!   full-conformal interval is computed **exactly in closed form** by a breakpoint
//!   scan, instead of the paper's grid search over trial values. The reported interval
//!   is the convex hull of the (possibly non-convex) conformal set, so coverage is
//!   preserved.
//! * Phase: the NW fit of a warp is the pointwise convex combination of warps (which is
//!   again a warp); scores are the Fisher–Rao distance
//!   `d_w(γ_1, γ_2) = arccos ∫ √γ̇_1 √γ̇_2`. The set `I_γ` is available exactly through
//!   [`PartialConformalPrediction::phase_set_contains`]; for display the prediction
//!   carries the accepted members of a deterministic trial family (geodesics on the
//!   Hilbert sphere from the predicted centre toward each calibration warp), their
//!   pointwise envelope and the largest accepted distance from the centre.
//! * Bandwidth: fixed (user supplied), the median of all pairwise predictor distances
//!   among the `n_2 + 1` points (a symmetric function, so validity is kept; the
//!   default), or **data-driven tuning** (see below).
//!
//! # Options beyond the pointwise prefix setting (GH #72)
//!
//! * **Simultaneous bands** ([`PartialAmplitudeBand::Simultaneous`]): max-type scores
//!   `S_i = sup_t |Y_i(t) − Ŷ_i(t)| / σ(t)` with a modulation `σ` fixed independently of
//!   the calibration / test curves ([`PartialBandScale`]). With the leave-self-out NW fit
//!   on the augmented set, `Y_i − Ŷ_i = a_i − b_i y` for the trial curve `y`, and every
//!   `y` in the full-conformal set satisfies `sup |y − ĉ|/σ ≤ R`, where `R` is the `k`-th
//!   smallest of `A_i / (1 − b_i)`, `A_i = sup_t |a_i(t) − b_i ĉ(t)| / σ(t)` (triangle
//!   inequality on `S_i ≤ A_i + b_i S_{n+1}`). The reported band `ĉ ± R σ` contains the
//!   conformal set, hence `P(Y_{n+1}(t) ∈ band ∀t) ≥ 1 − α` in finite samples. With
//!   `b_i = 0` it is exactly the split-conformal sup-norm band.
//! * **Bandwidth tuning** ([`PartialConformalConfig::tuning_fraction`],
//!   [`partial_conformal_select_bandwidth`]). The paper adapts Liang, Zhu & Foygel Barber
//!   (2024) and picks `h` minimising the average interval length on the *same augmented
//!   data*. Their ModSel-CP guarantee needs the selection to be re-run for every trial
//!   response (a symmetric function of the augmented set), which has no closed form for
//!   our affine full-conformal scores. fdars therefore uses the validity-preserving
//!   **held-out** form: a seeded part `T` of the calibration curves is reserved, and
//!   `h* = argmin_h` of the mean band width over the leave-one-out rotations *within `T`*
//!   (each tuning curve in turn predicted from the other tuning curves, under the new
//!   curve's observation pattern). `h*` is a function of `T` (and of the observation
//!   pattern, assumed independent of the curves) only, so the remaining calibration
//!   curves and the new curve stay exchangeable given `h*` and the finite-sample coverage
//!   of every set is exact. The price is the `|T|` curves not used for calibration.
//!   Candidates are user-supplied or lower quantiles of the pairwise predictor distances
//!   within `T`; the same `h*` is used for the phase set.
//! * **Fragmented / sparse observation** ([`PartialObservation`],
//!   [`partial_conformal_predict_observed`]; the paper's fragmented and sparse
//!   settings). Fragments `J = ∪_j [U_{j1}, U_{j2}]` (grid index ranges) use
//!   `d_prod(X, X′) = Σ_j λ_j d(f|J_j, f′|J_j)` with `λ_j ∝ |J_j|` (`Σ λ_j = 1`) and the
//!   configured per-fragment distance; sparse observation at arbitrary times
//!   `t_1 < … < t_N` uses the Euclidean distance of the values in `ℝ^N` (calibration
//!   curves linearly interpolated at those times). Responses are unchanged (registration
//!   of the complete calibration curves to the template), so all guarantees carry over
//!   provided the observation pattern is independent of the curves. Each fragment (or the
//!   sparse point set) is located in the template by [`elastic_partial_match`], for
//!   information only.
//! * The partial registration of the new prefix onto the template (an estimate of the
//!   observed part of `γ_{n+1}*`) is reported via
//!   [`elastic_partial_match`]; it is
//!   informative only and does not enter the conformal sets.

use crate::alignment::{
    elastic_align_pair, elastic_distance, elastic_partial_match, karcher_mean, srsf_transform,
    AlignmentResult, PartialMatchConfig, PartialMatchResult,
};
use crate::error::FdarError;
use crate::helpers::{cumulative_trapz, gradient, simpsons_weights, trapz};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::conformal_split;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Distance between observed prefixes used by the neighbourhood smoother.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PartialPredictorDistance {
    /// L² distance of the raw prefixes (cheap; ignores phase).
    #[default]
    L2,
    /// L² distance between the SRSFs of the prefixes (no alignment).
    Srsf,
    /// Elastic (amplitude) distance between the prefixes, after DP alignment on the
    /// observed window. Costs `O(n_2²)` pairwise alignments per prediction.
    Elastic,
}

/// Type of the amplitude prediction band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PartialAmplitudeBand {
    /// Pointwise intervals: `P(Y(t) ∈ I_t) ≥ 1 − α` for every `t` separately (the
    /// paper's SFCP; exact closed-form full-conformal intervals).
    #[default]
    Pointwise,
    /// Simultaneous band from the max-type score `sup_t |Y − Ŷ| / σ(t)`:
    /// `P(Y(t) ∈ band for all t) ≥ 1 − α` (see the module docs).
    Simultaneous,
}

/// Modulation `σ(t)` of the simultaneous band `ĉ(t) ± R·σ(t)`.
///
/// It must not depend on the calibration or new curves (validity); every option
/// below satisfies this.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum PartialBandScale {
    /// `σ ≡ 1` (constant-width band).
    #[default]
    Constant,
    /// Pointwise standard deviation of the aligned *training* curves (independent of
    /// the calibration split), floored at `0.05 · max_t sd(t)`. Only available for
    /// models from [`partial_conformal_fit`].
    TrainingSd,
    /// User-supplied positive modulation on `argvals` (length m).
    Custom(Vec<f64>),
}

/// How a new curve is observed (for [`partial_conformal_predict_observed`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PartialObservation {
    /// The grid prefix `argvals[0..n]` (the setting of [`partial_conformal_predict`]).
    Prefix(usize),
    /// Disjoint fragments given as inclusive grid index ranges `(start, end)`,
    /// strictly increasing, each with at least 2 points (3 for the elastic distance).
    /// Values are passed concatenated in fragment order.
    Fragments(Vec<(usize, usize)>),
    /// Sparse observation at arbitrary strictly increasing times inside the domain
    /// (at least 2). Calibration curves are linearly interpolated at these times.
    Sparse(Vec<f64>),
}

/// Outcome of held-out bandwidth tuning ([`partial_conformal_select_bandwidth`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PartialBandwidthSelection {
    /// Selected bandwidth `h*`.
    pub bandwidth: f64,
    /// Candidate bandwidths that were compared.
    pub candidates: Vec<f64>,
    /// Mean amplitude band width over the leave-one-out tuning rotations, per
    /// candidate (`∞` if a band is unbounded).
    pub mean_width: Vec<f64>,
    /// Number of tuning curves.
    pub n_tuning: usize,
}

/// Configuration for [`partial_conformal_fit`] / [`partial_conformal_predict`].
///
/// Construct via `PartialConformalConfig::default()`, then assign the fields you need.
/// This struct is `#[non_exhaustive]`, so external crates cannot build it with a struct
/// literal.
///
/// # Example
/// ```
/// use fdars_core::conformal::{PartialConformalConfig, PartialPredictorDistance};
///
/// let mut cfg = PartialConformalConfig::default();
/// cfg.alpha = 0.2;
/// cfg.distance = PartialPredictorDistance::Srsf;
/// assert_eq!(cfg.cal_fraction, 0.5);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PartialConformalConfig {
    /// Miscoverage level `α ∈ (0, 1)` (default 0.1). Requires
    /// `⌈(1 − α)(n_2 + 1)⌉ ≤ n_2` for a finite set.
    pub alpha: f64,
    /// Fraction of the curves used for calibration in [`partial_conformal_fit`]
    /// (default 0.5); the rest estimate the template.
    pub cal_fraction: f64,
    /// Seed of the training / calibration split (default 42).
    pub seed: u64,
    /// Predictor distance (default [`PartialPredictorDistance::L2`]).
    pub distance: PartialPredictorDistance,
    /// Kernel bandwidth `h > 0`; `None` (default) uses the median pairwise predictor
    /// distance among the calibration curves and the new curve.
    pub bandwidth: Option<f64>,
    /// Roughness penalty of the DP registrations (default 0.0).
    pub lambda: f64,
    /// Maximum Karcher-mean iterations for the template (default 20).
    pub karcher_max_iter: usize,
    /// Karcher-mean convergence tolerance (default 1e-3).
    pub karcher_tol: f64,
    /// Number of radii per geodesic direction in the phase trial family (default 25).
    pub n_phase_radii: usize,
    /// Amplitude band type (default [`PartialAmplitudeBand::Pointwise`]).
    pub band: PartialAmplitudeBand,
    /// Modulation of the simultaneous band (default [`PartialBandScale::Constant`]);
    /// ignored for pointwise bands.
    pub band_scale: PartialBandScale,
    /// Held-out bandwidth tuning (default `None` = off). `Some(f)` makes the fit
    /// functions reserve a seeded fraction `f ∈ (0, 1)` of the calibration curves as
    /// a tuning set, on which [`partial_conformal_predict`] selects `h` (unless
    /// `bandwidth` is set, which takes precedence). See the module docs.
    pub tuning_fraction: Option<f64>,
    /// Candidate bandwidths for tuning (default empty = the 5, 10, 20, 30, 50, 70 and
    /// 100 % quantiles of the pairwise predictor distances among the tuning curves).
    pub bandwidth_candidates: Vec<f64>,
}

impl Default for PartialConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            cal_fraction: 0.5,
            seed: 42,
            distance: PartialPredictorDistance::L2,
            bandwidth: None,
            lambda: 0.0,
            karcher_max_iter: 20,
            karcher_tol: 1e-3,
            n_phase_radii: 25,
            band: PartialAmplitudeBand::Pointwise,
            band_scale: PartialBandScale::Constant,
            tuning_fraction: None,
            bandwidth_candidates: Vec::new(),
        }
    }
}

/// Fitted model: template plus registered calibration curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PartialConformalModel {
    /// Evaluation grid (length m).
    pub argvals: Vec<f64>,
    /// Elastic template `f̄` (length m).
    pub template: Vec<f64>,
    /// Row indices (into the input data) used to estimate the template (empty when the
    /// template was supplied).
    pub training_indices: Vec<usize>,
    /// Row indices (into the input data) of the calibration curves.
    pub calibration_indices: Vec<usize>,
    /// Raw calibration curves (n_2 × m).
    pub calibration_data: FdMatrix,
    /// Warps `γ_i*` registering each calibration curve to the template (n_2 × m).
    pub calibration_gammas: FdMatrix,
    /// Aligned calibration curves `f_i ∘ γ_i*` (n_2 × m).
    pub calibration_aligned: FdMatrix,
    /// Row indices (into the input data) of the held-out bandwidth-tuning curves
    /// (empty unless `tuning_fraction` was set).
    pub tuning_indices: Vec<usize>,
    /// Raw tuning curves (`n_T × m`, `0 × m` without tuning).
    pub tuning_data: FdMatrix,
    /// Tuning curves aligned to the template (`n_T × m`).
    pub tuning_aligned: FdMatrix,
    /// Pointwise standard deviation of the aligned training curves (`Some` only for
    /// [`partial_conformal_fit`]); used by [`PartialBandScale::TrainingSd`].
    pub training_sd: Option<Vec<f64>>,
}

/// Prediction sets for one partially observed curve.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PartialConformalPrediction {
    /// Number of observed grid points (prefix length).
    pub n_observed: usize,
    /// End `s` of the observed window.
    pub observed_end: f64,
    /// Miscoverage level used.
    pub alpha: f64,
    /// Bandwidth used by the neighbourhood smoother.
    pub bandwidth: f64,
    /// Normalised NW weights of the new curve over the calibration curves (length n_2).
    pub weights: Vec<f64>,
    /// Elastic partial match of the observed prefix within the template.
    pub registration: PartialMatchResult,
    /// Full registration to the template; `Some` only when the curve is fully observed
    /// (`n_observed == m`) — then its `gamma` / `f_aligned` are the conformal targets.
    pub full_registration: Option<AlignmentResult>,
    /// Registration of every fragment within the template (one per fragment for
    /// [`PartialObservation::Fragments`], empty otherwise); `registration` is the one
    /// of the longest fragment.
    pub fragment_registrations: Vec<PartialMatchResult>,
    /// Band type of `amplitude_lower` / `amplitude_upper`.
    pub band: PartialAmplitudeBand,
    /// Simultaneous band radius `R` (band = centre ± `R·σ`); `None` for pointwise bands.
    pub amplitude_radius: Option<f64>,
    /// Bandwidth tuning outcome when `h` was selected on the model's tuning set.
    pub bandwidth_selection: Option<PartialBandwidthSelection>,
    /// Point prediction of the aligned curve `f ∘ γ*` (length m).
    pub amplitude_center: Vec<f64>,
    /// Lower end of the amplitude prediction band (length m).
    pub amplitude_lower: Vec<f64>,
    /// Upper end of the amplitude prediction band (length m).
    pub amplitude_upper: Vec<f64>,
    /// Point prediction of the warp `γ*` (length m, on the `argvals` domain).
    pub phase_center: Vec<f64>,
    /// Pointwise lower envelope of the accepted phase trial warps (length m).
    pub phase_lower: Vec<f64>,
    /// Pointwise upper envelope of the accepted phase trial warps (length m).
    pub phase_upper: Vec<f64>,
    /// Largest Fisher–Rao distance from `phase_center` among accepted trial warps.
    pub phase_radius: f64,
    /// Accepted trial warps (rows; `n_accepted × m`, `argvals` domain).
    pub phase_accepted: FdMatrix,
    state: PhaseState,
}

/// Internal state for exact phase-set membership.
#[derive(Debug, Clone, PartialEq)]
struct PhaseState {
    /// Grid mapped to [0, 1].
    time: Vec<f64>,
    t0: f64,
    span: f64,
    /// Calibration warps on [0, 1] (n_2 × m).
    y: FdMatrix,
    /// Leave-self-out NW centre over the other calibration warps (n_2 × m).
    m: FdMatrix,
    /// Weight of the new point in each calibration fit.
    b: Vec<f64>,
    /// Predicted centre on [0, 1].
    center: Vec<f64>,
    /// Rank threshold `k = ⌈(1 − α)(n_2 + 1)⌉`.
    k: usize,
}

impl PartialConformalPrediction {
    /// Exact membership of a warp `γ` (length m, on the `argvals` domain) in the
    /// split-conformal phase prediction set `I_γ`.
    ///
    /// `γ ∈ I_γ` iff fewer than `k = ⌈(1 − α)(n_2 + 1)⌉` calibration scores
    /// `d_w(γ_i*, Ŷ_i(γ))` are strictly smaller than the new score `d_w(γ, Ŷ_{n+1})`.
    ///
    /// # Errors
    /// Returns [`FdarError::InvalidDimension`] if `gamma.len() != m`.
    pub fn phase_set_contains(&self, gamma: &[f64]) -> Result<bool, FdarError> {
        let st = &self.state;
        if gamma.len() != st.time.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "gamma",
                expected: format!("{}", st.time.len()),
                actual: format!("{}", gamma.len()),
            });
        }
        let g01: Vec<f64> = gamma.iter().map(|&g| (g - st.t0) / st.span).collect();
        Ok(phase_accept(st, &g01))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fitting
// ═══════════════════════════════════════════════════════════════════════════

/// Fit the partial-curve conformal model: split, Karcher template, registration.
///
/// The rows of `data` are randomly split (seed `config.seed`) into a training part,
/// whose elastic Karcher mean is the template, and a calibration part
/// (`config.cal_fraction` of the curves) that is registered to the template.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 rows or its
/// column count differs from `argvals`, and [`FdarError::InvalidParameter`] for an
/// invalid configuration or non-increasing `argvals`.
///
/// # Example
/// ```
/// use fdars_core::conformal::{
///     partial_conformal_fit, partial_conformal_predict, PartialConformalConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 21;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 24;
/// let mut data = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     let a = 0.8 + 0.4 * (i as f64 / n as f64);
///     let shift = 0.1 * ((i * 7 % 5) as f64 / 4.0 - 0.5);
///     for j in 0..m {
///         data[(i, j)] = a * (-(t[j] - 0.5 - shift).powi(2) / 0.02).exp();
///     }
/// }
/// let mut cfg = PartialConformalConfig::default();
/// cfg.alpha = 0.25;
/// let model = partial_conformal_fit(&data, &t, &cfg).unwrap();
/// let observed: Vec<f64> = (0..12).map(|j| (-(t[j] - 0.5f64).powi(2) / 0.02).exp()).collect();
/// let pred = partial_conformal_predict(&model, &observed, &cfg).unwrap();
/// assert_eq!(pred.amplitude_lower.len(), m);
/// assert!(pred.phase_set_contains(&pred.phase_center).unwrap());
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn partial_conformal_fit(
    data: &FdMatrix,
    argvals: &[f64],
    config: &PartialConformalConfig,
) -> Result<PartialConformalModel, FdarError> {
    validate_config(config)?;
    validate_grid(data, argvals)?;
    if data.nrows() < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 4 curves".to_string(),
            actual: format!("{}", data.nrows()),
        });
    }
    if config.cal_fraction <= 0.0 || config.cal_fraction >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "cal_fraction",
            message: format!("must be in (0, 1), got {}", config.cal_fraction),
        });
    }
    let (train_idx, cal_idx) = conformal_split(data.nrows(), config.cal_fraction, config.seed);
    let train = select_rows(data, &train_idx);
    let km = karcher_mean(
        &train,
        argvals,
        config.karcher_max_iter,
        config.karcher_tol,
        config.lambda,
    );
    let cal = select_rows(data, &cal_idx);
    let training_sd = pointwise_sd(&km.aligned_data);
    let mut model = register_calibration(cal, argvals, km.mean, config.lambda);
    model.training_indices = train_idx;
    model.calibration_indices = cal_idx;
    model.training_sd = Some(training_sd);
    if let Some(f) = config.tuning_fraction {
        split_tuning(&mut model, f, config.seed)?;
    }
    Ok(model)
}

/// Pointwise sample standard deviation of the rows (0 for a single row).
fn pointwise_sd(x: &FdMatrix) -> Vec<f64> {
    let (n, m) = x.shape();
    (0..m)
        .map(|j| {
            if n < 2 {
                return 0.0;
            }
            let col = x.column(j);
            let mean = col.iter().sum::<f64>() / n as f64;
            (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        })
        .collect()
}

/// Move a seeded fraction `f` of the calibration curves into the tuning set.
fn split_tuning(model: &mut PartialConformalModel, f: f64, seed: u64) -> Result<(), FdarError> {
    let n = model.calibration_data.nrows();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "calibration",
            expected: "at least 4 calibration curves for tuning_fraction".to_string(),
            actual: format!("{n}"),
        });
    }
    let (tune, conf) = conformal_split(n, 1.0 - f, seed.wrapping_add(1));
    model.tuning_indices = tune.iter().map(|&i| model.calibration_indices[i]).collect();
    model.tuning_data = select_rows(&model.calibration_data, &tune);
    model.tuning_aligned = select_rows(&model.calibration_aligned, &tune);
    model.calibration_indices = conf.iter().map(|&i| model.calibration_indices[i]).collect();
    model.calibration_data = select_rows(&model.calibration_data, &conf);
    model.calibration_gammas = select_rows(&model.calibration_gammas, &conf);
    model.calibration_aligned = select_rows(&model.calibration_aligned, &conf);
    Ok(())
}

/// Fit the model with a user-supplied template that is independent of the
/// calibration curves (e.g. estimated from a separate sample, or known).
///
/// All rows of `calibration` are registered to `template`.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `calibration` has fewer than 2 rows or
/// dimensions disagree, and [`FdarError::InvalidParameter`] for an invalid
/// configuration or non-increasing `argvals`.
///
/// # Example
/// ```
/// use fdars_core::conformal::{partial_conformal_fit_with_template, PartialConformalConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 15;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let template: Vec<f64> = t.iter().map(|&x| (3.0 * x).sin()).collect();
/// let mut cal = FdMatrix::zeros(6, m);
/// for i in 0..6 {
///     for j in 0..m {
///         cal[(i, j)] = (1.0 + 0.1 * i as f64) * template[j];
///     }
/// }
/// let model =
///     partial_conformal_fit_with_template(&cal, &t, &template, &PartialConformalConfig::default())
///         .unwrap();
/// assert_eq!(model.calibration_gammas.nrows(), 6);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn partial_conformal_fit_with_template(
    calibration: &FdMatrix,
    argvals: &[f64],
    template: &[f64],
    config: &PartialConformalConfig,
) -> Result<PartialConformalModel, FdarError> {
    validate_config(config)?;
    validate_grid(calibration, argvals)?;
    if calibration.nrows() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "calibration",
            expected: "at least 2 curves".to_string(),
            actual: format!("{}", calibration.nrows()),
        });
    }
    if template.len() != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "template",
            expected: format!("{}", argvals.len()),
            actual: format!("{}", template.len()),
        });
    }
    let n = calibration.nrows();
    let mut model = register_calibration(
        calibration.clone(),
        argvals,
        template.to_vec(),
        config.lambda,
    );
    model.calibration_indices = (0..n).collect();
    if let Some(f) = config.tuning_fraction {
        split_tuning(&mut model, f, config.seed)?;
    }
    Ok(model)
}

fn register_calibration(
    cal: FdMatrix,
    argvals: &[f64],
    template: Vec<f64>,
    lambda: f64,
) -> PartialConformalModel {
    let (n, m) = cal.shape();
    let aligned: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let r = elastic_align_pair(&template, &cal.row(i), argvals, lambda);
            (r.gamma, r.f_aligned)
        })
        .collect();
    let mut gammas = FdMatrix::zeros(n, m);
    let mut f_al = FdMatrix::zeros(n, m);
    for (i, (g, f)) in aligned.iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = g[j];
            f_al[(i, j)] = f[j];
        }
    }
    PartialConformalModel {
        argvals: argvals.to_vec(),
        template,
        training_indices: Vec::new(),
        calibration_indices: Vec::new(),
        calibration_data: cal,
        calibration_gammas: gammas,
        calibration_aligned: f_al,
        tuning_indices: Vec::new(),
        tuning_data: FdMatrix::zeros(0, m),
        tuning_aligned: FdMatrix::zeros(0, m),
        training_sd: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Prediction
// ═══════════════════════════════════════════════════════════════════════════

/// Split-conformal amplitude band and phase set for a partially observed curve.
///
/// `observed` holds the new curve's values on the grid prefix
/// `argvals[0..observed.len()]` (so `s = argvals[observed.len() − 1]`); pass all `m`
/// values for a fully observed curve.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `observed` has fewer than 2 or more than
/// `m` values, [`FdarError::InvalidParameter`] if a value is non-finite, the
/// configuration is invalid, or `α < 1/(n_2 + 1)` (the set would be unbounded).
///
/// # Example
/// See [`partial_conformal_fit`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn partial_conformal_predict(
    model: &PartialConformalModel,
    observed: &[f64],
    config: &PartialConformalConfig,
) -> Result<PartialConformalPrediction, FdarError> {
    predict_impl(
        model,
        &PartialObservation::Prefix(observed.len()),
        observed,
        config,
    )
}

/// Split-conformal amplitude band and phase set for a curve observed on fragments or
/// at sparse points (paper's fragmented / sparse settings).
///
/// `values` are the observed values in the order of `observation` (fragments
/// concatenated). [`PartialObservation::Prefix`] reproduces
/// [`partial_conformal_predict`] exactly. The observation pattern must be independent
/// of the curves for the coverage guarantee.
///
/// # Errors
/// As [`partial_conformal_predict`]; additionally [`FdarError::InvalidDimension`] /
/// [`FdarError::InvalidParameter`] for empty, overlapping, out-of-range or too short
/// fragments, non-increasing or out-of-domain sparse times, or a `values` length that
/// does not match the pattern.
///
/// # Example
/// ```
/// use fdars_core::conformal::{
///     partial_conformal_fit, partial_conformal_predict_observed, PartialConformalConfig,
///     PartialObservation,
/// };
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 21;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 24;
/// let mut data = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     let a = 0.8 + 0.4 * (i as f64 / n as f64);
///     for j in 0..m {
///         data[(i, j)] = a * (-(t[j] - 0.5).powi(2) / 0.02).exp();
///     }
/// }
/// let mut cfg = PartialConformalConfig::default();
/// cfg.alpha = 0.25;
/// let model = partial_conformal_fit(&data, &t, &cfg).unwrap();
/// let curve = data.row(0);
/// // Two fragments: grid points 2..=6 and 12..=16.
/// let obs = PartialObservation::Fragments(vec![(2, 6), (12, 16)]);
/// let values: Vec<f64> = curve[2..=6].iter().chain(&curve[12..=16]).copied().collect();
/// let pred = partial_conformal_predict_observed(&model, &obs, &values, &cfg).unwrap();
/// assert_eq!(pred.fragment_registrations.len(), 2);
/// // Sparse observation at three arbitrary times.
/// let sparse = PartialObservation::Sparse(vec![0.13, 0.42, 0.77]);
/// let pred = partial_conformal_predict_observed(&model, &sparse, &[0.1, 0.6, 0.3], &cfg).unwrap();
/// assert_eq!(pred.n_observed, 3);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn partial_conformal_predict_observed(
    model: &PartialConformalModel,
    observation: &PartialObservation,
    values: &[f64],
    config: &PartialConformalConfig,
) -> Result<PartialConformalPrediction, FdarError> {
    predict_impl(model, observation, values, config)
}

/// Held-out bandwidth selection on the calibration curves of `tuning_model`.
///
/// Each of the `n_T` calibration curves of `tuning_model` is in turn treated as the new
/// curve (observed under `observation`) and predicted from the other `n_T − 1`; the
/// selected bandwidth minimises the mean amplitude band width (of type `config.band`)
/// over these rotations. Ties go to the first candidate. Because the result depends
/// only on the tuning curves, using it as `config.bandwidth` for a model fitted on
/// *other* curves keeps exact finite-sample coverage; this is what
/// [`PartialConformalConfig::tuning_fraction`] does automatically.
///
/// # Errors
/// [`FdarError::InvalidParameter`] for an invalid configuration, observation or
/// candidate list, or if `α` needs more than `n_T − 1` tuning curves;
/// [`FdarError::ComputationFailed`] if every candidate gives an unbounded band.
///
/// # Example
/// ```
/// use fdars_core::conformal::{
///     partial_conformal_fit_with_template, partial_conformal_select_bandwidth,
///     PartialConformalConfig, PartialObservation,
/// };
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 21;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let template: Vec<f64> = t.iter().map(|&x| (-(x - 0.5).powi(2) / 0.02).exp()).collect();
/// let mut tune = FdMatrix::zeros(12, m);
/// for i in 0..12 {
///     for j in 0..m {
///         tune[(i, j)] = (0.7 + 0.05 * i as f64) * template[j];
///     }
/// }
/// let mut cfg = PartialConformalConfig::default();
/// cfg.alpha = 0.25;
/// let tm = partial_conformal_fit_with_template(&tune, &t, &template, &cfg).unwrap();
/// let sel = partial_conformal_select_bandwidth(&tm, &PartialObservation::Prefix(8), &cfg)
///     .unwrap();
/// assert!(sel.candidates.contains(&sel.bandwidth));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn partial_conformal_select_bandwidth(
    tuning_model: &PartialConformalModel,
    observation: &PartialObservation,
    config: &PartialConformalConfig,
) -> Result<PartialBandwidthSelection, FdarError> {
    validate_config(config)?;
    validate_pattern(observation, None, &tuning_model.argvals, config.distance)?;
    let sigma = band_sigma(tuning_model, config)?;
    select_bandwidth_on(
        &tuning_model.calibration_data,
        &tuning_model.calibration_aligned,
        &tuning_model.argvals,
        observation,
        config,
        &sigma,
    )
}

fn predict_impl(
    model: &PartialConformalModel,
    obs: &PartialObservation,
    values: &[f64],
    config: &PartialConformalConfig,
) -> Result<PartialConformalPrediction, FdarError> {
    validate_config(config)?;
    let m = model.argvals.len();
    let n2 = model.calibration_data.nrows();
    validate_pattern(obs, Some(values), &model.argvals, config.distance)?;
    let k = rank_threshold(n2, config.alpha)?;
    let sigma = band_sigma(model, config)?;

    let preds = augmented_predictors(&model.calibration_data, &model.argvals, obs, Some(values));
    let dist = pattern_distances(obs, &model.argvals, &preds, config);
    let (h, bandwidth_selection) = choose_bandwidth(model, obs, config, &dist, &sigma)?;
    let nw = NwWeights::new(&dist, n2, h);

    let amp = amplitude_sets(&model.calibration_aligned, &nw, k, config.band, &sigma);
    let state = phase_state(model, &nw, k);
    let (accepted, radius) = phase_trials(&state, config.n_phase_radii);
    let (registration, full_registration, fragment_registrations) =
        register_observation(model, obs, values, config.lambda)?;

    let to_dom = |v: f64| state.t0 + state.span * v;
    let (ph_lo, ph_hi) = envelope(&accepted, m);
    let mut acc = FdMatrix::zeros(accepted.len(), m);
    for (r, g) in accepted.iter().enumerate() {
        for j in 0..m {
            acc[(r, j)] = to_dom(g[j]);
        }
    }
    Ok(PartialConformalPrediction {
        n_observed: values.len(),
        observed_end: observation_end(obs, &model.argvals),
        alpha: config.alpha,
        bandwidth: h,
        weights: nw.test.clone(),
        registration,
        full_registration,
        fragment_registrations,
        band: config.band,
        amplitude_radius: amp.radius,
        bandwidth_selection,
        amplitude_center: amp.center,
        amplitude_lower: amp.lower,
        amplitude_upper: amp.upper,
        phase_center: state.center.iter().map(|&v| to_dom(v)).collect(),
        phase_lower: ph_lo.into_iter().map(to_dom).collect(),
        phase_upper: ph_hi.into_iter().map(to_dom).collect(),
        phase_radius: radius,
        phase_accepted: acc,
        state,
    })
}

fn observation_end(obs: &PartialObservation, argvals: &[f64]) -> f64 {
    match obs {
        PartialObservation::Prefix(n) => argvals[n - 1],
        PartialObservation::Fragments(f) => argvals[f[f.len() - 1].1],
        PartialObservation::Sparse(times) => times[times.len() - 1],
    }
}

/// Fixed bandwidth, held-out tuning on the model's tuning set, or the median rule.
fn choose_bandwidth(
    model: &PartialConformalModel,
    obs: &PartialObservation,
    config: &PartialConformalConfig,
    dist: &[f64],
    sigma: &[f64],
) -> Result<(f64, Option<PartialBandwidthSelection>), FdarError> {
    if let Some(h) = config.bandwidth {
        return Ok((h, None));
    }
    if model.tuning_data.nrows() > 0 {
        let sel = select_bandwidth_on(
            &model.tuning_data,
            &model.tuning_aligned,
            &model.argvals,
            obs,
            config,
            sigma,
        )?;
        return Ok((sel.bandwidth, Some(sel)));
    }
    let n2 = model.calibration_data.nrows();
    Ok((resolve_bandwidth(dist, n2 + 1, None)?, None))
}

// ─── Validation helpers ─────────────────────────────────────────────────────

fn validate_config(config: &PartialConformalConfig) -> Result<(), FdarError> {
    if !(config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {}", config.alpha),
        });
    }
    if let Some(h) = config.bandwidth {
        if !(h.is_finite() && h > 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter: "bandwidth",
                message: format!("must be finite and > 0, got {h}"),
            });
        }
    }
    if !(config.lambda.is_finite() && config.lambda >= 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("must be finite and >= 0, got {}", config.lambda),
        });
    }
    validate_tuning(config)
}

fn validate_tuning(config: &PartialConformalConfig) -> Result<(), FdarError> {
    if let Some(f) = config.tuning_fraction {
        if !(f > 0.0 && f < 1.0) {
            return Err(FdarError::InvalidParameter {
                parameter: "tuning_fraction",
                message: format!("must be in (0, 1), got {f}"),
            });
        }
    }
    if config
        .bandwidth_candidates
        .iter()
        .any(|h| !(h.is_finite() && *h > 0.0))
    {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth_candidates",
            message: "must all be finite and > 0".to_string(),
        });
    }
    Ok(())
}

fn bad_obs(message: String) -> FdarError {
    FdarError::InvalidParameter {
        parameter: "observation",
        message,
    }
}

/// Number of observed values implied by a pattern (after validating its structure).
fn pattern_len(
    obs: &PartialObservation,
    argvals: &[f64],
    distance: PartialPredictorDistance,
) -> Result<usize, FdarError> {
    match obs {
        PartialObservation::Prefix(n) => Ok(*n),
        PartialObservation::Fragments(frags) => fragments_len(frags, argvals.len(), distance),
        PartialObservation::Sparse(times) => sparse_len(times, argvals),
    }
}

fn fragments_len(
    frags: &[(usize, usize)],
    m: usize,
    distance: PartialPredictorDistance,
) -> Result<usize, FdarError> {
    let min_pts = if distance == PartialPredictorDistance::Elastic {
        3
    } else {
        2
    };
    if frags.is_empty() {
        return Err(bad_obs("at least one fragment is required".to_string()));
    }
    let mut prev_end: Option<usize> = None;
    let mut total = 0;
    for &(a, b) in frags {
        let ok_order = prev_end.map_or(true, |e| a > e);
        if b >= m || b + 1 < a + min_pts || !ok_order {
            return Err(bad_obs(format!(
                "fragment ({a}, {b}) must lie in 0..{m}, have >= {min_pts} points and \
                 start after the previous fragment"
            )));
        }
        prev_end = Some(b);
        total += b - a + 1;
    }
    Ok(total)
}

fn sparse_len(times: &[f64], argvals: &[f64]) -> Result<usize, FdarError> {
    let (lo, hi) = (argvals[0], argvals[argvals.len() - 1]);
    let inside = times.iter().all(|t| t.is_finite() && *t >= lo && *t <= hi);
    if times.len() < 2 || !inside || times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(bad_obs(format!(
            "sparse times must be >= 2 strictly increasing points in [{lo}, {hi}]"
        )));
    }
    Ok(times.len())
}

/// Validate an observation pattern and (optionally) the observed values.
fn validate_pattern(
    obs: &PartialObservation,
    values: Option<&[f64]>,
    argvals: &[f64],
    distance: PartialPredictorDistance,
) -> Result<(), FdarError> {
    let m = argvals.len();
    let need = pattern_len(obs, argvals, distance)?;
    let prefix_bad = matches!(obs, PartialObservation::Prefix(n) if !(2..=m).contains(n));
    let len_bad = values.is_some_and(|v| v.len() != need);
    if prefix_bad || len_bad {
        return Err(FdarError::InvalidDimension {
            parameter: "observed",
            expected: match obs {
                PartialObservation::Prefix(_) => format!("between 2 and {m} values"),
                _ => format!("{need} values"),
            },
            actual: format!("{}", values.map_or(need, <[f64]>::len)),
        });
    }
    if values.is_some_and(|v| v.iter().any(|x| !x.is_finite())) {
        return Err(FdarError::InvalidParameter {
            parameter: "observed",
            message: "values must be finite".to_string(),
        });
    }
    Ok(())
}

/// Modulation `σ` of the simultaneous band (all ones for pointwise bands).
fn band_sigma(
    model: &PartialConformalModel,
    config: &PartialConformalConfig,
) -> Result<Vec<f64>, FdarError> {
    let m = model.argvals.len();
    if config.band == PartialAmplitudeBand::Pointwise {
        return Ok(vec![1.0; m]);
    }
    match &config.band_scale {
        PartialBandScale::Constant => Ok(vec![1.0; m]),
        PartialBandScale::TrainingSd => {
            let sd = model
                .training_sd
                .as_ref()
                .ok_or_else(|| FdarError::InvalidParameter {
                    parameter: "band_scale",
                    message: "TrainingSd needs a model from partial_conformal_fit".to_string(),
                })?;
            let max = sd.iter().copied().fold(0.0_f64, f64::max);
            Ok(if max > 0.0 {
                sd.iter().map(|&v| v.max(0.05 * max)).collect()
            } else {
                vec![1.0; m]
            })
        }
        PartialBandScale::Custom(v) => {
            if v.len() != m {
                return Err(FdarError::InvalidDimension {
                    parameter: "band_scale",
                    expected: format!("{m}"),
                    actual: format!("{}", v.len()),
                });
            }
            if v.iter().any(|x| !(x.is_finite() && *x > 0.0)) {
                return Err(FdarError::InvalidParameter {
                    parameter: "band_scale",
                    message: "custom scale must be finite and > 0".to_string(),
                });
            }
            Ok(v.clone())
        }
    }
}

fn validate_grid(data: &FdMatrix, argvals: &[f64]) -> Result<(), FdarError> {
    let m = argvals.len();
    if m < 3 || data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length >= 3 matching data columns ({})", data.ncols()),
            actual: format!("{m}"),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must be strictly increasing".to_string(),
        });
    }
    Ok(())
}

/// `k = ⌈(1 − α)(n_2 + 1)⌉`, which must not exceed `n_2`.
fn rank_threshold(n2: usize, alpha: f64) -> Result<usize, FdarError> {
    let k = ((n2 + 1) as f64 * (1.0 - alpha) - 1e-12).ceil() as usize;
    if k > n2 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!(
                "alpha = {alpha} needs at least {} calibration curves (have {n2})",
                ((1.0 / alpha).ceil() as usize).saturating_sub(1)
            ),
        });
    }
    Ok(k.max(1))
}

fn select_rows(data: &FdMatrix, idx: &[usize]) -> FdMatrix {
    let m = data.ncols();
    let mut out = FdMatrix::zeros(idx.len(), m);
    for (r, &i) in idx.iter().enumerate() {
        for j in 0..m {
            out[(r, j)] = data[(i, j)];
        }
    }
    out
}

// ─── Predictor distances and NW weights ─────────────────────────────────────

/// Observed part of a full grid curve under `obs`.
fn extract_observed(obs: &PartialObservation, argvals: &[f64], row: &[f64]) -> Vec<f64> {
    match obs {
        PartialObservation::Prefix(n) => row[..*n].to_vec(),
        PartialObservation::Fragments(frags) => frags
            .iter()
            .flat_map(|&(a, b)| row[a..=b].iter().copied())
            .collect(),
        PartialObservation::Sparse(times) => {
            times.iter().map(|&t| lin_interp(argvals, row, t)).collect()
        }
    }
}

/// Linear interpolation of `(x, y)` at `t ∈ [x_0, x_last]`.
fn lin_interp(x: &[f64], y: &[f64], t: f64) -> f64 {
    let j = x.partition_point(|&v| v <= t).clamp(1, x.len() - 1);
    let w = (t - x[j - 1]) / (x[j] - x[j - 1]);
    (1.0 - w) * y[j - 1] + w * y[j]
}

/// Predictors of the rows of `curves` (plus the new curve's `extra` values, if any).
fn augmented_predictors(
    curves: &FdMatrix,
    argvals: &[f64],
    obs: &PartialObservation,
    extra: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    let mut preds: Vec<Vec<f64>> = (0..curves.nrows())
        .map(|i| extract_observed(obs, argvals, &curves.row(i)))
        .collect();
    if let Some(v) = extra {
        preds.push(v.to_vec());
    }
    preds
}

/// Symmetric `np × np` distance matrix (row-major) of the predictors under `obs`.
fn pattern_distances(
    obs: &PartialObservation,
    argvals: &[f64],
    preds: &[Vec<f64>],
    config: &PartialConformalConfig,
) -> Vec<f64> {
    match obs {
        PartialObservation::Prefix(n) => segment_distances(preds, &argvals[..*n], config),
        PartialObservation::Fragments(frags) => {
            let np = preds.len();
            let total: f64 = frags.iter().map(|&(a, b)| argvals[b] - argvals[a]).sum();
            let mut d = vec![0.0; np * np];
            let mut off = 0;
            for &(a, b) in frags {
                let len = b - a + 1;
                let seg: Vec<Vec<f64>> = preds.iter().map(|p| p[off..off + len].to_vec()).collect();
                let lam = (argvals[b] - argvals[a]) / total;
                let dj = segment_distances(&seg, &argvals[a..=b], config);
                d.iter_mut().zip(&dj).for_each(|(x, y)| *x += lam * y);
                off += len;
            }
            d
        }
        PartialObservation::Sparse(_) => {
            let np = preds.len();
            let mut d = vec![0.0; np * np];
            for i in 0..np {
                for j in (i + 1)..np {
                    let v = preds[i]
                        .iter()
                        .zip(&preds[j])
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f64>()
                        .sqrt();
                    d[i * np + j] = v;
                    d[j * np + i] = v;
                }
            }
            d
        }
    }
}

/// Distance matrix of curve segments observed on `sub_t` with the configured distance.
fn segment_distances(
    prefixes: &[Vec<f64>],
    sub_t: &[f64],
    config: &PartialConformalConfig,
) -> Vec<f64> {
    let w = simpsons_weights(sub_t);
    let reps: Vec<Vec<f64>> = match config.distance {
        PartialPredictorDistance::Srsf => prefixes.iter().map(|p| srsf_of(p, sub_t)).collect(),
        _ => prefixes.to_vec(),
    };
    let np = prefixes.len();
    let pairs: Vec<(usize, usize)> = (0..np)
        .flat_map(|i| ((i + 1)..np).map(move |j| (i, j)))
        .collect();
    let vals: Vec<f64> = iter_maybe_parallel!(pairs.clone())
        .map(|(i, j)| match config.distance {
            PartialPredictorDistance::Elastic => {
                elastic_distance(&prefixes[i], &prefixes[j], sub_t, config.lambda)
            }
            _ => weighted_l2(&reps[i], &reps[j], &w),
        })
        .collect();
    let mut d = vec![0.0; np * np];
    for (&(i, j), &v) in pairs.iter().zip(vals.iter()) {
        d[i * np + j] = v;
        d[j * np + i] = v;
    }
    d
}

fn srsf_of(f: &[f64], t: &[f64]) -> Vec<f64> {
    let mat = FdMatrix::from_slice(f, 1, f.len()).expect("1 × len matrix");
    srsf_transform(&mat, t).row(0)
}

fn weighted_l2(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .zip(w)
        .map(|((&x, &y), &wi)| wi * (x - y) * (x - y))
        .sum::<f64>()
        .max(0.0)
        .sqrt()
}

fn resolve_bandwidth(dist: &[f64], np: usize, user: Option<f64>) -> Result<f64, FdarError> {
    if let Some(h) = user {
        return Ok(h);
    }
    let mut v: Vec<f64> = (0..np)
        .flat_map(|i| ((i + 1)..np).map(move |j| (i, j)))
        .map(|(i, j)| dist[i * np + j])
        .filter(|d| d.is_finite())
        .collect();
    if v.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "partial_conformal_predict",
            detail: "no finite predictor distances".to_string(),
        });
    }
    crate::helpers::sort_nan_safe(&mut v);
    let med = if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    };
    Ok(if med > 0.0 { med } else { 1.0 })
}

/// Leave-self-out Nadaraya–Watson weights on the augmented set.
struct NwWeights {
    /// `u[i][j]`: weight of calibration point j in calibration fit i, renormalised over
    /// the other calibration points (n_2 × n_2, row-major, zero diagonal).
    u: Vec<f64>,
    /// Weight of the new point in each calibration fit.
    b: Vec<f64>,
    /// Weights of the calibration points in the new point's fit.
    test: Vec<f64>,
    n2: usize,
}

impl NwWeights {
    fn new(dist: &[f64], n2: usize, h: f64) -> Self {
        let np = n2 + 1;
        let mut u = vec![0.0; n2 * n2];
        let mut b = vec![0.0; n2];
        for i in 0..n2 {
            let row = kernel_row(&dist[i * np..(i + 1) * np], i, h);
            let s_cal: f64 = row[..n2].iter().sum();
            let total = s_cal + row[n2];
            b[i] = row[n2] / total;
            if s_cal > 0.0 {
                for j in 0..n2 {
                    u[i * n2 + j] = row[j] / s_cal;
                }
            } else {
                b[i] = 1.0;
            }
        }
        let row = kernel_row(&dist[n2 * np..], n2, h);
        let s: f64 = row[..n2].iter().sum();
        let test = row[..n2].iter().map(|&v| v / s).collect();
        Self { u, b, test, n2 }
    }

    /// Leave-self-out centre of calibration fit i over the other calibration points.
    fn center_of(&self, i: usize, resp: &FdMatrix, j: usize) -> f64 {
        (0..self.n2)
            .map(|l| self.u[i * self.n2 + l] * resp[(l, j)])
            .sum()
    }
}

/// Gaussian kernel row with the self entry zeroed, scaled so its largest entry is 1.
fn kernel_row(drow: &[f64], self_idx: usize, h: f64) -> Vec<f64> {
    let dmin_sq = drow
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != self_idx)
        .map(|(_, &d)| d * d)
        .fold(f64::INFINITY, f64::min);
    drow.iter()
        .enumerate()
        .map(|(j, &d)| {
            if j == self_idx {
                0.0
            } else {
                (-(d * d - dmin_sq) / (2.0 * h * h)).exp()
            }
        })
        .collect()
}

// ─── Amplitude band (exact full conformal, affine scores) ───────────────────

fn amplitude_band(resp: &FdMatrix, nw: &NwWeights, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n2, m) = resp.shape();
    let mut center = vec![0.0; m];
    let mut lo = vec![0.0; m];
    let mut hi = vec![0.0; m];
    let mut a = vec![0.0; n2];
    for j in 0..m {
        let c: f64 = (0..n2).map(|l| nw.test[l] * resp[(l, j)]).sum();
        for (i, ai) in a.iter_mut().enumerate() {
            let mi = nw.center_of(i, resp, j);
            *ai = resp[(i, j)] - (1.0 - nw.b[i]) * mi;
        }
        let (l, u) = affine_conformal_interval(&a, &nw.b, c, k);
        center[j] = c;
        lo[j] = l;
        hi[j] = u;
    }
    (center, lo, hi)
}

/// Exact full-conformal interval (convex hull) for scores
/// `|a_i − b_i y|` (calibration) and `|y − c|` (new point): `y` is accepted iff fewer
/// than `k` calibration scores are strictly below the new score.
pub(super) fn affine_conformal_interval(a: &[f64], b: &[f64], c: f64, k: usize) -> (f64, f64) {
    let mut bps: Vec<f64> = Vec::with_capacity(2 * a.len() + 1);
    bps.push(c);
    for (&ai, &bi) in a.iter().zip(b) {
        bps.push((ai + c) / (1.0 + bi));
        if bi < 1.0 {
            bps.push((c - ai) / (1.0 - bi));
        }
    }
    bps.retain(|v| v.is_finite());
    crate::helpers::sort_nan_safe(&mut bps);
    bps.dedup();
    let accept = |y: f64| {
        let st = (y - c).abs();
        let tol = 1e-12 * (1.0 + st);
        let below = a
            .iter()
            .zip(b)
            .filter(|(&ai, &bi)| (ai - bi * y).abs() < st - tol)
            .count();
        below < k
    };
    let first = bps[0];
    let last = bps[bps.len() - 1];
    let pad = 1.0 + (last - first).abs();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let consider = |y: f64, lo: &mut f64, hi: &mut f64| {
        if accept(y) {
            *lo = lo.min(y);
            *hi = hi.max(y);
        }
    };
    for w in bps.windows(2) {
        consider(0.5 * (w[0] + w[1]), &mut lo, &mut hi);
    }
    for &y in &bps {
        consider(y, &mut lo, &mut hi);
    }
    if accept(first - pad) {
        lo = f64::NEG_INFINITY;
    }
    if accept(last + pad) {
        hi = f64::INFINITY;
    }
    (lo, hi)
}

/// Amplitude centre and band (pointwise or simultaneous).
struct AmpSets {
    center: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    radius: Option<f64>,
}

fn amplitude_sets(
    resp: &FdMatrix,
    nw: &NwWeights,
    k: usize,
    band: PartialAmplitudeBand,
    sigma: &[f64],
) -> AmpSets {
    match band {
        PartialAmplitudeBand::Simultaneous => simultaneous_band(resp, nw, k, sigma),
        _ => {
            let (center, lower, upper) = amplitude_band(resp, nw, k);
            AmpSets {
                center,
                lower,
                upper,
                radius: None,
            }
        }
    }
}

/// Simultaneous band `ĉ ± R σ` containing the full-conformal set of the max-type
/// score `sup_t |a_i(t) − b_i y(t)| / σ(t)` (see the module docs for the bound).
fn simultaneous_band(resp: &FdMatrix, nw: &NwWeights, k: usize, sigma: &[f64]) -> AmpSets {
    let (n2, m) = resp.shape();
    let center: Vec<f64> = (0..m)
        .map(|j| (0..n2).map(|l| nw.test[l] * resp[(l, j)]).sum())
        .collect();
    let mut bound: Vec<f64> = (0..n2)
        .map(|i| {
            let bi = nw.b[i];
            let a_sup = (0..m)
                .map(|j| {
                    let ai = resp[(i, j)] - (1.0 - bi) * nw.center_of(i, resp, j);
                    (ai - bi * center[j]).abs() / sigma[j]
                })
                .fold(0.0_f64, f64::max);
            if bi < 1.0 {
                a_sup / (1.0 - bi)
            } else {
                f64::INFINITY
            }
        })
        .collect();
    crate::helpers::sort_nan_safe(&mut bound);
    let r = bound[k - 1];
    AmpSets {
        lower: center.iter().zip(sigma).map(|(c, s)| c - r * s).collect(),
        upper: center.iter().zip(sigma).map(|(c, s)| c + r * s).collect(),
        center,
        radius: Some(r),
    }
}

// ─── Held-out bandwidth tuning ──────────────────────────────────────────────

fn select_bandwidth_on(
    raw: &FdMatrix,
    aligned: &FdMatrix,
    argvals: &[f64],
    obs: &PartialObservation,
    config: &PartialConformalConfig,
    sigma: &[f64],
) -> Result<PartialBandwidthSelection, FdarError> {
    let nt = raw.nrows();
    let k = rank_threshold(nt.saturating_sub(1), config.alpha).map_err(|_| {
        FdarError::InvalidParameter {
            parameter: "tuning_fraction",
            message: format!(
                "alpha = {} needs more tuning curves (have {nt})",
                config.alpha
            ),
        }
    })?;
    let preds = augmented_predictors(raw, argvals, obs, None);
    let d = pattern_distances(obs, argvals, &preds, config);
    let candidates = if config.bandwidth_candidates.is_empty() {
        default_candidates(&d, nt)?
    } else {
        config.bandwidth_candidates.clone()
    };
    let mean_width: Vec<f64> = candidates
        .iter()
        .map(|&h| loo_mean_width(&d, aligned, h, k, config.band, sigma))
        .collect();
    let best = mean_width
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_finite())
        .fold(None, |acc: Option<(usize, f64)>, (i, &w)| match acc {
            Some((_, bw)) if bw <= w => acc,
            _ => Some((i, w)),
        })
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "partial_conformal_select_bandwidth",
            detail: "every candidate bandwidth gives an unbounded band".to_string(),
        })?;
    Ok(PartialBandwidthSelection {
        bandwidth: candidates[best.0],
        candidates,
        mean_width,
        n_tuning: nt,
    })
}

/// Lower quantiles of the positive pairwise distances among the tuning curves.
fn default_candidates(d: &[f64], nt: usize) -> Result<Vec<f64>, FdarError> {
    let mut v: Vec<f64> = (0..nt)
        .flat_map(|i| ((i + 1)..nt).map(move |j| d[i * nt + j]))
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if v.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "partial_conformal_select_bandwidth",
            detail: "no positive predictor distances among the tuning curves".to_string(),
        });
    }
    crate::helpers::sort_nan_safe(&mut v);
    let mut c: Vec<f64> = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        .iter()
        .map(|q| v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)])
        .collect();
    c.dedup();
    Ok(c)
}

/// Mean band width when each tuning curve in turn is predicted from the others.
fn loo_mean_width(
    d: &[f64],
    aligned: &FdMatrix,
    h: f64,
    k: usize,
    band: PartialAmplitudeBand,
    sigma: &[f64],
) -> f64 {
    let nt = aligned.nrows();
    let widths: Vec<f64> = iter_maybe_parallel!(0..nt)
        .map(|tau| {
            let order: Vec<usize> = (0..nt).filter(|&i| i != tau).chain([tau]).collect();
            let dperm: Vec<f64> = order
                .iter()
                .flat_map(|&a| order.iter().map(move |&b| d[a * nt + b]))
                .collect();
            let resp = select_rows(aligned, &order[..nt - 1]);
            let nw = NwWeights::new(&dperm, nt - 1, h);
            let amp = amplitude_sets(&resp, &nw, k, band, sigma);
            amp.upper
                .iter()
                .zip(&amp.lower)
                .map(|(u, l)| u - l)
                .sum::<f64>()
                / amp.upper.len() as f64
        })
        .collect();
    widths.iter().sum::<f64>() / nt as f64
}

// ─── Phase set ──────────────────────────────────────────────────────────────

fn phase_state(model: &PartialConformalModel, nw: &NwWeights, k: usize) -> PhaseState {
    let t = &model.argvals;
    let m = t.len();
    let t0 = t[0];
    let span = t[m - 1] - t0;
    let time: Vec<f64> = t.iter().map(|&v| (v - t0) / span).collect();
    let n2 = nw.n2;
    let mut y = FdMatrix::zeros(n2, m);
    for i in 0..n2 {
        for j in 0..m {
            y[(i, j)] = (model.calibration_gammas[(i, j)] - t0) / span;
        }
    }
    let mut mm = FdMatrix::zeros(n2, m);
    let mut center = vec![0.0; m];
    for j in 0..m {
        for i in 0..n2 {
            mm[(i, j)] = nw.center_of(i, &y, j);
        }
        center[j] = (0..n2).map(|l| nw.test[l] * y[(l, j)]).sum();
    }
    PhaseState {
        time,
        t0,
        span,
        y,
        m: mm,
        b: nw.b.clone(),
        center,
        k,
    }
}

/// √γ̇ of a [0, 1] warp, normalised to unit L² norm.
fn unit_psi(g: &[f64], time: &[f64]) -> Vec<f64> {
    let mut psi: Vec<f64> = gradient(g, time)
        .into_iter()
        .map(|d| d.max(0.0).sqrt())
        .collect();
    let sq: Vec<f64> = psi.iter().map(|p| p * p).collect();
    let nrm = trapz(&sq, time).max(0.0).sqrt();
    if nrm > 1e-12 {
        psi.iter_mut().for_each(|p| *p /= nrm);
    }
    psi
}

fn sphere_dist(p1: &[f64], p2: &[f64], time: &[f64]) -> f64 {
    let prod: Vec<f64> = p1.iter().zip(p2).map(|(a, b)| a * b).collect();
    trapz(&prod, time).clamp(-1.0, 1.0).acos()
}

/// Fisher–Rao distance between two warps on [0, 1].
fn warp_dist(g1: &[f64], g2: &[f64], time: &[f64]) -> f64 {
    sphere_dist(&unit_psi(g1, time), &unit_psi(g2, time), time)
}

fn phase_accept(st: &PhaseState, g: &[f64]) -> bool {
    let s_new = warp_dist(g, &st.center, &st.time);
    let tol = 1e-12 * (1.0 + s_new);
    let n2 = st.b.len();
    let mut below = 0usize;
    let mut fit = vec![0.0; g.len()];
    for i in 0..n2 {
        let bi = st.b[i];
        for (j, f) in fit.iter_mut().enumerate() {
            *f = (1.0 - bi) * st.m[(i, j)] + bi * g[j];
        }
        if warp_dist(&st.y.row(i), &fit, &st.time) < s_new - tol {
            below += 1;
        }
    }
    below < st.k
}

/// Deterministic trial family: the centre, the calibration warps, and points on the
/// sphere geodesics from the centre toward each calibration warp at `n_radii` radii.
fn phase_trials(st: &PhaseState, n_radii: usize) -> (Vec<Vec<f64>>, f64) {
    let time = &st.time;
    let psi_c = unit_psi(&st.center, time);
    let n2 = st.b.len();
    let dirs: Vec<(Vec<f64>, f64)> = (0..n2)
        .filter_map(|i| {
            let v = crate::warping::inv_exp_map_sphere(&psi_c, &unit_psi(&st.y.row(i), time), time);
            let th = crate::warping::l2_norm_l2(&v, time);
            (th > 1e-10).then(|| (v.iter().map(|x| x / th).collect(), th))
        })
        .collect();
    let r_max = dirs.iter().map(|d| d.1).fold(0.0_f64, f64::max) * 1.5;
    let r_max = r_max.min(std::f64::consts::FRAC_PI_2);
    let mut cands: Vec<Vec<f64>> = vec![st.center.clone()];
    cands.extend((0..n2).map(|i| st.y.row(i)));
    for (v, _) in &dirs {
        for l in 1..=n_radii {
            let r = r_max * l as f64 / n_radii as f64;
            let step: Vec<f64> = v.iter().map(|x| x * r).collect();
            let psi = crate::warping::exp_map_sphere(&psi_c, &step, time);
            cands.push(psi_to_warp(&psi, time));
        }
    }
    let flags: Vec<bool> = iter_maybe_parallel!(0..cands.len())
        .map(|c| phase_accept(st, &cands[c]))
        .collect();
    let accepted: Vec<Vec<f64>> = cands
        .into_iter()
        .zip(flags)
        .filter_map(|(c, f)| f.then_some(c))
        .collect();
    let radius = accepted
        .iter()
        .map(|g| sphere_dist(&psi_c, &unit_psi(g, time), time))
        .fold(0.0_f64, f64::max);
    (accepted, radius)
}

fn psi_to_warp(psi: &[f64], time: &[f64]) -> Vec<f64> {
    let sq: Vec<f64> = psi.iter().map(|p| p * p).collect();
    let g = cumulative_trapz(&sq, time);
    let end = g[g.len() - 1].max(1e-15);
    g.iter().map(|v| v / end).collect()
}

fn envelope(rows: &[Vec<f64>], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for r in rows {
        for j in 0..m {
            lo[j] = lo[j].min(r[j]);
            hi[j] = hi[j].max(r[j]);
        }
    }
    (lo, hi)
}

// ─── Partial registration ───────────────────────────────────────────────────

/// Partial match of the segment `argvals[a..=b]` (values `vals`) within the template.
fn register_segment(
    model: &PartialConformalModel,
    vals: &[f64],
    a: usize,
    b: usize,
    lambda: f64,
) -> Result<PartialMatchResult, FdarError> {
    let m = model.argvals.len();
    let sub_t = &model.argvals[a..=b];
    let t0 = model.argvals[0];
    let frac = (model.argvals[b] - model.argvals[a]) / (model.argvals[m - 1] - t0);
    let min_span = if a == 0 && b == m - 1 {
        1.0
    } else {
        (0.5 * frac).max(2.0 / m as f64).min(1.0)
    };
    // Same-crate struct literal (the config is only sealed for external crates).
    let pm = PartialMatchConfig { lambda, min_span };
    elastic_partial_match(vals, &model.template, sub_t, &model.argvals, &pm)
}

type Registrations = (
    PartialMatchResult,
    Option<AlignmentResult>,
    Vec<PartialMatchResult>,
);

fn register_observation(
    model: &PartialConformalModel,
    obs: &PartialObservation,
    values: &[f64],
    lambda: f64,
) -> Result<Registrations, FdarError> {
    let m = model.argvals.len();
    let full_curve = |covers: bool| {
        covers.then(|| elastic_align_pair(&model.template, values, &model.argvals, lambda))
    };
    match obs {
        PartialObservation::Prefix(n) => {
            let reg = register_segment(model, values, 0, n - 1, lambda)?;
            Ok((reg, full_curve(*n == m), Vec::new()))
        }
        PartialObservation::Fragments(frags) => {
            let mut regs = Vec::with_capacity(frags.len());
            let mut off = 0;
            for &(a, b) in frags {
                let len = b - a + 1;
                regs.push(register_segment(
                    model,
                    &values[off..off + len],
                    a,
                    b,
                    lambda,
                )?);
                off += len;
            }
            let longest = (0..frags.len()).fold(0, |best, j| {
                let w = |i: usize| model.argvals[frags[i].1] - model.argvals[frags[i].0];
                if w(j) > w(best) {
                    j
                } else {
                    best
                }
            });
            let covers = frags.len() == 1 && frags[0] == (0, m - 1);
            Ok((regs[longest].clone(), full_curve(covers), regs))
        }
        PartialObservation::Sparse(times) => {
            let t0 = model.argvals[0];
            let frac = (times[times.len() - 1] - times[0]) / (model.argvals[m - 1] - t0);
            let min_span = (0.5 * frac).max(2.0 / m as f64).min(1.0);
            let pm = PartialMatchConfig { lambda, min_span };
            let reg = elastic_partial_match(values, &model.template, times, &model.argvals, &pm)?;
            Ok((reg, None, Vec::new()))
        }
    }
}

#[cfg(test)]
#[path = "partial_tests.rs"]
mod tests;
