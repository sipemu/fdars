//! Metric-space abstraction and the 1D-Wasserstein (density-response) backend.
//!
//! Provides the [`MetricSpace`] trait (a distance + weighted-Fréchet-mean solver)
//! and its first concrete implementation, [`WassersteinDensitySpace`], whose
//! objects are probability densities on a shared strictly-increasing grid and
//! whose metric is the 1D 2-Wasserstein distance ([`wasserstein2_distance`]).
//!
//! The density backend reuses DENS-01's quantile/Wasserstein machinery
//! ([`crate::density_fda::wasserstein_barycenter`] and its density→quantile→
//! density back-map) rather than re-deriving it.

use crate::density_fda::{dedup_adjacent, quantile_density_from_q, wasserstein_barycenter};
use crate::error::FdarError;
use crate::helpers::{cumulative_trapz, linear_interp, trapz};
use crate::matrix::FdMatrix;

/// A metric space: a distance function plus a weighted-Fréchet-mean solver over
/// its objects. Regression / statistics routines are generic over this trait.
///
/// Implementors must be `Send + Sync` so the statistics routines can parallelize.
/// `weighted_frechet_mean` expects **non-negative** weights (they are normalized
/// to sum to 1 by callers); the signed-weight regression path uses the private
/// [`signed_quantile_average`] helper instead, never `weighted_frechet_mean`.
pub trait MetricSpace: Send + Sync {
    /// The object type living in this metric space (e.g. a density on a grid).
    type Object;

    /// Distance between two objects.
    ///
    /// # Errors
    /// Returns [`FdarError`] on dimension mismatch or degenerate input.
    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError>;

    /// Weighted Fréchet mean (barycenter) of `objects` under non-negative `weights`.
    ///
    /// # Errors
    /// Returns [`FdarError`] on empty input, weight/length mismatch, or a
    /// degenerate barycenter.
    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError>;
}

/// The 1D-Wasserstein (density-response) metric space.
///
/// Objects are probability densities sampled on the shared strictly-increasing
/// grid `argvals`; the metric is the 1D 2-Wasserstein distance and the weighted
/// Fréchet mean is the Wasserstein barycenter (quantile average).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WassersteinDensitySpace {
    /// Shared strictly-increasing evaluation grid for all density objects.
    pub argvals: Vec<f64>,
}

impl WassersteinDensitySpace {
    /// Construct a density space over a strictly-increasing `argvals` grid.
    ///
    /// # Errors
    /// Returns [`FdarError::InvalidDimension`] if `argvals` has fewer than 2
    /// points, or [`FdarError::InvalidParameter`] if it is not strictly
    /// increasing.
    pub fn new(argvals: Vec<f64>) -> Result<Self, FdarError> {
        if argvals.len() < 2 {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals",
                expected: "at least 2 grid points".to_string(),
                actual: format!("{} points", argvals.len()),
            });
        }
        if argvals.windows(2).any(|w| w[1] <= w[0]) {
            return Err(FdarError::InvalidParameter {
                parameter: "argvals",
                message: "argvals must be strictly increasing".to_string(),
            });
        }
        Ok(Self { argvals })
    }
}

impl MetricSpace for WassersteinDensitySpace {
    type Object = Vec<f64>;

    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError> {
        wasserstein2_distance(a, b, &self.argvals)
    }

    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError> {
        let m = self.argvals.len();
        if objects.is_empty() {
            return Err(FdarError::InvalidDimension {
                parameter: "objects",
                expected: "at least 1 object".to_string(),
                actual: "0 objects".to_string(),
            });
        }
        let n = objects.len();
        let mut mat = FdMatrix::zeros(n, m);
        for (i, obj) in objects.iter().enumerate() {
            if obj.len() != m {
                return Err(FdarError::InvalidDimension {
                    parameter: "objects",
                    expected: format!("each object has {m} points"),
                    actual: format!("object {i} has {} points", obj.len()),
                });
            }
            for j in 0..m {
                mat[(i, j)] = obj[j];
            }
        }
        // Non-negative-weight sample barycenter — reuse DENS-01's solver.
        // (Signed-weight regression uses `signed_quantile_average` instead.)
        wasserstein_barycenter(&mat, &self.argvals, Some(weights))
    }
}

/// The 1D 2-Wasserstein distance between two densities on a shared grid.
///
/// Computed as the L² distance between quantile functions,
/// `W₂(F,G) = (∫₀¹ (Q_F(t) − Q_G(t))² dt)^{1/2}`, reusing the density→CDF→quantile
/// machinery of [`crate::density_fda`]. This is the metric behind
/// [`MetricSpace::distance`] for [`WassersteinDensitySpace`].
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `a`, `b`, and `argvals` lengths
/// differ or `argvals` has fewer than 2 points.
#[must_use = "returns the 2-Wasserstein distance; result should be examined"]
pub fn wasserstein2_distance(a: &[f64], b: &[f64], argvals: &[f64]) -> Result<f64, FdarError> {
    let m = argvals.len();
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: "at least 2 grid points".to_string(),
            actual: format!("{m} points"),
        });
    }
    if a.len() != m || b.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "a/b",
            expected: format!("both length {m} (matching argvals)"),
            actual: format!("a={}, b={}", a.len(), b.len()),
        });
    }
    let n_q = m.max(101);
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();
    let qa = density_to_quantile(a, argvals, &t_grid);
    let qb = density_to_quantile(b, argvals, &t_grid);
    let sq_diff: Vec<f64> = qa
        .iter()
        .zip(qb.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .collect();
    Ok(trapz(&sq_diff, &t_grid).sqrt())
}

/// Quantile function `Q(t)` of a density `row` on `argvals`, evaluated at `t_grid`.
///
/// Normalizes the density to integrate to 1, forms its CDF via
/// [`cumulative_trapz`], and inverts by interpolating the CDF at each probability
/// `t` — replicating the density→quantile step of
/// [`crate::density_fda::wasserstein_barycenter`].
#[inline]
fn density_to_quantile(row: &[f64], argvals: &[f64], t_grid: &[f64]) -> Vec<f64> {
    let integral = trapz(row, argvals);
    let inv = if integral.abs() < 1e-300 {
        1.0
    } else {
        1.0 / integral
    };
    let norm: Vec<f64> = row.iter().map(|&v| v * inv).collect();
    let cdf = cumulative_trapz(&norm, argvals);
    t_grid
        .iter()
        .map(|&t| linear_interp(&cdf, argvals, t))
        .collect()
}

/// Signed weighted quantile average → density, with a sort-based monotone
/// (isotonic) projection. **Reserved for the signed-weight regression path**
/// (global/local Fréchet regression); the non-negative-weight sample Fréchet mean
/// ([`WassersteinDensitySpace::weighted_frechet_mean`]) uses
/// [`crate::density_fda::wasserstein_barycenter`] instead and never calls this.
///
/// Computes `Q̄(t) = Σᵢ wᵢ · Qᵢ(t)` with possibly-**negative** weights (so it does
/// NOT call `wasserstein_barycenter`, which rejects negative weights), then sorts
/// `Q̄` to restore monotonicity and inverts it back to a density on `argvals`
/// using the same back-map as the Wasserstein barycenter.
///
/// # Divergence from R `frechet`
///
/// R's `GloWassReg`/`LocWassReg` enforce a monotone quantile via an `osqp`
/// quadratic-program projection. To avoid a new crate dependency this uses a
/// sort-based isotonic projection — equivalent on smooth quantile averages,
/// slightly more conservative on non-smooth ones.
///
/// # Errors
/// Returns [`FdarError`] on dimension mismatch or a degenerate (zero-range)
/// quantile average.
pub(crate) fn signed_quantile_average(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    weights: &[f64],
    n_q: usize,
) -> Result<Vec<f64>, FdarError> {
    let (n, m) = density_matrix.shape();
    if m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "density_matrix",
            expected: format!("{} columns (matching argvals)", argvals.len()),
            actual: format!("{m} columns"),
        });
    }
    if weights.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "weights",
            expected: format!("{n} weights (matching rows)"),
            actual: format!("{} weights", weights.len()),
        });
    }
    if n_q < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_q",
            message: "n_q must be at least 2".to_string(),
        });
    }
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();

    // Signed weighted average of quantile functions.
    let mut q_bar = vec![0.0_f64; n_q];
    for i in 0..n {
        let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
        let qi = density_to_quantile(&row, argvals, &t_grid);
        let wi = weights[i];
        for j in 0..n_q {
            q_bar[j] += wi * qi[j];
        }
    }

    // Sort-based monotone (isotonic) projection — the no-osqp alternative.
    q_bar.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Clamp the averaged quantile to the target support so the inverted density
    // stays on `argvals` (signed extrapolation weights can push Q̄ slightly past
    // the grid edges).
    let lb = argvals[0];
    let ub = argvals[m - 1];
    for v in q_bar.iter_mut() {
        *v = v.clamp(lb, ub);
    }
    let q_range = q_bar[n_q - 1] - q_bar[0];
    if q_range < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "signed_quantile_average",
            detail: "quantile average has zero range; degenerate weighted input".to_string(),
        });
    }

    // Invert Q̄ → density directly in x-units (Q̄ is a weighted average of quantile
    // functions, already on the argvals x-scale — no rescale-to-full-support, which
    // would spuriously stretch a narrow barycenter across the whole grid).
    let dens_raw = quantile_density_from_q(&q_bar, &t_grid);
    let (q_dedup, dens_dedup) = dedup_adjacent(&q_bar, &dens_raw);
    let dens: Vec<f64> = argvals
        .iter()
        .map(|&x| linear_interp(&q_dedup, &dens_dedup, x))
        .collect();
    let integral = trapz(&dens, argvals);
    if integral < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "signed_quantile_average",
            detail: "reconstructed density integrates to zero".to_string(),
        });
    }
    Ok(dens.iter().map(|&d| d / integral).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(m: usize, lb: f64, ub: f64) -> Vec<f64> {
        (0..m)
            .map(|j| lb + (ub - lb) * j as f64 / (m - 1) as f64)
            .collect()
    }

    fn gaussian(argvals: &[f64], mu: f64) -> Vec<f64> {
        let raw: Vec<f64> = argvals
            .iter()
            .map(|&x| (-(x - mu).powi(2) / 2.0).exp())
            .collect();
        let integral = trapz(&raw, argvals);
        raw.iter().map(|&d| d / integral).collect()
    }

    #[test]
    fn space_new_validates_grid() {
        assert!(WassersteinDensitySpace::new(uniform_grid(50, -5.0, 5.0)).is_ok());
        assert!(matches!(
            WassersteinDensitySpace::new(vec![0.0, 1.0, 0.5]).unwrap_err(),
            FdarError::InvalidParameter { parameter, .. } if parameter == "argvals"
        ));
        assert!(matches!(
            WassersteinDensitySpace::new(vec![0.0]).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
    }

    #[test]
    fn w2_identical_is_zero() {
        let argvals = uniform_grid(101, -5.0, 5.0);
        let d = gaussian(&argvals, 0.0);
        let w2 = wasserstein2_distance(&d, &d, &argvals).unwrap();
        assert!(w2 < 1e-8, "w2 = {w2}");
    }

    #[test]
    fn w2_matches_location_shift() {
        // For a location family, W₂ between N(0,1) and N(δ,1) equals δ.
        let argvals = uniform_grid(201, -8.0, 8.0);
        let d0 = gaussian(&argvals, 0.0);
        let d1 = gaussian(&argvals, 0.5);
        let w2 = wasserstein2_distance(&d0, &d1, &argvals).unwrap();
        assert!((w2 - 0.5).abs() < 0.05, "w2 = {w2}");
    }

    #[test]
    fn distance_delegates_to_w2() {
        let argvals = uniform_grid(101, -5.0, 5.0);
        let space = WassersteinDensitySpace::new(argvals.clone()).unwrap();
        let d0 = gaussian(&argvals, 0.0);
        let d1 = gaussian(&argvals, 0.3);
        let via_trait = space.distance(&d0, &d1).unwrap();
        let direct = wasserstein2_distance(&d0, &d1, &argvals).unwrap();
        assert!((via_trait - direct).abs() < 1e-12);
    }

    #[test]
    fn weighted_frechet_mean_of_identical_recovers_object() {
        let argvals = uniform_grid(101, -5.0, 5.0);
        let space = WassersteinDensitySpace::new(argvals.clone()).unwrap();
        let d = gaussian(&argvals, 0.0);
        let objects = vec![d.clone(), d.clone(), d.clone()];
        let weights = vec![1.0 / 3.0; 3];
        let mean = space.weighted_frechet_mean(&objects, &weights).unwrap();
        // The mean is the Wasserstein barycenter; its density→quantile→density
        // reconstruction (reused from DENS-01) has an inherent ~0.1 W₂ round-trip
        // floor, so recovery is within that documented tolerance, not machine eps.
        let w2 = wasserstein2_distance(&mean, &d, &argvals).unwrap();
        assert!(w2 < 0.15, "w2 = {w2}");
        // Exact agreement with DENS-01's Wasserstein mean (same underlying call).
        let bary = wasserstein_barycenter(
            &{
                let mut m = FdMatrix::zeros(3, argvals.len());
                for i in 0..3 {
                    for j in 0..argvals.len() {
                        m[(i, j)] = d[j];
                    }
                }
                m
            },
            &argvals,
            Some(&weights),
        )
        .unwrap();
        assert_eq!(mean, bary);
    }

    #[test]
    fn w2_rejects_length_mismatch() {
        let argvals = uniform_grid(50, -5.0, 5.0);
        let a = vec![0.0; 50];
        let b = vec![0.0; 49];
        assert!(matches!(
            wasserstein2_distance(&a, &b, &argvals).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
    }

    #[test]
    fn signed_quantile_average_uniform_weights_recovers_true_barycenter() {
        // With uniform non-negative weights, the signed quantile average of two
        // unit Gaussians at ±1 is the true Wasserstein barycenter N(0,1) (the
        // location-family barycenter is the mean-location Gaussian).
        let argvals = uniform_grid(101, -5.0, 5.0);
        let d0 = gaussian(&argvals, -1.0);
        let d1 = gaussian(&argvals, 1.0);
        let mut mat = FdMatrix::zeros(2, argvals.len());
        for j in 0..argvals.len() {
            mat[(0, j)] = d0[j];
            mat[(1, j)] = d1[j];
        }
        let w = vec![0.5, 0.5];
        let n_q = argvals.len().max(101);
        let signed = signed_quantile_average(&mat, &argvals, &w, n_q).unwrap();
        let truth = gaussian(&argvals, 0.0);
        let diff = wasserstein2_distance(&signed, &truth, &argvals).unwrap();
        assert!(diff < 0.15, "diff = {diff}");
    }

    #[test]
    fn signed_quantile_average_accepts_negative_weights() {
        // Negative weights must NOT error (the whole point of this helper).
        let argvals = uniform_grid(101, -6.0, 6.0);
        let d0 = gaussian(&argvals, -1.0);
        let d1 = gaussian(&argvals, 0.0);
        let d2 = gaussian(&argvals, 1.0);
        let mut mat = FdMatrix::zeros(3, argvals.len());
        for j in 0..argvals.len() {
            mat[(0, j)] = d0[j];
            mat[(1, j)] = d1[j];
            mat[(2, j)] = d2[j];
        }
        let w = vec![-0.2, 1.4, -0.2]; // sums to 1, has negatives
        let n_q = argvals.len().max(101);
        let res = signed_quantile_average(&mat, &argvals, &w, n_q).unwrap();
        assert_eq!(res.len(), argvals.len());
        assert!(res.iter().all(|v| v.is_finite() && *v >= -1e-9));
    }
}
