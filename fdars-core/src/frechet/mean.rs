//! Sample Fréchet mean and variance over any [`MetricSpace`].

use crate::error::FdarError;
use crate::frechet::space::MetricSpace;
use crate::helpers::NUMERICAL_EPS;

/// Resolve and normalize weights to sum 1 (uniform when `None`).
fn resolve_weights(n: usize, weights: Option<&[f64]>) -> Result<Vec<f64>, FdarError> {
    match weights {
        None => Ok(vec![1.0 / n as f64; n]),
        Some(w) => {
            if w.len() != n {
                return Err(FdarError::InvalidDimension {
                    parameter: "weights",
                    expected: format!("{n} weights"),
                    actual: format!("{} weights", w.len()),
                });
            }
            let s: f64 = w.iter().sum();
            if s.abs() < NUMERICAL_EPS {
                return Err(FdarError::InvalidParameter {
                    parameter: "weights",
                    message: "weights sum to zero".to_string(),
                });
            }
            Ok(w.iter().map(|&wi| wi / s).collect())
        }
    }
}

/// Fréchet mean (weighted barycenter) of a sample of metric-space objects.
///
/// Delegates to the space's [`MetricSpace::weighted_frechet_mean`]. With `None`
/// weights this is the unweighted sample Fréchet mean.
///
/// # Errors
/// Returns [`FdarError`] for an empty sample, a weight/length mismatch, or
/// weights summing to zero, and propagates the solver's errors.
#[must_use = "expensive Fréchet mean computation — store or use the returned object"]
pub fn frechet_mean<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    weights: Option<&[f64]>,
) -> Result<S::Object, FdarError> {
    if objects.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "objects",
            expected: "at least 1 object".to_string(),
            actual: "0 objects".to_string(),
        });
    }
    let w = resolve_weights(objects.len(), weights)?;
    space.weighted_frechet_mean(objects, &w)
}

/// Fréchet variance: the (weighted) mean squared distance to a given Fréchet mean.
///
/// `V̂ = Σᵢ wᵢ · d²(objectsᵢ, mean)` (Dubey & Müller 2019). Pass the mean returned
/// by [`frechet_mean`].
///
/// # Errors
/// Returns [`FdarError`] for an empty sample, a weight/length mismatch, or
/// weights summing to zero, and propagates the distance's errors.
pub fn frechet_variance<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    mean: &S::Object,
    weights: Option<&[f64]>,
) -> Result<f64, FdarError> {
    if objects.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "objects",
            expected: "at least 1 object".to_string(),
            actual: "0 objects".to_string(),
        });
    }
    let w = resolve_weights(objects.len(), weights)?;
    let mut var = 0.0;
    for (i, obj) in objects.iter().enumerate() {
        let d = space.distance(obj, mean)?;
        var += w[i] * d * d;
    }
    Ok(var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frechet::space::{wasserstein2_distance, WassersteinDensitySpace};
    use crate::helpers::trapz;

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
    fn mean_of_identical_recovers_object() {
        let argvals = uniform_grid(101, -5.0, 5.0);
        let space = WassersteinDensitySpace::new(argvals.clone()).unwrap();
        let d = gaussian(&argvals, 0.0);
        let objects = vec![d.clone(), d.clone(), d.clone()];
        let mean = frechet_mean(&space, &objects, None).unwrap();
        // Recovery within the barycenter reconstruction tolerance (see space.rs).
        let w2 = wasserstein2_distance(&mean, &d, &argvals).unwrap();
        assert!(w2 < 0.15, "w2 = {w2}");
    }

    #[test]
    fn variance_zero_for_identical_sample() {
        let argvals = uniform_grid(101, -5.0, 5.0);
        let space = WassersteinDensitySpace::new(argvals.clone()).unwrap();
        let d = gaussian(&argvals, 0.0);
        let objects = vec![d.clone(), d.clone(), d.clone(), d.clone()];
        let mean = frechet_mean(&space, &objects, None).unwrap();
        // "Zero within tolerance": the only residual is the barycenter's
        // reconstruction floor (~0.09 W₂ → ~0.008 squared), far below the
        // dispersion-driven variances (order 1+) the ANOVA/regression rely on.
        let var = frechet_variance(&space, &objects, &mean, None).unwrap();
        assert!(var < 0.02, "var = {var}");
    }

    #[test]
    fn variance_grows_with_dispersion() {
        let argvals = uniform_grid(201, -8.0, 8.0);
        let space = WassersteinDensitySpace::new(argvals.clone()).unwrap();
        let tight = vec![
            gaussian(&argvals, -0.5),
            gaussian(&argvals, 0.0),
            gaussian(&argvals, 0.5),
        ];
        let wide = vec![
            gaussian(&argvals, -2.0),
            gaussian(&argvals, 0.0),
            gaussian(&argvals, 2.0),
        ];
        let m_tight = frechet_mean(&space, &tight, None).unwrap();
        let m_wide = frechet_mean(&space, &wide, None).unwrap();
        let v_tight = frechet_variance(&space, &tight, &m_tight, None).unwrap();
        let v_wide = frechet_variance(&space, &wide, &m_wide, None).unwrap();
        assert!(v_wide > v_tight, "v_wide = {v_wide}, v_tight = {v_tight}");
    }

    #[test]
    fn rejects_bad_input() {
        let argvals = uniform_grid(50, -5.0, 5.0);
        let space = WassersteinDensitySpace::new(argvals.clone()).unwrap();
        let empty: Vec<Vec<f64>> = vec![];
        assert!(matches!(
            frechet_mean(&space, &empty, None).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
        let objects = vec![gaussian(&argvals, 0.0), gaussian(&argvals, 1.0)];
        // weight length mismatch
        assert!(matches!(
            frechet_mean(&space, &objects, Some(&[1.0])).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
        // zero-sum weights
        assert!(matches!(
            frechet_mean(&space, &objects, Some(&[0.0, 0.0])).unwrap_err(),
            FdarError::InvalidParameter { .. }
        ));
    }
}
