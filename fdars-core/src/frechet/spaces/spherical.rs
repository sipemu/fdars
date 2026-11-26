//! Spherical-data `MetricSpace` backend (FRE-02-03).
//!
//! Objects are unit vectors on the sphere `Sᵈ⁻¹`, stored as `Vec<f64>` of length
//! `d`. The metric is the geodesic (great-circle) distance
//! `arccos(clamp(⟨a,b⟩, −1, 1))`; the weighted Fréchet mean is the intrinsic
//! (Karcher) mean, computed by Riemannian gradient descent using the exponential
//! and logarithm maps, initialized at the normalized extrinsic (weighted-average)
//! mean.
//!
//! Callers must supply unit vectors; unit-norm is not re-checked per call for
//! performance. Antipodally-balanced inputs (whose extrinsic mean is the zero
//! vector, or whose Karcher log map hits an antipode) return an error rather than
//! an ill-defined mean.
//!
//! # Divergence from R `frechet` 0.3.0
//!
//! R's spherical Fréchet mean uses its own initialization and stopping rule; this
//! backend uses an extrinsic-mean initialization with `max_iter = 50` and
//! `tol = 1e-8`. The geodesic geometry is identical; iterates may differ at the
//! last few digits.

use super::check_mean_inputs;
use crate::error::FdarError;
use crate::frechet::MetricSpace;

const MAX_ITER: usize = 50;
const TOL: f64 = 1e-8;

/// Spherical-data response space on `Sᵈ⁻¹` (FRE-02-03).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SphericalSpace {
    /// Ambient dimension `d` (objects are unit vectors of length `d` on `Sᵈ⁻¹`).
    pub d: usize,
}

impl SphericalSpace {
    /// Construct a spherical space of ambient dimension `d`.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `d < 1`.
    pub fn new(d: usize) -> Result<Self, FdarError> {
        if d < 1 {
            return Err(FdarError::InvalidParameter {
                parameter: "d",
                message: "ambient dimension must be >= 1".to_string(),
            });
        }
        Ok(Self { d })
    }

    fn check_len(&self, obj: &[f64], name: &'static str) -> Result<(), FdarError> {
        if obj.len() != self.d {
            return Err(FdarError::InvalidDimension {
                parameter: name,
                expected: format!("{} elements", self.d),
                actual: format!("{} elements", obj.len()),
            });
        }
        Ok(())
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Geodesic (great-circle) distance between two unit vectors.
fn geodesic_distance(a: &[f64], b: &[f64]) -> f64 {
    dot(a, b).clamp(-1.0, 1.0).acos()
}

/// Exponential map: move from `x` along tangent `v`, staying on the sphere.
fn exp_map(x: &[f64], v: &[f64]) -> Vec<f64> {
    let nv = norm(v);
    if nv < 1e-12 {
        return x.to_vec();
    }
    let c = nv.cos();
    let s = nv.sin() / nv;
    x.iter()
        .zip(v.iter())
        .map(|(xi, vi)| c * xi + s * vi)
        .collect()
}

/// Logarithm map: tangent vector at `x` pointing toward `y`.
fn log_map(x: &[f64], y: &[f64]) -> Result<Vec<f64>, FdarError> {
    let theta = dot(x, y).clamp(-1.0, 1.0).acos();
    if theta < 1e-12 {
        return Ok(vec![0.0; x.len()]);
    }
    if theta > std::f64::consts::PI - 1e-8 {
        return Err(FdarError::ComputationFailed {
            operation: "SphericalSpace::weighted_frechet_mean",
            detail: "antipodal points have a non-unique logarithm; Karcher mean is undefined"
                .to_string(),
        });
    }
    let scale = theta / theta.sin();
    let ct = theta.cos();
    Ok(x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| scale * (yi - ct * xi))
        .collect())
}

/// Divide `x` by its norm in place; errors with `detail` if the norm is ~0.
fn normalize_or_err(x: &mut [f64], detail: &str) -> Result<(), FdarError> {
    let nx = norm(x);
    if nx < 1e-14 {
        return Err(FdarError::ComputationFailed {
            operation: "SphericalSpace::weighted_frechet_mean",
            detail: detail.to_string(),
        });
    }
    for xk in x.iter_mut() {
        *xk /= nx;
    }
    Ok(())
}

/// Extrinsic initialization: normalized weighted average of the objects.
fn extrinsic_init(objects: &[Vec<f64>], weights: &[f64], d: usize) -> Result<Vec<f64>, FdarError> {
    let mut x = vec![0.0f64; d];
    for (o, &w) in objects.iter().zip(weights.iter()) {
        for (k, xk) in x.iter_mut().enumerate() {
            *xk += w * o[k];
        }
    }
    normalize_or_err(
        &mut x,
        "extrinsic mean is ~0 (antipodally-balanced input); Karcher mean is undefined",
    )?;
    Ok(x)
}

/// Riemannian gradient at `x`: `sum_i w_i log_x(o_i)`.
fn weighted_log_sum(
    x: &[f64],
    objects: &[Vec<f64>],
    weights: &[f64],
) -> Result<Vec<f64>, FdarError> {
    let mut g = vec![0.0f64; x.len()];
    for (o, &w) in objects.iter().zip(weights.iter()) {
        let lm = log_map(x, o)?;
        for (k, gk) in g.iter_mut().enumerate() {
            *gk += w * lm[k];
        }
    }
    Ok(g)
}

impl MetricSpace for SphericalSpace {
    type Object = Vec<f64>;

    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError> {
        self.check_len(a, "a")?;
        self.check_len(b, "b")?;
        Ok(geodesic_distance(a, b))
    }

    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError> {
        check_mean_inputs(objects, weights, self.d)?;
        let mut x = extrinsic_init(objects, weights, self.d)?;
        // Riemannian gradient descent.
        for _ in 0..MAX_ITER {
            let g = weighted_log_sum(&x, objects, weights)?;
            if norm(&g) < TOL {
                return Ok(x);
            }
            x = exp_map(&x, &g);
            normalize_or_err(&mut x, "Karcher iterate collapsed to the origin")?;
        }
        Err(FdarError::ComputationFailed {
            operation: "SphericalSpace::weighted_frechet_mean",
            detail: "Karcher mean did not converge in 50 iterations".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn spherical_geodesic_antipodal_is_pi() {
        let s = SphericalSpace::new(2).unwrap();
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((s.distance(&a, &b).unwrap() - PI).abs() < 1e-12);
    }

    #[test]
    fn spherical_geodesic_identical_is_zero() {
        let s = SphericalSpace::new(3).unwrap();
        let a = vec![0.0, 1.0, 0.0];
        assert!(s.distance(&a, &a).unwrap() < 1e-12);
    }

    #[test]
    fn spherical_karcher_midpoint() {
        let s = SphericalSpace::new(2).unwrap();
        let a = vec![1.0, 0.0];
        let b = vec![0.1f64.cos(), 0.1f64.sin()];
        let m = s.weighted_frechet_mean(&[a, b], &[0.5, 0.5]).unwrap();
        let expected = [0.05f64.cos(), 0.05f64.sin()];
        for (x, y) in m.iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-6, "x={x} y={y}");
        }
    }

    #[test]
    fn spherical_karcher_of_identical_recovers() {
        let s = SphericalSpace::new(3).unwrap();
        let a = {
            let raw = [0.3f64, -0.4, 0.5];
            let n = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            raw.iter().map(|x| x / n).collect::<Vec<_>>()
        };
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone(), a.clone()], &[0.2, 0.3, 0.5])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-8, "x={x} y={y}");
        }
    }

    #[test]
    fn spherical_karcher_antipodal_balanced_fails() {
        let s = SphericalSpace::new(2).unwrap();
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(matches!(
            s.weighted_frechet_mean(&[a, b], &[0.5, 0.5]),
            Err(FdarError::ComputationFailed { .. })
        ));
    }

    #[test]
    fn spherical_rejects_dimension_mismatch() {
        let s = SphericalSpace::new(2).unwrap();
        let a = vec![1.0, 0.0];
        let bad = vec![1.0, 0.0, 0.0];
        assert!(matches!(
            s.distance(&a, &bad),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    /// Refactor pin (#75): Karcher means on pseudo-random clustered S^2 data
    /// stay bit-identical to values recorded on the pre-refactor code.
    #[test]
    fn spherical_karcher_bit_identity_pin() {
        use crate::test_helpers::{bits_fingerprint, lcg_values};
        let s = SphericalSpace::new(3).unwrap();
        let objs: Vec<Vec<f64>> = (0..7)
            .map(|k| {
                let r = lcg_values(3, 200 + k);
                let raw = [1.0 + 0.3 * r[0], 0.3 * r[1], 0.3 * r[2]];
                let n = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
                raw.iter().map(|x| x / n).collect()
            })
            .collect();
        let mut all = Vec::new();
        for seed in 0..5u64 {
            let w: Vec<f64> = lcg_values(7, 300 + seed)
                .iter()
                .map(|v| (v + 1.5) / 7.0)
                .collect();
            all.extend(s.weighted_frechet_mean(&objs, &w).unwrap());
        }
        let fp = bits_fingerprint(&all);
        assert_eq!(fp, 12_241_966_407_315_560_618);
    }
}
