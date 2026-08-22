//! SPD covariance-matrix `MetricSpace` backend (FRE-02-01).
//!
//! Objects are symmetric positive-definite (SPD) matrices stored as flat
//! column-major `Vec<f64>` of length `d*d` (element `(i, j)` at index `i + j*d`).
//! The metric is selected via [`SpdMetric`]:
//!
//! * **Frobenius** — Euclidean distance on the flat matrices; weighted Fréchet
//!   mean is the weighted average (SPD-closed under non-negative weights).
//! * **Power-α** — distance `‖Aᵅ − Bᵅ‖_F / α` (matrix power via
//!   [`nalgebra::SymmetricEigen`]); weighted mean `(Σ wᵢ Aᵢᵅ / Σ wᵢ)^{1/α}`.
//!   `α = 1` reproduces the Frobenius metric exactly; smaller α approaches the
//!   log-Euclidean geometry.
//! * **Log-Cholesky** — distance/mean in log-Cholesky coordinates (strictly-lower
//!   Cholesky entries as-is, log of the diagonal), then mapped back.
//!
//! # Divergence from R `frechet` 0.3.0
//!
//! R's SPD covariance-response geometry uses an affine-invariant (log-Euclidean)
//! metric family; this backend offers the Frobenius / power-α / log-Cholesky
//! family instead. Numeric results therefore differ; the capability (distance +
//! weighted Fréchet mean per metric) matches.

use crate::error::FdarError;
use crate::frechet::MetricSpace;
use crate::helpers::NUMERICAL_EPS;
use nalgebra::DMatrix;

/// The metric used by an [`SpdMatrixSpace`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpdMetric {
    /// Element-wise Frobenius distance; weighted-average mean.
    Frobenius,
    /// Power-α metric with `α > 0`. `α = 1` ≡ Frobenius.
    Power(f64),
    /// Log-Cholesky metric.
    LogCholesky,
}

/// SPD covariance-matrix response space (FRE-02-01).
///
/// Objects are `d×d` SPD matrices as flat column-major `Vec<f64>`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpdMatrixSpace {
    /// Matrix dimension `d` (objects are `d*d` flat vectors).
    pub d: usize,
    /// The metric on the SPD manifold.
    pub metric: SpdMetric,
}

impl SpdMatrixSpace {
    /// Construct an SPD matrix space of dimension `d` under `metric`.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `d < 1`, or if `metric` is
    /// `Power(α)` with `α <= 0`.
    pub fn new(d: usize, metric: SpdMetric) -> Result<Self, FdarError> {
        if d < 1 {
            return Err(FdarError::InvalidParameter {
                parameter: "d",
                message: "matrix dimension must be >= 1".to_string(),
            });
        }
        if let SpdMetric::Power(alpha) = metric {
            if alpha <= 0.0 || alpha.is_nan() {
                return Err(FdarError::InvalidParameter {
                    parameter: "alpha",
                    message: "power-metric exponent must be > 0".to_string(),
                });
            }
        }
        Ok(Self { d, metric })
    }

    fn check_len(&self, obj: &[f64], name: &'static str) -> Result<(), FdarError> {
        let dd = self.d * self.d;
        if obj.len() != dd {
            return Err(FdarError::InvalidDimension {
                parameter: name,
                expected: format!("{dd} elements (d*d)"),
                actual: format!("{} elements", obj.len()),
            });
        }
        Ok(())
    }

    fn validate_objects(&self, objects: &[Vec<f64>], weights: &[f64]) -> Result<(), FdarError> {
        if objects.is_empty() {
            return Err(FdarError::InvalidDimension {
                parameter: "objects",
                expected: "at least 1 object".to_string(),
                actual: "0 objects".to_string(),
            });
        }
        if weights.len() != objects.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "weights",
                expected: format!("{} weights (matching objects)", objects.len()),
                actual: format!("{} weights", weights.len()),
            });
        }
        for (i, o) in objects.iter().enumerate() {
            if o.len() != self.d * self.d {
                return Err(FdarError::InvalidDimension {
                    parameter: "objects",
                    expected: format!("each object has {} elements", self.d * self.d),
                    actual: format!("object {i} has {} elements", o.len()),
                });
            }
        }
        Ok(())
    }
}

/// Frobenius norm of the element-wise difference of two equal-length vectors.
fn frobenius_norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Normalized weighted average `Σ wᵢ oᵢ / Σ wᵢ` of equal-length vectors.
fn weighted_average(
    objects: &[Vec<f64>],
    weights: &[f64],
    len: usize,
) -> Result<Vec<f64>, FdarError> {
    let sw: f64 = weights.iter().sum();
    if sw.abs() < NUMERICAL_EPS {
        return Err(FdarError::ComputationFailed {
            operation: "SpdMatrixSpace::weighted_frechet_mean",
            detail: "sum of weights is ~0; cannot normalize the barycenter".to_string(),
        });
    }
    let mut acc = vec![0.0f64; len];
    for (o, &w) in objects.iter().zip(weights.iter()) {
        for (k, ak) in acc.iter_mut().enumerate() {
            *ak += w * o[k];
        }
    }
    for x in &mut acc {
        *x /= sw;
    }
    Ok(acc)
}

/// Matrix power `A^α` of a symmetric matrix via eigendecomposition
/// (eigenvalues clamped to `>= 0` before `powf`). Returns flat column-major `d×d`.
fn spd_power(mat_flat: &[f64], d: usize, alpha: f64) -> Vec<f64> {
    let mut mat = DMatrix::from_column_slice(d, d, mat_flat);
    for i in 0..d {
        for j in (i + 1)..d {
            let avg = 0.5 * (mat[(i, j)] + mat[(j, i)]);
            mat[(i, j)] = avg;
            mat[(j, i)] = avg;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(mat);
    let mut result = vec![0.0f64; d * d];
    for k in 0..d {
        let lk = eig.eigenvalues[k].max(0.0).powf(alpha);
        if lk == 0.0 {
            continue;
        }
        for i in 0..d {
            let vik = eig.eigenvectors[(i, k)];
            for j in 0..d {
                result[i + j * d] += vik * lk * eig.eigenvectors[(j, k)];
            }
        }
    }
    result
}

/// Log-Cholesky coordinates: strictly-lower Cholesky entries (row-major, `i>j`)
/// followed by the log of the diagonal. Length `d*(d+1)/2`.
fn log_cholesky_coords(mat_flat: &[f64], d: usize) -> Result<Vec<f64>, FdarError> {
    // cholesky_factor takes a symmetric flat matrix; column-major == row-major here.
    let l = crate::linalg::cholesky_factor(mat_flat, d)?; // row-major L: l[i*d + j]
    let mut coords = Vec::with_capacity(d * (d + 1) / 2);
    for i in 0..d {
        for j in 0..i {
            coords.push(l[i * d + j]);
        }
    }
    for i in 0..d {
        coords.push(l[i * d + i].ln());
    }
    Ok(coords)
}

/// Inverse of [`log_cholesky_coords`]: rebuild `L` then form `M = L Lᵀ`
/// (flat column-major `d×d`).
fn log_cholesky_reconstruct(coords: &[f64], d: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; d * d]; // row-major L
    let mut idx = 0;
    for i in 0..d {
        for j in 0..i {
            l[i * d + j] = coords[idx];
            idx += 1;
        }
    }
    for i in 0..d {
        l[i * d + i] = coords[idx].exp();
        idx += 1;
    }
    let mut result = vec![0.0f64; d * d];
    for i in 0..d {
        for k in 0..d {
            let mut s = 0.0;
            for j in 0..=i.min(k) {
                s += l[i * d + j] * l[k * d + j];
            }
            result[i + k * d] = s;
        }
    }
    result
}

impl MetricSpace for SpdMatrixSpace {
    type Object = Vec<f64>;

    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError> {
        self.check_len(a, "a")?;
        self.check_len(b, "b")?;
        match self.metric {
            SpdMetric::Frobenius => Ok(frobenius_norm_diff(a, b)),
            SpdMetric::Power(alpha) => {
                let pa = spd_power(a, self.d, alpha);
                let pb = spd_power(b, self.d, alpha);
                Ok(frobenius_norm_diff(&pa, &pb) / alpha)
            }
            SpdMetric::LogCholesky => {
                let ca = log_cholesky_coords(a, self.d)?;
                let cb = log_cholesky_coords(b, self.d)?;
                Ok(frobenius_norm_diff(&ca, &cb))
            }
        }
    }

    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError> {
        self.validate_objects(objects, weights)?;
        let dd = self.d * self.d;
        match self.metric {
            SpdMetric::Frobenius => weighted_average(objects, weights, dd),
            SpdMetric::Power(alpha) => {
                let transformed: Vec<Vec<f64>> = objects
                    .iter()
                    .map(|o| spd_power(o, self.d, alpha))
                    .collect();
                let avg = weighted_average(&transformed, weights, dd)?;
                Ok(spd_power(&avg, self.d, 1.0 / alpha))
            }
            SpdMetric::LogCholesky => {
                let mut coords: Vec<Vec<f64>> = Vec::with_capacity(objects.len());
                for o in objects {
                    coords.push(log_cholesky_coords(o, self.d)?);
                }
                let ncoord = self.d * (self.d + 1) / 2;
                let avg = weighted_average(&coords, weights, ncoord)?;
                Ok(log_cholesky_reconstruct(&avg, self.d))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2×2 SPD helpers (flat column-major).
    fn identity2() -> Vec<f64> {
        vec![1.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn spd_new_rejects_zero_dim() {
        assert!(matches!(
            SpdMatrixSpace::new(0, SpdMetric::Frobenius),
            Err(FdarError::InvalidParameter { parameter: "d", .. })
        ));
        assert!(matches!(
            SpdMatrixSpace::new(2, SpdMetric::Power(0.0)),
            Err(FdarError::InvalidParameter {
                parameter: "alpha",
                ..
            })
        ));
        assert!(matches!(
            SpdMatrixSpace::new(2, SpdMetric::Power(-1.0)),
            Err(FdarError::InvalidParameter {
                parameter: "alpha",
                ..
            })
        ));
    }

    #[test]
    fn spd_frobenius_distance_of_identical_is_zero() {
        let s = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        let a = vec![2.0, 0.5, 0.5, 3.0];
        assert!(s.distance(&a, &a).unwrap() < 1e-12);
    }

    #[test]
    fn spd_frobenius_mean_of_identical_recovers_matrix() {
        let s = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        let a = vec![2.0, 0.5, 0.5, 3.0];
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone(), a.clone()], &[0.2, 0.3, 0.5])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn spd_rejects_dimension_mismatch() {
        let s = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        let a = identity2();
        let bad = vec![1.0, 0.0, 0.0]; // length 3, not 4
        assert!(matches!(
            s.distance(&a, &bad),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            s.weighted_frechet_mean(&[], &[]),
            Err(FdarError::InvalidDimension {
                parameter: "objects",
                ..
            })
        ));
        assert!(matches!(
            s.weighted_frechet_mean(std::slice::from_ref(&a), &[1.0, 2.0]),
            Err(FdarError::InvalidDimension {
                parameter: "weights",
                ..
            })
        ));
    }

    #[test]
    fn spd_power_alpha_one_equals_frobenius() {
        let fro = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        let pow = SpdMatrixSpace::new(2, SpdMetric::Power(1.0)).unwrap();
        let a = vec![2.0, 0.5, 0.5, 3.0];
        let b = vec![1.5, -0.2, -0.2, 2.5];
        let df = fro.distance(&a, &b).unwrap();
        let dp = pow.distance(&a, &b).unwrap();
        assert!((df - dp).abs() < 1e-10, "df={df} dp={dp}");
    }

    #[test]
    fn spd_power_alpha_mean_of_identical_recovers() {
        let s = SpdMatrixSpace::new(2, SpdMetric::Power(0.5)).unwrap();
        let a = vec![2.0, 0.3, 0.3, 4.0];
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone()], &[0.5, 0.5])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-8, "x={x} y={y}");
        }
    }

    #[test]
    fn spd_log_cholesky_mean_identity_and_4i_is_2i() {
        let s = SpdMatrixSpace::new(2, SpdMetric::LogCholesky).unwrap();
        let i2 = vec![1.0, 0.0, 0.0, 1.0];
        let four_i = vec![4.0, 0.0, 0.0, 4.0];
        let m = s.weighted_frechet_mean(&[i2, four_i], &[0.5, 0.5]).unwrap();
        let expected = [2.0, 0.0, 0.0, 2.0];
        for (x, y) in m.iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8, "x={x} y={y}");
        }
    }

    #[test]
    fn spd_log_cholesky_mean_of_identical_recovers() {
        let s = SpdMatrixSpace::new(2, SpdMetric::LogCholesky).unwrap();
        let a = vec![2.0, 0.5, 0.5, 3.0];
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone()], &[0.4, 0.6])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10, "x={x} y={y}");
        }
    }

    #[test]
    fn spd_log_cholesky_rejects_non_pd() {
        let s = SpdMatrixSpace::new(2, SpdMetric::LogCholesky).unwrap();
        // Not positive-definite (zero matrix).
        let bad = vec![0.0, 0.0, 0.0, 0.0];
        let good = identity2();
        assert!(matches!(
            s.distance(&bad, &good),
            Err(FdarError::ComputationFailed { .. })
        ));
    }
}
