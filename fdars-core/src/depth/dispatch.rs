//! Unified depth dispatcher and depth-fence functional boxplot.
//!
//! This module provides a single [`functional_depth`] entry point that computes
//! the **self-depth** of a sample (each curve's depth with respect to the sample
//! itself, i.e. `data_obj == data_ori`) by dispatching to the existing depth
//! functions via the [`DepthMethod`] selector.
//!
//! The dispatcher only *wraps* the underlying depth functions — their signatures
//! are unchanged.

use crate::depth::{band_1d, fraiman_muniz_1d, modified_band_1d, random_projection_1d_seeded};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Depth measure selector for [`functional_depth`].
///
/// Each variant maps 1:1 to an existing self-depth call:
/// - `FraimanMuniz { scale }` → `fraiman_muniz_1d(data, data, scale)`
/// - `Band` → `band_1d(data, data)`
/// - `ModifiedBand` → `modified_band_1d(data, data)`
/// - `RandomProjection { nproj, seed }` → `random_projection_1d_seeded(data, data, nproj, Some(seed))`
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DepthMethod {
    /// Fraiman-Muniz depth. `scale` toggles the scaled `2·min(Fn, 1−Fn)` form.
    FraimanMuniz {
        /// Whether to scale the depth values.
        scale: bool,
    },
    /// Band depth (BD).
    Band,
    /// Modified band depth (MBD).
    ModifiedBand,
    /// Random projection depth. `seed` makes results bit-reproducible.
    RandomProjection {
        /// Number of random projection directions.
        nproj: usize,
        /// RNG seed for deterministic projections.
        seed: u64,
    },
}

/// Compute the **self-depth** of every curve in `data` w.r.t. the sample.
///
/// Passes `data` as both the object and reference matrix and dispatches to the
/// underlying depth function selected by `method`. Returns one depth per curve
/// (`Vec<f64>` of length `data.nrows()`).
///
/// # Errors
/// - `InvalidDimension` if `data` has zero rows or zero columns, or if
///   `Band`/`ModifiedBand` is requested with fewer than 2 curves (a band needs
///   two reference curves).
/// - `InvalidParameter` if `RandomProjection { nproj: 0, .. }` is requested.
pub fn functional_depth(data: &FdMatrix, method: DepthMethod) -> Result<Vec<f64>, FdarError> {
    let (n, m) = (data.nrows(), data.ncols());
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (nrows > 0 and ncols > 0)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }

    let depths = match method {
        DepthMethod::FraimanMuniz { scale } => fraiman_muniz_1d(data, data, scale),
        DepthMethod::Band => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for band depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            band_1d(data, data)
        }
        DepthMethod::ModifiedBand => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for modified band depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            modified_band_1d(data, data)
        }
        DepthMethod::RandomProjection { nproj, seed } => {
            if nproj == 0 {
                return Err(FdarError::InvalidParameter {
                    parameter: "nproj",
                    message: "must be >= 1".to_string(),
                });
            }
            random_projection_1d_seeded(data, data, nproj, Some(seed))
        }
    };

    Ok(depths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic sample: `n` mild sinusoids on an `m`-point grid.
    fn sample(n: usize, m: usize) -> FdMatrix {
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                // element (i, t) at index i + t*n (column-major)
                col_major[i + t * n] = (x * std::f64::consts::PI).sin() + 0.05 * i as f64;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    #[test]
    fn fraiman_muniz_dispatch_equals_underlying() {
        let data = sample(6, 12);
        for scale in [true, false] {
            let got = functional_depth(&data, DepthMethod::FraimanMuniz { scale }).unwrap();
            let want = fraiman_muniz_1d(&data, &data, scale);
            assert_eq!(got, want);
            assert_eq!(got.len(), data.nrows());
        }
    }

    #[test]
    fn band_dispatch_equals_underlying() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::Band).unwrap();
        assert_eq!(got, band_1d(&data, &data));
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn modified_band_dispatch_equals_underlying() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::ModifiedBand).unwrap();
        assert_eq!(got, modified_band_1d(&data, &data));
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn random_projection_dispatch_equals_underlying_and_is_reproducible() {
        let data = sample(6, 12);
        let method = DepthMethod::RandomProjection {
            nproj: 20,
            seed: 42,
        };
        let got = functional_depth(&data, method).unwrap();
        let want = random_projection_1d_seeded(&data, &data, 20, Some(42));
        assert_eq!(got, want);
        // Two dispatch calls with the same seed are bit-identical.
        let got2 = functional_depth(&data, method).unwrap();
        assert_eq!(got, got2);
    }

    #[test]
    fn empty_matrix_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(functional_depth(&empty, DepthMethod::FraimanMuniz { scale: true }).is_err());
    }

    #[test]
    fn too_few_curves_for_band_returns_err() {
        let one = sample(1, 8);
        assert!(functional_depth(&one, DepthMethod::Band).is_err());
        assert!(functional_depth(&one, DepthMethod::ModifiedBand).is_err());
    }

    #[test]
    fn zero_nproj_returns_err() {
        let data = sample(6, 12);
        assert!(
            functional_depth(&data, DepthMethod::RandomProjection { nproj: 0, seed: 1 }).is_err()
        );
    }
}
