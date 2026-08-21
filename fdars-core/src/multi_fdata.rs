//! Multi-domain functional data container.
//!
//! [`MultiFunData`] holds several [`FdComponent`] blocks that may live on
//! different domains/grids (the *multi-domain* feature). The only shared
//! invariant across components is that they must all have the same number of
//! observations (rows). Each component carries its own evaluation grid
//! (`argvals`), whose length must equal the number of columns in its
//! [`FdMatrix`].
//!
//! This mirrors the `funData::multiFunData` capability from the R `funData`
//! package (REP-01 SC2).
//!
//! # Examples
//!
//! ```
//! use fdars_core::matrix::FdMatrix;
//! use fdars_core::multi_fdata::{FdComponent, MultiFunData};
//!
//! // Two components: 5 observations on different grids (10-point and 4-point).
//! let data1 = FdMatrix::zeros(5, 10);
//! let argvals1: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
//! let comp1 = FdComponent { data: data1, argvals: argvals1 };
//!
//! let data2 = FdMatrix::zeros(5, 4);
//! let argvals2: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
//! let comp2 = FdComponent { data: data2, argvals: argvals2 };
//!
//! let mfd = MultiFunData::new(vec![comp1, comp2]).unwrap();
//! assert_eq!(mfd.n_obs(), 5);
//! assert_eq!(mfd.n_components(), 2);
//! ```

use crate::{matrix::FdMatrix, FdarError};

/// A single component of a [`MultiFunData`] object.
///
/// Bundles an [`FdMatrix`] (rows = observations, columns = evaluation points)
/// with its evaluation grid `argvals`. The invariant `argvals.len() ==
/// data.ncols()` is enforced by [`MultiFunData::new`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FdComponent {
    /// Functional data matrix (column-major; rows = observations, columns =
    /// evaluation points on this component's domain).
    pub data: FdMatrix,
    /// Evaluation grid for this component. Must satisfy `argvals.len() ==
    /// data.ncols()`.
    pub argvals: Vec<f64>,
}

/// Multi-domain functional data container.
///
/// Holds several [`FdComponent`] blocks that may live on **different** domains
/// (grids, lengths). The invariant shared across all components is that they
/// must all record the same number of observations (`data.nrows()`). Each
/// component keeps its own evaluation grid so that multi-domain data (e.g.
/// temperature + precipitation observed at different densities) can coexist
/// under a single object.
///
/// Constructed via [`MultiFunData::new`], which validates both invariants:
/// 1. All components must have `data.nrows() == n_obs` (equal observation count).
/// 2. Each component must have `argvals.len() == data.ncols()`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::multi_fdata::{FdComponent, MultiFunData};
///
/// let n = 8;
/// let comp_a = FdComponent {
///     data: FdMatrix::zeros(n, 20),
///     argvals: (0..20).map(|i| i as f64).collect(),
/// };
/// let comp_b = FdComponent {
///     data: FdMatrix::zeros(n, 5),
///     argvals: vec![0.0, 0.25, 0.5, 0.75, 1.0],
/// };
/// let mfd = MultiFunData::new(vec![comp_a, comp_b]).unwrap();
/// assert_eq!(mfd.n_obs(), n);
/// assert_eq!(mfd.n_components(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiFunData {
    components: Vec<FdComponent>,
}

impl MultiFunData {
    /// Construct a validated multi-domain functional data container.
    ///
    /// # Invariants enforced
    ///
    /// 1. `components` must be non-empty.
    /// 2. All components must share the same observation count (`data.nrows()`).
    /// 3. Each component must satisfy `argvals.len() == data.ncols()`.
    ///
    /// # Errors
    ///
    /// - [`FdarError::InvalidParameter`] — `components` is empty.
    /// - [`FdarError::InvalidDimension`] — observation count mismatch across
    ///   components, or `argvals.len() != data.ncols()` for any component.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    /// use fdars_core::multi_fdata::{FdComponent, MultiFunData};
    ///
    /// let comp = FdComponent {
    ///     data: FdMatrix::zeros(3, 6),
    ///     argvals: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    /// };
    /// let mfd = MultiFunData::new(vec![comp]).unwrap();
    /// assert_eq!(mfd.n_obs(), 3);
    /// ```
    pub fn new(components: Vec<FdComponent>) -> Result<Self, FdarError> {
        if components.is_empty() {
            return Err(FdarError::InvalidParameter {
                parameter: "components",
                message: "MultiFunData requires at least one component".to_string(),
            });
        }

        let n_obs = components[0].data.nrows();

        // Validate argvals length for the first component before iterating.
        if components[0].argvals.len() != components[0].data.ncols() {
            return Err(FdarError::InvalidDimension {
                parameter: "components[0].argvals",
                expected: format!("{}", components[0].data.ncols()),
                actual: format!("{}", components[0].argvals.len()),
            });
        }

        for (k, comp) in components.iter().enumerate().skip(1) {
            // Check observation count consistency.
            if comp.data.nrows() != n_obs {
                return Err(FdarError::InvalidDimension {
                    parameter: "components[k].data.nrows",
                    expected: format!("{n_obs} (same as component 0)"),
                    actual: format!("{} (component {k})", comp.data.nrows()),
                });
            }
            // Check argvals-length vs ncols.
            if comp.argvals.len() != comp.data.ncols() {
                return Err(FdarError::InvalidDimension {
                    parameter: "components[k].argvals",
                    expected: format!("{} (data.ncols for component {k})", comp.data.ncols()),
                    actual: format!("{}", comp.argvals.len()),
                });
            }
        }

        Ok(Self { components })
    }

    /// Number of observations shared by all components.
    ///
    /// # Panics
    ///
    /// Never — `components` is always non-empty after [`MultiFunData::new`].
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.components[0].data.nrows()
    }

    /// Number of components.
    #[inline]
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Return a reference to the `k`-th component.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `k >= n_components()`.
    pub fn component(&self, k: usize) -> Result<&FdComponent, FdarError> {
        if k >= self.components.len() {
            return Err(FdarError::InvalidParameter {
                parameter: "k",
                message: format!(
                    "component index {k} out of range (n_components = {})",
                    self.components.len()
                ),
            });
        }
        Ok(&self.components[k])
    }

    /// Return a reference to the evaluation grid of the `k`-th component.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `k >= n_components()`.
    pub fn argvals(&self, k: usize) -> Result<&[f64], FdarError> {
        if k >= self.components.len() {
            return Err(FdarError::InvalidParameter {
                parameter: "k",
                message: format!(
                    "argvals index {k} out of range (n_components = {})",
                    self.components.len()
                ),
            });
        }
        Ok(&self.components[k].argvals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    fn make_component(nrows: usize, ncols: usize) -> FdComponent {
        FdComponent {
            data: FdMatrix::zeros(nrows, ncols),
            argvals: (0..ncols).map(|i| i as f64).collect(),
        }
    }

    fn make_component_argvals(nrows: usize, argvals: Vec<f64>) -> FdComponent {
        let ncols = argvals.len();
        FdComponent {
            data: FdMatrix::zeros(nrows, ncols),
            argvals,
        }
    }

    // --- Constructor tests ---

    #[test]
    fn test_two_component_different_grids_ok() {
        // Multi-domain: two components with different ncols (different grids).
        let comp1 = make_component(5, 10);
        let comp2 = make_component(5, 4);
        let mfd = MultiFunData::new(vec![comp1, comp2]).unwrap();
        assert_eq!(mfd.n_obs(), 5);
        assert_eq!(mfd.n_components(), 2);
    }

    #[test]
    fn test_single_component_ok() {
        let comp = make_component(3, 6);
        let mfd = MultiFunData::new(vec![comp]).unwrap();
        assert_eq!(mfd.n_obs(), 3);
        assert_eq!(mfd.n_components(), 1);
    }

    #[test]
    fn test_three_components_same_nrows_ok() {
        let comp1 = make_component(7, 5);
        let comp2 = make_component(7, 10);
        let comp3 = make_component(7, 3);
        let mfd = MultiFunData::new(vec![comp1, comp2, comp3]).unwrap();
        assert_eq!(mfd.n_obs(), 7);
        assert_eq!(mfd.n_components(), 3);
    }

    #[test]
    fn test_empty_components_err() {
        let result = MultiFunData::new(vec![]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_mismatched_nrows_err() {
        let comp1 = make_component(5, 10);
        let comp2 = make_component(4, 10); // 4 rows, should be 5
        let result = MultiFunData::new(vec![comp1, comp2]);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    #[test]
    fn test_argvals_len_mismatch_first_component_err() {
        // argvals.len() != data.ncols() for component 0
        let comp = FdComponent {
            data: FdMatrix::zeros(5, 10),
            argvals: vec![0.0, 1.0, 2.0], // len=3, ncols=10
        };
        let result = MultiFunData::new(vec![comp]);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    #[test]
    fn test_argvals_len_mismatch_later_component_err() {
        // argvals.len() != data.ncols() for component 1
        let comp1 = make_component(5, 10);
        let comp2 = FdComponent {
            data: FdMatrix::zeros(5, 4),
            argvals: vec![0.0, 1.0], // len=2, ncols=4
        };
        let result = MultiFunData::new(vec![comp1, comp2]);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    // --- Accessor tests ---

    #[test]
    fn test_component_accessor_valid() {
        let argvals1: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let argvals2: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let comp1 = make_component_argvals(5, argvals1.clone());
        let comp2 = make_component_argvals(5, argvals2.clone());
        let mfd = MultiFunData::new(vec![comp1, comp2]).unwrap();

        let c0 = mfd.component(0).unwrap();
        assert_eq!(c0.argvals, argvals1);
        assert_eq!(c0.data.nrows(), 5);
        assert_eq!(c0.data.ncols(), 10);

        let c1 = mfd.component(1).unwrap();
        assert_eq!(c1.argvals, argvals2);
        assert_eq!(c1.data.ncols(), 4);
    }

    #[test]
    fn test_component_accessor_out_of_range_err() {
        let mfd = MultiFunData::new(vec![make_component(3, 5)]).unwrap();
        let result = mfd.component(1);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_argvals_accessor_valid() {
        let argvals: Vec<f64> = vec![0.0, 0.5, 1.0];
        let comp = make_component_argvals(4, argvals.clone());
        let mfd = MultiFunData::new(vec![comp]).unwrap();
        assert_eq!(mfd.argvals(0).unwrap(), argvals.as_slice());
    }

    #[test]
    fn test_argvals_accessor_out_of_range_err() {
        let mfd = MultiFunData::new(vec![make_component(3, 5)]).unwrap();
        let result = mfd.argvals(5);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_component_accessor_preserves_argvals_per_component() {
        // Ensure each component's argvals are preserved independently.
        let argvals1: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let argvals2: Vec<f64> = vec![10.0, 20.0];
        let comp1 = make_component_argvals(6, argvals1.clone());
        let comp2 = make_component_argvals(6, argvals2.clone());
        let mfd = MultiFunData::new(vec![comp1, comp2]).unwrap();

        assert_eq!(mfd.argvals(0).unwrap(), argvals1.as_slice());
        assert_eq!(mfd.argvals(1).unwrap(), argvals2.as_slice());
    }

    #[test]
    fn test_no_panic_on_out_of_range_component() {
        let mfd = MultiFunData::new(vec![make_component(2, 3)]).unwrap();
        // These must never panic — they return Err.
        assert!(mfd.component(100).is_err());
        assert!(mfd.argvals(100).is_err());
        assert!(mfd.component(usize::MAX).is_err());
    }

    // --- Trait derive tests ---

    #[test]
    fn test_debug_clone_partialeq() {
        let comp = make_component(2, 3);
        let mfd = MultiFunData::new(vec![comp]).unwrap();
        let mfd2 = mfd.clone();
        assert_eq!(mfd, mfd2);
        let s = format!("{:?}", mfd);
        assert!(s.contains("MultiFunData"));
    }

    #[test]
    fn test_fdcomponent_debug_clone_partialeq() {
        let comp = make_component(2, 4);
        let comp2 = comp.clone();
        assert_eq!(comp, comp2);
        let s = format!("{:?}", comp);
        assert!(s.contains("FdComponent"));
    }
}
