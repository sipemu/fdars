//! Non-density object-data `MetricSpace` backends (FRE-02).
//!
//! Each backend implements [`crate::frechet::MetricSpace`] for a specific object
//! geometry, consumed generically by the Fréchet regression / ANOVA solvers:
//!
//! * [`SpdMatrixSpace`] / [`SpdMetric`] — SPD covariance matrices (FRE-02-01).
//! * [`CorrelationMatrixSpace`] — correlation matrices (FRE-02-02).

mod correlation;
mod spd;

pub use correlation::CorrelationMatrixSpace;
pub use spd::{SpdMatrixSpace, SpdMetric};
