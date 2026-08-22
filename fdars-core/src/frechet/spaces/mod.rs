//! Non-density object-data `MetricSpace` backends (FRE-02).
//!
//! Each backend implements [`crate::frechet::MetricSpace`] for a specific object
//! geometry, consumed generically by the Fréchet regression / ANOVA solvers:
//!
//! * [`SpdMatrixSpace`] / [`SpdMetric`] — SPD covariance matrices (FRE-02-01).
//! * [`CorrelationMatrixSpace`] — correlation matrices (FRE-02-02).
//! * [`SphericalSpace`] — spherical data (FRE-02-03).
//! * [`NetworkSpace`] — graph-Laplacian networks (FRE-02-04).
//! * [`PointProcessSpace`] — intensity/count point processes (FRE-02-05).

mod correlation;
mod network;
mod point_process;
mod spd;
mod spherical;

pub use correlation::CorrelationMatrixSpace;
pub use network::NetworkSpace;
pub use point_process::PointProcessSpace;
pub use spd::{SpdMatrixSpace, SpdMetric};
pub use spherical::SphericalSpace;
