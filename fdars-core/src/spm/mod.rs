//! Statistical Process Monitoring (SPM) for functional data.
//!
//! This module implements a complete framework for monitoring functional data
//! processes using FPCA-based control charts. It supports both univariate and
//! multivariate functional settings.
//!
//! # Overview
//!
//! The SPM workflow follows two phases:
//!
//! - **Phase I** (training): Builds a monitoring chart from historical in-control
//!   data. The data is split into a tuning set (for FPCA) and a calibration set
//!   (for establishing control limits).
//!
//! - **Phase II** (monitoring): Projects new observations through the trained
//!   model and checks whether monitoring statistics exceed control limits.
//!
//! # Monitoring Statistics
//!
//! - **Hotelling T-squared**: Measures variation in the principal component
//!   subspace. Sensitive to shifts in the major modes of variation.
//!
//! - **SPE (Squared Prediction Error)**: Measures residual variation outside
//!   the PC subspace. Sensitive to new types of variation not captured by FPCA.
//!
//! # Modules
//!
//! - [`phase`]: Core Phase I/II framework for univariate and multivariate SPM
//! - [`ewma`]: EWMA smoothing for enhanced sensitivity to small persistent shifts
//! - [`frcc`]: Functional Regression Control Chart (adjusts for known covariates)
//! - [`mod@mfpca`]: Multivariate FPCA for multi-response monitoring
//! - [`stats`]: T-squared and SPE computation
//! - [`control`]: Control limit estimation
//! - [`contrib`]: Contribution diagnostics for fault identification
//!
//! # Example
//!
//! ```rust,no_run
//! use fdars_core::matrix::FdMatrix;
//! use fdars_core::spm::phase::{spm_phase1, spm_monitor, SpmConfig};
//!
//! // Phase I: build chart from in-control data
//! # let data = FdMatrix::zeros(40, 30);
//! # let argvals: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
//! let config = SpmConfig { ncomp: 3, alpha: 0.05, ..SpmConfig::default() };
//! let chart = spm_phase1(&data, &argvals, &config).unwrap();
//!
//! // Phase II: monitor new data
//! # let new_data = FdMatrix::zeros(10, 30);
//! let result = spm_monitor(&chart, &new_data, &argvals).unwrap();
//! // Check result.t2_alarm and result.spe_alarm for out-of-control signals
//! ```

pub(super) mod chi_squared;
pub mod contrib;
pub mod control;
pub mod ewma;
pub mod frcc;
pub mod mfpca;
pub mod phase;
pub mod stats;

#[cfg(test)]
mod tests;

// Re-export primary types and functions for convenience
pub use contrib::{spe_contributions, t2_contributions};
pub use control::{spe_control_limit, t2_control_limit, ControlLimit};
pub use ewma::{ewma_scores, spm_ewma_monitor, EwmaConfig, EwmaMonitorResult};
pub use frcc::{frcc_monitor, frcc_phase1, FrccChart, FrccConfig, FrccMonitorResult};
pub use mfpca::{mfpca, MfpcaConfig, MfpcaResult};
pub use phase::{
    mf_spm_monitor, mf_spm_phase1, spm_monitor, spm_phase1, MfSpmChart, SpmChart, SpmConfig,
    SpmMonitorResult,
};
pub use stats::{hotelling_t2, spe_multivariate, spe_univariate};
