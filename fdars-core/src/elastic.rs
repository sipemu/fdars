//! Unified access to elastic (SRSF-based) analysis methods.
//!
//! This module re-exports items from the individual `elastic_*` modules
//! for convenient importing:
//!
//! ```rust
//! use fdars_core::elastic::*;
//! ```

pub use crate::elastic_changepoint::*;
pub use crate::elastic_explain::*;
pub use crate::elastic_fpca::*;
pub use crate::elastic_regression::*;
