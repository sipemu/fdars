use std::fmt;

/// Error type for fdars operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FdarError {
    /// Input dimensions invalid (empty matrix, length mismatch).
    InvalidDimension {
        parameter: &'static str,
        expected: String,
        actual: String,
    },
    /// Parameter value out of allowed range.
    InvalidParameter {
        parameter: &'static str,
        message: String,
    },
    /// Numerical computation failed (SVD, matrix inversion, convergence).
    ComputationFailed {
        operation: &'static str,
        detail: String,
    },
    /// Enum conversion from integer failed.
    InvalidEnumValue { enum_name: &'static str, value: i32 },
}

impl fmt::Display for FdarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FdarError::InvalidDimension {
                parameter,
                expected,
                actual,
            } => write!(
                f,
                "invalid dimension for '{parameter}': expected {expected}, got {actual}"
            ),
            FdarError::InvalidParameter { parameter, message } => {
                write!(f, "invalid parameter '{parameter}': {message}")
            }
            FdarError::ComputationFailed { operation, detail } => {
                write!(f, "{operation} failed: {detail}")
            }
            FdarError::InvalidEnumValue { enum_name, value } => {
                write!(f, "invalid value {value} for enum '{enum_name}'")
            }
        }
    }
}

impl std::error::Error for FdarError {}
