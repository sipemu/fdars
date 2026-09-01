//! Dimensionality selector shared by the unified depth/fdata dispatchers.
//!
//! Several depth measures and `fdata::mean` historically shipped `_1d` / `_2d`
//! variants where the `_2d` path was a thin shim that simply forwarded to the
//! `_1d` primitive (the 2D computation never diverged from the 1D one — both
//! iterate the flattened column-major grid identically). The unified
//! dispatchers take an explicit [`Dim`] argument so callers get one ergonomic
//! `name(…, dim)` entry point, while the redundant `_2d` shims are retired via
//! `#[deprecated]`.
//!
//! The enum is `#[non_exhaustive]` so future dimensionalities (or a genuinely
//! divergent 2D path) can be added without a breaking change.

/// Dimensionality selector for the unified depth/fdata dispatchers.
///
/// Both arms currently forward to the same `_1d` primitive because the `_2d`
/// path never diverged; the `dim` argument makes caller intent explicit and
/// gives a single future seam should a real 2D specialization ever be needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Dim {
    /// 1D functional data (curves).
    One,
    /// 2D functional data (surfaces, flattened column-major).
    Two,
}
