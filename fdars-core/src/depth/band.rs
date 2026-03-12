//! Band depth and related measures (BD, MBD, MEI).

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::streaming_depth::{
    FullReferenceState, SortedReferenceState, StreamingBd, StreamingDepth, StreamingMbd,
};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute Band Depth (BD) for 1D functional data.
///
/// BD(x) = proportion of pairs (i,j) where x lies within the band formed by curves i and j.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn band_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() < 2 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = FullReferenceState::from_reference(data_ori);
    let streaming = StreamingBd::new(state);
    streaming.depth_batch(data_obj)
}

/// Compute Modified Band Depth (MBD) for 1D functional data.
///
/// MBD(x) = average over pairs of the proportion of the domain where x is inside the band.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modified_band_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() < 2 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = SortedReferenceState::from_reference(data_ori);
    let streaming = StreamingMbd::new(state);
    streaming.depth_batch(data_obj)
}

/// Compute Modified Epigraph Index (MEI) for 1D functional data.
///
/// MEI measures the proportion of time a curve is below other curves.
/// Matches R's `roahd::MEI()`: uses `<=` comparison with 0.5 adjustment for ties.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modified_epigraph_index_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total = 0.0;

            for j in 0..nori {
                let mut count = 0.0;

                for t in 0..n_points {
                    let xi = data_obj[(i, t)];
                    let xj = data_ori[(j, t)];

                    // R's roahd::MEI uses <= for the epigraph condition
                    if xi <= xj {
                        count += 1.0;
                    }
                }

                total += count / n_points as f64;
            }

            total / nori as f64
        })
        .collect()
}
