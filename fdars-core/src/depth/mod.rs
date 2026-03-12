//! Depth measures for functional data.
//!
//! This module provides various depth measures for assessing the centrality
//! of functional observations within a reference sample.

use crate::matrix::FdMatrix;
use crate::maybe_par_chunks_mut_enumerate;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub mod band;
pub mod fraiman_muniz;
pub mod modal;
pub mod random_projection;
pub mod random_tukey;
pub mod spatial;

#[cfg(test)]
mod tests;

// Re-export all public functions
pub use band::{band_1d, modified_band_1d, modified_epigraph_index_1d};
pub use fraiman_muniz::{fraiman_muniz_1d, fraiman_muniz_2d};
pub use modal::{modal_1d, modal_2d};
pub use random_projection::{random_projection_1d, random_projection_1d_seeded, random_projection_2d};
pub use random_tukey::{random_tukey_1d, random_tukey_1d_seeded, random_tukey_2d};
pub use spatial::{
    functional_spatial_1d, functional_spatial_2d, kernel_functional_spatial_1d,
    kernel_functional_spatial_2d,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Generate `nproj` unit-norm random projection vectors of dimension `m`.
///
/// Returns a flat buffer of length `nproj * m` where projection `p` occupies
/// `[p*m .. (p+1)*m]`.
///
/// If `seed` is Some, uses a deterministic RNG seeded from the given value.
pub(super) fn generate_random_projections(nproj: usize, m: usize, seed: Option<u64>) -> Vec<f64> {
    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };
    let mut projections = vec![0.0; nproj * m];
    for p_idx in 0..nproj {
        let base = p_idx * m;
        let mut norm_sq = 0.0;
        for t in 0..m {
            let v: f64 = rng.sample(StandardNormal);
            projections[base + t] = v;
            norm_sq += v * v;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for t in 0..m {
            projections[base + t] *= inv_norm;
        }
    }
    projections
}

/// Project each reference curve onto each projection direction and sort.
///
/// Returns a flat buffer of length `nproj * nori` where the sorted projections
/// for direction `p` occupy `[p*nori .. (p+1)*nori]`.
pub(super) fn project_and_sort_reference(
    data_ori: &FdMatrix,
    projections: &[f64],
    nproj: usize,
    nori: usize,
    m: usize,
) -> Vec<f64> {
    let mut sorted = vec![0.0; nproj * nori];
    maybe_par_chunks_mut_enumerate!(sorted, nori, |(p_idx, spo): (usize, &mut [f64])| {
        let proj = &projections[p_idx * m..(p_idx + 1) * m];
        for j in 0..nori {
            let mut dot = 0.0;
            for t in 0..m {
                dot += data_ori[(j, t)] * proj[t];
            }
            spo[j] = dot;
        }
        spo.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    });
    sorted
}

/// Shared implementation for random projection-based depth measures.
///
/// Generates `nproj` random projections, projects both object and reference
/// curves to scalars, and computes univariate depth via binary-search ranking.
/// The `aggregate` and `finalize` closures control how per-projection depths
/// are combined (e.g. average for RP depth, minimum for Tukey depth).
pub(super) fn random_depth_core(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    nproj: usize,
    seed: Option<u64>,
    init: f64,
    aggregate: impl Fn(f64, f64) -> f64 + Sync,
    finalize: impl Fn(f64, usize) -> f64 + Sync,
) -> Vec<f64> {
    use crate::iter_maybe_parallel;
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let m = data_obj.ncols();

    if nobj == 0 || nori == 0 || m == 0 || nproj == 0 {
        return Vec::new();
    }

    let projections = generate_random_projections(nproj, m, seed);
    let sorted_proj_ori = project_and_sort_reference(data_ori, &projections, nproj, nori, m);
    let denom = nori as f64 + 1.0;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut acc = init;
            for p_idx in 0..nproj {
                let proj = &projections[p_idx * m..(p_idx + 1) * m];
                let sorted_ori = &sorted_proj_ori[p_idx * nori..(p_idx + 1) * nori];

                let mut proj_i = 0.0;
                for t in 0..m {
                    proj_i += data_obj[(i, t)] * proj[t];
                }

                let below = sorted_ori.partition_point(|&v| v < proj_i);
                let above = nori - sorted_ori.partition_point(|&v| v <= proj_i);
                let depth = (below.min(above) as f64 + 1.0) / denom;
                acc = aggregate(acc, depth);
            }
            finalize(acc, nproj)
        })
        .collect()
}
