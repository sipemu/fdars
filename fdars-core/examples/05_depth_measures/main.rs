//! Example 05: Functional Depth Measures
//!
//! Depth measures rank curves from most central (high depth) to most
//! extreme (low depth). This example computes several depth measures
//! and shows how to identify the deepest (most typical) and most
//! outlying curves in a dataset.

use fdars_core::depth::{
    band_1d, fraiman_muniz_1d, functional_spatial_1d, modal_1d, modified_band_1d,
    modified_epigraph_index_1d, random_projection_1d, random_tukey_1d,
};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn rank_indices(depths: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..depths.len()).collect();
    indices.sort_by(|&a, &b| depths[b].partial_cmp(&depths[a]).unwrap());
    indices
}

fn main() {
    println!("=== Example 05: Functional Depth Measures ===\n");

    let n = 30;
    let m = 50;
    let big_m = 5;
    let t = uniform_grid(m);

    // Generate data: most curves from one process, a few shifted outliers
    let mut data_mat = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    // Inject 3 outliers by shifting them
    for j in 0..m {
        data_mat[(27, j)] += 3.0; // magnitude outlier
        data_mat[(28, j)] += (t[j] * 10.0).sin() * 2.0; // shape outlier
        data_mat[(29, j)] -= 2.5; // magnitude outlier (opposite direction)
    }

    println!("--- Dataset ---");
    println!(
        "  {n} curves ({} normal + 3 injected outliers: indices 27,28,29)",
        n - 3
    );
    println!("  {m} grid points on [0, 1]");

    let mat = data_mat;

    // --- Section 1: Fraiman-Muniz depth ---
    println!("\n--- Fraiman-Muniz Depth ---");
    let fm = fraiman_muniz_1d(&mat, &mat, true);
    let fm_rank = rank_indices(&fm);
    println!(
        "  Deepest 3:  {:?} (depths: {:.4}, {:.4}, {:.4})",
        &fm_rank[..3],
        fm[fm_rank[0]],
        fm[fm_rank[1]],
        fm[fm_rank[2]]
    );
    println!(
        "  Shallowest 3: {:?} (depths: {:.4}, {:.4}, {:.4})",
        &fm_rank[n - 3..],
        fm[fm_rank[n - 3]],
        fm[fm_rank[n - 2]],
        fm[fm_rank[n - 1]]
    );

    // --- Section 2: Band depth ---
    println!("\n--- Band Depth ---");
    let bd = band_1d(&mat, &mat);
    let bd_rank = rank_indices(&bd);
    println!("  Deepest 3:  {:?}", &bd_rank[..3]);
    println!("  Shallowest 3: {:?}", &bd_rank[n - 3..]);

    // --- Section 3: Modified Band Depth ---
    println!("\n--- Modified Band Depth ---");
    let mbd = modified_band_1d(&mat, &mat);
    let mbd_rank = rank_indices(&mbd);
    println!("  Deepest 3:  {:?}", &mbd_rank[..3]);
    println!("  Shallowest 3: {:?}", &mbd_rank[n - 3..]);

    // --- Section 4: Modal depth ---
    println!("\n--- Modal Depth (h=0.5) ---");
    let modal = modal_1d(&mat, &mat, 0.5);
    let modal_rank = rank_indices(&modal);
    println!("  Deepest 3:  {:?}", &modal_rank[..3]);
    println!("  Shallowest 3: {:?}", &modal_rank[n - 3..]);

    // --- Section 5: Random projection depth ---
    println!("\n--- Random Projection Depth (50 projections) ---");
    let rp = random_projection_1d(&mat, &mat, 50);
    let rp_rank = rank_indices(&rp);
    println!("  Deepest 3:  {:?}", &rp_rank[..3]);
    println!("  Shallowest 3: {:?}", &rp_rank[n - 3..]);

    // --- Section 6: Random Tukey depth ---
    println!("\n--- Random Tukey Depth (50 projections) ---");
    let rt = random_tukey_1d(&mat, &mat, 50);
    let rt_rank = rank_indices(&rt);
    println!("  Deepest 3:  {:?}", &rt_rank[..3]);
    println!("  Shallowest 3: {:?}", &rt_rank[n - 3..]);

    // --- Section 7: Functional spatial depth ---
    println!("\n--- Functional Spatial Depth ---");
    let fsd = functional_spatial_1d(&mat, &mat);
    let fsd_rank = rank_indices(&fsd);
    println!("  Deepest 3:  {:?}", &fsd_rank[..3]);
    println!("  Shallowest 3: {:?}", &fsd_rank[n - 3..]);

    // --- Section 8: Modified Epigraph Index ---
    println!("\n--- Modified Epigraph Index ---");
    let mei = modified_epigraph_index_1d(&mat, &mat);
    let mei_rank = rank_indices(&mei);
    println!("  Deepest 3:  {:?}", &mei_rank[..3]);
    println!("  Shallowest 3: {:?}", &mei_rank[n - 3..]);

    // --- Section 9: Outlier detection summary ---
    println!("\n--- Outlier Detection Summary ---");
    println!("  Injected outlier indices: 27, 28, 29");
    println!("  Which methods rank outliers in bottom 3?");
    let methods: Vec<(&str, &[usize])> = vec![
        ("Fraiman-Muniz", &fm_rank),
        ("Band", &bd_rank),
        ("Modified Band", &mbd_rank),
        ("Modal", &modal_rank),
        ("Random Proj.", &rp_rank),
        ("Random Tukey", &rt_rank),
        ("Func. Spatial", &fsd_rank),
        ("Mod. Epigraph", &mei_rank),
    ];
    for (name, rank) in methods {
        let bottom3: Vec<usize> = rank[n - 3..].to_vec();
        let detected: Vec<usize> = bottom3.iter().filter(|&&i| i >= 27).copied().collect();
        println!(
            "  {name:15}: bottom 3 = {bottom3:?}, outliers detected = {}/{}",
            detected.len(),
            3
        );
    }

    println!("\n=== Done ===");
}
