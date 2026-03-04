//! Example 17: Functional Equivalence Test (TOST)
//!
//! Demonstrates testing whether two functional groups are equivalent
//! within a margin δ using the sup-norm and bootstrap methods.

use fdars_core::fdata::mean_1d;
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::tolerance::{
    equivalence_test, equivalence_test_one_sample, EquivalenceBootstrap, MultiplierDistribution,
};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 17: Functional Equivalence Test (TOST) ===\n");

    let n = 40;
    let m = 60;
    let t = uniform_grid(m);

    // --- Section 1: Equivalent groups (same DGP, large delta) ---
    println!("--- Section 1: Equivalent groups ---");
    let data1 = sim_fundata(n, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
    let data2 = sim_fundata(n, &t, 5, EFunType::Fourier, EValType::Exponential, Some(99));

    let delta = 5.0;
    let result = equivalence_test(
        &data1,
        &data2,
        delta,
        0.05,
        500,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .expect("equivalence_test failed");

    println!("  delta = {delta}");
    println!(
        "  test_statistic (sup|d_hat|) = {:.4}",
        result.test_statistic
    );
    println!("  critical_value = {:.4}", result.critical_value);
    println!("  p_value = {:.4}", result.p_value);
    println!("  equivalent = {}", result.equivalent);

    // --- Section 2: Non-equivalent groups (shifted mean, small delta) ---
    println!("\n--- Section 2: Non-equivalent groups ---");
    let mut data2_shifted =
        sim_fundata(n, &t, 5, EFunType::Fourier, EValType::Exponential, Some(99));
    for i in 0..n {
        for j in 0..m {
            data2_shifted[(i, j)] += 5.0;
        }
    }

    let delta_small = 1.0;
    let result2 = equivalence_test(
        &data1,
        &data2_shifted,
        delta_small,
        0.05,
        500,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .expect("equivalence_test failed");

    println!("  delta = {delta_small}");
    println!(
        "  test_statistic (sup|d_hat|) = {:.4}",
        result2.test_statistic
    );
    println!("  p_value = {:.4}", result2.p_value);
    println!("  equivalent = {}", result2.equivalent);

    // --- Section 3: Percentile vs multiplier comparison ---
    println!("\n--- Section 3: Percentile vs Multiplier comparison ---");
    let r_mult = equivalence_test(
        &data1,
        &data2,
        delta,
        0.05,
        500,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();
    let r_pct = equivalence_test(
        &data1,
        &data2,
        delta,
        0.05,
        500,
        EquivalenceBootstrap::Percentile,
        42,
    )
    .unwrap();

    println!(
        "  Multiplier: p={:.4}, equivalent={}",
        r_mult.p_value, r_mult.equivalent
    );
    println!(
        "  Percentile: p={:.4}, equivalent={}",
        r_pct.p_value, r_pct.equivalent
    );

    // --- Section 4: One-sample test ---
    println!("\n--- Section 4: One-sample equivalence test ---");
    let mu0 = mean_1d(&data1);
    let r_one = equivalence_test_one_sample(
        &data1,
        &mu0,
        5.0,
        0.05,
        500,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();

    println!("  Testing data vs its own sample mean (delta=5.0)");
    println!("  test_statistic = {:.6}", r_one.test_statistic);
    println!("  p_value = {:.4}", r_one.p_value);
    println!("  equivalent = {}", r_one.equivalent);

    println!("\n=== Done ===");
}
