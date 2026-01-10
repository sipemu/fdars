use core::ptr::null;
use std::collections::HashMap;

use aligned_vec::avec;
use diol::Picoseconds;
use diol::config::*;
use diol::prelude::*;
use diol::result::BenchResult;
use faer::prelude::*;
use faer::reborrow::*;
use private_gemm_x86::*;
use rand::prelude::*;

fn bench_faer(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
	let rng = &mut StdRng::seed_from_u64(0);
	let mut cs = Ord::min(m.next_power_of_two(), m.next_multiple_of(8));
	if m > 48 {
		cs = Ord::max(4096, cs);
	}

	let params = gemm_common::cache::kernel_params(m, n, k, 4 * 8, 6, 8);

	let _ = params;

	let lhs = &mut *avec![[4096]| 0.0; cs * k];
	let rhs = &mut *avec![[4096]| 0.0; k * n];
	let dst = &mut *avec![[4096]| 0.0; cs * n];

	rng.fill(lhs);
	rng.fill(rhs);

	let lhs = MatRef::from_column_major_slice_with_stride(lhs, m, k, cs);
	let rhs = MatRef::from_column_major_slice_with_stride(rhs, k, n, k);
	let mut dst = unsafe { MatMut::from_raw_parts_mut(dst.as_mut_ptr(), m, n, 1, cs as isize) };

	bencher.bench(|| {
		faer::linalg::matmul::triangular::matmul(
			dst.rb_mut(),
			faer::linalg::matmul::triangular::BlockStructure::Rectangular,
			faer::Accum::Replace,
			lhs.rb(),
			faer::linalg::matmul::triangular::BlockStructure::Rectangular,
			rhs,
			faer::linalg::matmul::triangular::BlockStructure::Rectangular,
			1.0,
			Par::Seq,
		);
	});
}

fn bench_asm(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
	let rng = &mut StdRng::seed_from_u64(0);
	let n_threads = 1;

	let mut cs = Ord::min(m.next_power_of_two(), m.next_multiple_of(8));
	if m > 48 {
		cs = Ord::max(4096, cs);
	}

	let lhs = &mut *avec![[4096]| 0.0; cs * k];
	let rhs = &mut *avec![[4096]| 0.0; k * n];
	let dst = &mut *avec![[4096]| 0.0;  cs * n];

	rng.fill(lhs);
	rng.fill(rhs);

	// spindle::with_lock(n_threads, || {
	({
		bencher.bench(|| unsafe {
			gemm(
				DType::F64,
				IType::U64,
				InstrSet::Avx512,
				m,
				n,
				k,
				dst.as_mut_ptr() as *mut (),
				1,
				cs as isize,
				null(),
				null(),
				DstKind::Full,
				Accum::Replace,
				lhs.as_ptr() as *const (),
				1,
				cs as isize,
				false,
				null(),
				0,
				rhs.as_ptr() as *const (),
				1,
				k as isize,
				false,
				&raw const *&1.0 as *const (),
				n_threads,
			)
		});
	});
}

fn main() -> eyre::Result<()> {
	let config = &mut Config::from_args()?;
	config.plot_axis = PlotAxis::SemiLogX;
	let plot_dir = &config.plot_dir.0.take();

	for k in [8, 16, 32, 64, 128, 256, 512] {
		let mut args_small: [_; 16] = core::array::from_fn(|i| {
			let i = i as u32;
			if i % 2 == 0 {
				2usize.pow(1 + i / 2 as u32)
			} else {
				3 * 2usize.pow(i / 2 as u32)
			}
		});
		args_small.sort_unstable();
		let args_small = args_small.map(PlotArg);

		let mut args_big: [_; 11] = core::array::from_fn(|i| {
			let i = i as u32;
			if i % 2 == 0 {
				2usize.pow(3 + i / 2 as u32)
			} else {
				3 * 2usize.pow(2 + i / 2 as u32)
			}
		});
		args_big.sort_unstable();

		let args_big = args_big.map(PlotArg);

		let f = [bench_asm, bench_faer];

		if true {
			config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| (n * n * k) as f64 / time.to_secs()).with_name("flops");
			let bench = Bench::new(&config);

			let f = f.map(move |f| move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (n, n, k)));
			bench.register_many(
				&format!("k={k} m=n"),
				list![f[0].with_name(&format!("asm")), f[1].with_name(&format!("faer"))],
				args_big,
			);
			let results = bench.run()?.combine(
				&serde_json::from_str(
					&std::fs::read_to_string(&format!(
						"{}/timings {}.json",
						concat!(env!("CARGO_MANIFEST_DIR")),
						bench.groups.borrow().keys().next().unwrap()
					))
					.unwrap_or(String::new()),
				)
				.unwrap_or(BenchResult { groups: HashMap::new() }),
			);
			std::fs::write(
				format!(
					"{}/timings {}.json",
					concat!(env!("CARGO_MANIFEST_DIR")),
					bench.groups.borrow().keys().next().unwrap()
				),
				serde_json::to_string(&results)?,
			)?;

			if let Some(plot_dir) = plot_dir {
				results.plot(config, plot_dir)?;
			}
		}

		for PlotArg(m) in args_small {
			if true {
				config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| (n * m * k) as f64 / time.to_secs()).with_name("flops");
				let bench = Bench::new(&config);
				let f = f.map(move |f| move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (n, m, k)));
				bench.register_many(
					&format!("k={k} n={m}"),
					list![f[0].with_name(&format!("asm")), f[1].with_name(&format!("faer"))],
					args_big,
				);
				let results = bench.run()?.combine(
					&serde_json::from_str(
						&std::fs::read_to_string(&format!(
							"{}/timings {}.json",
							concat!(env!("CARGO_MANIFEST_DIR")),
							bench.groups.borrow().keys().next().unwrap()
						))
						.unwrap_or(String::new()),
					)
					.unwrap_or(BenchResult { groups: HashMap::new() }),
				);
				std::fs::write(
					format!(
						"{}/timings {}.json",
						concat!(env!("CARGO_MANIFEST_DIR")),
						bench.groups.borrow().keys().next().unwrap()
					),
					serde_json::to_string(&results)?,
				)?;
				if let Some(plot_dir) = plot_dir {
					results.plot(config, plot_dir)?;
				}
			}

			if true {
				config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| (n * m * k) as f64 / time.to_secs()).with_name("flops");
				let bench = Bench::new(&config);
				let f = f.map(move |f| move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (m, n, k)));
				bench.register_many(
					&format!("k={k} m={m}"),
					list![f[0].with_name(&format!("asm")), f[1].with_name(&format!("faer"))],
					args_big,
				);
				let results = bench.run()?.combine(
					&serde_json::from_str(
						&std::fs::read_to_string(&format!(
							"{}/timings {}.json",
							concat!(env!("CARGO_MANIFEST_DIR")),
							bench.groups.borrow().keys().next().unwrap()
						))
						.unwrap_or(String::new()),
					)
					.unwrap_or(BenchResult { groups: HashMap::new() }),
				);
				std::fs::write(
					format!(
						"{}/timings {}.json",
						concat!(env!("CARGO_MANIFEST_DIR")),
						bench.groups.borrow().keys().next().unwrap()
					),
					serde_json::to_string(&results)?,
				)?;
				if let Some(plot_dir) = plot_dir {
					results.plot(config, plot_dir)?;
				}
			}
		}
	}

	Ok(())
}
