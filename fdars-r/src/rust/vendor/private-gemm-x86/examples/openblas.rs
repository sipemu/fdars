use aligned_vec::avec;
use diol::Picoseconds;
use diol::config::*;
use diol::prelude::*;
use rand::prelude::*;

extern crate openblas_src;

fn bench_blas(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
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

	bencher.bench(|| unsafe {
		blas_sys::dgemm_(
			&(b'N' as i8),
			&(b'N' as i8),
			&raw const m as _,
			&raw const n as _,
			&raw const k as _,
			&1.0,
			lhs.as_ptr(),
			&raw const cs as _,
			rhs.as_ptr(),
			&raw const k as _,
			&0.0,
			dst.as_mut_ptr(),
			&raw const cs as _,
		);
	});
}

fn main() -> eyre::Result<()> {
	let mut config = Config::from_args()?;

	for k in [64, 128, 256, 512] {
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
				2usize.pow(9 + i / 2 as u32)
			} else {
				3 * 2usize.pow(8 + i / 2 as u32)
			}
		});
		args_big.sort_unstable();

		let args_big = args_big.map(PlotArg);

		let f = [bench_blas];

		{
			config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| (n * n * k) as f64 / time.to_secs()).with_name("flops");
			let bench = Bench::new(&config);

			let f = f.map(move |f| move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (n, n, k)));
			bench.register_many(&format!("k={k} m=n"), list![f[0].with_name("openblas")], args_big);
			std::fs::write(
				format!(
					"{}/timings {}.json",
					concat!(env!("CARGO_MANIFEST_DIR")),
					bench.groups.borrow().keys().next().unwrap()
				),
				serde_json::to_string(&bench.run()?)?,
			)?;
		}

		for PlotArg(m) in args_small {
			{
				config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| (n * m * k) as f64 / time.to_secs()).with_name("flops");
				let bench = Bench::new(&config);
				let f = f.map(move |f| move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (n, m, k)));
				bench.register_many(&format!("k={k} n={m}"), list![f[0].with_name("openblas")], args_big);
				std::fs::write(
					format!(
						"{}/timings {}.json",
						concat!(env!("CARGO_MANIFEST_DIR")),
						bench.groups.borrow().keys().next().unwrap()
					),
					serde_json::to_string(&bench.run()?)?,
				)?;
			}

			{
				config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| (n * m * k) as f64 / time.to_secs()).with_name("flops");
				let bench = Bench::new(&config);
				let f = f.map(move |f| move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (m, n, k)));
				bench.register_many(&format!("k={k} m={m}"), list![f[0].with_name("openblas")], args_big);
				std::fs::write(
					format!(
						"{}/timings {}.json",
						concat!(env!("CARGO_MANIFEST_DIR")),
						bench.groups.borrow().keys().next().unwrap()
					),
					serde_json::to_string(&bench.run()?)?,
				)?;
			}
		}
	}

	Ok(())
}
