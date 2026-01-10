#[derive(Default, Debug, Copy, Clone)]
pub struct CacheInfo {
	pub associativity: usize,
	pub cache_bytes: usize,
	pub cache_line_bytes: usize,
}

fn cache_info() -> Option<[CacheInfo; 3]> {
	if !cfg!(miri) {
		{
			#[cfg(all(target_os = "linux", feature = "std"))]
			{
				use std::fs;
				fn try_cache_info_linux() -> Result<[CacheInfo; 3], std::io::Error> {
					let mut all_info = [CacheInfo {
						associativity: 8,
						cache_bytes: 0,
						cache_line_bytes: 64,
					}; 3];

					let mut l1_shared_count = 1;
					for cpu_x in fs::read_dir("/sys/devices/system/cpu")? {
						let cpu_x = cpu_x?.path();
						let Some(cpu_x_name) = cpu_x.file_name().and_then(|f| f.to_str()) else {
							continue;
						};
						if !cpu_x_name.starts_with("cpu") {
							continue;
						}
						let cache = cpu_x.join("cache");
						if !cache.is_dir() {
							continue;
						}
						'index: for index_y in fs::read_dir(cache)? {
							let index_y = index_y?.path();
							if !index_y.is_dir() {
								continue;
							}
							let Some(index_y_name) = index_y.file_name().and_then(|f| f.to_str()) else {
								continue;
							};
							if !index_y_name.starts_with("index") {
								continue;
							}

							let mut cache_info = CacheInfo {
								associativity: 8,
								cache_bytes: 0,
								cache_line_bytes: 64,
							};
							let mut level: usize = 0;
							let mut shared_count: usize = 0;

							for entry in fs::read_dir(index_y)? {
								let entry = entry?.path();
								if let Some(name) = entry.file_name() {
									let contents = fs::read_to_string(&entry)?;
									let contents = contents.trim();
									if name == "type" && !matches!(contents, "Data" | "Unified") {
										continue 'index;
									}
									if name == "shared_cpu_list" {
										for item in contents.split(',') {
											if item.contains('-') {
												let mut item = item.split('-');
												let Some(start) = item.next() else {
													continue 'index;
												};
												let Some(end) = item.next() else {
													continue 'index;
												};

												let Ok(start) = start.parse::<usize>() else {
													continue 'index;
												};
												let Ok(end) = end.parse::<usize>() else {
													continue 'index;
												};

												shared_count += end + 1 - start;
											} else {
												shared_count += 1;
											}
										}
									}

									if name == "level" {
										let Ok(contents) = contents.parse::<usize>() else {
											continue 'index;
										};
										level = contents;
									}

									if name == "coherency_line_size" {
										let Ok(contents) = contents.parse::<usize>() else {
											continue 'index;
										};
										cache_info.cache_line_bytes = contents;
									}
									if name == "ways_of_associativity" {
										let Ok(contents) = contents.parse::<usize>() else {
											continue 'index;
										};
										cache_info.associativity = contents;
									}
									if name == "size" {
										if contents.ends_with("G") {
											let Ok(contents) = contents.trim_end_matches('G').parse::<usize>() else {
												continue 'index;
											};
											cache_info.cache_bytes = contents * 1024 * 1024 * 1024;
										} else if contents.ends_with("M") {
											let Ok(contents) = contents.trim_end_matches('M').parse::<usize>() else {
												continue 'index;
											};
											cache_info.cache_bytes = contents * 1024 * 1024;
										} else if contents.ends_with("K") {
											let Ok(contents) = contents.trim_end_matches('K').parse::<usize>() else {
												continue 'index;
											};
											cache_info.cache_bytes = contents * 1024;
										} else {
											let Ok(contents) = contents.parse::<usize>() else {
												continue 'index;
											};
											cache_info.cache_bytes = contents;
										}
									}
								}
							}
							if level == 1 {
								l1_shared_count = shared_count;
							}
							if level > 0 {
								if cache_info.cache_line_bytes >= all_info[level - 1].cache_line_bytes {
									all_info[level - 1].associativity = cache_info.associativity;
									all_info[level - 1].cache_line_bytes = cache_info.cache_line_bytes;
									all_info[level - 1].cache_bytes = cache_info.cache_bytes / shared_count;
								}
							}
						}
					}
					for info in &mut all_info {
						info.cache_bytes *= l1_shared_count;
					}
					all_info[2].cache_bytes *= num_cpus::get_physical();

					Ok(all_info)
				}
				if let Ok(info) = try_cache_info_linux() {
					return Some(info);
				}

				if let Ok(lscpu) = std::process::Command::new("lscpu")
					.arg("-B")
					.arg("-C=type,level,ways,coherency-size,one-size")
					.output()
				{
					if lscpu.status.success() {
						if let Ok(lscpu) = String::from_utf8(lscpu.stdout).as_deref() {
							let mut info = CACHE_INFO_DEFAULT;
							for line in lscpu.lines().skip(1) {
								let mut iter = line.split_whitespace();
								if let [Some(cache_type), Some(level), Some(ways), Some(coherency_size), Some(one_size)] =
									[iter.next(), iter.next(), iter.next(), iter.next(), iter.next()]
								{
									if let "Data" | "Unified" = cache_type {
										let level = level.parse::<usize>().unwrap();
										let ways = ways.parse::<usize>().unwrap();
										let coherency_size = coherency_size.parse::<usize>().unwrap();
										let one_size = one_size.parse::<usize>().unwrap();

										info[level - 1].associativity = ways;
										info[level - 1].cache_line_bytes = coherency_size;
										info[level - 1].cache_bytes = one_size;
									}
								}
							}
							return Some(info);
						}
					}
				}
			}
			#[cfg(all(target_vendor = "apple", feature = "std"))]
			{
				use sysctl::{Ctl, Sysctl};

				let mut all_info = CACHE_INFO_DEFAULT;
				if let Ok(l1) = Ctl::new("hw.perflevel0.l1dcachesize").and_then(|ctl| ctl.value_string()) {
					if let Ok(l1) = l1.parse::<usize>() {
						all_info[0].cache_bytes = l1;
					}
				}
				if let (Ok(physicalcpu), Ok(cpusperl2), Ok(l2)) = (
					Ctl::new("hw.perflevel0.physicalcpu").and_then(|ctl| ctl.value_string()),
					Ctl::new("hw.perflevel0.cpusperl2").and_then(|ctl| ctl.value_string()),
					Ctl::new("hw.perflevel0.l2cachesize").and_then(|ctl| ctl.value_string()),
				) {
					if let (Ok(_physicalcpu), Ok(cpusperl2), Ok(l2)) = (physicalcpu.parse::<usize>(), cpusperl2.parse::<usize>(), l2.parse::<usize>())
					{
						all_info[1].cache_bytes = l2 / cpusperl2;
					}
				}
				all_info[2].cache_bytes = 0;
				return Some(all_info);
			}
		}

		#[cfg(any(
			all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
			all(target_arch = "x86_64", not(target_env = "sgx"))
		))]
		{
			use raw_cpuid::CpuId;
			let cpuid = CpuId::new();

			if let Some(vf) = cpuid.get_vendor_info() {
				let vf = vf.as_str();
				if vf == "GenuineIntel" {
					if let Some(cparams) = cpuid.get_cache_parameters() {
						// not sure why, intel cpus seem to prefer smaller mc
						let mut info = [CacheInfo {
							cache_bytes: 0,
							associativity: 0,
							cache_line_bytes: 64,
						}; 3];

						for cache in cparams {
							use raw_cpuid::CacheType::*;
							match cache.cache_type() {
								Null | Instruction | Reserved => continue,
								Data | Unified => {
									let level = cache.level() as usize;
									let associativity = cache.associativity();
									let nsets = cache.sets();
									let cache_line_bytes = cache.coherency_line_size();
									if level > 0 && level < 4 {
										let info = &mut info[level - 1];
										info.cache_line_bytes = cache_line_bytes;
										info.associativity = associativity;
										info.cache_bytes = associativity * nsets * cache_line_bytes;
									}
								},
							}
						}
						return Some(info);
					}
				}

				if vf == "AuthenticAMD" {
					if let Some(l1) = cpuid.get_l1_cache_and_tlb_info() {
						if let Some(l23) = cpuid.get_l2_l3_cache_and_tlb_info() {
							let compute_info = |associativity: raw_cpuid::Associativity, cache_kb: usize, cache_line_bytes: u8| -> CacheInfo {
								let cache_bytes = cache_kb as usize * 1024;
								let cache_line_bytes = cache_line_bytes as usize;

								use raw_cpuid::Associativity::*;
								let associativity = match associativity {
									Unknown | Disabled => {
										return CacheInfo {
											associativity: 0,
											cache_bytes: 0,
											cache_line_bytes: 64,
										};
									},
									FullyAssociative => cache_bytes / cache_line_bytes,
									DirectMapped => 1,
									NWay(n) => n as usize,
								};

								CacheInfo {
									associativity,
									cache_bytes,
									cache_line_bytes,
								}
							};
							return Some([
								compute_info(l1.dcache_associativity(), l1.dcache_size() as usize, l1.dcache_line_size()),
								compute_info(l23.l2cache_associativity(), l23.l2cache_size() as usize, l23.l2cache_line_size()),
								compute_info(l23.l3cache_associativity(), l23.l3cache_size() as usize * 512, l23.l3cache_line_size()),
							]);
						}
					}
				}
			}
		}
	}
	None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
	CacheInfo {
		associativity: 8,
		cache_bytes: 32 * 1024, // 32KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 256 * 1024, // 256KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 2 * 1024 * 1024, // 2MiB
		cache_line_bytes: 64,
	},
];

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
static CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
	CacheInfo {
		associativity: 8,
		cache_bytes: 64 * 1024, // 64KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 512 * 1024, // 512KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 4 * 1024 * 1024, // 4MiB
		cache_line_bytes: 64,
	},
];

#[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64", target_arch = "x86", target_arch = "x86_64")))]
static CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
	CacheInfo {
		associativity: 8,
		cache_bytes: 16 * 1024, // 16KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 512 * 1024, // 512KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 1024 * 1024, // 1MiB
		cache_line_bytes: 64,
	},
];

pub struct CacheInfoDeref;

impl core::ops::Deref for CacheInfoDeref {
	type Target = [CacheInfo; 3];

	#[cfg(feature = "std")]
	#[inline]
	fn deref(&self) -> &Self::Target {
		static CACHE_INFO: std::sync::OnceLock<[CacheInfo; 3]> = std::sync::OnceLock::new();

		CACHE_INFO.get_or_init(|| {
			let mut val = cache_info().unwrap_or(CACHE_INFO_DEFAULT);
			val[0].cache_bytes = Ord::max(val[0].cache_bytes, CACHE_INFO_DEFAULT[0].cache_bytes);

			val[1].cache_bytes = Ord::max(val[1].cache_bytes, CACHE_INFO_DEFAULT[1].cache_bytes);
			val[1].cache_bytes = Ord::max(val[1].cache_bytes, val[0].cache_bytes.checked_mul(4).unwrap());

			val[2].cache_bytes = Ord::max(val[2].cache_bytes, CACHE_INFO_DEFAULT[2].cache_bytes);
			val[2].cache_bytes = Ord::max(val[2].cache_bytes, val[1].cache_bytes.checked_mul(4).unwrap());
			val
		})
	}

	#[cfg(not(feature = "std"))]
	#[inline]
	fn deref(&self) -> &Self::Target {
		&CACHE_INFO_DEFAULT
	}
}

pub static CACHE_INFO: CacheInfoDeref = CacheInfoDeref;
