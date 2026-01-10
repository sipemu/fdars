#![allow(non_upper_case_globals)]
#![allow(dead_code, unused_variables)]

const M: usize = 4;
const N: usize = 32;

use core::cell::RefCell;
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use cache::CACHE_INFO;

include!(concat!(env!("OUT_DIR"), "/asm.rs"));

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Position {
	pub row: usize,
	pub col: usize,
}

mod cache;

const FLAGS_ACCUM: usize = 1 << 0;
const FLAGS_CONJ_LHS: usize = 1 << 1;
const FLAGS_CONJ_NEQ: usize = 1 << 2;
const FLAGS_LOWER: usize = 1 << 3;
const FLAGS_UPPER: usize = 1 << 4;
const FLAGS_32BIT_IDX: usize = 1 << 5;
const FLAGS_CPLX: usize = 1 << 62;
const FLAGS_ROWMAJOR: usize = 1 << 63;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MicrokernelInfo {
	pub flags: usize,
	pub depth: usize,
	pub lhs_rs: isize,
	pub lhs_cs: isize,
	pub rhs_rs: isize,
	pub rhs_cs: isize,
	pub alpha: *const (),

	// dst
	pub ptr: *mut (),
	pub rs: isize,
	pub cs: isize,
	pub row_idx: *const (),
	pub col_idx: *const (),

	// diag
	pub diag_ptr: *const (),
	pub diag_stride: isize,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MillikernelInfo {
	pub lhs_rs: isize,
	pub packed_lhs_rs: isize,
	pub rhs_cs: isize,
	pub packed_rhs_cs: isize,
	pub micro: MicrokernelInfo,
}

#[inline(always)]
unsafe fn pack_rhs_imp<T: Copy>(dst: *mut T, src: *const (), depth: usize, stride: usize, nr: usize, rs: isize, cs: isize) {
	for i in 0..depth {
		unsafe {
			let dst = dst.add(i * stride);
			let src = src.byte_offset(i as isize * rs);

			for j in 0..nr {
				let dst = dst.add(j);
				let src = src.byte_offset(j as isize * cs) as *const T;

				*dst = *src;
			}
		}
	}
}

#[inline(never)]
unsafe fn pack_rhs(dst: *mut (), src: *const (), depth: usize, nr: usize, rs: isize, cs: isize, sizeof: usize) {
	if !src.is_null() && src != dst as *const () {
		unsafe {
			match sizeof {
				4 => pack_rhs_imp(dst as *mut f32, src, depth, nr, nr, rs, cs),
				8 => pack_rhs_imp(dst as *mut [f32; 2], src, depth, nr, nr, rs, cs),
				16 => pack_rhs_imp(dst as *mut [f64; 2], src, depth, nr, nr, rs, cs),
				_ => unreachable!(),
			}
		}
	}
}

#[inline(always)]
pub unsafe fn call_microkernel(
	microkernel: unsafe extern "C" fn(),
	lhs: *const (),
	packed_lhs: *mut (),

	rhs: *const (),
	packed_rhs: *mut (),

	mut nrows: usize,
	mut ncols: usize,

	micro: &MicrokernelInfo,
	position: &mut Position,
) -> (usize, usize) {
	unsafe {
		core::arch::asm! {
			"call r10",

			in("rax") lhs,
			in("r15") packed_lhs,
			in("rcx") rhs,
			in("rdx") packed_rhs,
			in("rdi") position,
			in("rsi") micro,
			inout("r8") nrows,
			inout("r9") ncols,
			in("r10") microkernel,

			out("zmm0") _,
			out("zmm1") _,
			out("zmm2") _,
			out("zmm3") _,
			out("zmm4") _,
			out("zmm5") _,
			out("zmm6") _,
			out("zmm7") _,
			out("zmm8") _,
			out("zmm9") _,
			out("zmm10") _,
			out("zmm11") _,
			out("zmm12") _,
			out("zmm13") _,
			out("zmm14") _,
			out("zmm15") _,
			out("zmm16") _,
			out("zmm17") _,
			out("zmm18") _,
			out("zmm19") _,
			out("zmm20") _,
			out("zmm21") _,
			out("zmm22") _,
			out("zmm23") _,
			out("zmm24") _,
			out("zmm25") _,
			out("zmm26") _,
			out("zmm27") _,
			out("zmm28") _,
			out("zmm29") _,
			out("zmm30") _,
			out("zmm31") _,
			out("k1") _,
			out("k2") _,
			out("k3") _,
			out("k4") _,
		}
	}
	(nrows, ncols)
}

pub unsafe fn millikernel_rowmajor(
	microkernel: unsafe extern "C" fn(),
	pack: unsafe extern "C" fn(),
	mr: usize,
	nr: usize,
	sizeof: usize,

	lhs: *const (),
	packed_lhs: *mut (),

	rhs: *const (),
	packed_rhs: *mut (),

	nrows: usize,
	ncols: usize,

	milli: &MillikernelInfo,

	pos: &mut Position,
) {
	let mut rhs = rhs;
	let mut nrows = nrows;
	let mut lhs = lhs;
	let mut packed_lhs = packed_lhs;

	let tril = milli.micro.flags & FLAGS_LOWER != 0;
	let triu = milli.micro.flags & FLAGS_UPPER != 0;
	let rectangular = !tril && !triu;

	loop {
		let rs = milli.micro.lhs_rs;
		unsafe {
			let mut rhs = rhs;
			let mut packed_rhs = packed_rhs;
			let mut ncols = ncols;
			let mut lhs = lhs;
			let col = pos.col;

			macro_rules! iter {
                ($($lhs: ident)?) => {{
                    $({
                        let _ = $lhs;
                        if lhs != packed_lhs && !lhs.is_null() && (!milli.micro.diag_ptr.is_null() || milli.micro.lhs_rs != sizeof as isize) {
                            pack_lhs(pack, milli, Ord::min(nrows, mr), packed_lhs, lhs, sizeof);
                            lhs = null();
                        }
                    })*

                    let row_chunk = Ord::min(nrows, mr);
                    let col_chunk = Ord::min(ncols, nr);

                    {
                        let mut rhs = rhs;
                        if rhs != packed_rhs && !rhs.is_null() {
                            pack_rhs(
                                packed_rhs,
                                rhs,
                                milli.micro.depth,
                                col_chunk,
                                milli.micro.rhs_rs,
                                milli.micro.rhs_cs,
                                sizeof,
                            );
                            rhs = null();
                        }


                        if rectangular || (tril && pos.row + mr > pos.col) || (triu && pos.col + col_chunk > pos.row) {
                            call_microkernel(
                                microkernel,
                                lhs,
                                packed_lhs,
                                rhs,
                                packed_rhs,
                                row_chunk,
                                col_chunk,
                                &milli.micro,
                                pos,
                            );
                        } else {
                            if lhs != packed_lhs && !lhs.is_null() {
                                pack_lhs(pack, milli, row_chunk, packed_lhs, lhs, sizeof);
                            }
                        }
                    }

                    pos.col += col_chunk;
                    ncols -= col_chunk;
                    if ncols == 0 {
                        pos.row += row_chunk;
                        nrows -= row_chunk;
                    }

                    if !rhs.is_null() {
                        rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
                    }
                    packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);

                    $(if lhs != packed_lhs {
                        $lhs = null();
                    })?
                }};
            }
			iter!(lhs);
			while ncols > 0 {
				iter!();
			}
			pos.col = col;
		}

		if !lhs.is_null() {
			lhs = lhs.wrapping_byte_offset(milli.lhs_rs);
		}
		packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs);
		if rhs != packed_rhs {
			rhs = null();
		}

		if nrows == 0 {
			break;
		}
	}
}

pub unsafe fn millikernel_colmajor(
	microkernel: unsafe extern "C" fn(),
	pack: unsafe extern "C" fn(),
	mr: usize,
	nr: usize,
	sizeof: usize,

	lhs: *const (),
	packed_lhs: *mut (),

	rhs: *const (),
	packed_rhs: *mut (),

	nrows: usize,
	ncols: usize,

	milli: &MillikernelInfo,

	pos: &mut Position,
) {
	let mut lhs = lhs;
	let mut ncols = ncols;
	let mut rhs = rhs;
	let mut packed_rhs = packed_rhs;

	let tril = milli.micro.flags & FLAGS_LOWER != 0;
	let triu = milli.micro.flags & FLAGS_UPPER != 0;
	let rectangular = !tril && !triu;

	let mut j = 0;

	loop {
		let cs = milli.micro.rhs_cs;
		unsafe {
			let mut lhs = lhs;
			let mut packed_lhs = packed_lhs;
			let mut nrows = nrows;
			let mut rhs = rhs;
			let row = pos.row;

			macro_rules! iter {
                ($($rhs: ident)?) => {{
                    {
                        let mut lhs = lhs;

                        let row_chunk = Ord::min(nrows, mr);
                        let col_chunk = Ord::min(ncols, nr);

                        if lhs != packed_lhs && !lhs.is_null() && (!milli.micro.diag_ptr.is_null() || milli.micro.lhs_rs != sizeof as isize) {
                            pack_lhs(pack, milli, row_chunk, packed_lhs, lhs, sizeof);
                            lhs = null();
                        }

                        $({
                            let _ = $rhs;
                            if rhs != packed_rhs && !rhs.is_null() {
                                pack_rhs(
                                    packed_rhs,
                                    rhs,
                                    milli.micro.depth,
                                    col_chunk,
                                    milli.micro.rhs_rs,
                                    milli.micro.rhs_cs,
                                    sizeof,
                                );
                                rhs = null();
                            }
                        })*
                        if rectangular || (tril && pos.row + mr > pos.col) || (triu && pos.col + col_chunk > pos.row) {
                            call_microkernel(
                                microkernel,
                                lhs,
                                packed_lhs,
                                rhs,
                                packed_rhs,
                                row_chunk,
                                col_chunk,
                                &milli.micro,
                                pos,
                            );
                        } else {
                            if lhs != packed_lhs && !lhs.is_null() {
                                pack_lhs(pack, milli, row_chunk, packed_lhs, lhs, sizeof);
                            }
                        }

                        pos.row += row_chunk;
                        nrows -= row_chunk;
                        if nrows == 0 {
                            pos.col += col_chunk;
                            ncols -= col_chunk;
                        }
                    }

                    if !lhs.is_null() {
                        lhs = lhs.wrapping_byte_offset(milli.lhs_rs);
                    }
                    packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs);

                    $(if rhs != packed_rhs {
                        $rhs = null();
                    })?
                }};
            }
			iter!(rhs);
			while nrows > 0 {
				iter!();
			}
			pos.row = row;
		}

		if !rhs.is_null() {
			rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
		}
		packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);
		if lhs != packed_lhs {
			lhs = null();
		}

		j += 1;
		if ncols == 0 {
			break;
		}
	}
}

pub unsafe fn millikernel_par(
	thd_id: usize,
	n_threads: usize,

	microkernel_job: &[AtomicU8],
	pack_lhs_job: &[AtomicU8],
	pack_rhs_job: &[AtomicU8],
	finished: &AtomicUsize,
	hyper: usize,

	mr: usize,
	nr: usize,
	sizeof: usize,

	mf: usize,
	nf: usize,

	microkernel: unsafe extern "C" fn(),
	pack: unsafe extern "C" fn(),

	lhs: *const (),
	packed_lhs: *mut (),

	rhs: *const (),
	packed_rhs: *mut (),

	nrows: usize,
	ncols: usize,

	milli: &MillikernelInfo,

	pos: Position,
	tall: bool,
) {
	let n_threads0 = nrows.div_ceil(mf * mr);
	let n_threads1 = ncols.div_ceil(nf * nr);

	let thd_id0 = thd_id % (n_threads0);
	let thd_id1 = thd_id / (n_threads0);

	let tril = milli.micro.flags & FLAGS_LOWER != 0;
	let triu = milli.micro.flags & FLAGS_UPPER != 0;
	let rectangular = !tril && !triu;

	let i = mf * thd_id0;
	let j = nf * thd_id1;

	let colmajor = !tall;

	for ij in 0..mf * nf {
		let (i, j) = if colmajor {
			(i + ij % mf, j + ij / mf)
		} else {
			(i + ij / nf, j + ij % nf)
		};

		let row = Ord::min(nrows, i * mr);
		let col = Ord::min(ncols, j * nr);

		let row_chunk = Ord::min(nrows - row, mr);
		let col_chunk = Ord::min(ncols - col, nr);

		if row_chunk == 0 || col_chunk == 0 {
			continue;
		}

		let packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs * i as isize);
		let packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs * j as isize);

		let mut lhs = lhs;
		let mut rhs = rhs;

		{
			if !lhs.is_null() {
				lhs = lhs.wrapping_byte_offset(milli.lhs_rs * i as isize);
			}

			if lhs != packed_lhs {
				let val = pack_lhs_job[i].load(Ordering::Acquire);

				if val == 2 {
					lhs = null();
				}
			}
		}

		{
			if !rhs.is_null() {
				rhs = rhs.wrapping_byte_offset(milli.rhs_cs * j as isize);
			}
			if rhs != packed_rhs {
				let val = pack_rhs_job[j].load(Ordering::Acquire);

				if val == 2 {
					rhs = null();
				}
			}

			unsafe {
				if lhs != packed_lhs && !lhs.is_null() && (!milli.micro.diag_ptr.is_null() || milli.micro.lhs_rs != sizeof as isize) {
					pack_lhs(pack, milli, row_chunk, packed_lhs, lhs, sizeof);

					lhs = null();
					pack_lhs_job[i].store(2, Ordering::Release);
				}
				if rhs != packed_rhs && !rhs.is_null() {
					pack_rhs(
						packed_rhs,
						rhs,
						milli.micro.depth,
						col_chunk,
						milli.micro.rhs_rs,
						milli.micro.rhs_cs,
						sizeof,
					);
					rhs = null();
					pack_rhs_job[j].store(2, Ordering::Release);
				}

				if rectangular || (tril && pos.row + mr > pos.col) || (triu && pos.col + col_chunk > pos.row) {
					call_microkernel(
						microkernel,
						lhs,
						packed_lhs,
						rhs,
						packed_rhs,
						row_chunk,
						col_chunk,
						&milli.micro,
						&mut Position {
							row: row + pos.row,
							col: col + pos.col,
						},
					);
				} else {
					if lhs != packed_lhs && !lhs.is_null() {
						pack_lhs(pack, milli, row_chunk, packed_lhs, lhs, sizeof);
					}
				}
			}

			if !lhs.is_null() && lhs != packed_lhs {
				pack_lhs_job[i].store(2, Ordering::Release);
			}
			if !rhs.is_null() && rhs != packed_rhs {
				pack_rhs_job[j].store(2, Ordering::Release);
			}
		}
	}
}

unsafe fn pack_lhs(pack: unsafe extern "C" fn(), milli: &MillikernelInfo, row_chunk: usize, packed_lhs: *mut (), lhs: *const (), sizeof: usize) {
	unsafe {
		{
			let mut dst_cs = row_chunk;
			core::arch::asm! {
				"call r10",
				in("r10") pack,
				in("rax") lhs,
				in("r15") packed_lhs,
				inout("r8") dst_cs,
				in("rsi") &milli.micro,

				out("zmm0") _,
				out("zmm1") _,
				out("zmm2") _,
				out("zmm3") _,
				out("zmm4") _,
				out("zmm5") _,
				out("zmm6") _,
				out("zmm7") _,
				out("zmm8") _,
				out("zmm9") _,
				out("zmm10") _,
				out("zmm11") _,
				out("zmm12") _,
				out("zmm13") _,
				out("zmm14") _,
				out("zmm15") _,
				out("zmm16") _,
				out("zmm17") _,
				out("zmm18") _,
				out("zmm19") _,
				out("zmm20") _,
				out("zmm21") _,
				out("zmm22") _,
				out("zmm23") _,
				out("zmm24") _,
				out("zmm25") _,
				out("zmm26") _,
				out("zmm27") _,
				out("zmm28") _,
				out("zmm29") _,
				out("zmm30") _,
				out("zmm31") _,
				out("k1") _,
				out("k2") _,
				out("k3") _,
				out("k4") _,
			};

			if milli.micro.lhs_rs != sizeof as isize && milli.micro.lhs_cs != sizeof as isize {
				for j in 0..milli.micro.depth {
					let dst = packed_lhs.byte_add(j * dst_cs);
					let src = lhs.byte_offset(j as isize * milli.micro.lhs_cs);
					let diag_ptr = milli.micro.diag_ptr.byte_offset(j as isize * milli.micro.diag_stride);

					if sizeof == 4 {
						let dst = dst as *mut f32;
						let src = src as *const f32;
						for i in 0..row_chunk {
							let dst = dst.add(i);
							let src = src.byte_offset(i as isize * milli.micro.lhs_rs);

							if diag_ptr.is_null() {
								*dst = *src;
							} else {
								*dst = *src * *(diag_ptr as *const f32);
							}
						}
					} else if sizeof == 16 {
						let dst = dst as *mut [f64; 2];
						let src = src as *const [f64; 2];
						for i in 0..row_chunk {
							let dst = dst.add(i);
							let src = src.byte_offset(i as isize * milli.micro.lhs_rs);

							if diag_ptr.is_null() {
								*dst = *src;
							} else {
								(*dst)[0] = (*src)[0] * *(diag_ptr as *const f64);
								(*dst)[1] = (*src)[1] * *(diag_ptr as *const f64);
							}
						}
					} else {
						if (milli.micro.flags >> 62) & 1 == 1 {
							let dst = dst as *mut [f32; 2];
							let src = src as *const [f32; 2];
							for i in 0..row_chunk {
								let dst = dst.add(i);
								let src = src.byte_offset(i as isize * milli.micro.lhs_rs);

								if diag_ptr.is_null() {
									*dst = *src;
								} else {
									(*dst)[0] = (*src)[0] * *(diag_ptr as *const f32);
									(*dst)[1] = (*src)[1] * *(diag_ptr as *const f32);
								}
							}
						} else {
							let dst = dst as *mut f64;
							let src = src as *const f64;
							for i in 0..row_chunk {
								let dst = dst.add(i);
								let src = src.byte_offset(i as isize * milli.micro.lhs_rs);

								if diag_ptr.is_null() {
									*dst = *src;
								} else {
									*dst = *src * *(diag_ptr as *const f64);
								}
							}
						}
					}
				}
			}
		}
	}
}

pub unsafe trait Millikernel {
	unsafe fn call(
		&mut self,

		microkernel: unsafe extern "C" fn(),
		pack: unsafe extern "C" fn(),

		lhs: *const (),
		packed_lhs: *mut (),

		rhs: *const (),
		packed_rhs: *mut (),

		nrows: usize,
		ncols: usize,

		milli: &MillikernelInfo,

		pos: Position,
	);
}

struct Milli {
	mr: usize,
	nr: usize,
	sizeof: usize,
}
#[cfg(feature = "rayon")]
struct MilliPar {
	mr: usize,
	nr: usize,
	hyper: usize,
	sizeof: usize,

	microkernel_job: Box<[AtomicU8]>,
	pack_lhs_job: Box<[AtomicU8]>,
	pack_rhs_job: Box<[AtomicU8]>,
	finished: AtomicUsize,
	n_threads: usize,
}

unsafe impl Millikernel for Milli {
	unsafe fn call(
		&mut self,

		microkernel: unsafe extern "C" fn(),
		pack: unsafe extern "C" fn(),

		lhs: *const (),
		packed_lhs: *mut (),

		rhs: *const (),
		packed_rhs: *mut (),

		nrows: usize,
		ncols: usize,

		milli: &MillikernelInfo,
		pos: Position,
	) {
		unsafe {
			(if milli.micro.flags >> 63 == 1 {
				millikernel_rowmajor
			} else {
				millikernel_colmajor
			})(
				microkernel,
				pack,
				self.mr,
				self.nr,
				self.sizeof,
				lhs,
				packed_lhs,
				rhs,
				packed_rhs,
				nrows,
				ncols,
				milli,
				&mut { pos },
			)
		}
	}
}

#[derive(Copy, Clone)]
pub struct ForceSync<T>(pub T);
unsafe impl<T> Sync for ForceSync<T> {}
unsafe impl<T> Send for ForceSync<T> {}

#[cfg(feature = "rayon")]
unsafe impl Millikernel for MilliPar {
	unsafe fn call(
		&mut self,

		microkernel: unsafe extern "C" fn(),
		pack: unsafe extern "C" fn(),

		lhs: *const (),
		packed_lhs: *mut (),

		rhs: *const (),
		packed_rhs: *mut (),

		nrows: usize,
		ncols: usize,

		milli: &MillikernelInfo,
		pos: Position,
	) {
		let lhs = ForceSync(lhs);
		let mut rhs = ForceSync(rhs);
		let packed_lhs = ForceSync(packed_lhs);
		let packed_rhs = ForceSync(packed_rhs);
		let milli = ForceSync(milli);

		self.microkernel_job.fill_with(|| AtomicU8::new(0));
		self.pack_lhs_job.fill_with(|| AtomicU8::new(0));
		self.pack_rhs_job.fill_with(|| AtomicU8::new(0));
		self.finished = AtomicUsize::new(0);

		let f = Ord::min(8, milli.0.micro.depth.div_ceil(64));
		let l3 = CACHE_INFO[2].cache_bytes / f;

		let tall = nrows >= l3;
		let wide = ncols >= 2 * nrows;

		let mut mf = Ord::clamp(nrows.div_ceil(self.mr).div_ceil(2 * self.n_threads), 2, 4);
		if tall {
			mf = 16 / f;
		}
		if wide {
			mf = 2;
		}
		let par_rows = nrows.div_ceil(mf * self.mr);
		let nf = Ord::clamp(ncols.div_ceil(self.nr).div_ceil(8 * self.n_threads) * par_rows, 1, 1024 / f);
		let nf = 32 / self.nr;

		let n = nrows.div_ceil(mf * self.mr) * ncols.div_ceil(nf * self.nr);

		let mr = self.mr;
		let nr = self.nr;

		if !rhs.0.is_null() && rhs.0 != packed_rhs.0 {
			let depth = { milli }.0.micro.depth;

			let div = depth / self.n_threads;
			let rem = depth % self.n_threads;

			if !wide {
				spindle::for_each_raw(self.n_threads, |j| {
					let mut start = j * div;
					if j <= rem {
						start += j;
					} else {
						start += rem;
					}
					let end = start + div + if j < rem { 1 } else { 0 };
					let milli = { milli }.0;

					for i in 0..ncols.div_ceil(nr) {
						let col = Ord::min(ncols, i * nr);
						let ncols = Ord::min(ncols - col, nr);

						let rs = ncols;
						let rhs = { rhs }.0.wrapping_byte_offset(milli.rhs_cs * i as isize);
						let packed_rhs = { packed_rhs }.0.wrapping_byte_offset(milli.packed_rhs_cs * i as isize);

						pack_rhs(
							packed_rhs.wrapping_byte_offset((start * rs * self.sizeof) as isize),
							rhs.wrapping_byte_offset(start as isize * milli.micro.rhs_rs),
							end - start,
							ncols,
							milli.micro.rhs_rs,
							milli.micro.rhs_cs,
							self.sizeof,
						);
					}
				});
				rhs.0 = null();
			}
		}

		let gtid = AtomicUsize::new(0);

		spindle::for_each_raw(self.n_threads, |_| unsafe {
			loop {
				let tid = gtid.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
				if tid >= n {
					return;
				}
				let milli = { milli }.0;

				millikernel_par(
					tid,
					n,
					&self.microkernel_job,
					&self.pack_lhs_job,
					&self.pack_rhs_job,
					&self.finished,
					self.hyper,
					self.mr,
					self.nr,
					self.sizeof,
					mf,
					nf,
					microkernel,
					pack,
					{ lhs }.0,
					{ packed_lhs }.0,
					{ rhs }.0,
					{ packed_rhs }.0,
					nrows,
					ncols,
					milli,
					pos,
					tall,
				);
			}
		});
	}
}

#[inline(never)]
unsafe fn kernel_imp(
	millikernel: &mut dyn Millikernel,

	microkernel: &[unsafe extern "C" fn()],
	pack: &[unsafe extern "C" fn()],

	mr: usize,
	nr: usize,

	lhs: *const (),
	packed_lhs: *mut (),

	rhs: *const (),
	packed_rhs: *mut (),

	nrows: usize,
	ncols: usize,

	row_chunk: &[usize],
	col_chunk: &[usize],
	lhs_rs: &[isize],
	rhs_cs: &[isize],
	packed_lhs_rs: &[isize],
	packed_rhs_cs: &[isize],

	row: usize,
	col: usize,

	pos: Position,
	info: &MicrokernelInfo,
) {
	let _ = mr;

	let mut stack: [(
		*const (),
		*mut (),
		*const (),
		*mut (),
		usize,
		usize,
		usize,
		usize,
		usize,
		usize,
		usize,
		usize,
		bool,
		bool,
		bool,
		bool,
	); 16] = const { [(null(), null_mut(), null(), null_mut(), 0, 0, 0, 0, 0, 0, 0, 0, false, false, false, false); 16] };

	stack[0] = (
		lhs, packed_lhs, rhs, packed_rhs, row, col, nrows, ncols, 0, 0, 0, 0, false, false, false, false,
	);

	let mut pos = pos;
	let mut depth = 0;
	let max_depth = row_chunk.len();

	let milli_rs = *lhs_rs.last().unwrap();
	let milli_cs = *rhs_cs.last().unwrap();

	let micro_rs = info.lhs_rs;
	let micro_cs = info.rhs_cs;

	let milli = MillikernelInfo {
		lhs_rs: milli_rs,
		packed_lhs_rs: *packed_lhs_rs.last().unwrap(),
		rhs_cs: milli_cs,
		packed_rhs_cs: *packed_rhs_cs.last().unwrap(),
		micro: *info,
	};
	let microkernel = microkernel[nr - 1];
	let pack = pack[0];

	let q = row_chunk.len();
	let row_chunk = &row_chunk[..q - 1];
	let col_chunk = &col_chunk[..q - 1];
	let lhs_rs = &lhs_rs[..q];
	let packed_lhs_rs = &packed_lhs_rs[..q];
	let rhs_cs = &rhs_cs[..q];
	let packed_rhs_cs = &packed_rhs_cs[..q];

	loop {
		let (lhs, packed_lhs, rhs, packed_rhs, row, col, nrows, ncols, i, j, ii, jj, is_packed_lhs, is_packed_rhs, row_rev, col_rev) = stack[depth];
		let row_rev = false;
		let col_rev = false;

		if depth + 1 == max_depth {
			let mut lhs = lhs;
			let mut rhs = rhs;

			pos.row = row;
			pos.col = col;

			if is_packed_lhs && lhs != packed_lhs {
				lhs = null();
			}
			if is_packed_rhs && rhs != packed_rhs {
				rhs = null();
			}

			unsafe {
				millikernel.call(microkernel, pack, lhs, packed_lhs, rhs, packed_rhs, nrows, ncols, &milli, pos);
			}

			while depth > 0 {
				depth -= 1;

				let (_, _, _, _, _, _, nrows, ncols, i, j, ii, jj, _, _, _, _) = &mut stack[depth];

				let col_chunk = col_chunk[depth];
				let row_chunk = row_chunk[depth];

				let j_chunk = Ord::min(col_chunk, *ncols - *j);
				let i_chunk = Ord::min(row_chunk, *nrows - *i);

				if milli.micro.flags & FLAGS_ROWMAJOR == 0 {
					*i += i_chunk;
					*ii += 1;
					if *i == *nrows {
						*i = 0;
						*ii = 0;
						*j += j_chunk;
						*jj += 1;

						if *j == *ncols {
							if depth == 0 {
								return;
							}

							*j = 0;
							*jj = 0;
							continue;
						}
					}
				} else {
					*j += j_chunk;
					*jj += 1;
					if *j == *ncols {
						*j = 0;
						*jj = 0;
						*i += i_chunk;
						*ii += 1;

						if *i == *nrows {
							*i = 0;
							*ii = 0;
							if depth == 0 {
								return;
							}
							continue;
						}
					}
				}
				break;
			}
		} else {
			let col_chunk = col_chunk[depth];
			let row_chunk = row_chunk[depth];
			let rhs_cs = rhs_cs[depth];
			let lhs_rs = lhs_rs[depth];
			let prhs_cs = packed_rhs_cs[depth];
			let plhs_rs = packed_lhs_rs[depth];

			let last_row_chunk = if nrows == 0 { 0 } else { ((nrows - 1) % row_chunk) + 1 };

			let last_col_chunk = if ncols == 0 { 0 } else { ((ncols - 1) % col_chunk) + 1 };

			let (i, ii) = if row_rev {
				(nrows - last_row_chunk - i, nrows.div_ceil(row_chunk) - 1 - ii)
			} else {
				(i, ii)
			};

			let (j, jj) = if col_rev {
				(ncols - last_col_chunk - j, ncols.div_ceil(col_chunk) - 1 - jj)
			} else {
				(j, jj)
			};
			assert!(i as isize >= 0);
			assert!(j as isize >= 0);

			let j_chunk = Ord::min(col_chunk, ncols - j);
			let i_chunk = Ord::min(row_chunk, nrows - i);

			depth += 1;
			stack[depth] = (
				lhs.wrapping_byte_offset(lhs_rs * ii as isize),
				packed_lhs.wrapping_byte_offset(plhs_rs * ii as isize),
				rhs.wrapping_byte_offset(rhs_cs * jj as isize),
				packed_rhs.wrapping_byte_offset(prhs_cs * jj as isize),
				row + i,
				col + j,
				i_chunk,
				j_chunk,
				0,
				0,
				0,
				0,
				is_packed_lhs || (j > 0 && packed_lhs_rs[depth - 1] != 0),
				is_packed_rhs || (i > 0 && packed_rhs_cs[depth - 1] != 0),
				jj % 2 == 1,
				ii % 2 == 1,
			);
			continue;
		}
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InstrSet {
	Avx256,
	Avx512,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
	F32,
	F64,
	C32,
	C64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Accum {
	Replace,
	Add,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IType {
	U32,
	U64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DstKind {
	Lower,
	Upper,
	Full,
}

pub unsafe fn gemm(
	dtype: DType,
	itype: IType,

	instr: InstrSet,
	nrows: usize,
	ncols: usize,
	depth: usize,

	dst: *mut (),
	dst_rs: isize,
	dst_cs: isize,
	dst_row_idx: *const (),
	dst_col_idx: *const (),
	dst_kind: DstKind,

	beta: Accum,

	lhs: *const (),
	lhs_rs: isize,
	lhs_cs: isize,
	conj_lhs: bool,

	real_diag: *const (),
	diag_stride: isize,

	rhs: *const (),
	rhs_rs: isize,
	rhs_cs: isize,
	conj_rhs: bool,

	alpha: *const (),

	n_threads: usize,
) {
	let (sizeof, cplx) = match dtype {
		DType::F32 => (4, false),
		DType::F64 => (8, false),
		DType::C32 => (8, true),
		DType::C64 => (16, true),
	};
	let mut lhs_rs = lhs_rs * sizeof as isize;
	let mut lhs_cs = lhs_cs * sizeof as isize;
	let mut rhs_rs = rhs_rs * sizeof as isize;
	let mut rhs_cs = rhs_cs * sizeof as isize;
	let mut dst_rs = dst_rs * sizeof as isize;
	let mut dst_cs = dst_cs * sizeof as isize;
	let real_diag_stride = diag_stride * sizeof as isize;

	if nrows == 0 || ncols == 0 || (depth == 0 && beta == Accum::Add) {
		return;
	}

	let mut nrows = nrows;
	let mut ncols = ncols;

	let mut dst = dst;
	let mut dst_row_idx = dst_row_idx;
	let mut dst_col_idx = dst_col_idx;
	let mut dst_kind = dst_kind;

	let mut lhs = lhs;
	let mut conj_lhs = conj_lhs;

	let mut rhs = rhs;
	let mut conj_rhs = conj_rhs;

	if dst_rs.unsigned_abs() > dst_cs.unsigned_abs() {
		use core::mem::swap;
		swap(&mut dst_rs, &mut dst_cs);
		swap(&mut dst_row_idx, &mut dst_col_idx);
		dst_kind = match dst_kind {
			DstKind::Lower => DstKind::Upper,
			DstKind::Upper => DstKind::Lower,
			DstKind::Full => DstKind::Full,
		};
		swap(&mut lhs, &mut rhs);
		swap(&mut lhs_rs, &mut rhs_cs);
		swap(&mut lhs_cs, &mut rhs_rs);
		swap(&mut conj_lhs, &mut conj_rhs);
		swap(&mut nrows, &mut ncols);
	}

	if dst_rs < 0 && dst_kind == DstKind::Full && dst_row_idx.is_null() {
		dst = dst.wrapping_byte_offset((nrows - 1) as isize * dst_rs);
		lhs = lhs.wrapping_byte_offset((nrows - 1) as isize * lhs_rs);
		dst_rs = -dst_rs;
		lhs_rs = -lhs_rs;
	}

	if lhs_cs < 0 && depth > 0 {
		lhs = lhs.wrapping_byte_offset((depth - 1) as isize * lhs_cs);
		rhs = rhs.wrapping_byte_offset((depth - 1) as isize * rhs_rs);

		lhs_cs = -lhs_cs;
		rhs_rs = -rhs_rs;
	}

	let (microkernel, pack, mr, nr) = match (instr, dtype) {
		(InstrSet::Avx256, DType::F32) => (F32_SIMD256.as_slice(), F32_SIMDpack_256.as_slice(), 24, 4),
		(InstrSet::Avx256, DType::F64) => (F64_SIMD256.as_slice(), F64_SIMDpack_256.as_slice(), 12, 4),
		(InstrSet::Avx256, DType::C32) => (C32_SIMD256.as_slice(), C32_SIMDpack_256.as_slice(), 12, 4),
		(InstrSet::Avx256, DType::C64) => (C64_SIMD256.as_slice(), C64_SIMDpack_256.as_slice(), 6, 4),
		(InstrSet::Avx512, DType::F32) => (F32_SIMD512x4.as_slice(), F32_SIMDpack_512.as_slice(), 96, 4),
		(InstrSet::Avx512, DType::F64) => {
			if nrows > 48 {
				(F64_SIMD512x4.as_slice(), F64_SIMDpack_512.as_slice(), 48, 4)
			} else {
				(F64_SIMD512x8.as_slice(), F64_SIMDpack_512.as_slice(), 24, 8)
			}
		},
		(InstrSet::Avx512, DType::C32) => (C32_SIMD512x4.as_slice(), C32_SIMDpack_512.as_slice(), 48, 4),
		(InstrSet::Avx512, DType::C64) => (C64_SIMD512x4.as_slice(), C64_SIMDpack_512.as_slice(), 24, 4),
	};

	let m = nrows;
	let n = ncols;

	let kc = Ord::min(depth, 512);

	let cache = *cache::CACHE_INFO;

	let l1 = cache[0].cache_bytes / sizeof;
	let l2 = cache[1].cache_bytes / sizeof;
	let l3 = cache[2].cache_bytes / sizeof;

	#[repr(align(4096))]
	struct Page([u8; 4096]);

	let lhs_size = (l3.next_multiple_of(16) * sizeof).div_ceil(size_of::<Page>());
	let rhs_size = (l3.next_multiple_of(nr) * sizeof).div_ceil(size_of::<Page>());

	thread_local! {
		static MEM: RefCell<Vec::<core::mem::MaybeUninit<Page>>> = {
			let cache = *cache::CACHE_INFO;
			let l3 = cache[2].cache_bytes;

			let lhs_size = l3.div_ceil(size_of::<Page>());
			let rhs_size = l3.div_ceil(size_of::<Page>());

			let mut mem = Vec::with_capacity(lhs_size + rhs_size);
			unsafe { mem.set_len(lhs_size + rhs_size) };
			RefCell::new(mem)
		};
	}

	MEM.with(|mem| {
		let mut storage;
		let mut alloc;

		let mem = match mem.try_borrow_mut() {
			Ok(mem) => {
				storage = mem;
				&mut *storage
			},
			Err(_) => {
				alloc = Vec::with_capacity(lhs_size + rhs_size);

				&mut alloc
			},
		};
		if mem.len() < lhs_size + rhs_size {
			mem.reserve_exact(lhs_size + rhs_size);
			unsafe { mem.set_len(lhs_size + rhs_size) };
		}

		let (packed_lhs, packed_rhs) = mem.split_at_mut(lhs_size);
		let (packed_rhs, _) = packed_rhs.split_at_mut(rhs_size);

		let lhs = ForceSync(lhs);
		let rhs = ForceSync(rhs);
		let dst = ForceSync(dst);
		let real_diag = ForceSync(real_diag);
		let dst_row_idx = ForceSync(dst_row_idx);
		let dst_col_idx = ForceSync(dst_col_idx);
		let alpha = ForceSync(alpha);
		let mut f = || {
			let mut k = 0;
			let mut beta = beta;
			let mut lhs = { lhs }.0;
			let mut rhs = { rhs }.0;
			let mut real_diag = { real_diag }.0;
			let dst = { dst }.0;
			while k < depth {
				let kc = Ord::min(depth - k, kc);

				let f = kc.div_ceil(64);
				let l1 = l1 / 64 / f;
				let l2 = l2 / 64 / f;
				let l3 = l3 / 64 / f;

				let tall = m >= 3 * n / 2 && m >= l3;
				let pack_lhs = !real_diag.is_null() || (n > 6 * nr && tall) || (n > 3 * nr * n_threads) || lhs_rs != sizeof as isize;
				let pack_rhs = tall;

				let rowmajor = if n_threads > 1 {
					false
				} else if tall {
					true
				} else {
					false
				};

				let info = MicrokernelInfo {
					flags: match beta {
						Accum::Replace => 0,
						Accum::Add => FLAGS_ACCUM,
					} | if conj_lhs { FLAGS_CONJ_LHS } else { 0 }
						| if conj_lhs != conj_rhs { FLAGS_CONJ_NEQ } else { 0 }
						| match itype {
							IType::U32 => FLAGS_32BIT_IDX,
							IType::U64 => 0,
						} | if cplx { FLAGS_CPLX } else { 0 }
						| match dst_kind {
							DstKind::Lower => FLAGS_LOWER,
							DstKind::Upper => FLAGS_UPPER,
							DstKind::Full => 0,
						} | if rowmajor { FLAGS_ROWMAJOR } else { 0 },
					depth: kc,
					lhs_rs,
					lhs_cs,
					rhs_rs,
					rhs_cs,
					alpha: { alpha }.0,
					ptr: dst,
					rs: dst_rs,
					cs: dst_cs,
					row_idx: { dst_row_idx }.0,
					col_idx: { dst_col_idx }.0,
					diag_ptr: real_diag,
					diag_stride: real_diag_stride,
				};

				if n_threads <= 1 && !rowmajor && m < l2 && n < l2 {
					let microkernel = microkernel[nr - 1];
					let pack = pack[0];
					millikernel_colmajor(
						microkernel,
						pack,
						mr,
						nr,
						sizeof,
						lhs,
						if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs as _ },
						rhs,
						if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs as _ },
						nrows,
						ncols,
						&MillikernelInfo {
							lhs_rs: lhs_rs * mr as isize,
							packed_lhs_rs: if pack_lhs { (sizeof * mr * kc) as isize } else { lhs_rs * mr as isize },
							rhs_cs: rhs_cs * nr as isize,
							packed_rhs_cs: if pack_rhs { (sizeof * nr * kc) as isize } else { rhs_cs * nr as isize },
							micro: info,
						},
						&mut Position { row: 0, col: 0 },
					);
				} else {
					let (row_chunk, col_chunk, rowmajor) = if n_threads > 1 {
						(
							//
							[m, m, m, l3 / 16 * 16, mr],
							[n, n, n, l3 / 16 * 16, nr],
							false,
						)
					} else if true {
						(
							//
							[m, l3, l2, l2 / 2, mr],
							[n, 2 * l3, l3 / 2, l2, nr],
							true,
						)
					} else {
						(
							//
							[2 * l3, l3 / 2, l3 / 2, l2, mr],
							[l3, l3 / 2, l2 / 2, l1, nr],
							false,
						)
					};

					let mut row_chunk = row_chunk.map(|r| if r == mr { mr } else { r.next_multiple_of(16) });
					let mut col_chunk = col_chunk.map(|c| c.next_multiple_of(nr));

					let q = row_chunk.len();
					{
						for i in (1..q - 1).rev() {
							row_chunk[i - 1] = Ord::max(row_chunk[i - 1].next_multiple_of(row_chunk[i]), row_chunk[i]);
							if row_chunk[i - 1] > l3 / 2 && row_chunk[i - 1] < l3 {
								row_chunk[i - 1] = l3 / 2;
							}
							if row_chunk[i - 1] >= l3 {
								row_chunk[i - 1] = Ord::min(row_chunk[i - 1], 2 * row_chunk[i]);
							}
						}
						for i in (1..q - 1).rev() {
							col_chunk[i - 1] = Ord::max(col_chunk[i - 1].next_multiple_of(col_chunk[i]), col_chunk[i]);
							if col_chunk[i - 1] > l3 / 2 && col_chunk[i - 1] < l3 {
								col_chunk[i - 1] = l3 / 2;
							}
							if col_chunk[i - 1] >= l3 {
								col_chunk[i - 1] = Ord::min(col_chunk[i - 1], 2 * col_chunk[i]);
							}
						}
					}

					let all_lhs_rs = row_chunk.map(|m| m as isize * lhs_rs);
					let all_rhs_cs = col_chunk.map(|n| n as isize * rhs_cs);

					let mut packed_lhs_rs = row_chunk.map(|x| if x > l3 / 2 { 0 } else { (x * kc * sizeof) as isize });
					let mut packed_rhs_cs = col_chunk.map(|x| if x > l3 / 2 { 0 } else { (x * kc * sizeof) as isize });
					packed_lhs_rs[0] = 0;
					packed_rhs_cs[0] = 0;

					assert!(lhs_size * size_of::<Page>() >= row_chunk[q - 2] * kc * sizeof);
					assert!(rhs_size * size_of::<Page>() >= col_chunk[q - 2] * kc * sizeof);

					unsafe {
						kernel(
							n_threads,
							microkernel,
							pack,
							mr,
							nr,
							sizeof,
							lhs,
							if pack_lhs { packed_lhs.as_mut_ptr() as *mut () } else { lhs as *mut () },
							rhs,
							if pack_rhs { packed_rhs.as_mut_ptr() as *mut () } else { rhs as *mut () },
							nrows,
							ncols,
							&row_chunk,
							&col_chunk,
							&all_lhs_rs,
							&all_rhs_cs,
							if pack_lhs { &packed_lhs_rs } else { &all_lhs_rs },
							if pack_rhs { &packed_rhs_cs } else { &all_rhs_cs },
							0,
							0,
							Position { row: 0, col: 0 },
							&info,
						)
					};
				}

				k += kc;
				lhs = lhs.wrapping_byte_offset(lhs_cs * kc as isize);
				rhs = rhs.wrapping_byte_offset(rhs_rs * kc as isize);
				real_diag = real_diag.wrapping_byte_offset(real_diag_stride * kc as isize);

				beta = Accum::Add;
			}
		};
		if n_threads <= 1 {
			f();
		} else {
			#[cfg(feature = "rayon")]
			spindle::with_lock(n_threads, f);

			#[cfg(not(feature = "rayon"))]
			f();
		}
	});
}

pub unsafe fn kernel(
	n_threads: usize,
	microkernel: &[unsafe extern "C" fn()],
	pack: &[unsafe extern "C" fn()],

	mr: usize,
	nr: usize,
	sizeof: usize,

	lhs: *const (),
	packed_lhs: *mut (),

	rhs: *const (),
	packed_rhs: *mut (),

	nrows: usize,
	ncols: usize,

	row_chunk: &[usize],
	col_chunk: &[usize],
	lhs_rs: &[isize],
	rhs_cs: &[isize],
	packed_lhs_rs: &[isize],
	packed_rhs_cs: &[isize],

	row: usize,
	col: usize,

	pos: Position,
	info: &MicrokernelInfo,
) {
	unsafe {
		let mut seq = Milli { mr, nr, sizeof };
		#[cfg(feature = "rayon")]
		let mut par;
		kernel_imp(
			#[cfg(feature = "rayon")]
			if n_threads > 1 {
				par = {
					let max_i = nrows.div_ceil(mr);
					let max_j = ncols.div_ceil(nr);
					let max_jobs = max_i * max_j;
					let c = max_i;

					MilliPar {
						mr,
						nr,
						sizeof,
						hyper: 1,
						microkernel_job: (0..c * max_j).map(|_| AtomicU8::new(0)).collect(),
						pack_lhs_job: (0..max_i).map(|_| AtomicU8::new(0)).collect(),
						pack_rhs_job: (0..max_j).map(|_| AtomicU8::new(0)).collect(),
						finished: AtomicUsize::new(0),
						n_threads,
					}
				};
				&mut par
			} else {
				&mut seq
			},
			#[cfg(not(feature = "rayon"))]
			&mut seq,
			microkernel,
			pack,
			mr,
			nr,
			lhs,
			packed_lhs,
			rhs,
			packed_rhs,
			nrows,
			ncols,
			row_chunk,
			col_chunk,
			lhs_rs,
			rhs_cs,
			packed_lhs_rs,
			packed_rhs_cs,
			row,
			col,
			pos,
			info,
		)
	};
}

#[cfg(test)]
mod tests_f64 {
	use core::ptr::null_mut;

	use super::*;

	use aligned_vec::*;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<f64>() as isize;
		let len = 64 / size_of::<f64>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), 2.5.into()] {
					let alpha: f64 = alpha;
					for m in 1..=48usize {
						for n in (1..=4usize).chain([5]) {
							for cs in [m.next_multiple_of(48)] {
								let acs = m.next_multiple_of(48);
								let k = 2usize;

								let packed_lhs: &mut [f64] = &mut *avec![0.0.into(); acs * k];
								let packed_rhs: &mut [f64] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
								let lhs: &mut [f64] = &mut *avec![0.0.into(); cs * k];
								let rhs: &mut [f64] = &mut *avec![0.0.into(); n * k];
								let dst: &mut [f64] = &mut *avec![0.0.into(); cs * n];
								let target = &mut *avec![0.0.into(); cs * n];

								rng.fill(lhs);
								rng.fill(rhs);

								for i in 0..m {
									for j in 0..n {
										let target = &mut target[i + cs * j];
										let mut acc = 0.0.into();
										for depth in 0..k {
											acc = f64::mul_add(lhs[i + cs * depth], rhs[depth + k * j], acc);
										}
										*target = f64::mul_add(acc, alpha, *target);
									}
								}

								unsafe {
									millikernel_colmajor(
										F64_SIMD512x4[3],
										F64_SIMDpack_512[0],
										48,
										4,
										8,
										lhs.as_ptr() as _,
										if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
										rhs.as_ptr() as _,
										if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
										m,
										n,
										&mut MillikernelInfo {
											lhs_rs: 48 * sizeof,
											packed_lhs_rs: if pack_lhs { 48 * sizeof * k as isize } else { 48 * sizeof },
											rhs_cs: 4 * sizeof * k as isize,
											packed_rhs_cs: 4 * sizeof * k as isize,
											micro: MicrokernelInfo {
												flags: 0,
												depth: k,
												lhs_rs: 1 * sizeof,
												lhs_cs: cs as isize * sizeof,
												rhs_rs: 1 * sizeof,
												rhs_cs: k as isize * sizeof,
												alpha: &raw const alpha as _,
												ptr: dst.as_mut_ptr() as _,
												rs: 1 * sizeof,
												cs: cs as isize * sizeof,
												row_idx: null_mut(),
												col_idx: null_mut(),
												diag_ptr: null(),
												diag_stride: 0,
											},
										},
										&mut Position { row: 0, col: 0 },
									)
								};
								assert_eq!(dst, target);
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_gemm() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<f64>() as isize;
		let len = 64 / size_of::<f64>();

		for instr in [InstrSet::Avx256, InstrSet::Avx512] {
			for pack_lhs in [false, true] {
				for pack_rhs in [false] {
					for alpha in [1.0.into(), 0.0.into(), 2.5.into()] {
						let alpha: f64 = alpha;
						for m in (1..=48usize).chain([513]) {
							for n in (1..=4usize).chain([512]) {
								for cs in [m.next_multiple_of(48)] {
									let acs = m.next_multiple_of(48);
									let k = 513usize;

									let packed_lhs: &mut [f64] = &mut *avec![0.0.into(); acs * k];
									let packed_rhs: &mut [f64] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
									let lhs: &mut [f64] = &mut *avec![0.0.into(); cs * k];
									let rhs: &mut [f64] = &mut *avec![0.0.into(); n * k];
									let dst: &mut [f64] = &mut *avec![0.0.into(); cs * n];
									let target = &mut *avec![0.0.into(); cs * n];

									rng.fill(lhs);
									rng.fill(rhs);

									for i in 0..m {
										for j in 0..n {
											let target = &mut target[i + cs * j];
											let mut acc = 0.0.into();
											for depth in 0..k {
												acc = f64::mul_add(lhs[i + cs * depth], rhs[depth + k * j], acc);
											}
											*target = f64::mul_add(acc, alpha, *target);
										}
									}

									unsafe {
										gemm(
											DType::F64,
											IType::U64,
											instr,
											m,
											n,
											k,
											dst.as_mut_ptr() as _,
											1,
											cs as isize,
											null(),
											null(),
											DstKind::Full,
											Accum::Add,
											lhs.as_ptr() as _,
											1,
											cs as isize,
											false,
											null(),
											0,
											rhs.as_ptr() as _,
											1,
											k as isize,
											false,
											&raw const alpha as _,
											1,
										)
									};
									let mut i = 0;
									for (&target, &dst) in core::iter::zip(&*target, &*dst) {
										if !((target - dst).abs() < 1e-6) {
											dbg!(i / cs, i % cs, target, dst);
											panic!();
										}
										i += 1;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	#[test]
	fn test_avx512_kernel() {
		let m = 1023usize;
		let n = 1023usize;
		let k = 5usize;

		let rng = &mut StdRng::seed_from_u64(0);
		let sizeof = size_of::<f64>() as isize;
		let cs = m.next_multiple_of(8);
		let cs = Ord::max(4096, cs);

		let lhs: &mut [f64] = &mut *avec![0.0; cs * k];
		let rhs: &mut [f64] = &mut *avec![0.0; k * n];
		let target: &mut [f64] = &mut *avec![0.0; cs * n];

		rng.fill(lhs);
		rng.fill(rhs);

		unsafe {
			gemm::gemm(
				m,
				n,
				k,
				target.as_mut_ptr(),
				cs as isize,
				1,
				true,
				lhs.as_ptr(),
				cs as isize,
				1,
				rhs.as_ptr(),
				k as isize,
				1,
				1.0,
				1.0,
				false,
				false,
				false,
				gemm::Parallelism::None,
			);
		}

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				let dst = &mut *avec![0.0; cs * n];
				let packed_lhs = &mut *avec![0.0f64; m.next_multiple_of(8) * k];
				let packed_rhs = &mut *avec![0.0; if pack_rhs { n.next_multiple_of(4) * k } else { 0 }];

				unsafe {
					let row_chunk = [48 * 32, 48 * 16, 48];
					let col_chunk = [48 * 64, 48 * 32, 48, 4];

					let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
					let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
					let packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
					let packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

					kernel(
						1,
						&F64_SIMD512x4[..24],
						&F64_SIMDpack_512,
						48,
						4,
						size_of::<f64>(),
						lhs.as_ptr() as _,
						if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
						rhs.as_ptr() as _,
						if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
						m,
						n,
						&row_chunk,
						&col_chunk,
						&lhs_rs,
						&rhs_cs,
						&if pack_lhs { packed_lhs_rs } else { lhs_rs },
						&if pack_rhs { packed_rhs_cs } else { rhs_cs },
						0,
						0,
						Position { row: 0, col: 0 },
						&MicrokernelInfo {
							flags: 0,
							depth: k,
							lhs_rs: sizeof,
							lhs_cs: cs as isize * sizeof,
							rhs_rs: sizeof,
							rhs_cs: k as isize * sizeof,
							alpha: &raw const *&1.0f64 as _,
							ptr: dst.as_mut_ptr() as _,
							rs: sizeof,
							cs: cs as isize * sizeof,
							row_idx: null_mut(),
							col_idx: null_mut(),
							diag_ptr: null(),
							diag_stride: 0,
						},
					);
				}
				let mut i = 0;
				for (&target, &dst) in core::iter::zip(&*target, &*dst) {
					if !((target - dst).abs() < 1e-6) {
						dbg!(i / cs, i % cs, target, dst);
						panic!();
					}
					i += 1;
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_c64 {
	use super::*;

	use aligned_vec::*;
	use bytemuck::*;
	use core::ptr::null_mut;
	use gemm::c64;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c64>() as isize;
		let len = 64 / size_of::<c64>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c64::new(0.0, 3.5), c64::new(2.5, 3.5)] {
					let alpha: c64 = alpha;
					for m in 1..=24usize {
						for n in (1..=4usize).into_iter().chain([8]) {
							for cs in [m.next_multiple_of(len), m] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										let conj_different = conj_lhs != conj_rhs;

										let acs = m.next_multiple_of(len);
										let k = 1usize;

										let packed_lhs: &mut [c64] = &mut *avec![0.0.into(); acs * k];
										let packed_rhs: &mut [c64] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
										let lhs: &mut [c64] = &mut *avec![0.0.into(); cs * k];
										let rhs: &mut [c64] = &mut *avec![0.0.into(); n * k];
										let dst: &mut [c64] = &mut *avec![0.0.into(); cs * n];
										let target: &mut [c64] = &mut *avec![0.0.into(); cs * n];

										rng.fill(cast_slice_mut::<c64, f64>(lhs));
										rng.fill(cast_slice_mut::<c64, f64>(rhs));

										for i in 0..m {
											for j in 0..n {
												let target = &mut target[i + cs * j];
												let mut acc: c64 = 0.0.into();
												for depth in 0..k {
													let mut l = lhs[i + cs * depth];
													let mut r = rhs[depth + k * j];
													if conj_lhs {
														l = l.conj();
													}
													if conj_rhs {
														r = r.conj();
													}

													acc = l * r + acc;
												}
												*target = acc * alpha + *target;
											}
										}

										unsafe {
											millikernel_colmajor(
												C64_SIMD512x4[3],
												C64_SIMDpack_512[0],
												24,
												4,
												16,
												lhs.as_ptr() as _,
												if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
												rhs.as_ptr() as _,
												if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
												m,
												n,
												&mut MillikernelInfo {
													lhs_rs: 24 * sizeof,
													packed_lhs_rs: 24 * sizeof * k as isize,
													rhs_cs: 4 * sizeof * k as isize,
													packed_rhs_cs: 4 * sizeof * k as isize,
													micro: MicrokernelInfo {
														flags: ((conj_lhs as usize) << 1) | ((conj_different as usize) << 2),
														depth: k,
														lhs_rs: 1 * sizeof,
														lhs_cs: cs as isize * sizeof,
														rhs_rs: 1 * sizeof,
														rhs_cs: k as isize * sizeof,
														alpha: &raw const alpha as _,
														ptr: dst.as_mut_ptr() as _,
														rs: 1 * sizeof,
														cs: cs as isize * sizeof,
														row_idx: null_mut(),
														col_idx: null_mut(),
														diag_ptr: null(),
														diag_stride: 0,
													},
												},
												&mut Position { row: 0, col: 0 },
											)
										};
										let mut i = 0;
										for (&target, &dst) in core::iter::zip(&*target, &*dst) {
											if !((target - dst).norm_sqr().sqrt() < 1e-6) {
												dbg!(i / cs, i % cs, target, dst);
												panic!();
											}
											i += 1;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_f32 {
	use core::ptr::null_mut;

	use super::*;

	use aligned_vec::*;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<f32>() as isize;
		let len = 64 / size_of::<f32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), 2.5.into()] {
					let alpha: f32 = alpha;
					for m in 1..=96usize {
						for n in (1..=4usize).into_iter().chain([8]) {
							for cs in [m.next_multiple_of(len), m] {
								let acs = m.next_multiple_of(len);
								let k = 1usize;

								let packed_lhs: &mut [f32] = &mut *avec![0.0.into(); acs * k];
								let packed_rhs: &mut [f32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
								let lhs: &mut [f32] = &mut *avec![0.0.into(); cs * k];
								let rhs: &mut [f32] = &mut *avec![0.0.into(); n * k];
								let dst: &mut [f32] = &mut *avec![0.0.into(); cs * n];
								let target = &mut *avec![0.0.into(); cs * n];

								rng.fill(lhs);
								rng.fill(rhs);

								for i in 0..m {
									for j in 0..n {
										let target = &mut target[i + cs * j];
										let mut acc = 0.0.into();
										for depth in 0..k {
											acc = f32::mul_add(lhs[i + cs * depth], rhs[depth + k * j], acc);
										}
										*target = f32::mul_add(acc, alpha, *target);
									}
								}

								unsafe {
									millikernel_rowmajor(
										F32_SIMD512x4[3],
										F32_SIMDpack_512[0],
										96,
										4,
										4,
										lhs.as_ptr() as _,
										if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
										rhs.as_ptr() as _,
										if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
										m,
										n,
										&mut MillikernelInfo {
											lhs_rs: 96 * sizeof,
											packed_lhs_rs: 96 * sizeof * k as isize,
											rhs_cs: 4 * sizeof * k as isize,
											packed_rhs_cs: 4 * sizeof * k as isize,
											micro: MicrokernelInfo {
												flags: (1 << 63),
												depth: k,
												lhs_rs: 1 * sizeof,
												lhs_cs: cs as isize * sizeof,
												rhs_rs: 1 * sizeof,
												rhs_cs: k as isize * sizeof,
												alpha: &raw const alpha as _,
												ptr: dst.as_mut_ptr() as _,
												rs: 1 * sizeof,
												cs: cs as isize * sizeof,
												row_idx: null_mut(),
												col_idx: null_mut(),
												diag_ptr: null(),
												diag_stride: 0,
											},
										},
										&mut Position { row: 0, col: 0 },
									)
								};
								assert_eq!(dst, target);
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_avx512_kernel() {
		let m = 6000usize;
		let n = 2000usize;
		let k = 5usize;

		let rng = &mut StdRng::seed_from_u64(0);
		let sizeof = size_of::<f32>() as isize;
		let cs = m.next_multiple_of(16);
		let cs = Ord::max(4096, cs);

		let lhs: &mut [f32] = &mut *avec![0.0; cs * k];
		let rhs: &mut [f32] = &mut *avec![0.0; k * n];
		let target: &mut [f32] = &mut *avec![0.0; cs * n];

		rng.fill(lhs);
		rng.fill(rhs);

		unsafe {
			gemm::gemm(
				m,
				n,
				k,
				target.as_mut_ptr(),
				cs as isize,
				1,
				true,
				lhs.as_ptr(),
				cs as isize,
				1,
				rhs.as_ptr(),
				k as isize,
				1,
				1.0,
				1.0,
				false,
				false,
				false,
				gemm::Parallelism::None,
			);
		}

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				let dst = &mut *avec![0.0; cs * n];
				let packed_lhs = &mut *avec![0.0f32; m.next_multiple_of(16) * k];
				let packed_rhs = &mut *avec![0.0; if pack_rhs { n.next_multiple_of(4) * k } else { 0 }];

				unsafe {
					let row_chunk = [96 * 32, 96 * 16, 96 * 4, 96];
					let col_chunk = [1024, 256, 64, 16, 4];

					let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
					let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
					let packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
					let mut packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
					packed_rhs_cs[0] = 0;

					kernel(
						1,
						&F32_SIMD512x4[..24],
						&F32_SIMDpack_512,
						96,
						4,
						size_of::<f32>(),
						lhs.as_ptr() as _,
						if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
						rhs.as_ptr() as _,
						if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
						m,
						n,
						&row_chunk,
						&col_chunk,
						&lhs_rs,
						&rhs_cs,
						&if pack_lhs { packed_lhs_rs } else { lhs_rs },
						&if pack_rhs { packed_rhs_cs } else { rhs_cs },
						0,
						0,
						Position { row: 0, col: 0 },
						&MicrokernelInfo {
							flags: 0,
							depth: k,
							lhs_rs: sizeof,
							lhs_cs: cs as isize * sizeof,
							rhs_rs: sizeof,
							rhs_cs: k as isize * sizeof,
							alpha: &raw const *&1.0f32 as _,
							ptr: dst.as_mut_ptr() as _,
							rs: sizeof,
							cs: cs as isize * sizeof,
							row_idx: null_mut(),
							col_idx: null_mut(),
							diag_ptr: null(),
							diag_stride: 0,
						},
					)
				}
				let mut i = 0;
				for (&target, &dst) in core::iter::zip(&*target, &*dst) {
					if !((target - dst).abs() < 1e-6) {
						dbg!(i / cs, i % cs, target, dst);
						panic!();
					}
					i += 1;
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_c32 {
	use super::*;

	use aligned_vec::*;
	use bytemuck::*;
	use core::ptr::null_mut;
	use gemm::c32;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in (1..=4usize).into_iter().chain([8]) {
							for cs in [m.next_multiple_of(len), m] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}
											let conj_different = conj_lhs != conj_rhs;

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													let target = &mut target[i + cs * j];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[i + cs * depth];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD512x4[3],
													C32_SIMDpack_512[0],
													48,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 48 * sizeof,
														packed_lhs_rs: 48 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) << 1) | ((conj_different as usize) << 2),
															depth: k,
															lhs_rs: 1 * sizeof,
															lhs_cs: cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: 1 * sizeof,
															cs: cs as isize * sizeof,
															row_idx: null_mut(),
															col_idx: null_mut(),
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_c32_lower {
	use super::*;

	use aligned_vec::*;
	use bytemuck::*;
	use core::ptr::null_mut;
	use gemm::c32;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in (1..=4usize).chain([8, 32]) {
							for cs in [m, m.next_multiple_of(len)] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}
											let conj_different = conj_lhs != conj_rhs;

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													if i < j {
														continue;
													}
													let target = &mut target[i + cs * j];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[i + cs * depth];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD512x4[3],
													C32_SIMDpack_512[0],
													48,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 48 * sizeof,
														packed_lhs_rs: 48 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) << 1) | ((conj_different as usize) << 2) | (1 << 3),
															depth: k,
															lhs_rs: 1 * sizeof,
															lhs_cs: cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: 1 * sizeof,
															cs: cs as isize * sizeof,
															row_idx: null_mut(),
															col_idx: null_mut(),
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_avx256microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in (1..=4usize).chain([8, 32]) {
							for cs in [m, m.next_multiple_of(len)] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}

											let conj_different = conj_lhs != conj_rhs;

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													if i < j {
														continue;
													}
													let target = &mut target[i + cs * j];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[i + cs * depth];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD256[3],
													C32_SIMDpack_256[0],
													12,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 12 * sizeof,
														packed_lhs_rs: 12 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) << 1) | ((conj_different as usize) << 2) | (1 << 3),
															depth: k,
															lhs_rs: 1 * sizeof,
															lhs_cs: cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: 1 * sizeof,
															cs: cs as isize * sizeof,
															row_idx: null_mut(),
															col_idx: null_mut(),
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_c32_lower_add {
	use super::*;

	use aligned_vec::*;
	use bytemuck::*;
	use gemm::c64;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel_rowmajor() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c64>() as isize;
		let len = 64 / size_of::<c64>();

		for alpha in [1.0.into(), 0.0.into(), c64::new(0.0, 3.5), c64::new(2.5, 3.5)] {
			let alpha: c64 = alpha;
			for m in 1..=127usize {
				let m = 4005usize;
				for n in (1..=4usize).chain([8, 32, 1024]) {
					let n = 2usize;
					for cs in [m, m.next_multiple_of(len)] {
						for conj_lhs in [false, true] {
							for conj_rhs in [false, true] {
								for diag_scale in [true, false] {
									let conj_different = conj_lhs != conj_rhs;

									let acs = m.next_multiple_of(24);
									let k = 4005usize;
									dbg!(m, n, k, diag_scale, conj_lhs, conj_rhs);

									let packed_lhs: &mut [c64] = &mut *avec![0.0.into(); acs * k];
									let packed_rhs: &mut [c64] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
									let lhs: &mut [c64] = &mut *avec![0.0.into(); m * k];
									let rhs: &mut [c64] = &mut *avec![0.0.into(); n * k];
									let dst: &mut [c64] = &mut *avec![0.0.into(); cs * n];
									rng.fill(cast_slice_mut::<c64, f64>(dst));

									let target0: &mut [c64] = &mut *dst.to_vec();
									let target1: &mut [c64] = &mut *dst.to_vec();

									let diag: &mut [c64] = &mut *avec![0.0.into(); k];

									rng.fill(cast_slice_mut::<c64, f64>(lhs));
									rng.fill(cast_slice_mut::<c64, f64>(rhs));

									for x in &mut *diag {
										x.re = rng.random();
									}

									for i in 0..m {
										for j in 0..n {
											let target = &mut target0[i + cs * j];
											let mut acc: c64 = 0.0.into();
											for depth in 0..k {
												let mut l = lhs[i * k + depth];
												let mut r = rhs[depth + k * j];
												let d = diag[depth];

												if conj_lhs {
													l = l.conj();
												}
												if conj_rhs {
													r = r.conj();
												}

												if diag_scale {
													acc += d * l * r;
												} else {
													acc += l * r;
												}
											}
											*target = acc * alpha;
										}
									}

									unsafe {
										gemm(
											DType::C64,
											IType::U64,
											InstrSet::Avx512,
											m,
											n,
											k,
											dst.as_mut_ptr() as _,
											1,
											cs as isize,
											null(),
											null(),
											DstKind::Full,
											Accum::Replace,
											lhs.as_ptr() as _,
											k as isize,
											1,
											conj_lhs,
											if diag_scale { diag.as_ptr() as _ } else { null() },
											if diag_scale { 1 } else { 0 },
											rhs.as_ptr() as _,
											1,
											k as isize,
											conj_rhs,
											&raw const alpha as _,
											1,
										)
									};

									let mut i = 0;
									for (&target, &dst) in core::iter::zip(&*target0, &*dst) {
										if !((target - dst).norm_sqr().sqrt() < 1e-4) {
											dbg!(i / cs, i % cs, target, dst);
											panic!();
										}
										i += 1;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_avx512_microkernel_colmajor() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c64>() as isize;
		let len = 64 / size_of::<c64>();

		for alpha in [1.0.into(), 0.0.into(), c64::new(0.0, 3.5), c64::new(2.5, 3.5)] {
			let alpha: c64 = alpha;
			for m in [4005usize] {
				for n in [2usize] {
					for cs in [4008] {
						for conj_lhs in [false, true] {
							for conj_rhs in [false, true] {
								for diag_scale in [true, false] {
									let conj_different = conj_lhs != conj_rhs;

									let acs = m.next_multiple_of(24);
									let k = 4005usize;
									dbg!(m, n, k, diag_scale, conj_lhs, conj_rhs);

									let lhs: &mut [c64] = &mut *avec![0.0.into(); cs * k];
									let rhs: &mut [c64] = &mut *avec![0.0.into(); n * cs];
									let dst: &mut [c64] = &mut *avec![0.0.into(); cs * n];
									rng.fill(cast_slice_mut::<c64, f64>(dst));

									let target0: &mut [c64] = &mut *dst.to_vec();
									let target1: &mut [c64] = &mut *dst.to_vec();

									let diag: &mut [c64] = &mut *avec![0.0.into(); k];

									rng.fill(cast_slice_mut::<c64, f64>(lhs));
									rng.fill(cast_slice_mut::<c64, f64>(rhs));

									for x in &mut *diag {
										x.re = rng.random();
									}

									for i in 0..m {
										for j in 0..n {
											let target = &mut target0[i + cs * j];
											let mut acc: c64 = 0.0.into();
											for depth in 0..k {
												let mut l = lhs[i + cs * depth];
												let mut r = rhs[depth + cs * j];
												let d = diag[depth];

												if conj_lhs {
													l = l.conj();
												}
												if conj_rhs {
													r = r.conj();
												}

												if diag_scale {
													acc += d * l * r;
												} else {
													acc += l * r;
												}
											}
											*target = acc * alpha;
										}
									}

									unsafe {
										gemm(
											DType::C64,
											IType::U64,
											InstrSet::Avx512,
											m,
											n,
											k,
											dst.as_mut_ptr() as _,
											1,
											cs as isize,
											null(),
											null(),
											DstKind::Full,
											Accum::Replace,
											lhs.as_ptr() as _,
											1,
											cs as isize,
											conj_lhs,
											if diag_scale { diag.as_ptr() as _ } else { null() },
											if diag_scale { 1 } else { 0 },
											rhs.as_ptr() as _,
											1,
											cs as isize,
											conj_rhs,
											&raw const alpha as _,
											2,
										)
									};

									let mut i = 0;
									for (&target, &dst) in core::iter::zip(&*target0, &*dst) {
										if !((target - dst).norm_sqr().sqrt() < 1e-8) {
											dbg!(i / cs, i % cs, target, dst);
											panic!();
										}
										i += 1;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_c32_upper {
	use super::*;

	use aligned_vec::*;
	use bytemuck::*;
	use core::ptr::null_mut;
	use gemm::c32;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in [8].into_iter().chain(1..=4usize).chain([8]) {
							for cs in [m, m.next_multiple_of(len)] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}
											let conj_different = conj_lhs != conj_rhs;

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													if i > j {
														continue;
													}
													let target = &mut target[i + cs * j];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[i + cs * depth];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD512x4[3],
													C32_SIMDpack_512[0],
													48,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 48 * sizeof,
														packed_lhs_rs: 48 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) << 1) | ((conj_different as usize) << 2) | (1 << 4),
															depth: k,
															lhs_rs: 1 * sizeof,
															lhs_cs: cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: 1 * sizeof,
															cs: cs as isize * sizeof,
															row_idx: null_mut(),
															col_idx: null_mut(),
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod transpose_tests {
	use super::*;
	use aligned_vec::avec;
	use rand::prelude::*;

	#[test]
	fn test_b128() {
		let rng = &mut StdRng::seed_from_u64(0);

		for m in 1..=24 {
			let n = 127;

			let src = &mut *avec![0u128; m * n];
			let dst = &mut *avec![0u128; m.next_multiple_of(8) * n];

			rng.fill(src);
			rng.fill(dst);

			let ptr = C64_SIMDpack_512[(24 - m) / 4];
			let info = MicrokernelInfo {
				flags: 0,
				depth: n,
				lhs_rs: (n * size_of::<u128>()) as isize,
				lhs_cs: size_of::<u128>() as isize,
				rhs_rs: 0,
				rhs_cs: 0,
				alpha: null(),
				ptr: null_mut(),
				rs: 0,
				cs: 0,
				row_idx: null(),
				col_idx: null(),
				diag_ptr: null(),
				diag_stride: 0,
			};

			unsafe {
				core::arch::asm! {"
                call r10
                ",
					in("r10") ptr,
					in("rax") src.as_ptr(),
					in("r15") dst.as_mut_ptr(),
					in("r8") m,
					in("rsi") &info,
				};
			}

			for j in 0..n {
				for i in 0..m {
					assert_eq!(src[i * n + j], dst[i + m.next_multiple_of(4) * j]);
				}
			}
		}
	}

	#[test]
	fn test_b64() {
		let rng = &mut StdRng::seed_from_u64(0);

		for m in 1..=48 {
			let n = 127;

			let src = &mut *avec![0u64; m * n];
			let dst = &mut *avec![0u64; m.next_multiple_of(8) * n];

			rng.fill(src);
			rng.fill(dst);

			let ptr = F64_SIMDpack_512[(48 - m) / 8];
			let info = MicrokernelInfo {
				flags: 0,
				depth: n,
				lhs_rs: (n * size_of::<u64>()) as isize,
				lhs_cs: size_of::<u64>() as isize,
				rhs_rs: 0,
				rhs_cs: 0,
				alpha: null(),
				ptr: null_mut(),
				rs: 0,
				cs: 0,
				row_idx: null(),
				col_idx: null(),
				diag_ptr: null(),
				diag_stride: 0,
			};

			unsafe {
				core::arch::asm! {"
                call r10
                ",
					in("r10") ptr,
					in("rax") src.as_ptr(),
					in("r15") dst.as_mut_ptr(),
					in("r8") m,
					in("rsi") &info,
				};
			}

			for j in 0..n {
				for i in 0..m {
					assert_eq!(src[i * n + j], dst[i + m.next_multiple_of(8) * j]);
				}
			}
		}
	}

	#[test]
	fn test_b32() {
		let rng = &mut StdRng::seed_from_u64(0);

		for m in 1..=96 {
			let n = 127;

			let src = &mut *avec![0u32; m * n];
			let dst = &mut *avec![0u32; m.next_multiple_of(16) * n];

			rng.fill(src);
			rng.fill(dst);

			let ptr = F32_SIMDpack_512[(96 - m) / 16];
			let info = MicrokernelInfo {
				flags: 0,
				depth: n,
				lhs_rs: (n * size_of::<f32>()) as isize,
				lhs_cs: size_of::<f32>() as isize,
				rhs_rs: 0,
				rhs_cs: 0,
				alpha: null(),
				ptr: null_mut(),
				rs: 0,
				cs: 0,
				row_idx: null(),
				col_idx: null(),
				diag_ptr: null(),
				diag_stride: 0,
			};

			unsafe {
				core::arch::asm! {"
                call r10
                ",
					in("r10") ptr,
					in("rax") src.as_ptr(),
					in("r15") dst.as_mut_ptr(),
					in("r8") m,
					in("rsi") &info,
				};
			}

			for j in 0..n {
				for i in 0..m {
					assert_eq!(src[i * n + j], dst[i + m.next_multiple_of(16) * j]);
				}
			}
		}
	}
}

#[cfg(test)]
mod tests_c32_gather_scatter {
	use super::*;

	use aligned_vec::*;
	use bytemuck::*;
	use core::ptr::null_mut;
	use gemm::c32;
	use rand::prelude::*;

	#[test]
	fn test_avx512_microkernel() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in [8].into_iter().chain(1..=4usize).chain([8]) {
							for cs in [m, m.next_multiple_of(len)] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}

											let m = 2usize;
											let cs = m;
											let conj_different = conj_lhs != conj_rhs;

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													if i > j {
														continue;
													}
													let target = &mut target[2 * (i + cs * j)];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[i + cs * depth];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD512x4[3],
													C32_SIMDpack_512[0],
													48,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 48 * sizeof,
														packed_lhs_rs: 48 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) << 1) | ((conj_different as usize) << 2) | (1 << 4),
															depth: k,
															lhs_rs: 1 * sizeof,
															lhs_cs: cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: 2 * sizeof,
															cs: 2 * cs as isize * sizeof,
															row_idx: null_mut(),
															col_idx: null_mut(),
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_avx512_microkernel2() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [false, true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in [8].into_iter().chain(1..=4usize).chain([8]) {
							for cs in [m, m.next_multiple_of(len)] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}
											let m = 2usize;
											let cs = m;
											let conj_different = conj_lhs != conj_rhs;
											let idx = (0..Ord::max(m, n)).map(|i| 2 * i as u32).collect::<Vec<_>>();

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													if i > j {
														continue;
													}
													let target = &mut target[2 * (i + cs * j)];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[i + cs * depth];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD512x4[3],
													C32_SIMDpack_512[0],
													48,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 48 * sizeof,
														packed_lhs_rs: 48 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) << 1)
																| ((conj_different as usize) << 2) | (1 << 4) | (1 << 5),
															depth: k,
															lhs_rs: 1 * sizeof,
															lhs_cs: cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: sizeof,
															cs: cs as isize * sizeof,
															row_idx: idx.as_ptr() as _,
															col_idx: idx.as_ptr() as _,
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_avx512_microkernel3() {
		let rng = &mut StdRng::seed_from_u64(0);

		let sizeof = size_of::<c32>() as isize;
		let len = 64 / size_of::<c32>();

		for pack_lhs in [true] {
			for pack_rhs in [false] {
				for alpha in [1.0.into(), 0.0.into(), c32::new(0.0, 3.5), c32::new(2.5, 3.5)] {
					let alpha: c32 = alpha;
					for m in 1..=127usize {
						for n in [8].into_iter().chain(1..=4usize).chain([8]) {
							for cs in [m, m.next_multiple_of(len)] {
								for conj_lhs in [false, true] {
									for conj_rhs in [false, true] {
										for diag_scale in [false, true] {
											if diag_scale && !pack_lhs {
												continue;
											}
											let m = 2usize;
											let cs = m;
											let conj_different = conj_lhs != conj_rhs;
											let idx = (0..Ord::max(m, n)).map(|i| 2 * i as u32).collect::<Vec<_>>();

											let acs = m.next_multiple_of(len);
											let k = 1usize;

											let packed_lhs: &mut [c32] = &mut *avec![0.0.into(); acs * k];
											let packed_rhs: &mut [c32] = &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
											let lhs: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * k];
											let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
											let dst: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * n];
											let target: &mut [c32] = &mut *avec![0.0.into(); 2 * cs * n];

											let diag: &mut [f32] = &mut *avec![0.0.into(); k];

											rng.fill(cast_slice_mut::<c32, f32>(lhs));
											rng.fill(cast_slice_mut::<c32, f32>(rhs));
											rng.fill(diag);

											for i in 0..m {
												for j in 0..n {
													if i > j {
														continue;
													}
													let target = &mut target[2 * (i + cs * j)];
													let mut acc: c32 = 0.0.into();
													for depth in 0..k {
														let mut l = lhs[2 * (i + cs * depth)];
														let mut r = rhs[depth + k * j];
														let d = diag[depth];

														if conj_lhs {
															l = l.conj();
														}
														if conj_rhs {
															r = r.conj();
														}

														if diag_scale {
															acc += d * l * r;
														} else {
															acc += l * r;
														}
													}
													*target = acc * alpha + *target;
												}
											}

											unsafe {
												millikernel_colmajor(
													C32_SIMD512x4[3],
													C32_SIMDpack_512[0],
													48,
													4,
													8,
													lhs.as_ptr() as _,
													if pack_lhs { packed_lhs.as_mut_ptr() as _ } else { lhs.as_ptr() as _ },
													rhs.as_ptr() as _,
													if pack_rhs { packed_rhs.as_mut_ptr() as _ } else { rhs.as_ptr() as _ },
													m,
													n,
													&mut MillikernelInfo {
														lhs_rs: 48 * sizeof,
														packed_lhs_rs: 48 * sizeof * k as isize,
														rhs_cs: 4 * sizeof * k as isize,
														packed_rhs_cs: 4 * sizeof * k as isize,
														micro: MicrokernelInfo {
															flags: ((conj_lhs as usize) * FLAGS_CONJ_LHS)
																| ((conj_different as usize) * FLAGS_CONJ_NEQ) | (1 * FLAGS_UPPER)
																| (1 * FLAGS_32BIT_IDX) | (1 * FLAGS_CPLX),
															depth: k,
															lhs_rs: 2 * sizeof,
															lhs_cs: 2 * cs as isize * sizeof,
															rhs_rs: 1 * sizeof,
															rhs_cs: k as isize * sizeof,
															alpha: &raw const alpha as _,
															ptr: dst.as_mut_ptr() as _,
															rs: sizeof,
															cs: cs as isize * sizeof,
															row_idx: idx.as_ptr() as _,
															col_idx: idx.as_ptr() as _,
															diag_ptr: if diag_scale { diag.as_ptr() as *const () } else { null() },
															diag_stride: if diag_scale { size_of::<f32>() as isize } else { 0 },
														},
													},
													&mut Position { row: 0, col: 0 },
												)
											};
											let mut i = 0;
											for (&target, &dst) in core::iter::zip(&*target, &*dst) {
												if !((target - dst).norm_sqr().sqrt() < 1e-4) {
													dbg!(i / cs, i % cs, target, dst);
													panic!();
												}
												i += 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
