#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code, unused_labels, unused_macros)]

const INFO_FLAGS: isize = 0 * WORD;
const INFO_DEPTH: isize = 1 * WORD;
const INFO_LHS_RS: isize = 2 * WORD;
const INFO_LHS_CS: isize = 3 * WORD;
const INFO_RHS_RS: isize = 4 * WORD;
const INFO_RHS_CS: isize = 5 * WORD;
const INFO_ALPHA: isize = 6 * WORD;
const INFO_PTR: isize = 7 * WORD;
const INFO_RS: isize = 8 * WORD;
const INFO_CS: isize = 9 * WORD;
const INFO_ROW_IDX: isize = 10 * WORD;
const INFO_COL_IDX: isize = 11 * WORD;
const INFO_DIAG_PTR: isize = 12 * WORD;
const INFO_DIAG_STRIDE: isize = 13 * WORD;

use std::fmt::{Display, Write};
use std::ops::*;
use std::path::Path;
use std::sync::LazyLock;
use std::{env, fs};

use defer::defer;
use interpol::{format, println, writeln};
use std::cell::{Cell, RefCell};
use std::ops::Index;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! setup {
	($ctx: ident $(,)?) => {
		macro_rules! ctx {
			() => {
				$ctx
			};
		}
	};

	($ctx: ident, $target: ident $(,)?) => {
		macro_rules! ctx {
			() => {
				$ctx
			};
		}
		macro_rules! target {
			() => {
				$target
			};
		}
	};
}

macro_rules! align {
	() => {
		asm!(".p2align 4")
	};
}

macro_rules! func {
	($name: tt) => {
		let __name__ = &format!($name);

		align!();
		asm!("{__name__}:");
		defer!(asm!("ret"));

		macro_rules! name {
			() => {
				__name__
			};
		}
	};

	(pub $name: tt) => {
		let __name__ = &format!($name);

		asm!(".globl {__name__}");
		align!();
		asm!("{__name__}:");
		defer!(asm!("ret"));

		macro_rules! name {
			() => {
				__name__
			};
		}
	};
}

macro_rules! asm {
	($code: tt) => {{
		asm!($code, "");
	}};

	($code: tt, $comment: tt) => {{
		use std::fmt::Write;

		let code = &mut *ctx!().code.borrow_mut();

		::interpol::writeln!(code, $code).unwrap();
	}};
}

macro_rules! reg {
	($name: ident) => {
		let $name = ctx!().reg(::std::stringify!($name));
		::defer::defer!(ctx!().reg_drop($name, ::std::stringify!($name)));
	};

	(&$name: ident) => {
		$name = ctx!().reg(::std::stringify!($name));
		::defer::defer!(ctx!().reg_drop($name, ::std::stringify!($name)));
	};
}

macro_rules! label {
    ({$(let $label: ident;)*}) => {$(
        let $label = Cell::new(!ctx!().label(""));
        defer!({
            let mut __label__ = $label.get();
            if (__label__ as isize) < 0 {
                __label__ = !__label__;
            }
            ctx!().label_drop(__label__, "");
        });
    )*};

    ($label: ident) => {{
        let __label__ = $label.get();
        if __label__ as isize >= 0 {
            format!("{__label__}b")
        } else {
            format!("{!__label__}f")
        }
    }};

    ($label: ident = _) => {{
        let __label__ = !$label.get();
        assert!(__label__ as isize > 0);
        $label.set(__label__);

        align!();
        asm!("{__label__}:");
    }};
}

macro_rules! vxor {
	(zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {{
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vxor(dst, lhs, rhs)}"),
		}
	}};
}

macro_rules! vadd {
	(zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {{
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vadd_mem(dst, lhs, Addr::from(rhs))}"),
		}
	}};

	(zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {{
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vadd(dst, lhs, rhs)}"),
		}
	}};
}

macro_rules! vadds {
	(xmm($dst: expr), xmm($lhs: expr), xmm($rhs: expr) $(,)?) => {{
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vadds(dst, lhs, rhs)}"),
		}
	}};
}

macro_rules! vmov {
	([$dst: expr][$mask: expr $(,)?], zmm($src: expr) $(,)?) => {{
		match ($dst, $mask, $src) {
			(dst, mask, src) => asm!("{target!().vstoremask(mask, Addr::from(dst), src)}"),
		}
	}};
	([$dst: expr], zmm($src: expr) $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().vstore(Addr::from(dst), src)}"),
		}
	}};
	(zmm($dst: expr)[$mask: expr $(,)?], [$src: expr] $(,)?) => {{
		match ($dst, $mask, $src) {
			(dst, mask, src) => asm!("{target!().vloadmask(mask, dst, Addr::from(src))}"),
		}
	}};
	(zmm($dst: expr), [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().vload(dst, Addr::from(src))}"),
		}
	}};

	(zmm($dst: expr), zmm($src: expr) $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().vmov(dst, src)}"),
		}
	}};
}

macro_rules! vswap {
	(zmm($src: expr) $(,)?) => {{
		match $src {
			src => asm!("{target!().vswap(src)}"),
		}
	}};
}

macro_rules! kmov {
	(k($dst: expr), [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{ target!().kload(dst, Addr::from(src)) }"),
		}
	}};
}

macro_rules! kand {
	(k($dst: expr), k($lhs: expr), k($rhs: expr) $(,)?) => {{
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{ target!().kand(dst, lhs, rhs) }"),
		}
	}};
}

macro_rules! vmul {
	(zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {{
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vmul_mem(dst, lhs, Addr::from(rhs))}"),
		}
	}};

	(zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vmul(dst, lhs, rhs)}"),
		}
	};
}

macro_rules! vfma231 {
	(zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vfma231_mem(dst, lhs, Addr::from(rhs))}"),
		}
	};

	(zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vfma231(dst, lhs, rhs)}"),
		}
	};
}

macro_rules! vfma231_conj {
	(zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vfma231_conj_mem(dst, lhs, Addr::from(rhs))}"),
		}
	};

	(zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vfma231_conj(dst, lhs, rhs)}"),
		}
	};
}

macro_rules! vfma213 {
	(zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vfma213(dst, lhs, rhs)}"),
		}
	};
	(zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {
		match ($dst, $lhs, $rhs) {
			(dst, lhs, rhs) => asm!("{target!().vfma213_mem(dst, lhs, Addr::from(rhs))}"),
		}
	};
}

macro_rules! vbroadcast {
	(zmm($dst: expr), [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().real().vbroadcast(dst, Addr::from(src))}"),
		}
	}};
}

macro_rules! vmovs {
	([$dst: expr], xmm($src: expr) $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().scalar().vstore(Addr::from(dst), src)}"),
		}
	}};
	(xmm($dst: expr), [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().scalar().vload(dst, Addr::from(src))}"),
		}
	}};
}

macro_rules! vmovsr {
	([$dst: expr], xmm($src: expr) $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().real().scalar().vstore(Addr::from(dst), src)}"),
		}
	}};
	(xmm($dst: expr), [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("{target!().real().scalar().vload(dst, Addr::from(src))}"),
		}
	}};
}

macro_rules! alloca {
	([$reg: expr]) => {
		let __reg__ = Addr::from($reg);
		asm!("push qword ptr {__reg__}");
		defer!(asm!("pop qword ptr {__reg__}"));
	};
	($reg: expr) => {
		let __reg__ = $reg;
		asm!("push {__reg__}");
		defer!(asm!("pop {__reg__}"));
	};
}

macro_rules! cmovz {
	($dst: expr, $src: expr $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("cmovz {dst}, {src}"),
		}
	}};
}
macro_rules! cmovc {
	($dst: expr, $src: expr $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("cmovc {dst}, {src}"),
		}
	}};
}

macro_rules! cmovl {
	($dst: expr, $src: expr $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("cmovl {dst}, {src}"),
		}
	}};
}

macro_rules! cmovg {
	($dst: expr, $src: expr $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("cmovg {dst}, {src}"),
		}
	}};
}

macro_rules! mov {
	([$dst: expr], $src: expr $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("mov qword ptr { Addr::from(dst) }, {src}"),
		}
	}};

	($dst: expr, [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("mov {dst}, { Addr::from(src) }"),
		}
	}};

	($dst: expr, $src: expr $(,)?) => {{
		match ($dst, $src) {
			(dst, src) => asm!("mov {dst}, {src}"),
		}
	}};
}

macro_rules! movzx {
	($dst: expr, [$src: expr] $(,)?) => {{
		match ($dst, $src) {
			(dst, src) if dst >= r8 => asm!("mov {dst}d, dword ptr { Addr::from(src) }"),
			(dst, src) => {
				let dst = dst.to_string();
				let dst = dst.replace('r', "e");
				asm!("mov {dst}, dword ptr { Addr::from(src) }")
			},
		}
	}};
}

macro_rules! cmp {
	([$lhs: expr], $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("cmp qword ptr {Addr::from(lhs)}, {rhs}"),
		}
	}};

	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("cmp {lhs}, {rhs}"),
		}
	}};
}

macro_rules! test {
	([$lhs: expr], $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("test qword ptr {Addr::from(lhs)}, {rhs}"),
		}
	}};

	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("test {lhs}, {rhs}"),
		}
	}};
}

macro_rules! bt {
	([$lhs: expr], $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("bt qword ptr {Addr::from(lhs)}, {rhs}"),
		}
	}};

	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("bt {lhs}, {rhs}"),
		}
	}};
}

macro_rules! add {
	($lhs: expr, [$rhs: expr] $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("add {lhs}, qword ptr {Addr::from(rhs)}"),
		}
	}};

	([$lhs: expr], $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("add qword ptr {Addr::from(lhs)}, {rhs}"),
		}
	}};

	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("add {lhs}, {rhs}"),
		}
	}};
}

macro_rules! shl {
	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("shl {lhs}, {rhs}"),
		}
	}};
}

macro_rules! shr {
	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("shr {lhs}, {rhs}"),
		}
	}};
}

macro_rules! imul {
	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("imul {lhs}, {rhs}"),
		}
	}};
}

macro_rules! neg {
	($inout: expr $(,)?) => {{
		match $inout {
			inout => asm!("neg {inout}"),
		}
	}};
}

macro_rules! dec {
	($inout: expr $(,)?) => {{
		match $inout {
			inout => asm!("dec {inout}"),
		}
	}};
}

macro_rules! inc {
	($inout: expr $(,)?) => {{
		match $inout {
			inout => asm!("inc {inout}"),
		}
	}};
}

macro_rules! sub {
	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("sub {lhs}, {rhs}"),
		}
	}};
}

macro_rules! and {
	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("and {lhs}, {rhs}"),
		}
	}};
}
macro_rules! xor {
	($lhs: expr, $rhs: expr $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("xor {lhs}, {rhs}"),
		}
	}};
}

macro_rules! lea {
	($lhs: expr, [$rhs: expr] $(,)?) => {{
		match ($lhs, $rhs) {
			(lhs, rhs) => asm!("lea {lhs}, { Addr::from(rhs) }"),
		}
	}};
}

macro_rules! jmp {
    ($label: ident) => {{
        let __label__ = label!($label);
        asm!("jmp {__label__}");
    }};
    ($($label: tt)*) => {
        match format!($($label)*) {
            label => asm!("jmp {label}"),
        }
    };
}

macro_rules! call {
    ($($label: tt)*) => {
        match format!($($label)*) {
            label => asm!("call {label}"),
        }
    };
}

macro_rules! pop {
	($e: expr) => {
		match $e {
			e => asm!("pop {e}"),
		}
	};
}

macro_rules! jnz {
	($label: ident) => {
		let name = label!($label);
		asm!("jnz {name}");
	};
}

macro_rules! jl {
	($label: ident) => {
		let name = label!($label);
		asm!("jl {name}");
	};
}

macro_rules! jnl {
	($label: ident) => {
		let name = label!($label);
		asm!("jnl {name}");
	};
}

macro_rules! jg {
	($label: ident) => {
		let name = label!($label);
		asm!("jg {name}");
	};
}

macro_rules! jng {
	($label: ident) => {
		let name = label!($label);
		asm!("jng {name}");
	};
}

macro_rules! jz {
	($label: ident) => {
		let name = label!($label);
		asm!("jz {name}");
	};
}

macro_rules! jnc {
	($label: ident) => {
		let name = label!($label);
		asm!("jnc {name}");
	};
}

macro_rules! jc {
	($label: ident) => {
		let name = label!($label);
		asm!("jc {name}");
	};
}

macro_rules! brk {
	() => {
		asm!("int3");
	};
}

macro_rules! abort {
	() => {
		asm!("ud2");
	};
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Reg {
	rax = 0,
	rbx = 1,
	rcx = 2,
	rdx = 3,
	rdi = 4,
	rsi = 5,
	rbp = 6,
	rsp = 7,
	r8 = 8,
	r9 = 9,
	r10 = 10,
	r11 = 11,
	r12 = 12,
	r13 = 13,
	r14 = 14,
	r15 = 15,
	rip = 16,
}

#[derive(Copy, Clone, Debug)]
struct IndexScale {
	index: Reg,
	scale: isize,
}

#[derive(Copy, Clone, Debug)]
struct PtrIndexScale {
	ptr: Reg,
	index: Reg,
	scale: isize,
}

#[derive(Copy, Clone, Debug)]
struct PtrOffset {
	ptr: Reg,
	offset: isize,
}

#[derive(Copy, Clone, Debug)]
struct PtrStatic<'a> {
	ptr: Reg,
	offset: &'a str,
}

#[derive(Copy, Clone, Debug)]
struct Addr<'a> {
	ptr: Reg,
	index: Reg,
	scale: isize,
	offset: isize,
	static_offset: Option<&'a str>,
}

impl Mul<isize> for Reg {
	type Output = IndexScale;

	fn mul(self, rhs: isize) -> Self::Output {
		IndexScale { index: self, scale: rhs }
	}
}

impl Mul<Reg> for isize {
	type Output = IndexScale;

	fn mul(self, rhs: Reg) -> Self::Output {
		rhs * self
	}
}

impl Add<isize> for Reg {
	type Output = PtrOffset;

	fn add(self, rhs: isize) -> Self::Output {
		PtrOffset { ptr: self, offset: rhs }
	}
}

impl<'a> Add<&'a str> for Reg {
	type Output = PtrStatic<'a>;

	fn add(self, rhs: &'a str) -> Self::Output {
		PtrStatic { ptr: self, offset: rhs }
	}
}

impl<'a> Add<&'a String> for Reg {
	type Output = PtrStatic<'a>;

	fn add(self, rhs: &'a String) -> Self::Output {
		self + &**rhs
	}
}

impl Add<IndexScale> for Reg {
	type Output = PtrIndexScale;

	fn add(self, rhs: IndexScale) -> Self::Output {
		PtrIndexScale {
			ptr: self,
			index: rhs.index,
			scale: rhs.scale,
		}
	}
}

impl Add<isize> for PtrIndexScale {
	type Output = Addr<'static>;

	fn add(self, rhs: isize) -> Self::Output {
		Addr {
			ptr: self.ptr,
			index: self.index,
			scale: self.scale,
			offset: rhs,
			static_offset: None,
		}
	}
}

impl<'a> Add<isize> for PtrStatic<'a> {
	type Output = Addr<'a>;

	fn add(self, rhs: isize) -> Self::Output {
		Addr {
			ptr: self.ptr,
			index: rsp,
			scale: 0,
			offset: rhs,
			static_offset: Some(self.offset),
		}
	}
}

impl From<Reg> for Addr<'_> {
	fn from(value: Reg) -> Self {
		Self {
			ptr: value,
			index: rsp,
			scale: 0,
			offset: 0,
			static_offset: None,
		}
	}
}

impl From<PtrIndexScale> for Addr<'_> {
	fn from(value: PtrIndexScale) -> Self {
		Self {
			ptr: value.ptr,
			index: value.index,
			scale: value.scale,
			offset: 0,
			static_offset: None,
		}
	}
}

impl From<PtrOffset> for Addr<'_> {
	fn from(value: PtrOffset) -> Self {
		Self {
			ptr: value.ptr,
			index: rsp,
			scale: 0,
			offset: value.offset,
			static_offset: None,
		}
	}
}

impl<'a> From<PtrStatic<'a>> for Addr<'a> {
	fn from(value: PtrStatic<'a>) -> Self {
		Self {
			ptr: value.ptr,
			index: rsp,
			scale: 0,
			offset: 0,
			static_offset: Some(value.offset),
		}
	}
}

use Reg::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Float {
	F32,
	C32,
	F64,
	C64,
}

impl Reg {
	const ALL: &[Self] = &[
		Self::rax,
		Self::rbx,
		Self::rcx,
		Self::rdx,
		Self::rdi,
		Self::rsi,
		Self::rbp,
		Self::rsp,
		Self::r8,
		Self::r9,
		Self::r10,
		Self::r11,
		Self::r12,
		Self::r13,
		Self::r14,
		Self::r15,
	];
}

impl Float {
	fn sizeof(self) -> isize {
		match self {
			Float::F32 => 4,
			Float::C32 => 8,
			Float::F64 => 8,
			Float::C64 => 16,
		}
	}
}

impl std::fmt::Display for Reg {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		std::fmt::Debug::fmt(self, f)
	}
}

impl Index<Reg> for Ctx {
	type Output = Cell<bool>;

	fn index(&self, index: Reg) -> &Self::Output {
		&self.reg_busy[index as usize]
	}
}

type Code = RefCell<String>;

struct Ctx {
	reg_busy: [Cell<bool>; 16],
	label: Cell<usize>,
	code: Code,
}

impl Ctx {
	fn new() -> Self {
		Self {
			reg_busy: [const { Cell::new(false) }; 16],
			label: Cell::new(200000),
			code: RefCell::new(String::new()),
		}
	}

	#[track_caller]
	fn reg(&self, _: &str) -> Reg {
		setup!(self);

		for &reg in Reg::ALL {
			if !self[reg].get() {
				asm!("push {reg}");
				self[reg].set(true);
				return reg;
			}
		}

		panic!();
	}

	fn reg_drop(&self, reg: Reg, _: &str) {
		setup!(self);

		self[reg].set(false);
		asm!("pop {reg}");
	}

	fn label(&self, _: &str) -> usize {
		let label = self.label.get();
		self.label.set(label + 1);
		label
	}

	fn label_drop(&self, label: usize, _: &str) {
		self.label.set(label);
	}
}

const VERSION_MAJOR: usize = 0;

const PRETTY: LazyLock<bool> = LazyLock::new(|| false);
const PREFIX: LazyLock<String> = LazyLock::new(|| {
	if *PRETTY {
		format!("[gemm.x86 v{VERSION_MAJOR}]")
	} else {
		format!("gemm_v{VERSION_MAJOR}")
	}
});
const WORD: isize = 8;
const QUOTE: char = '"';

fn func_name(pieces: &str, params: &str, quote: bool) -> String {
	let pieces = pieces.split('.').collect::<Vec<_>>();
	let params = params
		.split('.')
		.filter_map(|p| {
			if p.is_empty() {
				return None;
			}

			let mut iter = p.split('=');
			Some((iter.next().unwrap(), iter.next().unwrap()))
		})
		.collect::<Vec<_>>();

	if *PRETTY {
		let name = pieces.iter().map(|i| i.as_ref()).collect::<Vec<_>>().join(".");
		let params = params.iter().map(|(k, v)| format!("{k} = {v}")).collect::<Vec<_>>().join(", ");

		if params.is_empty() {
			format!("{QUOTE}{*PREFIX} {name}{QUOTE}")
		} else {
			format!("{QUOTE}{*PREFIX} {name} [with {params}]{QUOTE}")
		}
	} else {
		let name = pieces.iter().map(|i| i.as_ref()).collect::<Vec<_>>().join("_");
		let params = params.iter().map(|(k, v)| format!("{k}{v}")).collect::<Vec<_>>().join("_");

		let name = if params.is_empty() {
			format!("{*PREFIX}_{name}")
		} else {
			format!("{*PREFIX}_{name}_{params}")
		};

		if quote { format!("{QUOTE}{name}{QUOTE}") } else { format!("{name}") }
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Simd {
	_512,
	_256,
	_128,
	_64,
	_32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Ty {
	F32,
	F64,
	C32,
	C64,
}

#[derive(Copy, Clone, Debug)]
struct Target {
	ty: Ty,
	simd: Simd,
}

impl Simd {
	fn dedicated_mask(self) -> bool {
		matches!(self, Simd::_512)
	}

	fn reg(self) -> String {
		match self {
			Simd::_512 => "zmm",
			Simd::_256 => "ymm",
			_ => "xmm",
		}
		.to_string()
	}

	fn num_regs(self) -> isize {
		if self.sizeof() == 64 { 32 } else { 16 }
	}

	fn sizeof(self) -> isize {
		match self {
			Simd::_512 => 512 / 8,
			Simd::_256 => 256 / 8,
			Simd::_128 => 128 / 8,
			Simd::_64 => 64 / 8,
			Simd::_32 => 32 / 8,
		}
	}
}

impl Ty {
	fn sizeof(self) -> isize {
		match self {
			Ty::F32 => 4,
			Ty::F64 => 8,
			Ty::C32 => 2 * 4,
			Ty::C64 => 2 * 8,
		}
	}

	fn suffix(self) -> String {
		match self {
			Ty::F32 | Ty::C32 => "s",
			Ty::F64 | Ty::C64 => "d",
		}
		.to_string()
	}
}

impl Display for Ty {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(match *self {
			Ty::F32 => "f32",
			Ty::F64 => "f64",
			Ty::C32 => "c32",
			Ty::C64 => "c64",
		})
	}
}

impl Display for Addr<'_> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let &Self {
			ptr,
			index,
			scale,
			offset,
			static_offset,
		} = self;

		let mut out = format!("{ptr}");

		if scale > 0 {
			assert!((scale as usize).is_power_of_two());
			assert!(scale <= 8);
			if scale == 1 {
				out = format!("{out} + {index}");
			} else {
				out = format!("{out} + {scale} * {index}");
			}
		}

		if offset > 0 {
			out = format!("{out} + {offset}");
		}
		if offset < 0 {
			out = format!("{out} - {-offset}");
		}

		if let Some(offset) = static_offset {
			assert_eq!(ptr, rip);
			out = format!("{out} + {offset}");
		}

		write!(f, "[{out}]")
	}
}

impl Target {
	fn load_imp(self, mask: Option<isize>, dst: isize, src: Addr) -> String {
		let Self { ty: _, simd } = self;

		let reg = simd.reg();

		let instr = match simd {
			Simd::_32 => format!("vmovss"),
			Simd::_64 => format!("vmovsd"),
			_ => {
				if mask.is_none() || simd.dedicated_mask() {
					format!("vmovup{self.ty.suffix()}")
				} else {
					format!("vmaskmovp{self.ty.suffix()}")
				}
			},
		};

		match (mask, simd) {
			(None, _) => format!("{instr} {reg}{dst}, {src}"),
			(Some(mask), Simd::_512) => {
				format!("{instr} {reg}{dst} {{{{k{mask}}}}}{{{{z}}}}, {src}")
			},
			(Some(mask), _) => {
				format!("{instr} {reg}{dst}, {reg}{mask}, {src}")
			},
		}
	}

	fn store_imp(self, mask: Option<isize>, dst: Addr, src: isize) -> String {
		let Self { ty: _, simd } = self;

		let reg = simd.reg();

		let instr = match simd {
			Simd::_32 => format!("vmovss"),
			Simd::_64 => format!("vmovsd"),
			_ => {
				if mask.is_none() || simd.dedicated_mask() {
					format!("vmovup{self.ty.suffix()}")
				} else {
					format!("vmaskmovp{self.ty.suffix()}")
				}
			},
		};

		match (mask, simd) {
			(None, _) => format!("{instr} {dst}, {reg}{src}"),
			(Some(mask), Simd::_512) => {
				format!("{instr} {dst} {{{{k{mask}}}}}, {reg}{src}")
			},
			(Some(mask), _) => {
				format!("{instr} {dst}, {reg}{mask}, {reg}{src}")
			},
		}
	}

	fn is_scalar(self) -> bool {
		matches!((self.ty, self.simd), (Ty::F32, Simd::_32) | (Ty::F64, Simd::_64),)
	}

	fn is_cplx(self) -> bool {
		matches!(self.ty, Ty::C32 | Ty::C64)
	}

	fn is_real(self) -> bool {
		!self.is_cplx()
	}

	fn real(self) -> Target {
		let ty = match self.ty {
			Ty::F32 | Ty::C32 => Ty::F32,
			Ty::F64 | Ty::C64 => Ty::F64,
		};
		Target { ty, simd: self.simd }
	}

	fn mask_sizeof(self) -> isize {
		if self.simd.dedicated_mask() {
			if self.is_cplx() { 2 * self.len() / 8 } else { self.len() / 8 }
		} else {
			self.simd.sizeof()
		}
	}

	fn len(self) -> isize {
		self.simd.sizeof() / self.ty.sizeof()
	}

	fn scalar_suffix(self) -> String {
		if !self.is_cplx() && self.is_scalar() { "s" } else { "p" }.to_string()
	}

	fn vswap(self, reg: isize) -> String {
		let bits = match self.ty {
			Ty::C64 => 0b01010101,
			Ty::C32 => 0b10110001,
			_ => panic!(),
		};
		format!("vpermilp{self.ty.suffix()} {self.simd.reg()}{reg}, {self.simd.reg()}{reg}, {bits}")
	}

	fn vmov(self, dst: isize, src: isize) -> String {
		format!("vmovaps {self.simd.reg()}{dst}, {self.simd.reg()}{src}")
	}

	fn vload(self, dst: isize, src: Addr) -> String {
		self.load_imp(None, dst, src)
	}

	fn vstore(self, dst: Addr, src: isize) -> String {
		self.store_imp(None, dst, src)
	}

	fn kload(self, dst: isize, src: Addr) -> String {
		if self.simd.dedicated_mask() {
			match self.ty {
				Ty::F32 | Ty::C32 => format!("kmovw k{dst}, {src}"),
				Ty::F64 | Ty::C64 => format!("kmovb k{dst}, {src}"),
			}
		} else {
			self.vload(dst, src)
		}
	}

	fn kand(self, dst: isize, lhs: isize, rhs: isize) -> String {
		if self.simd.dedicated_mask() {
			match self.ty {
				Ty::F32 | Ty::C32 => format!("kandw k{dst}, k{lhs}, k{rhs}"),
				Ty::F64 | Ty::C64 => format!("kandb k{dst}, k{lhs}, k{rhs}"),
			}
		} else {
			let Self { ty, simd } = self;
			let reg = simd.reg();
			format!("vandp{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		}
	}

	fn scalar(self) -> Self {
		Self {
			ty: self.ty,
			simd: match self.ty {
				Ty::F32 => Simd::_32,
				Ty::F64 => Simd::_64,
				Ty::C32 => Simd::_64,
				Ty::C64 => Simd::_128,
			},
		}
	}

	fn vloadmask(self, mask: isize, dst: isize, src: Addr) -> String {
		self.load_imp(Some(mask), dst, src)
	}

	fn vstoremask(self, mask: isize, dst: Addr, src: isize) -> String {
		self.store_imp(Some(mask), dst, src)
	}

	fn vxor(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vxorp{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
	}

	fn vadd(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vadd{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
	}

	fn vadds(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd: _ } = self;
		let reg = "xmm";
		if self.is_cplx() {
			format!("vaddp{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		} else {
			format!("vadds{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		}
	}

	fn vadd_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vadd{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs}")
	}

	fn vmul(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vmul{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
	}

	fn vmul_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vmul{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs}")
	}

	fn vfma231(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		if self.is_cplx() {
			format!("vfmaddsub231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		} else {
			format!("vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		}
	}

	fn vfma231_conj(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		if self.is_cplx() {
			format!("vfmsubadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		} else {
			format!("vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
		}
	}

	fn vfma231_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		if self.is_cplx() {
			format!("vfmaddsub231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}")
		} else {
			format!("vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}")
		}
	}

	fn vfma231_conj_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();

		if self.is_cplx() {
			format!("vfmsubadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}")
		} else {
			format!("vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}")
		}
	}

	fn vfma213(self, dst: isize, lhs: isize, rhs: isize) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vfmadd213{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
	}

	fn vfma213_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
		let Self { ty, simd } = self;
		let reg = simd.reg();
		format!("vfmadd213{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs}")
	}

	fn vbroadcast(self, dst: isize, src: Addr) -> String {
		let this = self.real();

		let instr = if this.is_scalar() {
			return this.vload(dst, src);
		} else if (this.ty, this.simd) == (Ty::F64, Simd::_128) {
			format!("vmovddup")
		} else {
			format!("vbroadcasts{this.ty.suffix()}")
		};

		format!("{instr} {this.simd.reg()}{dst}, {src}")
	}

	fn transpose(self, n_regs: usize) -> (String, usize, bool) {
		let mut t = self.simd.num_regs() as usize - 1;

		if n_regs == 1 {
			return (String::new(), t, false);
		}

		let ctx = Ctx::new();
		setup!(ctx, self);

		let unpacklo32 = "vunpcklps";
		let unpackhi32 = "vunpckhps";
		let unpacklo64 = "vunpcklpd";
		let unpackhi64 = "vunpckhpd";
		let permute2x128 = "vperm2f128";
		let shuffle64x2 = "vshuff64x2";

		let in_t = match self.ty {
			Ty::F32 => match n_regs {
				2 => {
					t = 15;
					let r = |i: usize| format!("xmm{i}");
					// 128 bit register, want to get low and high 64 bits
					asm!("{unpacklo32} {r(t)}, {r(0)}, {r(1)}");
					asm!("{unpacklo64} {r(0)}, {r(t)}, {r(t)}");
					asm!("{unpackhi64} {r(1)}, {r(t)}, {r(t)}");
					false
				},
				4 => {
					t = 15;
					let r = |i: usize| format!("xmm{i}");
					let t = |i: usize| format!("xmm{t - i}");

					for i in 0..2 {
						let t0 = t(2 * i + 0);
						let t1 = t(2 * i + 1);
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						asm!("{unpacklo32} {t0}, {r0}, {r1}");
						asm!("{unpackhi32} {t1}, {r0}, {r1}");
					}
					for i in 0..2 {
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						let t0 = t(i + 0);
						let t1 = t(i + 2);
						asm!("{unpacklo64} {r0}, {t0}, {t1}");
						asm!("{unpackhi64} {r1}, {t0}, {t1}");
					}
					false
				},
				8 => {
					t = 15;
					let r = |i: usize| format!("ymm{i}");
					let t = |i: usize| format!("ymm{t - i}");

					for i in 0..4 {
						let t0 = t(2 * i + 0);
						let t1 = t(2 * i + 1);
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						asm!("{unpacklo32} {t0}, {r0}, {r1}");
						asm!("{unpackhi32} {t1}, {r0}, {r1}");
					}
					for j in 0..2 {
						for i in 0..2 {
							let r0 = r(4 * j + 2 * i + 0);
							let r1 = r(4 * j + 2 * i + 1);
							let t0 = t(4 * j + i + 0);
							let t1 = t(4 * j + i + 2);
							asm!("{unpacklo64} {r0}, {t0}, {t1}");
							asm!("{unpackhi64} {r1}, {t0}, {t1}");
						}
					}

					let idx = [0b00100000, 0b00110001];
					for i in 0..4 {
						let r0 = r(i);
						let r1 = r(i + 4);
						let t0 = t(i);
						let t1 = t(i + 4);

						asm!("{permute2x128} {t0}, {r0}, {r1}, {idx[0]}");
						asm!("{permute2x128} {t1}, {r0}, {r1}, {idx[1]}");
					}
					true
				},
				16 => {
					let r = |i: usize| format!("zmm{i}");
					let t = |i: usize| format!("zmm{t - i}");

					for i in 0..8 {
						let t0 = t(2 * i + 0);
						let t1 = t(2 * i + 1);
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						asm!("{unpacklo32} {t0}, {r0}, {r1}");
						asm!("{unpackhi32} {t1}, {r0}, {r1}");
					}
					for j in 0..4 {
						for i in 0..2 {
							let r0 = r(4 * j + 2 * i + 0);
							let r1 = r(4 * j + 2 * i + 1);
							let t0 = t(4 * j + i + 0);
							let t1 = t(4 * j + i + 2);
							asm!("{unpacklo64} {r0}, {t0}, {t1}");
							asm!("{unpackhi64} {r1}, {t0}, {t1}");
						}
					}

					let idx = [0b10001000, 0b11011101];
					for j in 0..2 {
						for i in 0..4 {
							let r0 = r(8 * j + i + 0);
							let r1 = r(8 * j + i + 4);
							let t0 = t(8 * j + 2 * i + 0);
							let t1 = t(8 * j + 2 * i + 1);

							asm!("{shuffle64x2} {t0}, {r0}, {r1}, {idx[0]}");
							asm!("{shuffle64x2} {t1}, {r0}, {r1}, {idx[1]}");
						}
					}

					for j in 0..2 {
						for i in 0..4 {
							let t0 = t(j + 2 * i + 0);
							let t1 = t(j + 2 * i + 8);
							let r0 = r(4 * j + i + 0);
							let r1 = r(4 * j + i + 8);

							asm!("{shuffle64x2} {r0}, {t0}, {t1}, {idx[0]}");
							asm!("{shuffle64x2} {r1}, {t0}, {t1}, {idx[1]}");
						}
					}

					false
				},
				_ => unreachable!(),
			},
			Ty::F64 | Ty::C32 => match n_regs {
				2 => {
					t = 15;
					let r = |i: usize| format!("xmm{i}");
					let t = |i: usize| format!("xmm{t - i}");

					let t0 = t(0);
					let t1 = t(1);
					let r0 = r(0);
					let r1 = r(1);
					asm!("{unpacklo64} {t0}, {r0}, {r1}");
					asm!("{unpackhi64} {t1}, {r0}, {r1}");
					true
				},
				4 => {
					t = 15;
					let r = |i: usize| format!("ymm{i}");
					let t = |i: usize| format!("ymm{t - i}");

					for i in 0..2 {
						let t0 = t(2 * i + 0);
						let t1 = t(2 * i + 1);
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						asm!("{unpacklo64} {t0}, {r0}, {r1}");
						asm!("{unpackhi64} {t1}, {r0}, {r1}");
					}

					let idx = [0b00100000, 0b00110001];
					for i in 0..2 {
						let t0 = t(i + 0);
						let t1 = t(i + 2);
						let r0 = r(i + 0);
						let r1 = r(i + 2);
						asm!("{permute2x128} {r0}, {t0}, {t1}, {idx[0]}");
						asm!("{permute2x128} {r1}, {t0}, {t1}, {idx[1]}");
					}
					false
				},
				8 => {
					let r = |i: usize| format!("zmm{i}");
					let t = |i: usize| format!("zmm{t - i}");

					for i in 0..4 {
						let t0 = t(2 * i + 0);
						let t1 = t(2 * i + 1);
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						asm!("{unpacklo64} {t0}, {r0}, {r1}");
						asm!("{unpackhi64} {t1}, {r0}, {r1}");
					}

					let idx = [0b10001000, 0b11011101];
					for j in 0..2 {
						for i in 0..2 {
							let t0 = t(4 * j + i + 0);
							let t1 = t(4 * j + i + 2);
							let r0 = r(4 * j + 2 * i + 0);
							let r1 = r(4 * j + 2 * i + 1);
							asm!("{shuffle64x2} {r0}, {t0}, {t1}, {idx[0]}");
							asm!("{shuffle64x2} {r1}, {t0}, {t1}, {idx[1]}");
						}
					}

					for j in 0..2 {
						for i in 0..2 {
							let t0 = t(2 * j + i + 0);
							let t1 = t(2 * j + i + 4);
							let r0 = r(j + 2 * i + 0);
							let r1 = r(j + 2 * i + 4);
							asm!("{shuffle64x2} {t0}, {r0}, {r1}, {idx[0]}");
							asm!("{shuffle64x2} {t1}, {r0}, {r1}, {idx[1]}");
						}
					}

					true
				},
				_ => unreachable!(),
			},
			Ty::C64 => match n_regs {
				2 => {
					t = 15;
					let r = |i: usize| format!("ymm{i}");
					let t = |i: usize| format!("ymm{t - i}");

					let idx = [0b00100000, 0b00110001];

					let t0 = t(0);
					let t1 = t(1);
					let r0 = r(0);
					let r1 = r(1);
					asm!("{permute2x128} {t0}, {r0}, {r1}, {idx[0]}");
					asm!("{permute2x128} {t1}, {r0}, {r1}, {idx[1]}");

					true
				},
				4 => {
					let r = |i: usize| format!("zmm{i}");
					let t = |i: usize| format!("zmm{t - i}");

					let idx = [0b10001000, 0b11011101];

					for i in 0..2 {
						let r0 = r(2 * i + 0);
						let r1 = r(2 * i + 1);
						let t0 = t(2 * i + 0);
						let t1 = t(2 * i + 1);

						asm!("{shuffle64x2} {t0}, {r0}, {r1}, {idx[0]}");
						asm!("{shuffle64x2} {t1}, {r0}, {r1}, {idx[1]}");
					}

					for i in 0..2 {
						let r0 = r(i + 0);
						let r1 = r(i + 2);
						let t0 = t(i + 0);
						let t1 = t(i + 2);

						asm!("{shuffle64x2} {r0}, {t0}, {t1}, {idx[0]}");
						asm!("{shuffle64x2} {r1}, {t0}, {t1}, {idx[1]}");
					}

					false
				},
				_ => unreachable!(),
			},
		};

		let x = (ctx.code.borrow().clone(), t, in_t);
		x
	}

	fn pack_lhs(self, m: isize) -> (String, String) {
		let Self { ty, simd } = self;
		let bits = simd.sizeof() * 8;

		let ctx = Ctx::new();
		setup!(ctx, self);
		let suffix = &format!("m={m * self.len()}");
		let src = rax;
		let dst = r15;
		let nrows = r8;
		let info = rsi;
		let prefix = &format!("gemm.pack.{ty}.simd{bits}");

		ctx[rsp].set(true);
		ctx[src].set(true);
		ctx[dst].set(true);
		ctx[nrows].set(true);
		ctx[info].set(true);

		let dst_cs = simd.sizeof() * m;
		let need_mask = simd.sizeof() >= 16;

		let main = {
			func!(pub "{func_name(prefix, &suffix, false)}");
			{
				label!({
					let good;
				});
				if m > 1 {
					cmp!(nrows, (m - 1) * self.len() + 1);
					jnc!(good);
					let prev = &format!("m={(m - 1) * self.len()}");
					jmp!("{func_name(prefix, prev, false)}");
				}
				label!(good = _);
			}
			{
				label!({
					let cont;
				});
				cmp!([info + INFO_LHS_RS], ty.sizeof());
				jz!(cont);

				cmp!([info + INFO_LHS_CS], ty.sizeof());
				jz!(cont);

				mov!(nrows, m * simd.sizeof());
				asm!("ret");

				label!(cont = _);
			}

			{
				alloca!(src);
				alloca!(dst);
				reg!(src_rs);
				reg!(depth);
				reg!(depth_down);
				reg!(diag_ptr);
				reg!(diag_stride);

				label!({
					let colmajor;
					let rowmajor;
					let end;
				});

				mov!(depth, [info + INFO_DEPTH]);
				mov!(diag_ptr, [info + INFO_DIAG_PTR]);
				mov!(diag_stride, [info + INFO_DIAG_STRIDE]);

				test!(depth, depth);
				jz!(end);

				cmp!([info + INFO_LHS_RS], ty.sizeof());
				jz!(colmajor);

				cmp!([info + INFO_LHS_CS], ty.sizeof());
				jz!(rowmajor);

				abort!();
				label!(colmajor = _);
				{
					label!({
						let loop_begin;
						let loop_begin_d;
					});

					let src_cs = src_rs;
					let mask_ptr = depth_down;

					if need_mask {
						reg!(tmp);
						{
							let name = func_name(&format!("gemm.microkernel.{ty}.simd{bits}.mask.data"), "", true);

							lea!(tmp, [rip + &name]);
						}
						if self.mask_sizeof() <= 8 {
							lea!(mask_ptr, [tmp + nrows * self.mask_sizeof() + -(m - 1) * self.len() * self.mask_sizeof()]);
						} else {
							lea!(mask_ptr, [tmp + nrows * 8 + -(m - 1) * self.len() * self.mask_sizeof()]);
							for _ in 0..self.mask_sizeof() / 8 - 1 {
								lea!(mask_ptr, [mask_ptr + nrows * 8]);
							}
						}
						kmov!(k(2), [mask_ptr]);
					}
					mov!(src_cs, [info + INFO_LHS_CS]);

					test!(diag_ptr, diag_ptr);
					jnz!(loop_begin_d);

					label!(loop_begin = _);
					for i in 0..m {
						if i + 1 == m {
							if need_mask {
								vmov!(zmm(0)[2], [src + i * simd.sizeof()]);
							} else {
								abort!();
							}
						} else {
							vmov!(zmm(0), [src + i * simd.sizeof()]);
						}
						vmov!([dst + i * simd.sizeof()], zmm(0));
					}

					add!(src, src_cs);
					add!(dst, dst_cs);
					dec!(depth);
					jnz!(loop_begin);

					jmp!(end);

					label!(loop_begin_d = _);
					vbroadcast!(zmm(1), [diag_ptr]);

					for i in 0..m {
						if i + 1 == m {
							if need_mask {
								vmov!(zmm(0)[2], [src + i * simd.sizeof()]);
							} else {
								abort!();
							}
						} else {
							vmov!(zmm(0), [src + i * simd.sizeof()]);
						}
						vmul!(zmm(0), zmm(0), zmm(1));
						vmov!([dst + i * simd.sizeof()], zmm(0));
					}

					add!(src, src_cs);
					add!(dst, dst_cs);
					add!(diag_ptr, diag_stride);
					dec!(depth);
					jnz!(loop_begin_d);

					jmp!(end);
				}

				label!(rowmajor = _);
				{
					mov!(src_rs, [info + INFO_LHS_RS]);

					let min_nrows = (m - 1) * self.len();

					for j in 0..simd.num_regs() {
						vxor!(zmm(j), zmm(j), zmm(j));
					}

					let mut len = self.len();

					while len >= 1 {
						label!({
							let loop_begin;
							let loop_end;
						});

						assert!(len <= 16);

						let target = Target {
							ty,
							simd: match len * ty.sizeof() * 8 {
								512 => Simd::_512,
								256 => Simd::_256,
								128 => Simd::_128,
								64 => Simd::_64,
								32 => Simd::_32,
								_ => unreachable!(),
							},
						};

						mov!(depth_down, depth);
						and!(depth_down, -len as i8);
						sub!(depth, depth_down);

						test!(depth_down, depth_down);
						jz!(loop_end);
						label!(loop_begin = _);
						{
							alloca!(src);

							for i in 0..min_nrows / len {
								for j in 0..len {
									asm!("{target.vload(j, Addr::from(src))}");
									add!(src, src_rs);
								}

								let (c, t, in_t) = self.transpose(len as _);
								*ctx.code.borrow_mut() += &c;

								let d = if in_t { 0 } else { t as isize };
								{
									alloca!(diag_ptr);
									for j in 0..len {
										let reg = if in_t { t as isize - j } else { j };
										assert_ne!(d, reg);
										label!({
											let cont;
										});

										test!(diag_ptr, diag_ptr);
										jz!(cont);
										asm!("{target.vbroadcast(d, Addr::from(diag_ptr))}");
										asm!("{target.vmul(reg, reg, d)}");
										add!(diag_ptr, diag_stride);

										label!(cont = _);
										asm!(
											"{
                                        target.vstore(
                                            Addr::from(dst + (i * len * ty.sizeof() + j * dst_cs)),
                                            reg,
                                        )
                                    }"
										);
									}
								}
							}
							let mut i = min_nrows / len;
							while i < m * self.len() / len {
								label!({
									let end;
								});

								for j in 0..len {
									cmp!(nrows, i * len + j);
									jng!(end);

									asm!("{target.vload(j, Addr::from(src))}");
									add!(src, src_rs);
								}
								label!(end = _);

								let (c, t, in_t) = self.transpose(len as _);
								*ctx.code.borrow_mut() += &c;

								let d = if in_t { 0 } else { t as isize };
								{
									alloca!(diag_ptr);
									for j in 0..len {
										let reg = if in_t { t as isize - j } else { j };
										assert_ne!(d, reg);

										label!({
											let cont;
										});

										test!(diag_ptr, diag_ptr);
										jz!(cont);
										asm!("{target.vbroadcast(d, Addr::from(diag_ptr))}");
										asm!("{target.vmul(reg, reg, d)}");
										add!(diag_ptr, diag_stride);

										label!(cont = _);
										asm!(
											"{
                                        target.vstore(
                                            Addr::from(dst + (i * len * ty.sizeof() + j * dst_cs)),
                                            reg,
                                        )
                                    }"
										);
									}
								}

								i += 1;
							}
						}

						add!(src, len * ty.sizeof());
						add!(dst, len * dst_cs);
						if len == 16 {
							lea!(diag_ptr, [diag_ptr + 8 * diag_stride]);
							lea!(diag_ptr, [diag_ptr + 8 * diag_stride]);
						} else {
							lea!(diag_ptr, [diag_ptr + len * diag_stride]);
						}

						if len == 1 {
							dec!(depth_down);
						} else {
							sub!(depth_down, len);
						}
						jnz!(loop_begin);
						label!(loop_end = _);

						len /= 2;
					}
				}

				label!(end = _);
			}
			name!().clone()
		};

		let x = (main, ctx.code.borrow().clone());
		x
	}

	fn microkernel(self, m: isize, n: isize) -> (String, String) {
		let Self { ty, simd } = self;
		let bits = simd.sizeof() * 8;
		let need_mask = simd.sizeof() >= 16;

		let unroll = if m * n <= 6 && !need_mask { 2 } else { 1 };

		let ctx = Ctx::new();
		setup!(ctx, self);

		let suffix = &format!("m={m * self.len()}.n={n}");
		let lhs = rax;
		let packed_lhs = r15;
		let rhs = rcx;
		let packed_rhs = rdx;
		let position = rdi;
		let info = rsi;
		let nrows = r8;
		let ncols = r9;

		let flags_accum = 0;
		let flags_conj_lhs = 1;
		let flags_conj_diff = 2;
		let flags_lower = 3;
		let flags_upper = 4;
		let flags_32bit = 5;

		ctx[rsp].set(true);
		ctx[lhs].set(true);
		ctx[packed_lhs].set(true);
		ctx[rhs].set(true);
		ctx[packed_rhs].set(true);
		ctx[position].set(true);
		ctx[nrows].set(true);
		ctx[ncols].set(true);
		ctx[info].set(true);

		let prefix = &format!("gemm.microkernel.{ty}.simd{bits}");

		let mask_ptr;
		let main = {
			let depth;
			let lhs_rs;
			let lhs_cs;
			let rhs_rs;
			let rhs_cs;

			let main = {
				label!({
					let row_check;
					let col_check;
					let prologue;
				});

				func!(pub "{func_name(prefix, suffix, false)}");
				label!(row_check = _);
				if m > 1 {
					cmp!(nrows, (m - 1) * self.len() + 1);
					jnc!(col_check);
					let prev = &format!("m={(m - 1) * self.len()}.n={n}");
					jmp!("{func_name(prefix, prev, false)}");
				}
				label!(col_check = _);
				if n > 1 {
					cmp!(ncols, n);
					jnc!(prologue);
					let prev = &format!("m={m * self.len()}.n={n - 1}");
					jmp!("{func_name(prefix, prev, false)}");
				}
				label!(prologue = _);

				alloca!(lhs);
				alloca!(rhs);
				alloca!(packed_lhs);
				alloca!(packed_rhs);

				reg!(&lhs_rs);
				reg!(&lhs_cs);
				reg!(&rhs_rs);
				reg!(&rhs_cs);
				reg!(&mask_ptr);

				mov!(lhs_rs, [info + INFO_LHS_RS]);
				mov!(lhs_cs, [info + INFO_LHS_CS]);
				mov!(rhs_rs, [info + INFO_RHS_RS]);
				mov!(rhs_cs, [info + INFO_RHS_CS]);

				{
					reg!(tmp);

					test!(lhs, lhs);
					mov!(tmp, m * simd.sizeof());
					cmovz!(lhs_cs, tmp);
					mov!(tmp, ty.sizeof());
					cmovz!(lhs_rs, tmp);
					cmovz!(lhs, packed_lhs);

					test!(rhs, rhs);
					cmovz!(rhs_cs, tmp);
					mov!(tmp, n * ty.sizeof());
					cmovz!(rhs_rs, tmp);
					cmovz!(rhs, packed_rhs);
				}

				for i in 0..m * n * unroll {
					vxor!(zmm(i), zmm(i), zmm(i));
				}

				if need_mask {
					label!({
						let no_mask;
					});

					sub!(nrows, m * self.len());

					jnc!(no_mask);

					{
						reg!(tmp);
						lea!(tmp, [rip + &func_name(&format!("{prefix}.mask.data"), "", false)]);

						if self.mask_sizeof() <= 8 {
							lea!(mask_ptr, [tmp + nrows * self.mask_sizeof() + self.len() * self.mask_sizeof()]);
						} else {
							lea!(mask_ptr, [tmp + nrows * 8 + self.len() * self.mask_sizeof()]);
							for _ in 0..self.mask_sizeof() / 8 - 1 {
								lea!(mask_ptr, [mask_ptr + nrows * 8]);
							}
						}
					}
					label!(no_mask = _);

					add!(nrows, m * self.len());
				}

				reg!(&depth);
				mov!(depth, [info + INFO_DEPTH]);

				label!({
					let colmajor;
					let strided;
					let load;
					let mask;
					let epilogue;
					let epilogue_any;
					let epilogue_store;
					let epilogue_mask;
					let epilogue_store_overwrite;
					let epilogue_store_add;
					let epilogue_mask_overwrite;
					let epilogue_mask_add;
					let epilogue_lower;
					let epilogue_lower_overwrite;
					let epilogue_lower_add;
					let epilogue_upper;
					let epilogue_upper_overwrite;
					let epilogue_upper_add;
					let end;
				});

				cmp!(lhs_rs, ty.sizeof());
				jz!(colmajor);

				label!(strided = _);
				{
					abort!();
					jmp!(end);
				}

				label!(colmajor = _);
				{
					cmp!(nrows, m * self.len());
					jnc!(load);

					if need_mask {
						test!(lhs, simd.sizeof() - 1);
						jnz!(mask);

						test!(lhs_cs, simd.sizeof() - 1);
						jnz!(mask);
					} else {
						abort!();
						jmp!(end);
					}
				}

				label!(load = _);
				{
					label!({
						let load_A;
						let load_A;
						let load_noA;
					});
					cmp!(packed_lhs, lhs);
					jnz!(load_A);
					cmp!(packed_rhs, rhs);

					label!(load_noA = _);
					{
						let f = &format!("{prefix}.load");
						call!("{func_name(f, suffix, false)}");
						jmp!(epilogue);
					}

					label!(load_A = _);
					{
						let f = &format!("{prefix}.load.packA");
						call!("{func_name(f, suffix, false)}");
						jmp!(epilogue);
					}
				}

				label!(mask = _);
				{
					label!({
						let mask_A;
						let mask_A;
						let mask_noA;
					});

					if need_mask {
						cmp!(packed_lhs, lhs);
						jnz!(mask_A);

						label!(mask_noA = _);
						{
							let f = &format!("{prefix}.mask");
							call!("{func_name(f, suffix, false)}");
							jmp!(epilogue);
						}
						label!(mask_A = _);
						{
							let f = &format!("{prefix}.mask.packA");
							call!("{func_name(f, suffix, false)}");
							jmp!(epilogue);
						}
					}
				}
				label!(epilogue = _);

				{
					let alpha_ptr = lhs;
					let alpha_re = m * n;
					let alpha_im = m * n + 1;
					let tmp = m * n + 3;

					if self.is_cplx() {
						vmov!(
							zmm(alpha_re),
							[rip + &func_name(&format!("gemm.microkernel.{ty}.flip.re.data"), "", true)]
						);
						vmov!(
							zmm(alpha_im),
							[rip + &func_name(&format!("gemm.microkernel.{ty}.flip.im.data"), "", true)]
						);
						label!({
							let xor;
							let conj_lhs;
							let conj_lhs_conj_rhs;
							let conj_lhs_no_conj_rhs;
							let conj_lhs_no_conj_rhs;
							let no_conj_lhs;
							let no_conj_lhs_conj_rhs;
							let no_conj_lhs_no_conj_rhs;
							let no_conj_lhs_no_conj_rhs;
						});
						bt!([info + INFO_FLAGS], flags_conj_lhs);
						jnc!(no_conj_lhs);

						label!(conj_lhs = _);
						{
							bt!([info + INFO_FLAGS], flags_conj_diff);
							jc!(conj_lhs_no_conj_rhs);

							label!(conj_lhs_conj_rhs = _);
							{
								vxor!(zmm(alpha_re), zmm(alpha_re), zmm(alpha_im));
								jmp!(xor);
							}
							label!(conj_lhs_no_conj_rhs = _);
							{
								vxor!(zmm(alpha_re), zmm(alpha_re), zmm(alpha_re));
								jmp!(xor);
							}
						}

						label!(no_conj_lhs = _);
						{
							bt!([info + INFO_FLAGS], flags_conj_diff);
							jnc!(no_conj_lhs_no_conj_rhs);

							label!(no_conj_lhs_conj_rhs = _);
							{
								vmov!(zmm(alpha_re), zmm(alpha_im));
								jmp!(xor);
							}
							label!(no_conj_lhs_no_conj_rhs = _);
							{}
						}
						label!(xor = _);
						for i in 0..m * n {
							vxor!(zmm(i), zmm(i), zmm(alpha_re));
						}
					}

					mov!(alpha_ptr, [info + INFO_ALPHA]);
					vbroadcast!(zmm(alpha_re), [alpha_ptr]);
					if self.is_cplx() {
						vbroadcast!(zmm(alpha_im), [alpha_ptr + ty.sizeof() / 2]);
					}

					for j in 0..n {
						for i in 0..m {
							let src = m * j + i;

							if self.is_cplx() {
								vswap!(zmm(src));
								vmul!(zmm(tmp), zmm(src), zmm(alpha_im));
								vswap!(zmm(src));
								vfma231!(zmm(tmp), zmm(src), zmm(alpha_re));
								vmov!(zmm(src), zmm(tmp));
							} else {
								vmul!(zmm(src), zmm(src), zmm(alpha_re));
							};
						}
					}
				}

				cmp!([info + INFO_ROW_IDX], 0);
				jnz!(epilogue_any);
				cmp!([info + INFO_COL_IDX], 0);
				jnz!(epilogue_any);
				cmp!([info + INFO_RS], ty.sizeof());
				jnz!(epilogue_any);

				bt!([info + INFO_FLAGS], flags_lower);
				jc!(epilogue_lower);

				bt!([info + INFO_FLAGS], flags_upper);
				jc!(epilogue_upper);

				cmp!(nrows, m * self.len());
				jnc!(epilogue_store);

				label!(epilogue_mask = _);
				{
					if need_mask {
						bt!([info + INFO_FLAGS], flags_accum);
						jnc!(epilogue_mask_overwrite);

						label!(epilogue_mask_add = _);
						let f = &format!("{prefix}.epilogue.mask.add");
						call!("{func_name(f, suffix, false)}");
						jmp!(end);

						label!(epilogue_mask_overwrite = _);
						let f = &format!("{prefix}.epilogue.mask.overwrite");
						call!("{func_name(f, suffix, false)}");
						jmp!(end);
					} else {
						let f = &format!("{prefix}.epilogue.any");
						call!("{func_name(f, suffix, false)}");
						jmp!(end);
					}
				}

				label!(epilogue_store = _);
				{
					bt!([info + INFO_FLAGS], flags_accum);
					jnc!(epilogue_store_overwrite);

					label!(epilogue_store_add = _);
					let f = &format!("{prefix}.epilogue.store.add");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);

					label!(epilogue_store_overwrite = _);
					let f = &format!("{prefix}.epilogue.store.overwrite");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);
				}

				label!(epilogue_lower = _);
				if need_mask {
					bt!([info + INFO_FLAGS], flags_accum);
					jnc!(epilogue_lower_overwrite);

					label!(epilogue_lower_add = _);
					let f = &format!("{prefix}.epilogue.mask.lower.add");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);

					label!(epilogue_lower_overwrite = _);
					let f = &format!("{prefix}.epilogue.mask.lower.overwrite");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);
				}

				label!(epilogue_upper = _);
				if need_mask {
					bt!([info + INFO_FLAGS], flags_accum);
					jnc!(epilogue_upper_overwrite);

					label!(epilogue_upper_add = _);
					let f = &format!("{prefix}.epilogue.mask.upper.add");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);

					label!(epilogue_upper_overwrite = _);
					let f = &format!("{prefix}.epilogue.mask.upper.overwrite");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);
				}

				label!(epilogue_any = _);
				{
					let f = &format!("{prefix}.epilogue.any");
					call!("{func_name(f, suffix, false)}");
					jmp!(end);
				}

				label!(end = _);

				name!().clone()
			};

			ctx[depth].set(true);
			ctx[rhs_cs].set(true);
			ctx[rhs_rs].set(true);
			ctx[lhs_cs].set(true);
			ctx[lhs_rs].set(true);
			ctx[mask_ptr].set(true);

			for mask in if need_mask { vec![false, true] } else { vec![false] } {
				let __mask__ = if mask { ".mask" } else { ".load" };

				for pack_lhs in [false, true] {
					let __pack_lhs__ = if pack_lhs { ".packA" } else { "" };

					for pack_rhs in [false] {
						let __pack_rhs__ = if pack_rhs { ".packB" } else { "" };

						for conj in if self.is_cplx() { vec![false, true] } else { vec![false] } {
							let __conj__ = if conj { ".conj" } else { "" };

							let f = &format!("{prefix}{__conj__}{__mask__}{__pack_lhs__}{__pack_rhs__}");
							func!("{func_name(f, suffix, false)}");
							label!({
								let start0;
								let start1;
							});
							if self.is_cplx() && !conj {
								bt!([info + INFO_FLAGS], flags_conj_diff);
								jnc!(start0);

								let f = &format!("{prefix}.conj{__mask__}{__pack_lhs__}{__pack_rhs__}");
								jmp!("{func_name(f, suffix, false)}");
							}
							label!(start0 = _);

							let rhs_neg_cs = lhs_rs;

							mov!(rhs_neg_cs, rhs_cs);
							neg!(rhs_neg_cs);
							add!(rhs, rhs_cs);

							if mask && simd.dedicated_mask() {
								kmov!(k(1), [mask_ptr]);
							}

							reg!(depth_down);

							let mut unroll0 = unroll;
							while unroll0 > 0 {
								label!({
									let nanokernel;
									let loop_end;
								});

								mov!(depth_down, depth);
								and!(depth_down, -unroll0 as i8);
								sub!(depth, depth_down);
								test!(depth_down, depth_down);
								jz!(loop_end);

								label!(nanokernel = _);

								let bcst = bits == 512 && m == 1 && !pack_rhs;
								for iter in 0..unroll0 {
									for i in 0..m {
										if !mask || i + 1 < m {
											vmov!(zmm(m * n * unroll + i), [lhs + simd.sizeof() * i],);
										} else {
											if simd.dedicated_mask() {
												vmov!(zmm(m * n * unroll + i)[1], [lhs + simd.sizeof() * i],);
											} else {
												vmov!(zmm(m * n * unroll + i), [mask_ptr]);
												vmov!(zmm(m * n * unroll + i)[m * n * unroll + i], [lhs + simd.sizeof() * i],);
											}
										}
										if pack_lhs {
											vmov!([packed_lhs + simd.sizeof() * i], zmm(m * n * unroll + i));
										}
									}

									for j in 0..n {
										let rhs_addr = if j % 4 == 0 { rhs + 1 * rhs_neg_cs } else { rhs + (j % 4 - 1) * rhs_cs };

										if !bcst {
											vbroadcast!(zmm(m * n * unroll + m), [rhs_addr]);
											if self.is_real() && n > 4 {
												if j + 1 == 4 {
													lea!(rhs, [rhs + rhs_cs * 4]);
												}
											}

											if self.is_real() && j + 1 == n {
												if n > 4 {
													lea!(rhs, [rhs + rhs_neg_cs * 4]);
												}
												add!(rhs, rhs_rs);
											}
										}

										if pack_rhs {
											vmovsr!([packed_rhs + j * ty.sizeof()], xmm(m * n * unroll + m));
											if self.is_real() && j + 1 == n {
												add!(packed_rhs, n * ty.sizeof());
											}
										}

										for i in 0..m {
											if bcst {
												if conj {
													vfma231_conj!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), [rhs_addr],);
												} else {
													vfma231!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), [rhs_addr],);
												}

												if self.is_real() && n > 4 {
													if j + 1 == 4 {
														lea!(rhs, [rhs + rhs_cs * 4]);
													}
												}

												if self.is_real() && j + 1 == n {
													if n > 4 {
														lea!(rhs, [rhs + rhs_neg_cs * 4]);
													}
													add!(rhs, rhs_rs);
												}
											} else {
												if conj {
													vfma231_conj!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), zmm(m * n * unroll + m),);
												} else {
													vfma231!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), zmm(m * n * unroll + m),);
												}
											}
										}

										if j == 0 {
											add!(lhs, lhs_cs);
											if pack_lhs {
												add!(packed_lhs, simd.sizeof() * m);
											}
										}
									}

									if self.is_cplx() {
										for j in 0..n {
											let rhs_addr = if j % 4 == 0 {
												rhs + 1 * rhs_neg_cs + ty.sizeof() / 2
											} else {
												rhs + (j % 4 - 1) * rhs_cs + ty.sizeof() / 2
											};

											if !bcst {
												vbroadcast!(zmm(m * n * unroll + m), [rhs_addr]);
												if n > 4 {
													if (j + 1) % 4 == 0 {
														lea!(rhs, [rhs + rhs_cs * 4]);
													}
												}

												if j + 1 == n {
													if n > 4 {
														lea!(rhs, [rhs + rhs_neg_cs * 4]);
													}
													add!(rhs, rhs_rs);
												}
											}

											if pack_rhs {
												vmovsr!([packed_rhs + (j * ty.sizeof() + ty.sizeof() / 2)], xmm(m * n * unroll + m));
												if j + 1 == n {
													add!(packed_rhs, n * ty.sizeof());
												}
											}

											for i in 0..m {
												if j == 0 {
													vswap!(zmm(m * n * unroll + i));
												}
												if bcst {
													if conj {
														vfma231_conj!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), [rhs_addr],);
													} else {
														vfma231!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), [rhs_addr],);
													}

													if n > 4 {
														if j + 1 == 4 {
															lea!(rhs, [rhs + rhs_cs * 4]);
														}
													}

													if j + 1 == n {
														if n > 4 {
															lea!(rhs, [rhs + rhs_neg_cs * 4]);
														}
														add!(rhs, rhs_rs);
													}
												} else {
													if conj {
														vfma231_conj!(
															zmm(m * n * iter + m * j + i),
															zmm(m * n * unroll + i),
															zmm(m * n * unroll + m),
														);
													} else {
														vfma231!(zmm(m * n * iter + m * j + i), zmm(m * n * unroll + i), zmm(m * n * unroll + m),);
													}
												}
											}
										}
									}
								}
								if unroll0 == 1 {
									dec!(depth_down);
								} else {
									sub!(depth_down, unroll);
								}
								jnz!(nanokernel);
								label!(loop_end = _);
								unroll0 /= 2;
							}

							for iter in 1..unroll {
								for i in 0..m * n {
									vadd!(zmm(i), zmm(i), zmm(m * n * iter + i));
								}
							}
						}
					}
				}
			}

			ctx[depth].set(false);
			ctx[rhs_cs].set(false);
			ctx[rhs_rs].set(false);
			ctx[lhs_cs].set(false);
			ctx[lhs_rs].set(false);
			ctx[lhs].set(false);
			ctx[packed_lhs].set(false);
			ctx[rhs].set(false);
			ctx[packed_rhs].set(false);

			main
		};

		{
			let f = &format!("{prefix}.epilogue.any");
			func!("{func_name(f, suffix, false)}");

			reg!(stack);
			reg!(tmp);
			reg!(ptr);
			reg!(rs);
			reg!(cs);
			reg!(row);
			reg!(col);
			reg!(row_idx);
			reg!(col_idx);
			reg!(row_min);
			let row_max = mask_ptr;

			let vtmp1 = m * n + 2;
			let vtmp2 = m * n + 3;

			{
				label!({
					let rowmajor;
					let colmajor;
					let end;
				});

				mov!(rs, [info + INFO_RS]);
				mov!(cs, [info + INFO_CS]);
				mov!(row_idx, [info + INFO_ROW_IDX]);
				mov!(col_idx, [info + INFO_COL_IDX]);

				for j in 0..n {
					mov!(stack, rsp);
					and!(rsp, -128);
					sub!(rsp, 512);

					for i in 0..m {
						vmov!([rsp + i * simd.sizeof()], zmm(m * j + i));
					}

					mov!(col, [position + WORD]);
					add!(col, j);

					mov!(row_min, [position]);
					mov!(row, [position]);
					cmp!(row, col);

					mov!(row_max, m * self.len());
					cmp!(nrows, row_max);
					cmovc!(row_max, nrows);
					add!(row_max, [position]);

					{
						label!({
							let cont;
						});

						bt!([info + INFO_FLAGS], flags_lower);
						jnc!(cont);
						cmp!(row_min, col);
						jg!(cont);
						sub!(row_min, col);
						neg!(row_min);
						if ty.sizeof() <= 8 {
							lea!(rsp, [rsp + ty.sizeof() * row_min]);
						} else {
							assert_eq!(ty.sizeof(), 16);
							lea!(rsp, [rsp + 8 * row_min]);
							lea!(rsp, [rsp + 8 * row_min]);
						}

						mov!(row_min, col);

						label!(cont = _);
					}
					{
						label!({
							let cont;
						});

						inc!(col);

						bt!([info + INFO_FLAGS], flags_upper);
						jnc!(cont);

						cmp!(row_max, col);
						jl!(cont);
						mov!(row_max, col);

						label!(cont = _);

						dec!(col);
					}

					{
						label!({
							let cont;
							let cont32;
						});
						test!(col_idx, col_idx);
						jz!(cont);

						bt!([info + INFO_FLAGS], flags_32bit);
						jc!(cont32);

						{
							mov!(col, [col_idx + WORD * col]);
							jmp!(cont);
						}
						label!(cont32 = _);

						{
							movzx!(col, [col_idx + 4 * col]);
						}
						label!(cont = _);
					}
					imul!(col, cs);
					mov!(ptr, [info + INFO_PTR]);
					add!(ptr, col);

					{
						label!({
							let cont;
							let end;
						});

						test!(row_idx, row_idx);
						jz!(cont);

						{
							label!({
								let begin;
								let skip_add;
								let cont32;
								let cont64;
							});
							cmp!(row_min, row_max);
							jnl!(end);

							mov!(row, row_min);

							label!(begin = _);
							{
								bt!([info + INFO_FLAGS], flags_32bit);
								jc!(cont32);

								{
									mov!(tmp, [row_idx + WORD * row]);
									jmp!(cont64);
								}
								label!(cont32 = _);

								{
									movzx!(tmp, [row_idx + 4 * row]);
								}
								label!(cont64 = _);

								imul!(tmp, rs);
								lea!(tmp, [ptr + 1 * tmp]);

								vmovs!(xmm(vtmp1), [rsp]);
								add!(rsp, ty.sizeof());

								bt!([info + INFO_FLAGS], flags_accum);
								jnc!(skip_add);
								{
									vmovs!(xmm(vtmp2), [tmp]);
									vadds!(xmm(vtmp1), xmm(vtmp1), xmm(vtmp2));
								}
								label!(skip_add = _);

								vmovs!([tmp], xmm(vtmp1));

								inc!(row);
								cmp!(row, row_max);
								jl!(begin);
							}

							jmp!(end);
						}

						label!(cont = _);
						{
							label!({
								let begin;
								let skip_add;
							});
							cmp!(row_min, row_max);
							jnl!(end);

							mov!(row, row_min);

							label!(begin = _);
							{
								mov!(tmp, row);
								imul!(tmp, rs);
								lea!(tmp, [ptr + 1 * tmp]);

								vmovs!(xmm(vtmp1), [rsp]);
								add!(rsp, ty.sizeof());

								bt!([info + INFO_FLAGS], flags_accum);
								jnc!(skip_add);
								{
									vmovs!(xmm(vtmp2), [tmp]);
									vadds!(xmm(vtmp1), xmm(vtmp1), xmm(vtmp2));
								}
								label!(skip_add = _);

								vmovs!([tmp], xmm(vtmp1));

								inc!(row);
								cmp!(row, row_max);
								jl!(begin);
							}

							jmp!(end);
						}

						label!(end = _);
					}
					mov!(rsp, stack);
				}

				label!(end = _);
			}
		}

		let __triangle__ = [".lower", ".upper", ""];

		for triangle in 0..3 {
			let __triangle__ = __triangle__[triangle];

			let mut mask = vec![];
			if need_mask {
				mask.push(true);
			}
			if triangle == 2 {
				mask.push(false);
			}

			for &mask in &mask {
				let __mask__ = if mask { ".mask" } else { ".store" };

				for add in [false, true] {
					let __add__ = if add { ".add" } else { ".overwrite" };

					let f = &format!("{prefix}.epilogue{__mask__}{__triangle__}{__add__}");
					func!("{func_name(f, suffix, false)}");
					label!({
						let rowmajor;
						let colmajor;
						let end;
					});

					reg!(ptr);
					reg!(rs);
					reg!(cs);
					reg!(row);
					reg!(col);

					mov!(ptr, [info + INFO_PTR]);
					mov!(rs, [info + INFO_RS]);
					mov!(cs, [info + INFO_CS]);
					mov!(row, [position]);
					mov!(col, [position + WORD]);

					if triangle == 0 {
						sub!(row, col);

						{
							label!({
								let cont;
							});

							cmp!(row, -(self.len() - 1));
							jnl!(cont);
							{
								if m > 1 {
									{
										for j in 0..n {
											for i in 1..m {
												vmov!(zmm((i - 1) + (m - 1) * j), zmm(i + m * j));
											}
										}
										alloca!([position]);
										alloca!(nrows);
										add!([position], self.len());
										sub!(nrows, self.len());

										let suffix = &format!("m={(m - 1) * self.len()}.n={n}");
										call!("{func_name(f, suffix, false)}");
									}
									jmp!(end);
								} else {
									jmp!(end);
								}
							}

							label!(cont = _);
						}

						{
							label!({
								let cont;
							});

							cmp!(row, n - 1);
							jl!(cont);

							{
								pop!(col);
								pop!(row);
								pop!(cs);
								pop!(rs);
								pop!(ptr);
								let f = &format!("{prefix}.epilogue{__mask__}{__add__}");
								jmp!("{func_name(f, suffix, false)}");
							}
							label!(cont = _);
						}

						add!(row, col);
					}
					if triangle == 1 {
						sub!(col, row);
						{
							label!({
								let cont;
							});

							// x x x
							// 0 x x
							// 0 0 x
							// 0 0[0] // row + (m - 1) * simd_len > col + ncols - 1 => col - row < (m - 1) * simd_len - (ncols -
							// 1) 0 0 0
							// 0 0 0
							// col - row <
							cmp!(col, (m - 1) * self.len() - (n - 1));
							jnl!(cont);

							{
								if m > 1 {
									{
										for j in 1..n {
											for i in 0..m - 1 {
												vmov!(zmm(i + (m - 1) * j), zmm(i + m * j));
											}
										}

										pop!(col);
										pop!(row);
										pop!(cs);
										pop!(rs);
										pop!(ptr);
										let suffix = &format!("m={(m - 1) * self.len()}.n={n}");
										jmp!("{func_name(f, suffix, false)}");
										abort!();
									}
								} else {
									jmp!(end);
								}
							}

							label!(cont = _);
						}

						{
							label!({
								let cont;
							});

							add!(col, 1);
							cmp!(col, nrows);
							jl!(cont);

							{
								pop!(col);
								pop!(row);
								pop!(cs);
								pop!(rs);
								pop!(ptr);
								let f = &format!("{prefix}.epilogue{__mask__}{__add__}");
								jmp!("{func_name(f, suffix, false)}");
								abort!();
							}
							label!(cont = _);
							sub!(col, 1);
						}
						add!(col, row);
					}

					if mask && triangle == 2 {
						label!({
							let cont;
						});
						cmp!(nrows, m * self.len());
						jc!(cont);

						{
							pop!(col);
							pop!(row);
							pop!(cs);
							pop!(rs);
							pop!(ptr);
							let f = &format!("{prefix}.epilogue.store{__triangle__}{__add__}");
							jmp!("{func_name(f, suffix, false)}");
						}
						label!(cont = _);
					}

					imul!(col, cs);
					add!(ptr, col);

					imul!(row, rs);
					add!(ptr, row);

					{
						mov!(row, [position]);
						mov!(col, [position + WORD]);
						sub!(row, col);
						let diff = row;
						shl!(diff, self.mask_sizeof().ilog2());
						if triangle == 1 {
							neg!(diff);
						}

						let mask_ = if simd.dedicated_mask() { 1 } else { m * n };
						let mask2_ = if simd.dedicated_mask() { 2 } else { m * n + 1 };
						let tmp = m * n + 2;

						if mask {
							label!({
								let load_mask;
								let cont;
							});

							cmp!(nrows, m * self.len());
							jc!(load_mask);

							lea!(mask_ptr, [rip + &func_name(&format!("{prefix}.mask.data"), "", false)]);
							add!(mask_ptr, self.len() * self.mask_sizeof());
							kmov!(k(mask_), [mask_ptr]);
							jmp!(cont);

							label!(load_mask = _);

							lea!(mask_ptr, [rip + &func_name(&format!("{prefix}.mask.data"), "", false)]);

							if self.mask_sizeof() <= 8 {
								kmov!(
									k(mask_),
									[mask_ptr + self.mask_sizeof() * nrows + -(m - 1) * self.len() * self.mask_sizeof()]
								);
							} else {
								for _ in 0..self.mask_sizeof() / 8 - 1 {
									lea!(mask_ptr, [mask_ptr + 8 * nrows]);
								}
								kmov!(k(mask_), [mask_ptr + 8 * nrows + -(m - 1) * self.len() * self.mask_sizeof()]);
							}

							label!(cont = _);

							if triangle == 0 {
								lea!(mask_ptr, [rip + &func_name(&format!("{prefix}.rmask.data"), "", false)]);
							}
							if triangle == 1 {
								lea!(mask_ptr, [rip + &func_name(&format!("{prefix}.mask.data"), "", false)]);
							}
						}

						let idk = 8usize.div_ceil(self.len() as usize) as isize;

						for j in 0..n {
							for i in 0..m {
								let src = m * j + i;
								let ptr = ptr + simd.sizeof() * i;

								if (!mask || i + 1 < m) && (triangle != 0 || i >= idk) && (triangle != 1 || i + 1 <= m - idk) {
									if add {
										vadd!(zmm(src), zmm(src), [ptr]);
									}
									vmov!([ptr], zmm(src));
								} else {
									let mask_ = if triangle == 0 && i < idk {
										alloca!(diff);
										reg!(tmp);
										xor!(tmp, tmp);
										add!(diff, self.mask_sizeof() * (self.len() + i * self.len() - j),);
										mov!(tmp, self.mask_sizeof() * self.len());
										cmp!(diff, tmp);
										cmovg!(diff, tmp);

										kmov!(k(mask2_), [mask_ptr + 1 * diff]);
										if i + 1 == m {
											kand!(k(mask2_), k(mask2_), k(mask_));
										}

										mask2_
									} else if triangle == 1 && i + 1 > m - idk {
										alloca!(diff);
										reg!(tmp);
										xor!(tmp, tmp);
										add!(diff, self.mask_sizeof() * (j + 1 - i * self.len()),);
										mov!(tmp, self.mask_sizeof() * self.len());
										cmp!(diff, tmp);
										cmovg!(diff, tmp);

										kmov!(k(mask2_), [mask_ptr + 1 * diff]);
										if i + 1 == m {
											kand!(k(mask2_), k(mask2_), k(mask_));
										}

										mask2_
									} else {
										mask_
									};

									if add {
										vmov!(zmm(tmp)[mask_], [ptr]);
										vadd!(zmm(src), zmm(src), zmm(tmp));
									}
									vmov!([ptr][mask_], zmm(src));
								}
							}
							add!(ptr, cs);
						}
					}

					label!(end = _);
				}
			}
		}

		(main, ctx.code.into_inner())
	}
}

fn main() -> Result {
	let mut code = String::new();

	let mut pack_c64_simd512 = vec![];
	let mut pack_c32_simd512 = vec![];
	let mut pack_f64_simd512 = vec![];
	let mut pack_f32_simd512 = vec![];

	let mut pack_c64_simd256 = vec![];
	let mut pack_c32_simd256 = vec![];
	let mut pack_f64_simd256 = vec![];
	let mut pack_f32_simd256 = vec![];

	let mut pack_c64_simd128 = vec![];
	let mut pack_c32_simd128 = vec![];
	let mut pack_f64_simd128 = vec![];
	let mut pack_f32_simd128 = vec![];

	let mut pack_f64_simd64 = vec![];
	let mut pack_c32_simd64 = vec![];
	let mut pack_f32_simd64 = vec![];

	let mut pack_f32_simd32 = vec![];

	let mut f32_simd512 = vec![];
	let mut c32_simd512 = vec![];
	let mut f64_simd512 = vec![];
	let mut c64_simd512 = vec![];

	let mut f32_simd512x8 = vec![];
	let mut c32_simd512x8 = vec![];
	let mut f64_simd512x8 = vec![];
	let mut c64_simd512x8 = vec![];

	let mut f32_simd256 = vec![];
	let mut c32_simd256 = vec![];
	let mut f64_simd256 = vec![];
	let mut c64_simd256 = vec![];

	let mut f32_simd128 = vec![];
	let mut c32_simd128 = vec![];
	let mut f64_simd128 = vec![];
	let mut c64_simd128 = vec![];

	let mut f32_simd64 = vec![];
	let mut c32_simd64 = vec![];
	let mut f64_simd64 = vec![];

	let mut f32_simd32 = vec![];

	for (out, pack, ty) in [
		(&mut f32_simd512, &mut pack_f32_simd512, Ty::F32),
		(&mut c32_simd512, &mut pack_c32_simd512, Ty::C32),
		(&mut f64_simd512, &mut pack_f64_simd512, Ty::F64),
		(&mut c64_simd512, &mut pack_c64_simd512, Ty::C64),
	] {
		for m in (1..=6).rev() {
			let target = Target { ty, simd: Simd::_512 };

			let last = if m == 1 { 8 } else { 4 };

			let (name, f) = target.pack_lhs(m);

			pack.push(name);
			code += &f;

			for n in 1..=last {
				let (name, f) = target.microkernel(m, n);

				code += &f;
				out.push(name);
			}
		}
	}

	for (out, ty) in [
		(&mut f32_simd512x8, Ty::F32),
		(&mut c32_simd512x8, Ty::C32),
		(&mut f64_simd512x8, Ty::F64),
		(&mut c64_simd512x8, Ty::C64),
	] {
		for m in (1..=3).rev() {
			for n in 1..=8 {
				let target = Target { ty, simd: Simd::_512 };

				let (name, f) = target.microkernel(m, n);
				if m > 1 && n > 4 {
					code += &f;
				}
				out.push(name);
			}
		}
	}

	for (out, pack, ty) in [
		(&mut f32_simd256, &mut pack_f32_simd256, Ty::F32),
		(&mut c32_simd256, &mut pack_c32_simd256, Ty::C32),
		(&mut f64_simd256, &mut pack_f64_simd256, Ty::F64),
		(&mut c64_simd256, &mut pack_c64_simd256, Ty::C64),
	] {
		for m in (1..=3).rev() {
			let target = Target { ty, simd: Simd::_256 };

			let last = if m == 1 { 8 } else { 4 };

			let (name, f) = target.pack_lhs(m);

			pack.push(name);
			code += &f;

			for n in 1..=last {
				let (name, f) = target.microkernel(m, n);
				code += &f;
				out.push(name);
			}
		}
	}

	for (out, pack, ty, simd) in [
		(&mut f32_simd128, &mut pack_f32_simd128, Ty::F32, Simd::_128),
		(&mut c32_simd128, &mut pack_c32_simd128, Ty::C32, Simd::_128),
		(&mut f64_simd128, &mut pack_f64_simd128, Ty::F64, Simd::_128),
		(&mut c64_simd128, &mut pack_c64_simd128, Ty::C64, Simd::_128),
		(&mut f32_simd64, &mut pack_f32_simd64, Ty::F32, Simd::_64),
		(&mut c32_simd64, &mut pack_c32_simd64, Ty::C32, Simd::_64),
		(&mut f64_simd64, &mut pack_f64_simd64, Ty::F64, Simd::_64),
		(&mut f32_simd32, &mut pack_f32_simd32, Ty::F32, Simd::_32),
	] {
		for m in (1..=1).rev() {
			let target = Target { ty, simd };

			let (name, f) = target.pack_lhs(m);

			pack.push(name);
			code += &f;

			for n in 1..=8 {
				let (name, f) = target.microkernel(m, n);
				code += &f;
				out.push(name);
			}
		}
	}

	let out_dir = env::var_os("OUT_DIR").unwrap();
	let dest_path = Path::new(&out_dir).join("asm.s");
	writeln!(
		code,
		"
            .p2align 6
            {func_name(\"gemm.microkernel.c64.flip.re.data\", \"\", true)}:
            .quad 0x8000000000000000,0,0x8000000000000000,0,0x8000000000000000,0,0x8000000000000000,0
            .p2align 6
            {func_name(\"gemm.microkernel.c64.flip.im.data\", \"\", true)}:
            .quad 0,0x8000000000000000,0,0x8000000000000000,0,0x8000000000000000,0,0x8000000000000000

            .p2align 6
            {func_name(\"gemm.microkernel.c32.flip.re.data\", \"\", true)}:
            .int 0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0
            .p2align 6
            {func_name(\"gemm.microkernel.c32.flip.im.data\", \"\", true)}:
            .int 0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000




            

            .p2align 4
            {func_name(\"gemm.microkernel.c64.simd128.rmask.data\", \"\", true)}:
            {func_name(\"gemm.microkernel.c64.simd128.mask.data\", \"\", true)}:
            .octa 0, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

            .p2align 5
            {func_name(\"gemm.microkernel.c64.simd256.rmask.data\", \"\", true)}:
            .octa 0,0, 0,0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

            .p2align 5
            {func_name(\"gemm.microkernel.c64.simd256.mask.data\", \"\", true)}:
            .octa 0,0, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,0, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


            {func_name(\"gemm.microkernel.c64.simd512.rmask.data\", \"\", true)}:
            .byte 0b00000000, 0b11000000, 0b11110000, 0b11111100, 0b11111111

            {func_name(\"gemm.microkernel.c64.simd512.mask.data\", \"\", true)}:
            .byte 0b00000000, 0b00000011, 0b00001111, 0b00111111, 0b11111111

            {func_name(\"gemm.microkernel.f64.simd512.rmask.data\", \"\", true)}:
            .byte 0b00000000, 0b10000000, 0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111

            {func_name(\"gemm.microkernel.f64.simd512.mask.data\", \"\", true)}:
            .byte 0b00000000, 0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111





            .p2align 4
            {func_name(\"gemm.microkernel.c32.simd128.rmask.data\", \"\", true)}:
            {func_name(\"gemm.microkernel.f64.simd128.rmask.data\", \"\", true)}:
            .quad 0,0, 0,-1, -1,-1

            .p2align 4
            {func_name(\"gemm.microkernel.c32.simd128.mask.data\", \"\", true)}:
            {func_name(\"gemm.microkernel.f64.simd128.mask.data\", \"\", true)}:
            .quad 0,0, -1,0, -1,-1
            
            .p2align 5
            {func_name(\"gemm.microkernel.c32.simd256.rmask.data\", \"\", true)}:
            {func_name(\"gemm.microkernel.f64.simd256.rmask.data\", \"\", true)}:
            .quad 0,0,0,0, 0,0,0,-1, 0,0,-1,-1, 0,-1,-1,-1, -1,-1,-1,-1

            .p2align 5
            {func_name(\"gemm.microkernel.c32.simd256.mask.data\", \"\", true)}:
            {func_name(\"gemm.microkernel.f64.simd256.mask.data\", \"\", true)}:
            .quad 0,0,0,0, -1,0,0,0, -1,-1,0,0, -1,-1,-1,0, -1,-1,-1,-1


            .p2align 1
            {func_name(\"gemm.microkernel.c32.simd512.rmask.data\", \"\", true)}:
            .word 0b0000000000000000, 0b1100000000000000, 0b1111000000000000, 0b1111110000000000, 0b1111111100000000, 0b1111111111000000, \
		 0b1111111111110000, 0b1111111111111100, 0b1111111111111111

            .p2align 1
            {func_name(\"gemm.microkernel.c32.simd512.mask.data\", \"\", true)}:
            .word 0b0000000000000000, 0b0000000000000011, 0b0000000000001111, 0b0000000000111111, 0b0000000011111111, 0b0000001111111111, \
		 0b0000111111111111, 0b0011111111111111, 0b1111111111111111




            
            .p2align 4
            {func_name(\"gemm.microkernel.f32.simd128.rmask.data\", \"\", true)}:
            .int 0,0,0,0, 0,0,0,-1, 0,0,-1,-1, 0,-1,-1,-1, -1,-1,-1,-1

            .p2align 4
            {func_name(\"gemm.microkernel.f32.simd128.mask.data\", \"\", true)}:
            .int 0,0,0,0, -1,0,0,0, -1,-1,0,0, -1,-1,-1,0, -1,-1,-1,-1
            
            .p2align 5
            {func_name(\"gemm.microkernel.f32.simd256.rmask.data\", \"\", true)}:
            .int 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,-1, 0,0,0,0,0,0,-1,-1, 0,0,0,0,0,-1,-1,-1, 0,0,0,0,-1,-1,-1,-1, 0,0,0,-1,-1,-1,-1,-1, \
		 0,0,-1,-1,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1

            .p2align 5
            {func_name(\"gemm.microkernel.f32.simd256.mask.data\", \"\", true)}:
            .int 0,0,0,0,0,0,0,0, -1,0,0,0,0,0,0,0, -1,-1,0,0,0,0,0,0, -1,-1,-1,0,0,0,0,0, -1,-1,-1,-1,0,0,0,0, -1,-1,-1,-1,-1,0,0,0, \
		 -1,-1,-1,-1,-1,-1,0,0, -1,-1,-1,-1,-1,-1,-1,0, -1,-1,-1,-1,-1,-1,-1,-1


            .p2align 1
            {func_name(\"gemm.microkernel.f32.simd512.rmask.data\", \"\", true)}:
            .word 0b0000000000000000, 0b1000000000000000, 0b1100000000000000, 0b1110000000000000, 0b1111000000000000, 0b1111100000000000, \
		 0b1111110000000000, 0b1111111000000000, 0b1111111100000000, 0b1111111110000000, 0b1111111111000000, 0b1111111111100000, \
		 0b1111111111110000, 0b1111111111111000, 0b1111111111111100, 0b1111111111111110, 0b1111111111111111

            .p2align 1
            {func_name(\"gemm.microkernel.f32.simd512.mask.data\", \"\", true)}:
            .word 0b0000000000000000, 0b0000000000000001, 0b0000000000000011, 0b0000000000000111, 0b0000000000001111, 0b0000000000011111, \
		 0b0000000000111111, 0b0000000001111111, 0b0000000011111111, 0b0000000111111111, 0b0000001111111111, 0b0000011111111111, \
		 0b0000111111111111, 0b0001111111111111, 0b0011111111111111, 0b0111111111111111, 0b1111111111111111
        "
	)?;

	fs::write(&dest_path, &code)?;

	{
		let dest_path = Path::new(&out_dir).join("asm.rs");

		let mut code = format!("::core::arch::global_asm!{{ include_str!(concat!(env!({QUOTE}OUT_DIR{QUOTE}), {QUOTE}/asm.s{QUOTE})) }}");

		for (names, ty, bits) in [
			(&f32_simd512x8, Ty::F32, "512x8"),
			(&c32_simd512x8, Ty::C32, "512x8"),
			(&f64_simd512x8, Ty::F64, "512x8"),
			(&c64_simd512x8, Ty::C64, "512x8"),
			(&f32_simd512, Ty::F32, "512x4"),
			(&c32_simd512, Ty::C32, "512x4"),
			(&f64_simd512, Ty::F64, "512x4"),
			(&c64_simd512, Ty::C64, "512x4"),
			(&f32_simd256, Ty::F32, "256"),
			(&c32_simd256, Ty::C32, "256"),
			(&f64_simd256, Ty::F64, "256"),
			(&c64_simd256, Ty::C64, "256"),
			(&f32_simd128, Ty::F32, "128"),
			(&c32_simd128, Ty::C32, "128"),
			(&f64_simd128, Ty::F64, "128"),
			(&c64_simd128, Ty::C64, "128"),
			(&f32_simd64, Ty::F32, "64"),
			(&c32_simd64, Ty::C32, "64"),
			(&f64_simd64, Ty::F64, "64"),
			(&pack_f32_simd512, Ty::F32, "pack_512"),
			(&pack_f64_simd512, Ty::F64, "pack_512"),
			(&pack_c32_simd512, Ty::C32, "pack_512"),
			(&pack_c64_simd512, Ty::C64, "pack_512"),
			(&pack_f32_simd256, Ty::F32, "pack_256"),
			(&pack_f64_simd256, Ty::F64, "pack_256"),
			(&pack_c32_simd256, Ty::C32, "pack_256"),
			(&pack_c64_simd256, Ty::C64, "pack_256"),
			(&pack_f32_simd128, Ty::F32, "pack_128"),
			(&pack_f64_simd128, Ty::F64, "pack_128"),
			(&pack_c32_simd128, Ty::C32, "pack_128"),
			(&pack_c64_simd128, Ty::C64, "pack_128"),
			(&pack_f32_simd64, Ty::F32, "pack_64"),
			(&pack_c32_simd64, Ty::C32, "pack_64"),
			(&pack_f64_simd64, Ty::F64, "pack_64"),
		] {
			for (i, name) in names.iter().enumerate() {
				let name = if name.starts_with(QUOTE) {
					name.clone()
				} else {
					format!("{QUOTE}{name}{QUOTE}")
				};
				code += &format!(
					"
                unsafe extern {QUOTE}C{QUOTE} {{
                    #[link_name = ::core::concat!(\"\\x01\", {name})]
                    unsafe fn __decl_{ty}_simd{bits}_{i}__();
                }}
                "
				);
			}

			let upper = format!("{ty}").to_uppercase();
			code += &format!("pub static {upper}_SIMD{bits}: [unsafe extern {QUOTE}C{QUOTE} fn(); {names.len()}] = [");
			for i in 0..names.len() {
				code += &format!("__decl_{ty}_simd{bits}_{i}__,");
			}
			code += "];";
		}

		fs::write(&dest_path, &code)?;
	}

	println!("cargo::rerun-if-changed=build.rs");

	Ok(())
}
