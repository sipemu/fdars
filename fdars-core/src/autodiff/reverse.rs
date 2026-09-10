//! Reverse-mode (VJP) automatic differentiation — stub placeholder.
//!
//! Full implementation added in Task 2.

use super::Scalar;

// Temporary stub so the module compiles for Task 1 verification.
// The full `Var`, `Node`, `TAPE`, and `vjp` are added in Task 2/3.

#[derive(Debug, Clone, Copy)]
pub struct Var {
    pub(crate) value: f64,
    pub(crate) node: usize,
}

#[must_use]
pub fn vjp<F: Fn(&[Var]) -> Var>(_f: F, x: &[f64]) -> (f64, Vec<f64>) {
    unimplemented!("vjp not yet implemented — added in Task 3");
    #[allow(unreachable_code)]
    (0.0, vec![0.0; x.len()])
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for Var {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Scalar for Var {
    fn zero() -> Self {
        Var {
            value: 0.0,
            node: usize::MAX,
        }
    }
    fn one() -> Self {
        Var {
            value: 1.0,
            node: usize::MAX,
        }
    }
    fn from_f64(v: f64) -> Self {
        Var {
            value: v,
            node: usize::MAX,
        }
    }
    fn infinity() -> Self {
        Var {
            value: f64::INFINITY,
            node: usize::MAX,
        }
    }
    fn sqrt(self) -> Self {
        unimplemented!()
    }
    fn exp(self) -> Self {
        unimplemented!()
    }
    fn ln(self) -> Self {
        unimplemented!()
    }
    fn sin(self) -> Self {
        unimplemented!()
    }
    fn cos(self) -> Self {
        unimplemented!()
    }
    fn powf(self, _p: f64) -> Self {
        unimplemented!()
    }
    fn abs(self) -> Self {
        unimplemented!()
    }
    fn signum(self) -> Self {
        unimplemented!()
    }
}

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

impl Add for Var {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
impl Sub for Var {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
impl Mul for Var {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
impl Div for Var {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
impl Neg for Var {
    type Output = Self;
    fn neg(self) -> Self {
        unimplemented!()
    }
}
impl AddAssign for Var {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for Var {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for Var {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
