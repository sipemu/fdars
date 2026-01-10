//! `coe-rs` is a Rust library for coercing a value of a given type into the same type, in cases
//! where the compiler can't prove the two types are equal.  
//! This can be used to emulate specialization in to a limited extent.
#![no_std]

use core::any::TypeId;
use core::mem::transmute;

/// Returns `true` if `T` and `U` are the same type.
#[inline(always)]
pub fn is_same<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// Checks if `T` and `U` are the same type, and panics if that's not the case.
#[track_caller]
#[inline(always)]
pub fn assert_same<T: 'static, U: 'static>() {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
}

/// Trait for performing coercion from one type to another, where the types
/// are identical but the compiler can't prove it.
///
/// # Example
/// ```
/// use coe::{Coerce, is_same};
/// use core::ops::Add;
///
/// fn foo<T: 'static + Copy + Add<Output = T>>(slice: &mut [T]) {
///     if is_same::<f64, T>() {
///         // use some optimized SIMD implementation
///         // ...
///         println!("using SIMD operations");
///         let slice: &mut [f64] = slice.coerce();
///     } else {
///         for value in slice {
///             println!("using fallback implementation");
///             *value = *value + *value;
///         }
///     }
/// }
///
/// foo(&mut [1, 2, 3u64]); // calls fallback implementation
/// foo(&mut [1.0, 2.0, 3.0f64]); // calls SIMD implementation
/// ```
pub trait Coerce<U> {
    fn coerce(self) -> U;
}

impl<'a, T: 'static, U: 'static> Coerce<&'a U> for &'a T {
    #[inline(always)]
    #[track_caller]
    fn coerce(self) -> &'a U {
        assert_same::<T, U>();
        unsafe { transmute(self) }
    }
}

impl<'a, T: 'static, U: 'static> Coerce<&'a mut U> for &'a mut T {
    #[inline(always)]
    #[track_caller]
    fn coerce(self) -> &'a mut U {
        assert_same::<T, U>();
        unsafe { transmute(self) }
    }
}

impl<'a, T: 'static, U: 'static> Coerce<&'a [U]> for &'a [T] {
    #[inline(always)]
    #[track_caller]
    fn coerce(self) -> &'a [U] {
        assert_same::<T, U>();
        unsafe { transmute(self) }
    }
}

impl<'a, T: 'static, U: 'static> Coerce<&'a mut [U]> for &'a mut [T] {
    #[inline(always)]
    #[track_caller]
    fn coerce(self) -> &'a mut [U] {
        assert_same::<T, U>();
        unsafe { transmute(self) }
    }
}

/// Free function that defers to the `Coerce` trait implementation.
#[inline(always)]
pub fn coerce<T: Coerce<U>, U>(value: T) -> U {
    value.coerce()
}

/// Similar to [`coerce`] but operates on any lifetime-free type.
#[inline(always)]
pub fn coerce_static<T: 'static, U: 'static>(value: T) -> U {
    assert_same::<T, U>();
    unsafe { core::mem::transmute_copy(&core::mem::ManuallyDrop::new(value)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coerce() {
        let mut ints = [0, 1, 2u32];
        let mut floats = [0.0, 1.0, 2.0f64];

        pub fn generic_fn<T: 'static>(factor: T, slice: &mut [T]) {
            if is_same::<u32, T>() {
                let slice: &mut [u32] = slice.coerce();
                let factor: u32 = coerce_static(factor);
                for x in slice {
                    *x = 2 * factor * *x;
                }
            } else if is_same::<f64, T>() {
                let slice: &mut [f64] = slice.coerce();
                let factor: f64 = coerce_static(factor);
                for x in slice {
                    *x = factor * *x;
                }
            }
        }

        generic_fn(2, &mut ints);
        generic_fn(2.0, &mut floats);

        assert_eq!(ints, [0, 4, 8]);
        assert_eq!(floats, [0.0, 2.0, 4.0]);
    }
}
