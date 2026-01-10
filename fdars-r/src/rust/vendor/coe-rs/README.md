[![Documentation](https://docs.rs/coe-rs/badge.svg)](https://docs.rs/coe-rs)
[![Crate](https://img.shields.io/crates/v/coe-rs.svg)](https://crates.io/crates/coe-rs)

`coe-rs` is a Rust library for coercing a value of a given type into the same type, in cases
where the compiler can't prove the two types are equal.  
This can be used to emulate specialization in to a limited extent.

# Example

```rust
use coe::{Coerce, is_same};
use core::ops::Add;
fn foo<T: 'static + Copy + Add<Output = T>>(slice: &mut [T]) {
    if is_same::<f64, T>() {
        // use some optimized SIMD implementation
        // ...
        println!("using SIMD operations");
        let slice: &mut [f64] = slice.coerce();
    } else {
        for value in slice {
            println!("using fallback implementation");
            *value = *value + *value;
        }
    }
}
foo(&mut [1, 2, 3u64]); // calls fallback implementation
foo(&mut [1.0, 2.0, 3.0f64]); // calls SIMD implementation
```
