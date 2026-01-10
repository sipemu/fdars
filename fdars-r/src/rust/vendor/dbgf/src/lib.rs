#![no_std]

#[macro_export]
macro_rules! dbgf {
    ($fmt: tt $(,)?) => {{
            extern crate std as __std;
            __std::eprintln!("[{}:{}:{}]", __std::file!(), __std::line!(), __std::column!())
    }};
    ($fmt: tt, $val:expr $(,)?) => {
        match $val {
            tmp => {{
                extern crate std as __std;
                __std::eprintln!(__std::concat!("[{}:{}:{}] {} = {:", $fmt, "}"),
                __std::file!(), __std::line!(), __std::column!(), __std::stringify!($val), &tmp);
                tmp
            }}
        }
    };
    ($fmt: tt $(, $val:expr)+ $(,)?) => {
        ($($crate::dbgf!($fmt, $val)),+,)
    };
}

#[cfg(test)]
mod tests {
    extern crate std as extern_std;

    use super::*;

    #[test]
    fn it_works() {
        #[derive(Debug, Clone)]
        struct S {
            i: extern_std::vec::Vec<f32>,
        }

        let s = S {
            i: extern_std::vec![11.0 / 3.0; 10],
        };
        extern_std::dbg!(&s, &s.i);
        dbgf!("5.3?", &s, &s.i);
    }
}
