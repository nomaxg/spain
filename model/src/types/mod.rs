pub mod afloat;
pub mod hp_f128;
pub mod tfloat;
pub use afloat::*;
pub use hp_f128::*;
pub use tfloat::*;

use i256::{I256, I512};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rug::{Float, Integer};
use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

pub trait ToPrimitiveExt {
    fn to_i256(&self) -> I256;
    fn to_i512(&self) -> I512;
    fn to_rug_integer(&self) -> Integer;
    fn to_rug_float(&self) -> Float;
}

pub trait HighPrecision:
    Sized
    + Display
    + Clone
    + Debug
    + ToPrimitive
    + FromPrimitive
    + ToPrimitiveExt
    + Zero
    + Div<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Default
    + PartialEq
    + PartialOrd
{
    fn sqrt(&self) -> Self;
    fn is_nan(&self) -> bool;
    fn powf(&self, x: f64) -> Self;
    fn max(&self, other: &Self) -> Self;
    fn recip(&self) -> Self;
}

macro_rules! default_implement_to_primitive_ext {
    ($t:ty) => {
        impl ToPrimitiveExt for $t {
            // NOTE potentially exploding i128 here, but fine for our import
            fn to_i256(&self) -> I256 {
                I256::from(*self as i128)
            }

            fn to_i512(&self) -> I512 {
                I512::from(*self as i128)
            }

            fn to_rug_integer(&self) -> Integer {
                Integer::from(*self as i128)
            }

            fn to_rug_float(&self) -> Float {
                Float::with_val(FBITS, *self)
            }
        }
    };
}

macro_rules! default_implement_high_precision {
    ($t:ty) => {
        impl HighPrecision for $t {
            fn sqrt(&self) -> Self {
                Self::sqrt(*self)
            }

            fn is_nan(&self) -> bool {
                Self::is_nan(*self)
            }

            fn powf(&self, x: f64) -> Self {
                (*self).powf(x as Self)
            }

            fn max(&self, other: &Self) -> Self {
                Self::max(*self, *other)
            }

            fn recip(&self) -> Self {
                Self::recip(*self)
            }
        }
    };
}

default_implement_to_primitive_ext!(f64);
default_implement_to_primitive_ext!(f32);
default_implement_high_precision!(f32);
default_implement_high_precision!(f64);
