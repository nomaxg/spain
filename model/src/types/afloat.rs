use super::{HighPrecision, ToPrimitiveExt};
use core::fmt;
use i256::{I256, I512};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rug::ops::Pow;
use rug::{Float, Integer};
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Sub};

pub static FBITS: u32 = 128;
#[derive(Debug, Clone)]
pub struct AFloat(pub Float);

impl Add for AFloat {
    type Output = AFloat;

    fn add(self, other: Self) -> Self {
        Self(Float::with_val(FBITS, &self.0 + &other.0))
    }
}

impl Sub for AFloat {
    type Output = AFloat;

    fn sub(self, other: Self) -> Self {
        Self(Float::with_val(FBITS, &self.0 - &other.0))
    }
}

impl Mul for AFloat {
    type Output = AFloat;

    fn mul(self, other: Self) -> Self {
        Self(Float::with_val(FBITS, &self.0 * &other.0))
    }
}

impl Div for AFloat {
    type Output = AFloat;

    fn div(self, other: Self) -> Self {
        Self(Float::with_val(FBITS, &self.0 / &other.0))
    }
}

impl Zero for AFloat {
    fn zero() -> Self {
        Self(Float::with_val(FBITS, 0))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl FromPrimitive for AFloat {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(Float::with_val(FBITS, n)))
    }

    fn from_i128(n: i128) -> Option<Self> {
        Some(Self(Float::with_val(FBITS, n)))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(Float::with_val(FBITS, n)))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self(Float::with_val(FBITS, n)))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self(Float::with_val(FBITS, n)))
    }
}

impl ToPrimitive for AFloat {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_integer().unwrap().to_i64()
    }

    fn to_i128(&self) -> Option<i128> {
        self.0.to_integer().unwrap().to_i128()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_integer().unwrap().to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0.to_f32())
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0.to_f64())
    }
}

impl PartialEq for AFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for AFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Default for AFloat {
    fn default() -> Self {
        Self::from_f64(0_f64).unwrap()
    }
}

impl Display for AFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ToPrimitiveExt for AFloat {
    fn to_i256(&self) -> I256 {
        I256::from_str_radix(self.to_rug_integer().to_string_radix(10).as_str(), 10).unwrap()
    }

    fn to_i512(&self) -> I512 {
        I512::from_str_radix(self.to_rug_integer().to_string_radix(10).as_str(), 10).unwrap()
    }

    fn to_rug_integer(&self) -> Integer {
        self.0.to_integer().unwrap()
    }

    fn to_rug_float(&self) -> Float {
        self.0.clone()
    }
}

impl HighPrecision for AFloat {
    fn sqrt(&self) -> Self {
        AFloat(self.0.clone().sqrt())
    }

    fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    fn powf(&self, x: f64) -> Self {
        AFloat(self.0.clone().pow(Float::with_val(FBITS, x)))
    }

    fn max(&self, other: &Self) -> Self {
        AFloat(Float::max(self.0.clone(), &other.0))
    }

    fn recip(&self) -> Self {
        AFloat(Float::recip(self.0.clone()))
    }
}
