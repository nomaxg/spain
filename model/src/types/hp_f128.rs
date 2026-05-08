use super::{FBITS, HighPrecision, ToPrimitiveExt};
use core::fmt;
use i256::{I256, I512};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rug::{Float, Integer};
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct F128(f128);

impl Add for F128 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for F128 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul for F128 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl Div for F128 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl Zero for F128 {
    fn zero() -> Self {
        Self(0_f128)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0_f128
    }
}

impl FromPrimitive for F128 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(F128(n as f128))
    }

    fn from_i128(n: i128) -> Option<Self> {
        Some(F128(n as f128))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(F128(n as f128))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(F128(n as f128))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(F128(n as f128))
    }
}

impl ToPrimitive for F128 {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }

    fn to_i128(&self) -> Option<i128> {
        Some(self.0 as i128)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0 as f64)
    }
}

impl PartialEq for F128 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for F128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Default for F128 {
    fn default() -> Self {
        Self::from_f64(0_f64).unwrap()
    }
}

impl Display for F128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 as f64)
    }
}

impl ToPrimitiveExt for F128 {
    fn to_i256(&self) -> I256 {
        I256::from_str_radix(self.to_rug_integer().to_string_radix(10).as_str(), 10).unwrap()
    }

    fn to_i512(&self) -> I512 {
        I512::from_str_radix(self.to_rug_integer().to_string_radix(10).as_str(), 10).unwrap()
    }

    fn to_rug_integer(&self) -> Integer {
        self.to_rug_float().to_integer().unwrap()
    }

    fn to_rug_float(&self) -> Float {
        // use f128::f128;
        Float::with_val(FBITS, self.0 as i128)
    }
}

impl HighPrecision for F128 {
    fn sqrt(&self) -> Self {
        F128(self.0.sqrt())
    }

    fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    fn powf(&self, x: f64) -> Self {
        F128(self.0.powf(x as f128))
    }

    fn max(&self, other: &Self) -> Self {
        F128(f128::max(self.0, other.0))
    }

    fn recip(&self) -> Self {
        F128(f128::recip(self.0))
    }
}
