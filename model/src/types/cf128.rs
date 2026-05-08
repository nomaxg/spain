use super::{FBITS, HighPrecision, ToPrimitiveExt};
use core::fmt;
use f128::f128;
use i256::{I256, I512};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rug::{Float, Integer};
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct CF128(f128);

impl Add for CF128 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for CF128 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul for CF128 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl Div for CF128 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl Zero for CF128 {
    fn zero() -> Self {
        Self(0.into())
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.into()
    }
}

impl FromPrimitive for CF128 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(CF128(n.into()))
    }

    fn from_i128(n: i128) -> Option<Self> {
        Some(CF128(n.into()))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(CF128(n.into()))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(CF128(n.into()))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(CF128(n.into()))
    }
}

impl ToPrimitive for CF128 {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0.into())
    }

    fn to_i128(&self) -> Option<i128> {
        Some(self.0.into())
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0.into())
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0.into())
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0.into())
    }
}

impl PartialEq for CF128 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for CF128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Default for CF128 {
    fn default() -> Self {
        Self::from_f64(0_f64).unwrap()
    }
}

impl Display for CF128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ToPrimitiveExt for CF128 {
    fn to_i256(&self) -> I256 {
        // TODO potential speed up?
        I256::from_str_radix(self.to_rug_integer().to_string_radix(10).as_str(), 10).unwrap()
    }

    fn to_i512(&self) -> I512 {
        I512::from_str_radix(self.to_rug_integer().to_string_radix(10).as_str(), 10).unwrap()
    }

    fn to_rug_integer(&self) -> Integer {
        self.to_rug_float().to_integer().unwrap()
    }

    fn to_rug_float(&self) -> Float {
        Float::with_val(
            FBITS,
            Float::parse(format!("{}", self.0.to_string_fmt("%.35Qf").unwrap())).unwrap(),
        )
    }
}

impl HighPrecision for CF128 {
    fn sqrt(&self) -> Self {
        use num_traits::Float;
        CF128(self.0.sqrt())
    }

    fn is_nan(&self) -> bool {
        use num_traits::Float;
        self.0.is_nan()
    }

    fn powf(&self, x: f64) -> Self {
        use num_traits::Float;
        CF128(self.0.powf(x.into()))
    }

    fn max(&self, other: &Self) -> Self {
        use num_traits::Float;
        CF128(f128::max(self.0, other.0))
    }

    fn recip(&self) -> Self {
        use num_traits::Float;
        CF128(f128::recip(self.0))
    }
}
