use super::{FBITS, HighPrecision, ToPrimitiveExt};
use core::fmt;
use i256::{I256, I512};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rug::{Float, Integer};
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Sub};
use twofloat::TwoFloat;

#[derive(Debug, Clone)]
pub struct TFloat(TwoFloat);

impl Add for TFloat {
    type Output = TFloat;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for TFloat {
    type Output = TFloat;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul for TFloat {
    type Output = TFloat;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl Div for TFloat {
    type Output = TFloat;

    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl Zero for TFloat {
    fn zero() -> Self {
        Self(TwoFloat::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl FromPrimitive for TFloat {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(TwoFloat::from_i64(n)?))
    }

    fn from_i128(n: i128) -> Option<Self> {
        Some(Self(TwoFloat::from_i128(n)?))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(TwoFloat::from_u64(n)?))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self(TwoFloat::from_f64(n)))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self(TwoFloat::from_f64(n as f64)))
    }
}

impl ToPrimitive for TFloat {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_i128(&self) -> Option<i128> {
        self.0.to_i128()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        self.0.to_f64().map(|v| v as f32)
    }

    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl PartialEq for TFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for TFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Default for TFloat {
    fn default() -> Self {
        Self::from_f64(0_f64).unwrap()
    }
}

impl Display for TFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            Float::with_val(128, self.0.lo()) + Float::with_val(128, self.0.hi())
        )
    }
}

impl ToPrimitiveExt for TFloat {
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
        Float::with_val(FBITS, self.0.lo()) + Float::with_val(FBITS, self.0.hi())
    }
}

impl HighPrecision for TFloat {
    fn sqrt(&self) -> Self {
        TFloat(self.0.sqrt())
    }

    fn is_nan(&self) -> bool {
        use num_traits::Float;
        self.0.is_nan()
    }

    fn powf(&self, x: f64) -> Self {
        TFloat(self.0.powf(TwoFloat::from(x)))
    }

    fn max(&self, other: &Self) -> Self {
        TFloat(TwoFloat::max(self.0, other.0))
    }

    fn recip(&self) -> Self {
        TFloat(TwoFloat::recip(self.0))
    }
}
