use std::ops::Mul;

use i256::{I256, U256};
use rug::Integer;
use serde::{Deserialize, Serialize};
// traditional modular arithmetic operations
// not super optimized yet, but should be good enough for now

// zero-cost abstraction for u128 in montgomery form
// this just helps ensure correctness
#[derive(PartialEq, Eq, Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct M128(u128);

#[inline(always)]
pub fn neg_mod(x: u128, m: u128) -> u128 {
    if x == 0 { x } else { m - x }
}

#[inline(always)]
pub fn add_mod(x: u128, y: u128, m: u128) -> u128 {
    let s = x.wrapping_add(y);
    if x >= m - y { s.wrapping_sub(m) } else { s }
}

#[inline(always)]
pub fn sub_mod(x: u128, y: u128, m: u128) -> u128 {
    let d = x.wrapping_sub(y);
    if x < y { d.wrapping_add(m) } else { d }
}

#[inline(always)]
pub fn mul_mod(x: u128, y: u128, m: u128) -> u128 {
    ((U256::from(x) * U256::from(y)) % U256::from(m)).as_u128()
}

// montgomery operations adapted from the following sources
// https://en.algorithmica.org/hpc/number-theory/montgomery/
// https://github.com/cp-algorithms/cp-algorithms/blob/main/src/algebra/montgomery_multiplication.md
#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Mont {
    m: u128,   // modulus
    mr: u128,  // inverse of m mod 2^64
    r2: u128,  // r^2 mod m
    one: M128, // 1 in montgomery form
}

impl Mont {
    // create a new montgomery context with modulus m
    #[inline(always)]
    pub fn new(m: u128) -> Mont {
        // compute inverse of m mod 2^128 using newton-raphson
        let mut mr = 1u128;
        for _ in 0..7 {
            mr = mr.wrapping_mul(2u128.wrapping_sub(mr.wrapping_mul(m)));
        }
        // compute r^2 mod m
        // r = 2^128, so r^2 = 2^256
        let r: U256 = (U256::from(1u128) << 128) % U256::from(m);
        let r2 = ((r.mul(r)).rem_euclid(U256::from(m))).as_u128();
        // compute 1 in montgomery form
        let one = M128(((U256::from(1u128) << 128u32).rem_euclid(U256::from(m))).as_u128());
        // build the montgomery context
        Mont { m, mr, r2, one }
    }
    #[inline(always)]
    pub fn from_bigint(&self, x: Integer) -> M128 {
        let tmp = x.clone().abs() % self.modulus();
        let mut n = tmp.to_u128_wrapping();
        if x.is_negative() {
            n = self.modulus() - n; // Adjust for negative values
        }
        self.to_mont(n)
    }
    #[inline(always)]
    pub fn from_i128(&self, x: i128) -> M128 {
        let n = self.to_mont(x.unsigned_abs());
        if x < 0 { self.neg(n) } else { n }
    }
    #[inline(always)]
    pub fn from_i256(&self, x: I256) -> M128 {
        let is_negative = x < I256::from(0);
        let abs_x = if is_negative { -x } else { x };
        let limbs = abs_x.to_le_limbs();
        let base = self.to_mont(1u128 << 64);
        let mut n = self.zero();
        for limb in limbs.iter().rev() {
            n = self.mul(n, base);
            n = self.add(n, self.to_mont(*limb as u128));
        }
        if is_negative { self.neg(n) } else { n }
    }
    // exclusively for testing
    #[inline(always)]
    pub fn literal(&self, x: u128) -> M128 {
        M128(x)
    }
    // get modulus
    #[inline(always)]
    pub fn modulus(&self) -> u128 {
        self.m
    }
    // convert to montgomery form
    #[inline(always)]
    pub fn to_mont(&self, x: u128) -> M128 {
        if x == 0 {
            self.zero()
        } else if x == 1 {
            self.one
        } else {
            self.reduce(U256::from(x) * U256::from(self.r2))
        }
    }
    // convert from montgomery form to normal form
    #[inline(always)]
    pub fn to_normal(&self, x: M128) -> u128 {
        // reduce to normal form
        self.reduce(U256::from(x.0)).0
    }
    // convert from montgomery form to normal form as an Integer
    #[inline(always)]
    pub fn to_integer(&self, x: M128) -> Integer {
        Integer::from(self.to_normal(x))
    }
    // get zero in montgomery form
    #[inline(always)]
    pub fn zero(&self) -> M128 {
        M128(0)
    }
    // get one in montgomery form
    #[inline(always)]
    pub fn one(&self) -> M128 {
        self.one
    }
    // is 0 in montgomery form
    // this is the same as checking if x == 0 in normal form
    #[inline(always)]
    pub fn is_zero(&self, x: M128) -> bool {
        x == M128(0)
    }
    // check if x is 1 in montgomery form
    #[inline(always)]
    pub fn is_one(&self, x: M128) -> bool {
        x == self.one
    }
    // redc (reduce mod m and eliminate an extra montgomery factor)
    #[inline(always)]
    fn reduce(&self, x: U256) -> M128 {
        let q = (U256::from(x.as_u128()) * U256::from(self.mr)).as_u128();
        let n = U256::from(q) * U256::from(self.m);
        let y = ((x.wrapping_sub(n)) >> 128u32).as_u128();
        M128(if x < n { y.wrapping_add(self.m) } else { y })
    }
    // multiply in montgomery form
    #[inline(always)]
    pub fn mul(&self, x: M128, y: M128) -> M128 {
        if x == M128(0) || y == M128(0) {
            self.zero()
        } else if x == self.one {
            y
        } else if y == self.one {
            x
        } else {
            self.reduce(U256::from(x.0) * U256::from(y.0))
        }
    }
    // divide
    #[inline(always)]
    pub fn div(&self, x: M128, y: M128) -> M128 {
        if self.is_zero(y) {
            panic!("Attempt to divide by zero");
        }
        // x * y^-1
        let y_inv = self.inv(y);
        self.mul(x, y_inv)
    }
    // square in montgomery form
    #[inline(always)]
    pub fn sqr(&self, x: M128) -> M128 {
        let x = U256::from(x.0);
        self.reduce(x * x)
    }
    // add
    #[inline(always)]
    pub fn add(&self, x: M128, y: M128) -> M128 {
        M128(add_mod(x.0, y.0, self.m))
    }
    // subtract
    #[inline(always)]
    pub fn sub(&self, x: M128, y: M128) -> M128 {
        M128(sub_mod(x.0, y.0, self.m))
    }
    // negate
    #[inline(always)]
    pub fn neg(&self, x: M128) -> M128 {
        M128(neg_mod(x.0, self.m))
    }
    // exponentiate
    #[inline(always)]
    pub fn exp(&self, x: M128, e: u128) -> M128 {
        let mut r = self.one;
        let mut b = x;
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(r, b);
            }
            e >>= 1;
            b = self.sqr(b);
        }
        r
    }
    // invert in montgomery form using a call to exp
    // Slower than the extended euclidean algorithm but good enough for now
    // note, if you are calling this on each element in an array of elements
    // you should use the batch inversion algorithm!
    #[inline(always)]
    pub fn inv(&self, x: M128) -> M128 {
        if self.is_zero(x) {
            panic!("Attempt to invert zero");
        }
        self.exp(x, self.m - 2)
    }
}
