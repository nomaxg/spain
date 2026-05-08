use rug::Integer;
use serde::{Deserialize, Serialize};
// traditional modular arithmetic operations
// not super optimized yet, but should be good enough for now

// zero-cost abstraction for u64 in montgomery form
// this just helps ensure correctness
#[derive(PartialEq, Eq, Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct M64(u64);

#[inline(always)]
pub fn neg_mod(x: u64, m: u64) -> u64 {
    if x == 0 { x } else { m - x }
}

#[inline(always)]
pub fn add_mod(x: u64, y: u64, m: u64) -> u64 {
    let s = x.wrapping_add(y);
    if x >= m - y { s.wrapping_sub(m) } else { s }
}

#[inline(always)]
pub fn sub_mod(x: u64, y: u64, m: u64) -> u64 {
    let d = x.wrapping_sub(y);
    if x < y { d.wrapping_add(m) } else { d }
}

#[inline(always)]
pub fn mul_mod(x: u64, y: u64, m: u64) -> u64 {
    ((x as u128 * y as u128) % m as u128) as u64
}

// montgomery operations adapted from the following sources
// https://en.algorithmica.org/hpc/number-theory/montgomery/
// https://github.com/cp-algorithms/cp-algorithms/blob/main/src/algebra/montgomery_multiplication.md
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct Mont {
    m: u64,   // modulus
    mr: u64,  // inverse of m mod 2^64
    r2: u64,  // r^2 mod m
    one: M64, // 1 in montgomery form
}

impl Mont {
    // create a new montgomery context with modulus m
    #[inline(always)]
    pub fn new(m: u64) -> Mont {
        // compute inverse of m mod 2^64
        // using the extended euclidean algorithm
        let mut mr = 1u64;
        for _ in 0..6 {
            mr = mr.wrapping_mul(2u64.wrapping_sub(mr.wrapping_mul(m)));
        }
        // compute r^2 mod m
        // r = 2^64, so r^2 = 2^128
        let r = (1u128 << 64) % (m as u128);
        let r2 = ((r * r) % (m as u128)) as u64;
        // compute 1 in montgomery form
        let one = M64(((1u128 << 64) % (m as u128)) as u64);
        // build the montgomery context
        Mont { m, mr, r2, one }
    }
    #[inline(always)]
    pub fn from_bigint(&self, x: Integer) -> M64 {
        let tmp = x.clone().abs() % self.modulus();
        let mut n = tmp.to_u64_wrapping();
        if x.is_negative() {
            n = self.modulus() - n; // Adjust for negative values
        }
        self.to_mont(n)
    }
    #[inline(always)]
    pub fn from_i128(&self, x: i128) -> M64 {
        let tmp = x.abs();
        let low = tmp as u64;
        let high = (tmp >> 64) as u64;
        let n = self.add(
            self.to_mont(low),
            self.mul(self.to_mont(high), M64(self.r2)),
        );
        if x < 0 { self.neg(n) } else { n }
    }
    // exclusively for testing
    #[inline(always)]
    pub fn literal(&self, x: u64) -> M64 {
        M64(x)
    }
    // get modulus
    #[inline(always)]
    pub fn modulus(&self) -> u64 {
        self.m
    }
    // convert to montgomery form
    #[inline(always)]
    pub fn to_mont(&self, x: u64) -> M64 {
        if x == 0 {
            self.zero()
        } else if x == 1 {
            self.one
        } else {
            self.reduce((x as u128) * (self.r2 as u128))
        }
    }
    // convert from montgomery form to normal form
    #[inline(always)]
    pub fn to_normal(&self, x: M64) -> u64 {
        // reduce to normal form
        self.reduce(x.0 as u128).0
    }
    // get zero in montgomery form
    #[inline(always)]
    pub fn zero(&self) -> M64 {
        M64(0)
    }
    // get one in montgomery form
    #[inline(always)]
    pub fn one(&self) -> M64 {
        self.one
    }
    // is 0 in montgomery form
    // this is the same as checking if x == 0 in normal form
    #[inline(always)]
    pub fn is_zero(&self, x: M64) -> bool {
        x == M64(0)
    }
    // check if x is 1 in montgomery form
    #[inline(always)]
    pub fn is_one(&self, x: M64) -> bool {
        x == self.one
    }
    // redc (reduce mod m and eliminate an extra montgomery factor)
    #[inline(always)]
    fn reduce(&self, x: u128) -> M64 {
        let q = (((x as u64) as u128) * (self.mr as u128)) as u64;
        let n = (q as u128) * (self.m as u128);
        let y = ((x.wrapping_sub(n)) >> 64) as u64;
        M64(if x < n { y.wrapping_add(self.m) } else { y })
    }
    // multiply in montgomery form
    #[inline(always)]
    pub fn mul(&self, x: M64, y: M64) -> M64 {
        if x == M64(0) || y == M64(0) {
            self.zero()
        } else if x == self.one {
            y
        } else if y == self.one {
            x
        } else {
            self.reduce(x.0 as u128 * y.0 as u128)
        }
    }
    // divide
    #[inline(always)]
    pub fn div(&self, x: M64, y: M64) -> M64 {
        if self.is_zero(y) {
            panic!("Attempt to divide by zero");
        }
        // x * y^-1
        let y_inv = self.inv(y);
        self.mul(x, y_inv)
    }
    // square in montgomery form
    #[inline(always)]
    pub fn sqr(&self, x: M64) -> M64 {
        let x = x.0 as u128;
        self.reduce(x * x)
    }
    // add
    #[inline(always)]
    pub fn add(&self, x: M64, y: M64) -> M64 {
        M64(add_mod(x.0, y.0, self.m))
    }
    // subtract
    #[inline(always)]
    pub fn sub(&self, x: M64, y: M64) -> M64 {
        M64(sub_mod(x.0, y.0, self.m))
    }
    // negate
    #[inline(always)]
    pub fn neg(&self, x: M64) -> M64 {
        M64(neg_mod(x.0, self.m))
    }
    // exponentiate
    #[inline(always)]
    pub fn exp(&self, x: M64, e: u64) -> M64 {
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
    pub fn inv(&self, x: M64) -> M64 {
        if self.is_zero(x) {
            panic!("Attempt to invert zero");
        }
        self.exp(x, self.m - 2)
    }
}
