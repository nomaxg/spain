// 64 bit prime utilities

use crate::ops::mul_mod;
use rand::Rng;

// Determinized Miller-Rabin primality test for 64 bit integers
// SPRP base sets from: https://miller-rabin.appspot.com/

// modular exponentiation
fn mod_pow(b: u64, e: u64, m: u64) -> u64 {
    let mut r = 1;
    let mut b = b % m;
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(r, b, m);
        }
        e >>= 1;
        b = mul_mod(b, b, m);
    }
    r
}

// Miller-Rabin primality test
fn rabin(d: u64, s: u32, n: u64, a: u64) -> bool {
    let mut x = mod_pow(a, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    for _ in 0..s {
        x = mul_mod(x, x, n);
        if x == 1 {
            return false;
        }
        if x == n - 1 {
            return true;
        }
    }
    false
}

// run multiple tests with different bases
fn is_prime(n: u64) -> bool {
    // first check if n is a multiple of a few small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        if n.is_multiple_of(p) {
            return false;
        }
    }
    // if not, move on to the Miller-Rabin test
    let mut d = n - 1;
    let s = d.leading_zeros();
    d <<= s;
    rabin(d, s, n, 2)
        && rabin(d, s, n, 325)
        && rabin(d, s, n, 9375)
        && rabin(d, s, n, 28178)
        && rabin(d, s, n, 450775)
        && rabin(d, s, n, 9780504)
        && rabin(d, s, n, 1795265022)
}

// get random 64 bit prime with msb and lsb set
pub fn rand_prime<R: Rng>(rng: &mut R) -> u64 {
    loop {
        let n = rng.random::<u64>();
        if is_prime(n) {
            return n;
        }
    }
}

// get random element in the field
// use rejection sampling to make it fair
pub fn rand_elem<R: Rng>(p: u64, rng: &mut R) -> u64 {
    loop {
        let x = rng.random::<u64>();
        if x < p {
            return x;
        }
    }
}
