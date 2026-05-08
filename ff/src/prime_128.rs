use crate::ops_128::mul_mod;
use rand::Rng;

fn mod_pow(b: u128, e: u128, m: u128) -> u128 {
    let mut r = 1u128;
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

fn rabin(d: u128, s: u32, n: u128, a: u128) -> bool {
    let mut x = mod_pow(a % n, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    for _ in 1..s {
        x = mul_mod(x, x, n);
        if x == n - 1 {
            return true;
        }
        if x == 1 {
            return false;
        }
    }
    false
}

fn is_prime(n: u128) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    let mut d = n - 1;
    let s = d.trailing_zeros();
    d >>= s;
    for _ in 0..24 {
        let a = rand::random::<u128>() % (n - 3) + 2;
        if !rabin(d, s, n, a) {
            return false;
        }
    }
    true
}

pub fn rand_prime<R: Rng>(rng: &mut R) -> u128 {
    loop {
        let n = rng.random::<u128>() | (1u128 << 127) | 1;
        if is_prime(n) {
            return n;
        }
    }
}

pub fn rand_elem<R: Rng>(p: u128, rng: &mut R) -> u128 {
    loop {
        let x = rng.random::<u128>();
        if x < p {
            return x;
        }
    }
}
