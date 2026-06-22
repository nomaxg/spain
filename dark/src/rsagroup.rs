use once_cell::sync::Lazy;
use rug::{Complete, Integer};
use std::fmt;
use std::ops::{Div, Mul};

// Basic unoptimized RSAGroup implementation.
// https://github.com/bbuenz/dark_prototype/blob/master/rsagroup.py used as reference.

// 768‐bit RSA modulus N
static N: Lazy<Integer> = Lazy::new(|| {
    Integer::from_str_radix(
        "\
        1230186684530117755130494958384962720772853569595334792197322452151726400507263657518745202199786469389956474942774063845925192557326303453731548268507917026122142913461670429214311602221240479274737794080665351419597459856902143413
        ",
        10,
    )
    .unwrap()
});

// lambda(N) = lcm(p-1, q-1)
pub static LAMBDA_N: Lazy<Integer> = Lazy::new(|| {
    Integer::from_str_radix(
        "\
        307546671132529438782623739596240680193213392398833698049330613037931600126815914379686300549946617347489118735693498405452456700209272291231975106966116760542248714151364710187659700899445022498304201487967355716559907310924458752",
        10,
    )
    .unwrap()
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RSAGroup {
    pub value: Integer,
}

impl RSAGroup {
    pub fn new<V: Into<Integer>>(v: V) -> Self {
        let mut x = v.into();
        x %= &*N;
        RSAGroup { value: x }
    }

    pub fn generator() -> Self {
        RSAGroup::new(2)
    }

    pub fn pow(&self, exp: &Integer) -> RSAGroup {
        let r = self
            .value
            .clone()
            .pow_mod(exp, &N)
            .expect("modular exponentiation failed");
        RSAGroup { value: r }
    }

    pub fn trapdoor_pow(&self, exp: &Integer) -> RSAGroup {
        let exp = exp.clone().modulo(&LAMBDA_N);
        let r = self
            .value
            .clone()
            .pow_mod(&exp, &N)
            .expect("modular exponentiation failed");
        RSAGroup { value: r }
    }

    pub fn pow_u64(&self, exp: u64) -> RSAGroup {
        self.pow(&Integer::from(exp))
    }

    pub fn mul_mut(&mut self, other: &RSAGroup) {
        let value = (&self.value * &other.value).complete() % &*N;
        self.value = value;
    }

    pub fn inv(&self) -> RSAGroup {
        self.value
            .clone()
            .invert(&N)
            .map(|inv| RSAGroup { value: inv })
            .expect("inversion failed, this is unlikely")
    }
}

impl Mul for RSAGroup {
    type Output = RSAGroup;
    fn mul(self, rhs: RSAGroup) -> RSAGroup {
        let value = (&self.value * &rhs.value).complete() % &*N;
        RSAGroup::new(value)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for RSAGroup {
    type Output = RSAGroup;
    fn div(self, rhs: RSAGroup) -> RSAGroup {
        let inv_b = rhs.inv();
        RSAGroup::new(self.value * inv_b.value)
    }
}

impl fmt::Display for RSAGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} mod {}", self.value, *N)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rug::ops::Pow;

    use super::*;

    #[test]
    fn smoke() {
        let g = RSAGroup::generator();
        let a = RSAGroup::new(42);
        let b = RSAGroup::new(17);
        let ab = a.clone() * b.clone();
        assert_eq!(ab, RSAGroup::new(42 * 17));
        assert_eq!(g.pow(&5u32.into()), RSAGroup::new(32));
        let c = a.clone() / b.clone();
        assert_eq!(c * b, a);
    }

    #[test]
    fn pow_trapdoor() {
        let g = RSAGroup::generator();
        let exp = Integer::from(12345678241241249u64).pow(50);
        let naive_start = Instant::now();
        let y1 = g.pow(&exp);
        println!("Naive pow time: {:?}", naive_start.elapsed());
        let trapdoor_start = Instant::now();
        let y2 = g.trapdoor_pow(&exp);
        println!("Trapdoor pow time: {:?}", trapdoor_start.elapsed());
        assert_eq!(y1, y2);
    }
}
