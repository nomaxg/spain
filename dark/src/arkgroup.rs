use ark_ff::{
    fields::{MontBackend, MontConfig},
    BigInt, Field, Fp768,
};
use rug::{Complete, Integer};

// bit of a hack, though this isn't a finite field, we can still use ark's montgomery implementation for fast modular exponentiation

// 512 bit modulus
// #[modulus = "10941738641570527421809707322040357612003732945449205990913842131476349984288934784717997257891267332497625752899781833797076537244027146743531593354333897"]
// 768 bit modulus
#[derive(MontConfig)]
#[modulus = "1230186684530117755130494958384962720772853569595334792197322452151726400507263657518745202199786469389956474942774063845925192557326303453731548268507917026122142913461670429214311602221240479274737794080665351419597459856902143413"]
// #[modulus = "10941738641570527421809707322040357612003732945449205990913842131476349984288934784717997257891267332497625752899781833797076537244027146743531593354333897"]
#[generator = "2"]
pub struct FqConfig;
pub type ArkRsa512 = Fp768<MontBackend<FqConfig, 12>>;

pub fn ark_integer_pow(base: &ArkRsa512, exp: &Integer) -> ArkRsa512 {
    let mask = Integer::from(u64::MAX);
    let mut limbs = [0u64; 12];
    let mut e = exp.clone();
    for i in 0..12 {
        limbs[i] = (&e & &mask).complete().to_u64_wrapping();
        e >>= 64;
    }
    base.pow(BigInt::<12>::new(limbs))
}

// tests
#[cfg(test)]
mod tests {
    use ark_ff::FftField;

    use crate::rsagroup::RSAGroup;

    use super::*;
    #[test]
    fn test_ark_rsa512() {
        let mut a = ArkRsa512::GENERATOR;
        for _ in 0..1000 {
            a *= a;
        }
        let mut b = RSAGroup::generator();
        for _ in 0..1000 {
            b = b.clone() * b.clone();
        }
        // print a and b
        println!("ArkRsa512: {}", a);
        println!("RSAGroup: {}", b.value);
        assert_eq!(a.to_string(), b.value.to_string());
    }
}
