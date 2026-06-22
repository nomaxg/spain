// Montgomery form large integer arithmetic
// Hand-rolled to support fixed-size big integers
// Without compile time moduli
// specialized to odd moduli with high bit 0
// based heavily on:
// https://github.com/arkworks-rs/algebra/blob/master/ff/src/fields/models/fp/montgomery_backend.rs

#![allow(dead_code)]

use crate::arith::*;
use ark_ff::{BigInt, BigInteger};
use rand_old::rngs::StdRng;
use rand_old::SeedableRng;
use rsa::traits::{PrivateKeyParts, PublicKeyParts};
use rsa::{BigUint, RsaPrivateKey};
use rug::Integer;
use serde::{de::SeqAccess, ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};

// zero-cost abstraction for u64 in montgomery form
// this just helps ensure correctness
#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub struct MontInt<const N: usize>(BigInt<N>);
pub type RSAGroup = MontInt<12>;

impl<const N: usize> Serialize for MontInt<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let limbs: &[u64; N] = &self.0 .0;
        let mut seq = serializer.serialize_seq(Some(N))?;
        for limb in limbs {
            seq.serialize_element(limb)?;
        }
        seq.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for MontInt<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MontIntVisitor<const N: usize>;

        impl<'de, const N: usize> serde::de::Visitor<'de> for MontIntVisitor<N> {
            type Value = MontInt<N>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "array of {} little-endian limbs", N)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut limbs = [0u64; N];
                for (i, limb) in limbs.iter_mut().enumerate() {
                    *limb = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(MontInt(BigInt::new(limbs)))
            }
        }

        deserializer.deserialize_seq(MontIntVisitor::<N>)
    }
}

// aux info for working in montgomery form
#[derive(Clone, Debug)]
pub struct Mont<const N: usize> {
    m: BigInt<N>,  // modulus
    r: BigInt<N>,  // 2^{64 N} mod m (which is 1 in montgomery form)
    r2: BigInt<N>, // r^2 mod m
    inv: u64,      // -m^{-1} mod 2^64
}

impl<const N: usize> Mont<N>
where
    [(); 2 * N]:,
{
    // check that modulus is well formed
    fn check_modulus(m_rug: &Integer) {
        // m > 0
        assert!(m_rug.cmp0() == std::cmp::Ordering::Greater);
        // m is exactly N limbs
        let m_limbs = m_rug.to_digits::<u64>(rug::integer::Order::Lsf);
        assert!(m_limbs.len() == N);
        // low bit of m is 1
        assert!(m_limbs[0] & 1 == 1);
        // high bit of m is 0
        assert!((m_limbs[N - 1] >> 63) & 1 == 0);
    }
    // convert from rug Integer to BigInt<N>
    // panic if x does not fit in N limbs
    fn rug_to_big(x: &Integer) -> BigInt<N> {
        let x_limbs = x.to_digits::<u64>(rug::integer::Order::Lsf);
        assert!(x_limbs.len() <= N);
        let mut x_limbs_fixed = [0u64; N];
        for (i, &limb) in x_limbs.iter().enumerate() {
            x_limbs_fixed[i] = limb;
        }
        BigInt::new(x_limbs_fixed)
    }
    // convert from biginteger to rug
    fn big_to_rug(x: &BigInt<N>) -> Integer {
        let mut res = Integer::from(0);
        for i in (0..N).rev() {
            res <<= 64;
            res += x.0[i];
        }
        res
    }
    fn inv_u64(m0: u64) -> u64 {
        // compute -m^{-1} mod 2^64 using extended euclidean algorithm
        // as -(m mod 2^64)^{-1} mod 2^64
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(inv.wrapping_mul(m0)));
        }
        inv = inv.wrapping_neg();
        inv
    }
    // slow constructor from rug Integer modulus
    pub fn new(m_rug: Integer) -> Self {
        Self::check_modulus(&m_rug);
        let m = Self::rug_to_big(&m_rug);
        let tmp = Integer::from(1) << (64 * N as u32);
        let r = Self::rug_to_big(&(tmp.clone() % &m_rug));
        let r2 = Self::rug_to_big(&((tmp.clone() * tmp) % &m_rug));
        let inv = Self::inv_u64(m_rug.to_digits::<u64>(rug::integer::Order::Lsf)[0]);
        Mont { m, r, r2, inv }
    }
    pub fn zero(&self) -> MontInt<N> {
        MontInt(BigInt::zero())
    }
    pub fn one(&self) -> MontInt<N> {
        MontInt(self.r)
    }
    // a = a + b
    // copied almost verbatim from arkworks
    pub fn add_assign(&self, a: &mut MontInt<N>, b: &MontInt<N>) {
        let _ = a.0.add_with_carry(&b.0);
        if a.0 >= self.m {
            a.0.sub_with_borrow(&self.m);
        }
    }
    // a = a - b
    // copied almost verbatim from arkworks
    pub fn sub_assign(&self, a: &mut MontInt<N>, b: &MontInt<N>) {
        if b.0 > a.0 {
            a.0.add_with_carry(&self.m);
        }
        a.0.sub_with_borrow(&b.0);
    }
    // a = -a
    // copied almost verbatim from arkworks
    pub fn neg_in_place(&self, a: &mut MontInt<N>) {
        if !a.0.is_zero() {
            let mut tmp = self.m;
            tmp.sub_with_borrow(&a.0);
            a.0 = tmp;
        }
    }
    // a = a * b
    // copied almost verbatim from arkworks
    pub fn mul_assign(&self, a: &mut MontInt<N>, b: &MontInt<N>) {
        let mut r = [0u64; N];
        for i in 0..N {
            let mut carry1 = 0u64;
            r[0] = mac(r[0], (a.0).0[0], (b.0).0[i], &mut carry1);

            let k = r[0].wrapping_mul(self.inv);

            let mut carry2 = 0u64;
            mac_discard(r[0], k, self.m.0[0], &mut carry2);

            for j in 1..N {
                r[j] = mac_with_carry(r[j], (a.0).0[j], (b.0).0[i], &mut carry1);
                r[j - 1] = mac_with_carry(r[j], k, self.m.0[j], &mut carry2);
            }
            r[N - 1] = carry1 + carry2;
        }
        (a.0).0.copy_from_slice(&r);
        if a.0 >= self.m {
            a.0.sub_with_borrow(&self.m);
        }
    }
    pub fn div_assign(&self, a: &mut MontInt<N>, b: &MontInt<N>) {
        let b_inv = self.inverse(&mut b.clone()).expect("inversion failed");
        self.mul_assign(a, &b_inv);
    }
    // copied almost verbatim from arkworks
    pub fn inverse(&self, a: &mut MontInt<N>) -> Option<MontInt<N>> {
        if a.0.is_zero() {
            return None;
        }
        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to
        // Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let one = BigInt::from(1u64);

        let mut u = a.0;
        let mut v = self.m;
        let mut b = MontInt(self.r2);
        let mut c = self.zero();

        while u != one && v != one {
            while u.is_even() {
                u.div2();

                if b.0.is_even() {
                    b.0.div2();
                } else {
                    let carry = b.0.add_with_carry(&self.m);
                    b.0.div2();
                    if carry {
                        (b.0).0[N - 1] |= 1 << 63;
                    }
                }
            }

            while v.is_even() {
                v.div2();

                if c.0.is_even() {
                    c.0.div2();
                } else {
                    let carry = c.0.add_with_carry(&self.m);
                    c.0.div2();
                    if carry {
                        (c.0).0[N - 1] |= 1 << 63;
                    }
                }
            }

            if v < u {
                u.sub_with_borrow(&v);
                self.sub_assign(&mut b, &c);
            } else {
                v.sub_with_borrow(&u);
                self.sub_assign(&mut c, &b);
            }
        }

        if u == one {
            Some(b)
        } else {
            Some(c)
        }
    }
    // a = a * a
    // copied almost verbatim from arkworks
    pub fn square_in_place(&self, a: &mut MontInt<N>) {
        let mut r = [0u64; { 2 * N }];

        let mut carry = 0;
        for i in 0..(N - 1) {
            for j in (i + 1)..N {
                r[i + j] = mac_with_carry(r[i + j], (a.0).0[i], (a.0).0[j], &mut carry);
            }
            r[i + N] = carry;
            carry = 0;
        }

        r[2 * N - 1] = r[2 * N - 2] >> 63;
        for i in 2..(2 * N - 1) {
            r[2 * N - i] = (r[2 * N - i] << 1) | (r[2 * N - (i + 1)] >> 63);
        }
        r[1] <<= 1;

        for i in 0..N {
            r[2 * i] = mac_with_carry(r[2 * i], (a.0).0[i], (a.0).0[i], &mut carry);
            carry = adc(&mut r[2 * i + 1], 0, carry);
        }
        // Montgomery reduction
        let mut carry2 = 0;
        for i in 0..N {
            let k = r[i].wrapping_mul(self.inv);
            carry = 0;
            mac_discard(r[i], k, self.m.0[0], &mut carry);
            for j in 1..N {
                r[j + i] = mac_with_carry(r[j + i], k, self.m.0[j], &mut carry);
            }
            carry2 = adc(&mut r[i + N], carry, carry2);
        }
        (a.0).0.copy_from_slice(&r[N..2 * N]);
        if a.0 >= self.m {
            a.0.sub_with_borrow(&self.m);
        }
    }
    // convert from Integer to montgomery form
    // assumed to be smaller than modulus
    // copied almost verbatim from arkworks
    pub fn to_montgomery(&self, a: &Integer) -> MontInt<N> {
        if a.is_zero() {
            MontInt(BigInt::zero())
        } else {
            let mut a_mont = MontInt(Self::rug_to_big(a));
            self.mul_assign(&mut a_mont, &MontInt(self.r2));
            a_mont
        }
    }
    // convert from montgomery form to normal form
    // copied almost verbatim from arkworks
    pub fn to_normal(&self, a: &MontInt<N>) -> Integer {
        let mut r = a.0;
        // Montgomery Reduction
        for i in 0..N {
            let k = r.0[i].wrapping_mul(self.inv);
            let mut carry = 0u64;

            mac_with_carry(r.0[i], k, self.m.0[0], &mut carry);
            for j in 1..N {
                r.0[(j + i) % N] = mac_with_carry(r.0[(j + i) % N], k, self.m.0[j], &mut carry);
            }
            r.0[i] = carry;
        }
        if r >= self.m {
            r.sub_with_borrow(&self.m);
        }
        // convert to rug Integer
        Self::big_to_rug(&r)
    }
    // slow big-exp via square-and-multiply
    pub fn exp(&self, base: &MontInt<N>, exp: &Integer) -> MontInt<N> {
        let mut res = self.one();
        let mut base_acc = *base;
        let mut exp_copy = exp.clone();
        while exp_copy > 0 {
            if exp_copy.is_odd() {
                self.mul_assign(&mut res, &base_acc);
            }
            self.square_in_place(&mut base_acc);
            exp_copy >>= 1;
        }
        res
    }
    // fast big-exp via Carmichael function of modulus
    // Recall that for prime p and q, car(pq) = lcm(p-1, q-1)
    pub fn fast_exp(&self, base: &MontInt<N>, exp: &Integer, car: &Integer) -> MontInt<N> {
        // first compute exp mod car
        let exp_mod = exp.clone() % car;
        // then use slow exp
        self.exp(base, &exp_mod)
    }
}

fn biguint_to_rug(x: &BigUint) -> Integer {
    let mut res = Integer::from(0);
    for limb in x.to_bytes_le().chunks(8).rev() {
        res <<= 64;
        let mut limb_arr = [0u8; 8];
        for (i, &b) in limb.iter().enumerate() {
            limb_arr[i] = b;
        }
        let limb_u64 = u64::from_le_bytes(limb_arr);
        res += limb_u64;
    }
    res
}

// generate RSA keypair
pub fn generate_rsa_group(bits: usize) -> ((Integer, Integer), Integer) {
    // deterministic rng so that verifier can precompute bases
    let mut rng = StdRng::seed_from_u64(0);
    let key = RsaPrivateKey::new(&mut rng, bits).expect("failed to generate a key");
    let p = biguint_to_rug(&key.primes()[0]);
    let q = biguint_to_rug(&key.primes()[1]);
    let m = biguint_to_rug(key.n());
    ((p, q), m)
}
pub fn precompute_bases<const N: usize>(
    mont: &Mont<N>,
    g: &MontInt<N>,
    q: &Integer,
    len: usize,
) -> Vec<MontInt<N>>
where
    [(); 2 * N]:,
{
    let mut table = Vec::with_capacity(len);
    table.push(*g);
    table.push(mont.exp(g, q));
    for i in 2..len {
        let last = &table[i - 1];
        let next = mont.exp(last, q);
        table.push(next);
    }
    table
}

// pippenger's algorithm for multi-exponentiation
pub fn pippenger_exp<const N: usize>(
    mont: &Mont<N>,
    bases: &[MontInt<N>],
    exponents: &[Integer],
) -> MontInt<N>
where
    [(); 2 * N]:,
{
    // length of bases must be at least as long as exponents
    assert!(bases.len() >= exponents.len());
    // determine max exponent bit length
    let mut max_len = 0;
    for exp in exponents {
        let bit_len = exp.significant_bits() as usize;
        if bit_len > max_len {
            max_len = bit_len;
        }
    }
    // window size
    let w = 8;
    let bucket_size = 1 << w;
    let num_windows = max_len.div_ceil(w);
    // initialize accumulator to 1
    let mut acc = mont.one();
    // process windows from high to low
    for i in (0..num_windows).rev() {
        // double acc w times
        for _ in 0..w {
            mont.square_in_place(&mut acc);
        }
        // initialize buckets
        let mut buckets = vec![None; bucket_size];
        // fill buckets
        for (base, exp) in bases.iter().zip(exponents.iter()) {
            let mut bit_chunk = 0usize;
            for j in 0..w {
                let bit_index = i * w + j;
                if bit_index < exp.significant_bits() as usize && exp.get_bit(bit_index as u32) {
                    bit_chunk |= 1 << j;
                }
            }
            if bit_chunk != 0 {
                if let Some(bucket_val) = &mut buckets[bit_chunk] {
                    mont.mul_assign(bucket_val, base);
                } else {
                    buckets[bit_chunk] = Some(*base);
                }
            }
        }
        // accumulate buckets
        let mut running = None;
        for j in (1..bucket_size).rev() {
            if let Some(bucket_val) = &buckets[j] {
                running = Some(match running {
                    None => *bucket_val,
                    Some(mut r) => {
                        mont.mul_assign(&mut r, bucket_val);
                        r
                    }
                });
            }
            if let Some(r) = &running {
                mont.mul_assign(&mut acc, r);
            }
        }
    }
    acc
}

impl Serialize for Mont<12> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let m_rug = Self::big_to_rug(&self.m);
        m_rug.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Mont<12> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let m_rug = Integer::deserialize(deserializer)?;
        Ok(Self::new(m_rug))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode;
    use rand::{prelude::StdRng, Rng, SeedableRng};
    use serde_json;

    fn setup<const N: usize>() -> (StdRng, Mont<N>, Integer)
    where
        [(); 2 * N]:,
    {
        // init seeded rng
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // set up modulus
        let mut m = Integer::from(0);
        for _ in 0..N {
            m <<= 64;
            let limb: u64 = rng.random();
            m += limb;
        }
        // ensure odd and high bit clear
        m.set_bit(0, true);
        m.set_bit(64 * N as u32 - 1, false);
        // create montgomery context
        (rng, Mont::new(m.clone()), m)
    }

    // biased rand, but its fine.
    fn rand_element<const N: usize, R: Rng>(rng: &mut R, m: &Integer) -> Integer {
        let mut res = Integer::from(0);
        for _ in 0..N {
            res <<= 64;
            res += rng.random::<u64>();
        }
        res %= m;
        res
    }

    #[test]
    fn test_conversion_consistency() {
        // set up modulus
        let (mut rng, mont, m) = setup::<4>();
        // test conversions
        for _ in 0..100 {
            let big = rand_element::<4, _>(&mut rng, &m);
            let mont_int = mont.to_montgomery(&big);
            let back = mont.to_normal(&mont_int);
            assert!(big == back, "Conversion mismatch: {:?} != {:?}", big, back);
        }
    }

    #[test]
    fn test_pippenger_consistency() {
        // set up modulus
        let (mut rng, mont, m) = setup::<4>();
        // test pippenger
        for _ in 0..10 {
            let num_bases = 50;
            let mut bases = Vec::with_capacity(num_bases);
            let mut exponents = Vec::with_capacity(num_bases);
            for _ in 0..num_bases {
                let base = rand_element::<4, _>(&mut rng, &m);
                let exp = Integer::from(rng.random::<u64>() % 10000);
                bases.push(mont.to_montgomery(&base));
                exponents.push(exp);
            }
            let pippenger_res = pippenger_exp(&mont, &bases, &exponents);
            // compute normal result
            let mut normal_res = Integer::from(1);
            for (base_mont, exp) in bases.iter().zip(exponents.iter()) {
                let base_normal = mont.to_normal(base_mont);
                let base_rug = base_normal % &m;
                let term = base_rug.pow_mod(exp, &m).unwrap();
                normal_res = (normal_res * term) % &m;
            }
            let back = mont.to_normal(&pippenger_res);
            assert!(
                normal_res == back,
                "Pippenger mismatch: {:?} != {:?}",
                normal_res,
                back
            );
        }
    }

    #[test]
    fn test_add_consistency() {
        // set up modulus
        let (mut rng, mont, m) = setup::<4>();
        // test arithmetic
        for _ in 0..100 {
            let a = rand_element::<4, _>(&mut rng, &m);
            let b = rand_element::<4, _>(&mut rng, &m);
            let mut mont_a = mont.to_montgomery(&a);
            let mont_b = mont.to_montgomery(&b);
            // addition
            mont.add_assign(&mut mont_a, &mont_b);
            let normal_res = (a + b) % &m;
            let back = mont.to_normal(&mont_a);
            assert!(
                normal_res == back,
                "Addition mismatch: {:?} != {:?}",
                normal_res,
                back
            );
        }
    }

    #[test]
    fn test_mul_consistency() {
        // set up modulus
        let (mut rng, mont, m) = setup::<4>();
        // test arithmetic
        for _ in 0..100 {
            let a = rand_element::<4, _>(&mut rng, &m);
            let b = rand_element::<4, _>(&mut rng, &m);
            let mut mont_a = mont.to_montgomery(&a);
            let mont_b = mont.to_montgomery(&b);
            // multiplication
            mont.mul_assign(&mut mont_a, &mont_b);
            let normal_res = (a * b) % &m;
            let back = mont.to_normal(&mont_a);
            assert!(
                normal_res == back,
                "Multiplication mismatch: {:?} != {:?}",
                normal_res,
                back
            );
        }
    }

    #[test]
    fn test_square_consistency() {
        // set up modulus
        let (mut rng, mont, m) = setup::<4>();
        // test arithmetic
        for _ in 0..100 {
            let a = rand_element::<4, _>(&mut rng, &m);
            let mut mont_a = mont.to_montgomery(&a);
            // squaring
            mont.square_in_place(&mut mont_a);
            let normal_res = (a.clone() * a) % &m;
            let back = mont.to_normal(&mont_a);
            assert!(
                normal_res == back,
                "Squaring mismatch: {:?} != {:?}",
                normal_res,
                back
            );
        }
    }

    #[test]
    fn test_inverse() {
        // prime modulus
        // special modulus setup, RSA challenge primes
        let p = Integer::from_str_radix(
            "6122421090493547576937037317561418841225758554253106999",
            10,
        )
        .unwrap();
        let q = Integer::from_str_radix(
            "5846418214406154678836553182979162384198610505601062333",
            10,
        )
        .unwrap();
        let m = p.clone() * q.clone();
        let mont = Mont::<6>::new(m.clone());
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for _ in 0..100 {
            let a = rand_element::<6, _>(&mut rng, &m);
            if a.is_zero() {
                continue;
            }
            let mut mont_a = mont.to_montgomery(&a);
            let mont_inv = mont.inverse(&mut mont_a).expect("inverse failed");
            // check that a * a^{-1} = 1
            let mut check = mont_a.clone();
            mont.mul_assign(&mut check, &mont_inv);
            let back = mont.to_normal(&check);
            assert!(
                back == Integer::from(1),
                "Inverse mismatch: {:?} != 1",
                back
            );
        }
    }

    // test fast exponentiation consistency
    #[test]
    fn test_fast_exp_consistency() {
        // special modulus setup, RSA challenge primes
        let p = Integer::from_str_radix(
            "6122421090493547576937037317561418841225758554253106999",
            10,
        )
        .unwrap();
        let q = Integer::from_str_radix(
            "5846418214406154678836553182979162384198610505601062333",
            10,
        )
        .unwrap();
        let m = p.clone() * q.clone();
        let pm1 = Integer::from(&p - 1);
        let qm1 = Integer::from(&q - 1);
        let car = Integer::lcm(pm1, &qm1);
        let mont = Mont::<6>::new(m.clone());
        // set up rng
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // test arithmetic
        for _ in 0..10 {
            let base = rand_element::<6, _>(&mut rng, &m);
            let mut exp = Integer::from(0);
            for _ in 0..1000 {
                exp <<= 64;
                exp += rng.random::<u64>();
            }
            let mont_base = mont.to_montgomery(&base);
            // exponentiation
            let mont_res = mont.fast_exp(&mont_base, &exp, &car);
            let normal_res = base.pow_mod(&exp, &m).unwrap();
            let back = mont.to_normal(&mont_res);
            assert!(
                normal_res == back,
                "Fast exponentiation mismatch: {:?} != {:?}",
                normal_res,
                back
            );
        }
    }

    #[test]
    fn test_rsa_keygen() {
        let ((p, q), m) = generate_rsa_group(767);
        // check that m = p * q
        let mut pq = Integer::from(1);
        let pm1 = Integer::from(&p - 1);
        let qm1 = Integer::from(&q - 1);
        let car = Integer::lcm(pm1, &qm1);
        dbg!(&car);
        pq *= p;
        pq *= q;
        dbg!(&pq);
        assert!(pq == m);
        // create montgomery context from m
        let _mont = Mont::<12>::new(m.clone());
    }

    #[test]
    fn test_montint_serde_roundtrip() {
        let limbs = [42u64, 7, 0, 1];
        let value = MontInt::<4>(BigInt::new(limbs));

        let bin = bincode::serialize(&value).expect("serialize");
        let json = serde_json::to_string(&value).expect("json serialize");

        let from_bin: MontInt<4> = bincode::deserialize(&bin).expect("deserialize");
        let from_json: MontInt<4> = serde_json::from_str(&json).expect("json deserialize");

        assert_eq!(value, from_bin);
        assert_eq!(value, from_json);
    }
}
