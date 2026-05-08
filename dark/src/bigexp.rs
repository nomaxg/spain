// optimized exponentiation implementations
// given a vector of u64s, compute g^{sum_i v_i * 2^{64*i}}

use crate::rsagroup::RSAGroup;
use rug::Integer;

pub struct IntExp;

impl IntExp {
    pub fn new() -> Self {
        IntExp {}
    }
    pub fn exp(&self, v: &[u64]) -> RSAGroup {
        // build integer from limbs
        let mut v_int = Integer::from(*v.last().unwrap());
        for vi in v.iter().rev().skip(1) {
            v_int <<= 64;
            v_int += Integer::from(*vi);
        }
        // exponentiate g^{v_int}
        let g = RSAGroup::new(2);
        g.pow(&v_int)
    }
}

pub struct NaiveExp;

impl NaiveExp {
    pub fn new() -> Self {
        NaiveExp {}
    }
    pub fn exp(&self, v: &[u64]) -> RSAGroup {
        // start from highest bit of last v and work backwards
        let mut res = RSAGroup::new(1);
        let gen = RSAGroup::new(2);
        for vi in v.iter().rev() {
            for j in (0..64).rev() {
                res = res.clone() * res.clone();
                if (vi >> j) & 1 == 1 {
                    res = res * gen.clone();
                }
            }
        }
        res
    }
}

// inner bitlength assumed to be 16
pub struct CombExp {
    max_len: usize,             // maximum number of u64 limbs
    outer_len: usize,           // number of u64s per outer chunk
    precomputed: Vec<RSAGroup>, // vector of precomputed g^{2^(64 * i)} for i in [0, outer_size, 2*outer_size, ...]
}

impl CombExp {
    pub fn new(max_len: usize, outer_len: usize) -> Self {
        // precompute g^{2^(64 * i)} for i in [0, outer_size, 2*outer_size, ...]
        let outer_bitlength = 64 * outer_len;
        let mut precomputed = Vec::new();
        let mut acc = RSAGroup::new(2);
        let num_powers = max_len / outer_len;
        precomputed.push(acc.clone()); // g^1
        for _ in 1..num_powers {
            for _ in 0..outer_bitlength {
                acc = acc.clone() * acc.clone();
            }
            precomputed.push(acc.clone()); // g^{2^(64 * i)}
        }
        CombExp {
            max_len,
            outer_len,
            precomputed,
        }
    }
    // compute g^{sum_i v_i * 2^{64*i}} using comb method
    // that is to group windows of size inner_bitlength
    pub fn exp(&self, v: &[u64]) -> RSAGroup {
        // ensure precomputed is large enough
        assert!(v.len() <= self.max_len);
        // fill buckets
        // buckets: for each index in [0, 2^16), store vector of (RSAGroup, bool) of length outer_len
        // these hold products of precomputed values for each limb position
        let mut buckets = vec![vec![(RSAGroup::new(1), false); 4 * self.outer_len]; 1 << 16];
        for i in 0..v.len() / self.outer_len {
            for j in 0..self.outer_len {
                let limb = v[i * self.outer_len + j];
                for k in 0..4 {
                    let idx = ((limb >> (16 * k)) & 0xFFFF) as usize;
                    if idx != 0 {
                        let sel = 4 * j + k;
                        if !buckets[idx][sel].1 {
                            buckets[idx][sel].0 = self.precomputed[i].clone();
                            buckets[idx][sel].1 = true;
                        } else {
                            buckets[idx][sel].0 =
                                buckets[idx][sel].0.clone() * self.precomputed[i].clone();
                        }
                    }
                }
            }
        }
        // compress inner buckets
        for idx in 1..buckets.len() {
            let mut acc = RSAGroup::new(1);
            let mut first = true;
            for v in buckets[idx].iter().rev() {
                if !first {
                    for _ in 0..16 {
                        acc = acc.clone() * acc.clone();
                    }
                }
                if v.1 {
                    acc = acc.clone() * v.0.clone();
                    first = false;
                }
            }
            buckets[idx] = vec![(acc, !first)];
        }
        // accumulate outer buckets
        let mut res = RSAGroup::new(1);
        for idx in 1..buckets.len() {
            // res *= buckets[idx]^{idx}
            if buckets[idx][0].1 {
                let acc = buckets[idx][0].0.pow_u64(idx as u64);
                res = res * acc;
            }
        }
        res
    }
}

// split exponentiation
// let t_i = 2^{64 * i * len / splits}
// precompute all g^{t_i}
// then get a table of size 2^splits
// for each index in [0, 2^splits), store product of g^{t_i} where i-th bit of index is 1
// when computing exponentiation, index into table for contribution
pub struct SplitExp {
    max_len: usize,
    splits: usize,
    precomputed: Vec<RSAGroup>,
}

impl SplitExp {
    pub fn new(max_len: usize, splits: usize) -> Self {
        // let start = std::time::Instant::now();
        let mut helper = Vec::new();
        let mut g = RSAGroup::new(2);
        let b = 64 * ((max_len + splits - 1) / splits);
        // print b
        println!("SplitExp: b = {}", b);
        helper.push(g.clone());
        for _ in 1..splits {
            for _ in 0..b {
                g = g.clone() * g.clone();
            }
            helper.push(g.clone());
        }
        // let elapsed = start.elapsed();
        // println!(" Setup 1 | {:?}", elapsed);
        // print helper
        /*println!("SplitExp helper precomputed:");
        for (i, h) in helper.iter().enumerate() {
            println!("  helper[{}] = {}", i, h);
        }*/
        // let start = std::time::Instant::now();
        let mut precomputed = Vec::new();
        for i in 0..(1 << splits) {
            let mut acc = RSAGroup::new(1);
            for j in 0..splits {
                if (i >> j) & 1 == 1 {
                    acc = acc * helper[j].clone();
                }
            }
            precomputed.push(acc);
        }
        // let elapsed = start.elapsed();
        // println!(" Setup 2 | {:?}", elapsed);
        SplitExp {
            max_len,
            splits,
            precomputed,
        }
    }
    pub fn exp(&self, v: &[u64]) -> RSAGroup {
        // ensure precomputed is large enough
        assert!(v.len() <= self.max_len);
        let mut res = RSAGroup::new(1);
        let gap = (self.max_len + self.splits - 1) / self.splits;
        // loop over gap size in reverse
        for i in (0..gap).rev() {
            for j in (0..64).rev() {
                // square
                res = res.clone() * res.clone();
                // build index
                let mut idx = 0;
                for k in 0..self.splits {
                    let l = i + gap * k;
                    if l >= v.len() {
                        break;
                    }
                    idx |= ((v[l] >> j) & 1) << k;
                }
                if idx != 0 {
                    res = res * self.precomputed[idx as usize].clone();
                }
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // generate random u64 vector of given length
    fn rand_u64_vec(len: usize) -> Vec<u64> {
        let mut rng = rand::rng();
        (0..len).map(|_| rng.random::<u64>()).collect()
    }

    #[test]
    fn methods_equal() {
        let max_len = (1 << 10) - 23;
        let int_exp = IntExp::new();
        let naive_exp = NaiveExp::new();
        let comb_exp = CombExp::new(max_len, 4);
        let split_exp = SplitExp::new(max_len, 8);
        // get random vector
        let v = rand_u64_vec(max_len);
        // compute with int exp
        let res_int = int_exp.exp(&v);
        // compute naively
        let _res_naive = naive_exp.exp(&v);
        // compute with comb
        let _res_comb = comb_exp.exp(&v);
        // compute with split
        let res_split = split_exp.exp(&v);
        // check that lit matches all
        //assert_eq!(res_int.value, res_naive.value);
        //assert_eq!(res_int.value, res_comb.value);
        assert_eq!(res_int.value, res_split.value);
    }
}
