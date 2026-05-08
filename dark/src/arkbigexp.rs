// optimized exponentiation implementations
// given a vector of u64s, compute g^{sum_i v_i * 2^{64*i}}

use ark_ff::{FftField, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rug::Integer;

use crate::{
    arkgroup::{ark_integer_pow, ArkRsa512},
    rsagroup::LAMBDA_N,
};

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
pub fn verifier_generate_exp_helper(
    max_len: usize,
    splits: usize,
    folds: usize,
) -> Vec<Vec<ArkRsa512>> {
    let b = 64 * ceil_div(max_len, splits * folds);
    let two = Integer::from(2);
    let c = two.pow_mod(&Integer::from(b as u64), &*LAMBDA_N).unwrap();

    let mut helper = Vec::with_capacity(folds);
    let mut e = Integer::from(1);

    for _ in 0..folds {
        let mut row = Vec::with_capacity(splits);
        for _ in 0..splits {
            row.push(ark_integer_pow(&ArkRsa512::GENERATOR, &e));
            e *= &c;
            e %= &*LAMBDA_N;
        }
        helper.push(row);
    }
    helper
}

// split exponentiation
// let t_i = 2^{64 * i * len / splits}
// precompute all g^{t_i}
// then get a table of size 2^splits
// for each index in [0, 2^splits), store product of g^{t_i} where i-th bit of index is 1
// when computing exponentiation, index into table for contribution
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ArkSplitExp {
    max_len: usize,
    splits: usize,
    precomputed: Vec<ArkRsa512>,
}

impl ArkSplitExp {
    pub fn new(max_len: usize, splits: usize) -> Self {
        // let start = std::time::Instant::now();
        let mut helper = Vec::new();
        let mut g = ArkRsa512::GENERATOR;
        //let b = 64 * (max_len / splits);
        let b = 64 * ceil_div(max_len, splits);
        helper.push(g.clone());
        for _ in 1..splits {
            for _ in 0..b {
                //g *= g;
                g.square_in_place();
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
            let mut acc = ArkRsa512::ONE;
            for j in 0..splits {
                if (i >> j) & 1 == 1 {
                    acc *= helper[j];
                }
            }
            precomputed.push(acc);
        }
        // let elapsed = start.elapsed();
        // println!(" Setup 2 | {:?}", elapsed);
        ArkSplitExp {
            max_len,
            splits,
            precomputed,
        }
    }
    pub fn exp(&self, v: &[u64]) -> ArkRsa512 {
        // ensure precomputed is large enough
        assert!(v.len() <= self.max_len);
        let mut res = ArkRsa512::ONE;
        //let gap = self.max_len / self.splits;
        let gap = ceil_div(self.max_len, self.splits);
        // loop over gap size in reverse
        for i in (0..gap).rev() {
            for j in (0..64).rev() {
                // square
                //res *= res;
                res.square_in_place();
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
                    res *= self.precomputed[idx as usize];
                }
            }
        }
        res
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ArkFoldSplitExp {
    pub max_len: usize,
    splits: usize,
    folds: usize,
    precomputed: Vec<Vec<ArkRsa512>>,
}

impl ArkFoldSplitExp {
    pub fn new(max_len: usize, splits: usize, folds: usize) -> Self {
        let mut helper = Vec::new();
        let mut g = ArkRsa512::GENERATOR;
        //let b = 64 * (max_len / (splits * folds));
        let b = 64 * ceil_div(max_len, splits * folds);
        for i in 0..folds {
            let mut tmp = Vec::new();
            for j in 0..splits {
                tmp.push(g.clone());
                if i + 1 == folds && j + 1 == splits {
                    break;
                }
                for _ in 0..b {
                    g.square_in_place();
                }
            }
            helper.push(tmp);
        }
        Self::new_with_gen_helper(max_len, splits, folds, helper)
    }
    pub fn new_with_gen_helper(
        max_len: usize,
        splits: usize,
        folds: usize,
        helper: Vec<Vec<ArkRsa512>>,
    ) -> Self {
        let mut precomputed = Vec::new();
        for i in 0..folds {
            let mut fold_precomp = Vec::new();
            for j in 0..(1 << splits) {
                let mut acc = ArkRsa512::ONE;
                for k in 0..splits {
                    if (j >> k) & 1 == 1 {
                        acc *= helper[i][k].clone();
                    }
                }
                fold_precomp.push(acc);
            }
            precomputed.push(fold_precomp);
        }
        ArkFoldSplitExp {
            max_len,
            splits,
            folds,
            precomputed,
        }
    }
    pub fn exp(&self, v: &[u64]) -> ArkRsa512 {
        // ensure precomputed is large enough
        assert!(v.len() <= self.max_len);
        let mut res = ArkRsa512::ONE;
        //let gap = self.max_len / (self.splits * self.folds);
        let gap = ceil_div(self.max_len, self.splits * self.folds);
        // loop over gap size in reverse
        for i in (0..gap).rev() {
            for j in (0..64).rev() {
                // square
                res.square_in_place();
                // multiply in each fold
                for f in 0..self.folds {
                    // build index
                    let mut idx = 0;
                    for k in 0..self.splits {
                        let l = i + gap * (k + self.splits * f);
                        if l >= v.len() {
                            break;
                        }
                        idx |= ((v[l] >> j) & 1) << k;
                    }
                    if idx != 0 {
                        res *= self.precomputed[f][idx as usize];
                    }
                }
            }
        }
        res
    }
}

// check that two ArkRsa512 are equal
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_consistency() {
        let splits = 8;
        let folds = 2;
        let len = 1 << 16;
        let exp = ArkSplitExp::new(len, splits);
        let fold_exp = ArkFoldSplitExp::new(len, splits, folds);
        let v: Vec<u64> = (0..len as u64).collect();
        let res1 = exp.exp(&v);
        let res2 = fold_exp.exp(&v);
        assert_eq!(res1, res2);
    }
}
