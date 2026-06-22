use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use crate::bigrsa::{generate_rsa_group, precompute_bases, Mont as BigMont};
use ff::{ops_128::Mont, prime_128::rand_prime, FieldMont};
use i256::{I256, I512, U256, U512};
use rug::Integer;
use serde::{Deserialize, Serialize};

use crate::bigrsa::RSAGroup;

pub static MOD_BITS: usize = 767;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicParams {
    pub pippenger_bases: Vec<RSAGroup>,
    pub cached_const_comms: Vec<RSAGroup>,
    pub initial_pad_comm: Option<RSAGroup>,
    pub rsa_gen: RSAGroup,
    pub small_mont: Mont,
    pub mont: BigMont<12>,
    pub q: Integer,
    pub q_bits: usize,
    pub num_chunks: usize,
    pub precision: u16,
    pub total_rounds: usize,
}

impl PublicParams {
    pub fn new(q_bits: usize, num_vars: usize, num_chunks: usize, precision: u16) -> Self {
        if precision != 64 && precision != 128 && precision != 256 && precision != 512 {
            panic!("Unsupported precision {precision}; expected 64, 128, 256, 512");
        }
        // Generate MOD_BITS bit RSA group
        let ((p_prime, q_prime), m) = generate_rsa_group(MOD_BITS);
        let pm1 = Integer::from(&p_prime - 1);
        let qm1 = Integer::from(&q_prime - 1);
        let mont = BigMont::<12>::new(m);
        let modulo = Integer::from(&p_prime * &q_prime);
        let rsa_gen = mont.to_montgomery(&Integer::from(2));
        let car = Integer::lcm(pm1, &qm1);
        let q_exp = (Integer::from(1u32) << q_bits).modulo(&car);
        let q = Integer::from(2).pow_mod(&q_exp, &modulo).unwrap();
        let small_mont = Mont::new(rand_prime(&mut rand::rng()));

        PublicParams {
            q_bits,
            num_chunks,
            precision,
            total_rounds: num_vars,
            q: q.clone(),
            rsa_gen,
            pippenger_bases: vec![],
            cached_const_comms: vec![],
            small_mont,
            mont: mont.clone(),
            initial_pad_comm: None,
        }
    }

    pub fn set_small_mont(&mut self, modulus: FieldMont) {
        self.small_mont = modulus
    }

    pub fn int_shift_base(&self) -> Integer {
        match self.precision {
            64 => Integer::from(i64::MAX),
            128 => Integer::from(i128::MAX),
            256 => Integer::from_str_radix(I256::MAX.to_string().as_str(), 10).unwrap(),
            512 => Integer::from_str_radix(I512::MAX.to_string().as_str(), 10).unwrap(),
            _ => unreachable!("precision validated in DARK::new"),
        }
    }
    pub fn uint_shift_base(&self) -> Integer {
        match self.precision {
            64 => Integer::from(u64::MAX),
            128 => Integer::from(u128::MAX),
            256 => Integer::from_str_radix(U256::MAX.to_string().as_str(), 10).unwrap(),
            512 => Integer::from_str_radix(U512::MAX.to_string().as_str(), 10).unwrap(),
            _ => unreachable!("precision validated in DARK::new"),
        }
    }

    pub fn chunk_size(&self) -> usize {
        let chunk_vars = self.total_rounds - self.num_chunks.trailing_zeros() as usize;
        1usize << chunk_vars
    }

    // Returns the number of verifier in the head rounds, log of the number of chunks
    pub fn num_verifier_in_head_rounds(&self) -> usize {
        self.num_chunks.trailing_zeros() as usize
    }

    // Precompute bases H_i = g^{q^i} for i in [0, 2^{num_vars}).
    // This uses `trapdoor_pow` for each exponent q^i.
    pub fn build_pippenger_bases(&mut self) {
        let len = self.chunk_size();
        let num_vars = len.trailing_zeros() as usize;
        let base_dir = PathBuf::from(if cfg!(feature = "nyuhpc") {
            "./"
        } else {
            env!("CARGO_MANIFEST_DIR")
        });
        let prefix = "dark_pip_base_";
        let candidates: Vec<(usize, PathBuf)> = match fs::read_dir(&base_dir) {
            Ok(entries) => entries
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    let name = path.file_name()?.to_str()?;
                    let name = name.strip_prefix(prefix)?;
                    let mut parts = name.split('_');
                    let num_vars_part = parts.next()?.parse::<usize>().ok()?;
                    let q_bits_part = parts.next()?.parse::<usize>().ok()?;
                    if q_bits_part == self.q_bits {
                        Some((num_vars_part, path))
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => Vec::new(),
        };

        let chosen: Option<(usize, PathBuf)> = candidates
            .iter()
            .filter(|(n, _)| *n >= num_vars)
            .min_by_key(|(n, _)| *n)
            .cloned()
            .or_else(|| candidates.iter().max_by_key(|(n, _)| *n).cloned());

        let target_cache_path = base_dir.join(format!("{prefix}{num_vars}_{}", self.q_bits));

        let bases: Vec<RSAGroup> = match chosen {
            Some((stored_vars, path)) => {
                eprintln!(
                    "Using cached pippenger bases (num_vars {}, q_bits {}) from {:?}",
                    stored_vars, self.q_bits, path
                );
                let mut loaded: Vec<RSAGroup> = bincode::deserialize(
                    &fs::read(&path).expect("failed to read cached pippenger bases"),
                )
                .expect("failed to deserialize cached pippenger bases");
                let mut extended = false;
                if loaded.len() < len {
                    eprintln!(
                        "Extending cached pippenger bases from length {} to {}",
                        loaded.len(),
                        len
                    );
                    while loaded.len() < len {
                        let last = *loaded.last().expect("cached pippenger bases are empty");
                        let next = self.mont.exp(&last, &self.q);
                        loaded.push(next);
                    }
                    extended = true;
                }
                if extended {
                    let serialized =
                        bincode::serialize(&loaded).expect("failed to serialize pippenger bases");
                    fs::write(&target_cache_path, serialized)
                        .expect("failed to persist pippenger bases");
                    let _ = fs::remove_file(&path);
                }
                loaded
            }
            None => {
                eprintln!(
                    "Precomputing pippenger bases of length 2^{} (q_bits {})",
                    num_vars, self.q_bits
                );
                let computed = precompute_bases(&self.mont, &self.rsa_gen, &self.q, len);
                let serialized =
                    bincode::serialize(&computed).expect("failed to serialize pippenger bases");
                eprintln!("{:?}", &target_cache_path);
                io::stderr().flush().unwrap();
                fs::write(&target_cache_path, serialized)
                    .expect("failed to persist pippenger bases");
                computed
            }
        };

        self.pippenger_bases = bases.clone();
    }

    pub fn build_pippenger_bench(&self) {
        let len = self.chunk_size();
        let num_vars = len.trailing_zeros() as usize;
        println!(
            "Precomputing pippenger bases of length 2^{} (q_bits {})",
            num_vars, self.q_bits
        );
        let _computed = precompute_bases(&self.mont, &self.rsa_gen, &self.q, len);
    }
}
