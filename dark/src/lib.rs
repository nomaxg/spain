#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
pub mod actor;
pub mod arith;
pub mod bigexp;
pub mod bigrsa;
pub mod prover;
pub mod public;
pub mod rsagroup;
pub mod test;
pub mod verifier;

use ff::ops_128::{Mont, M128};
use ff::poly::int::MLE as IntMLE;
use ff::prime_128::rand_elem;
use rand::Rng;
use rug::Complete;
use rug::Integer;
use std::time::Duration;

use crate::prover::{ChunkedComm, ProverState};
use crate::public::PublicParams;
use crate::test::simulate_simple;
use crate::verifier::VerifierState;

#[derive(Clone, Debug, Default)]
pub struct EvaluationResult {
    pub poly_fold: Duration,
    pub prover_dealloc_time: Duration,
    pub prover_exp_prepare_time: Duration,
    pub poly_split: Duration,
    pub poly_reduce: Duration,
    pub poly_eval: Duration,
    pub commit: Duration,
    pub verifier_time: Duration,
    pub prover_time: Duration,
}

impl EvaluationResult {
    pub fn calc_prover_total(&mut self) {
        self.prover_time = self.poly_reduce
            + self.poly_split
            + self.poly_eval
            + self.poly_fold
            + self.commit
            + self.prover_exp_prepare_time
            + self.prover_dealloc_time;
    }
}

// Samples random lambda bit alpha for the folding step of the DARK protocol (both int and
// montgomery forms)
pub fn sample_alpha<R: Rng>(mont: &Mont, rng: &mut R) -> (Integer, M128) {
    let alpha_u128 = rand_elem(mont.modulus(), rng);
    let alpha_int = Integer::from(alpha_u128);
    let alpha_p = mont.from_bigint(alpha_int.clone());
    (alpha_int, alpha_p)
}
// Returns the constant poly of w variables at the provided point
pub fn const_mle(value: &Integer, num_vars: usize) -> IntMLE {
    let evals = vec![value.clone(); 1 << num_vars];
    IntMLE::from_buffer(evals, num_vars)
}

pub fn get_mle_coefficients(poly: &IntMLE) -> Vec<u128> {
    (0..poly.num_vars())
        .map(|i| {
            poly.evals[i]
                .clone()
                .try_into()
                .expect("Integer does not fit into u128")
        })
        .collect()
}

fn mod_pow_u64(base: &Integer, exp: u64, modulus: &Integer) -> Integer {
    let mut result = Integer::from(1);
    let mut b = base.clone() % modulus;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * &b) % modulus;
        }
        b = (&b * &b).complete() % modulus;
        e >>= 1;
    }
    result
}

fn precompute_series_mod(q: &Integer, lambda: &Integer, max_k: usize) -> Vec<Integer> {
    let mut series = Vec::with_capacity(max_k);
    if max_k == 0 {
        return series;
    }
    let one = Integer::from(1);
    let mut s = one.clone();
    series.push(s.clone());
    let mut a = q.clone() % lambda;
    for _k in 1..max_k {
        let mut tmp = a.clone();
        tmp += &one;
        s = (s * &tmp) % lambda;
        series.push(s.clone());
        a = (&a * &a).complete() % lambda;
    }
    series
}

#[derive(Clone, Debug)]
pub struct DARK {
    pub public: PublicParams,
    pub verifier: VerifierState,
    pub prover: ProverState,
}

impl DARK {
    // Initialize DARK scheme with base q and num_vars for MLE
    pub fn new(q_bits: usize, num_vars: usize, num_chunks: usize, precision: u16) -> Self {
        let public = PublicParams::new(q_bits, num_vars, num_chunks, precision);
        let verifier = VerifierState::new(&public);
        let prover = ProverState::default();

        DARK {
            public,
            verifier,
            prover,
        }
    }
    // Simulate the full protocol as described in section 4.3 of the paper
    pub fn run_protocol<R: Rng>(
        &mut self,
        chunked_comm: ChunkedComm,
        int_poly: IntMLE,
        eval_point: &[M128],
        y: M128,
        rng: &mut R,
    ) -> EvaluationResult {
        let mut prover = self.prover.clone();
        let mut verifier = self.verifier.clone();
        simulate_simple(
            chunked_comm,
            int_poly,
            eval_point,
            y,
            &mut prover,
            &mut verifier,
            &mut self.public.clone(),
            rng,
        )
    }
}
