use rand::Rng;
use rug::Integer;

use crate::prover::{commit_chunked_pippenger, ChunkedComm, ProverState};
use crate::public::PublicParams;
use crate::verifier::{RoundChallenge, VerifierState};
use crate::{EvaluationResult, DARK};
use ff::ops_128::{Mont, M128};
use ff::poly::int::MLE as IntMLE;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct MockDARK {
    pub dark: DARK,
    pub mle: IntMLE,
}

impl MockDARK {
    pub fn new(num_vars: usize) -> Self {
        // According to Theorem 1 of the DARK paper, q must be sufficiently large:
        // i.e q > 2^O(mu*lambda)
        let q_bits = 30000;
        // let q_bits = 64;
        let num_chunks = 16;
        println!("Initializing DARK");
        let dark = DARK::new(q_bits, num_vars, num_chunks, 128);
        // create the polynomial
        println!("Creating mock polynomial");
        let start = Instant::now();
        let poly = IntMLE::from_buffer(
            (0..(1 << num_vars))
                .map(|_| {
                    // Random values based on gpt-2 witness magnitudes
                    Integer::from(rand::rng().random_range(0..999778222080i64))
                })
                .collect::<Vec<_>>(),
            num_vars,
        );
        let time = start.elapsed();
        println!("Generated random MLE in {:?}", time);
        MockDARK { dark, mle: poly }
    }
    pub fn simulate(&mut self) -> EvaluationResult {
        let mut rng = rand::rng();
        let num_vars = self.mle.num_vars();
        let eval_point_p = (0..num_vars)
            .map(|i| self.dark.public.small_mont.to_mont((i as u128) * 7 + 3))
            .collect::<Vec<_>>();
        let poly = self.mle.clone();
        let mut mont_poly = poly.reduce_to_mont(&self.dark.public.small_mont);
        let chunked_comm = commit_chunked_pippenger(poly.clone(), &self.dark.public);
        let y = mont_poly.eval(&eval_point_p, &self.dark.public.small_mont);
        self.dark
            .run_protocol(chunked_comm, poly, &eval_point_p, y, &mut rng)
    }
}

// Simple simulate method (more of a smoke test) with coupled prover/verifier logic
// for DARK evaluation protocol
#[allow(clippy::too_many_arguments)]
pub fn simulate_simple<R: Rng>(
    chunked_comm: ChunkedComm,
    int_poly: IntMLE,
    eval_point: &[M128],
    y: M128,
    prover: &mut ProverState,
    verifier: &mut VerifierState,
    public: &mut PublicParams,
    rng: &mut R,
) -> EvaluationResult {
    let mut eval_result = EvaluationResult::default();

    // Verifier first constructs constant commitments used to shift/fold commitment chunks
    // for verifier in the head rounds
    let verifier_start = Instant::now();
    verifier.compute_const_comms(public);
    eval_result.verifier_time += verifier_start.elapsed();

    // Initialize verifier and prover params
    prover.set_poly(int_poly);
    prover.set_eval_point(eval_point.to_vec());
    verifier.set_claim(y, eval_point.to_vec());
    verifier.set_commit(chunked_comm);

    // Now emulate the protocol based on section 4.3 pseudocode

    // Prover reduces poly to montgomery form
    let prover_start = Instant::now();
    prover.poly_reduce(public);
    eval_result.poly_reduce += prover_start.elapsed();

    let mut total_verifier_time = Duration::ZERO;
    let mut last_time = Instant::now();

    for _ in 0..public.total_rounds {
        // Account for some deallocation time for large prover polys
        eval_result.prover_dealloc_time += last_time.elapsed();
        prover.round += 1;

        // Sample alpha
        let verifier_sample_start = Instant::now();
        let RoundChallenge { alpha_int, alpha_p } = verifier.start_round(public, rng);
        total_verifier_time += verifier_sample_start.elapsed();

        // Split poly
        let prover_start = Instant::now();
        let (fl, fr, fl_int, fr_int) = prover.poly_split(public);
        eval_result.poly_split += prover_start.elapsed();

        // Determine if this is a verifier-in-head round
        let verifier_in_head_round = prover.round - 1 < public.num_verifier_in_head_rounds();

        // Prover computes Comm(fl) and Comm(fr)
        let (comm_fl, comm_fr) = if verifier_in_head_round {
            let verifier_start = std::time::Instant::now();
            let (cl, cr) = verifier.get_derived_comms(public);
            total_verifier_time += verifier_start.elapsed();
            (cl, cr)
        } else {
            let prover_start = Instant::now();
            let (cl, cr) = prover.poly_split_comm(&fl_int, &fr_int, public);
            eval_result.commit += prover_start.elapsed();
            (cl, cr)
        };

        // Prover computes y_l, y_r
        let prover_start = Instant::now();
        let (y_l, y_r) = prover.poly_split_eval(&fl, &fr, public);
        eval_result.poly_eval += prover_start.elapsed();

        // Verifier checks that y = y_l + X_2 * y_r
        let verifier_start = std::time::Instant::now();
        verifier.check_y_claim(&y_l, &y_r, public);
        total_verifier_time += verifier_start.elapsed();

        // Verifier checks commitment consistency
        if !verifier_in_head_round {
            let verifier_start = std::time::Instant::now();
            verifier.check_commitment_consistency(comm_fl, comm_fr, public);
            total_verifier_time += verifier_start.elapsed();
        }
        // Verifier folds commitments
        let verifier_start = std::time::Instant::now();
        verifier.update_commitment_and_claim(y_l, y_r, comm_fl, comm_fr, public);
        total_verifier_time += verifier_start.elapsed();

        // Prover computes new polynomial evaluations
        let prover_start = Instant::now();
        prover.update_polys(fl, fr, fl_int, fr_int, alpha_p, &alpha_int, public);
        eval_result.poly_fold += prover_start.elapsed();
        last_time = Instant::now();
    }

    // Verifier performs final check
    let verifier_start = std::time::Instant::now();
    let final_constant = prover.mont_poly.as_ref().unwrap().evals[0];
    let final_constant_int = prover.int_poly.as_ref().unwrap().evals[0].clone();
    verifier.final_check(&final_constant, &final_constant_int, public);

    // Final time accounting
    total_verifier_time += verifier_start.elapsed();
    eval_result.verifier_time += total_verifier_time;
    eval_result
}

// Simulate method (more of a smoke test) with decoupled prover/verifier logic
pub fn simulate<R: Rng>(
    chunked_comm: ChunkedComm,
    eval_point: &[M128],
    prover: &mut ProverState,
    verifier: &mut VerifierState,
    public: &mut PublicParams,
    rng: &mut R,
) {
    // Setup
    let y = prover.gen_y_claim(eval_point.to_vec(), public);
    verifier.set_claim(y, eval_point.to_vec());
    verifier.set_commit(chunked_comm);
    prover.poly_reduce(public);

    // Run DARK eval protocol
    for _ in 0..public.total_rounds {
        let challenge = verifier.start_round(public, rng);
        let round_claim = prover.respond_to_challenge(&challenge, public);
        verifier.verify_round(&round_claim, public);
    }
}

pub fn mock_poly(num_vars: usize) -> IntMLE {
    let total_points = (1usize).checked_shl(num_vars as u32).unwrap();
    let mut rng = rand::rng();
    let evals = (0..total_points)
        .map(|_| {
            let sample: i64 = rng.random_range(0..999_778_222_080i64);
            Integer::from(sample)
        })
        .collect::<Vec<_>>();
    IntMLE::from_buffer(evals, num_vars)
}

pub fn mock_eval_point(mont: &Mont, num_vars: usize) -> Vec<M128> {
    (0..num_vars)
        .map(|i| mont.to_mont((i as u128) * 7 + 3))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_dark(num_vars: usize, num_chunks: usize) {
        let mut rng = rand::rng();

        // Generate public params
        let q_bits = 30000;
        let precision = 128;
        let mut public = PublicParams::new(q_bits, num_vars, num_chunks, precision);

        // Create prover/verifier
        let mut prover = ProverState::default();
        let mut verifier = VerifierState::new(&public);

        // preprocessing
        verifier.compute_const_comms(&mut public);
        public.build_pippenger_bases();

        // Create mock poly/eval point
        let poly = mock_poly(num_vars);
        let eval_point = mock_eval_point(&public.small_mont, num_vars);

        // Prover commit
        let comm = prover.commit(poly.clone(), &public);

        // Run DARK
        simulate(
            comm,
            &eval_point,
            &mut prover,
            &mut verifier,
            &mut public,
            &mut rng,
        );
    }
    #[test]
    fn smoke_test() {
        test_dark(10, 16);
    }

    #[test]
    fn smoke_test_single_chunk() {
        test_dark(10, 1);
    }
}
