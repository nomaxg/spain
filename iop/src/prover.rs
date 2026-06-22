use std::time::Duration;

use crate::traits::SumCheckPoly;
use crate::verifier::VerifierState;

use ff::{FieldElem, FieldMont};

pub struct EvaluationResult {
    pub prover_time: Duration,
    pub verifier_time: Duration,
}

#[derive(Clone)]
pub struct ProverState<P: SumCheckPoly> {
    poly: P,
    mont: FieldMont,
}

impl<P: SumCheckPoly> ProverState<P> {
    pub fn new(poly: P, mont: FieldMont) -> Self {
        Self { poly, mont }
    }

    pub fn prove_round(&mut self, r: Option<FieldElem>) -> Vec<FieldElem> {
        assert!(
            self.poly.num_vars() > 1,
            "Tried to prove beyond the last variable"
        );
        // if r is Some, bind the outer variable
        if let Some(r) = r {
            self.poly.bind(r, &self.mont);
        }
        // get the evaluations of the polynomial as a univariate polynomial and return
        self.poly.as_poly(&self.mont)
    }

    pub fn final_evals(&mut self, r: FieldElem) -> Vec<FieldElem> {
        // assert number of variables is 1
        assert!(
            self.poly.num_vars() == 1,
            "Tried to get final evaluations when there are still variables left"
        );
        // bind final variable
        self.poly.bind(r, &self.mont);
        // return the final evaluations
        self.poly.final_evals()
    }

    pub fn num_vars(&self) -> usize {
        self.poly.num_vars()
    }

    pub fn last_round(&self) -> bool {
        self.poly.num_vars() == 1
    }

    // for benchmarking only! This simulates the protocol (and verifier checks) with local randomness
    // returns randomness and final evaluations
    pub fn simulate(
        &mut self,
        sum: FieldElem,
        aux: &[FieldElem],
        verbose: bool,
    ) -> (Vec<FieldElem>, Vec<FieldElem>, EvaluationResult) {
        // timing information
        let mut prover_time = Duration::ZERO;
        let mut verifier_time = Duration::ZERO;
        if verbose {
            println!("Simulating sum-check");
        }
        // create verifier
        let mut verifier =
            VerifierState::new(self.poly.num_vars(), self.poly.degree(), sum, self.mont);
        // create a vector to hold the random responses
        let mut challenges: Vec<FieldElem> = vec![];
        let mut r = None;
        // simulate rounds
        loop {
            // run prover
            let prover_start = std::time::Instant::now();
            let mut p = self.prove_round(r);
            prover_time += prover_start.elapsed();
            // run verifier
            let verifier_start = std::time::Instant::now();
            r = Some(
                verifier
                    .verify_round(&mut p, &mut rand::rng())
                    .expect("Verification failed"),
            );
            verifier_time += verifier_start.elapsed();
            // add to challenges
            challenges.push(r.unwrap());
            // exit if done after final check
            if self.poly.num_vars() == 1 {
                // check final evaluations
                let prover_start = std::time::Instant::now();
                let evals = self.final_evals(r.unwrap());
                prover_time += prover_start.elapsed();
                let verifier_start = std::time::Instant::now();
                let mut check_aux = Vec::with_capacity(aux.len() + challenges.len());
                check_aux.extend_from_slice(aux);
                check_aux.extend_from_slice(&challenges);
                P::check_final_evals(&self.mont, &p, r.unwrap(), &check_aux, &evals)
                    .expect("Final evaluations did not match");
                verifier_time += verifier_start.elapsed();
                return (
                    challenges,
                    evals,
                    EvaluationResult {
                        prover_time,
                        verifier_time,
                    },
                );
            }
        }
        // return the challenges and final evaluations
        //(challenges, self.final_evals(r.unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::ProverState;
    use ff::{FieldMont, outer_eq::OuterPolyEq, poly::cmont::MLE, prime_128::rand_prime};
    use rand::{SeedableRng, rngs::StdRng};
    use stream::bigvec::BigVec;

    #[test]
    fn simulate_outer_poly_eq() {
        let num_vars = 4usize;
        let size = 1usize << num_vars;
        let mut rng = StdRng::seed_from_u64(7);
        let mont = FieldMont::new(rand_prime(&mut rng));

        let az = MLE::from_buffer(BigVec::new(size).unwrap(), vec![0..size]);
        let bz = MLE::from_buffer(BigVec::new(size).unwrap(), vec![0..size]);
        let cz = MLE::from_buffer(BigVec::new(size).unwrap(), vec![0..size]);
        let tau = (0..num_vars)
            .map(|i| mont.to_mont((i as u128) + 2))
            .collect::<Vec<_>>();

        let poly = OuterPolyEq::from_buffers(az, bz, cz, &tau, &mont);
        let mut prover = ProverState::new(poly, mont);
        let (challenges, evals, _) = prover.simulate(mont.zero(), &tau, false);

        assert_eq!(challenges.len(), num_vars);
        assert_eq!(evals.len(), 4);
    }
}
