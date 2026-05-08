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
                P::check_final_evals(&self.mont, &p, r.unwrap(), aux, &evals)
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
