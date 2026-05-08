use ff::FieldElem;
use protocol::machine::ProtocolState;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::prover::ProverState;
use crate::traits::SumCheckPoly;
use crate::verifier::VerifierState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SumcheckMessage {
    RoundPoly(Vec<FieldElem>),
    Challenge(FieldElem),
    FinalClaim {
        claim: FieldElem,
        evals: Vec<FieldElem>,
    },
}

pub struct SumcheckProver<P: SumCheckPoly> {
    state: ProverState<P>,
    started: bool,
    finished: bool,
}

impl<P: SumCheckPoly> SumcheckProver<P> {
    pub fn new(state: ProverState<P>) -> Self {
        Self {
            state,
            started: false,
            finished: false,
        }
    }
}

impl<P: SumCheckPoly> ProtocolState<SumcheckMessage> for SumcheckProver<P> {
    fn init_message(&mut self) -> anyhow::Result<SumcheckMessage> {
        if self.started {
            return Err(anyhow::anyhow!("sumcheck prover already started"));
        }
        self.started = true;
        Ok(SumcheckMessage::RoundPoly(self.state.prove_round(None)))
    }

    fn handle_message(&mut self, m: &SumcheckMessage) -> Option<SumcheckMessage> {
        if self.finished {
            return None;
        }
        match m {
            SumcheckMessage::Challenge(r) => {
                if self.state.num_vars() == 1 {
                    let evals = self.state.final_evals(*r);
                    let Some(final_claim) = evals.first().copied() else {
                        panic!("prover produced empty final evaluation vector");
                    };
                    self.finished = true;
                    Some(SumcheckMessage::FinalClaim {
                        claim: final_claim,
                        evals,
                    })
                } else {
                    Some(SumcheckMessage::RoundPoly(self.state.prove_round(Some(*r))))
                }
            }
            SumcheckMessage::RoundPoly(_) | SumcheckMessage::FinalClaim { .. } => {
                panic!("prover received unexpected message type")
            }
        }
    }
}

pub struct SumcheckVerifier<P: SumCheckPoly> {
    state: VerifierState,
    aux: Vec<FieldElem>,
    rng: ThreadRng,
    last_round_poly: Option<Vec<FieldElem>>,
    last_challenge: Option<FieldElem>,
    finished: bool,
    _marker: PhantomData<P>,
}

impl<P: SumCheckPoly> SumcheckVerifier<P> {
    pub fn new(state: VerifierState, aux: Vec<FieldElem>) -> Self {
        Self {
            state,
            aux,
            rng: rand::rng(),
            last_round_poly: None,
            last_challenge: None,
            finished: false,
            _marker: PhantomData,
        }
    }
}

impl<P: SumCheckPoly> ProtocolState<SumcheckMessage> for SumcheckVerifier<P> {
    fn init_message(&mut self) -> anyhow::Result<SumcheckMessage> {
        Err(anyhow::anyhow!("sumcheck verifier does not initiate"))
    }

    fn handle_message(&mut self, m: &SumcheckMessage) -> Option<SumcheckMessage> {
        if self.finished {
            return None;
        }
        match m {
            SumcheckMessage::RoundPoly(poly) => {
                let mut p = poly.clone();
                match self.state.verify_round(&mut p, &mut self.rng) {
                    Ok(challenge) => {
                        self.last_round_poly = Some(p);
                        self.last_challenge = Some(challenge);
                        Some(SumcheckMessage::Challenge(challenge))
                    }
                    Err(e) => panic!("verifier round check failed: {e}"),
                }
            }
            SumcheckMessage::FinalClaim { claim: _, evals } => {
                let p = self
                    .last_round_poly
                    .as_ref()
                    .expect("final round poly missing");
                let r = self.last_challenge.expect("missing final challenge");
                match P::check_final_evals(&self.state.mont(), p, r, &self.aux, evals) {
                    Ok(()) => {
                        self.finished = true;
                        None
                    }
                    Err(e) => panic!("verifier final check failed: {e}"),
                }
            }
            SumcheckMessage::Challenge(_) => panic!("verifier received unexpected message type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use ff::poly::mont::{MLE, lagrange_interpolate};
    use ff::{FieldElem, FieldMont, prime_128::rand_prime};
    use protocol::machine::simulate;
    use rand::Rng;

    use crate::actor::{SumcheckProver, SumcheckVerifier};
    use crate::prover::ProverState;
    use crate::traits::SumCheckPoly;
    use crate::verifier::VerifierState;

    #[derive(Clone)]
    struct DummyPoly {
        mle: MLE,
    }

    impl DummyPoly {
        fn new(mle: MLE) -> Self {
            Self { mle }
        }
    }

    impl SumCheckPoly for DummyPoly {
        fn degree(&self) -> usize {
            1
        }

        fn num_vars(&self) -> usize {
            self.mle.num_vars()
        }

        fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
            let mut p_1 = mont.zero();
            for i in 0..(self.mle.evals.len() / 2) {
                p_1 = mont.add(p_1, self.mle.evals[2 * i + 1]);
            }
            vec![p_1]
        }

        fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
            self.mle.bind(x, mont);
        }

        fn final_evals(&self) -> Vec<FieldElem> {
            vec![self.mle.evals[0]]
        }

        fn check_final_evals(
            mont: &FieldMont,
            p: &[FieldElem],
            r: FieldElem,
            _aux: &[FieldElem],
            evals: &[FieldElem],
        ) -> Result<(), String> {
            let actual = evals[0];
            let expected = lagrange_interpolate(p, r, mont);
            if actual == expected {
                Ok(())
            } else {
                Err(format!(
                    "Final evaluations did not match: expected {:?}, got {:?}",
                    expected, actual
                ))
            }
        }
    }

    #[test]
    fn simulate_sumcheck() {
        let mont = FieldMont::new(rand_prime(&mut rand::rng()));
        let num_vars = 4;
        let degree = 1;
        let mut rng = rand::rng();
        let mut mle = MLE::new(num_vars);
        for i in 0..mle.evals.len() {
            mle.evals[i] = mont.to_mont(rng.random::<u128>() % mont.modulus());
        }
        // Compute the real sum-check claim
        let mut initial_sum = mont.zero();
        for i in 0..mle.evals.len() {
            initial_sum = mont.add(initial_sum, mle.evals[i]);
        }

        // Generate a fake point and ensure MLE evaluation behaves as expected
        let fake_eval_point: Vec<FieldElem> = (0..num_vars)
            .map(|_| mont.to_mont(rng.random::<u128>() % mont.modulus()))
            .collect();
        let mut mle_for_eval = mle.clone();
        let eval_at_fake_point = mle_for_eval.eval(&fake_eval_point, &mont);
        assert!(eval_at_fake_point == mle_for_eval.evals[0]);

        let poly = DummyPoly::new(mle);
        let prover_state = ProverState::new(poly, mont);
        let verifier_state = VerifierState::new(num_vars, degree, initial_sum, mont);

        let prover = SumcheckProver::new(prover_state);
        let verifier = SumcheckVerifier::<DummyPoly>::new(verifier_state, vec![]);

        simulate(prover, verifier).expect("sumcheck actor simulation failed");
    }
}
