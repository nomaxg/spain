use std::{fmt::Debug, str::FromStr};

use anyhow::{Result, anyhow};
use dark::{
    prover::{ChunkedComm, RoundClaim},
    public::PublicParams,
    verifier::RoundChallenge,
};
use ff::{FieldElem, FieldMont};
use model::HighPrecision;
use parse::{generalized::HighPrecisionInt, mat::Matrix};
use protocol::machine::ProtocolState;
use serde::{Deserialize, Serialize};

use crate::{
    EvaluationResult,
    prover::ProverState,
    timer::{ProverPhase as ProverTimerPhase, Timer, VerifierPhase as VerifierTimerPhase},
    traits::{MatrixIntOps, R1CSInstance, ToI512},
    verifier::{VerifierState, ZRangeOpening},
};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: ToString",
    deserialize = "T: FromStr, <T as FromStr>::Err: std::fmt::Debug"
))]
pub enum SpainMessage<T = i128> {
    Setup(PublicParams),
    ErrorClaim(String),
    RequestCommitment,
    Commitment(ChunkedComm),
    SampleMont {
        small_modulus: u128,
    },
    Randomness(Vec<u64>),
    OuterRoundClaim(Vec<FieldElem>),
    OuterChallenge(FieldElem),
    OuterFinalEvals(Vec<FieldElem>),
    StartInner {
        r1: FieldElem,
        r2: FieldElem,
    },
    InnerStart {
        num_vars: usize,
        claim: Vec<FieldElem>,
    },
    InnerRoundClaim(Vec<FieldElem>),
    InnerChallenge(FieldElem),
    InnerFinalEvals(Vec<FieldElem>),
    DarkEvalPoint(Vec<FieldElem>),
    DarkClaim(FieldElem),
    DarkChallenge(RoundChallenge),
    DarkRoundResponse(RoundClaim),
    RequestWitnessOpenings,
    WitnessOpenings(Vec<ZRangeOpening<T>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProverPhase {
    WaitingSetup,
    WaitingRandomness,
    WaitingMont,
    Outer,
    WaitingInnerStart,
    Inner,
    DarkClaim,
    DarkRounds,
    Done,
}

pub struct SpainProver<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> where
    Matrix<T>: MatrixIntOps,
{
    state: ProverState<T, P, E>,
    phase: ProverPhase,
    timer: Timer,
}

impl<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> SpainProver<T, P, E>
where
    Matrix<T>: MatrixIntOps,
{
    pub fn new(state: ProverState<T, P, E>) -> Self {
        let batch_size = state.batch_size;
        Self {
            state,
            phase: ProverPhase::WaitingSetup,
            timer: Timer::from_eval_result(EvaluationResult {
                batch_size,
                ..Default::default()
            }),
        }
    }

    fn initialize(&mut self, public_params: PublicParams) -> SpainMessage<T> {
        self.timer.prover(ProverTimerPhase::ComputeWitness, || {
            self.state.compute_commit_witness()
        });
        self.timer.prover(ProverTimerPhase::Preprocessing, || {
            self.state.set_dark_public_params(public_params)
        });
        self.phase = ProverPhase::WaitingRandomness;
        let comm = self
            .timer
            .prover(ProverTimerPhase::PolyCommit, || self.state.commit());
        SpainMessage::Commitment(comm)
    }

    pub fn print_eval(&mut self) {
        let result = self.timer.finish();
        eprintln!("Eval report");
        eprintln!("sys: spain_decoupled_prover");
        eprintln!("{result:#?}");
        eprintln!("End eval");
    }

    pub fn get_eval_result(&mut self) -> EvaluationResult {
        self.timer.finish()
    }

    pub fn num_constraints(&self) -> usize {
        self.state.num_constraints()
    }

    pub fn set_eval_model_name(&mut self, model_name: impl Into<String>) {
        self.timer.eval_result_mut().model_name = model_name.into();
    }

    pub fn is_done(&self) -> bool {
        self.phase == ProverPhase::Done
    }
}

impl<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> ProtocolState<SpainMessage<T>> for SpainProver<T, P, E>
where
    Matrix<T>: MatrixIntOps,
{
    fn init_message(&mut self) -> Result<SpainMessage<T>> {
        Err(anyhow!("spain prover does not initiate"))
    }

    fn handle_message(&mut self, m: &SpainMessage<T>) -> Option<SpainMessage<T>> {
        Some(match (self.phase, m) {
            (ProverPhase::WaitingSetup, SpainMessage::Setup(public_params)) => {
                self.initialize(public_params.clone())
            }
            (ProverPhase::WaitingRandomness, SpainMessage::Randomness(randomness)) => {
                self.phase = ProverPhase::WaitingMont;
                self.timer.prover(ProverTimerPhase::Misc, || {
                    self.state.set_randomness(randomness.clone());
                    self.state.inject_randomness();
                });
                self.timer.prover(ProverTimerPhase::ComputeWitness, || {
                    self.state.compute_full_witness();
                });
                let squared_error = self
                    .timer
                    .prover(ProverTimerPhase::ComputeSquaredError, || {
                        self.state.compute_squared_error()
                    });
                SpainMessage::ErrorClaim(squared_error.to_string())
            }
            (ProverPhase::WaitingMont, SpainMessage::SampleMont { small_modulus }) => {
                self.timer.prover(ProverTimerPhase::PrepareOuterSc, || {
                    self.state.set_mont(FieldMont::new(*small_modulus));
                    self.state.convert_instance_to_mont();
                    self.state.prepare_outer_sc();
                });
                self.phase = ProverPhase::Outer;
                let claim = self
                    .timer
                    .prover(ProverTimerPhase::RunOuterSc, || self.state.outer_sc_claim());
                SpainMessage::OuterRoundClaim(claim)
            }
            (ProverPhase::Outer, SpainMessage::OuterChallenge(r)) => {
                if self.state.outer_last_round() {
                    self.phase = ProverPhase::WaitingInnerStart;
                    let evals = self.timer.prover(ProverTimerPhase::RunOuterSc, || {
                        self.state.outer_final_evals(*r)
                    });
                    SpainMessage::OuterFinalEvals(evals)
                } else {
                    let claim = self.timer.prover(ProverTimerPhase::RunOuterSc, || {
                        self.state.outer_sc_prove(Some(*r))
                    });
                    SpainMessage::OuterRoundClaim(claim)
                }
            }
            (ProverPhase::WaitingInnerStart, SpainMessage::StartInner { r1, r2 }) => {
                let num_vars = self.timer.prover(ProverTimerPhase::PrepareInnerSc, || {
                    self.state.prepare_inner_sc(*r1, *r2)
                });
                self.phase = ProverPhase::Inner;
                let claim = self
                    .timer
                    .prover(ProverTimerPhase::RunInnerSc, || self.state.inner_sc_claim());
                SpainMessage::InnerStart { num_vars, claim }
            }
            (ProverPhase::Inner, SpainMessage::InnerChallenge(r)) => {
                if self.state.inner_last_round() {
                    self.phase = ProverPhase::DarkClaim;
                    let evals = self.timer.prover(ProverTimerPhase::RunInnerSc, || {
                        self.state.inner_final_evals(*r)
                    });
                    SpainMessage::InnerFinalEvals(evals)
                } else {
                    let claim = self.timer.prover(ProverTimerPhase::RunInnerSc, || {
                        self.state.inner_sc_prove(Some(*r))
                    });
                    SpainMessage::InnerRoundClaim(claim)
                }
            }
            (ProverPhase::DarkClaim, SpainMessage::DarkEvalPoint(eval_point)) => {
                self.phase = ProverPhase::DarkRounds;
                let claim = self.timer.prover(ProverTimerPhase::PolyEval, || {
                    self.state.dark_mle_eval(eval_point)
                });
                SpainMessage::DarkClaim(claim)
            }
            (ProverPhase::DarkRounds, SpainMessage::DarkChallenge(challenge)) => {
                let response = self.timer.prover(ProverTimerPhase::PolyEval, || {
                    self.state.dark_respond(challenge.clone())
                });
                SpainMessage::DarkRoundResponse(response)
            }
            (ProverPhase::DarkRounds, SpainMessage::RequestWitnessOpenings) => {
                self.phase = ProverPhase::Done;
                let openings = self
                    .timer
                    .prover(ProverTimerPhase::Misc, || self.state.witness_openings());
                SpainMessage::WitnessOpenings(openings)
            }
            _ => panic!(
                "prover received unexpected message {m:?} in phase {:?}",
                self.phase
            ),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierPhase {
    BeforeSetup,
    WaitingErrorClaim,
    WaitingCommitment,
    Outer,
    WaitingInnerStart,
    Inner,
    WaitingDarkClaim,
    DarkRounds,
    WaitingWitnessOpenings,
    Done,
}

pub struct SpainVerifier<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> where
    Matrix<T>: MatrixIntOps,
{
    state: VerifierState<T, P, E>,
    phase: VerifierPhase,
    last_outer_round_poly: Option<Vec<FieldElem>>,
    last_inner_round_poly: Option<Vec<FieldElem>>,
    timer: Timer,
}

impl<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> SpainVerifier<T, P, E>
where
    Matrix<T>: MatrixIntOps,
{
    pub fn new(state: VerifierState<T, P, E>) -> Self {
        let batch_size = state.batch_size;
        Self {
            state,
            phase: VerifierPhase::BeforeSetup,
            last_outer_round_poly: None,
            last_inner_round_poly: None,
            timer: Timer::from_eval_result(EvaluationResult {
                batch_size,
                ..Default::default()
            }),
        }
    }

    pub fn print_eval(&mut self) {
        let result = self.timer.finish();
        eprintln!("Eval report");
        eprintln!("sys: spain_decoupled_verifier");
        eprintln!("{result:#?}");
        eprintln!("End eval");
    }

    pub fn get_eval_result(&mut self) -> EvaluationResult {
        self.timer.finish()
    }

    pub fn set_eval_model_name(&mut self, model_name: impl Into<String>) {
        self.timer.eval_result_mut().model_name = model_name.into();
    }

    pub fn num_constraints(&self) -> usize {
        self.state.num_constraints()
    }

    pub fn is_done(&self) -> bool {
        matches!(self.phase, VerifierPhase::Done)
    }
}

impl<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> ProtocolState<SpainMessage<T>> for SpainVerifier<T, P, E>
where
    Matrix<T>: MatrixIntOps,
{
    fn init_message(&mut self) -> Result<SpainMessage<T>> {
        let setup_msg = self.timer.verifier(VerifierTimerPhase::Setup, || {
            self.state.dark_setup();
            self.phase = VerifierPhase::WaitingCommitment;
            SpainMessage::Setup(self.state.get_dark_public_params())
        });
        Ok(setup_msg)
    }

    fn handle_message(&mut self, m: &SpainMessage<T>) -> Option<SpainMessage<T>> {
        match (self.phase, m) {
            (VerifierPhase::WaitingCommitment, SpainMessage::Commitment(comm)) => {
                self.phase = VerifierPhase::WaitingErrorClaim;
                let randomness = self.timer.verifier(VerifierTimerPhase::Misc, || {
                    self.state.set_commit(comm.clone());
                    let randomness = self.state.sample_normal_randomness();
                    self.state.inject_randomness();
                    randomness
                });
                Some(SpainMessage::Randomness(randomness))
            }
            (VerifierPhase::WaitingErrorClaim, SpainMessage::ErrorClaim(squared_error)) => {
                self.phase = VerifierPhase::Outer;
                let squared_error =
                    i256::I512::from_str(squared_error).expect("invalid squared error encoding");
                self.timer.verifier(VerifierTimerPhase::EpsilonCheck, || {
                    self.state.epsilon_check(&squared_error)
                });
                let mont = self
                    .timer
                    .verifier(VerifierTimerPhase::Sample, || self.state.sample_mont());
                Some(SpainMessage::SampleMont {
                    small_modulus: mont.modulus(),
                })
            }
            (VerifierPhase::Outer, SpainMessage::OuterRoundClaim(poly)) => {
                if self.state.outer_state.is_none() {
                    self.timer.verifier(VerifierTimerPhase::RunOuterSc, || {
                        self.state.prepare_outer_sc()
                    });
                }
                let mut poly = poly.clone();
                let challenge = self.timer.verifier(VerifierTimerPhase::RunOuterSc, || {
                    self.state
                        .outer_sc_verify(&mut poly)
                        .expect("outer sc verify failed")
                });
                self.last_outer_round_poly = Some(poly);
                Some(SpainMessage::OuterChallenge(challenge))
            }
            (VerifierPhase::Outer, SpainMessage::OuterFinalEvals(evals)) => {
                let round_poly = self
                    .last_outer_round_poly
                    .as_ref()
                    .expect("missing final outer round polynomial");
                self.timer.verifier(VerifierTimerPhase::RunOuterSc, || {
                    self.state
                        .outer_sc_check_final_evals(round_poly, evals)
                        .expect("outer sc final evals check failed")
                });
                let (r1, r2) = self.timer.verifier(VerifierTimerPhase::Sample, || {
                    self.state.sample_lc_challenges()
                });
                self.phase = VerifierPhase::WaitingInnerStart;
                Some(SpainMessage::StartInner { r1, r2 })
            }
            (VerifierPhase::WaitingInnerStart, SpainMessage::InnerStart { num_vars, claim }) => {
                self.timer.verifier(VerifierTimerPhase::RunInnerSc, || {
                    self.state.prepare_inner_sc(*num_vars)
                });
                let mut claim = claim.clone();
                let challenge = self.timer.verifier(VerifierTimerPhase::RunInnerSc, || {
                    self.state
                        .inner_sc_verify(&mut claim)
                        .expect("inner sc verify failed")
                });
                self.last_inner_round_poly = Some(claim);
                self.phase = VerifierPhase::Inner;
                Some(SpainMessage::InnerChallenge(challenge))
            }
            (VerifierPhase::Inner, SpainMessage::InnerRoundClaim(claim)) => {
                let mut claim = claim.clone();
                let challenge = self.timer.verifier(VerifierTimerPhase::RunInnerSc, || {
                    self.state
                        .inner_sc_verify(&mut claim)
                        .expect("inner sc verify failed")
                });
                self.last_inner_round_poly = Some(claim);
                Some(SpainMessage::InnerChallenge(challenge))
            }
            (VerifierPhase::Inner, SpainMessage::InnerFinalEvals(evals)) => {
                let round_poly = self
                    .last_inner_round_poly
                    .as_ref()
                    .expect("missing final inner round polynomial");
                self.timer.verifier(VerifierTimerPhase::RunInnerSc, || {
                    self.state
                        .inner_sc_check_final_evals(round_poly, evals)
                        .expect("inner sc final evals check failed")
                });
                self.phase = VerifierPhase::WaitingDarkClaim;
                let eval_point = self.timer.verifier(VerifierTimerPhase::PolyEval, || {
                    self.state.dark_eval_point()
                });
                Some(SpainMessage::DarkEvalPoint(eval_point))
            }
            (VerifierPhase::WaitingDarkClaim, SpainMessage::DarkClaim(claim)) => {
                self.timer.verifier(VerifierTimerPhase::PolyEval, || {
                    self.state.set_dark_claim(*claim)
                });
                self.phase = VerifierPhase::DarkRounds;
                let challenge = self.timer.verifier(VerifierTimerPhase::PolyEval, || {
                    self.state.start_dark_round()
                });
                Some(SpainMessage::DarkChallenge(challenge))
            }
            (VerifierPhase::DarkRounds, SpainMessage::DarkRoundResponse(round_claim)) => {
                self.timer.verifier(VerifierTimerPhase::PolyEval, || {
                    self.state.verify_dark_round(round_claim)
                });
                if round_claim.final_claim.is_some() {
                    self.phase = VerifierPhase::WaitingWitnessOpenings;
                    Some(SpainMessage::RequestWitnessOpenings)
                } else {
                    let challenge = self.timer.verifier(VerifierTimerPhase::PolyEval, || {
                        self.state.start_dark_round()
                    });
                    Some(SpainMessage::DarkChallenge(challenge))
                }
            }
            (VerifierPhase::WaitingWitnessOpenings, SpainMessage::WitnessOpenings(z_openings)) => {
                self.timer.verifier(VerifierTimerPhase::SmartEval, || {
                    self.state.matrices_claim_check()
                });
                self.timer
                    .verifier(VerifierTimerPhase::ClaimInterpolate, || {
                        self.state.witness_claim_check(z_openings)
                    });
                eprintln!("Final verification success!!!");
                self.phase = VerifierPhase::Done;
                None
            }
            _ => panic!(
                "verifier received unexpected message {m:?} in phase {:?}",
                self.phase
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SpainProver, SpainVerifier};
    use crate::{
        actor::SpainMessage,
        inputs::{DEFAULT_DATA_DIR, import_metadata},
        prover::ProverState,
        verifier::VerifierState,
        witness_gen::OnnxExecutor,
    };
    use model::AFloat;
    use protocol::machine::simulate;
    use std::path::PathBuf;

    #[test]
    fn simulate_spain_actor_smoke() {
        let model = "layernorm_32x768";
        let path = PathBuf::from(DEFAULT_DATA_DIR);
        let meta_path = path.join(model).join("meta.json");
        if !meta_path.exists() {
            eprintln!("skipping test: missing {}", meta_path.display());
            return;
        }

        let metadata = import_metadata(&path, model);
        let scale_factor_bits = 70;
        let max_epsilon = 0.1;
        let num_chunks = 16;
        let precision = 128;
        let q_bits = 30_000;
        let batch_size = 1;
        let scale_factor = crate::prover::scale_factor::<AFloat>(scale_factor_bits);
        let wit_exec = OnnxExecutor::new(model.to_string(), path.clone(), metadata.clone(), true);

        let prover_state: ProverState<i128, AFloat, OnnxExecutor<AFloat>> = ProverState::new(
            wit_exec.clone(),
            scale_factor,
            metadata.clone(),
            batch_size,
            false,
        );
        let verifier_state: VerifierState<i128, AFloat, OnnxExecutor<AFloat>> = VerifierState::new(
            max_epsilon,
            batch_size,
            scale_factor_bits,
            q_bits,
            precision,
            num_chunks,
            false,
            wit_exec,
            metadata,
        );

        let prover = SpainProver::new(prover_state);
        let verifier = SpainVerifier::new(verifier_state);
        simulate::<SpainMessage, _, _>(prover, verifier).expect("spain actor simulate failed");
    }
}
