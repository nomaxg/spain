use std::time::{Duration, Instant};

use crate::EvaluationResult;

#[derive(Debug, Clone, Copy)]
pub enum ProverPhase {
    Preprocessing,
    ComputeWitness,
    ComputeSquaredError,
    PrepareOuterSc,
    RunOuterSc,
    PrepareInnerSc,
    RunInnerSc,
    PolyCommit,
    PolyEval,
    Misc,
}

#[derive(Debug, Clone, Copy)]
pub enum VerifierPhase {
    Setup,
    EpsilonCheck,
    Sample,
    RunOuterSc,
    RunInnerSc,
    PolyEval,
    Spark,
    ClaimInterpolate,
    SmartEval,
    Misc,
}

#[derive(Debug, Clone)]
pub struct Timer {
    eval_result: EvaluationResult,
    protocol_start: Instant,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Self {
        Self::from_eval_result(EvaluationResult::default())
    }

    pub fn from_eval_result(eval_result: EvaluationResult) -> Self {
        Self {
            eval_result,
            protocol_start: Instant::now(),
        }
    }

    pub fn prover<T>(&mut self, phase: ProverPhase, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        self.add_prover(phase, start.elapsed());
        result
    }

    pub fn verifier<T>(&mut self, phase: VerifierPhase, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        self.add_verifier(phase, start.elapsed());
        result
    }

    pub fn add_prover(&mut self, phase: ProverPhase, duration: Duration) {
        match phase {
            ProverPhase::Preprocessing => self.eval_result.prover_preprocessing_time += duration,
            ProverPhase::ComputeWitness => self.eval_result.prover_compute_witness += duration,
            ProverPhase::ComputeSquaredError => {
                self.eval_result.prover_compute_square_error_time += duration
            }
            ProverPhase::PrepareOuterSc => {
                self.eval_result.prover_prepare_outer_sc_time += duration
            }
            ProverPhase::RunOuterSc => self.eval_result.prover_run_outer_sc_time += duration,
            ProverPhase::PrepareInnerSc => {
                self.eval_result.prover_prepare_inner_sc_time += duration
            }
            ProverPhase::RunInnerSc => self.eval_result.prover_run_inner_sc_time += duration,
            ProverPhase::PolyCommit => self.eval_result.prover_poly_commit_time += duration,
            ProverPhase::PolyEval => self.eval_result.prover_poly_eval_time += duration,
            ProverPhase::Misc => self.eval_result.prover_misc_time += duration,
        }
    }

    pub fn add_verifier(&mut self, phase: VerifierPhase, duration: Duration) {
        match phase {
            VerifierPhase::Setup => self.eval_result.verifier_setup_time += duration,
            VerifierPhase::EpsilonCheck => self.eval_result.verifier_epsilon_check_time += duration,
            VerifierPhase::Sample => self.eval_result.verifier_sample_time += duration,
            VerifierPhase::RunOuterSc => self.eval_result.verifier_run_outer_sc_time += duration,
            VerifierPhase::RunInnerSc => self.eval_result.verifier_run_inner_sc_time += duration,
            VerifierPhase::PolyEval => self.eval_result.verifier_poly_eval_time += duration,
            VerifierPhase::Spark => self.eval_result.verifier_spark_time += duration,
            VerifierPhase::ClaimInterpolate => {
                self.eval_result.verifier_claim_interpolate_time += duration
            }
            VerifierPhase::SmartEval => self.eval_result.verifier_smart_eval_time += duration,
            VerifierPhase::Misc => self.eval_result.verifier_misc_time += duration,
        }
    }

    pub fn eval_result_mut(&mut self) -> &mut EvaluationResult {
        &mut self.eval_result
    }

    pub fn finish(&mut self) -> EvaluationResult {
        self.eval_result.total_protocol_time = self.protocol_start.elapsed();
        self.eval_result.calc_totals();
        self.eval_result.clone()
    }
}
