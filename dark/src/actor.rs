use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use ff::ops_128::M128;
use ff::poly::int::MLE as IntMLE;
use protocol::broker::JsonBroker;
use protocol::machine::{run_actor, ProtocolState};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};

use crate::prover::{ChunkedComm, ProverState, RoundClaim};
use crate::public::PublicParams;
use crate::test::{mock_eval_point, mock_poly};
use crate::verifier::{RoundChallenge, VerifierState};

pub fn run() {
    run_with_cli(Cli::parse())
}

pub fn run_with_cli(cli: Cli) {
    let config = SharedConfig::from(&cli);
    let mut prover = DarkProver::new();
    let mut verifier = DarkVerifier::new(config).expect("verifier init failed");
    let num_vars = cli.num_vars;
    prover.set_poly(mock_poly(num_vars));
    verifier.set_eval_point(mock_eval_point(&verifier.public.small_mont, num_vars));
    match cli.role {
        Role::Prover => run_actor(&mut prover, JsonBroker::new()).unwrap(),
        Role::Verifier => run_actor(&mut verifier, JsonBroker::new()).unwrap(),
    };
}

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(long, value_enum)]
    role: Role,
    #[arg(long, default_value_t = 10)]
    pub num_vars: usize,
    #[arg(long, default_value_t = 16)]
    pub num_chunks: usize,
    #[arg(long, default_value_t = 30_000)]
    pub q_bits: usize,
    #[arg(long, default_value_t = 128)]
    pub precision: u16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Role {
    Prover,
    Verifier,
}

#[derive(Debug, Clone, Copy)]
pub struct SharedConfig {
    pub num_vars: usize,
    pub num_chunks: usize,
    pub q_bits: usize,
    pub precision: u16,
}

impl From<&Cli> for SharedConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            num_vars: cli.num_vars,
            num_chunks: cli.num_chunks,
            q_bits: cli.q_bits,
            precision: cli.precision,
        }
    }
}

impl SharedConfig {
    pub fn validate(self) -> Result<()> {
        assert!(
            self.precision == 64 || self.precision == 128 || self.precision == 256,
            "precision must be 64, 128, or 256, got {}",
            self.precision
        );
        assert!(self.num_vars > 0, "num_vars must be positive");
        assert!(
            self.num_chunks > 0 && self.num_chunks.is_power_of_two(),
            "num_chunks must be a power of two, got {}",
            self.num_chunks
        );
        let total_points = (1usize)
            .checked_shl(self.num_vars as u32)
            .ok_or_else(|| anyhow!("num_vars {} is too large", self.num_vars))?;
        assert!(
            self.num_chunks < total_points,
            "num_chunks {} must be smaller than polynomial length {}",
            self.num_chunks,
            total_points
        );
        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DarkMessage {
    Setup(PublicParams),
    ProverCommitment(ChunkedComm),
    EvalPoint(Vec<M128>),
    InitialClaim(M128),
    Challenge(RoundChallenge),
    RoundResponse(RoundClaim),
}

#[derive(Default)]
pub struct DarkProver {
    public: Option<PublicParams>,
    state: ProverState,
    poly: Option<IntMLE>,
}

impl DarkProver {
    pub fn new() -> Self {
        Self {
            public: None,
            state: ProverState::default(),
            poly: None,
        }
    }

    pub fn set_poly(&mut self, poly: IntMLE) {
        self.poly = Some(poly);
    }

    fn handle_setup(&mut self, payload: &PublicParams) -> DarkMessage {
        let poly = self.poly.clone().expect("dark prover poly not initialized");
        let mut public = payload.clone();
        public.build_pippenger_bases();
        let chunked_comm = self.state.commit(poly, &public);
        self.public = Some(public);
        DarkMessage::ProverCommitment(chunked_comm)
    }

    fn handle_eval_point(&mut self, payload: &[M128]) -> DarkMessage {
        let public = self
            .public
            .as_ref()
            .expect("public params missing before eval point");
        let y = self.state.gen_y_claim(payload.to_vec(), public);
        DarkMessage::InitialClaim(y)
    }

    fn handle_challenge(&mut self, payload: &RoundChallenge) -> DarkMessage {
        let public = self
            .public
            .as_ref()
            .expect("public params missing before challenge");
        let round_claim = self.state.respond_to_challenge(payload, public);
        DarkMessage::RoundResponse(round_claim)
    }
}

impl ProtocolState<DarkMessage> for DarkProver {
    fn init_message(&mut self) -> Result<DarkMessage> {
        Err(anyhow!("dark prover does not initiate"))
    }

    fn handle_message(&mut self, m: &DarkMessage) -> Option<DarkMessage> {
        Some(match m {
            DarkMessage::Setup(payload) => self.handle_setup(payload),
            DarkMessage::EvalPoint(payload) => self.handle_eval_point(payload),
            DarkMessage::Challenge(payload) => self.handle_challenge(payload),
            other => panic!("prover received unexpected message {other:?}"),
        })
    }
}

pub struct DarkVerifier {
    public: PublicParams,
    state: VerifierState,
    eval_point: Option<Vec<M128>>,
    rng: ThreadRng,
}

impl DarkVerifier {
    pub fn new(config: SharedConfig) -> Result<Self> {
        config.validate()?;
        let mut public = PublicParams::new(
            config.q_bits,
            config.num_vars,
            config.num_chunks,
            config.precision,
        );
        let mut state = VerifierState::new(&public);
        state.compute_const_comms(&mut public);
        Ok(Self {
            public,
            state,
            eval_point: None,
            rng: rand::rng(),
        })
    }

    pub fn set_eval_point(&mut self, eval_point: Vec<M128>) {
        self.eval_point = Some(eval_point);
    }

    fn next_challenge(&mut self) -> DarkMessage {
        DarkMessage::Challenge(self.state.start_round(&self.public, &mut self.rng))
    }

    fn handle_commitment(&mut self, payload: &ChunkedComm) -> DarkMessage {
        let eval_point = self
            .eval_point
            .clone()
            .expect("verifier eval point not initialized");
        self.state.set_commit(payload.clone());
        DarkMessage::EvalPoint(eval_point)
    }

    fn handle_initial_claim(&mut self, y: M128) -> DarkMessage {
        let eval_point = self
            .eval_point
            .clone()
            .expect("verifier eval point missing before initial claim");
        self.state.set_claim(y, eval_point);
        self.next_challenge()
    }

    fn handle_round_response(&mut self, payload: &RoundClaim) -> Option<DarkMessage> {
        self.state.verify_round(payload, &self.public);

        if self.state.round == self.public.total_rounds {
            let final_claim = payload
                .final_claim
                .clone()
                .expect("missing final claim on final round");
            self.state.final_check(
                &final_claim.final_constant,
                &final_claim.final_constant_int,
                &self.public,
            );
            None
        } else {
            assert!(
                payload.final_claim.is_none(),
                "unexpected final claim before final round"
            );
            Some(self.next_challenge())
        }
    }
}

impl ProtocolState<DarkMessage> for DarkVerifier {
    fn init_message(&mut self) -> Result<DarkMessage> {
        Ok(DarkMessage::Setup(self.public.clone()))
    }

    fn handle_message(&mut self, m: &DarkMessage) -> Option<DarkMessage> {
        match m {
            DarkMessage::ProverCommitment(payload) => Some(self.handle_commitment(payload)),
            DarkMessage::InitialClaim(y) => Some(self.handle_initial_claim(*y)),
            DarkMessage::RoundResponse(payload) => self.handle_round_response(payload),
            other => panic!("verifier received unexpected message {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::{mock_eval_point, mock_poly};
    use protocol::machine::simulate;

    #[test]
    fn simulate_dark_protocol_smoke() {
        let num_vars = 10;
        let config = SharedConfig {
            num_vars,
            num_chunks: 16,
            q_bits: 30_000,
            precision: 128,
        };

        let mut prover = DarkProver::new();
        let mut verifier = DarkVerifier::new(config).expect("verifier init failed");
        prover.set_poly(mock_poly(num_vars));
        verifier.set_eval_point(mock_eval_point(&verifier.public.small_mont, num_vars));

        simulate(prover, verifier).expect("dark actor simulation failed");
    }
}
