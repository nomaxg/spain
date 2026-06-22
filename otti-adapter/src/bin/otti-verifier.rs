use clap::Parser;
use model::AFloat;
use otti_adapter::{LPSpain, OttiExec};
use parse::generalized::I256;
use protocol::machine::run_actor;
use rayon::ThreadPoolBuilder;
use spain::actor::{SpainMessage, SpainVerifier};
use spain::broker::SpainBroker;
use spain::traits::R1CSInstance;
use spain::verifier::VerifierState;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about = "Otti LP verifier actor over SpainBroker")]
struct Cli {
    #[arg(long, default_value = "./datasets/sc105.mps")]
    mps_path: PathBuf,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 34)]
    scale_factor_bits: usize,
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,
    #[arg(long, default_value_t = 128)]
    precision: u16,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    otti_sid: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    phase_breakdown: bool,
}

fn main() {
    env_logger::init();
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let cli = Cli::parse();
    let mps_path = cli
        .mps_path
        .to_str()
        .expect("MPS path must be valid UTF-8")
        .to_string();
    let model_name = dataset_name_from_mps_path(&cli.mps_path);

    let eval_result = if cli.otti_sid {
        let exec = OttiExec::new(&model_name);
        let metadata = exec.get_meta();
        let verifier_state: VerifierState<I256, AFloat, OttiExec> = VerifierState::new(
            cli.max_epsilon,
            cli.batch_size,
            cli.scale_factor_bits,
            cli.q_bits,
            256,
            cli.chunk_size,
            true,
            exec,
            metadata,
        );
        let mut verifier = SpainVerifier::new(verifier_state);
        verifier.set_eval_model_name(model_name);

        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<I256>, _, _>(&mut verifier, SpainBroker::new())
                .expect("otti verifier actor loop failed");

        let mut eval_result = verifier.get_eval_result();
        if !verifier.is_done() {
            panic!(
                "Verifier terminated without completing, ensure that are running the prover with sufficient memory."
            );
        }

        eval_result.num_constraints = verifier.num_constraints();
        eval_result.verifier_bytes_sent = bytes_sent;
        eval_result.total_verifier_actor_time = time_spent;
        eval_result.verifier_serialization_time = serialization_time;
        eval_result.calc_totals();
        eval_result
    } else {
        let lp = LPSpain::parse_mps(&mps_path);
        let metadata = lp.get_metadata();
        let verifier_state: VerifierState<i128, AFloat, LPSpain> = VerifierState::new(
            cli.max_epsilon,
            cli.batch_size,
            cli.scale_factor_bits,
            cli.q_bits,
            cli.precision,
            cli.chunk_size,
            false,
            lp,
            metadata,
        );
        let mut verifier = SpainVerifier::new(verifier_state);
        verifier.set_eval_model_name(mps_path);

        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<i128>, _, _>(&mut verifier, SpainBroker::new())
                .expect("otti verifier actor loop failed");

        let mut eval_result = verifier.get_eval_result();
        if !verifier.is_done() {
            panic!(
                "Verifier terminated without completing, ensure that are running the prover with sufficient memory."
            );
        }

        eval_result.num_constraints = verifier.num_constraints();
        eval_result.verifier_bytes_sent = bytes_sent;
        eval_result.total_verifier_actor_time = time_spent;
        eval_result.verifier_serialization_time = serialization_time;
        eval_result.calc_totals();
        eval_result
    };

    eprintln!("Num constraints: {}", eval_result.num_constraints);

    if cli.phase_breakdown {
        eval_result.report_verifier_phases();
    } else {
        eval_result.report_verifier_time();
        eprintln!(
            "Verifier bytes sent to prover: {:#?}",
            eval_result.verifier_bytes_sent
        );
    }
}

fn dataset_name_from_mps_path(mps_path: &Path) -> String {
    mps_path
        .file_stem()
        .expect("MPS path must have a file stem")
        .to_str()
        .expect("MPS path stem must be valid UTF-8")
        .to_string()
}
