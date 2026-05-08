use clap::Parser;
use model::AFloat;
use otti_adapter::LPSpain;
use protocol::broker::JsonBroker;
use protocol::machine::run_actor;
use rayon::ThreadPoolBuilder;
use spain::actor::{SpainMessage, SpainVerifier};
use spain::verifier::VerifierState;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Otti LP verifier actor over JsonBroker")]
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
    let lp = LPSpain::parse_mps(&mps_path);
    let metadata = lp.get_metadata();

    let verifier_state: VerifierState<i128, AFloat, LPSpain> = VerifierState::new(
        cli.max_epsilon,
        cli.batch_size,
        cli.scale_factor_bits,
        cli.q_bits,
        cli.precision,
        cli.chunk_size,
        lp,
        metadata,
    );
    let mut verifier = SpainVerifier::new(verifier_state);
    verifier.set_eval_model_name(mps_path);

    run_actor::<SpainMessage, _, _>(&mut verifier, JsonBroker::new())
        .expect("otti verifier actor loop failed");
    let eval_result = verifier.get_eval_result();
    eval_result.report_verifier_time();
}
