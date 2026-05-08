use clap::Parser;
use model::AFloat;
use otti_adapter::LPSpain;
use protocol::broker::JsonBroker;
use protocol::machine::run_actor;
use rayon::ThreadPoolBuilder;
use spain::actor::{SpainMessage, SpainProver};
use spain::prover::{ProverState, scale_factor};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Otti LP prover actor over JsonBroker")]
struct Cli {
    #[arg(long, default_value = "./datasets/sc105.mps")]
    mps_path: PathBuf,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 50)]
    scale_factor_bits: usize,
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
    let scale_factor: AFloat = scale_factor(cli.scale_factor_bits);

    let prover_state: ProverState<i128, AFloat, LPSpain> =
        ProverState::new(lp, scale_factor, metadata, cli.batch_size);
    let mut prover = SpainProver::new(prover_state);
    prover.set_eval_model_name(mps_path);

    run_actor::<SpainMessage, _, _>(&mut prover, JsonBroker::new())
        .expect("otti prover actor loop failed");

    let eval_result = prover.get_eval_result();
    eprintln!("Num constraints: {}", prover.num_constraints());
    eval_result.report_prover_time();
}
