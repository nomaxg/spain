use clap::Parser;
use model::F128;
use protocol::broker::JsonBroker;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainVerifier};
use spain::inputs::{DEFAULT_DATA_DIR, import_metadata};
use spain::verifier::VerifierState;
use spain::witness_gen::OnnxExecutor;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Spain verifier actor over JsonBroker")]
struct Cli {
    #[arg(long, default_value = "layer_norm")]
    model: String,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,
    #[arg(long, default_value_t = 128)]
    precision: u16,
    #[arg(long, value_name = "DIR", default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_same_input: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    phase_breakdown: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let metadata = import_metadata(&cli.data_dir, &cli.model);
    let wit_exec = OnnxExecutor::new(
        cli.model.clone(),
        cli.data_dir.clone(),
        metadata.clone(),
        cli.use_same_input,
    );

    let verifier_state: VerifierState<i128, F128, OnnxExecutor<F128>> = VerifierState::new(
        cli.max_epsilon,
        cli.batch_size,
        cli.scale_factor_bits,
        cli.q_bits,
        cli.precision,
        cli.chunk_size,
        wit_exec,
        metadata,
    );
    let mut verifier = SpainVerifier::new(verifier_state);
    verifier.set_eval_model_name(cli.model);

    run_actor::<SpainMessage, _, _>(&mut verifier, JsonBroker::new())
        .expect("spain verifier actor loop failed");

    let eval_result = verifier.get_eval_result();

    if !verifier.is_done() {
        panic!(
            "Verifier terminated without completing, ensure that are running the prover with sufficient memory."
        );
    }

    if cli.phase_breakdown {
        eval_result.report_verifier_phases();
    } else {
        eval_result.report_verifier_time();
    }
}
