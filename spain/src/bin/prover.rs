use clap::Parser;
use model::F128;
use protocol::broker::JsonBroker;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainProver};
use spain::inputs::{DEFAULT_DATA_DIR, import_metadata};
use spain::prover::{ProverState, scale_factor};
use spain::witness_gen::OnnxExecutor;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Spain prover actor over JsonBroker")]
struct Cli {
    #[arg(long, default_value = "layernorm_32x768")]
    model: String,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,
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
    let scale_factor: F128 = scale_factor(cli.scale_factor_bits);
    let wit_exec = OnnxExecutor::new(
        cli.model.clone(),
        cli.data_dir.clone(),
        metadata.clone(),
        cli.use_same_input,
    );

    let prover_state: ProverState<i128, F128, OnnxExecutor<F128>> =
        ProverState::new(wit_exec, scale_factor, metadata, cli.batch_size);
    let mut prover = SpainProver::new(prover_state);
    prover.set_eval_model_name(cli.model);

    run_actor::<SpainMessage, _, _>(&mut prover, JsonBroker::new())
        .expect("spain prover actor loop failed");

    let eval_result = prover.get_eval_result();

    if cli.phase_breakdown {
        eval_result.report_prover_phases();
    } else {
        eprintln!("Num constraints: {}", prover.num_constraints());
        eval_result.report_prover_time();
    }
}
