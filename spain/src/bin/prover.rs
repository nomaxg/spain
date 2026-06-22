use clap::Parser;
use model::F128;
use parse::generalized::I256;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainProver};
use spain::broker::SpainBroker;
use spain::executor::{SpainExecutor, build_spain_executor, build_zklp_executor};
use spain::inputs::DEFAULT_DATA_DIR;
use spain::prover::{ProverState, scale_factor};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Spain prover actor over SpainBroker")]
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
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    zklp: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let scale_factor: F128 = scale_factor(cli.scale_factor_bits);
    if cli.zklp {
        let (wit_exec, metadata) = build_zklp_executor::<F128>(
            cli.model.clone(),
            cli.data_dir.clone(),
            cli.scale_factor_bits,
        );
        let prover_state: ProverState<I256, F128, SpainExecutor<F128, I256>> =
            ProverState::new(wit_exec, scale_factor, metadata, cli.batch_size, cli.zklp);
        let mut prover = SpainProver::new(prover_state);
        prover.set_eval_model_name(cli.model);
        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<I256>, _, _>(&mut prover, SpainBroker::new())
                .expect("spain prover actor loop failed");

        if !prover.is_done() {
            panic!(
                "Prover terminated before protocol completion, verifier may have panicked or disconnected",
            );
        }

        let mut eval_result = prover.get_eval_result();
        eval_result.prover_bytes_sent = bytes_sent;
        eval_result.total_prover_actor_time = time_spent;
        eval_result.prover_serialization_time = serialization_time;

        eprintln!("Num constraints: {}", prover.num_constraints());

        if cli.phase_breakdown {
            eval_result.report_prover_phases();
        } else {
            eval_result.report_prover_time();
        }
    } else {
        let (wit_exec, metadata) = build_spain_executor::<F128>(
            cli.model.clone(),
            cli.data_dir.clone(),
            cli.use_same_input,
        );
        let prover_state: ProverState<i128, F128, SpainExecutor<F128, i128>> =
            ProverState::new(wit_exec, scale_factor, metadata, cli.batch_size, cli.zklp);
        let mut prover = SpainProver::new(prover_state);
        prover.set_eval_model_name(cli.model);
        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<i128>, _, _>(&mut prover, SpainBroker::new())
                .expect("spain prover actor loop failed");

        if !prover.is_done() {
            panic!(
                "Prover terminated before protocol completion, verifier may have panicked or disconnected",
            );
        }

        let mut eval_result = prover.get_eval_result();
        eval_result.prover_bytes_sent = bytes_sent;
        eval_result.total_prover_actor_time = time_spent;
        eval_result.prover_serialization_time = serialization_time;

        eprintln!("Num constraints: {}", prover.num_constraints());

        if cli.phase_breakdown {
            eval_result.report_prover_phases();
        } else {
            eval_result.report_prover_time();
        }
    }
}
