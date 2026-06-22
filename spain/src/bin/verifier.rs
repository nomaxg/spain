use clap::Parser;
use model::F128;
use parse::generalized::I256;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainVerifier};
use spain::broker::SpainBroker;
use spain::executor::{SpainExecutor, build_spain_executor, build_zklp_executor};
use spain::inputs::DEFAULT_DATA_DIR;
use spain::verifier::VerifierState;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Spain verifier actor over SpainBroker")]
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
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    zklp: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    if cli.zklp {
        let (wit_exec, metadata) = build_zklp_executor::<F128>(
            cli.model.clone(),
            cli.data_dir.clone(),
            cli.scale_factor_bits,
        );
        let verifier_state: VerifierState<I256, F128, SpainExecutor<F128, I256>> =
            VerifierState::new(
                cli.max_epsilon,
                cli.batch_size,
                cli.scale_factor_bits,
                cli.q_bits,
                cli.precision,
                cli.chunk_size,
                cli.zklp,
                wit_exec,
                metadata,
            );
        let mut verifier = SpainVerifier::new(verifier_state);
        verifier.set_eval_model_name(cli.model);

        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<I256>, _, _>(&mut verifier, SpainBroker::new())
                .expect("spain verifier actor loop failed");

        let mut eval_result = verifier.get_eval_result();
        eval_result.verifier_bytes_sent = bytes_sent;
        eval_result.total_verifier_actor_time = time_spent;
        eval_result.verifier_serialization_time = serialization_time;

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
    } else {
        let (wit_exec, metadata) = build_spain_executor::<F128>(
            cli.model.clone(),
            cli.data_dir.clone(),
            cli.use_same_input,
        );
        let verifier_state: VerifierState<i128, F128, SpainExecutor<F128, i128>> =
            VerifierState::new(
                cli.max_epsilon,
                cli.batch_size,
                cli.scale_factor_bits,
                cli.q_bits,
                cli.precision,
                cli.chunk_size,
                cli.zklp,
                wit_exec,
                metadata,
            );
        let mut verifier = SpainVerifier::new(verifier_state);
        verifier.set_eval_model_name(cli.model);

        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<i128>, _, _>(&mut verifier, SpainBroker::new())
                .expect("spain verifier actor loop failed");

        let mut eval_result = verifier.get_eval_result();
        eval_result.verifier_bytes_sent = bytes_sent;
        eval_result.total_verifier_actor_time = time_spent;
        eval_result.verifier_serialization_time = serialization_time;

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
}
