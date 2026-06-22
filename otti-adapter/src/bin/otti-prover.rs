use clap::Parser;
use model::AFloat;
use otti_adapter::{LPSpain, OttiExec};
use parse::generalized::I256;
use protocol::machine::run_actor;
use rayon::ThreadPoolBuilder;
use spain::actor::{SpainMessage, SpainProver};
use spain::broker::SpainBroker;
use spain::prover::{ProverState, scale_factor};
use spain::traits::R1CSInstance;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about = "Otti LP prover actor over SpainBroker")]
struct Cli {
    #[arg(long, default_value = "./datasets/sc105.mps")]
    mps_path: PathBuf,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 50)]
    scale_factor_bits: usize,
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
    let scale_factor: AFloat = scale_factor(cli.scale_factor_bits);
    let model_name = dataset_name_from_mps_path(&cli.mps_path);

    let eval_result = if cli.otti_sid {
        let exec = OttiExec::new(&model_name);
        let metadata = exec.get_meta();
        let prover_state: ProverState<I256, AFloat, OttiExec> =
            ProverState::new(exec, scale_factor, metadata, cli.batch_size, true);
        let mut prover = SpainProver::new(prover_state);
        prover.set_eval_model_name(model_name);

        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<I256>, _, _>(&mut prover, SpainBroker::new())
                .expect("otti prover actor loop failed");

        if !prover.is_done() {
            panic!(
                "Prover terminated before protocol completion, verifier may have panicked or disconnected",
            );
        }

        let mut eval_result = prover.get_eval_result();
        eval_result.num_constraints = prover.num_constraints();
        eval_result.prover_bytes_sent = bytes_sent;
        eval_result.total_prover_actor_time = time_spent;
        eval_result.prover_serialization_time = serialization_time;
        eval_result.calc_totals();

        eval_result
    } else {
        let lp = LPSpain::parse_mps(&mps_path);
        let metadata = lp.get_metadata();
        let prover_state: ProverState<i128, AFloat, LPSpain> =
            ProverState::new(lp, scale_factor, metadata, cli.batch_size, false);
        let mut prover = SpainProver::new(prover_state);
        prover.set_eval_model_name(mps_path);

        let (bytes_sent, time_spent, serialization_time) =
            run_actor::<SpainMessage<i128>, _, _>(&mut prover, SpainBroker::new())
                .expect("otti prover actor loop failed");

        if !prover.is_done() {
            panic!(
                "Prover terminated before protocol completion, verifier may have panicked or disconnected",
            );
        }

        let mut eval_result = prover.get_eval_result();
        eval_result.num_constraints = prover.num_constraints();
        eval_result.prover_bytes_sent = bytes_sent;
        eval_result.total_prover_actor_time = time_spent;
        eval_result.prover_serialization_time = serialization_time;
        eval_result.calc_totals();

        eval_result
    };

    eprintln!("Num constraints: {}", eval_result.num_constraints);

    if cli.phase_breakdown {
        eval_result.report_prover_phases();
    } else {
        eval_result.report_prover_time();
        eprintln!(
            "Prover bytes sent to verifier: {:#?}",
            eval_result.prover_bytes_sent
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
