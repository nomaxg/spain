use clap::Parser;
use examples::zklp::ZKLPExecutor;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainProver};
use spain::broker::SpainBroker;
use spain::prover::{ProverState, scale_factor};
use spain::traits::R1CSInstance;

#[derive(Parser, Debug)]
#[command(author, version, about = "ZKLP on Spain")]
struct Cli {
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    phase_breakdown: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let content = "3FE6B9A15D56C309 3FE2749BA27893E4 F 179BA4 0 16EDD5"
        .split_whitespace()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect::<Vec<_>>();
    let &[lat, lng, res, result_i, result_j, result_k] = content[0..6].try_into().unwrap();
    let lat = f64::from_bits(lat) as f64;
    let lng = f64::from_bits(lng) as f64;
    let alpha_lat = (lat / 2.).tan();
    let gamma_lat = (lat / 2.).sin();
    let delta_lat = (lat / 2.).cos();
    let beta_lat = 2. * gamma_lat * delta_lat;
    let alpha_lng = (lng / 2.).tan();
    let gamma_lng = (lng / 2.).sin();
    let delta_lng = (lng / 2.).cos();
    let beta_lng = 2. * gamma_lng * delta_lng;
    let exec = ZKLPExecutor::new(
        lat, lng, res, result_i, result_j, result_k, alpha_lat, beta_lat, gamma_lat, delta_lat,
        alpha_lng, beta_lng, gamma_lng, delta_lng,
    );
    let metadata = exec.get_meta();
    let scale_factor: f64 = scale_factor(cli.scale_factor_bits);

    let prover_state: ProverState<i128, f64, ZKLPExecutor> =
        ProverState::new(exec, scale_factor, metadata, cli.batch_size, false);
    let mut prover = SpainProver::new(prover_state);
    prover.set_eval_model_name(format!("zklp-spain-batch-size-{}", cli.batch_size));

    let (bytes_sent, time_spent, serialization_time) =
        run_actor::<SpainMessage, _, _>(&mut prover, SpainBroker::new())
            .expect("physics prover actor loop failed");

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

    eprintln!("Num constraints: {}", eval_result.num_constraints);

    if cli.phase_breakdown {
        eval_result.report_prover_phases();
    } else {
        eval_result.report_prover_time();
    }
}
