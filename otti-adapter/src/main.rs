use clap::Parser;
use dark::DARK;
use model::{AFloat, FBITS};
use otti_adapter::LPSpain;
use rayon::ThreadPoolBuilder;
use rug::{Float, ops::Pow};
use spain::EvaluationResult;
use spain::inputs::R1CSMatrices;
use spain::prover::simulate;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Standalone runner for the LP Spain adapter",
    long_about = None
)]
struct Cli {
    /// Path to the MPS file describing the LP instance
    #[arg(long, default_value = "./datasets/sc105.mps")]
    mps_path: PathBuf,

    /// Run all bundled MPS computations in the datasets directory
    #[arg(long, default_value_t = false)]
    run_all: bool,

    /// Batch size
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Scale factor bits for fixed-point conversion
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,

    /// Max epsilon for approximate checks
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,

    /// Chunk size for DARK commitment
    #[arg(long, default_value_t = 4)]
    chunk_size: usize,

    /// Number of bits for q in DARK
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,
}

fn bundled_mps_paths() -> Vec<PathBuf> {
    let datasets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets");
    let mut paths: Vec<PathBuf> = fs::read_dir(&datasets_dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", datasets_dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().is_some_and(|ext| ext == "mps") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths
}

fn run_single(
    mps_path: &Path,
    batch_size: usize,
    scale_factor_bits: usize,
    max_epsilon: f64,
    chunk_size: usize,
    q_bits: usize,
) {
    let mps_path_str = mps_path.to_str().expect("MPS path must be valid UTF-8");
    println!("Loading LP from {}", mps_path_str);
    let lp = LPSpain::parse_mps(mps_path_str);
    println!("Solving LP...");
    let solve_start = std::time::Instant::now();
    let sols = lp.solve();
    let solve_duration = solve_start.elapsed();
    println!("LP solved in {:?}", solve_duration);

    let metadata = lp.get_metadata();
    println!("Witness ranges: {:?}", metadata.get_ranges());

    let scale_factor = AFloat(Float::with_val(FBITS, 2).pow(scale_factor_bits as u32));
    let witness_gen_start = Instant::now();
    let z = lp
        .build_witness(&sols, scale_factor.clone())
        .repeat_column(batch_size);
    let witness_gen_time = witness_gen_start.elapsed();
    let ranges = metadata.get_ranges();
    let witness_rows = ranges[1].len();
    let witness_cols = z.width();
    let num_row_vars = witness_rows.next_power_of_two().trailing_zeros() as usize;
    let num_col_vars = witness_cols.next_power_of_two().trailing_zeros() as usize;
    let num_z_vars = num_row_vars + num_col_vars;

    println!("Setting up DARK instance...");
    let mut dark = DARK::new(q_bits, num_z_vars, chunk_size, 128);
    dark.verifier.compute_const_comms(&mut dark.public);
    dark.public.build_pippenger_bases();

    let (a, b, c) = lp.generate_r1cs_matrices(scale_factor.clone());
    let num_cons = a.height();
    let r1cs = R1CSMatrices { a, b, c };

    let model_name: String = Path::new(&mps_path_str)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let mut eval_result = EvaluationResult {
        model_name,
        batch_size,
        ..Default::default()
    };

    simulate(
        &r1cs,
        &z,
        &metadata,
        scale_factor_bits,
        scale_factor,
        max_epsilon,
        &mut dark,
        true,
        &mut eval_result,
    );

    println!(
        "Batch size: {} | Average time: {:?}",
        batch_size,
        eval_result.total_protocol_time / batch_size.try_into().unwrap()
    );
    eval_result.calc_totals();
    dbg!("Eval report");
    dbg!("sys: spain");
    dbg!(&eval_result);
    println!("prover_witness_generation_time:{:?}", witness_gen_time);
    println!("num_constraints: {}", &num_cons);
}

fn main() {
    env_logger::init();
    let Cli {
        mps_path,
        run_all,
        batch_size,
        scale_factor_bits,
        max_epsilon,
        chunk_size,
        q_bits,
    } = Cli::parse();

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let mps_paths = if run_all {
        bundled_mps_paths()
    } else {
        vec![mps_path]
    };

    for mps_path in mps_paths {
        run_single(
            &mps_path,
            batch_size,
            scale_factor_bits,
            max_epsilon,
            chunk_size,
            q_bits,
        );
    }
}
