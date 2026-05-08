use clap::Parser;
use dark::DARK;
use model::{AFloat, F128, FromPrimitive, HighPrecision, TFloat};
use parse::generalized::HighPrecisionInt;
use parse::mat::Matrix;
use spain::EvaluationResult;
use spain::inputs::{
    DEFAULT_DATA_DIR, FromF64Matrix, Metadata, import_full_r1cs_int_deprecated, import_metadata,
    import_raw_r1cs_deprecated, import_witness_int_deprecated, import_witness_raw_deprecated,
};
use spain::prover::{compute_squared_error_raw, simulate, simulate_hp};
use spain::traits::{MatrixIntOps, ToI512, ToI1024};
use spain::witness_gen::{compute_squared_error_raw_hp, compute_witness, compute_witness_raw};
use std::fmt::Debug;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Spain DARK protocol runner",
    long_about = None
)]
struct Cli {
    /// Model name
    #[arg(long, default_value = "layer_norm")]
    model: String,

    /// Batch size
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Scale factor bits used for fixed-point representation
    #[arg(long, default_value_t = 34)]
    scale_factor_bits: usize,

    /// Max epsilon for approximate checks
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,

    /// Chunk size for DARK commitment
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,

    /// Number of bits for q in DARK
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,

    /// Directory containing model data files
    #[arg(long, value_name = "DIR", default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,

    /// Integer precision for witness/R1CS values (64, 128, 1280, 2560, 5120)
    #[arg(long, default_value_t = 1280)]
    precision: u16,

    /// Option only valid for custom onnx runner, where we use the precomputed input instead of a
    /// randomly generated one
    #[arg(long, default_value_t = false)]
    use_same_input: bool,

    /// Run the witness as {"AFloat", "TFloat", "F128", "CF128"}
    #[arg(long, default_value = "AFloat")]
    run_as: String,
}

fn main() {
    env_logger::init();
    let Cli {
        model,
        batch_size,
        chunk_size,
        q_bits,
        max_epsilon,
        scale_factor_bits,
        data_dir,
        precision,
        use_same_input,
        run_as,
    } = Cli::parse();

    match run_as.as_str() {
        "AFloat" => run::<AFloat>(
            data_dir,
            model,
            batch_size,
            scale_factor_bits,
            max_epsilon,
            chunk_size,
            q_bits,
            precision,
            use_same_input,
        ),
        "F128" => run::<F128>(
            data_dir,
            model,
            batch_size,
            scale_factor_bits,
            max_epsilon,
            chunk_size,
            q_bits,
            precision,
            use_same_input,
        ),
        "TFloat" => run::<TFloat>(
            data_dir,
            model,
            batch_size,
            scale_factor_bits,
            max_epsilon,
            chunk_size,
            q_bits,
            precision,
            use_same_input,
        ),
        // "CF128" => run::<CF128>(
        //     data_dir,
        //     model,
        //     batch_size,
        //     scale_factor_bits,
        //     max_epsilon,
        //     chunk_size,
        //     q_bits,
        //     precision,
        //     use_same_input,
        // ),
        _ => panic!("unknown runtime type. choose from AFloat, F128, or TFloat"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run<T: HighPrecision>(
    data_dir: PathBuf,
    model: String,
    batch_size: usize,
    scale_factor_bits: usize,
    max_epsilon: f64,
    chunk_size: usize,
    q_bits: usize,
    precision: u16,
    use_same_input: bool,
) {
    let metadata = import_metadata(data_dir.as_path(), &model);
    let raw_matrices = import_raw_r1cs_deprecated(data_dir.as_path(), &model, true);

    if precision > 128 {
        let (raw_z_hp, names, _) =
            compute_witness_raw::<T>(&data_dir, &model, &metadata, false, false);
        println!(
            "Raw error from computed input: {:e}",
            compute_squared_error_raw_hp(&raw_matrices, &raw_z_hp, &names, true)
                .to_f64()
                .unwrap()
                .sqrt()
        );
    } else {
        let raw_z = import_witness_raw_deprecated(data_dir.as_path(), &model, &metadata, true);
        let raw_error = compute_squared_error_raw(&raw_matrices, &raw_z, true);
        println!("Raw error from exported input: {:e}", raw_error.sqrt());
    }
    let scale_factor = 2_f64.powf(scale_factor_bits as f64);
    match precision {
        64 => run_with_precision::<i64>(
            data_dir.as_path(),
            &model,
            &metadata,
            batch_size,
            scale_factor_bits,
            AFloat::from_f64(scale_factor).unwrap(),
            max_epsilon,
            chunk_size,
            q_bits,
            precision,
        ),
        128 => run_with_precision::<i128>(
            data_dir.as_path(),
            &model,
            &metadata,
            batch_size,
            scale_factor_bits,
            AFloat::from_f64(scale_factor).unwrap(),
            max_epsilon,
            chunk_size,
            q_bits,
            precision,
        ),
        1280 => run_with_high_precision::<T, i128>(
            data_dir.as_path(),
            &model,
            &metadata,
            batch_size,
            scale_factor_bits,
            T::from_f64(scale_factor).unwrap(),
            max_epsilon,
            chunk_size,
            q_bits,
            128,
            use_same_input,
        ),
        // 2560 => run_with_high_precision::<T, I256>(
        //     data_dir.as_path(),
        //     &model,
        //     &metadata,
        //     batch_size,
        //     scale_factor_bits,
        //     T::from_f64(scale_factor).unwrap(),
        //     max_epsilon,
        //     chunk_size,
        //     q_bits,
        //     256,
        //     use_same_input,
        // ),
        // 5120 => run_with_high_precision::<T, I512>(
        //     data_dir.as_path(),
        //     &model,
        //     &metadata,
        //     batch_size,
        //     scale_factor_bits,
        //     T::from_f64(scale_factor).unwrap(),
        //     max_epsilon,
        //     chunk_size,
        //     q_bits,
        //     512,
        //     use_same_input,
        // ),
        _ => {
            eprintln!("Invalid precision {precision}. Use 64 or 128.");
            std::process::exit(2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_with_precision<T>(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    batch_size: usize,
    scale_factor_bits: usize,
    scale_factor: AFloat,
    max_epsilon: f64,
    chunk_size: usize,
    q_bits: usize,
    precision: u16,
) where
    T: FromF64Matrix + Copy + Clone + Default + PartialEq + Debug + ToI512,
    Matrix<T>: MatrixIntOps,
{
    let z =
        import_witness_int_deprecated::<T>(data_dir, model, metadata, scale_factor.clone(), true)
            .repeat_column(batch_size);
    let ranges = metadata.get_ranges();
    let witness_rows = ranges[1].len();
    let witness_cols = z.width();
    let num_row_vars = witness_rows.next_power_of_two().trailing_zeros() as usize;
    let num_col_vars = witness_cols.next_power_of_two().trailing_zeros() as usize;
    let num_z_vars = num_row_vars + num_col_vars;

    println!("Setting up DARK instance...");
    let dark_precision = precision.max(128);
    let mut dark = DARK::new(q_bits, num_z_vars as usize, chunk_size, dark_precision);
    dark.verifier.compute_const_comms(&mut dark.public);
    dark.public.build_pippenger_bases();
    let r1cs_matrices = import_full_r1cs_int_deprecated::<T>(
        data_dir,
        model,
        scale_factor.clone(),
        metadata,
        &z,
        true,
    );
    let mut eval_result = EvaluationResult {
        model_name: model.to_string(),
        batch_size,
        ..Default::default()
    };
    simulate(
        &r1cs_matrices,
        &z,
        metadata,
        scale_factor_bits,
        scale_factor,
        max_epsilon,
        &mut dark,
        true,
        &mut eval_result,
    );
    report_eval(batch_size, &mut eval_result);
}

#[allow(clippy::too_many_arguments)]
fn run_with_high_precision<P, T>(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    batch_size: usize,
    scale_factor_bits: usize,
    scale_factor: P,
    max_epsilon: f64,
    chunk_size: usize,
    q_bits: usize,
    precision: u16,
    use_same_input: bool,
) where
    P: HighPrecision,
    T: Copy
        + Clone
        + Default
        + PartialEq
        + HighPrecisionInt
        + FromF64Matrix
        + Debug
        + ToI512
        + ToI1024,
    Matrix<T>: MatrixIntOps,
{
    let mut eval_result = EvaluationResult {
        model_name: model.to_string(),
        batch_size,
        ..Default::default()
    };

    let compute_witness_timer = std::time::Instant::now();
    let z = compute_witness(
        data_dir,
        model,
        metadata,
        scale_factor.clone(),
        use_same_input,
        true,
    );
    eval_result.prover_compute_witness = compute_witness_timer.elapsed();
    let ranges = metadata.get_ranges();
    let witness_rows = ranges[1].len();
    let witness_cols = z.width();
    let num_row_vars = witness_rows.next_power_of_two().trailing_zeros() as usize;
    let num_col_vars = witness_cols.next_power_of_two().trailing_zeros() as usize;
    let num_z_vars = num_row_vars + num_col_vars;

    println!("Setting up DARK instance...");
    let dark_precision = precision.max(128);
    let mut dark = DARK::new(q_bits, num_z_vars as usize, chunk_size, dark_precision);
    dark.verifier.compute_const_comms(&mut dark.public);
    dark.public.build_pippenger_bases();
    let r1cs_matrices = import_full_r1cs_int_deprecated::<T>(
        data_dir,
        model,
        AFloat(scale_factor.to_rug_float()),
        metadata,
        &z,
        true,
    );
    simulate_hp(
        &r1cs_matrices,
        &z,
        metadata,
        scale_factor_bits,
        AFloat(scale_factor.to_rug_float()),
        max_epsilon,
        &mut dark,
        true,
        &mut eval_result,
    );
    report_eval(batch_size, &mut eval_result);
}

fn report_eval(batch_size: usize, eval_result: &mut EvaluationResult) {
    println!(
        "Batch size: {} | Average time: {:?}",
        batch_size,
        eval_result.total_protocol_time / batch_size as u32
    );
    // Report measurements
    dbg!("Eval report");
    dbg!("sys: spain");
    eval_result.calc_totals();
    dbg!(eval_result);
}
