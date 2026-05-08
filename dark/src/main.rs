pub mod rsagroup;
use dark::{prover::commit_chunked_pippenger, test::MockDARK};
use std::time::Instant;

fn main() {
    let num_vars = 10;

    // Setup Mock Dark
    println!("Initializing DARK with {} variable MLE", num_vars);
    let MockDARK {
        mle: poly,
        mut dark,
    } = MockDARK::new(num_vars);
    let mont = dark.public.small_mont;

    // Reduce to mont poly and eval
    let mut mont_poly = poly.reduce_to_mont(&mont);
    let eval_point_p = (0..num_vars)
        .map(|i| mont.to_mont((i as u128) * 7 + 3))
        .collect::<Vec<_>>();
    let mut rng = rand::rng();

    // Protocol setup
    dark.verifier.compute_const_comms(&mut dark.public);
    dark.public.build_pippenger_bases();

    // Initial commitment
    let initial_comm_time = Instant::now();
    dark.prover.int_poly = Some(poly.clone());
    let chunked_comm = commit_chunked_pippenger(poly.clone(), &dark.public);
    println!("Commitment time {:?}", initial_comm_time.elapsed());

    let y = mont_poly.eval(&eval_point_p, &mont);
    println!("Starting protocol");
    let protocol_start = Instant::now();
    let mut eval_result = dark.run_protocol(chunked_comm, poly, &eval_point_p, y, &mut rng);
    println!("Protocol time {:?}", protocol_start.elapsed());
    eval_result.calc_prover_total();
    println!("Eval result: {:?}", eval_result);
    println!("Total verifier time: {:?}", eval_result.verifier_time);
    println!("Total prover time: {:?}", eval_result.prover_time);
}
