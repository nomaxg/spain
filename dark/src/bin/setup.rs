use dark::DARK;
use std::time::Instant;

fn main() {
    let setup_start = Instant::now();
    let num_chunks = 1024;
    let num_vars = 30;
    let q_bits = 30000;
    let mut dark = DARK::new(q_bits, num_vars, num_chunks, 128);
    dark.verifier.compute_const_comms(&mut dark.public);
    dark.public.build_pippenger_bench();
    println!("Precompute time {:?}", setup_start.elapsed());
}
