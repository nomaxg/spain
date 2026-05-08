use criterion::{Criterion, criterion_group, criterion_main};
use ff::poly::mont::MLE;
use ff::prime_128::{rand_elem, rand_prime};
use ff::{FieldElem, FieldMont};
// seeded randomness
use rand::SeedableRng;
use stream::bigvec::BigVec;

fn setup_bench(num_vars: usize) -> (FieldMont, MLE, Vec<FieldElem>) {
    // init seeded rng
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    // create a random prime modulus
    let p = rand_prime(&mut rng);
    // create a new montgomery context
    let mont = FieldMont::new(p);
    // create MLE with num_vars
    let mut evals = BigVec::new(1usize << num_vars).unwrap();
    evals[0] = mont.zero();
    for i in 1..(1usize << num_vars) {
        evals[i] = mont.add(evals[i - 1], mont.one());
    }
    let poly = MLE::from_buffer(evals, num_vars);
    // create a random point in the field
    let point = (0..num_vars)
        .map(|_| mont.to_mont(rand_elem(p, &mut rng)))
        .collect::<Vec<_>>();
    (mont, poly, point)
}

// benchmark mle evaluation (sequence of bind ops)
fn bench_eval_mle(_: &mut Criterion) {
    for num_vars in 2..=32 {
        let start = std::time::Instant::now();
        let (mont, mut poly, point) = setup_bench(num_vars);
        let elapsed = start.elapsed();
        println!("Setup for {} variables took: {:?}", num_vars, elapsed);
        let start = std::time::Instant::now();
        poly.eval(&point, &mont);
        let elapsed = start.elapsed();
        println!("Binding {} variables took: {:?}", num_vars, elapsed);
    }
}

criterion_group!(benches, bench_eval_mle,);
criterion_main!(benches);
