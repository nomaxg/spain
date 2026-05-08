use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use stream::bigvec::BigVec;

// helper function to initialize a BigVec with random elements
fn init_rand_bigvec(len: usize) -> BigVec<f64> {
    let mut bigvec = BigVec::<f64>::new(len).expect("Failed to create BigVec");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    // fill the BigVec with random f64 values
    for i in 0..len {
        bigvec[i] = rng.random();
    }
    bigvec
}

// bench the throughput for BigVec
// sequential read
fn bench_seq_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq-read");
    let mut len = 1024; // start with 1024 elements
    loop {
        // initialize a BigVec with random elements
        let mut vec = init_rand_bigvec(len);
        vec.advise_seq();
        // get throughput based on the number of elements
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(&format!("seq_read_{}", len), |b| {
            b.iter(|| {
                // read all elements sequentially
                for i in 0..len {
                    black_box(vec[i]);
                }
            });
        });
        len *= 4; // increase the number of elements by 4x each iteration
        // stop when the length exceeds 32 * 1024^2
        if len > 32 * 1024 * 1024 {
            break;
        }
    }
    group.finish();
}

// sequential write
fn bench_seq_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq-write");
    let mut len = 1024; // start with 1024 elements
    loop {
        // initialize a BigVec with random elements
        let mut vec = init_rand_bigvec(len);
        vec.advise_seq();
        // get throughput based on the number of elements
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(format!("Seq Write {}", len), |b| {
            b.iter(|| {
                // sequentially write all elements
                for i in 0..vec.len() {
                    vec[i] = black_box(i as f64);
                }
            });
        });
        len *= 4; // increase the number of elements by 4x each iteration
        // stop when the length exceeds 32 * 1024^2
        if len > 32 * 1024 * 1024 {
            break;
        }
    }
    group.finish();
}

// random read
fn bench_random_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand-read");
    let mut len = 1024; // start with 1024 elements
    loop {
        // initialize a BigVec with random elements
        let vec = init_rand_bigvec(len);
        // get throughput based on the number of elements
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(&format!("random_read_{}", len), |b| {
            b.iter(|| {
                // read "random" elements with Numerical recipes LCG
                let mask = len - 1; // mask to ensure idx is within bounds
                let mut idx = 1;
                for _ in 0..len {
                    idx = (1664525 * idx + 1013904223) & mask; // LCG formula
                    black_box(vec[idx]);
                }
            });
        });
        len *= 4; // increase the number of elements by 4x each iteration
        // stop when the length exceeds 32 * 1024^2
        if len > 32 * 1024 * 1024 {
            break;
        }
    }
    group.finish();
}

// random write
fn bench_random_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand-write");
    let mut len = 1024; // start with 1024 elements
    loop {
        // initialize a BigVec with random elements
        let mut vec = init_rand_bigvec(len);
        // get throughput based on the number of elements
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(format!("random_write {}", len), |b| {
            b.iter(|| {
                // write "random" elements with Numerical recipes LCG
                let mask = len - 1; // mask to ensure idx is within bounds
                let mut idx = 1;
                for i in 0..len {
                    idx = (1664525 * idx + 1013904223) & mask; // LCG formula
                    vec[idx] = black_box(i as f64);
                }
            });
        });
        len *= 4; // increase the number of elements by 4x each iteration
        // stop when the length exceeds 32 * 1024^2
        if len > 32 * 1024 * 1024 {
            break;
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_seq_read,
    bench_seq_write,
    bench_random_read,
    bench_random_write
);
criterion_main!(benches);
