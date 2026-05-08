use criterion::{Criterion, criterion_group, criterion_main};
use ff::ops::{M64, Mont, add_mod, mul_mod};
use ff::prime::{rand_elem, rand_prime};
use std::hint::black_box;
// seeded randomness
use rand::SeedableRng;

fn init_harness() -> (u64, u64, Mont, M64) {
    // init seeded rng
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    // create a random prime modulus
    let p = rand_prime(&mut rng);
    // get a random element in the field
    let x = rand_elem(p, &mut rng);
    // create a new montgomery context
    let mont = Mont::new(p);
    // convert x to montgomery form
    let xm = mont.to_mont(x);
    // return all
    (p, x, mont, xm)
}

// benchmark for naive multiplication w/ mod
fn bench_naive_mul(c: &mut Criterion) {
    // init harness
    let (p, mut x, _, _) = init_harness();
    // benchmark the naive multiplication
    c.bench_function("naive multiplication", |b| {
        b.iter(|| {
            x = mul_mod(black_box(x), black_box(x), black_box(p));
        })
    });
}

// benchmark for montgomery multiplication
fn bench_mont_mul(c: &mut Criterion) {
    // init harness
    let (_, _, mont, mut xm) = init_harness();
    // benchmark the montgomery multiplication
    c.bench_function("montgomery multiplication", |b| {
        b.iter(|| {
            xm = mont.mul(black_box(xm), black_box(xm));
        })
    });
}

// benchmark for conversion to montgomery form
fn bench_mont_conv(c: &mut Criterion) {
    // init harness
    let (_, x, mont, _) = init_harness();
    // benchmark the conversion to montgomery form
    c.bench_function("montgomery conversion", |b| {
        b.iter(|| {
            black_box(mont.to_mont(black_box(x)));
        })
    });
}

// benchmark for montgomery square
fn bench_mont_sqr(c: &mut Criterion) {
    // init harness
    let (_, _, mont, mut xm) = init_harness();
    // benchmark the montgomery multiplication
    c.bench_function("montgomery square", |b| {
        b.iter(|| {
            xm = mont.sqr(black_box(xm));
        })
    });
}

// benchmark for modular addition
fn bench_add_mod(c: &mut Criterion) {
    // init harness
    let (p, mut x, _, _) = init_harness();
    // benchmark the modular addition
    c.bench_function("modular addition", |b| {
        b.iter(|| {
            x = add_mod(black_box(x), black_box(x), p);
        })
    });
}

// benchmark for modular subtraction
fn bench_sub_mod(c: &mut Criterion) {
    // init harness
    let (p, mut x, _, _) = init_harness();
    // benchmark the modular subtraction
    c.bench_function("modular subtraction", |b| {
        b.iter(|| {
            x = add_mod(black_box(x), black_box(x), p);
        })
    });
}

fn bench_mont_inv(c: &mut Criterion) {
    // init harness
    let (_, _, mont, mut xm) = init_harness();
    // benchmark the montgomery multiplication
    c.bench_function("montgomery inverse", |b| {
        b.iter(|| {
            xm = mont.inv(black_box(xm));
        })
    });
}

// benchmark for u64 multiplication, no mod
fn bench_u64_mul(c: &mut Criterion) {
    // init harness
    let (_, mut x, _, _) = init_harness();
    // benchmark the u64 multiplication
    c.bench_function("u64 multiplication", |b| {
        b.iter(|| {
            x = black_box(x).wrapping_mul(black_box(x));
        })
    });
}

criterion_group!(
    benches,
    bench_naive_mul,
    bench_mont_mul,
    bench_mont_conv,
    bench_mont_sqr,
    bench_add_mod,
    bench_sub_mod,
    bench_mont_inv,
    bench_u64_mul,
);
criterion_main!(benches);
