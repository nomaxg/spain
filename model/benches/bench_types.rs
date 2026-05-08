#![feature(f128)]
use criterion::{Criterion, criterion_group, criterion_main};
use model::AFloat;
use num_traits::FromPrimitive;
use std::hint::black_box;
use twofloat::TwoFloat;

fn bench_afloat_mul(c: &mut Criterion) {
    let x = AFloat::from_f32(1.3).unwrap();
    let y = AFloat::from_f32(2.4).unwrap();
    let mut z = AFloat::from_f32(0.).unwrap();
    c.bench_function("bench_AFloat_mul", |b| {
        b.iter(|| {
            z = black_box(x.clone()) * black_box(y.clone());
        });
    });
}

fn bench_f32_mul(c: &mut Criterion) {
    let x = f32::from_f32(1.3).unwrap();
    let y = f32::from_f32(2.4).unwrap();
    let mut z = f32::from_f32(0.).unwrap();
    c.bench_function("bench_f32_mul", |b| {
        b.iter(|| {
            z = black_box(x) * black_box(y);
        });
    });
}

fn bench_f64_mul(c: &mut Criterion) {
    let x = f64::from_f32(1.3).unwrap();
    let y = f64::from_f32(2.4).unwrap();
    let mut z = f64::from_f32(0.).unwrap();
    c.bench_function("bench_f64_mul", |b| {
        b.iter(|| {
            z = black_box(x) * black_box(y);
        });
    });
}

fn bench_f128_mul(c: &mut Criterion) {
    let x = 1.3_f128;
    let y = 1.3_f128;
    let mut z = 0_f128;
    c.bench_function("bench_f128_mul", |b| {
        b.iter(|| {
            z = black_box(x) * black_box(y);
        });
    });
}

fn bench_twofloat_mul(c: &mut Criterion) {
    let x = TwoFloat::from_f32(1.3).unwrap();
    let y = TwoFloat::from_f32(2.4).unwrap();
    let mut z = TwoFloat::from_f32(0.).unwrap();
    c.bench_function("bench_TwoFloat_mul", |b| {
        b.iter(|| {
            z = black_box(x) * black_box(y);
        });
    });
}

criterion_group!(
    benches,
    bench_afloat_mul,
    bench_f32_mul,
    bench_f64_mul,
    bench_f128_mul,
    bench_twofloat_mul
);
criterion_main!(benches);
