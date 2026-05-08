#![feature(f128)]
use criterion::{Criterion, criterion_group, criterion_main};
use model::*;
use num_traits::ToPrimitive;
use std::hint::black_box;
use tract_onnx::prelude::*;

fn bench_tract_onnx_mul(c: &mut Criterion) {
    let path = "data/mul.onnx";
    let runner = OnnxRunner::<TFloat>::read(path).unwrap();
    let input = runner.rand_input();
    let model = onnx()
        .model_for_path(path)
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    let mut input_tract = tvec![];
    for outlet in model.model().input_outlets().unwrap() {
        let name = model.model().node(outlet.node).name.clone();
        let tensor = input.get(&name).expect("missing input");
        input_tract.push(
            tensor
                .data
                .clone()
                .unwrap()
                .mapv(|v| v.to_f32().unwrap()) // manual type conversion here
                .into_tvalue(),
        );
    }

    c.bench_function("bench_tract_onnx_mul", |b| {
        b.iter(|| {
            let _ = black_box(model.clone()).run(black_box(input_tract.clone()));
        });
    });
}

fn bench_tract_onnx_primary(c: &mut Criterion) {
    let path = "data/primary_model.onnx";
    let runner = OnnxRunner::<TFloat>::read(path).unwrap();
    let input = runner.rand_input();
    let model = onnx()
        .model_for_path(path)
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    let mut input_tract = tvec![];
    for outlet in model.model().input_outlets().unwrap() {
        let name = model.model().node(outlet.node).name.clone();
        let tensor = input.get(&name).expect("missing input");
        input_tract.push(
            tensor
                .data
                .clone()
                .unwrap()
                .mapv(|v| v.to_f64().unwrap()) // manual type conversion here
                .into_tvalue(),
        );
    }

    c.bench_function("bench_tract_onnx_primary", |b| {
        b.iter(|| {
            let _ = black_box(model.clone()).run(black_box(input_tract.clone()));
        });
    });
}

fn bench_my_onnx_mul_tfloat(c: &mut Criterion) {
    let path = "data/mul.onnx";
    let mut runner = OnnxRunner::<TFloat>::read(path).unwrap();
    let input = runner.rand_input();

    c.bench_function("bench_my_onnx_mul_tfloat", |b| {
        b.iter(|| {
            let _ = runner.run(black_box(input.clone()));
        });
    });
}

fn bench_my_onnx_mul_afloat(c: &mut Criterion) {
    let path = "data/mul.onnx";
    let mut runner = OnnxRunner::<AFloat>::read(path).unwrap();
    let input = runner.rand_input();

    c.bench_function("bench_my_onnx_mul_afloat", |b| {
        b.iter(|| {
            let _ = runner.run(black_box(input.clone()));
        });
    });
}

fn bench_my_onnx_mul_f64(c: &mut Criterion) {
    let path = "data/mul.onnx";
    let mut runner = OnnxRunner::<f64>::read(path).unwrap();
    let input = runner.rand_input();

    c.bench_function("bench_my_onnx_mul_f64", |b| {
        b.iter(|| {
            let _ = runner.run(black_box(input.clone()));
        });
    });
}

fn bench_my_onnx_primary_tfloat(c: &mut Criterion) {
    let path = "data/primary_model.onnx";
    let mut runner = OnnxRunner::<TFloat>::read(path).unwrap();
    let input = runner.rand_input();

    c.bench_function("bench_my_onnx_primary_tfloat", |b| {
        b.iter(|| {
            let _ = runner.run(black_box(input.clone()));
        });
    });
}

fn bench_my_onnx_primary_afloat(c: &mut Criterion) {
    let path = "data/primary_model.onnx";
    let mut runner = OnnxRunner::<AFloat>::read(path).unwrap();
    let input = runner.rand_input();

    c.bench_function("bench_my_onnx_primary_afloat", |b| {
        b.iter(|| {
            let _ = runner.run(black_box(input.clone()));
        });
    });
}

fn bench_my_onnx_primary_f64(c: &mut Criterion) {
    let path = "data/primary_model.onnx";
    let mut runner = OnnxRunner::<f64>::read(path).unwrap();
    let input = runner.rand_input();

    c.bench_function("bench_my_onnx_primary_f64", |b| {
        b.iter(|| {
            let _ = runner.run(black_box(input.clone()));
        });
    });
}

criterion_group!(
    benches,
    bench_tract_onnx_mul,
    bench_tract_onnx_primary,
    bench_my_onnx_mul_afloat,
    bench_my_onnx_mul_tfloat,
    bench_my_onnx_mul_f64,
    bench_my_onnx_primary_afloat,
    bench_my_onnx_primary_tfloat,
    bench_my_onnx_primary_f64
);
criterion_main!(benches);
