#![feature(generic_const_exprs)]
use criterion::{criterion_group, criterion_main, Criterion};
use dark::{
    arkbigexp::{ArkFoldSplitExp, ArkSplitExp},
    arkgroup::ArkRsa512,
    bigexp::{CombExp, IntExp, NaiveExp, SplitExp},
    rsagroup::RSAGroup,
    DARK,
};
//use ff::{poly::int::MLE};
use dark::bigrsa::{Mont, MontInt};
use rug::{ops::Pow, Integer};
// seeded randomness
use rand::{rngs::StdRng, Rng, SeedableRng};

// bench commitment to an MLE with n variables
// fn bench_comm(_c: &mut Criterion) {
//     // make group for these benchmarks
//     // and set number of samples to 10
//     //c.benchmark_group("commitment to MLE");
//     // loop over number of variables
//     for num_vars in 2..=20 {
//         // setup dark and polynomial
//         let (poly, dark) = setup_dark(num_vars);
//         // run the benchmark exactly once don't repeat or do lots of samples
//         // don't even invoke criterion, just time start and end
//         let start = std::time::Instant::now();
//         let comm = dark.commit(poly.clone());
//         let elapsed = start.elapsed();
//         println!(
//             "Commitment to MLE with {} vars took: {:?}",
//             num_vars, elapsed
//         );
//         /*c.bench_function(&format!("commitment to MLE with {} vars", num_vars), |b| {
//             b.iter(|| {
//                 black_box(dark.commit(poly.clone()));
//             })
//         })
//         .sample_size(10);*/
//     }
// }
//
// generate random u64 vector of given length
fn rand_u64_vec(len: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..len).map(|_| rng.gen::<u64>()).collect()
}

fn bench_int_exp(_c: &mut Criterion) {
    let exp = IntExp::new();
    for i in [10, 12, 14, 16, 18, 20] {
        let v = rand_u64_vec(1 << i);
        let start = std::time::Instant::now();
        let _res = exp.exp(&v);
        let elapsed = start.elapsed();
        println!("IntExp with 2^{} u64s:\n Online | {:?}", i, elapsed);
    }
}

fn bench_naive_exp(_c: &mut Criterion) {
    let exp = NaiveExp::new();
    for i in [24] {
        let v = rand_u64_vec(1 << i);
        let start = std::time::Instant::now();
        let _res = exp.exp(&v);
        let elapsed = start.elapsed();
        println!("Naive exp with 2^{} u64s:\n Online | {:?}", i, elapsed);
    }
}

fn bench_comb_exp(_c: &mut Criterion) {
    let outer_len = 1;
    for i in [24] {
        let exp = CombExp::new(1 << i, outer_len);
        let v = rand_u64_vec(1 << i);
        let start = std::time::Instant::now();
        let _res = exp.exp(&v);
        let elapsed = start.elapsed();
        println!("Comb exp with 2^{} u64s took: {:?}", i, elapsed);
    }
}

fn bench_split_exp(_c: &mut Criterion) {
    let splits = 16;
    for i in [10, 12, 14, 16, 18] {
        println!("SplitExp({}) with 2^{} u64s:", splits, i);
        let start = std::time::Instant::now();
        let exp = SplitExp::new(1 << i, splits);
        let elapsed_setup = start.elapsed();
        println!(" Setup  | {:?}", elapsed_setup);
        let v = rand_u64_vec(1 << i);
        let start = std::time::Instant::now();
        let _res = exp.exp(&v);
        let elapsed = start.elapsed();
        println!(" Online | {:?}", elapsed);
    }
}

fn bench_ark_split_exp(_c: &mut Criterion) {
    let splits = 16;
    for i in [16, 18] {
        println!("ARK SplitExp({}) with 2^{} u64s:", splits, i);
        let start = std::time::Instant::now();
        let exp = ArkSplitExp::new(1 << i, splits);
        let elapsed_setup = start.elapsed();
        println!(" Setup  | {:?}", elapsed_setup);
        let v = rand_u64_vec(1 << i);
        let start = std::time::Instant::now();
        let _res = exp.exp(&v);
        let elapsed = start.elapsed();
        println!(" Online | {:?}", elapsed);
    }
}

fn bench_ark_fold_split_exp(_c: &mut Criterion) {
    let splits = 16;
    //let folds = 3;
    for i in [16, 18] {
        for folds in 1..5 {
            println!(
                "ARK ArkFoldSplitExp({},{}) with 2^{} u64s:",
                splits, folds, i
            );
            let start = std::time::Instant::now();
            let exp = ArkFoldSplitExp::new(1 << i, splits, folds);
            let elapsed_setup = start.elapsed();
            println!(" Setup  | {:?}", elapsed_setup);
            let v = rand_u64_vec(1 << i);
            let start = std::time::Instant::now();
            let _res = exp.exp(&v);
            let elapsed = start.elapsed();
            println!(" Online | {:?}", elapsed);
        }
    }
}

// compare performance of ArkRsa512 and RSAGroup for a series of 100 modular multiplications
fn bench_mod_mul(c: &mut Criterion) {
    let mut group_a = RSAGroup::generator();
    let mut group_b = ArkRsa512::from(2u64);
    let muls = 100;
    c.bench_function("RSAGroup modular multiplications", |b| {
        b.iter(|| {
            for _ in 0..muls {
                group_a = group_a.clone() * group_a.clone();
            }
        })
    });
    c.bench_function("ArkRsa512 modular multiplications", |b| {
        b.iter(|| {
            for _ in 0..muls {
                group_b = group_b * group_b;
            }
        })
    });
}

fn setup<const N: usize>() -> (StdRng, Mont<N>, Integer)
where
    [(); 2 * N]:,
{
    // init seeded rng
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    // set up modulus
    let mut m = Integer::from(0);
    for _ in 0..N {
        m <<= 64;
        let limb: u64 = rng.random();
        m += limb;
    }
    // ensure odd and high bit clear
    m.set_bit(0, true);
    m.set_bit(64 * N as u32 - 1, false);
    // create montgomery context
    (rng, Mont::new(m.clone()), m)
}

// biased rand, but its fine.
fn rand_element<const N: usize, R: Rng>(rng: &mut R, m: &Integer) -> Integer {
    let mut res = Integer::from(0);
    for _ in 0..N {
        res <<= 64;
        res += rng.random::<u64>();
    }
    res %= m;
    res
}

// bench N limb arithmetic
fn bench_limb<const N: usize>(c: &mut Criterion)
where
    [(); 2 * N]:,
{
    // setup
    let (mut rng, mont, m) = setup::<N>();
    let mut a_int = rand_element::<N, _>(&mut rng, &m);
    let b_int = rand_element::<N, _>(&mut rng, &m);
    let mut a_mont = mont.to_montgomery(&a_int);
    let b_mont = mont.to_montgomery(&b_int);
    // benchmarks
    let name = format!("Rug mod mul N={}", N);
    c.bench_function(&name, |b| {
        b.iter(|| {
            a_int = (a_int.clone() * b_int.clone()) % m.clone();
        })
    });
    let name = format!("Custom mod mul N={}", N);
    c.bench_function(&name, |b| {
        b.iter(|| {
            mont.mul_assign(&mut a_mont, &b_mont);
        })
    });
    let name = format!("Rug mod square N={}", N);
    c.bench_function(&name, |b| {
        b.iter(|| {
            mont.square_in_place(&mut a_mont);
        })
    });
}

fn bench_arith(c: &mut Criterion) {
    // random N limb modulus
    bench_limb::<8>(c);
    bench_limb::<12>(c);
    bench_limb::<16>(c);
}

criterion_group!(
    benches,
    //bench_comm,
    //bench_int_exp,
    //bench_split_exp,
    // bench_ark_split_exp,
    //bench_ark_fold_split_exp, //bench_naive_exp,
    //bench_comb_exp
    //bench_mod_mul,
    bench_arith,
);
criterion_main!(benches);

/*
Stashing results from my machine
Commitment to MLE with 2 vars took: 587.394µs
Commitment to MLE with 3 vars took: 258.79µs
Commitment to MLE with 4 vars took: 684.214µs
Commitment to MLE with 5 vars took: 1.309788ms
Commitment to MLE with 6 vars took: 2.617935ms
Commitment to MLE with 7 vars took: 5.891522ms
Commitment to MLE with 8 vars took: 13.351032ms
Commitment to MLE with 9 vars took: 28.497986ms
Commitment to MLE with 10 vars took: 62.763989ms
Commitment to MLE with 11 vars took: 136.516096ms
Commitment to MLE with 12 vars took: 295.695094ms
Commitment to MLE with 13 vars took: 638.528998ms
Commitment to MLE with 14 vars took: 1.373376709s
Commitment to MLE with 15 vars took: 2.936755253s
Commitment to MLE with 16 vars took: 6.263402749s
Commitment to MLE with 17 vars took: 13.306546821s
Commitment to MLE with 18 vars took: 28.176753641s
Commitment to MLE with 19 vars took: 59.615412889s
Commitment to MLE with 20 vars took: 129.150567286s
*/

/* NG: results from my machine
Commitment to MLE with 2 vars took: 203.584µs
Commitment to MLE with 3 vars took: 115.625µs
Commitment to MLE with 4 vars took: 251µs
Commitment to MLE with 5 vars took: 520.667µs
Commitment to MLE with 6 vars took: 1.078167ms
Commitment to MLE with 7 vars took: 2.219291ms
Commitment to MLE with 8 vars took: 5.092708ms
Commitment to MLE with 9 vars took: 11.111625ms
Commitment to MLE with 10 vars took: 23.570959ms
Commitment to MLE with 11 vars took: 51.855625ms
Commitment to MLE with 12 vars took: 114.535125ms
Commitment to MLE with 13 vars took: 244.644834ms
Commitment to MLE with 14 vars took: 523.973666ms
Commitment to MLE with 15 vars took: 1.126083333s
Commitment to MLE with 16 vars took: 2.398435s
Commitment to MLE with 17 vars took: 5.109984209s
Commitment to MLE with 18 vars took: 10.809208417s
Commitment to MLE with 19 vars took: 23.188066583s
Commitment to MLE with 20 vars took: 49.719270209s
*/
