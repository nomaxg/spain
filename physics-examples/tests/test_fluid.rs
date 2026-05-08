use std::ops::Div;

use ndarray::{Array, IxDyn};
use physics_examples::fluid::{
    ADVECT_RADIUS, DIFF, DIFFUSION_ITERS, DT, FluidState, N, PROJECT_ITERS, STIR_DYE_STRENGTH,
    STIR_RADIUS, STIR_STRENGTH, VISC, flatten_inputs, initial_state, simulate_ops,
};
use physics_examples::r1cs::{ConstraintGenerator, R1CSExec, WitnessGenerator, check_constraints};

const EPSILON: f64 = 1e-8;

fn add_source_reference(x: &mut Array<f64, IxDyn>, s: &Array<f64, IxDyn>) {
    for i in 0..N {
        for j in 0..N {
            x[[i, j]] += DT * s[[i, j]];
        }
    }
}

fn diffuse_reference(x: &mut Array<f64, IxDyn>, x0: &Array<f64, IxDyn>, diff: f64) {
    let a = DT * diff * (N * N) as f64;
    let inv_denom = 1.0 / (1.0 + 4.0 * a);

    for _ in 0..DIFFUSION_ITERS {
        let x_old = x.clone();
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                let neighbors =
                    x_old[[i + 1, j]] + x_old[[i - 1, j]] + x_old[[i, j + 1]] + x_old[[i, j - 1]];
                x[[i, j]] = (x0[[i, j]] + a * neighbors) * inv_denom;
            }
        }
    }
}

fn advect_alt_reference(
    d: &mut Array<f64, IxDyn>,
    d0: &Array<f64, IxDyn>,
    u: &Array<f64, IxDyn>,
    v: &Array<f64, IxDyn>,
) {
    let dt0 = DT * N as f64;
    let low = ADVECT_RADIUS as f64 - 0.5;
    let high = N as f64 - ADVECT_RADIUS as f64 - 0.5;

    for i in ADVECT_RADIUS..(N - ADVECT_RADIUS) {
        for j in ADVECT_RADIUS..(N - ADVECT_RADIUS) {
            let x = (i as f64 - dt0 * u[[i, j]]).clamp(low, high);
            let y = (j as f64 - dt0 * v[[i, j]]).clamp(low, high);

            let mut val = 0.0;
            for di in -(ADVECT_RADIUS as isize)..=(ADVECT_RADIUS as isize) {
                let sx = (1.0 - (x - (i as f64 + di as f64)).abs()).max(0.0);
                for dj in -(ADVECT_RADIUS as isize)..=(ADVECT_RADIUS as isize) {
                    let sy = (1.0 - (y - (j as f64 + dj as f64)).abs()).max(0.0);
                    val += sx * sy * d0[[(i as isize + di) as usize, (j as isize + dj) as usize]];
                }
            }

            d[[i, j]] = val;
        }
    }
}

fn project_reference(
    u: &mut Array<f64, IxDyn>,
    v: &mut Array<f64, IxDyn>,
    p: &mut Array<f64, IxDyn>,
    div_field: &mut Array<f64, IxDyn>,
) {
    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            div_field[[i, j]] =
                -0.5 * (u[[i + 1, j]] - u[[i - 1, j]] + v[[i, j + 1]] - v[[i, j - 1]]) / N as f64;
        }
    }

    p.fill(0.0);

    for _ in 0..PROJECT_ITERS {
        let p_old = p.clone();
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                p[[i, j]] = (div_field[[i, j]]
                    + p_old[[i + 1, j]]
                    + p_old[[i - 1, j]]
                    + p_old[[i, j + 1]]
                    + p_old[[i, j - 1]])
                    / 4.0;
            }
        }
    }

    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            u[[i, j]] -= 0.5 * N as f64 * (p[[i + 1, j]] - p[[i - 1, j]]);
            v[[i, j]] -= 0.5 * N as f64 * (p[[i, j + 1]] - p[[i, j - 1]]);
        }
    }
}

fn stir_reference(
    u_prev: &mut Array<f64, IxDyn>,
    v_prev: &mut Array<f64, IxDyn>,
    dye_prev: &mut Array<f64, IxDyn>,
) {
    let cx = N.div(2);
    let cy = N.div(2);

    for i in cx - STIR_RADIUS..cx + STIR_RADIUS {
        for j in cy - STIR_RADIUS..cy + STIR_RADIUS {
            let dx = i as f64 - cx as f64;
            let dy = j as f64 - cy as f64;
            let dist2 = dx * dx + dy * dy;
            let r_squared = (STIR_RADIUS * STIR_RADIUS) as f64;
            if dist2 < r_squared {
                let falloff = f64::exp(-dist2 / r_squared);
                u_prev[[i, j]] += -dy * falloff * STIR_STRENGTH;
                v_prev[[i, j]] += dx * falloff * STIR_STRENGTH;
                dye_prev[[i, j]] += falloff * STIR_DYE_STRENGTH;
            }
        }
    }
}

fn step_reference(state: &mut FluidState<f64>) {
    stir_reference(&mut state.u_prev, &mut state.v_prev, &mut state.dye_prev);

    add_source_reference(&mut state.u, &state.u_prev);
    add_source_reference(&mut state.v, &state.v_prev);
    add_source_reference(&mut state.dye, &state.dye_prev);

    state.u_prev = state.u.clone();
    state.v_prev = state.v.clone();
    diffuse_reference(&mut state.u, &state.u_prev, VISC);
    diffuse_reference(&mut state.v, &state.v_prev, VISC);

    project_reference(&mut state.u, &mut state.v, &mut state.p, &mut state.div);

    state.u_prev = state.u.clone();
    state.v_prev = state.v.clone();
    advect_alt_reference(&mut state.u, &state.u_prev, &state.u_prev, &state.v_prev);
    advect_alt_reference(&mut state.v, &state.v_prev, &state.u_prev, &state.v_prev);

    project_reference(&mut state.u, &mut state.v, &mut state.p, &mut state.div);

    state.dye_prev = state.dye.clone();
    diffuse_reference(&mut state.dye, &state.dye_prev, DIFF);
    state.dye_prev = state.dye.clone();
    advect_alt_reference(&mut state.dye, &state.dye_prev, &state.u, &state.v);
}

// For N=4, r=1. Derived from fluid.py
fn known_answer_dye() -> Array<f64, IxDyn> {
    Array::from_shape_vec(
        IxDyn(&[N, N]),
        vec![
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.00003072,
            100.00073547,
            100.0,
            100.0,
            100.76562586,
            101.17540956,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )
    .expect("known answer dye shape should be valid")
}

fn assert_array_close(actual: &Array<f64, IxDyn>, expected: &Array<f64, IxDyn>, label: &str) {
    for (idx, expected_val) in expected.indexed_iter() {
        let actual_val = actual[idx.clone()];
        assert!(
            (actual_val - expected_val).abs() < EPSILON,
            "{label} mismatch at {:?}: {} != {}",
            idx,
            actual_val,
            expected_val
        );
    }
}

fn eval_array(
    symbolic: &Array<physics_examples::builder::LC, IxDyn>,
    witness: &[f64],
) -> Array<f64, IxDyn> {
    Array::from_shape_fn(symbolic.raw_dim(), |idx| symbolic[idx].eval(witness))
}

fn assert_witness_output_matches(witness: &[f64], out_start: usize, expected: &Array<f64, IxDyn>) {
    let expected_flat = expected.clone().into_raw_vec_and_offset().0;
    for (idx, expected_val) in expected_flat.iter().enumerate() {
        let actual_val = witness[out_start + idx];
        assert!(
            (actual_val - expected_val).abs() < EPSILON,
            "output witness mismatch at flat index {}: {} != {}",
            idx,
            actual_val,
            expected_val
        );
    }
}

#[test]
fn step_consistency() {
    let mut expected = initial_state();
    let start_tmie = std::time::Instant::now();
    step_reference(&mut expected);
    let elapsed = start_tmie.elapsed();
    println!("Reference fluid step took {:.2?}", elapsed);
    let expected_output = expected.dye.clone();

    let inputs = flatten_inputs(&initial_state());

    let start_time = std::time::Instant::now();
    let mut exp_gen = WitnessGenerator::new_from_tensored_inputs(inputs.clone());
    let out_start = exp_gen.inputs.len() + 1;
    let actual = simulate_ops(&mut exp_gen, 1);
    let witness = exp_gen.finish();
    let elapsed = start_time.elapsed();
    println!("Witness generation took {:.2?}", elapsed);

    let start_time = std::time::Instant::now();
    let mut cons_gen = ConstraintGenerator::new();
    let symbolic = simulate_ops(&mut cons_gen, 1);
    let constraints = cons_gen.finish();
    let elapsed = start_time.elapsed();
    println!("Constraint generation took {:.2?}", elapsed);

    let start_time = std::time::Instant::now();
    check_constraints(&constraints, &witness);
    let elapsed = start_time.elapsed();
    println!(
        "Constraint checking took {:.2?} for {} constraints",
        elapsed,
        constraints.constraints.len()
    );

    assert_witness_output_matches(&witness, out_start, &expected_output);
    assert_array_close(&actual, &expected.dye, "dye");
    assert_array_close(
        &eval_array(&symbolic, &witness),
        &expected.dye,
        "symbolic dye",
    );
}

#[test]
#[ignore]
fn known_answer_test() {
    let mut expected = initial_state();
    step_reference(&mut expected);

    let known_answer = known_answer_dye();
    assert_array_close(&expected.dye, &known_answer, "reference known-answer dye");

    let mut exp_gen = WitnessGenerator::new_from_tensored_inputs(flatten_inputs(&initial_state()));
    let actual = simulate_ops(&mut exp_gen, 1);

    assert_array_close(&actual, &known_answer, "witness known-answer dye");
}
