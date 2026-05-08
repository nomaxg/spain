use std::ops::Div;

use ndarray::{Array, IxDyn};

use crate::r1cs::R1CSExec;

pub const N: usize = 4;
pub const DT: f64 = 0.1;
pub const DIFF: f64 = 0.0001;
pub const VISC: f64 = 0.0001;
pub const DIFFUSION_ITERS: usize = 20;
pub const PROJECT_ITERS: usize = 50;
pub const ADVECT_RADIUS: usize = 1;
pub const STIR_RADIUS: usize = 1;
pub const STIR_STRENGTH: f64 = 10.0;
pub const STIR_DYE_STRENGTH: f64 = 100.0;

#[derive(Clone)]
pub struct FluidState<T: Clone> {
    pub u: Array<T, IxDyn>,
    pub v: Array<T, IxDyn>,
    pub u_prev: Array<T, IxDyn>,
    pub v_prev: Array<T, IxDyn>,
    pub p: Array<T, IxDyn>,
    pub div: Array<T, IxDyn>,
    pub dye: Array<T, IxDyn>,
    pub dye_prev: Array<T, IxDyn>,
}

pub fn flatten_inputs(state: &FluidState<f64>) -> Vec<Array<f64, IxDyn>> {
    vec![
        state.u.clone(),
        state.v.clone(),
        state.u_prev.clone(),
        state.v_prev.clone(),
        state.p.clone(),
        state.div.clone(),
        state.dye.clone(),
        state.dye_prev.clone(),
    ]
}

pub fn initial_state() -> FluidState<f64> {
    let zeros = || Array::from_shape_fn(IxDyn(&[N, N]), |_| 0.0);
    let mut state = FluidState {
        u: zeros(),
        v: zeros(),
        u_prev: zeros(),
        v_prev: zeros(),
        p: zeros(),
        div: zeros(),
        dye: zeros(),
        dye_prev: zeros(),
    };

    fn clamped_range(center: usize) -> std::ops::Range<usize> {
        let start = center.saturating_sub(5);
        let end = (center + 5).min(N);
        start..end
    }

    let px = N / 3;
    let py = N / 3;
    for i in clamped_range(px) {
        for j in clamped_range(py) {
            state.dye[[i, j]] = 100.0;
            state.u[[i, j]] = 2.0;
            state.v[[i, j]] = 1.0;
        }
    }

    let qx = 2 * N / 3;
    let qy = 2 * N / 3;
    for i in clamped_range(qx) {
        for j in clamped_range(qy) {
            state.dye[[i, j]] = 100.0;
            state.u[[i, j]] = 2.0;
            state.v[[i, j]] = -1.0;
        }
    }

    state
}

fn clone_cell<T: Clone>(tensor: &Array<T, IxDyn>, idx: [usize; 2]) -> T {
    tensor.get(idx).expect("index out of bounds").clone()
}

fn set_cell_value<T>(tensor: &mut Array<T, IxDyn>, idx: [usize; 2], value: T) {
    *tensor.get_mut(idx).expect("index out of bounds") = value;
}

fn clamp<E: R1CSExec>(exec: &mut E, atom: E::Atom, lower: f64, upper: f64) -> E::Atom {
    let lowered = exec.max(atom, lower);
    exec.min(lowered, upper)
}

fn linear_weight<E: R1CSExec>(exec: &mut E, coord: E::Atom, sample_idx: f64) -> E::Atom {
    let delta = coord - sample_idx;
    let abs_delta = exec.abs(delta);
    let one_minus_abs = abs_delta * -1.0 + 1.0;
    exec.max(one_minus_abs, 0.0)
}

fn add_source_ops<E: R1CSExec>(
    exec: &mut E,
    x: &mut Array<E::Atom, IxDyn>,
    s: &Array<E::Atom, IxDyn>,
) {
    for i in 0..N {
        for j in 0..N {
            let scaled = clone_cell(s, [i, j]) * DT;
            let updated = exec.add(clone_cell(x, [i, j]), scaled);
            set_cell_value(x, [i, j], updated);
        }
    }
}

fn diffuse_ops<E: R1CSExec>(
    exec: &mut E,
    x: &mut Array<E::Atom, IxDyn>,
    x0: &Array<E::Atom, IxDyn>,
    diff: f64,
) {
    let a = DT * diff * (N * N) as f64;
    let inv_denom = 1.0 / (1.0 + 4.0 * a);

    for _ in 0..DIFFUSION_ITERS {
        let x_old = x.clone();
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                let left_right = exec.add(
                    clone_cell(&x_old, [i + 1, j]),
                    clone_cell(&x_old, [i - 1, j]),
                );
                let up_down = exec.add(
                    clone_cell(&x_old, [i, j + 1]),
                    clone_cell(&x_old, [i, j - 1]),
                );
                let neighbors = exec.add(left_right, up_down);
                let scaled_neighbors = neighbors * a;
                let numerator = exec.add(clone_cell(x0, [i, j]), scaled_neighbors);
                let updated = numerator * inv_denom;
                set_cell_value(x, [i, j], updated);
            }
        }
    }
}

fn advect_alt_ops<E: R1CSExec>(
    exec: &mut E,
    d: &mut Array<E::Atom, IxDyn>,
    d0: &Array<E::Atom, IxDyn>,
    u: &Array<E::Atom, IxDyn>,
    v: &Array<E::Atom, IxDyn>,
) {
    let dt0 = DT * N as f64;
    let low = ADVECT_RADIUS as f64 - 0.5;
    let high = N as f64 - ADVECT_RADIUS as f64 - 0.5;

    for i in ADVECT_RADIUS..(N - ADVECT_RADIUS) {
        for j in ADVECT_RADIUS..(N - ADVECT_RADIUS) {
            let x = clone_cell(u, [i, j]) * -dt0 + i as f64;
            let y = clone_cell(v, [i, j]) * -dt0 + j as f64;

            let x = clamp(exec, x, low, high);
            let y = clamp(exec, y, low, high);

            let mut val = E::constant(0.0);
            for di in -(ADVECT_RADIUS as isize)..=(ADVECT_RADIUS as isize) {
                let sx = linear_weight(exec, x.clone(), i as f64 + di as f64);
                for dj in -(ADVECT_RADIUS as isize)..=(ADVECT_RADIUS as isize) {
                    let sy = linear_weight(exec, y.clone(), j as f64 + dj as f64);
                    let weight = exec.mul(sx.clone(), sy);
                    let sample =
                        clone_cell(d0, [(i as isize + di) as usize, (j as isize + dj) as usize]);
                    let weighted = exec.mul(weight, sample);
                    val = exec.add(val, weighted);
                }
            }

            set_cell_value(d, [i, j], val);
        }
    }
}

fn project_ops<E: R1CSExec>(
    exec: &mut E,
    u: &mut Array<E::Atom, IxDyn>,
    v: &mut Array<E::Atom, IxDyn>,
    p: &mut Array<E::Atom, IxDyn>,
    div_field: &mut Array<E::Atom, IxDyn>,
) {
    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            let du = clone_cell(u, [i + 1, j]) - clone_cell(u, [i - 1, j]);
            let dv = clone_cell(v, [i, j + 1]) - clone_cell(v, [i, j - 1]);
            let divergence = exec.add(du, dv);
            let scaled_divergence = divergence * (-0.5 / N as f64);
            set_cell_value(div_field, [i, j], scaled_divergence);
        }
    }

    for i in 0..N {
        for j in 0..N {
            set_cell_value(p, [i, j], E::constant(0.0));
        }
    }

    for _ in 0..PROJECT_ITERS {
        let p_old = p.clone();
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                let left_right = exec.add(
                    clone_cell(&p_old, [i + 1, j]),
                    clone_cell(&p_old, [i - 1, j]),
                );
                let up_down = exec.add(
                    clone_cell(&p_old, [i, j + 1]),
                    clone_cell(&p_old, [i, j - 1]),
                );
                let neighbors = exec.add(left_right, up_down);
                let numerator = exec.add(clone_cell(div_field, [i, j]), neighbors);
                let updated = numerator * 0.25;
                set_cell_value(p, [i, j], updated);
            }
        }
    }

    for i in 1..(N - 1) {
        for j in 1..(N - 1) {
            let p_x = clone_cell(p, [i + 1, j]) - clone_cell(p, [i - 1, j]);
            let p_y = clone_cell(p, [i, j + 1]) - clone_cell(p, [i, j - 1]);
            let u_delta = p_x * (0.5 * N as f64);
            let v_delta = p_y * (0.5 * N as f64);
            let u_new = clone_cell(u, [i, j]) - u_delta;
            let v_new = clone_cell(v, [i, j]) - v_delta;
            set_cell_value(u, [i, j], u_new);
            set_cell_value(v, [i, j], v_new);
        }
    }
}

fn stir_ops<E: R1CSExec>(
    exec: &mut E,
    u_prev: &mut Array<E::Atom, IxDyn>,
    v_prev: &mut Array<E::Atom, IxDyn>,
    dye_prev: &mut Array<E::Atom, IxDyn>,
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
                let idx = [i, j];

                add_source_constant(exec, u_prev, idx, -dy * falloff * STIR_STRENGTH);
                add_source_constant(exec, v_prev, idx, dx * falloff * STIR_STRENGTH);
                add_source_constant(exec, dye_prev, idx, falloff * STIR_DYE_STRENGTH);
            }
        }
    }
}

fn add_source_constant<E: R1CSExec>(
    exec: &mut E,
    tensor: &mut Array<E::Atom, IxDyn>,
    idx: [usize; 2],
    delta: f64,
) {
    let updated = exec.add(clone_cell(tensor, idx), E::constant(delta));
    set_cell_value(tensor, idx, updated);
}

pub fn add_input_state<E: R1CSExec>(exec: &mut E) -> FluidState<E::Atom> {
    FluidState {
        u: exec.add_tensor_input(&[N, N]),
        v: exec.add_tensor_input(&[N, N]),
        u_prev: exec.add_tensor_input(&[N, N]),
        v_prev: exec.add_tensor_input(&[N, N]),
        p: exec.add_tensor_input(&[N, N]),
        div: exec.add_tensor_input(&[N, N]),
        dye: exec.add_tensor_input(&[N, N]),
        dye_prev: exec.add_tensor_input(&[N, N]),
    }
}

pub fn step_ops<E: R1CSExec>(exec: &mut E, state: &mut FluidState<E::Atom>) {
    stir_ops(
        exec,
        &mut state.u_prev,
        &mut state.v_prev,
        &mut state.dye_prev,
    );

    add_source_ops(exec, &mut state.u, &state.u_prev);
    add_source_ops(exec, &mut state.v, &state.v_prev);
    add_source_ops(exec, &mut state.dye, &state.dye_prev);

    state.u_prev = state.u.clone();
    state.v_prev = state.v.clone();
    diffuse_ops(exec, &mut state.u, &state.u_prev, VISC);
    diffuse_ops(exec, &mut state.v, &state.v_prev, VISC);

    project_ops(
        exec,
        &mut state.u,
        &mut state.v,
        &mut state.p,
        &mut state.div,
    );

    state.u_prev = state.u.clone();
    state.v_prev = state.v.clone();
    advect_alt_ops(
        exec,
        &mut state.u,
        &state.u_prev,
        &state.u_prev,
        &state.v_prev,
    );
    advect_alt_ops(
        exec,
        &mut state.v,
        &state.v_prev,
        &state.u_prev,
        &state.v_prev,
    );

    project_ops(
        exec,
        &mut state.u,
        &mut state.v,
        &mut state.p,
        &mut state.div,
    );

    state.dye_prev = state.dye.clone();
    diffuse_ops(exec, &mut state.dye, &state.dye_prev, DIFF);
    state.dye_prev = state.dye.clone();
    advect_alt_ops(exec, &mut state.dye, &state.dye_prev, &state.u, &state.v);
}

pub fn simulate_ops<E: R1CSExec>(exec: &mut E, steps: usize) -> Array<E::Atom, IxDyn> {
    let mut state = add_input_state(exec);
    exec.add_tensor_output(&[N, N]);

    for _ in 0..steps {
        step_ops(exec, &mut state);
    }

    exec.tensor_output(&state.dye);

    state.dye
}
