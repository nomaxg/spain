use ndarray::{Array, IxDyn};

use crate::r1cs::R1CSExec;

pub const DEFAULT_GRID_SIZE: usize = 8;
pub const DT: f64 = 0.1;
pub const DIFF: f64 = 0.0001;
pub const VISC: f64 = 0.0001;
pub const DIFFUSION_ITERS: usize = 20;
pub const PROJECT_ITERS: usize = 50;
pub const ADVECT_RADIUS: usize = 1;
pub const STIR_RADIUS: usize = 1;
pub const STIR_STRENGTH: f64 = 10.0;
pub const STIR_DYE_STRENGTH: f64 = 100.0;

#[derive(Clone, Debug)]
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

pub fn initial_state(grid_size: usize) -> FluidState<f64> {
    assert!(grid_size >= 2, "grid_size must be >= 2");

    let zeros = || Array::from_shape_fn(IxDyn(&[grid_size, grid_size]), |_| 0.0);
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

    fn clamped_range(center: usize, grid_size: usize) -> std::ops::Range<usize> {
        let start = center.saturating_sub(5);
        let end = (center + 5).min(grid_size);
        start..end
    }

    let px = grid_size / 3;
    let py = grid_size / 3;
    for i in clamped_range(px, grid_size) {
        for j in clamped_range(py, grid_size) {
            state.dye[[i, j]] = 100.0;
            state.u[[i, j]] = 2.0;
            state.v[[i, j]] = 1.0;
        }
    }

    let qx = 2 * grid_size / 3;
    let qy = 2 * grid_size / 3;
    for i in clamped_range(qx, grid_size) {
        for j in clamped_range(qy, grid_size) {
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
    grid_size: usize,
) {
    for i in 0..grid_size {
        for j in 0..grid_size {
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
    grid_size: usize,
) {
    let a = DT * diff * (grid_size * grid_size) as f64;
    let inv_denom = 1.0 / (1.0 + 4.0 * a);

    for _ in 0..DIFFUSION_ITERS {
        let x_old_prev = x.clone();
        let mut x_old = Array::from_elem(IxDyn(&[grid_size, grid_size]), E::constant(0.0));
        for i in 0..grid_size {
            for j in 0..grid_size {
                let old = clone_cell(&x_old_prev, [i, j]);
                let old = exec.condense(old);
                set_cell_value(&mut x_old, [i, j], old);
            }
        }
        for i in 1..(grid_size - 1) {
            for j in 1..(grid_size - 1) {
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
                let updated = exec.condense(numerator * inv_denom);
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
    grid_size: usize,
) {
    let dt0 = DT * grid_size as f64;
    let low = ADVECT_RADIUS as f64 - 0.5;
    let high = grid_size as f64 - ADVECT_RADIUS as f64 - 0.5;

    for i in ADVECT_RADIUS..(grid_size - ADVECT_RADIUS) {
        for j in ADVECT_RADIUS..(grid_size - ADVECT_RADIUS) {
            let x = exec.condense(clone_cell(u, [i, j]) * -dt0 + i as f64);
            let y = exec.condense(clone_cell(v, [i, j]) * -dt0 + j as f64);

            let x = clamp(exec, x, low, high);
            let y = clamp(exec, y, low, high);

            let mut val = E::constant(0.0);
            for di in -(ADVECT_RADIUS as isize)..=(ADVECT_RADIUS as isize) {
                let raw_sx = linear_weight(exec, x.clone(), i as f64 + di as f64);
                let sx = exec.condense(raw_sx);
                for dj in -(ADVECT_RADIUS as isize)..=(ADVECT_RADIUS as isize) {
                    let raw_sy = linear_weight(exec, y.clone(), j as f64 + dj as f64);
                    let sy = exec.condense(raw_sy);
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
    grid_size: usize,
) {
    for i in 1..(grid_size - 1) {
        for j in 1..(grid_size - 1) {
            let du = exec.sub(clone_cell(u, [i + 1, j]), clone_cell(u, [i - 1, j]));
            let dv = exec.sub(clone_cell(v, [i, j + 1]), clone_cell(v, [i, j - 1]));
            let divergence = exec.add(du, dv);
            let scaled_divergence = divergence * (-0.5 / grid_size as f64);
            let scaled_divergence = exec.condense(scaled_divergence);
            set_cell_value(div_field, [i, j], scaled_divergence);
        }
    }

    for i in 0..grid_size {
        for j in 0..grid_size {
            set_cell_value(p, [i, j], E::constant(0.0));
        }
    }

    for _ in 0..PROJECT_ITERS {
        let p_old = p.clone();
        for i in 1..(grid_size - 1) {
            for j in 1..(grid_size - 1) {
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
                let updated = exec.condense(updated);
                set_cell_value(p, [i, j], updated);
            }
        }
    }

    for i in 1..(grid_size - 1) {
        for j in 1..(grid_size - 1) {
            let p_x = exec.sub(clone_cell(p, [i + 1, j]), clone_cell(p, [i - 1, j]));
            let p_y = exec.sub(clone_cell(p, [i, j + 1]), clone_cell(p, [i, j - 1]));
            let u_delta = p_x * (0.5 * grid_size as f64);
            let v_delta = p_y * (0.5 * grid_size as f64);
            let u_new = exec.sub(clone_cell(u, [i, j]), u_delta);
            let v_new = exec.sub(clone_cell(v, [i, j]), v_delta);
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
    grid_size: usize,
) {
    let cx = grid_size / 2;
    let cy = grid_size / 2;

    for i in cx.saturating_sub(STIR_RADIUS)..(cx + STIR_RADIUS).min(grid_size) {
        for j in cy.saturating_sub(STIR_RADIUS)..(cy + STIR_RADIUS).min(grid_size) {
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
    _exec: &mut E,
    tensor: &mut Array<E::Atom, IxDyn>,
    idx: [usize; 2],
    delta: f64,
) {
    let updated = clone_cell(tensor, idx) + delta;
    set_cell_value(tensor, idx, updated);
}

pub fn add_input_state<E: R1CSExec>(exec: &mut E) -> FluidState<E::Atom> {
    add_input_state_with_grid(exec, DEFAULT_GRID_SIZE)
}

pub fn add_input_state_with_grid<E: R1CSExec>(
    exec: &mut E,
    grid_size: usize,
) -> FluidState<E::Atom> {
    assert!(grid_size >= 2, "grid_size must be >= 2");
    FluidState {
        u: exec.add_tensor_input(&[grid_size, grid_size]),
        v: exec.add_tensor_input(&[grid_size, grid_size]),
        u_prev: exec.add_tensor_input(&[grid_size, grid_size]),
        v_prev: exec.add_tensor_input(&[grid_size, grid_size]),
        p: exec.add_tensor_input(&[grid_size, grid_size]),
        div: exec.add_tensor_input(&[grid_size, grid_size]),
        dye: exec.add_tensor_input(&[grid_size, grid_size]),
        dye_prev: exec.add_tensor_input(&[grid_size, grid_size]),
    }
}

pub fn step_ops<E: R1CSExec>(exec: &mut E, state: &mut FluidState<E::Atom>, grid_size: usize) {
    stir_ops(
        exec,
        &mut state.u_prev,
        &mut state.v_prev,
        &mut state.dye_prev,
        grid_size,
    );

    add_source_ops(exec, &mut state.u, &state.u_prev, grid_size);
    add_source_ops(exec, &mut state.v, &state.v_prev, grid_size);
    add_source_ops(exec, &mut state.dye, &state.dye_prev, grid_size);

    state.u_prev = state.u.clone();
    state.v_prev = state.v.clone();
    diffuse_ops(exec, &mut state.u, &state.u_prev, VISC, grid_size);
    diffuse_ops(exec, &mut state.v, &state.v_prev, VISC, grid_size);

    project_ops(
        exec,
        &mut state.u,
        &mut state.v,
        &mut state.p,
        &mut state.div,
        grid_size,
    );

    state.u_prev = state.u.clone();
    state.v_prev = state.v.clone();
    advect_alt_ops(
        exec,
        &mut state.u,
        &state.u_prev,
        &state.u_prev,
        &state.v_prev,
        grid_size,
    );
    advect_alt_ops(
        exec,
        &mut state.v,
        &state.v_prev,
        &state.u_prev,
        &state.v_prev,
        grid_size,
    );

    project_ops(
        exec,
        &mut state.u,
        &mut state.v,
        &mut state.p,
        &mut state.div,
        grid_size,
    );

    state.dye_prev = state.dye.clone();
    diffuse_ops(exec, &mut state.dye, &state.dye_prev, DIFF, grid_size);
    state.dye_prev = state.dye.clone();
    advect_alt_ops(
        exec,
        &mut state.dye,
        &state.dye_prev,
        &state.u,
        &state.v,
        grid_size,
    );
}

pub fn simulate_ops<E: R1CSExec>(
    exec: &mut E,
    steps: usize,
    grid_size: usize,
) -> Array<E::Atom, IxDyn> {
    simulate_ops_with_progress(exec, steps, grid_size, |_, _| {})
}

pub fn simulate_ops_with_progress<E: R1CSExec, F: FnMut(usize, usize)>(
    exec: &mut E,
    steps: usize,
    grid_size: usize,
    mut on_step: F,
) -> Array<E::Atom, IxDyn> {
    let mut state = add_input_state_with_grid(exec, grid_size);
    exec.add_tensor_output(&[grid_size, grid_size]);

    for step_idx in 0..steps {
        on_step(step_idx + 1, steps);
        step_ops(exec, &mut state, grid_size);
    }

    exec.tensor_output(&state.dye);

    state.dye
}
