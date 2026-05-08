use std::{
    collections::VecDeque,
    ops::{Add, Mul, Sub},
};

use model::AFloat;
use ndarray::{Array, IxDyn};
use parse::generalized::HighPrecisionInt;
use parse::mat::{Matrix, MatrixData};
use spain::inputs::Metadata;
use stream::bigvec::BigVec;

use crate::builder::{LC, R1CSBuilder, Var};

pub trait R1CSExec {
    type Atom: Clone
        + Add<Self::Atom, Output = Self::Atom>
        + Add<f64, Output = Self::Atom>
        + Sub<Self::Atom, Output = Self::Atom>
        + Sub<f64, Output = Self::Atom>
        + Mul<f64, Output = Self::Atom>;
    type Output;

    // outputs/inputs/constants
    fn add_input(&mut self) -> Self::Atom;

    fn add_tensor_input(&mut self, shape: &[usize]) -> Array<Self::Atom, IxDyn> {
        Array::from_shape_fn(IxDyn(shape), |_| self.add_input())
    }

    fn add_output(&mut self);

    fn add_tensor_output(&mut self, shape: &[usize]) {
        let total_size = shape.iter().product();
        for _ in 0..total_size {
            self.add_output();
        }
    }

    fn constant(c: f64) -> Self::Atom;

    // Ops
    fn add(&mut self, lhs: Self::Atom, rhs: Self::Atom) -> Self::Atom;

    fn mul(&mut self, lhs: Self::Atom, rhs: Self::Atom) -> Self::Atom;

    fn max(&mut self, lhs: Self::Atom, c: f64) -> Self::Atom;

    fn min(&mut self, lhs: Self::Atom, c: f64) -> Self::Atom;

    fn abs(&mut self, x: Self::Atom) -> Self::Atom;

    fn tensor_output(&mut self, tensor: &Array<Self::Atom, IxDyn>) {
        for atom in tensor.iter() {
            self.output(atom.clone());
        }
    }

    fn output(&mut self, atom: Self::Atom);

    // Returns the "trace" of whatever exec context we are in, either a witness or
    // set of constraints
    fn finish(self) -> Self::Output;
}
#[derive(Clone, Debug)]
pub struct WitnessGenerator {
    pub inputs: Vec<f64>,
    out_idx: usize,
    next_input: usize,
    pub witness: Vec<f64>,
}

impl WitnessGenerator {
    pub fn new(inputs: Vec<f64>) -> Self {
        Self {
            inputs,
            out_idx: 0,
            next_input: 0,
            witness: vec![1f64],
        }
    }
    pub fn new_from_tensored_inputs(inputs: Vec<Array<f64, IxDyn>>) -> Self {
        let flattened_inputs = inputs
            .into_iter()
            .flat_map(|arr| arr.into_raw_vec_and_offset().0)
            .collect();
        Self::new(flattened_inputs)
    }

    pub fn metadata(&self) -> Metadata {
        let num_inputs_and_outputs = self.out_idx - 1;
        let num_witness_values = self.witness.len() - num_inputs_and_outputs;
        Metadata {
            num_public_values: num_inputs_and_outputs,
            num_random_values: 0,
            num_witness_values,
            num_secondary_witness_values: 0,
            num_secondary_constraint_variables: 0,
            primary_output_labels: Vec::new(),
            secondary_output_labels: Vec::new(),
        }
    }

    pub fn witness_int<T: HighPrecisionInt>(&self, scale_factor: AFloat) -> Matrix<T> {
        let witness_matrix = Matrix::new(
            MatrixData::Dense(BigVec::from_vec(self.witness.clone())),
            1,
            self.witness.len(),
            None,
            "physics example witness".to_string(),
        );
        let mut scaled = Matrix::<T>::from_f64(&witness_matrix, scale_factor, None);
        let ranges = self.metadata().get_ranges();
        scaled.set_ranges(&ranges);
        scaled
    }

    pub fn witness(&self, scale_factor: AFloat) -> Matrix<i64> {
        self.witness_int(scale_factor)
    }
}

impl R1CSExec for WitnessGenerator {
    type Atom = f64;
    type Output = Vec<f64>;

    fn add_input(&mut self) -> f64 {
        let v = *self
            .inputs
            .get(self.next_input)
            .expect("input index out of bounds, please ensure input is supplied");
        self.next_input += 1;
        self.witness.push(v);
        v
    }

    fn add_output(&mut self) {
        self.witness.push(0f64);
        if self.out_idx == 0 {
            self.out_idx = self.witness.len() - 1;
        }
    }

    fn output(&mut self, atom: f64) {
        self.witness[self.out_idx] = atom;
        self.out_idx += 1;
    }

    fn constant(c: f64) -> f64 {
        c
    }

    fn add(&mut self, lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }

    fn mul(&mut self, lhs: f64, rhs: f64) -> f64 {
        self.witness.push(lhs * rhs);
        lhs * rhs
    }

    fn max(&mut self, lhs: f64, c: f64) -> f64 {
        let z = lhs.max(c);

        // z
        self.witness.push(z);

        // b0
        self.witness
            .push(if (lhs - z).abs() < 1e-6 { 1f64 } else { 0f64 });

        // b1
        self.witness
            .push(if (c - z).abs() < 1e-6 { 1f64 } else { 0f64 });

        // sqrt (z - lhs)
        self.witness.push((z - lhs).sqrt());

        // sqrt (z - c)
        self.witness.push((z - c).sqrt());

        // b0 * x
        self.witness
            .push(if (lhs - z).abs() < 1e-6 { lhs } else { 0f64 });

        z
    }

    fn min(&mut self, lhs: f64, c: f64) -> f64 {
        -self.max(-lhs, -c)
    }

    fn abs(&mut self, x: Self::Atom) -> Self::Atom {
        let z = x.abs();

        // z
        self.witness.push(z);

        // sqrt(z)
        self.witness.push(z.sqrt());

        // sign
        self.witness.push(if x < 0f64 { -1f64 } else { 1f64 });

        z
    }

    fn finish(self) -> Vec<f64> {
        self.witness
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConstraintGenerator {
    builder: R1CSBuilder,
    input_vars: Vec<Var>,
    output_vars: VecDeque<Var>,
}

impl ConstraintGenerator {
    pub fn new() -> Self {
        Self {
            builder: R1CSBuilder::new(),
            input_vars: Vec::new(),
            output_vars: VecDeque::new(),
        }
    }
}

impl R1CSExec for ConstraintGenerator {
    type Atom = LC;
    type Output = R1CSBuilder;

    fn add_input(&mut self) -> Self::Atom {
        let v = self.builder.new_variable();
        self.input_vars.push(v.clone());
        LC::var(v)
    }

    fn constant(c: f64) -> Self::Atom {
        LC::constant(c)
    }

    fn add_output(&mut self) {
        let var = self.builder.new_variable();
        self.output_vars.push_back(var);
    }

    fn output(&mut self, atom: Self::Atom) {
        let output_var = self
            .output_vars
            .pop_front()
            .expect("output index out of bounds, please ensure output is declared");
        self.builder
            .add_constraint(&atom, &LC::one(), &LC::var(output_var));
    }

    fn add(&mut self, lhs: Self::Atom, rhs: Self::Atom) -> Self::Atom {
        lhs + &rhs
    }

    fn mul(&mut self, lhs: Self::Atom, rhs: Self::Atom) -> Self::Atom {
        let z = self.builder.new_variable();
        let z_lc = LC::var(z);
        self.builder.add_constraint(&lhs, &rhs, &z_lc);
        z_lc
    }

    fn max(&mut self, lhs: Self::Atom, c: f64) -> Self::Atom {
        // var z represents max of lhs and c
        let z = self.builder.new_lc();

        // b_0 == 1 implies z == lhs
        let b_0 = self.builder.new_lc();

        // b_1 == 1 implies z == c
        let b_1 = self.builder.new_lc();

        // sqrt(z - lhs)
        let t_1 = self.builder.new_lc();

        // sqrt(z - c)
        let t_2 = self.builder.new_lc();

        // b_0 * x
        let b_0_times_x = self.builder.new_lc();

        // enforce b_0 and b_1 are boolean
        self.builder
            .add_constraint(&(1f64 - &b_0), &b_0, &LC::zero());

        self.builder
            .add_constraint(&(1f64 - &b_1), &b_1, &LC::zero());

        // enforce t_1^2 == z - lhs
        self.builder.add_constraint(&t_1, &t_1, &(z.clone() - &lhs));

        // enforce t_2^2 == z - c
        self.builder.add_constraint(&t_2, &t_2, &(z.const_add(-c)));

        // enforce exactly one of b_0 and b_1 is 1
        self.builder
            .add_constraint(&(b_0.clone() + &b_1), &LC::one(), &LC::one());

        // enforce b_0 * x == b_0 * lhs
        self.builder.add_constraint(&b_0, &lhs, &b_0_times_x);

        // enforce z == b_0 * lhs + b_1 * c
        self.builder
            .add_constraint(&LC::one(), &(b_0_times_x + &b_1.const_mul(c)), &z);

        z
    }

    fn min(&mut self, lhs: Self::Atom, c: f64) -> Self::Atom {
        self.max(lhs * -1.0, -c) * -1.0
    }

    fn abs(&mut self, x: Self::Atom) -> Self::Atom {
        // z = abs(x)
        let z = self.builder.new_lc();

        // sqrt(z)
        let t = self.builder.new_lc();

        // sign is -1 if x < 0 and 1 if x >= 0
        let sign = self.builder.new_lc();

        // enforce t^2 == z
        self.builder.add_constraint(&t, &t, &z);

        // enforce sign^2 == 1
        self.builder.add_constraint(&sign, &sign, &LC::one());

        // enforce sign * z == x
        self.builder.add_constraint(&sign, &z, &x);

        z
    }

    fn finish(self) -> Self::Output {
        if !self.output_vars.is_empty() {
            panic!(
                "Not all outputs were constrained, please ensure all outputs are used in constraints"
            );
        }
        self.builder
    }
}

// Simple utility to check constraint satisifcation. Panics if any constraint is not satisfied
// beyond some reasonable epsilon.
pub fn check_constraints(constraints: &R1CSBuilder, witness: &[f64]) {
    for (a, b, c) in &constraints.constraints {
        let a_val = a.eval(witness);
        let b_val = b.eval(witness);
        let c_val = c.eval(witness);
        if (a_val * b_val - c_val).abs() > 1e-6 {
            panic!("Constraint not satisfied: {a_val} * {b_val} != {c_val}",);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Div;

    use super::*;

    // Physics example constants
    const N: usize = 64;
    const DT: f64 = 0.1;
    const DIFFUSION_COEFF: f64 = 0.0001;
    const DIFFUSION_ITERS: usize = 20;

    fn simple_program<E: R1CSExec>(exec: &mut E) -> E::Atom {
        let a = exec.add_input();
        let b = exec.add_input();
        let c = exec.mul(a.clone(), b);
        exec.add(c, E::constant(1.0))
    }

    fn exp_n_times<E: R1CSExec>(n: usize, exec: &mut E) -> E::Atom {
        let base = exec.add_input();
        let mut res = base.clone();
        for _ in 0..n {
            res = exec.mul(res.clone(), base.clone());
        }
        res
    }
    fn add_constant_to_cell<E: R1CSExec>(
        _exec: &mut E,
        tensor: &mut Array<E::Atom, IxDyn>,
        idx: [usize; 2],
        delta: f64,
    ) {
        let cell = tensor.get_mut(idx).expect("index out of bounds");
        *cell = cell.clone() + delta;
    }

    fn clone_cell<T: Clone>(tensor: &Array<T, IxDyn>, idx: [usize; 2]) -> T {
        tensor.get(idx).expect("index out of bounds").clone()
    }

    fn set_cell_value<T>(tensor: &mut Array<T, IxDyn>, idx: [usize; 2], value: T) {
        *tensor.get_mut(idx).expect("index out of bounds") = value;
    }

    fn stir_ops<E: R1CSExec>(exec: &mut E) -> Array<E::Atom, IxDyn> {
        let cx = N.div(2);
        let cy = N.div(2);
        let r = 5;
        let strength = 100f64;

        // Mock inputs
        let mut u_prev = exec.add_tensor_input(&[N, N]);
        let mut v_prev = exec.add_tensor_input(&[N, N]);
        let mut dye_prev = exec.add_tensor_input(&[N, N]);

        // State transition for stir
        for i in cx - r..cx + r {
            for j in cy - r..cy + r {
                let dx = (i as f64) - (cx as f64);
                let dy = (j as f64) - (cy as f64);
                let dist2 = dx * dx + dy * dy;
                let r_squared = (r * r) as f64;
                if dist2 < r_squared {
                    let falloff = f64::exp(-dist2 / r_squared);

                    // inject rotational velocity (perpendicular to radius)
                    let idx = [i, j];
                    let velocity_delta = -strength * falloff * dy;
                    add_constant_to_cell(exec, &mut u_prev, idx, velocity_delta);
                    add_constant_to_cell(exec, &mut v_prev, idx, velocity_delta);

                    // inject dye
                    add_constant_to_cell(exec, &mut dye_prev, idx, strength * falloff);
                }
            }
        }
        dye_prev
    }

    fn diffuse_ops<E: R1CSExec>(exec: &mut E) -> Array<E::Atom, IxDyn> {
        let mut x = exec.add_tensor_input(&[N, N]);
        let x0 = exec.add_tensor_input(&[N, N]);

        let grid_area = (N * N) as f64;
        let a = DT * DIFFUSION_COEFF * grid_area;
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
                    let numerator = exec.add(clone_cell(&x0, [i, j]), scaled_neighbors);
                    let updated = numerator * inv_denom;
                    set_cell_value(&mut x, [i, j], updated);
                }
            }
        }

        x
    }

    #[test]
    fn gen_wit_example() {
        // Inputs
        let a = 5f64;
        let b = 10f64;
        let mut wit_gen = WitnessGenerator::new(vec![a, b]);

        // Gen result
        let res = simple_program(&mut wit_gen);

        // Check consistency
        let expected = a * b + 1f64;
        assert_eq!(res, expected);
    }

    #[test]
    fn cube_example() {
        let base = 2f64;

        // Execute and get witness
        let mut exp_gen = WitnessGenerator::new(vec![base]);
        let cube = exp_n_times(2, &mut exp_gen);
        let witness = exp_gen.finish();

        // Now symbolically execute to get constraints and ensure constraint satisfaction
        let mut cons_gen = ConstraintGenerator::new();
        let _ = exp_n_times(2, &mut cons_gen);
        let constraints = cons_gen.finish();
        check_constraints(&constraints, &witness);

        assert_eq!(cube, 8f64);
        assert_eq!(witness, vec![1f64, 2f64, 4f64, 8f64]);
    }

    #[test]
    fn test_stir_ops() {
        // Mock inputs
        let inp1 = Array::from_shape_fn(IxDyn(&[N, N]), |_| 1f64);
        let inp2 = Array::from_shape_fn(IxDyn(&[N, N]), |_| 1f64);
        let inp3 = Array::from_shape_fn(IxDyn(&[N, N]), |_| 1f64);

        let mut exp_gen = WitnessGenerator::new_from_tensored_inputs(vec![inp1, inp2, inp3]);
        let _lit_res = stir_ops(&mut exp_gen);
        let witness = exp_gen.finish();

        let mut cons_gen = ConstraintGenerator::new();
        let _symb_res = stir_ops(&mut cons_gen);
        let constraints = cons_gen.finish();

        check_constraints(&constraints, &witness);
    }

    #[test]
    fn test_diffuse_ops() {
        // Mock inputs
        let x_input = Array::from_shape_fn(IxDyn(&[N, N]), |_| 0f64);
        let x0_input =
            Array::from_shape_fn(IxDyn(&[N, N]), |idx| (idx[0] as f64 + idx[1] as f64) * 0.01);

        // Witness generator
        let mut exp_gen = WitnessGenerator::new_from_tensored_inputs(vec![x_input, x0_input]);
        let res = diffuse_ops(&mut exp_gen);
        let witness = exp_gen.finish();
        assert_eq!(res.shape(), &[N, N]);

        // Constraints generator
        let mut cons_gen = ConstraintGenerator::new();
        let _ = diffuse_ops(&mut cons_gen);
        let constraints = cons_gen.finish();

        // Check constraints
        check_constraints(&constraints, &witness);
    }
    #[test]
    fn test_max() {
        fn max<E: R1CSExec>(exec: &mut E, c: f64) -> E::Atom {
            let a = exec.add_input();
            exec.max(a, c)
        }

        let a = 5f64;
        let c = 3f64;

        // Gen witness
        let mut exp_gen = WitnessGenerator::new(vec![a]);
        let res = max(&mut exp_gen, c);
        let witness = exp_gen.finish();

        // Gen constraints
        let mut cons_gen = ConstraintGenerator::new();
        let symb_res = max(&mut cons_gen, c);
        let constraints = cons_gen.finish();

        assert_eq!(constraints.constraints.len(), 7);

        // Check constraints
        check_constraints(&constraints, &witness);

        assert_eq!(res, a.max(c));
        assert_eq!(symb_res.eval(&witness), a.max(c));
    }

    #[test]
    fn test_min() {
        fn min<E: R1CSExec>(exec: &mut E, c: f64) -> E::Atom {
            let a = exec.add_input();
            exec.min(a, c)
        }

        let a = 5f64;
        let c = 7f64;

        // Gen witness
        let mut exp_gen = WitnessGenerator::new(vec![a]);
        let res = min(&mut exp_gen, c);
        let witness = exp_gen.finish();

        // Gen constraints
        let mut cons_gen = ConstraintGenerator::new();
        let symb_res = min(&mut cons_gen, c);
        let constraints = cons_gen.finish();

        assert_eq!(constraints.constraints.len(), 7);

        // Check constraints
        check_constraints(&constraints, &witness);

        assert_eq!(res, a.min(c));
        assert_eq!(symb_res.eval(&witness), a.min(c));
    }

    #[test]
    fn test_abs() {
        fn abs<E: R1CSExec>(exec: &mut E) -> E::Atom {
            let x = exec.add_input();
            exec.abs(x)
        }

        let x = -5f64;

        let mut exp_gen = WitnessGenerator::new(vec![x]);
        let res = abs(&mut exp_gen);
        let witness = exp_gen.finish();

        let mut cons_gen = ConstraintGenerator::new();
        let _ = abs(&mut cons_gen);
        let constraints = cons_gen.finish();

        assert_eq!(constraints.constraints.len(), 3);
        check_constraints(&constraints, &witness);
        assert_eq!(res, x.abs());
    }

    #[test]
    fn test_output() {
        fn fn_with_output<E: R1CSExec>(exec: &mut E) -> E::Atom {
            let x = exec.add_input();
            let y = exec.add_input();
            exec.add_output();
            exec.add_output();

            let z = exec.mul(x.clone(), y.clone());
            // Two outputs for fun
            exec.output(z.clone());
            exec.output(z.clone());

            z
        }
        let x = 5f64;
        let y = 10f64;

        let mut exp_gen = WitnessGenerator::new(vec![x, y]);
        let res = fn_with_output(&mut exp_gen);
        let witness = exp_gen.finish();

        let mut cons_gen = ConstraintGenerator::new();
        let _ = fn_with_output(&mut cons_gen);
        let constraints = cons_gen.finish();

        assert_eq!(constraints.constraints.len(), 3);
        check_constraints(&constraints, &witness);
        assert_eq!(res, x * y);
        assert_eq!(witness[3], x * y);
    }

    #[test]
    fn test_tensor_output() {
        fn fn_with_tensor_output<E: R1CSExec>(exec: &mut E) -> Array<E::Atom, IxDyn> {
            let x = exec.add_tensor_input(&[2, 2]);
            exec.add_tensor_output(&[2, 2]);
            exec.tensor_output(&x);
            x
        }

        let inputs = vec![0.0, 1.0, 2.0, 3.0];
        let input = Array::from_shape_vec(IxDyn(&[2, 2]), inputs).unwrap();

        let mut exp_gen = WitnessGenerator::new_from_tensored_inputs(vec![input.clone()]);
        let res = fn_with_tensor_output(&mut exp_gen);
        let witness = exp_gen.finish();

        let mut cons_gen = ConstraintGenerator::new();
        let _ = fn_with_tensor_output(&mut cons_gen);
        let constraints = cons_gen.finish();

        assert_eq!(constraints.constraints.len(), 4);
        check_constraints(&constraints, &witness);

        assert_eq!(res.shape(), &[2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(res[[i, j]], input[[i, j]]);
                assert_eq!(witness[4 + i * 2 + j + 1], input[[i, j]]);
            }
        }
    }

    #[test]
    fn test_get_metadata() {
        fn simple_program<E: R1CSExec>(exec: &mut E) {
            let x = exec.add_input();
            exec.add_output();
            exec.output(x);
        }

        let mut exp_gen = WitnessGenerator::new(vec![3.14]);
        simple_program(&mut exp_gen);
        let metadata = exp_gen.metadata();
        assert_eq!(metadata.num_public_values, 2);
        assert_eq!(metadata.num_random_values, 0);
        assert_eq!(metadata.num_secondary_witness_values, 0);
        assert_eq!(metadata.num_secondary_constraint_variables, 0);

        let _ = exp_gen.finish();
        assert_eq!(metadata.num_witness_values, 1);
    }
}
