use core::panic;
use std::fmt::Debug;

use crate::{
    inputs::{Metadata, R1CSMatrices},
    traits::{MatrixIntOps, R1CSInstance, ToI512, ToI1024},
    verifier::ZRangeOpening,
};
use dark::{prover::ChunkedComm, public::PublicParams};
use dark::{
    prover::{ProverState as DarkProverState, RoundClaim},
    verifier::RoundChallenge,
};
use ff::{FieldElem, FieldMont, inner2::InnerPoly, outer::OuterPoly};
use i256::{I512, I1024};
use iop::prover::ProverState as SCProverState;
use model::HighPrecision;
use parse::{
    generalized::{HighPrecisionInt, InjectionInfo},
    mat::{Matrix, MatrixData},
};
use rug::Integer;

pub use crate::simulate::{simulate, simulate_hp};

pub struct ProverState<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512 + ToI1024,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> where
    Matrix<T>: MatrixIntOps,
{
    pub error: Option<I1024>,
    pub scale_factor: P,
    pub batch_size: usize,
    pub metadata: Metadata,
    pub r1cs_matrices_int: Option<R1CSMatrices<T>>,
    pub r1cs_matrices_mont: Option<R1CSMatrices<FieldElem>>,
    pub wit_exec: E,
    pub witness: Option<Matrix<T>>,
    pub randomness: Option<Vec<P>>,
    pub inject_info: Option<InjectionInfo>,
    pub witness_mont: Option<Matrix<FieldElem>>,
    pub r_outer: Vec<FieldElem>,
    pub dark_public_params: Option<PublicParams>,
    pub dark_prover: DarkProverState,
    pub scale_den: Option<FieldElem>,
    pub mont: Option<FieldMont>,
    pub outer_prover: Option<SCProverState<OuterPoly>>,
    pub inner_prover: Option<SCProverState<InnerPoly>>,
    _phantom: std::marker::PhantomData<P>,
}

impl<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512 + ToI1024,
    E: R1CSInstance<P, T>,
    P: HighPrecision,
> ProverState<T, P, E>
where
    Matrix<T>: MatrixIntOps,
{
    pub fn new(wit_exec: E, scale_factor: P, metadata: Metadata, batch_size: usize) -> Self {
        let mut ret = Self {
            error: None,
            metadata,
            batch_size,
            scale_factor,
            wit_exec,
            r1cs_matrices_int: None,
            r1cs_matrices_mont: None,
            witness: None,
            randomness: None,
            inject_info: None,
            witness_mont: None,
            outer_prover: None,
            inner_prover: None,
            dark_public_params: None,
            dark_prover: DarkProverState::default(),
            scale_den: None,
            mont: None,
            r_outer: Vec::new(),
            _phantom: std::marker::PhantomData::<P>,
        };
        let (matrices, inject_info) = ret.wit_exec.get_matrices(ret.scale_factor.clone(), None);
        ret.r1cs_matrices_int = Some(matrices);
        ret.inject_info = inject_info;
        ret
    }

    pub fn import_r1cs_matrices(&mut self) {
        let randomness = self
            .randomness
            .to_owned()
            .unwrap()
            .into_iter()
            .map(|v| T::from_hp(v * self.scale_factor.clone()))
            .collect::<Vec<_>>();
        let (matrices, _) = self
            .wit_exec
            .get_matrices(self.scale_factor.clone(), Some(&randomness));
        self.r1cs_matrices_int = Some(matrices);
    }

    pub fn inject_randomness(&mut self) {
        if self.inject_info.is_none() {
            eprintln!("nothing to inject, skipping inject");
            return;
        }
        let randomness = self
            .randomness
            .to_owned()
            .expect("inject randomness with no randomness set")
            .into_iter()
            .map(|v| T::from_hp(v * self.scale_factor.clone()))
            .collect::<Vec<_>>();
        let mut matrix = self.r1cs_matrices_int.take().unwrap();
        if let MatrixData::COO(c_data) = matrix.c.mut_data() {
            self.inject_info
                .as_ref()
                .unwrap()
                .iter()
                .for_each(|&(mat_index, rand_index)| {
                    assert_eq!(
                        c_data[mat_index].2,
                        T::from_i128(0),
                        "injecting randomness at a location where data isn't zero"
                    );
                    c_data[mat_index] = (
                        c_data[mat_index].0,
                        c_data[mat_index].1,
                        randomness[rand_index],
                    );
                });
        } else {
            panic!()
        }
        self.r1cs_matrices_int = Some(matrix);
    }

    pub fn compute_commit_witness(&mut self) {
        self.witness = Some(
            self.wit_exec
                .compute_commit_witness(self.scale_factor.clone(), self.batch_size),
        );
    }

    pub fn set_randomness(&mut self, randomness: Vec<P>) {
        self.randomness = Some(randomness);
    }

    pub fn compute_full_witness(&mut self) {
        let witness = self.wit_exec.compute_full_witness(
            &self.metadata,
            self.randomness.to_owned().unwrap(),
            self.scale_factor.clone(),
        );
        self.witness = Some(witness);
    }

    pub fn compute_squared_error(&mut self) -> I1024 {
        let witness = self
            .witness
            .as_ref()
            .expect("witness must be computed before computing error");
        let error = compute_squared_error_i1024(
            self.r1cs_matrices_int
                .as_ref()
                .expect("R1CS matrices must be set before computing error"),
            witness,
            &self.scale_factor.to_i512(),
            true,
        );
        self.error = Some(error);
        error
    }

    pub fn set_dark_public_params(&mut self, mut dark_public_params: PublicParams) {
        dark_public_params.build_pippenger_bases();
        self.dark_public_params = Some(dark_public_params);
    }

    pub fn dark_respond(&mut self, challenge: RoundChallenge) -> RoundClaim {
        self.dark_prover.respond_to_challenge(
            &challenge,
            self.dark_public_params
                .as_ref()
                .expect("Dark public params should be set"),
        )
    }

    pub fn dark_mle_eval(&mut self, eval_point: &[FieldElem]) -> FieldElem {
        self.dark_prover.gen_y_claim(
            eval_point.to_vec(),
            self.dark_public_params
                .as_ref()
                .expect("Dark public params should be set"),
        )
    }

    pub fn witness_openings(&self) -> Vec<ZRangeOpening>
    where
        T: Into<i128>,
    {
        let z_int = self
            .witness
            .as_ref()
            .expect("witness should be generated before witness openings");
        let ranges = self.metadata.get_ranges();
        ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 1)
            .map(|(i, range)| ZRangeOpening {
                range_index: i,
                width: z_int.width(),
                values: extract_dense_rows_i128(z_int, range),
            })
            .collect()
    }
    pub fn set_mont(&mut self, mont: FieldMont) {
        // Set the small mont and cache the scale denominator for later use
        let scale_den = get_scale_mont(&mont, &self.scale_factor);
        let scale_den = mont.inv(scale_den);
        self.scale_den = Some(scale_den);
        self.mont = Some(mont);
        self.dark_public_params
            .as_mut()
            .expect("public params should be set")
            .set_small_mont(mont);
    }

    pub fn commit(&mut self) -> ChunkedComm {
        let dark_params = self
            .dark_public_params
            .as_ref()
            .expect("dark public params not set before commit");
        let w_mle = self
            .witness
            .take()
            .expect("witness not generated yet")
            .extract_rows_to_mle(None);
        self.dark_prover.commit(w_mle, dark_params)
    }

    pub fn convert_instance_to_mont(&mut self) {
        let start = std::time::Instant::now();
        let (r1cs_mont, z_mont) = convert_instance_to_mont(
            self.mont.as_ref().expect("Mont should be set"),
            self.r1cs_matrices_int
                .as_ref()
                .expect("R1CS matrices should be set"),
            self.witness.as_ref().expect("Witness should be generated"),
            *self
                .scale_den
                .as_ref()
                .expect("Scale denominator should be set"),
            true,
        );
        self.r1cs_matrices_mont = Some(r1cs_mont);
        self.witness_mont = Some(z_mont);
        let duration = start.elapsed();
        eprintln!("Converted instance to montgomery form in {:?}", duration);
    }

    pub fn prepare_outer_sc(&mut self) {
        let sc_state = prepare_outer_sum_check(
            self.mont.as_ref().expect("Mont should be set"),
            self.r1cs_matrices_mont
                .as_ref()
                .expect("R1CS mont matrices should be set"),
            self.witness_mont
                .as_ref()
                .expect("Witness mont should be generated"),
            true,
        );
        self.outer_prover = Some(sc_state);
    }

    pub fn prepare_inner_sc(&mut self, r1: FieldElem, r2: FieldElem) -> usize {
        let a_height = self
            .r1cs_matrices_mont
            .as_ref()
            .expect("R1CS mont matrices should be set")
            .a
            .height();
        let split_point = a_height.next_power_of_two().ilog2() as usize;
        let r_row = self.r_outer[0..split_point].to_vec();
        let r_z_col = self.r_outer[split_point..].to_vec();

        let sc_state = prepare_inner_sum_check(
            self.mont.as_ref().expect("Mont should be set"),
            self.r1cs_matrices_mont
                .as_ref()
                .expect("R1CS mont matrices should be set"),
            self.witness_mont
                .as_ref()
                .expect("Witness mont should be generated"),
            r1,
            r2,
            &r_row,
            &r_z_col,
            true,
        );
        let num_vars = sc_state.num_vars();
        self.inner_prover = Some(sc_state);
        num_vars
    }

    pub fn outer_sc_claim(&mut self) -> Vec<FieldElem> {
        self.outer_sc_prove(None)
    }

    pub fn outer_sc_prove(&mut self, r: Option<FieldElem>) -> Vec<FieldElem> {
        let sc_prover = self
            .outer_prover
            .as_mut()
            .expect("Outer sum-check prover should be prepared");

        if let Some(r) = r {
            self.r_outer.push(r);
        }

        sc_prover.prove_round(r)
    }

    pub fn outer_final_evals(&mut self, r: FieldElem) -> Vec<FieldElem> {
        let sc_prover = self
            .outer_prover
            .as_mut()
            .expect("Outer sum-check prover should be prepared");

        self.r_outer.push(r);

        sc_prover.final_evals(r)
    }

    pub fn outer_last_round(&self) -> bool {
        let sc_prover = self
            .outer_prover
            .as_ref()
            .expect("Outer sum-check prover should be prepared");
        sc_prover.last_round()
    }

    pub fn inner_sc_claim(&mut self) -> Vec<FieldElem> {
        self.inner_sc_prove(None)
    }

    pub fn inner_sc_prove(&mut self, r: Option<FieldElem>) -> Vec<FieldElem> {
        let sc_prover = self
            .inner_prover
            .as_mut()
            .expect("Inner sum-check prover should be prepared");

        sc_prover.prove_round(r)
    }
    pub fn inner_last_round(&self) -> bool {
        let sc_prover = self
            .inner_prover
            .as_ref()
            .expect("Inner sum-check prover should be prepared");
        sc_prover.last_round()
    }

    pub fn inner_final_evals(&mut self, r: FieldElem) -> Vec<FieldElem> {
        let sc_prover = self
            .inner_prover
            .as_mut()
            .expect("Inner sum-check prover should be prepared");

        sc_prover.final_evals(r)
    }

    pub fn num_constraints(&self) -> usize {
        self.r1cs_matrices_int.as_ref().unwrap().a.height()
    }
}

fn extract_dense_rows_i128<T>(mat: &Matrix<T>, row_range: &std::ops::Range<usize>) -> Vec<i128>
where
    T: Copy + Clone + Default + PartialEq + Into<i128>,
{
    match mat.data() {
        MatrixData::Dense(values) => {
            let width = mat.width();
            let mut out = Vec::with_capacity(row_range.len() * width);
            for r in row_range.clone() {
                let base = r * width;
                for c in 0..width {
                    out.push(values[base + c].into());
                }
            }
            out
        }
        _ => panic!("z openings currently require dense witness matrix"),
    }
}

pub(crate) fn limbs_to_u128(limbs: &[u64]) -> u128 {
    let low = limbs.first().copied().unwrap_or(0) as u128;
    let high = limbs.get(1).copied().unwrap_or(0) as u128;
    low | (high << 64)
}

fn r1cs_rat_to_mont<T>(
    matrices: &R1CSMatrices<T>,
    den_mont: FieldElem,
    mont: &FieldMont,
) -> R1CSMatrices<FieldElem>
where
    T: Clone + Default + PartialEq,
    Matrix<T>: MatrixIntOps,
{
    R1CSMatrices {
        a: matrices.a.rat_to_mont(den_mont, mont),
        b: matrices.b.rat_to_mont(den_mont, mont),
        c: matrices.c.rat_to_mont(den_mont, mont),
    }
}

fn mul_to_vec_i512<T>(a: &Matrix<T>, z: &Matrix<T>) -> Vec<I512>
where
    T: Copy + Clone + Default + PartialEq + ToI512,
{
    assert_eq!(
        a.width(),
        z.height(),
        "Incompatible matrix dimensions for multiplication"
    );
    let mut out = vec![I512::from(0); a.height() * z.width()];
    match (a.data(), z.data()) {
        (MatrixData::COO(a_entries), MatrixData::Dense(z_values)) => {
            let z_width = z.width();
            for (r, c, val) in a_entries.iter() {
                let row_offset = r * z_width;
                let z_row_offset = c * z_width;
                let a_val = (*val).to_i512();
                for col in 0..z_width {
                    let z_val = z_values[z_row_offset + col].to_i512();
                    out[row_offset + col] += a_val * z_val;
                }
            }
        }
        (_, _) => panic!("not supported"),
    }
    out
}

fn mul_to_vec_i1024<T>(a: &Matrix<T>, z: &Matrix<T>) -> Vec<I1024>
where
    T: Copy + Clone + Default + PartialEq + ToI1024,
{
    assert_eq!(
        a.width(),
        z.height(),
        "Incompatible matrix dimensions for multiplication"
    );
    let mut out = vec![I1024::from(0); a.height() * z.width()];
    match (a.data(), z.data()) {
        (MatrixData::COO(a_entries), MatrixData::Dense(z_values)) => {
            let z_width = z.width();
            for &(r, c, val) in a_entries.iter() {
                let row_offset = r * z_width;
                let z_row_offset = c * z_width;
                let a_val = val.to_i1024();
                for col in 0..z_width {
                    let z_val = z_values[z_row_offset + col].to_i1024();
                    out[row_offset + col] += a_val * z_val;
                }
            }
        }
        (_, _) => panic!("not supported"),
    }
    out
}

fn mul_to_vec_rug<T>(a: &Matrix<T>, z: &Matrix<T>) -> Vec<Integer>
where
    T: Copy + Clone + Default + PartialEq + HighPrecisionInt,
{
    assert_eq!(
        a.width(),
        z.height(),
        "Incompatible matrix dimensions for multiplication"
    );
    let mut out = vec![Integer::from(0); a.height() * z.width()];
    match (a.data(), z.data()) {
        (MatrixData::COO(a_entries), MatrixData::Dense(z_values)) => {
            let z_width = z.width();
            for (r, c, val) in a_entries.iter() {
                let row_offset = r * z_width;
                let z_row_offset = c * z_width;
                let a_val = (*val).to_rug_int();
                for col in 0..z_width {
                    let z_val = z_values[z_row_offset + col].to_rug_int();
                    out[row_offset + col] = out[row_offset + col].clone() + a_val.clone() * z_val;
                }
            }
        }
        (_, _) => panic!("not supported"),
    }
    out
}

// Given a, b, c, z in integer form compute error (numerator)
pub fn compute_squared_error<T>(
    tensors: &R1CSMatrices<T>,
    z: &Matrix<T>,
    scale_factor: &I512,
    verbose: bool,
) -> I512
where
    T: Copy + Clone + Default + PartialEq + ToI512 + Debug,
{
    // get scale factor squared in wide integer form
    if verbose {
        eprintln!("Computing squared error");
    }
    let scale_squared = *scale_factor * *scale_factor;
    let az = mul_to_vec_i512(&tensors.a, z);
    let bz = mul_to_vec_i512(&tensors.b, z);
    let cz = mul_to_vec_i512(&tensors.c, z);
    az.iter()
        .zip(bz.iter())
        .zip(cz.iter())
        .map(|((a, b), c)| {
            let error = (*a * *b) - (*c * scale_squared);
            error * error
        })
        .fold(I512::from(0), |acc, x| acc + x)
}

pub fn compute_squared_error_i1024<T>(
    tensors: &R1CSMatrices<T>,
    z: &Matrix<T>,
    scale_factor: &I512,
    verbose: bool,
) -> I1024
where
    T: Copy + Clone + Default + PartialEq + ToI1024 + Debug,
{
    // get scale factor squared in wide integer form
    if verbose {
        eprintln!("Computing squared error");
    }
    let scale_squared =
        I1024::from_str_radix((*scale_factor * *scale_factor).to_string().as_str(), 10).unwrap();
    let az = mul_to_vec_i1024(&tensors.a, z);
    let bz = mul_to_vec_i1024(&tensors.b, z);
    let cz = mul_to_vec_i1024(&tensors.c, z);
    az.iter()
        .zip(bz.iter())
        .zip(cz.iter())
        .map(|((a, b), c)| {
            let error = (*a * *b) - (*c * scale_squared);
            error * error
        })
        .fold(I1024::from(0), |acc, x| acc + x)
}

pub fn compute_squared_error_hp<T>(
    tensors: &R1CSMatrices<T>,
    z: &Matrix<T>,
    scale_factor: &I512,
    verbose: bool,
) -> Integer
where
    T: Copy + Clone + Default + PartialEq + ToI512 + Debug + HighPrecisionInt,
{
    // get scale factor squared in wide integer form
    if verbose {
        eprintln!("Computing squared error");
    }
    let scale_squared =
        Integer::from_str_radix((*scale_factor * *scale_factor).to_string().as_str(), 10).unwrap();
    let az = mul_to_vec_rug(&tensors.a, z);
    let bz = mul_to_vec_rug(&tensors.b, z);
    let cz = mul_to_vec_rug(&tensors.c, z);
    az.iter()
        .zip(bz.iter())
        .zip(cz.iter())
        .map(|((a, b), c)| {
            let error = (a.clone() * b.clone()) - (c.clone() * scale_squared.clone());
            error.clone() * error.clone()
        })
        .fold(Integer::from(0), |acc, x| {
            acc + Integer::from_str_radix(x.to_string().as_str(), 10).unwrap()
        })
}

pub fn error_to_mont(
    mont: &FieldMont,
    error: I512,
    scale_mont: FieldElem,
    verbose: bool,
) -> FieldElem {
    if verbose {
        eprintln!("Converting error to montgomery form");
    }
    let error_int = Integer::from_str_radix(error.to_string().as_str(), 10).unwrap();
    let modulus_int = Integer::from(mont.modulus());
    let num = mont.from_bigint(error_int % modulus_int);
    let mut den = scale_mont;
    den = mont.sqr(den); // scale^2
    den = mont.sqr(den); // scale^4
    den = mont.sqr(den); // scale^8
    mont.div(num, den) // error is num / den
}

pub fn error_to_mont_i1024(
    mont: &FieldMont,
    error: I1024,
    scale_mont: FieldElem,
    verbose: bool,
) -> FieldElem {
    if verbose {
        eprintln!("Converting error to montgomery form");
    }
    let error_int = Integer::from_str_radix(error.to_string().as_str(), 10).unwrap();
    let modulus_int = Integer::from(mont.modulus());
    let num = mont.from_bigint(error_int % modulus_int);
    let mut den = scale_mont;
    den = mont.sqr(den); // scale^2
    den = mont.sqr(den); // scale^4
    den = mont.sqr(den); // scale^8
    mont.div(num, den) // error is num / den
}

pub fn scale_factor<P: HighPrecision>(scale_factor_bits: usize) -> P {
    P::from_i128(2_i128.pow(scale_factor_bits as u32)).unwrap()
}

pub fn get_scale_mont<P: HighPrecision>(mont: &FieldMont, scale_factor: &P) -> FieldElem {
    let scale_factor_int = scale_factor.to_i512();
    let scale_limbs = (scale_factor_int % I512::from(mont.modulus())).to_le_limbs();
    let scale_mod = limbs_to_u128(&scale_limbs);

    mont.to_mont(scale_mod)
}

pub fn error_to_mont_rug(
    mont: &FieldMont,
    error: Integer,
    scale_mont: FieldElem,
    verbose: bool,
) -> FieldElem {
    if verbose {
        eprintln!("Converting error to montgomery form");
    }
    let num = mont.from_bigint(error % Integer::from(mont.modulus()));
    let mut den = scale_mont;
    den = mont.sqr(den); // scale^2
    den = mont.sqr(den); // scale^4
    den = mont.sqr(den); // scale^8
    mont.div(num, den) // error is num / den
}

// Given mont and a, b, c, z in integer form and scale_bits, convert to FieldElem form
pub fn convert_instance_to_mont<T>(
    mont: &FieldMont,
    matrices: &R1CSMatrices<T>,
    z: &Matrix<T>,
    den_mont: FieldElem,
    _: bool,
) -> (R1CSMatrices<FieldElem>, Matrix<FieldElem>)
where
    T: Clone + Default + PartialEq,
    Matrix<T>: MatrixIntOps,
{
    let r1cs_mont = r1cs_rat_to_mont(matrices, den_mont, mont);
    (r1cs_mont, z.rat_to_mont(den_mont, mont))
}

// Given mont and a, b, c, z in FieldElem form, set up prover state for outer sum-check protocol protocol gives back back r_outer: Vec<FieldElem> and claims for az(r_outer), bz(r_outer), cz(r_outer)
pub fn prepare_outer_sum_check(
    mont: &FieldMont,
    tensors: &R1CSMatrices<FieldElem>,
    z: &Matrix<FieldElem>,
    // ranges: &Vec<Range<usize>>, TO DO, Add in if we add in SPARK and split A0, A1, etc.
    verbose: bool,
) -> SCProverState<OuterPoly> {
    if verbose {
        eprintln!("Preparing outer sum-check");
    }
    let az = tensors.a.mont_mul(z, mont).to_mle(mont);
    let bz = tensors.b.mont_mul(z, mont).to_mle(mont);
    let cz = tensors.c.mont_mul(z, mont).to_mle(mont);
    let prover_poly = OuterPoly::from_buffers(az, bz, cz);
    SCProverState::new(prover_poly, *mont)
}

// Given a, b, c, z, r1, r2, r_outer, get cx1 * a, r1 * cx1 * b, r2 * cx1 * c, cx2 * z to prepare to run inner sum-check protocol (interactively) which will return get back r_inner: Vec<FieldElem> and claims for a(r_inner, r_outer), r1b(r_inner, r_outer), r2c(r_inner, r_outer), z(r_inner)
#[allow(clippy::too_many_arguments)]
pub fn prepare_inner_sum_check(
    mont: &FieldMont,
    matrices: &R1CSMatrices<FieldElem>,
    z: &Matrix<FieldElem>,
    r1: FieldElem,
    r2: FieldElem,
    r_row: &[FieldElem],
    r_z_col: &[FieldElem],
    verbose: bool,
) -> SCProverState<InnerPoly> {
    if verbose {
        eprintln!("Preparing inner sum-check");
    }
    // get kronecker product of first portion of r_outer
    let cx1_mont = Matrix::trunc_kronecker_prod(r_row, matrices.a.height(), false, mont);
    // cx1_mont * A
    let mut a_cx1 = cx1_mont.mont_mul(&matrices.a, mont);
    a_cx1.set_ranges(z.ranges().unwrap());
    let a_mle = a_cx1.to_mle(mont);
    // cx1_mont * B
    let mut b_cx1 = cx1_mont.mont_mul(&matrices.b, mont);
    b_cx1.set_ranges(z.ranges().unwrap());
    let b_mle = b_cx1.to_mle(mont);
    // cx1_mont * C
    let mut c_cx1 = cx1_mont.mont_mul(&matrices.c, mont);
    c_cx1.set_ranges(z.ranges().unwrap());
    let c_mle = c_cx1.to_mle(mont);
    // and Z * cx2_mont
    let z_mle = if !r_z_col.is_empty() {
        let cx2_mont = Matrix::trunc_kronecker_prod(r_z_col, z.width(), true, mont);
        let mut z_cx2 = z.mont_mul(&cx2_mont, mont);
        z_cx2.set_ranges(z.ranges().unwrap());
        z_cx2.to_mle(mont)
    } else {
        z.to_mle(mont)
    };
    // construct inner polynomial
    let prover_poly = InnerPoly::from_buffers(a_mle, &b_mle, &c_mle, z_mle, r1, r2, mont);
    // construct prover state
    SCProverState::new(prover_poly, *mont)
}

pub fn compute_squared_error_raw(
    tensors: &R1CSMatrices<f64>,
    z: &Matrix<f64>,
    verbose: bool,
) -> f64 {
    if verbose {
        eprintln!("Computing squared error (raw)");
    }

    fn mul_to_vec(a: &Matrix<f64>, z: &Matrix<f64>) -> Vec<f64> {
        assert_eq!(
            a.width(),
            z.height(),
            "Incompatible matrix dimensions for multiplication"
        );
        let mut out = vec![0.0f64; a.height() * z.width()];
        match (a.data(), z.data()) {
            (MatrixData::COO(a_entries), MatrixData::Dense(z_values)) => {
                let z_width = z.width();
                for (r, c, val) in a_entries.iter() {
                    let row_offset = r * z_width;
                    let z_row_offset = c * z_width;
                    for col in 0..z_width {
                        out[row_offset + col] += val * z_values[z_row_offset + col];
                    }
                }
            }
            (_, _) => panic!("not supported"),
        }
        out
    }

    let az = mul_to_vec(&tensors.a, z);
    let bz = mul_to_vec(&tensors.b, z);
    let cz = mul_to_vec(&tensors.c, z);

    az.iter()
        .zip(bz.iter())
        .zip(cz.iter())
        .map(|((a, b), c)| {
            let err = (a * b) - c;
            err * err
        })
        .sum()
}
