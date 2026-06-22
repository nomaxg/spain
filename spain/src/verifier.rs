use std::fmt::Debug;
use std::{ops::Range, str::FromStr};

use dark::prover::{ChunkedComm, RoundClaim};
use dark::public::PublicParams;
use dark::verifier::{RoundChallenge, VerifierState as DarkVerifierState};
use ff::inner2::InnerPoly;
use ff::outer::OuterPoly;
use ff::outer_eq::OuterPolyEq;
use ff::{FieldElem, FieldMont, prime_128};
use i256::I512;
use iop::verifier::VerifierState as SCVerifierState;
use model::HighPrecision;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use parse::generalized::{HighPrecisionInt, InjectionInfo};
use parse::mat::{Matrix, MatrixData};
use rug::{Float, Integer};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::inputs::{Metadata, R1CSMatrices};
use crate::prover::{deterministic_tau, error_to_mont, get_scale_mont, scale_factor};
use crate::traits::{MatrixIntOps, R1CSInstance, ToI512};

#[derive(Debug, Clone)]
pub struct ZRangeOpening<T> {
    pub range_index: usize,
    pub width: usize,
    pub values: Vec<T>,
}

#[derive(Serialize, Deserialize)]
struct SerializableZRangeOpening {
    range_index: usize,
    width: usize,
    values: Vec<String>,
}

impl<T> Serialize for ZRangeOpening<T>
where
    T: ToString,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerializableZRangeOpening {
            range_index: self.range_index,
            width: self.width,
            values: self.values.iter().map(|v| v.to_string()).collect(),
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for ZRangeOpening<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = SerializableZRangeOpening::deserialize(deserializer)?;
        Ok(Self {
            range_index: raw.range_index,
            width: raw.width,
            values: raw
                .values
                .into_iter()
                .map(|v| {
                    T::from_str(&v).map_err(|err| serde::de::Error::custom(format!("{err:?}")))
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

#[derive(Debug)]
pub struct VerifierState<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> where
    Matrix<T>: MatrixIntOps,
{
    pub max_epsilon: f64,
    pub batch_size: usize,
    pub scale_factor_bits: usize,
    pub scale_factor: P,
    pub scale_mont: Option<FieldElem>,
    pub q_bits: usize,
    pub precision: u16,
    pub r1cs_matrices_int: Option<R1CSMatrices<T>>,
    pub mont: Option<FieldMont>,
    pub wit_exec: E,
    pub error: Option<I512>,
    pub num_chunks: usize,
    pub metadata: Metadata,
    pub randomness: Option<Vec<P>>,
    pub inject_info: Option<InjectionInfo>,
    pub spartan_poly: bool,
    // SC-specific state
    pub r_col: Vec<FieldElem>,
    pub inner_state: Option<SCVerifierState>,
    pub outer_state: Option<SCVerifierState>,
    pub outer_claims: Vec<FieldElem>,
    pub inner_claims: Vec<FieldElem>,
    pub r1: Option<FieldElem>,
    pub r2: Option<FieldElem>,
    pub dark_claim: Option<FieldElem>,
    // Dark state
    pub dark_public_params: Option<PublicParams>,
    pub dark_verifier: DarkVerifierState,
    // Misc.
    _phantom_p: std::marker::PhantomData<P>,
    _phantom_t: std::marker::PhantomData<T>,
}

impl<
    T: Clone + Default + Debug + PartialEq + HighPrecisionInt + ToI512,
    P: HighPrecision,
    E: R1CSInstance<P, T>,
> VerifierState<T, P, E>
where
    Matrix<T>: MatrixIntOps,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_epsilon: f64,
        batch_size: usize,
        scale_factor_bits: usize,
        q_bits: usize,
        precision: u16,
        num_chunks: usize,
        spartan_poly: bool,
        wit_exec: E,
        metadata: Metadata,
    ) -> Self {
        let scale_factor: P = if spartan_poly {
            P::from_i128(1).unwrap()
        } else {
            scale_factor(scale_factor_bits)
        };

        let mut ret = VerifierState {
            max_epsilon,
            batch_size,
            scale_factor_bits,
            scale_factor,
            scale_mont: None,
            r_col: Vec::new(),
            mont: None,
            dark_public_params: None,
            randomness: None,
            q_bits,
            error: None,
            precision,
            num_chunks,
            dark_verifier: DarkVerifierState::default(),
            inner_state: None,
            outer_state: None,
            outer_claims: vec![],
            inner_claims: vec![],
            r1: None,
            r2: None,
            dark_claim: None,
            r1cs_matrices_int: None,
            inject_info: None,
            spartan_poly,
            metadata,
            wit_exec,
            _phantom_p: std::marker::PhantomData::<P>,
            _phantom_t: std::marker::PhantomData::<T>,
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

    pub fn epsilon_check(&mut self, squared_error: &I512) {
        if self.spartan_poly {
            self.error = Some(I512::from(0));
        } else {
            self.error = Some(*squared_error);
            epsilon_check(
                squared_error,
                self.scale_factor_bits,
                self.max_epsilon,
                true,
            );
        }
    }

    pub fn sample_mont(&mut self) -> FieldMont {
        let modulus = prime_128::rand_prime(&mut rand::rng());
        let mont = FieldMont::new(modulus);
        self.scale_mont = Some(get_scale_mont(&mont, &self.scale_factor));
        self.mont = Some(mont);

        self.dark_public_params
            .as_mut()
            .expect("public params should be set")
            .set_small_mont(mont);
        mont
    }

    pub fn sample_lc_challenges(&mut self) -> (FieldElem, FieldElem) {
        let mont = self.mont();
        let r1 = mont.to_mont(ff::prime_128::rand_elem(mont.modulus(), &mut rand::rng()));
        let r2 = mont.to_mont(ff::prime_128::rand_elem(mont.modulus(), &mut rand::rng()));
        self.r1 = Some(r1);
        self.r2 = Some(r2);
        (r1, r2)
    }

    pub fn num_constraints(&self) -> usize {
        self.r1cs_matrices_int.as_ref().unwrap().a.height()
    }

    pub fn mont(&self) -> FieldMont {
        *self
            .mont
            .as_ref()
            .expect("Montgomery context not set, call sample_mont")
    }

    pub fn prepare_outer_sc(&mut self) {
        let mont = self.mont();
        let outer_claim = if self.spartan_poly {
            mont.zero()
        } else {
            error_to_mont(
                &mont,
                *self.error.as_ref().expect("error not set"),
                self.scale_mont.expect("scale denominator not set"),
                true,
            )
        };

        self.outer_state = Some(SCVerifierState::new(
            self.r1cs_matrices_int.as_ref().unwrap().a.height(),
            if self.spartan_poly { 3 } else { 4 },
            outer_claim,
            mont,
        ));
    }

    pub fn outer_sc_verify(&mut self, p: &mut Vec<FieldElem>) -> Result<FieldElem, String> {
        self.outer_state
            .as_mut()
            .expect("Outer sum-check state not initialized, call prepare_outer_sc")
            .verify_round(p, &mut rand::rng())
    }

    pub fn outer_sc_check_final_evals(
        &mut self,
        p: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        let r = self
            .outer_state
            .as_ref()
            .expect("Outer sum-check state not initialized, call prepare_outer_sc")
            .challenges();
        if self.spartan_poly {
            let tau = deterministic_tau(&self.mont.unwrap(), r.len());
            // Cache claims for the inner sum-check (exclude eq claim)
            self.outer_claims = evals[1..evals.len()].to_vec();
            let aux = [tau.clone(), r.clone()].concat();
            OuterPolyEq::check_final_evals(&self.mont.unwrap(), p, *r.last().unwrap(), &aux, evals)
        } else {
            // Cache claims for the inner sum-check.
            self.outer_claims = evals.to_owned();
            OuterPoly::check_final_evals(&self.mont.unwrap(), p, *r.last().unwrap(), &[], evals)
        }
    }

    pub fn prepare_inner_sc(&mut self, num_vars: usize) {
        let mont = self.mont();
        let mut claim = self.outer_claims[0];
        claim = mont.add(claim, mont.mul(self.r1.unwrap(), self.outer_claims[1]));
        claim = mont.add(claim, mont.mul(self.r2.unwrap(), self.outer_claims[2]));
        self.inner_state = Some(SCVerifierState::new(num_vars, 2, claim, mont));
    }

    pub fn inner_sc_verify(&mut self, p: &mut Vec<FieldElem>) -> Result<FieldElem, String> {
        self.inner_state
            .as_mut()
            .expect("Inner sum-check state not initialized, call prepare_inner_sc")
            .verify_round(p, &mut rand::rng())
    }

    pub fn inner_sc_check_final_evals(
        &mut self,
        claim: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        let r = self
            .inner_state
            .as_ref()
            .expect("Inner sum-check state not initialized, call prepare_inner_sc")
            .challenges()
            .last()
            .expect("should have a final challenge");
        self.inner_claims = evals.to_owned();
        InnerPoly::check_final_evals(
            &self.mont.unwrap(),
            claim,
            *r,
            &[self.r1.unwrap(), self.r2.unwrap()],
            evals,
        )
    }

    pub fn get_dark_public_params(&self) -> PublicParams {
        self.dark_public_params
            .as_ref()
            .expect("Dark public parameters not set, call dark setup")
            .clone()
    }

    pub fn set_commit(&mut self, comm: ChunkedComm) {
        self.dark_verifier.set_commit(comm);
    }

    pub fn set_dark_claim(&mut self, claim: FieldElem) {
        self.dark_claim = Some(claim);
        let eval_point = self.dark_eval_point();
        self.dark_verifier.set_claim(claim, eval_point);
    }

    pub fn dark_setup(&mut self) {
        let witness_rows = self.metadata.num_witness_values;
        let witness_cols = self.batch_size;
        let num_row_vars = witness_rows.next_power_of_two().trailing_zeros() as usize;
        let num_col_vars = witness_cols.next_power_of_two().trailing_zeros() as usize;
        let num_z_vars = num_row_vars + num_col_vars;
        let mut public_params =
            PublicParams::new(self.q_bits, num_z_vars, self.num_chunks, self.precision);
        self.dark_verifier = DarkVerifierState::new(&public_params);
        self.dark_verifier.compute_const_comms(&mut public_params);
        self.dark_public_params = Some(public_params);
    }

    pub fn dark_eval_point(&self) -> Vec<FieldElem> {
        let has_randomness = self.metadata.num_random_values > 0;
        let num_range_variables: usize = { if has_randomness { 2 } else { 1 } };
        let a_height = self.r1cs_matrices_int.as_ref().unwrap().a.height();
        let split_point = a_height.next_power_of_two().ilog2() as usize;
        let r_outer = self
            .outer_state
            .as_ref()
            .expect("Outer sum-check state not initialized, call prepare_outer_sc")
            .challenges();
        let mut eval_point = vec![];
        let r_col = self.inner_state.as_ref().unwrap().challenges();
        let r_z_col = r_outer[split_point..].to_vec();

        eval_point.extend(r_z_col.clone());
        eval_point.extend(r_col[..r_col.len() - num_range_variables].to_vec());
        eval_point
    }

    pub fn start_dark_round(&mut self) -> RoundChallenge {
        self.dark_verifier
            .start_round(&self.get_dark_public_params(), &mut rand::rng())
    }

    pub fn verify_dark_round(&mut self, round_claim: &RoundClaim) {
        let public = self.get_dark_public_params();
        self.dark_verifier.verify_round(round_claim, &public);
    }

    pub fn matrices_claim_check(&self) {
        let mont = self.mont();
        let scale_den = mont.inv(self.scale_mont.expect("scale mont not set"));
        let tensors_mont = R1CSMatrices {
            a: self
                .r1cs_matrices_int
                .as_ref()
                .unwrap()
                .a
                .rat_to_mont(scale_den, &mont),
            b: self
                .r1cs_matrices_int
                .as_ref()
                .unwrap()
                .b
                .rat_to_mont(scale_den, &mont),
            c: self
                .r1cs_matrices_int
                .as_ref()
                .unwrap()
                .c
                .rat_to_mont(scale_den, &mont),
        };
        let ranges = self.metadata.get_ranges();
        let a_height = self.r1cs_matrices_int.as_ref().unwrap().a.height();
        let split_point = a_height.next_power_of_two().ilog2() as usize;
        let r_outer = self
            .outer_state
            .as_ref()
            .expect("Outer sum-check state not initialized, call prepare_outer_sc")
            .challenges();
        let r_row = r_outer[0..split_point].to_vec();
        let r_col = self
            .inner_state
            .as_ref()
            .expect("Inner sum-check state not initialized, call prepare_inner_sc")
            .challenges();
        let inner_claims = &self.inner_claims;

        check_claim(
            &mont,
            &tensors_mont,
            &r_row,
            r_col,
            &ranges,
            self.r1.expect("r1 not set"),
            self.r2.expect("r2 not set"),
            inner_claims[0],
        );
    }

    pub fn witness_claim_check(&self, z_openings: &[ZRangeOpening<T>]) {
        let mont = self.mont();
        let scale_den = mont.inv(self.scale_mont.expect("scale mont not set"));
        let has_randomness = self.metadata.num_random_values > 0;
        let has_secondary_constraints = self.metadata.num_secondary_constraint_variables > 0;
        let ranges = self.metadata.get_ranges();
        let a_height = self.r1cs_matrices_int.as_ref().unwrap().a.height();
        let split_point = a_height.next_power_of_two().ilog2() as usize;
        let r_outer = self
            .outer_state
            .as_ref()
            .expect("Outer sum-check state not initialized, call prepare_outer_sc")
            .challenges();
        let r_z_col = r_outer[split_point..].to_vec();
        let r_col = self
            .inner_state
            .as_ref()
            .expect("Inner sum-check state not initialized, call prepare_inner_sc")
            .challenges();
        let inner_claims = &self.inner_claims;

        check_final_claim_from_openings(
            &mont,
            mont.mul(self.dark_claim.expect("dark claim not set"), scale_den),
            z_openings,
            scale_den,
            self.batch_size,
            r_col,
            &r_z_col,
            inner_claims,
            &ranges,
            has_randomness,
            has_secondary_constraints,
        );
    }

    pub fn final_verify(&self, z_openings: &[ZRangeOpening<T>]) {
        self.matrices_claim_check();
        self.witness_claim_check(z_openings);
        eprintln!("Final verification success!!!");
    }

    pub fn sample_normal_randomness(&mut self) -> Vec<u64> {
        let randomness = Array::random(
            vec![self.metadata.num_random_values],
            Normal::new(0., 1.).unwrap(),
        )
        .into_iter()
        .collect::<Vec<_>>();
        self.randomness = Some(
            randomness
                .iter()
                .map(|&v| P::from_f64(v).unwrap())
                .collect(),
        );
        randomness.into_iter().map(|v| v.to_bits()).collect()
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
}

pub fn remap_ranges(ranges: &[Range<usize>]) -> Vec<Range<usize>> {
    let subrange_len = ranges
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap_or(0)
        .next_power_of_two();

    ranges
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let start = i * subrange_len;
            let end = start + r.len();
            start..end
        })
        .collect::<Vec<_>>()
}

pub fn smart_mat_eval_sub_range(
    mont: &FieldMont,
    mat: &Matrix<FieldElem>,
    row_range: &Range<usize>,
    r_row_tbl: &[FieldElem],
    r_col_tbl: &[FieldElem],
) -> FieldElem {
    assert!(row_range.end <= mat.height(), "row range out of bounds");
    let row_offset = row_range.start;

    match &mat.data() {
        MatrixData::Dense(values) => {
            let mut res = mont.zero();
            let width = mat.width();

            for r in row_range.clone() {
                let row_eval = r_row_tbl[r - row_offset];
                let base = r * width;

                let mut row_sum = mont.zero();
                for c in 0..width {
                    row_sum = mont.add(row_sum, mont.mul(values[base + c], r_col_tbl[c]));
                }

                res = mont.add(res, mont.mul(row_eval, row_sum));
            }

            res
        }
        MatrixData::COO(entries) => {
            let mut res = mont.zero();
            for (r, c, v) in entries.iter() {
                if *r < row_range.start || *r >= row_range.end {
                    continue;
                }
                let row_eval = r_row_tbl
                    .get(*r - row_offset)
                    .expect("row table too small for sub-range");
                res = mont.add(res, mont.mul(*v, mont.mul(*row_eval, r_col_tbl[*c])));
            }
            res
        }
    }
}

fn smart_mat_eval(
    mont: &FieldMont,
    mat: &Matrix<FieldElem>,
    r_row_tbl: &[FieldElem],
    r_col_tbl: &[FieldElem],
) -> FieldElem {
    // sum val * eq(r_row, row) * eq(r_col, col)
    match &mat.data() {
        MatrixData::Dense(values) => {
            // compute the evaluation
            let mut res = mont.zero();
            for r in 0..mat.height() {
                for c in 0..mat.width() {
                    res = mont.add(
                        res,
                        mont.mul(
                            values[r * mat.width() + c],
                            mont.mul(r_row_tbl[r], r_col_tbl[c]),
                        ),
                    );
                }
            }
            res
        }
        MatrixData::COO(entries) => {
            // compute the evaluation
            let mut res = mont.zero();
            for (r, c, v) in entries.iter() {
                res = mont.add(res, mont.mul(*v, mont.mul(r_row_tbl[*r], r_col_tbl[*c])));
            }
            res
        }
    }
}

fn build_compressed_eval_tbl(
    mont: &FieldMont,
    r: &[FieldElem],
    ranges: &[Range<usize>],
) -> Vec<FieldElem> {
    // build eval table for r
    let tbl = build_eval_tbl(mont, r);
    // compress it according to ranges
    let mut new_evals = Vec::with_capacity(ranges.iter().map(|r| r.len()).sum::<usize>());
    for range in ranges.iter() {
        for i in range.clone() {
            new_evals.push(tbl[i]);
        }
    }
    // return the compressed evals
    new_evals
}

fn build_eval_tbl(mont: &FieldMont, r: &[FieldElem]) -> Vec<FieldElem> {
    let n = r.len();
    let final_len = 1usize << n;
    let mut tbl = vec![mont.zero(); final_len];

    tbl[0] = mont.one();
    let mut cur_len = 1;

    for &ri in r.iter().rev() {
        let one = mont.one();
        let omr = mont.sub(one, ri); // 1 - ri

        for j in (0..cur_len).rev() {
            let t = tbl[j];
            let base_idx = 2 * j;
            tbl[base_idx] = mont.mul(t, omr); // t * (1 - ri)
            tbl[base_idx + 1] = mont.mul(t, ri); // t * ri
        }

        cur_len <<= 1;
    }

    tbl
}

// Checks whether sqrt(J) is sufficiently low, panics if
// computed epsilon is higher than max epsilon
pub fn epsilon_check(
    squared_error: &I512,
    scale_factor_bits: usize,
    max_epsilon: f64,
    verbose: bool,
) {
    let num_dbg = Integer::from_str(&squared_error.to_string()).unwrap();
    let scale8_bits = 8 * scale_factor_bits;
    let scale8 = Integer::from(1) << scale8_bits;
    let denom_dbg = Float::with_val(256, scale8);
    let computed_epsilon = (num_dbg / denom_dbg).sqrt();
    let max_epsilon = Float::with_val(256, max_epsilon);

    if computed_epsilon > max_epsilon {
        panic!(
            "Computed model deviation {computed_epsilon} is greater than acceptable value {max_epsilon}"
        );
    }

    if verbose {
        eprintln!("Error epsilon: {computed_epsilon}");
        eprintln!("{}", scale8_bits);
    }
}

pub fn epsilon_check_rug(
    squared_error: &Integer,
    scale_factor_bits: usize,
    max_epsilon: f64,
    verbose: bool,
) {
    let num_dbg = Integer::from_str(&squared_error.to_string()).unwrap();
    let scale8_bits = 8 * scale_factor_bits;
    let scale8 = Integer::from(1) << scale8_bits;
    let denom_dbg = Float::with_val(256, scale8);
    let computed_epsilon = (num_dbg / denom_dbg).sqrt();
    let max_epsilon = Float::with_val(256, max_epsilon);

    if computed_epsilon > max_epsilon {
        panic!(
            "Computed model deviation {computed_epsilon} is greater than acceptable value {max_epsilon}"
        );
    }

    if verbose {
        eprintln!("Error epsilon: {computed_epsilon}");
        eprintln!("{}", scale8_bits);
    }
}

// Verifier uses "smart" mat evaluation to compute eval points specified by row and column challenges of A/B/C matrices
pub fn smart_r1cs_eval(
    mont: &FieldMont,
    tensors_mont: &R1CSMatrices<FieldElem>,
    r_row: &[FieldElem],
    r_col: &[FieldElem],
    ranges: &[Range<usize>],
) -> (FieldElem, FieldElem, FieldElem) {
    let r_row_tbl = build_eval_tbl(mont, r_row);
    let r_col_tbl = build_compressed_eval_tbl(mont, r_col, &remap_ranges(ranges));
    let a_eval = smart_mat_eval(mont, &tensors_mont.a, &r_row_tbl, &r_col_tbl);
    let b_eval = smart_mat_eval(mont, &tensors_mont.b, &r_row_tbl, &r_col_tbl);
    let c_eval = smart_mat_eval(mont, &tensors_mont.c, &r_row_tbl, &r_col_tbl);
    (a_eval, b_eval, c_eval)
}

// Checks that a random linear combo of a/b/c evals matches some claim
pub fn check_r1cs_claim(
    mont: &FieldMont,
    a_eval: FieldElem,
    b_eval: FieldElem,
    c_eval: FieldElem,
    r1: FieldElem,
    r2: FieldElem,
    claim: FieldElem,
) {
    if mont.add(mont.add(a_eval, mont.mul(r1, b_eval)), mont.mul(r2, c_eval)) != claim {
        panic!("WARNING: Matrix evaluation does not match claim");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn check_claim(
    mont: &FieldMont,
    tensors_mont: &R1CSMatrices<FieldElem>,
    r_row: &[FieldElem],
    r_col: &[FieldElem],
    ranges: &[Range<usize>],
    r1: FieldElem,
    r2: FieldElem,
    claim: FieldElem,
) {
    let (a_eval, b_eval, c_eval) = smart_r1cs_eval(mont, tensors_mont, r_row, r_col, ranges);
    check_r1cs_claim(mont, a_eval, b_eval, c_eval, r1, r2, claim);
}

fn eval_opened_z_range<T: HighPrecisionInt>(
    mont: &FieldMont,
    opening: &ZRangeOpening<T>,
    range_len: usize,
    row_evals: &[FieldElem],
    col_evals: &[FieldElem],
    den_mont: FieldElem,
) -> FieldElem {
    assert_eq!(
        opening.values.len(),
        range_len * opening.width,
        "opening value count mismatch for range {}",
        opening.range_index
    );

    let mut res = mont.zero();
    for (local_r, &row_eval) in row_evals.iter().enumerate().take(range_len) {
        let src_base = local_r * opening.width;
        let mut row_sum = mont.zero();
        for (c, val) in opening.values[src_base..src_base + opening.width]
            .iter()
            .enumerate()
        {
            let value = mont.mul(val.to_field_elem(mont), den_mont);
            row_sum = mont.add(row_sum, mont.mul(value, col_evals[c]));
        }
        res = mont.add(res, mont.mul(row_eval, row_sum));
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn combine_final_claim_evals(
    mont: &FieldMont,
    ez0: FieldElem,
    ez1: FieldElem,
    ez2: Option<FieldElem>,
    ez3: Option<FieldElem>,
    r_col: &[FieldElem],
    has_randomness: bool,
    has_secondary_constraints: bool,
) -> FieldElem {
    if has_secondary_constraints {
        let ez2 = ez2.expect("missing secondary witness evaluation");
        let ez3 = ez3.expect("missing secondary constraint evaluation");
        let r_col_e1 = r_col[r_col.len() - 1];
        let r_col_e2 = r_col[r_col.len() - 2];
        let r_col_e1m1 = mont.sub(mont.one(), r_col_e1);
        let r_col_e2m1 = mont.sub(mont.one(), r_col_e2);
        mont.add(
            mont.mul(
                r_col_e1m1,
                mont.add(mont.mul(r_col_e2m1, ez0), mont.mul(r_col_e2, ez1)),
            ),
            mont.mul(
                r_col_e1,
                mont.add(mont.mul(r_col_e2m1, ez2), mont.mul(r_col_e2, ez3)),
            ),
        )
    } else if has_randomness {
        let ez2 = ez2.expect("missing secondary witness evaluation");
        let r_col_e1 = r_col[r_col.len() - 1];
        let r_col_e2 = r_col[r_col.len() - 2];
        let r_col_e1m1 = mont.sub(mont.one(), r_col_e1);
        let r_col_e2m1 = mont.sub(mont.one(), r_col_e2);
        mont.add(
            mont.mul(
                r_col_e1m1,
                mont.add(mont.mul(r_col_e2m1, ez0), mont.mul(r_col_e2, ez1)),
            ),
            mont.mul(r_col_e1, mont.mul(r_col_e2m1, ez2)),
        )
    } else {
        let r_col_e1 = r_col[r_col.len() - 1];
        let r_col_e1m1 = mont.sub(mont.one(), r_col_e1);
        mont.add(mont.mul(r_col_e1m1, ez0), mont.mul(r_col_e1, ez1))
    }
}

// Verifier checks the final claim by evaluating the z mle.
// Shared portions of the mle are evaluated manually and interpolated with the dark claim for MLE representing the
// primary witness value range
#[allow(clippy::too_many_arguments)]
pub fn check_final_claim(
    mont: &FieldMont,
    ez1: FieldElem,
    z_mont: &Matrix<FieldElem>,
    r_col: &[FieldElem],
    r_z_col: &[FieldElem],
    claims: &[FieldElem],
    ranges: &[Range<usize>],
    has_randomness: bool,
    has_secondary_constraints: bool,
) {
    let r_z_col_tbl = build_eval_tbl(mont, r_z_col);
    let r_col_tbl = if has_randomness {
        build_eval_tbl(mont, &r_col[..r_col.len() - 2])
    } else {
        let eval_tbl_time = std::time::Instant::now();
        let res = build_eval_tbl(mont, &r_col[..r_col.len() - 1]);
        dbg!("build eval tbl time: {:?}", eval_tbl_time.elapsed());
        res
    };
    let ez0 = smart_mat_eval_sub_range(mont, z_mont, &ranges[0], &r_col_tbl, &r_z_col_tbl);
    let ez2 = if has_randomness {
        Some(smart_mat_eval_sub_range(
            mont,
            z_mont,
            &ranges[2],
            &r_col_tbl,
            &r_z_col_tbl,
        ))
    } else {
        None
    };
    let ez3 = if has_secondary_constraints {
        Some(smart_mat_eval_sub_range(
            mont,
            z_mont,
            &ranges[3],
            &r_col_tbl,
            &r_z_col_tbl,
        ))
    } else {
        None
    };
    let z_eval = combine_final_claim_evals(
        mont,
        ez0,
        ez1,
        ez2,
        ez3,
        r_col,
        has_randomness,
        has_secondary_constraints,
    );
    assert!(
        z_eval == claims[1],
        "Z matrix evaluation does not match claim"
    );
}

#[allow(clippy::too_many_arguments)]
pub fn check_final_claim_from_openings<T: HighPrecisionInt>(
    mont: &FieldMont,
    ez1: FieldElem,
    z_openings: &[ZRangeOpening<T>],
    den_mont: FieldElem,
    z_width: usize,
    r_col: &[FieldElem],
    r_z_col: &[FieldElem],
    claims: &[FieldElem],
    ranges: &[Range<usize>],
    has_randomness: bool,
    has_secondary_constraints: bool,
) {
    let r_z_col_tbl = build_eval_tbl(mont, r_z_col);
    let r_col_tbl = if has_randomness {
        build_eval_tbl(mont, &r_col[..r_col.len() - 2])
    } else {
        build_eval_tbl(mont, &r_col[..r_col.len() - 1])
    };
    let mut ez0 = None;
    let mut ez2 = None;
    let mut ez3 = None;

    for opening in z_openings {
        assert_eq!(
            opening.width, z_width,
            "opening width mismatch for range {}",
            opening.range_index
        );
        let range = ranges
            .get(opening.range_index)
            .expect("opening range index out of bounds");
        assert_eq!(
            range.len() * z_width,
            opening.values.len(),
            "opening value count mismatch for range {}",
            opening.range_index
        );

        match opening.range_index {
            0 => {
                ez0 = Some(eval_opened_z_range(
                    mont,
                    opening,
                    range.len(),
                    &r_col_tbl,
                    &r_z_col_tbl,
                    den_mont,
                ));
            }
            2 if has_randomness => {
                ez2 = Some(eval_opened_z_range(
                    mont,
                    opening,
                    range.len(),
                    &r_col_tbl,
                    &r_z_col_tbl,
                    den_mont,
                ));
            }
            3 if has_secondary_constraints => {
                ez3 = Some(eval_opened_z_range(
                    mont,
                    opening,
                    range.len(),
                    &r_col_tbl,
                    &r_z_col_tbl,
                    den_mont,
                ));
            }
            _ => {}
        }
    }

    let z_eval = combine_final_claim_evals(
        mont,
        ez0.expect("missing public opening"),
        ez1,
        ez2,
        ez3,
        r_col,
        has_randomness,
        has_secondary_constraints,
    );
    assert!(
        z_eval == claims[1],
        "Z matrix evaluation does not match claim"
    );
}
