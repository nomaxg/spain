use std::f64::consts::PI;

use model::{AFloat, FromPrimitive};
use parse::mat::Matrix;
use spain::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};

use crate::{
    r1cs::{ConstraintGenerator, R1CSExec, WitnessGenerator},
    utils::{
        AP7ROT, AZIMUTH, COS_FACE_LAT, FACE_CENTER_GEO_LNG, FACE_CENTER_POINT, RESCONST,
        SIN_FACE_LAT, SIN60,
    },
};

pub fn circuit_zklp<E: R1CSExec>(exec: &mut E) -> (E::Atom, E::Atom, E::Atom) {
    // TODO Deal with all the clones?
    // INPUTS
    let lat = exec.add_input();
    let lng = exec.add_input();
    let resolution = exec.add_input();
    // resolution is seen as an integer, even if we treat it as float; this is OK because it's public
    let input_i = exec.add_input();
    let input_j = exec.add_input();
    let input_k = exec.add_input();
    // "HINTS"
    let alpha_lat = exec.add_input();
    let beta_lat = exec.add_input();
    let gamma_lat = exec.add_input();
    let delta_lat = exec.add_input();
    let alpha_lng = exec.add_input();
    let beta_lng = exec.add_input();
    let gamma_lng = exec.add_input();
    let delta_lng = exec.add_input();
    // OUTPUTS
    // no outputs; just circuit sat

    let pi = E::constant(PI);
    let half_pi = E::constant(PI / 2.);
    let max_resolution = E::constant(15.);
    exec.assert_greater_equal_than(half_pi, lat.clone());
    exec.assert_greater_equal_than(pi, lng.clone());
    exec.assert_greater_equal_than(max_resolution, resolution.clone());

    let delta_lat_squared = exec.mul(delta_lat.clone(), delta_lat.clone());
    let gamma_lat_squared = exec.mul(gamma_lat.clone(), gamma_lat.clone());
    let identity_lat = exec.add(gamma_lat_squared.clone(), delta_lat_squared.clone());
    let delta_lng_squared = exec.mul(delta_lng.clone(), delta_lng.clone());
    let gamma_lng_squared = exec.mul(gamma_lng.clone(), gamma_lng.clone());
    let identity_lng = exec.add(gamma_lng_squared.clone(), delta_lng_squared.clone());
    // identity == 1
    exec.assert_equal(identity_lat, E::constant(1.));
    exec.assert_equal(identity_lng, E::constant(1.));
    // alpha * delta = gamma
    let tmp = exec.mul(alpha_lat.clone(), delta_lat.clone());
    exec.assert_equal(tmp, gamma_lat.clone());
    let tmp = exec.mul(alpha_lng.clone(), delta_lng.clone());
    exec.assert_equal(tmp, gamma_lng.clone());
    // 2 * gamma * delta = beta
    let tmp = exec.mul(gamma_lat, delta_lat) * 2.;
    exec.assert_equal(beta_lat.clone(), tmp);
    let tmp = exec.mul(gamma_lng, delta_lng) * 2.;
    exec.assert_equal(beta_lng.clone(), tmp);

    let cos_lat = exec.add(delta_lat_squared, gamma_lat_squared * -1.);
    let cos_lng = exec.add(delta_lng_squared, gamma_lng_squared * -1.);
    let z = beta_lat.clone();
    let sin_lng = beta_lng.clone();
    let x = exec.mul(cos_lat.clone(), cos_lng.clone());
    let y = exec.mul(cos_lat.clone(), sin_lng.clone());

    // Closest Face Calcuation
    let mut sq_dist = E::constant(5.0); // calc[0]
    let mut sin_face_lat = E::constant(0.); // calc[1]
    let mut cos_face_lat = E::constant(0.); // so on...
    let mut sin_face_lng = E::constant(0.);
    let mut cos_face_lng = E::constant(0.);
    let mut sin_az = E::constant(0.);
    let mut cos_az = E::constant(0.);
    let mut sin_az_rot = E::constant(0.);
    let mut cos_az_rot = E::constant(0.);
    for i in (0..60).step_by(3) {
        let d = exec.add(E::constant(FACE_CENTER_POINT[i]), x.clone() * -1.);
        let s1 = exec.mul(d.clone(), d);
        let d = exec.add(E::constant(FACE_CENTER_POINT[i + 1]), y.clone() * -1.);
        let s2 = exec.mul(d.clone(), d);
        let d = exec.add(E::constant(FACE_CENTER_POINT[i + 2]), z.clone() * -1.);
        let s3 = exec.mul(d.clone(), d);
        let tmp = exec.add(s1, s2);
        let dist = exec.add(tmp, s3);
        let check = exec.greater_equal_than(sq_dist.clone(), dist.clone());
        let face = i / 3;
        sq_dist = exec.select(check.clone(), dist.clone(), sq_dist.clone());
        sin_face_lat = exec.select(check.clone(), E::constant(SIN_FACE_LAT[face]), sin_face_lat);
        cos_face_lat = exec.select(check.clone(), E::constant(COS_FACE_LAT[face]), cos_face_lat);
        sin_face_lng = exec.select(
            check.clone(),
            E::constant(f64::sin(FACE_CENTER_GEO_LNG[face])),
            sin_face_lng,
        );
        cos_face_lng = exec.select(
            check.clone(),
            E::constant(f64::cos(FACE_CENTER_GEO_LNG[face])),
            cos_face_lng,
        );
        sin_az = exec.select(check.clone(), E::constant(f64::sin(AZIMUTH[face])), sin_az);
        cos_az = exec.select(check.clone(), E::constant(f64::cos(AZIMUTH[face])), cos_az);
        sin_az_rot = exec.select(
            check.clone(),
            E::constant(f64::sin(AZIMUTH[face] - AP7ROT)),
            sin_az_rot,
        );
        cos_az_rot = exec.select(
            check.clone(),
            E::constant(f64::cos(AZIMUTH[face] - AP7ROT)),
            cos_az_rot,
        );
    }

    // Calculate R
    let tmp = exec.add(E::constant(4.), sq_dist.clone() * -1.);
    let nominator = exec.mul(tmp, sq_dist.clone());
    let divisor = exec.add(E::constant(2.), sq_dist.clone() * -1.);
    let sqr_nom = exec.sqrt(nominator);
    let quotient = exec.div(sqr_nom, divisor);
    let r = exec.div(quotient, E::constant(RESCONST));

    // Scale R, used the dot product select into by Zach
    let multiplier = exec.select_pow_of_sqrt7(resolution.clone());
    let r = exec.mul(multiplier, r);

    // Calculate Hex 2D
    let is_class_3 = exec.select_is_class_3(resolution.clone());
    let sin_lat = z;
    let tmp1 = exec.mul(sin_lng.clone(), cos_face_lng.clone());
    let tmp2 = exec.mul(cos_lng.clone(), sin_face_lng.clone());
    let tmp3 = exec.add(tmp1, tmp2 * -1.);
    let y = exec.mul(cos_lat.clone(), tmp3);
    let tmp1 = exec.mul(cos_lng.clone(), cos_face_lng.clone());
    let tmp2 = exec.mul(sin_lng.clone(), sin_face_lng.clone());
    let tmp3 = exec.add(tmp1, tmp2); // f.Add(f.Mul(cosLng, cosFaceLng), f.Mul(sinLng, sinFaceLng)),
    let tmp4 = exec.mul(sin_face_lat.clone(), cos_lat.clone());
    let tmp5 = exec.mul(tmp3, tmp4);
    let tmp6 = exec.mul(cos_face_lat.clone(), sin_lat.clone());
    let x = exec.add(tmp6, tmp5 * -1.);
    let sin_az = exec.select(is_class_3.clone(), sin_az_rot, sin_az);
    let cos_az = exec.select(is_class_3.clone(), cos_az_rot, cos_az);
    let tmp1 = exec.mul(x.clone(), x.clone());
    let tmp2 = exec.mul(y.clone(), y.clone());
    let tmp3 = exec.add(tmp1, tmp2);
    let z = exec.sqrt(tmp3);
    let sin_p = exec.div(y.clone(), z.clone());
    let cos_p = exec.div(x.clone(), z.clone());
    let tmp1 = exec.mul(sin_az.clone(), cos_p.clone());
    let tmp2 = exec.mul(cos_az.clone(), sin_p.clone());
    let sin = exec.add(tmp1, tmp2 * -1.);
    let tmp1 = exec.mul(cos_az.clone(), cos_p.clone());
    let tmp2 = exec.mul(sin_az.clone(), sin_p.clone());
    let cos = exec.add(tmp1, tmp2);
    let hex2d_x = exec.mul(cos, r.clone());
    let hex2d_y = exec.mul(sin, r.clone());

    // Hex2d to Coord IJK
    let a1 = exec.abs(hex2d_x.clone());
    let a2 = exec.abs(hex2d_y.clone());
    let x2 = a2 * (1. / SIN60);
    let tmp = x2.clone() * (1. / 2.);
    let x1 = exec.add(a1, tmp);

    let m1 = exec.floor_32(x1.clone());
    let m2 = exec.floor_32(x2.clone());
    let r1 = exec.add(x1.clone(), m1.clone() * -1.);
    let r2 = exec.add(x2.clone(), m2.clone() * -1.);
    let double_r1 = r1.clone() * 2.;
    let m1_plus_one = m1.clone() + 1.;
    let m2_plus_one = m2.clone() + 1.;

    let r1_case_a = exec.greater_equal_than(E::constant(0.5), r1.clone());
    let r1_case_a1 = exec.greater_equal_than(E::constant(1. / 3.), r1.clone());
    let r1_case_b1 = exec.greater_equal_than(E::constant(2. / 3.), r1.clone());
    let one_minus = r1.clone() * -1. + 1.;
    let i_case_a2_first = exec.greater_equal_than(r2.clone(), one_minus.clone());
    let i_case_a2_second = exec.greater_equal_than(double_r1.clone(), r2.clone());
    let double_one_minus = double_r1.clone() - 1.;
    let i_case_b1_first = exec.greater_equal_than(r1.clone(), double_one_minus);
    let i_case_b1_second = exec.greater_equal_than(one_minus.clone(), r2.clone());

    let a_inner_most = exec.select(i_case_a2_second, m1_plus_one.clone(), m1.clone());
    let a_inner = exec.select(i_case_a2_first, a_inner_most, m1.clone());
    let branch_a = exec.select(r1_case_a1.clone(), m1.clone(), a_inner);
    let b_inner_most = exec.select(i_case_b1_second, m1.clone(), m1_plus_one.clone());
    let b_inner = exec.select(i_case_b1_first, b_inner_most, m1_plus_one.clone());
    let branch_b = exec.select(r1_case_b1.clone(), b_inner, m1_plus_one.clone());
    let i_coord = exec.select(r1_case_a.clone(), branch_a, branch_b);

    let one_plus = r1.clone() + 1.0;
    let value_r2_path_a = one_plus.clone() * (1. / 2.0);
    let value_r2_path_b = r1.clone() * (1. / 2.0);
    let check1 = exec.greater_equal_than(value_r2_path_a, r2.clone());
    let check2 = exec.greater_equal_than(one_minus.clone(), r2.clone());
    let check3 = exec.greater_equal_than(value_r2_path_b, r2);
    let case_a_inner_true = exec.select(check1, m2.clone(), m2_plus_one.clone());
    let case_a_inner_false = exec.select(check2.clone(), m2.clone(), m2_plus_one.clone());
    let case_a_j_coord = exec.select(r1_case_a1.clone(), case_a_inner_true, case_a_inner_false);
    let case_b_inner_true = exec.select(check2, m2.clone(), m2_plus_one.clone());
    let case_b_inner_false = exec.select(check3, m2.clone(), m2_plus_one.clone());
    let case_b_j_coord = exec.select(r1_case_b1, case_b_inner_true, case_b_inner_false);
    let j_coord = exec.select(r1_case_a, case_a_j_coord, case_b_j_coord);

    let x_sign = exec.greater_equal_than(E::constant(0.), hex2d_x.clone());
    let y_sign = exec.greater_equal_than(E::constant(0.), hex2d_y.clone());
    let i_greater = exec.greater_equal_than(i_coord.clone() - j_coord.clone(), E::constant(0.));
    let one_minus_i_greater = E::constant(1.0) - i_greater.clone();
    let icoord_neg_x_true = exec.select(y_sign.clone(), E::constant(1.0), i_greater.clone());
    let icoord_neg_x_false = exec.select(y_sign.clone(), one_minus_i_greater, E::constant(0.0));
    let i_coord_negative = exec.select(x_sign.clone(), icoord_neg_x_true, icoord_neg_x_false);
    let diff_select = exec.abs(i_coord.clone() - j_coord.clone());
    let icoord_x_true = exec.select(y_sign.clone(), i_coord.clone(), diff_select.clone());
    let icoord_x_false = exec.select(y_sign.clone(), diff_select, i_coord);
    let i_coord = exec.select(x_sign.clone(), icoord_x_true, icoord_x_false);

    let j_coord_negative = y_sign;
    let i_greater_j = exec.greater_equal_than(i_coord.clone() - j_coord.clone(), E::constant(0.));
    let j_tmp_inner = exec.abs(i_coord.clone() - j_coord.clone());
    let j_tmp = exec.select(
        j_coord_negative.clone(),
        j_tmp_inner,
        i_coord.clone() + j_coord.clone(),
    );
    let j_tmp_negative = exec.select(
        j_coord_negative.clone(),
        E::constant(1.0) - i_greater_j,
        E::constant(0.0),
    );

    let k_coord = E::constant(0.);
    let k_tmp = i_coord.clone() + k_coord.clone();
    let k_tmp_negative = E::constant(0.0);

    // if i < 0
    let i_coord = exec.select(i_coord_negative.clone(), E::constant(0.0), i_coord);
    let j_coord = exec.select(i_coord_negative.clone(), j_tmp, j_coord);
    let j_coord_negative = exec.select(i_coord_negative.clone(), j_tmp_negative, j_coord_negative);
    let k_coord = exec.select(i_coord_negative.clone(), k_tmp, k_coord);
    let k_coord_negative = exec.select(i_coord_negative, k_tmp_negative, E::constant(0.));

    let j_greater_k = exec.greater_equal_than(j_coord.clone() - k_coord.clone(), E::constant(0.));
    let k_tmp_inner_2 = exec.abs(j_coord.clone() - k_coord.clone());
    let k_tmp = exec.select(
        k_coord_negative.clone(),
        k_tmp_inner_2,
        j_coord.clone() + k_coord.clone(),
    );
    let k_tmp_negative = exec.select(
        k_coord_negative.clone(),
        E::constant(1.0) - j_greater_k,
        E::constant(0.0),
    );

    // if j < 0
    let i_coord = exec.select(
        j_coord_negative.clone(),
        i_coord.clone() + j_coord.clone(),
        i_coord,
    );
    let j_coord = exec.select(j_coord_negative.clone(), E::constant(0.0), j_coord);
    let k_coord = exec.select(j_coord_negative.clone(), k_tmp, k_coord);
    let k_coord_negative = exec.select(j_coord_negative, k_tmp_negative, k_coord_negative);

    // if k < 0
    let i_coord = exec.select(
        k_coord_negative.clone(),
        i_coord.clone() + k_coord.clone(),
        i_coord,
    );
    let j_coord = exec.select(
        k_coord_negative.clone(),
        j_coord.clone() + k_coord.clone(),
        j_coord,
    );
    let k_coord = exec.select(k_coord_negative, E::constant(0.0), k_coord);

    let i_greater_j = exec.greater_equal_than(i_coord.clone() - j_coord.clone(), E::constant(0.));
    let min = exec.select(i_greater_j, j_coord.clone(), i_coord.clone());
    let min_greater_k = exec.greater_equal_than(min.clone() - k_coord.clone(), E::constant(0.));
    let min = exec.select(min_greater_k, k_coord.clone(), min);

    let i = i_coord - min.clone();
    let j = j_coord - min.clone();
    let k = k_coord - min;

    exec.assert_equal(i.clone(), input_i);
    exec.assert_equal(j.clone(), input_j);
    exec.assert_equal(k.clone(), input_k);

    (i, j, k)
}

#[derive(Clone, Debug)]
pub struct ZKLPExecutor {
    witness: Option<Matrix<i128>>,
    witness_gen: WitnessGenerator,
}

impl ZKLPExecutor {
    pub fn new(
        lat: f64,
        lng: f64,
        res: u64,
        result_i: u64,
        result_j: u64,
        result_k: u64,
        alpha_lat: f64,
        beta_lat: f64,
        gamma_lat: f64,
        delta_lat: f64,
        alpha_lng: f64,
        beta_lng: f64,
        gamma_lng: f64,
        delta_lng: f64,
    ) -> Self {
        let mut wit_gen = WitnessGenerator::new(vec![
            lat,
            lng,
            res as f64,
            result_i as f64,
            result_j as f64,
            result_k as f64,
            alpha_lat,
            beta_lat,
            gamma_lat,
            delta_lat,
            alpha_lng,
            beta_lng,
            gamma_lng,
            delta_lng,
        ]);
        let _ret = circuit_zklp(&mut wit_gen);

        ZKLPExecutor {
            witness: None,
            witness_gen: wit_gen,
        }
    }
}

impl R1CSInstance<f64, i128> for ZKLPExecutor {
    fn get_matrices(
        &self,
        scale_factor: f64,
        _randomness: Option<&Vec<i128>>,
    ) -> (
        R1CSMatrices<i128>,
        Option<parse::generalized::InjectionInfo>,
    ) {
        let mut cons_gen = ConstraintGenerator::new();
        circuit_zklp(&mut cons_gen);
        let constraints = cons_gen.finish();
        let (a, b, c) = constraints.to_r1cs_int::<i128>(AFloat::from_f64(scale_factor).unwrap());
        (R1CSMatrices { a, b, c }, None)
    }

    fn get_meta(&self) -> Metadata {
        self.witness_gen.metadata()
    }

    fn compute_commit_witness(&mut self, scale_factor: f64, batch_size: usize) -> Matrix<i128> {
        let mut batched_witness = vec![];
        for _ in 0..batch_size {
            let witness = self
                .witness_gen
                .witness_int::<i128>(AFloat::from_f64(scale_factor).unwrap());
            batched_witness.push(witness);
        }
        let ranges = batched_witness[0].ranges().unwrap();
        let mut batched_witness =
            Matrix::stack_dense_matrices_horizontally(batched_witness.iter().collect());
        batched_witness.set_ranges(ranges);
        self.witness = Some(batched_witness);
        let ret = self.witness.as_ref().unwrap();
        let ret = ret.extract_rows(&ret.ranges().unwrap()[1]);
        ret
    }

    fn compute_full_witness(
        &mut self,
        _metadata: &Metadata,
        _random_values: Vec<f64>,
        _scale_factor: f64,
    ) -> Matrix<i128> {
        return self.witness.take().unwrap();
    }
}
