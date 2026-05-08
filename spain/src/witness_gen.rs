use crate::{
    Witness,
    inputs::{FromF64Matrix, Metadata, R1CSMatrices, data_file, import_full_r1cs},
    traits::R1CSInstance,
};
use model::{AFloat, HighPrecision, OnnxRunner, Tensor};
use ndarray::ArrayD;
use parse::{
    generalized::{HighPrecisionInt, InjectionInfo},
    mat::{Matrix, MatrixData},
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct OnnxExecutor<P: HighPrecision> {
    pub model: String,
    pub data_dir: PathBuf,
    pub metadata: Metadata,
    pub use_same_input: bool,
    pub partial_witness: Option<Vec<Witness<P>>>,
    pub primary_output: Option<Vec<HashMap<String, ArrayD<P>>>>,
    pub primary_output_order: Option<Vec<String>>,
}

impl<P, T> R1CSInstance<P, T> for OnnxExecutor<P>
where
    P: HighPrecision,
    T: HighPrecisionInt + FromF64Matrix,
{
    fn compute_commit_witness(
        &mut self,
        scale_factor: P,
        batch_size: usize,
        // TODO should add input given from verifier here
    ) -> Matrix<T> {
        // TODO potentially, improve batch witness computing speed by moving batching inside onnx
        self.partial_witness = Some(vec![]);
        self.primary_output = Some(vec![]);
        let mut ret = vec![];
        let mut runner_primary = OnnxRunner::<P>::read(
            data_file(
                &self.data_dir,
                format!("{}/primary_model.onnx", &self.model).as_str(),
            )
            .to_str()
            .unwrap(),
        )
        .unwrap();
        let primary_output_order = runner_primary.get_output_order();
        for _ in 0..batch_size {
            let mut witness = Witness::<P>::default();

            let primary_input = runner_primary.rand_input();
            let primary_input_order = runner_primary.get_input_order();
            let primary_output = runner_primary.run(primary_input.clone());
            primary_input_order.iter().for_each(|name| {
                primary_input[name]
                    .data
                    .as_ref()
                    .unwrap()
                    .iter()
                    .for_each(|value| witness.extend("input", value.clone(), name));
            });
            primary_output_order
                .iter()
                .zip(self.metadata.primary_output_labels.iter())
                .for_each(|(name, label)| {
                    primary_output[name]
                        .iter()
                        .for_each(|value| witness.extend(label, value.clone(), name));
                });
            ret.push(witness.to_scaled_primary_matrix::<T>(scale_factor.clone()));
            self.primary_output.as_mut().unwrap().push(primary_output);
            self.partial_witness.as_mut().unwrap().push(witness);
        }
        self.primary_output_order = Some(primary_output_order);
        Matrix::stack_dense_matrices_horizontally(ret.iter().collect())
    }

    fn get_matrices(
        &self,
        scale_factor: P,
        randomness: Option<&Vec<T>>,
    ) -> (R1CSMatrices<T>, Option<InjectionInfo>) {
        import_full_r1cs(
            &self.data_dir,
            &self.model,
            AFloat(scale_factor.to_rug_float()),
            &self.metadata,
            randomness,
            true,
        )
    }

    fn get_meta(&self) -> Metadata {
        self.metadata.clone()
    }

    fn compute_full_witness(
        &mut self,
        metadata: &Metadata, // TODO: consider remove metadata b/c already in struct
        random_values: Vec<P>,
        scale_factor: P,
    ) -> Matrix<T> {
        let mut partial_witness = self
            .partial_witness
            .take()
            .expect("cannot calculate full witness before primary witness");
        let primary_output = self
            .primary_output
            .take()
            .expect("cannot calculate full witness before primary witness");
        let primary_output_order = self
            .primary_output_order
            .take()
            .expect("cannot calculate full witness before primary witness");
        let mut runner_secondary = OnnxRunner::<P>::read(
            data_file(
                &self.data_dir,
                format!("{}/secondary_model.onnx", &self.model).as_str(),
            )
            .to_str()
            .unwrap(),
        )
        .unwrap();
        for i in 0..partial_witness.len() {
            let mut secondary_input = HashMap::<String, Tensor<P>>::new();
            let randomness_shape = runner_secondary
                .get_randomness_shape()
                .expect("randomness is not part of secondary model input");
            secondary_input.insert(
                "R".to_string(),
                Tensor {
                    name: "R".to_string(),
                    data: Some(
                        ArrayD::from_shape_vec(randomness_shape.clone(), random_values.clone())
                            .unwrap(),
                    ),
                    dim: randomness_shape,
                },
            );
            primary_output_order
                .iter()
                .zip(metadata.primary_output_labels.iter())
                .filter(|(_name, label)| label.contains("second"))
                .enumerate()
                .for_each(|(index, (name, _label))| {
                    secondary_input.insert(
                        format!("Input_{}", index),
                        Tensor {
                            name: format!("Input_{}", index),
                            data: Some(primary_output[i][name].clone()),
                            dim: primary_output[i][name].shape().to_vec(),
                        },
                    );
                });
            let secondary_output_order = runner_secondary.get_output_order();
            let secondary_output = runner_secondary.run(secondary_input.clone());

            secondary_output_order
                .iter()
                .zip(metadata.secondary_output_labels.iter())
                .for_each(|(name, label)| {
                    secondary_output[name]
                        .iter()
                        .for_each(|value| partial_witness[i].extend(label, value.clone(), name));
                });
            secondary_input["R"]
                .data
                .as_ref()
                .unwrap()
                .iter()
                .for_each(|value| partial_witness[i].extend("random", value.clone(), "R"));
        }
        let ret = partial_witness
            .into_iter()
            .map(|v| v.into_matrix().0)
            .collect::<Vec<_>>();
        let mut ret = Matrix::stack_dense_matrices_horizontally(ret.iter().collect());
        ret.set_ranges(&metadata.get_ranges());
        Matrix::from_hp(&ret, scale_factor)
    }
}

impl<P: HighPrecision> OnnxExecutor<P> {
    pub fn new(model: String, data_dir: PathBuf, metadata: Metadata, use_same_input: bool) -> Self {
        Self {
            model,
            data_dir,
            metadata,
            use_same_input,
            primary_output: None,
            primary_output_order: None,
            partial_witness: None,
        }
    }
}

pub fn compute_squared_error_raw_hp<T: HighPrecision>(
    tensors: &R1CSMatrices<f64>,
    z: &Matrix<T>,
    names: &[String],
    verbose: bool,
) -> T {
    fn mul_to_vec<T: HighPrecision>(a: &Matrix<f64>, z: &Matrix<T>) -> Vec<T> {
        assert_eq!(
            a.width(),
            z.height(),
            "Incompatible matrix dimensions for multiplication"
        );
        let mut out = vec![T::from_f64(0.0f64).unwrap(); a.height() * z.width()];
        match (a.data(), z.data()) {
            (MatrixData::COO(a_entries), MatrixData::Dense(z_values)) => {
                let z_width = z.width();
                for &(r, c, val) in a_entries.iter() {
                    let row_offset = r * z_width;
                    let z_row_offset = c * z_width;
                    for col in 0..z_width {
                        out[row_offset + col] = out[row_offset + col].clone()
                            + T::from_f64(val).unwrap() * z_values[z_row_offset + col].clone();
                    }
                }
            }
            (_, _) => panic!("not supported"),
        }
        out
    }

    fn mat_to_name(a: &Matrix<f64>, names: &[String]) -> Vec<Vec<String>> {
        assert_eq!(a.width(), names.len());
        let mut ret: Vec<Vec<String>> = Vec::new();
        for _ in 0..a.height() {
            ret.push(Vec::new());
        }
        if let MatrixData::COO(a_entries) = a.data() {
            for (r, c, _val) in a_entries.iter() {
                ret[*r].push(names[*c].clone());
            }
        }
        ret
    }

    if verbose {
        println!("Computing high precision squared error (raw)");
    }

    let az = mul_to_vec(&tensors.a, z);
    let bz = mul_to_vec(&tensors.b, z);
    let cz = mul_to_vec(&tensors.c, z);

    let a_names = mat_to_name(&tensors.a, names);
    let b_names = mat_to_name(&tensors.b, names);
    let c_names = mat_to_name(&tensors.c, names);
    let all_names = a_names
        .into_iter()
        .zip(b_names)
        .zip(c_names)
        .map(|((mut a, b), c)| {
            a.extend(b);
            a.extend(c);
            a.dedup();
            a
        })
        .collect::<Vec<_>>();

    println!("The following onnx outputs, if any, results in low precision:");
    let mut error_contribution = HashMap::<String, f64>::new();
    az.iter()
        .zip(bz.iter())
        .zip(cz.iter())
        .enumerate()
        .for_each(|(index, ((a, b), c))| {
            let err = (a.clone() * b.clone()) - c.clone();
            let err = (err.clone() * err.clone()).to_f64().unwrap();
            all_names[index].iter().for_each(|x| {
                error_contribution
                    .entry(x.clone())
                    .and_modify(|v| *v += err)
                    .or_insert(0.);
            });
        });
    let mut error_contribution = error_contribution.into_iter().collect::<Vec<_>>();
    error_contribution.sort_by(|a, b| f64::total_cmp(&a.1, &b.1));
    error_contribution
        .iter()
        .filter(|(_, value)| *value > 1e-40)
        .for_each(|(name, value)| println!("{:?}: {:?}", name, value));

    az.iter()
        .zip(bz.iter())
        .zip(cz.iter())
        .map(|((a, b), c)| {
            let err = (a.clone() * b.clone()) - c.clone();
            err.clone() * err.clone()
        })
        .reduce(|acc, e| acc.clone() + e.clone())
        .unwrap()
}

pub fn get_precomputed_input(data_dir: &Path, model: &str, len: usize) -> Vec<f64> {
    let z = data_file(data_dir, format!("{}/Z.bin", model).as_str());
    let (_, z) = Matrix::<f64>::from_file(&z).expect("Failed to read Z");
    z.as_dense_vector()
        .iter()
        .skip(1)
        .take(len)
        .cloned()
        .collect()
}

pub fn witness_pairwise_equal_up_to_epsilon<T: HighPrecision>(
    witness: &Matrix<T>,
    gt: &Matrix<f64>,
) -> bool {
    const EPSILON: f64 = 1e-5;
    if let MatrixData::Dense(gt_data) = gt.data() {
        if let MatrixData::Dense(witness_data) = witness.data() {
            gt_data
                .iter()
                .zip(witness_data.iter())
                .all(|(a, b)| (a - b.to_f64().unwrap()).abs() < EPSILON)
        } else {
            false
        }
    } else {
        println!("data not of the same type!");
        false
    }
}

pub fn compute_witness_raw<T: HighPrecision>(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    use_same_input: bool,
    show_perf: bool,
) -> (Matrix<T>, Vec<String>, Witness<T>) {
    let mut witness = Witness::<T>::default();

    let mut runner_primary = OnnxRunner::<T>::read(
        data_file(data_dir, format!("{}/primary_model.onnx", model).as_str())
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let primary_input = if use_same_input {
        runner_primary.given_input(get_precomputed_input(
            data_dir,
            model,
            runner_primary.input_vec_len(),
        ))
    } else {
        runner_primary.rand_input()
    };
    let primary_input_order = runner_primary.get_input_order();
    let primary_output_order = runner_primary.get_output_order();
    let (primary_output, primary_perf) =
        runner_primary.run_with_perf_breakdown(primary_input.clone());

    let mut runner_secondary = OnnxRunner::<T>::read(
        data_file(data_dir, format!("{}/secondary_model.onnx", model).as_str())
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let mut secondary_input = runner_secondary.randn_input(); // NOTE randn needed for soundness
    primary_output_order
        .iter()
        .zip(metadata.primary_output_labels.iter())
        .filter(|(_name, label)| label.contains("second"))
        .enumerate()
        .for_each(|(index, (name, _label))| {
            secondary_input.insert(
                format!("Input_{}", index),
                Tensor {
                    name: format!("Input_{}", index),
                    data: Some(primary_output[name].clone()),
                    dim: primary_output[name].shape().to_vec(),
                },
            );
        });
    let secondary_output_order = runner_secondary.get_output_order();
    let (secondary_output, secondary_perf) =
        runner_secondary.run_with_perf_breakdown(secondary_input.clone());

    primary_input_order.iter().for_each(|name| {
        primary_input[name]
            .data
            .as_ref()
            .unwrap()
            .iter()
            .for_each(|value| witness.extend("input", value.clone(), name));
    });
    primary_output_order
        .iter()
        .zip(metadata.primary_output_labels.iter())
        .for_each(|(name, label)| {
            primary_output[name]
                .iter()
                .for_each(|value| witness.extend(label, value.clone(), name));
        });
    let partial_witness = witness.clone();
    secondary_output_order
        .iter()
        .zip(metadata.secondary_output_labels.iter())
        .for_each(|(name, label)| {
            secondary_output[name]
                .iter()
                .for_each(|value| witness.extend(label, value.clone(), name));
        });
    secondary_input["R"]
        .data
        .as_ref()
        .unwrap()
        .iter()
        .for_each(|value| witness.extend("random", value.clone(), "R"));

    if show_perf {
        eprintln!(
            "Compute Witness Perf Breakdown:\nPrimary Model:\n{:#?}\nSecondary Model:\n{:#?}",
            primary_perf, secondary_perf
        );
    }

    let (mut ret, names) = witness.into_matrix();
    ret.set_ranges(&metadata.get_ranges());
    (ret, names, partial_witness)
}

pub fn compute_witness<T: HighPrecision, Q: HighPrecisionInt>(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    scale_factor: T,
    use_same_input: bool,
    show_perf: bool,
) -> Matrix<Q> {
    let (witness, _, _) = compute_witness_raw(data_dir, model, metadata, use_same_input, show_perf);
    Matrix::from_hp(&witness, scale_factor)
}

pub fn compute_partial_witness<T: HighPrecision, Q: HighPrecisionInt>(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    scale_factor: T,
    use_same_input: bool,
    show_perf: bool,
) -> (Matrix<Q>, Witness<T>) {
    let (witness, _, partial_witness) =
        compute_witness_raw(data_dir, model, metadata, use_same_input, show_perf);
    (Matrix::from_hp(&witness, scale_factor), partial_witness)
}
