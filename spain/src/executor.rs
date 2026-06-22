use std::path::PathBuf;

use eval_utils::{ComputationType, get_instance_size_lower_bound};
use model::HighPrecision;
use parse::{
    generalized::{HighPrecisionInt, I256, InjectionInfo},
    mat::Matrix,
};

use crate::{
    inputs::{FromF64Matrix, Metadata, R1CSMatrices},
    synthetic::SyntheticR1CS,
    traits::R1CSInstance,
    witness_gen::OnnxExecutor,
};

#[derive(Clone)]
pub enum SpainExecutor<P: HighPrecision, T: HighPrecisionInt> {
    Onnx(OnnxExecutor<P>),
    Synthetic(SyntheticR1CS<T>),
}

impl<P, T> R1CSInstance<P, T> for SpainExecutor<P, T>
where
    P: HighPrecision,
    T: HighPrecisionInt + FromF64Matrix,
{
    fn get_matrices(
        &self,
        scale_factor: P,
        randomness: Option<&Vec<T>>,
    ) -> (R1CSMatrices<T>, Option<InjectionInfo>) {
        match self {
            Self::Onnx(exec) => exec.get_matrices(scale_factor, randomness),
            Self::Synthetic(exec) => exec.get_matrices(scale_factor, randomness),
        }
    }

    fn get_meta(&self) -> Metadata {
        match self {
            Self::Onnx(exec) => <OnnxExecutor<P> as R1CSInstance<P, T>>::get_meta(exec),
            Self::Synthetic(exec) => <SyntheticR1CS<T> as R1CSInstance<P, T>>::get_meta(exec),
        }
    }

    fn compute_commit_witness(&mut self, scale_factor: P, batch_size: usize) -> Matrix<T> {
        match self {
            Self::Onnx(exec) => exec.compute_commit_witness(scale_factor, batch_size),
            Self::Synthetic(exec) => exec.compute_commit_witness(scale_factor, batch_size),
        }
    }

    fn compute_full_witness(
        &mut self,
        metadata: &Metadata,
        random_values: Vec<P>,
        scale_factor: P,
    ) -> Matrix<T> {
        match self {
            Self::Onnx(exec) => exec.compute_full_witness(metadata, random_values, scale_factor),
            Self::Synthetic(exec) => {
                exec.compute_full_witness(metadata, random_values, scale_factor)
            }
        }
    }
}

pub fn build_spain_executor<P: HighPrecision>(
    model: String,
    data_dir: PathBuf,
    use_same_input: bool,
) -> (SpainExecutor<P, i128>, Metadata) {
    let metadata = crate::inputs::import_metadata(&data_dir, &model);
    let exec = OnnxExecutor::new(model, data_dir, metadata.clone(), use_same_input);
    (SpainExecutor::Onnx(exec), metadata)
}

pub fn build_zklp_executor<P: HighPrecision>(
    model: String,
    _data_dir: PathBuf,
    scale_factor_bits: usize,
) -> (SpainExecutor<P, I256>, Metadata) {
    let computation_type = computation_type_for_model(&model)
        .unwrap_or_else(|| panic!("unsupported model for --zklp sizing: {model}"));
    let size = get_instance_size_lower_bound(computation_type);
    let exec = SyntheticR1CS::<I256>::new(size.target_num_cons, size.num_inputs, scale_factor_bits);
    let metadata = exec.get_metadata();
    (SpainExecutor::Synthetic(exec), metadata)
}

fn computation_type_for_model(model: &str) -> Option<ComputationType> {
    match model {
        "layernorm-32x768" => Some(ComputationType::LayerNorm),
        "gelu-32x3072" => Some(ComputationType::Gelu),
        "softmax-32x32" => Some(ComputationType::Softmax),
        "physics-d8-t10" => Some(ComputationType::PhysicsSmall),
        "physics-d16-t10" => Some(ComputationType::PhysicsLarge),
        "debug" => Some(ComputationType::Debug),
        _ => None,
    }
}
