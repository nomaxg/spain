#![feature(f128)]
mod ops;
mod types;

use crate::ops::*;
pub use crate::types::{AFloat, F128, FBITS, HighPrecision, TFloat, ToPrimitiveExt};
use anyhow::{Context, Result};
use ndarray::{Array, ArrayD};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};
pub use num_traits::cast::{FromPrimitive, ToPrimitive};
use onnx_extractor::{DataType, OnnxModel};
use std::clone::Clone;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{self, Write};
use std::ops::{Add, Div, Mul, Sub};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Tensor<T: HighPrecision> {
    pub name: String,
    pub data: Option<ArrayD<T>>,
    pub dim: Vec<usize>,
}

pub struct OnnxRunner<T: HighPrecision> {
    pub model: OnnxModel,
    pub tensors: HashMap<String, Tensor<T>>,
    pub data_path: PathBuf,
}

impl<T> OnnxRunner<T>
where
    T: HighPrecision,
{
    pub fn read(path: &str) -> Result<Self> {
        let model = OnnxModel::load_from_file(path)?;
        let mut tensors = HashMap::new();
        for tensor_name in model.tensor_names() {
            let tensor = model
                .get_tensor(tensor_name)
                .context("tensor does not exist")?;
            tensors.insert(
                tensor.name().to_string(),
                Tensor {
                    name: tensor.name().to_string(),
                    data: None,
                    dim: tensor.shape().iter().map(|v| *v as usize).collect(),
                },
            );
        }
        for tensor in model.get_weight_tensors() {
            let shape = tensor
                .shape()
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>();
            let tensor_entry = tensors
                .get_mut(tensor.name())
                .context("tensor does not exist")?;
            tensor_entry.data = match tensor.data_type() {
                DataType::Float => Some(
                    Array::from_shape_vec(
                        shape,
                        tensor
                            .copy_data_as::<f32>()?
                            .iter()
                            .map(|v| T::from_f32(*v).unwrap())
                            .collect(),
                    )
                    .unwrap(),
                ),
                DataType::Double => Some(
                    Array::from_shape_vec(
                        shape,
                        tensor
                            .copy_data_as::<f64>()?
                            .iter()
                            .map(|v| T::from_f64(*v).unwrap())
                            .collect(),
                    )
                    .unwrap(),
                ),
                DataType::Int64 => Some(
                    Array::from_shape_vec(
                        shape,
                        tensor
                            .copy_data_as::<i64>()?
                            .iter()
                            .map(|v| T::from_f64(*v as f64).unwrap())
                            .collect(),
                    )
                    .unwrap(),
                ),
                x => {
                    eprintln!("unknown datatype: {:?}", x);
                    None
                }
            }
        }
        Ok(Self {
            model,
            tensors,
            data_path: PathBuf::from(path),
        })
    }

    pub fn get_input_order(&self) -> Vec<String> {
        let mut ret = Vec::new();
        for tensor in self.model.get_input_tensors() {
            ret.push(tensor.name().to_string());
        }
        ret
    }

    pub fn get_output_order(&self) -> Vec<String> {
        let mut ret = Vec::new();
        for tensor in self.model.get_output_tensors() {
            ret.push(tensor.name().to_string());
        }
        ret
    }

    pub fn get_randomness_shape(&self) -> Option<Vec<usize>> {
        self.model
            .get_tensor("R")
            .map(|x| x.shape().iter().map(|&v| v as usize).collect())
    }

    pub fn rand_input(&self) -> HashMap<String, Tensor<T>> {
        let mut ret = HashMap::new();
        for tensor in self.model.get_input_tensors() {
            let shape = tensor
                .shape()
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>();
            ret.insert(
                tensor.name().to_string(),
                Tensor {
                    name: tensor.name().to_string(),
                    data: Some(
                        Array::random(shape.clone(), Uniform::new(0., 10.).unwrap())
                            .mapv(|v| T::from_f64(v).unwrap()),
                    ),
                    dim: shape.clone(),
                },
            );
        }
        ret
    }

    pub fn randn_input(&self) -> HashMap<String, Tensor<T>> {
        let mut ret = HashMap::new();
        for tensor in self.model.get_input_tensors() {
            let shape = tensor
                .shape()
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>();
            ret.insert(
                tensor.name().to_string(),
                Tensor {
                    name: tensor.name().to_string(),
                    data: Some(
                        Array::random(shape.clone(), Normal::new(0., 1.).unwrap())
                            .mapv(|v| T::from_f64(v).unwrap()),
                    ),
                    dim: shape.clone(),
                },
            );
        }
        ret
    }

    pub fn given_input(&self, data: Vec<f64>) -> HashMap<String, Tensor<T>> {
        let mut ret = HashMap::new();
        let mut count = 0;
        for tensor in self.model.get_input_tensors() {
            let shape = tensor
                .shape()
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>();
            let len = shape.iter().copied().reduce(|acc, e| acc * e).unwrap();
            ret.insert(
                tensor.name().to_string(),
                Tensor {
                    name: tensor.name().to_string(),
                    data: Some(
                        ArrayD::from_shape_vec(
                            shape.clone(),
                            data.iter()
                                .skip(count)
                                .take(len)
                                .map(|&v| T::from_f64(v).unwrap())
                                .collect(),
                        )
                        .unwrap(),
                    ),
                    dim: shape.clone(),
                },
            );
            count += len;
        }
        ret
    }

    // the length of all input when flattned, used for getting the inputs from the python witness
    pub fn input_vec_len(&self) -> usize {
        let mut count = 0;
        for tensor in self.model.get_input_tensors() {
            count += tensor
                .shape()
                .iter()
                .copied()
                .reduce(|acc, e| acc * e)
                .unwrap() as usize;
        }
        count
    }

    // get all 1s input for testing
    pub fn all_ones_input(&self) -> HashMap<String, Tensor<T>> {
        let mut ret = HashMap::new();
        for tensor in self.model.get_input_tensors() {
            let shape = tensor
                .shape()
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>();
            ret.insert(
                tensor.name().to_string(),
                Tensor {
                    name: tensor.name().to_string(),
                    data: Some(
                        Array::from_shape_vec(
                            shape.clone(),
                            vec![T::from_f32(1.).unwrap(); shape.iter().product()],
                        )
                        .unwrap(),
                    ),
                    dim: shape.clone(),
                },
            );
        }
        ret
    }

    pub fn run(&mut self, input: HashMap<String, Tensor<T>>) -> HashMap<String, ArrayD<T>>
    where
        T: HighPrecision + Div<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        let (ret, _) = self.run_with_perf_breakdown(input);
        ret
    }

    pub fn run_with_perf_breakdown(
        &mut self,
        input: HashMap<String, Tensor<T>>,
    ) -> (HashMap<String, ArrayD<T>>, HashMap<String, Duration>)
    where
        T: HighPrecision + Div<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        self.run_with_perf_breakdown_retain(input, vec![])
    }

    pub fn run_with_perf_breakdown_retain(
        &mut self,
        input: HashMap<String, Tensor<T>>,
        retain: Vec<String>,
    ) -> (HashMap<String, ArrayD<T>>, HashMap<String, Duration>)
    where
        T: HighPrecision + Div<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        let mut perf_breakdown = HashMap::<String, Duration>::new();
        for tensor in self.model.get_input_tensors() {
            let tensor_name: String = tensor.name().to_string();
            let input_tensor = self.tensors.get_mut(&tensor_name).unwrap();
            input_tensor.data = input.get(&tensor_name).unwrap().data.clone();
        }
        let ops: Vec<_> = self.model.execution_order().unwrap().into_iter().collect();
        eprintln!("Executing {} ops: ", ops.len());
        for op in ops.iter() {
            let op_timer = Instant::now();
            match op.op_type.as_str() {
                "Slice" => {
                    op_slice(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.inputs[2],
                        if let Some(axes) = op.inputs.get(3) {
                            Some(axes.as_str())
                        } else {
                            None
                        },
                        if let Some(steps) = op.inputs.get(4) {
                            Some(steps.as_str())
                        } else {
                            None
                        },
                        &op.outputs[0],
                    );
                }
                "Add" => {
                    op_add(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Mul" => {
                    op_mul(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Sub" => {
                    op_sub(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Div" => {
                    op_div(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Identity" => {
                    op_identity(&mut self.tensors, &op.inputs[0], &op.outputs[0]);
                }
                "Constant" => {
                    op_constant(
                        &mut self.tensors,
                        op.attributes["value"].as_tensor().unwrap(),
                        &op.outputs[0],
                    );
                }
                "ReduceMean" => {
                    op_reduce_mean(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        op.attributes["keepdims"].as_int().unwrap() != 0,
                        &op.outputs[0],
                    );
                }
                "ReduceSum" => {
                    op_reduce_sum(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        op.attributes["keepdims"].as_int().unwrap() != 0,
                        &op.outputs[0],
                    );
                }
                "ReduceMax" => {
                    op_reduce_max(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        op.attributes["keepdims"].as_int().unwrap() != 0,
                        &op.outputs[0],
                    );
                }
                "Sqrt" => {
                    op_sqrt(&mut self.tensors, &op.inputs[0], &op.outputs[0]);
                }
                "Reciprocal" => {
                    op_reciprocal(&mut self.tensors, &op.inputs[0], &op.outputs[0]);
                }
                "MatMul" => {
                    op_matmul(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Equal" => {
                    op_equal(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Less" => {
                    op_less(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Greater" => {
                    op_greater(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                "Cast" => {
                    // we treat cast as noop because all tensor types are defined by us (T)
                    op_identity(&mut self.tensors, &op.inputs[0], &op.outputs[0]);
                }
                "Gather" => {
                    op_gather(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        if let Some(value) = op.attributes.get("axis") {
                            value.as_int().unwrap()
                        } else {
                            0
                        },
                        &op.outputs[0],
                    );
                }
                "Split" => {
                    op_split(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        if let Some(value) = op.attributes.get("axis") {
                            value.as_int().unwrap()
                        } else {
                            0
                        },
                        &op.outputs.iter().map(|v| v.as_str()).collect::<Vec<&str>>(),
                    );
                }
                "Reshape" => {
                    op_reshape(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        if let Some(value) = op.attributes.get("allow_zero") {
                            value.as_int().unwrap() == 1
                        } else {
                            false
                        },
                        &op.outputs[0],
                    );
                }
                "Transpose" => {
                    op_transpose(
                        &mut self.tensors,
                        &op.inputs[0],
                        if let Some(value) = op.attributes.get("perm") {
                            Some(value.as_ints().unwrap())
                        } else {
                            None
                        },
                        &op.outputs[0],
                    );
                }
                "Unsqueeze" => {
                    op_unsqueeze(
                        &mut self.tensors,
                        &op.inputs[0],
                        &op.inputs[1],
                        &op.outputs[0],
                    );
                }
                _ => {
                    panic!("unkown operation: {:?}", op);
                }
            }
            perf_breakdown.insert(
                op.op_type.clone(),
                perf_breakdown
                    .get(&op.op_type)
                    .unwrap_or(&Duration::default())
                    .to_owned()
                    + op_timer.elapsed(),
            );
            eprint!(".");
            io::stderr().flush().unwrap();
        }
        eprintln!();
        let mut ret: HashMap<String, ArrayD<T>> = HashMap::new();
        for tensor_name in self
            .model
            .get_output_tensors()
            .iter()
            .map(|t| t.name().to_string())
        {
            ret.insert(
                tensor_name.clone(),
                self.tensors
                    .get(tensor_name.as_str())
                    .as_ref()
                    .unwrap()
                    .data
                    .to_owned()
                    .unwrap(),
            );
        }
        retain.iter().for_each(|tensor_name| {
            ret.insert(
                tensor_name.clone(),
                self.tensors
                    .get(tensor_name.as_str())
                    .as_ref()
                    .unwrap()
                    .data
                    .to_owned()
                    .unwrap(),
            );
        });
        (ret, perf_breakdown)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AFloat, TFloat};
    use crate::{F128, OnnxRunner};
    use ndarray::ArrayD;
    use num_traits::ToPrimitive;
    use tract_onnx::prelude::*;

    const EPSILON: f64 = 1e-5;

    fn elementwise_equal_up_to_epsilon(a: &ArrayD<f64>, b: &ArrayD<f64>) {
        assert_eq!(a.shape(), b.shape());
        eprintln!("mean diff: {:?}", (a - b).sum() / (a - b).len() as f64);
        let mut max_abs_diff = 0.;
        for (a, b) in a.iter().zip(b.iter()) {
            let diff = (a - b).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }
        eprintln!("max diff: {:?}", max_abs_diff);
        assert!((a - b).iter().all(|&v| v.abs() < EPSILON));
    }

    #[test]
    fn test_mul() {
        let path = "data/mul/mul.onnx";
        // because tract_onnx use slightly different tensor name, we need to manually define the
        // order of outputs if there are multiple ones
        let output_order = ["add_1"];
        let mut runner = OnnxRunner::<TFloat>::read(path).unwrap();
        let input = runner.rand_input();

        let model = onnx()
            .model_for_path(path)
            .unwrap()
            .into_runnable()
            .unwrap();
        let mut input_tract = tvec![];
        for outlet in model.model().input_outlets().unwrap() {
            let name = model.model().node(outlet.node).name.clone();
            let data = input
                .get(&name)
                .expect("missing input")
                .data
                .clone()
                .unwrap()
                .map(|v| v.to_f32().unwrap());
            let data_tract = tract_ndarray::ArrayD::from_shape_vec(
                data.shape(),
                data.clone().into_iter().collect(),
            )
            .unwrap();
            input_tract.push(data_tract.into_tvalue());
        }

        let outputs = runner.run(input);
        let outputs_tract = model.run(input_tract).unwrap();
        for (output_tract, output_name) in outputs_tract.iter().zip(output_order.iter()) {
            let data = outputs[*output_name].clone().mapv(|v| v.to_f64().unwrap());
            let data_tract = output_tract
                .clone()
                .into_tensor()
                .into_array::<f32>()
                .unwrap()
                .mapv(|v| v as f64);
            let data_tract = ArrayD::from_shape_vec(
                data_tract.shape(),
                data_tract.clone().into_iter().collect(),
            )
            .unwrap();
            elementwise_equal_up_to_epsilon(&data, &data_tract);
        }
    }

    #[test]
    fn test_a_vs_tfloat() {
        let path = "data/mul/mul.onnx";
        let mut runner_a = OnnxRunner::<AFloat>::read(path).unwrap();
        let mut runner_t = OnnxRunner::<TFloat>::read(path).unwrap();
        let input_a = runner_a.all_ones_input();
        let input_t = runner_t.all_ones_input();
        let outputs_a = runner_a.run(input_a);
        let outputs_t = runner_t.run(input_t);
        for (output_name, output_a) in outputs_a.iter() {
            let output_t = outputs_t.get(output_name).unwrap();
            let data_a = output_a.mapv(|v| v.to_f64().unwrap());
            let data_t = output_t.mapv(|v| v.to_f64().unwrap());
            elementwise_equal_up_to_epsilon(&data_a, &data_t);
        }
    }

    #[test]
    fn test_primary() {
        let path = "data/layer_norm/primary_model.onnx";
        let output_order = [
            "LayerNorm_32x768.identity",
            "LayerNorm_32x768.reduce_mean",
            "LayerNorm_32x768.squared",
            "LayerNorm_32x768.std",
            "LayerNorm_32x768.inv_std",
            "LayerNorm_32x768.normalized",
            "Y",
        ];
        let mut runner = OnnxRunner::<AFloat>::read(path).unwrap();
        let input = runner.rand_input();

        let model = onnx()
            .model_for_path(path)
            .unwrap()
            .into_runnable()
            .unwrap();
        let mut input_tract = tvec![];
        for outlet in model.model().input_outlets().unwrap() {
            let name = model.model().node(outlet.node).name.clone();
            let data = input
                .get(&name)
                .expect("missing input")
                .data
                .clone()
                .unwrap()
                .map(|v| v.to_f64().unwrap());
            let data_tract = tract_ndarray::ArrayD::from_shape_vec(
                data.shape(),
                data.clone().into_iter().collect(),
            )
            .unwrap();
            input_tract.push(data_tract.into_tvalue());
        }

        let outputs = runner.run(input);
        let outputs_tract = model.run(input_tract).unwrap();
        for (output_tract, output_name) in outputs_tract.iter().zip(output_order.iter()) {
            let data = outputs[*output_name].clone().mapv(|v| v.to_f64().unwrap());
            let data_tract = output_tract
                .clone()
                .into_tensor()
                .into_array::<f64>()
                .unwrap();
            let data_tract = ArrayD::from_shape_vec(
                data_tract.shape(),
                data_tract.clone().into_iter().collect(),
            )
            .unwrap();
            elementwise_equal_up_to_epsilon(&data, &data_tract);
        }
    }

    #[test]
    fn test_gpt_primary() {
        let path = "../spain/data/gpt/primary_model.onnx";
        let mut runner = OnnxRunner::<F128>::read(path).unwrap();
        let output_order = runner.get_output_order();
        let input = runner.rand_input();

        let model = onnx()
            .model_for_path(path)
            .unwrap()
            .into_runnable()
            .unwrap();
        let mut input_tract = tvec![];
        for outlet in model.model().input_outlets().unwrap() {
            let name = model.model().node(outlet.node).name.clone();
            let data = input
                .get(&name)
                .expect("missing input")
                .data
                .clone()
                .unwrap()
                .map(|v| v.to_i64().unwrap());
            let data_tract = tract_ndarray::ArrayD::from_shape_vec(
                data.shape(),
                data.clone().into_iter().collect(),
            )
            .unwrap();
            input_tract.push(data_tract.into_tvalue());
        }

        let outputs = runner.run(input);
        let outputs_tract = model.run(input_tract).unwrap();
        for (output_tract, output_name) in outputs_tract.iter().zip(output_order.iter()) {
            let data = outputs[output_name].clone().mapv(|v| v.to_f64().unwrap());
            let data_tract = output_tract
                .clone()
                .into_tensor()
                .into_array::<f64>()
                .unwrap();
            let data_tract = ArrayD::from_shape_vec(
                data_tract.shape(),
                data_tract.clone().into_iter().collect(),
            )
            .unwrap();
            elementwise_equal_up_to_epsilon(&data, &data_tract);
        }
    }

    #[test]
    fn test_gpt_secondary() {
        let path = "../spain/data/gpt/secondary_model.onnx";
        let mut runner = OnnxRunner::<F128>::read(path).unwrap();
        let output_order = runner.get_output_order();
        let input = runner.rand_input();

        let model = onnx()
            .model_for_path(path)
            .unwrap()
            .into_runnable()
            .unwrap();
        let mut input_tract = tvec![];
        for outlet in model.model().input_outlets().unwrap() {
            let name = model.model().node(outlet.node).name.clone();
            let data = input
                .get(&name)
                .expect("missing input")
                .data
                .clone()
                .unwrap()
                .map(|v| v.to_f64().unwrap());
            let data_tract = tract_ndarray::ArrayD::from_shape_vec(
                data.shape(),
                data.clone().into_iter().collect(),
            )
            .unwrap();
            input_tract.push(data_tract.into_tvalue());
        }

        let outputs = runner.run(input);
        let outputs_tract = model.run(input_tract).unwrap();
        for (output_tract, output_name) in outputs_tract.iter().zip(output_order.iter()) {
            let data = outputs[output_name].clone().mapv(|v| v.to_f64().unwrap());
            let data_tract = output_tract
                .clone()
                .into_tensor()
                .into_array::<f64>()
                .unwrap();
            let data_tract = ArrayD::from_shape_vec(
                data_tract.shape(),
                data_tract.clone().into_iter().collect(),
            )
            .unwrap();
            elementwise_equal_up_to_epsilon(&data, &data_tract);
        }
    }
}
