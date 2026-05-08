use ndarray::{Array, ArrayD, ArrayViewD, Axis, Slice, concatenate, stack};
use onnx_extractor::{DataType, OnnxTensor};
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

#[allow(unused)]
use crate::{AFloat, HighPrecision, Tensor};

fn manual_co_broadcast<'a, T>(
    x: &'a ArrayD<T>,
    y: &'a ArrayD<T>,
    except: usize,
) -> (ArrayViewD<'a, T>, ArrayViewD<'a, T>) {
    fn fill_with_one(shape: &[usize], len: usize) -> Vec<usize> {
        let mut ret = vec![1; len - shape.len()];
        ret.extend_from_slice(shape);
        ret
    }
    let mut x_shape = x.shape().to_vec();
    let mut y_shape = y.shape().to_vec();
    if x_shape.len() < y_shape.len() {
        x_shape = fill_with_one(&x_shape, y_shape.len());
    } else if x_shape.len() > y_shape.len() {
        y_shape = fill_with_one(&y_shape, x_shape.len());
    }
    x_shape.truncate(x_shape.len() - except);
    y_shape.truncate(y_shape.len() - except);
    let broadcast_shape =
        x_shape
            .iter()
            .zip(y_shape.iter())
            .fold(Vec::new(), |mut acc, (&xi, &yi)| {
                assert!(
                    xi == yi || xi == 1 || yi == 1,
                    "invalid broadcast shape: {:?}, {:?}",
                    x_shape,
                    y_shape
                );
                acc.push(xi.max(yi));
                acc
            });
    x_shape = broadcast_shape.clone();
    y_shape = broadcast_shape.clone();
    x_shape.extend_from_slice(&x.shape()[(x.shape().len() - except)..]);
    y_shape.extend_from_slice(&y.shape()[(y.shape().len() - except)..]);
    (x.broadcast(x_shape).unwrap(), y.broadcast(y_shape).unwrap())
}

pub fn op_add<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    out_: &str,
) where
    ArrayD<T>: Add<Output = ArrayD<T>>,
{
    let in1 = &tensors[in1_];
    let in2 = &tensors[in2_];
    let data = in1.data.as_ref().unwrap() + in2.data.as_ref().unwrap();
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_sub<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    out_: &str,
) where
    ArrayD<T>: Sub<Output = ArrayD<T>>,
{
    let in1 = &tensors[in1_];
    let in2 = &tensors[in2_];
    let data = in1.data.as_ref().unwrap() - in2.data.as_ref().unwrap();
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_mul<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    out_: &str,
) where
    ArrayD<T>: Mul<Output = ArrayD<T>>,
{
    let in1 = &tensors[in1_];
    let in2 = &tensors[in2_];
    let data = in1.data.as_ref().unwrap() * in2.data.as_ref().unwrap();
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_div<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    out_: &str,
) where
    ArrayD<T>: Div<Output = ArrayD<T>>,
{
    let in1 = &tensors[in1_];
    let in2 = &tensors[in2_];
    let data = in1.data.as_ref().unwrap() / in2.data.as_ref().unwrap();
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_identity<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    out_: &str,
) {
    let in1 = &tensors[in1_];
    let data = in1.data.clone().unwrap();
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

// constant will be stored in type T no matter what onnx file says
pub fn op_constant<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1: &OnnxTensor,
    out_: &str,
) {
    let shape = in1.shape().iter().map(|v| *v as usize).collect::<Vec<_>>();
    let data = match in1.data_type() {
        DataType::Int64 => in1
            .copy_data_as::<i64>()
            .unwrap()
            .iter()
            .map(|v| T::from_i64(*v).unwrap())
            .collect(),
        DataType::Double => in1
            .copy_data_as::<f64>()
            .unwrap()
            .iter()
            .map(|v| T::from_f64(*v).unwrap())
            .collect(),
        DataType::Float => in1
            .copy_data_as::<f32>()
            .unwrap()
            .iter()
            .map(|v| T::from_f32(*v).unwrap())
            .collect(),
        _ => panic!("have not implemented op_constant for {:?}", in1.data_type()),
    };
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(Array::from_shape_vec(shape.clone(), data).unwrap()),
            dim: shape,
        },
    );
}

pub fn op_reduce_mean<T>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    axes_: &str,
    keepdims: bool,
    out_: &str,
) where
    T: HighPrecision + Div<Output = T>,
{
    let in1 = &tensors[in1_];
    let axes = &tensors[axes_];
    assert_eq!(
        axes.dim,
        vec![1],
        "only implemtned reduce over one explicit axis"
    );
    let x = axes.data.clone().unwrap()[[0]].to_i64().unwrap();
    let real_axis = if x < 0 {
        (in1.dim.len() as i64 + x) as usize
    } else {
        axes.data.clone().unwrap()[[0]].to_usize().unwrap()
    };
    let mut data = in1
        .data
        .as_ref()
        .unwrap()
        .view()
        .mean_axis(Axis(real_axis))
        .unwrap();
    if keepdims {
        data.insert_axis_inplace(Axis(real_axis));
    }
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_reduce_sum<T>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    axes_: &str,
    keepdims: bool,
    out_: &str,
) where
    T: HighPrecision + Div<Output = T>,
{
    let in1 = &tensors[in1_];
    let axes = &tensors[axes_];
    assert_eq!(
        axes.dim,
        vec![1],
        "only implemtned reduce over one explicit axis"
    );
    let x = axes.data.clone().unwrap()[[0]].to_i64().unwrap();
    let real_axis = if x < 0 {
        (in1.dim.len() as i64 + x) as usize
    } else {
        axes.data.clone().unwrap()[[0]].to_usize().unwrap()
    };
    let mut data = in1.data.as_ref().unwrap().view().sum_axis(Axis(real_axis));
    if keepdims {
        data.insert_axis_inplace(Axis(real_axis));
    }
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_reduce_max<T>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    axes_: &str,
    keepdims: bool,
    out_: &str,
) where
    T: HighPrecision + Div<Output = T>,
{
    let in1 = &tensors[in1_];
    let axes = &tensors[axes_];
    assert_eq!(
        axes.dim,
        vec![1],
        "only implemtned reduce over one explicit axis"
    );
    let x = axes.data.clone().unwrap()[[0]].to_i64().unwrap();
    let real_axis = if x < 0 {
        (in1.dim.len() as i64 + x) as usize
    } else {
        axes.data.clone().unwrap()[[0]].to_usize().unwrap()
    };
    let mut data = in1
        .data
        .as_ref()
        .unwrap()
        .view()
        .map_axis(Axis(real_axis), |view| {
            let ret = view.iter().cloned().reduce(|acc, e| acc.max(&e)).unwrap();
            ret.clone()
        });
    // NOTE potentially inefficient here, everything is copied when comparing max
    if keepdims {
        data.insert_axis_inplace(Axis(real_axis));
    }
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_sqrt<T: HighPrecision>(tensors: &mut HashMap<String, Tensor<T>>, in1_: &str, out_: &str) {
    let in1 = &tensors[in1_];
    let mut data = in1.data.clone().unwrap();
    data.mapv_inplace(|v| v.sqrt());
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

// NOTE potentially slow, create AFloat for each array element
pub fn op_reciprocal<T>(tensors: &mut HashMap<String, Tensor<T>>, in1_: &str, out_: &str)
where
    T: HighPrecision + Div<Output = T>,
{
    let in1 = &tensors[in1_];
    let mut data = in1.data.clone().unwrap();
    data.mapv_inplace(|v| v.recip());
    let shape = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: shape,
        },
    );
}

pub fn op_matmul<T>(tensors: &mut HashMap<String, Tensor<T>>, in1_: &str, in2_: &str, out_: &str)
where
    T: HighPrecision + Mul<Output = T> + Add<Output = T>,
{
    // NOTE: the dot product provided by ndarray does not work because it only works with rust
    // primitives. Attempts to use AFloat/TFloat will result in atrocious compiler errors.
    // In fact, the dot product feature itself for arrayd is added no long ago... not stable
    // Suspect the problem is custom types dont implement copy, etc...
    // Here, using a very unoptimized n^3 matmul by hand
    let in1 = tensors[in1_].data.as_ref().unwrap();
    let in2 = tensors[in2_].data.as_ref().unwrap();
    assert!(in1.ndim() >= 2 && in2.ndim() >= 2);
    assert_eq!(
        in1.shape().last().unwrap().to_owned(),
        in2.shape()[in2.ndim() - 2],
        "matmul require the last and second to last dims of in1 and in2 to match"
    );
    // implementing co-broadcasting by hand, not supported by ndarray
    // SEE: https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html
    let (in1, in2) = manual_co_broadcast(in1, in2, 2);

    fn do_matmul<'a, T: HighPrecision>(
        in1: ArrayViewD<'a, T>,
        in2: ArrayViewD<'a, T>,
    ) -> ArrayD<T> {
        assert!(in1.ndim() == in2.ndim());
        if in1.ndim() == 2 {
            let shape = [in1.shape()[0], in2.shape()[1]];
            let len = shape.iter().product();
            let common_len = in1.shape()[1];
            let mut data = Vec::<T>::with_capacity(len);
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let mut acc = T::zero();
                    for k in 0..common_len {
                        acc = acc + in1[[i, k]].clone() * in2[[k, j]].clone();
                    }
                    data.push(acc);
                }
            }
            ArrayD::from_shape_vec(shape.to_vec(), data).unwrap()
        } else {
            assert_eq!(in1.shape()[0], in2.shape()[0]);
            let mut ret = Vec::with_capacity(in1.shape()[0]);
            for (in1_layer, in2_layer) in in1.axis_iter(Axis(0)).zip(in2.axis_iter(Axis(0))) {
                ret.push(do_matmul(in1_layer, in2_layer));
            }
            stack(Axis(0), &ret.iter().map(|v| v.view()).collect::<Vec<_>>()).unwrap()
        }
    }

    let data = do_matmul(in1, in2);
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            dim: data.shape().to_vec(),
            data: Some(data),
        },
    );
}

// NOTE ndarray support limited broadcasting
pub fn op_equal<T>(tensors: &mut HashMap<String, Tensor<T>>, in1_: &str, in2_: &str, out_: &str)
where
    T: HighPrecision + Sub<Output = T>,
{
    let in1 = tensors[in1_].data.as_ref().unwrap();
    let in2 = tensors[in2_].data.as_ref().unwrap();
    // note here we treat bool as float
    let data = (in1 - in2).mapv_into(|v| T::from_f64((v == T::zero()) as i64 as f64).unwrap());
    let dim = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim,
        },
    );
}

pub fn op_greater<T>(tensors: &mut HashMap<String, Tensor<T>>, in1_: &str, in2_: &str, out_: &str)
where
    T: HighPrecision + Sub<Output = T>,
{
    let in1 = tensors[in1_].data.as_ref().unwrap();
    let in2 = tensors[in2_].data.as_ref().unwrap();
    // note here we treat bool as float
    let data = (in1 - in2).mapv_into(|v| T::from_f64((v > T::zero()) as i64 as f64).unwrap());
    let dim = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim,
        },
    );
}

pub fn op_less<T>(tensors: &mut HashMap<String, Tensor<T>>, in1_: &str, in2_: &str, out_: &str)
where
    T: HighPrecision + Sub<Output = T>,
{
    let in1 = tensors[in1_].data.as_ref().unwrap();
    let in2 = tensors[in2_].data.as_ref().unwrap();
    // note here we treat bool as float
    let data = (in1 - in2).mapv_into(|v| T::from_f64((v < T::zero()) as i64 as f64).unwrap());
    let dim = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim,
        },
    );
}

pub fn op_gather<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    axis: i64,
    out_: &str,
) {
    let in1 = &tensors[in1_];
    let in2 = &tensors[in2_];

    let axis = if axis < 0 {
        (in1.dim.len() as i64 + axis) as usize
    } else {
        axis as usize
    };
    let indices = in2.data.as_ref().unwrap().map(|v| {
        let v = v.to_i64().unwrap();
        if v < 0 {
            (in1.dim[axis] as i64 + v) as usize
        } else {
            v as usize
        }
    });

    fn do_gather<T: HighPrecision>(
        indices: ArrayD<usize>,
        data: &ArrayD<T>,
        axis: usize,
    ) -> ArrayD<T> {
        if indices.ndim() == 1 {
            let indices = indices.as_slice().unwrap();
            data.select(Axis(axis), indices)
        } else {
            let mut ret = Vec::new();
            for layer in indices.axis_iter(Axis(0)) {
                ret.push(do_gather(layer.into_owned(), data, axis).insert_axis(Axis(axis)))
            }
            concatenate(
                Axis(axis),
                &ret.iter().map(|v| v.view()).collect::<Vec<_>>(),
            )
            .unwrap()
        }
    }

    let data = in1.data.as_ref().unwrap();
    let data = do_gather(indices, data, axis);
    let dim = data.shape().to_vec();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim,
        },
    );
}

pub fn op_split<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    axis: i64,
    outputs_: &[&str],
) {
    let input = tensors[in1_].data.as_ref().unwrap().clone();
    let split: Vec<_> = tensors[in2_]
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(|v| v.to_i64().unwrap() as usize)
        .collect();

    let r = input.ndim();
    let real_axis = if axis < 0 {
        (r as i64 + axis) as usize
    } else {
        axis as usize
    };
    let mut start_idx = 0;
    for (i, size) in split.iter().enumerate() {
        let end_idx = start_idx + size;
        let slice = input.slice_axis(Axis(real_axis), (start_idx..end_idx).into());
        let out_shape = slice.shape().to_vec();
        tensors.insert(
            outputs_[i].to_string(),
            Tensor {
                name: outputs_[i].to_string(),
                data: Some(slice.to_owned()),
                dim: out_shape,
            },
        );
        start_idx = end_idx;
    }
}

pub fn op_reshape<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    in2_: &str,
    allow_zero: bool,
    out_: &str,
) {
    if allow_zero {
        panic!("did not implement allow zero");
    }
    let in1 = &tensors[in1_];
    let in2 = &tensors[in2_];

    let mut new_shape: Vec<i64> = in2
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(|v| v.to_i64().unwrap())
        .collect();

    let total_elements: usize = in1.dim.iter().product();
    if let Some(neg_idx) = new_shape.iter().position(|&x| x == -1) {
        let known_product: usize = new_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != neg_idx)
            .map(|(_, &x)| x as usize)
            .product();
        new_shape[neg_idx] = (total_elements / known_product) as i64;
    }

    let new_shape_usize: Vec<usize> = new_shape.iter().map(|&x| x as usize).collect();
    let data = in1
        .data
        .to_owned()
        .unwrap()
        .into_shape_clone(new_shape_usize.clone())
        .unwrap();

    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            data: Some(data),
            dim: new_shape_usize,
        },
    );
}

pub fn op_transpose<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    perm_: Option<&[i64]>,
    out_: &str,
) {
    let in1 = &tensors[in1_];

    let perm = match perm_ {
        Some(value) => value
            .iter()
            .map(|&v| {
                if v < 0 {
                    (in1.dim.len() as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .collect::<Vec<_>>(),
        None => (0..in1.dim.len()).rev().collect(),
    };

    let data = in1.data.to_owned().unwrap().permuted_axes(perm).to_owned();
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            dim: data.shape().to_vec(),
            data: Some(data),
        },
    );
}

pub fn op_unsqueeze<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    data_: &str,
    axes_: &str,
    out_: &str,
) {
    let data = tensors[data_].data.to_owned().unwrap();
    let axes = tensors[axes_]
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(|v| v.to_i64().unwrap())
        .map(|v| {
            if v < 0 {
                (v + data.ndim() as i64) as usize
            } else {
                v as usize
            }
        })
        .collect::<Vec<_>>();
    let mut shape_iter = data.shape().iter();
    let mut new_shape = Vec::new();
    for i in 0..data.ndim() + axes.len() {
        if axes.contains(&i) {
            new_shape.push(1_usize);
        } else {
            new_shape.push(shape_iter.next().unwrap().to_owned());
        }
    }
    assert!(
        shape_iter.next().is_none(),
        "did not consume all shape iteration when calculating new shape"
    );

    let data = data.into_shape_clone(new_shape).unwrap();

    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            dim: data.shape().to_vec(),
            data: Some(data),
        },
    );
}

pub fn op_slice<T: HighPrecision>(
    tensors: &mut HashMap<String, Tensor<T>>,
    in1_: &str,
    starts_: &str,
    ends_: &str,
    axes_: Option<&str>,
    steps_: Option<&str>,
    out_: &str,
) {
    let mut data = tensors[in1_].data.to_owned().unwrap();
    let rank = data.ndim();
    assert!(tensors[starts_].dim.len() <= rank);
    assert!(tensors[ends_].dim.len() <= rank);
    let axes: &[usize] = if let Some(x) = axes_ {
        &tensors[x]
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| {
                let x = v.to_i64().unwrap();
                if x < 0 {
                    (x + rank as i64) as usize
                } else {
                    x as usize
                }
            })
            .collect::<Vec<_>>()
    } else {
        &(0..rank).collect::<Vec<_>>()
    };
    let starts: &[usize] = &tensors[starts_]
        .data
        .as_ref()
        .unwrap()
        .iter()
        .zip(axes.iter())
        .map(|(v, &i)| {
            let mut x = v.to_i64().unwrap();
            if x < 0 {
                x += data.shape()[i] as i64;
            }
            x.clamp(0, data.shape()[i] as i64) as usize
        })
        .collect::<Vec<_>>();
    let ends: &[usize] = &tensors[ends_]
        .data
        .as_ref()
        .unwrap()
        .iter()
        .zip(axes.iter())
        .map(|(v, &i)| {
            let mut x = v.to_i64().unwrap();
            if x < 0 {
                x += data.shape()[i] as i64;
            }
            x.clamp(0, data.shape()[i] as i64) as usize
        })
        .collect::<Vec<_>>();
    let steps: &[isize] = if let Some(x) = steps_ {
        &tensors[x].data.as_ref().unwrap().iter().map(|v| {
            let x = v.to_i64().unwrap();
            assert!(x > 0, "negative steps for op_slice not implemented, because ndarray behaves differently from numpy with negative steps");
            x as isize
        }).collect::<Vec<_>>()
    } else {
        &vec![1; rank]
    };
    assert_eq!(axes.len(), rank);
    assert_eq!(starts.len(), ends.len());
    assert!(steps.len() == rank || steps.len() == starts.len()); // sanity
    for (((&ax, &start), &end), &step) in axes
        .iter()
        .zip(starts.iter())
        .zip(ends.iter())
        .zip(steps.iter())
    {
        data = data.slice_axis_move(Axis(ax), Slice::from(start..end).step_by(step))
    }
    tensors.insert(
        out_.to_string(),
        Tensor {
            name: out_.to_string(),
            dim: data.shape().to_vec(),
            data: Some(data),
        },
    );
}
