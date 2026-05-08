use model::{AFloat, HighPrecision};
use parse::{
    generalized::{HighPrecisionInt, InjectionInfo},
    mat::{Matrix, MatrixData},
};
use serde::Deserialize;
use std::{
    ops::Range,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct R1CSMatrices<T: Clone + Default + PartialEq> {
    pub a: Matrix<T>,
    pub b: Matrix<T>,
    pub c: Matrix<T>,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct Metadata {
    pub num_witness_values: usize,
    pub num_public_values: usize,
    pub num_random_values: usize,
    pub num_secondary_witness_values: usize,
    pub num_secondary_constraint_variables: usize,
    pub primary_output_labels: Vec<String>,
    pub secondary_output_labels: Vec<String>,
}

impl Metadata {
    pub fn secondary_constraint_shift(&self) -> usize {
        self.num_public_values
            + self.num_random_values
            + self.num_witness_values
            + self.num_secondary_witness_values
    }

    pub fn r_vec_shift(&self) -> usize {
        self.num_public_values
    }

    pub fn secondary_witness_shift(&self) -> usize {
        self.num_public_values + self.num_random_values + self.num_witness_values
    }

    // Returns ranges of each subvector in the witness
    // Assuming z = (public, random, witness, secondary_witness, secondary_constraints)
    pub fn get_ranges(&self) -> Vec<Range<usize>> {
        let pub_end = self.num_public_values + self.num_random_values;
        let wit_end = pub_end + self.num_witness_values;
        let secondary_wit_end = wit_end + self.num_secondary_witness_values;
        let secondary_constraints_end = secondary_wit_end + self.num_secondary_constraint_variables;
        let mut ranges = vec![0..pub_end, pub_end..wit_end];
        if self.num_secondary_witness_values > 0 {
            ranges.push(wit_end..secondary_wit_end);
        }
        if self.num_secondary_constraint_variables > 0 {
            ranges.push(secondary_wit_end..secondary_constraints_end);
        }
        ranges
    }

    pub fn get_witness_size(&self) -> usize {
        let pub_end = self.num_public_values + self.num_random_values;
        let wit_end = pub_end + self.num_witness_values;
        let secondary_wit_end = wit_end + self.num_secondary_witness_values;

        secondary_wit_end + self.num_secondary_constraint_variables
    }
}

pub const DEFAULT_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../circuit/export");

pub fn data_file(data_dir: &Path, filename: &str) -> PathBuf {
    data_dir.join(filename)
}

pub fn import_metadata(data_dir: &Path, model: &str) -> Metadata {
    let meta_file = data_file(data_dir, format!("{model}/meta.json").as_str());
    let contents = std::fs::read_to_string(meta_file).expect("Failed to read meta.json");
    serde_json::from_str::<Metadata>(&contents).expect("Failed to parse meta.json")
}

// given scale_factor, load in a, b, c as matrices i64 form
pub fn import_r1cs0(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    verbose: bool,
) -> (Matrix<i64>, Matrix<i64>, Matrix<i64>) {
    import_r1cs0_int_deprecated::<i64>(data_dir, model, scale_factor, verbose)
}

pub fn import_r1cs0_i128(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    verbose: bool,
) -> (Matrix<i128>, Matrix<i128>, Matrix<i128>) {
    import_r1cs0_int_deprecated::<i128>(data_dir, model, scale_factor, verbose)
}

pub trait FromF64Matrix: Clone + Default + PartialEq + HighPrecisionInt {
    fn f64_to_int(
        matrix: &Matrix<f64>,
        scale_factor: AFloat,
        shift: Option<usize>,
    ) -> Matrix<Self> {
        Matrix::from_f64(matrix, scale_factor, shift)
    }
    fn f64_index_to_int(matrix: &Matrix<f64>, z: &[Self], shift: usize) -> Matrix<Self> {
        Matrix::from_f64_index(matrix, z, shift)
    }
}
impl<T> FromF64Matrix for T where T: Clone + Default + PartialEq + HighPrecisionInt {}

pub trait FromHPMatrix: Clone + Default + PartialEq {
    type T: HighPrecision;
    fn hp_to_int(
        matrix: &Matrix<Self::T>,
        scale_factor: Self::T,
        shift: Option<usize>,
    ) -> Matrix<Self>;
}

pub fn import_sub_r1cs<T: FromF64Matrix>(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    verbose: bool,
    r1cs_index: usize,
    randomness: Option<&Vec<T>>,
    metadata: Option<&Metadata>,
) -> (Matrix<T>, Matrix<T>, Matrix<T>, Option<InjectionInfo>) {
    assert!(r1cs_index == 1 || r1cs_index == 0);
    if verbose {
        eprintln!("Importing R1CS{} for model: {}", r1cs_index, model);
    }
    let a = data_file(data_dir, format!("{model}/A{r1cs_index}.bin").as_str());
    let b = data_file(data_dir, format!("{model}/B{r1cs_index}.bin").as_str());
    let c = data_file(data_dir, format!("{model}/C{r1cs_index}.bin").as_str());
    let (_, a) = Matrix::from_file(&a).expect("Failed to read matrix A");
    let (_, b) = Matrix::from_file(&b).expect("Failed to read matrix B");
    let (_, c) = Matrix::from_file(&c).expect("Failed to read matrix C");
    if verbose {
        eprintln!("A: {} rows, {} columns", a.height(), a.width());
        eprintln!("B: {} rows, {} columns", b.height(), b.width());
        eprintln!("C: {} rows, {} columns", c.height(), c.width());
    }
    let mut inject_info = None;
    let (a, b, c) = (
        if r1cs_index == 0 {
            T::f64_to_int(&a, scale_factor.clone(), None)
        } else {
            T::f64_to_int(
                &a,
                scale_factor.clone(),
                Some(metadata.unwrap().secondary_witness_shift()),
            )
        },
        T::f64_to_int(&b, scale_factor.clone(), None),
        if r1cs_index == 0 {
            T::f64_to_int(&c, scale_factor, None)
        } else {
            if let Some(randomness) = randomness {
                T::f64_index_to_int(&c, randomness, 0)
            } else {
                let (ret, info) = Matrix::from_f64_inject(&c, 0);
                inject_info = Some(info);
                ret
            }
        },
    );
    (a, b, c, inject_info)
}

pub fn import_full_r1cs<T: FromF64Matrix>(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    metadata: &Metadata,
    randomness: Option<&Vec<T>>,
    verbose: bool,
) -> (R1CSMatrices<T>, Option<InjectionInfo>) {
    if verbose {
        eprintln!("Importing full R1CS for model: {model}");
    }
    let r1cs0 = import_sub_r1cs::<T>(
        data_dir,
        model,
        scale_factor.clone(),
        verbose,
        0,
        None,
        None,
    );
    if metadata.num_secondary_witness_values == 0
        && metadata.num_secondary_constraint_variables == 0
    {
        (
            R1CSMatrices {
                a: Matrix::stack_sparse_matrices(vec![&r1cs0.0]),
                b: Matrix::stack_sparse_matrices(vec![&r1cs0.1]),
                c: Matrix::stack_sparse_matrices(vec![&r1cs0.2]),
            },
            None,
        )
    } else if metadata.num_secondary_constraint_variables == 0
        && metadata.num_secondary_witness_values > 0
    {
        let r1cs1 = import_sub_r1cs::<T>(
            data_dir,
            model,
            scale_factor,
            verbose,
            1,
            randomness,
            Some(metadata),
        );
        (
            R1CSMatrices {
                a: Matrix::stack_sparse_matrices(vec![&r1cs0.0, &r1cs1.0]),
                b: Matrix::stack_sparse_matrices(vec![&r1cs0.1, &r1cs1.1]),
                c: Matrix::stack_sparse_matrices(vec![&r1cs0.2, &r1cs1.2]),
            },
            if let Some(inject_info) = r1cs1.3 {
                let injection_offset = if let MatrixData::COO(data) = r1cs0.2.data() {
                    data.len()
                } else {
                    panic!()
                };
                Some(
                    inject_info
                        .iter()
                        .map(|&(i, v)| (i + injection_offset, v))
                        .collect::<InjectionInfo>(),
                )
            } else {
                None
            },
        )
    } else {
        panic!("invalid path (deprecated model?)");
    }
}

//
// ========== Below are deprecated, used for non-interactive simulate only =======
//

pub fn import_raw_r1cs_deprecated(
    data_dir: &Path,
    model: &str,
    verbose: bool,
) -> R1CSMatrices<f64> {
    if verbose {
        eprintln!("Importing R1CS for model: {}", model);
    }
    // open data files
    let a = data_file(data_dir, format!("{model}/A0.bin").as_str());
    let b = data_file(data_dir, format!("{model}/B0.bin").as_str());
    let c = data_file(data_dir, format!("{model}/C0.bin").as_str());
    // load to matrices
    let (_, a) = Matrix::<f64>::from_file(&a).expect("Failed to read matrix A");
    let (_, b) = Matrix::<f64>::from_file(&b).expect("Failed to read matrix B");
    let (_, c) = Matrix::<f64>::from_file(&c).expect("Failed to read matrix C");
    // print dimensions of a
    if verbose {
        eprintln!("A: {} rows, {} columns", a.height(), a.width());
        eprintln!("B: {} rows, {} columns", b.height(), b.width());
        eprintln!("C: {} rows, {} columns", c.height(), c.width());
    }
    R1CSMatrices { a, b, c }
}

pub fn import_full_r1cs_int_deprecated<T: FromF64Matrix>(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    metadata: &Metadata,
    z: &Matrix<T>,
    verbose: bool,
) -> R1CSMatrices<T> {
    if verbose {
        eprintln!("Importing full R1CS for model: {model}");
    }
    let (a0, b0, c0) =
        import_r1cs0_int_deprecated::<T>(data_dir, model, scale_factor.clone(), verbose);
    if metadata.num_secondary_witness_values == 0
        && metadata.num_secondary_constraint_variables == 0
    {
        R1CSMatrices {
            a: Matrix::stack_sparse_matrices(vec![&a0]),
            b: Matrix::stack_sparse_matrices(vec![&b0]),
            c: Matrix::stack_sparse_matrices(vec![&c0]),
        }
    } else if metadata.num_secondary_constraint_variables == 0
        && metadata.num_secondary_witness_values > 0
    {
        let (a1, b1, c1) =
            import_r1cs1_int_deprecated::<T>(data_dir, model, scale_factor, metadata, z, verbose);
        R1CSMatrices {
            a: Matrix::stack_sparse_matrices(vec![&a0, &a1]),
            b: Matrix::stack_sparse_matrices(vec![&b0, &b1]),
            c: Matrix::stack_sparse_matrices(vec![&c0, &c1]),
        }
    } else {
        panic!("deprecated path");
    }
}

pub fn import_witness_int_deprecated<T: FromF64Matrix>(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    scale_factor: AFloat,
    verbose: bool,
) -> Matrix<T> {
    if verbose {
        eprintln!("Importing witness for model: {}", model);
    }
    let z = data_file(data_dir, format!("{}/Z.bin", model).as_str());
    let (_, z) = Matrix::from_file(&z).expect("Failed to read Z");
    let mut z = T::f64_to_int(&z, scale_factor, None);
    z.set_ranges(&metadata.get_ranges());
    z
}

pub fn import_witness_raw_deprecated(
    data_dir: &Path,
    model: &str,
    metadata: &Metadata,
    verbose: bool,
) -> Matrix<f64> {
    if verbose {
        eprintln!("Importing witness for model: {}", model);
    }
    let z = data_file(data_dir, format!("{}/Z.bin", model).as_str());
    let (_, mut z) = Matrix::from_file(&z).expect("Failed to read Z");
    z.set_ranges(&metadata.get_ranges());
    z
}

pub fn import_r1cs0_int_deprecated<T: FromF64Matrix>(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    verbose: bool,
) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
    if verbose {
        eprintln!("Importing R1CS for model: {}", model);
    }
    // open data files
    let a = data_file(data_dir, format!("{model}/A0.bin").as_str());
    let b = data_file(data_dir, format!("{model}/B0.bin").as_str());
    let c = data_file(data_dir, format!("{model}/C0.bin").as_str());
    // load to matrices
    let (_, a) = Matrix::from_file(&a).expect("Failed to read matrix A");
    let (_, b) = Matrix::from_file(&b).expect("Failed to read matrix B");
    let (_, c) = Matrix::from_file(&c).expect("Failed to read matrix C");
    // print dimensions of a
    if verbose {
        eprintln!("A: {} rows, {} columns", a.height(), a.width());
        eprintln!("B: {} rows, {} columns", b.height(), b.width());
        eprintln!("C: {} rows, {} columns", c.height(), c.width());
    }
    // convert all matrices to integer precision
    (
        T::f64_to_int(&a, scale_factor.clone(), None),
        T::f64_to_int(&b, scale_factor.clone(), None),
        T::f64_to_int(&c, scale_factor, None),
    )
}

pub fn import_r1cs1_int_deprecated<T: FromF64Matrix>(
    data_dir: &Path,
    model: &str,
    scale_factor: AFloat,
    metadata: &Metadata,
    z: &Matrix<T>,
    verbose: bool,
) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
    if verbose {
        eprintln!("Importing R1CS for model: {}", model);
    }
    // open data files
    let a = data_file(data_dir, format!("{model}/A1.bin").as_str());
    let b = data_file(data_dir, format!("{model}/B1.bin").as_str());
    let c = data_file(data_dir, format!("{model}/C1.bin").as_str());
    // load to matrices
    let (_, a) = Matrix::from_file(&a).expect("Failed to read matrix A1");
    let (_, b) = Matrix::from_file(&b).expect("Failed to read matrix B1");
    let (_, c) = Matrix::from_file(&c).expect("Failed to read matrix C1");
    // print dimensions of a
    if verbose {
        eprintln!("A1: {} rows, {} columns", a.height(), a.width());
        eprintln!("B1: {} rows, {} columns", b.height(), b.width());
        eprintln!("C1: {} rows, {} columns", c.height(), c.width());
    }
    // convert all matrices to integer precision
    let z = if let MatrixData::Dense(z) = z.data() {
        z.clone().iter().copied().collect::<Vec<_>>()
    } else {
        panic!()
    };
    (
        T::f64_to_int(
            &a,
            scale_factor.clone(),
            Some(metadata.secondary_witness_shift()),
        ),
        T::f64_to_int(&b, scale_factor, None),
        T::f64_index_to_int(&c, &z, metadata.r_vec_shift()),
    )
}
