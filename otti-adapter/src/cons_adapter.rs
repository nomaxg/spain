use parse::generalized::I256;
use parse::mat::{Matrix, MatrixData};
use spain::inputs::R1CSMatrices;
use spain::prover::mul_to_vec_i1024;
use spzk::{R1cs, R1csReader};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use stream::bigvec::BigVec;

type SpartanTriplet = (usize, usize, [u8; 32]);

fn read_file_bytes(path: &PathBuf) -> Vec<u8> {
    let msg = format!("{}", path.display());
    let mut file = File::open(path).expect(&msg);
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect(&msg);
    buffer
}

fn get_otti_paths(module_name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lp-zkif");
    let header_path = base.join(format!("{module_name}.mps.inp.zkif"));
    let circuit_path = base.join(format!("{module_name}.mps.zkif"));
    let witness_path = base.join(format!("{module_name}.mps.wit.zkif"));
    (header_path, circuit_path, witness_path)
}

fn read_otti_buffers(module_name: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (header_path, circuit_path, witness_path) = get_otti_paths(module_name);
    let header_buf = read_file_bytes(&header_path);
    let circuit_buf = read_file_bytes(&circuit_path);
    let witness_buf = read_file_bytes(&witness_path);
    (header_buf, circuit_buf, witness_buf)
}

fn col_spartan_to_otti(col: usize, num_witness: usize, num_inputs: usize) -> usize {
    if col < num_witness {
        1 + num_inputs + col
    } else {
        col - num_witness
    }
}

fn coeff_bytes_to_i256(bytes: [u8; 32]) -> I256 {
    I256::from_le_bytes(bytes)
}

fn map_triplets_to_matrix(
    triplets: &[SpartanTriplet],
    num_witness: usize,
    num_inputs: usize,
    num_cols: usize,
    num_rows: usize,
    label: &str,
) -> Matrix<I256> {
    let entries = triplets
        .iter()
        .map(|(row, col, coeff)| {
            (
                *row,
                col_spartan_to_otti(*col, num_witness, num_inputs),
                coeff_bytes_to_i256(*coeff),
            )
        })
        .collect::<Vec<_>>();
    Matrix::new(
        MatrixData::COO(BigVec::from_vec(entries)),
        num_cols,
        num_rows,
        None,
        label.to_string(),
    )
}

fn filter_matrix_rows(matrix: &Matrix<I256>, keep_row: &[bool]) -> Matrix<I256> {
    match matrix.data() {
        MatrixData::COO(entries) => {
            let mut old_to_new = vec![usize::MAX; keep_row.len()];
            let mut num_removed = 0usize;
            for (row, keep) in keep_row.iter().enumerate() {
                if !*keep {
                    num_removed += 1;
                } else {
                    old_to_new[row] = row - num_removed;
                }
            }
            let filtered = entries
                .iter()
                .filter_map(|(r, c, v)| {
                    if keep_row[*r] {
                        Some((old_to_new[*r], *c, *v))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            Matrix::new(
                MatrixData::COO(BigVec::from_vec(filtered)),
                matrix.width(),
                keep_row.len() - num_removed,
                None,
                format!("filtered {}", matrix.comment()),
            )
        }
        _ => panic!("Expected COO matrix for R1CS filtering"),
    }
}

pub fn filter_failing_constraints(
    matrices: &R1CSMatrices<I256>,
    witness: &Matrix<I256>,
) -> (R1CSMatrices<I256>, usize) {
    eprintln!(
        "filter_failing_constraints: start (constraints={}, witness_rows={}, witness_cols={})",
        matrices.a.height(),
        witness.height(),
        witness.width()
    );
    let az = mul_to_vec_i1024(&matrices.a, witness);
    let bz = mul_to_vec_i1024(&matrices.b, witness);
    let cz = mul_to_vec_i1024(&matrices.c, witness);
    let z_width = witness.width();
    let mut keep_row = vec![true; matrices.a.height()];
    let mut raw_failed = 0usize;

    for (idx, ((a_i, b_i), c_i)) in az.iter().zip(bz.iter()).zip(cz.iter()).enumerate() {
        if (*a_i * *b_i) != *c_i {
            let row = idx / z_width;
            if keep_row[row] {
                keep_row[row] = false;
                raw_failed += 1;
            }
        }
    }

    let filtered = R1CSMatrices {
        a: filter_matrix_rows(&matrices.a, &keep_row),
        b: filter_matrix_rows(&matrices.b, &keep_row),
        c: filter_matrix_rows(&matrices.c, &keep_row),
    };
    (filtered, raw_failed)
}

pub fn get_otti_r1cs(module_name: &str) -> R1cs {
    eprintln!("parsing raw R1CS for {module_name}");
    let (mut header_buf, mut circuit_buf, mut witness_buf) = read_otti_buffers(module_name);

    let reader = R1csReader::new(&mut header_buf, &mut circuit_buf, &mut witness_buf);
    let r1cs = R1cs::new(reader);
    r1cs
}

pub fn get_otti_r1cs_matrices(module_name: &str) -> (R1CSMatrices<I256>, Matrix<I256>) {
    let (mut header_buf, mut circuit_buf, mut witness_buf) = read_otti_buffers(module_name);
    let reader = R1csReader::new(&mut header_buf, &mut circuit_buf, &mut witness_buf);
    let r1cs = R1cs::new(reader);

    let num_constraints = r1cs.constraints.len();
    let num_inputs = r1cs.inputs.len();
    let num_witness = r1cs.witness.len();
    let num_cols = 1 + num_inputs + num_witness;

    let mut a_raw = Vec::new();
    let mut b_raw = Vec::new();
    let mut c_raw = Vec::new();
    r1cs.instance(&mut a_raw, &mut b_raw, &mut c_raw);

    let a = map_triplets_to_matrix(
        &a_raw,
        num_witness,
        num_inputs,
        num_cols,
        num_constraints,
        "spzk A",
    );
    let b = map_triplets_to_matrix(
        &b_raw,
        num_witness,
        num_inputs,
        num_cols,
        num_constraints,
        "spzk B",
    );
    let c = map_triplets_to_matrix(
        &c_raw,
        num_witness,
        num_inputs,
        num_cols,
        num_constraints,
        "spzk C",
    );

    let input_values = r1cs.inputs.iter().map(|v| v.value).collect::<Vec<_>>();
    let witness_values = r1cs.witness.iter().map(|v| v.value).collect::<Vec<_>>();
    assert_eq!(input_values.len(), num_inputs);
    assert_eq!(witness_values.len(), num_witness);

    let mut z = Vec::with_capacity(num_cols);
    z.push(I256::from(1));
    z.extend(input_values.into_iter().map(coeff_bytes_to_i256));
    z.extend(witness_values.into_iter().map(coeff_bytes_to_i256));

    let witness = Matrix::new(
        MatrixData::Dense(BigVec::from_vec(z)),
        1,
        num_cols,
        None,
        "spzk witness [1; public_input; witness]".to_string(),
    );

    let matrices = R1CSMatrices { a, b, c };
    let (_, num_failed) = filter_failing_constraints(&matrices, &witness);
    let failed_pct = (num_failed as f64) * 100.0 / (num_constraints as f64);
    eprintln!(
        "filtered constraints: removed {} ({:.4}%), kept {}",
        num_failed,
        failed_pct,
        matrices.a.height()
    );

    (matrices, witness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_otti_r1cs() {
        let r1cs = get_otti_r1cs("afiro");
        let num_cons = r1cs.constraints.len();
        // check against Otti's reported results
        assert_eq!(num_cons, 36811);
    }

    #[test]
    fn test_get_otti_r1cs_matrices() {
        let otti_r1cs = get_otti_r1cs("afiro");
        let (r1cs, witness) = get_otti_r1cs_matrices("afiro");
        assert_eq!(r1cs.a.height(), 36811);
        assert_eq!(r1cs.b.height(), 36811);
        assert_eq!(r1cs.c.height(), 36811);
        assert_eq!(witness.width(), 1);
        assert_eq!(
            witness.height(),
            otti_r1cs.inputs.len() + otti_r1cs.witness.len() + 1
        );
    }

    #[test]
    // Constraint failures in i1024 SHOULD be due to field dependent reductions
    fn test_constraints() {
        let otti_r1cs = get_otti_r1cs("afiro");
        let (matrices, witness) = get_otti_r1cs_matrices("afiro");
        let raw_num_constraints = matrices.a.height();
        let (filtered, _) = filter_failing_constraints(&matrices, &witness);

        let prime = I256::from_le_bytes(otti_r1cs.field_max) + I256::from(1);
        let prime_1024 = spain::traits::ToI1024::to_i1024(prime);
        let zero = prime_1024 - prime_1024;

        let az = mul_to_vec_i1024(&matrices.a, &witness);
        let bz = mul_to_vec_i1024(&matrices.b, &witness);
        let cz = mul_to_vec_i1024(&matrices.c, &witness);

        let raw_failed = az
            .iter()
            .zip(bz.iter())
            .zip(cz.iter())
            .filter(|((a_i, b_i), c_i)| (**a_i * **b_i) != **c_i)
            .count();
        let mod_failed = az
            .iter()
            .zip(bz.iter())
            .zip(cz.iter())
            .filter(|((a_i, b_i), c_i)| ((**a_i * **b_i) - **c_i) % prime_1024 != zero)
            .count();

        let raw_pct = 100.0 * (raw_failed as f64) / (matrices.a.height() as f64);
        let mod_pct = 100.0 * (mod_failed as f64) / (matrices.a.height() as f64);
        println!(
            "raw_failed={} ({:.4}%), mod_failed={} ({:.4}%), prime={}",
            raw_failed, raw_pct, mod_failed, mod_pct, prime
        );

        assert!(
            mod_failed < raw_failed,
            "mod reduction did not reduce failures: raw_failed={}, mod_failed={}",
            raw_failed,
            mod_failed
        );
        assert_eq!(mod_failed, 0);
        assert_eq!(filtered.a.height(), raw_num_constraints - raw_failed);
        assert_eq!(filtered.b.height(), raw_num_constraints - raw_failed);
        assert_eq!(filtered.c.height(), raw_num_constraints - raw_failed);
    }
}
