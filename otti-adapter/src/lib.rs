use lpsolve::prelude::*;
use model::{AFloat, HighPrecision};
use mps::{Parser, model::Model, types::RowType};
use parse::{
    generalized::InjectionInfo,
    mat::{Matrix, MatrixData},
};
use spain::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};
use sprs::{CsMat, CsVec, TriMat};
use std::{collections::HashMap, fs, ops::Range, path::Path};
use stream::bigvec::BigVec;

pub mod cons_adapter;
pub mod otti_exec;
pub use otti_exec::OttiExec;

type SparseMat = CsMat<f64>;
type Witness = Vec<f64>;
type SolutionVectors = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>); // primal, eq_dual, leq_dual, geq_dual
type LinearCombo = Vec<(usize, f64)>;
type Constraint = (LinearCombo, LinearCombo, LinearCombo);

#[derive(Clone, Debug)]
pub struct LPSpain {
    leq_matrix: SparseMat,
    geq_matrix: SparseMat,
    equality_matrix: SparseMat,
    b_eq: Vec<f64>,
    b_leq: Vec<f64>,
    b_geq: Vec<f64>,
    c: Vec<f64>,
    num_vars: usize,
    num_eqs: usize,
    num_leqs: usize,
    num_geqs: usize,
    leq_to_row: HashMap<String, usize>,
    geq_to_row: HashMap<String, usize>,
    eq_to_row: HashMap<String, usize>,
    model: Model<f64>,
    cons: Vec<Constraint>,
    dataset_path: String,
    witness: Option<Matrix<i128>>,
    solution: Option<SolutionVectors>,
}

// Loads MPS file and converts to Model<f64>
fn load_mps<P: AsRef<Path>>(path: P) -> Model<f64> {
    let contents = fs::read_to_string(path).expect("Failed to read MPS file");
    let parsed = Parser::<f64>::parse(&contents)
        .map_err(|e| format!("mps parse error: {e:?}"))
        .unwrap();
    Model::try_from(parsed).expect("Failed to convert parsed MPS to Model")
}

impl LPSpain {
    // Creates LPSpain instance from mps file path
    pub fn parse_mps(mps_path: &str) -> Self {
        let model = load_mps(mps_path);
        let num_equations = model.row_types.0.len();
        // Variable name => column idx
        let mut x_index_map = HashMap::new();
        // Inequality eq string idx => row idx
        let mut leq_to_row = HashMap::new();
        let mut geq_to_row = HashMap::new();
        // Equality eq string idx => row idx
        let mut eq_to_row = HashMap::new();
        // Create index map for variable names
        for (eq_idx, _) in model.values.0.iter() {
            let x_index_map_len = x_index_map.len();
            x_index_map
                .entry(eq_idx.1.clone())
                .or_insert_with(|| x_index_map_len);
        }
        // Create index maps for inequality and equality matrices
        for (row_name, row_type) in model.row_types.0.iter() {
            match row_type {
                RowType::Leq => {
                    let leq_matrix_len = leq_to_row.len();
                    leq_to_row
                        .entry(row_name.clone())
                        .or_insert_with(|| leq_matrix_len);
                }
                RowType::Geq => {
                    let geq_matrix_len = geq_to_row.len();
                    geq_to_row
                        .entry(row_name.clone())
                        .or_insert_with(|| geq_matrix_len);
                }
                RowType::Eq => {
                    let eq_matrix_len = eq_to_row.len();
                    eq_to_row
                        .entry(row_name.clone())
                        .or_insert_with(|| eq_matrix_len);
                }
                RowType::Nr => {}
            }
        }
        let num_vars = x_index_map.len();
        let num_leqs = leq_to_row.len();
        let num_geqs = geq_to_row.len();
        let num_eqs = eq_to_row.len();
        // Sanity check, subtract 1 for objective
        assert_eq!(num_eqs + num_leqs + num_geqs, num_equations - 1);
        let mut b_eq = vec![0.0; num_eqs];
        let mut b_leq = vec![0.0; num_leqs];
        let mut b_geq = vec![0.0; num_geqs];
        let mut c = vec![0.0; num_vars];
        let mut leq_trimat = TriMat::<f64>::new((num_leqs, num_vars));
        let mut geq_trimat = TriMat::<f64>::new((num_geqs, num_vars));
        let mut eq_trimat = TriMat::<f64>::new((num_eqs, num_vars));
        for ((eq_name, var_name), coeff) in model.values.0.iter() {
            let row_type = model.row_types.0.get(eq_name).expect("Row type not found");
            let col_idx = *x_index_map.get(var_name).expect("Variable index not found");
            match row_type {
                // If type E, add to equality matrix
                RowType::Eq => {
                    let row_idx = *eq_to_row
                        .get(eq_name)
                        .expect("Equality row index not found");
                    eq_trimat.add_triplet(row_idx, col_idx, *coeff);
                }
                // If type L or G, add to inequality matrix (with appropriate sign)
                RowType::Leq => {
                    let row_idx = *leq_to_row
                        .get(eq_name)
                        .expect("Inequality row index not found");
                    leq_trimat.add_triplet(row_idx, col_idx, *coeff);
                }
                RowType::Geq => {
                    let row_idx = *geq_to_row
                        .get(eq_name)
                        .expect("Inequality row index not found");
                    geq_trimat.add_triplet(row_idx, col_idx, *coeff);
                }
                // If type OBJ, set c, the objective coefficients
                RowType::Nr => {
                    c[col_idx] = *coeff;
                }
            }
        }
        let leq_matrix = leq_trimat.to_csr();
        let geq_matrix = geq_trimat.to_csr();
        let equality_matrix = eq_trimat.to_csr();
        // Iterate over rhs to populate b_eq and b_ineq vectors
        assert_eq!(model.rhs.0.len(), 1);
        for (row_name, value) in model.rhs.0.first().unwrap().1.iter() {
            let row_type = model
                .row_types
                .0
                .get(row_name)
                .expect("Row type not found for RHS");
            match row_type {
                RowType::Eq => {
                    let row_idx = *eq_to_row
                        .get(row_name)
                        .expect("Equalirty row index not found for RHS");
                    b_eq[row_idx] = *value;
                }
                RowType::Leq => {
                    let row_idx = *leq_to_row
                        .get(row_name)
                        .expect("Inequality row index not found for RHS");
                    b_leq[row_idx] = *value;
                }
                RowType::Geq => {
                    let row_idx = *geq_to_row
                        .get(row_name)
                        .expect("Inequality row index not found for RHS");
                    b_geq[row_idx] = *value;
                }
                RowType::Nr => {}
            }
        }
        let mut lp = LPSpain {
            leq_matrix,
            geq_matrix,
            equality_matrix,
            dataset_path: mps_path.to_string(),
            b_eq,
            b_leq,
            b_geq,
            c,
            leq_to_row,
            geq_to_row,
            eq_to_row,
            model,
            num_vars,
            num_eqs,
            num_leqs,
            num_geqs,
            cons: vec![],
            witness: None,
            solution: None,
        };
        lp.solution = Some(lp.solve());
        lp.generate_constraint();
        lp
    }

    pub fn generate_constraint(&mut self) {
        // Witness vector layout:
        // [1 | x | y_eq | y_leq | y_geq | sqrt_x | sqrt_y_leq | sqrt_y_geq | sqrt_leq_slack | sqrt_geq_slack | sqrt_dual_slack]
        let offset_x = 1;
        let offset_y_eq = offset_x + self.num_vars;
        let offset_y_leq = offset_y_eq + self.num_eqs;
        let offset_y_geq = offset_y_leq + self.num_leqs;
        let offset_sqrt_x = offset_y_geq + self.num_geqs;
        let offset_sqrt_y_leq = offset_sqrt_x + self.num_vars;
        let offset_sqrt_y_geq = offset_sqrt_y_leq + self.num_leqs;
        let offset_sqrt_slack_leq = offset_sqrt_y_geq + self.num_geqs;
        let offset_sqrt_slack_geq = offset_sqrt_slack_leq + self.num_leqs;
        let offset_sqrt_dual_slack = offset_sqrt_slack_geq + self.num_geqs;

        let mut constraints = Vec::new();

        for i in 0..self.num_vars {
            let sqrt_idx = offset_sqrt_x + i;
            let lc = vec![(sqrt_idx, 1.0)];
            constraints.push((lc.clone(), lc.clone(), vec![(offset_x + i, 1.0)]));
        }

        for i in 0..self.num_leqs {
            let sqrt_idx = offset_sqrt_y_leq + i;
            let lc = vec![(sqrt_idx, 1.0)];
            constraints.push((lc.clone(), lc.clone(), vec![(offset_y_leq + i, -1.0)]));
        }

        for i in 0..self.num_geqs {
            let sqrt_idx = offset_sqrt_y_geq + i;
            let lc = vec![(sqrt_idx, 1.0)];
            constraints.push((lc.clone(), lc.clone(), vec![(offset_y_geq + i, 1.0)]));
        }

        // Equality constraints: A_eq x = b_eq
        let one_lc = vec![(0, 1.0)];
        for (row_idx, row) in self.equality_matrix.outer_iterator().enumerate() {
            let mut lhs = Vec::with_capacity(row.nnz() + 1);
            for (col_idx, coeff) in row.iter() {
                lhs.push((offset_x + col_idx, *coeff));
            }
            if self.b_eq[row_idx] != 0.0 {
                lhs.push((0, -self.b_eq[row_idx]));
            }
            constraints.push((lhs, one_lc.clone(), Vec::new()));
        }

        // <= constraints: slack = b - A x = sqrt^2
        for (row_idx, row) in self.leq_matrix.outer_iterator().enumerate() {
            let sqrt_idx = offset_sqrt_slack_leq + row_idx;
            let sqrt_lc = vec![(sqrt_idx, 1.0)];
            let mut slack_lc = Vec::with_capacity(row.nnz() + 1);
            if self.b_leq[row_idx] != 0.0 {
                slack_lc.push((0, self.b_leq[row_idx]));
            }
            for (col_idx, coeff) in row.iter() {
                slack_lc.push((offset_x + col_idx, -*coeff));
            }
            constraints.push((sqrt_lc.clone(), sqrt_lc, slack_lc));
        }

        // >= constraints: slack = A x - b = sqrt^2
        for (row_idx, row) in self.geq_matrix.outer_iterator().enumerate() {
            let sqrt_idx = offset_sqrt_slack_geq + row_idx;
            let sqrt_lc = vec![(sqrt_idx, 1.0)];
            let mut slack_lc = Vec::with_capacity(row.nnz() + 1);
            if self.b_geq[row_idx] != 0.0 {
                slack_lc.push((0, -self.b_geq[row_idx]));
            }
            for (col_idx, coeff) in row.iter() {
                slack_lc.push((offset_x + col_idx, *coeff));
            }
            constraints.push((sqrt_lc.clone(), sqrt_lc, slack_lc));
        }

        // Dual feasibility: sqrt(c - A^T y)^2 = c - A^T y
        let mut dual_rhs: Vec<LinearCombo> = (0..self.num_vars)
            .map(|j| {
                let mut lc = Vec::new();
                if self.c[j] != 0.0 {
                    lc.push((0, self.c[j]));
                }
                lc
            })
            .collect();
        for (row_idx, row) in self.equality_matrix.outer_iterator().enumerate() {
            for (col_idx, coeff) in row.iter() {
                dual_rhs[col_idx].push((offset_y_eq + row_idx, -*coeff));
            }
        }
        for (row_idx, row) in self.leq_matrix.outer_iterator().enumerate() {
            for (col_idx, coeff) in row.iter() {
                dual_rhs[col_idx].push((offset_y_leq + row_idx, -*coeff));
            }
        }
        for (row_idx, row) in self.geq_matrix.outer_iterator().enumerate() {
            for (col_idx, coeff) in row.iter() {
                dual_rhs[col_idx].push((offset_y_geq + row_idx, -*coeff));
            }
        }
        for (var_idx, rhs) in dual_rhs.iter().enumerate() {
            let sqrt_idx = offset_sqrt_dual_slack + var_idx;
            let sqrt_lc = vec![(sqrt_idx, 1.0)];
            constraints.push((sqrt_lc.clone(), sqrt_lc, rhs.clone()));
        }

        // Strong duality constraint: c^T x = b_eq^T y_eq + b_leq^T y_leq + b_geq^T y_geq
        let mut primal_obj_lc = Vec::with_capacity(self.num_vars);
        for (j, coeff) in self.c.iter().enumerate() {
            if *coeff != 0.0 {
                primal_obj_lc.push((offset_x + j, *coeff));
            }
        }
        let mut dual_obj_lc = Vec::with_capacity(self.num_eqs + self.num_leqs + self.num_geqs);
        for (i, coeff) in self.b_eq.iter().enumerate() {
            if *coeff != 0.0 {
                dual_obj_lc.push((offset_y_eq + i, *coeff));
            }
        }
        for (i, coeff) in self.b_leq.iter().enumerate() {
            if *coeff != 0.0 {
                dual_obj_lc.push((offset_y_leq + i, *coeff));
            }
        }
        for (i, coeff) in self.b_geq.iter().enumerate() {
            if *coeff != 0.0 {
                dual_obj_lc.push((offset_y_geq + i, *coeff));
            }
        }
        constraints.push((primal_obj_lc, vec![(0, 1.0)], dual_obj_lc));
        self.cons = constraints;
    }

    pub fn solve(&self) -> SolutionVectors {
        let mut problem =
            ProblemBuilder::from_fixedmps_file(self.dataset_path.clone(), MPSOptions::empty())
                .unwrap();
        match problem.solve() {
            SolveStatus::Optimal => {}
            _ => panic!("LP did not solve to optimality"),
        };
        // Extarct primal
        let mut primal = vec![0.0; self.num_vars];
        let primal = problem
            .get_solution_variables(&mut primal)
            .unwrap()
            .to_vec();
        assert_eq!(primal.len(), self.num_vars);
        // Extract dual
        let cols = self.num_vars;
        let rows = self.num_eqs + self.num_leqs + self.num_geqs + 1;
        let mut dual_buf = vec![0.0; rows + cols];
        let dual_all = problem.get_dual_solution(&mut dual_buf).unwrap();
        let row_duals = &dual_all[1..rows];
        // Remap dual based on our row ordering
        let mut eq_dual = vec![0.0; self.num_eqs];
        let mut leq_dual = vec![0.0; self.num_leqs];
        let mut geq_dual = vec![0.0; self.num_geqs];
        let mut constr_idx = 0;
        for (row_name, row_type) in self.model.row_types.0.iter() {
            match row_type {
                RowType::Nr => {
                    // objective row: no dual
                }
                RowType::Eq => {
                    let i = self.eq_to_row[row_name];
                    eq_dual[i] = row_duals[constr_idx];
                    constr_idx += 1;
                }
                RowType::Leq => {
                    let i = self.leq_to_row[row_name];
                    leq_dual[i] = row_duals[constr_idx];
                    constr_idx += 1;
                }
                RowType::Geq => {
                    let i = self.geq_to_row[row_name];
                    geq_dual[i] = row_duals[constr_idx];
                    constr_idx += 1;
                }
            }
        }
        assert_eq!(constr_idx, row_duals.len());
        (primal, eq_dual, leq_dual, geq_dual)
    }

    pub fn check_opt_certifiate(&self, sols: &SolutionVectors) {
        let (primal, eq_dual, leq_dual, geq_dual) = sols;
        // Check that primal is correct w.r.t to our parsed lp
        let x_vec = CsVec::new(self.num_vars, (0..self.num_vars).collect(), primal.clone());
        // Check equality constraints
        let ax_eq = (&self.equality_matrix * &x_vec).to_dense();
        for (i, val) in ax_eq.into_iter().enumerate() {
            assert!((val - self.b_eq[i]).abs() < 1e-8);
        }
        // Check <= constraints
        let ax_leq = (&self.leq_matrix * &x_vec).to_dense();
        for (i, val) in ax_leq.iter().enumerate() {
            assert!(*val <= self.b_leq[i] + 1e-8);
        }
        // Check >= constraints
        let ax_geq = (&self.geq_matrix * &x_vec).to_dense();
        for (i, val) in ax_geq.iter().enumerate() {
            assert!(*val + 1e-8 >= self.b_geq[i]);
        }
        // Assert that x is non-negative
        for val in primal.iter() {
            assert!(*val >= -1e-8);
        }
        // Check the dual is correct w.r.t to our parsed lp
        let z = eq_dual;
        let y_leq = leq_dual;
        let y_geq = geq_dual;

        for (i, &yi) in y_leq.iter().enumerate() {
            assert!(yi <= 1e-8, "leq dual y_leq[{}] positive", i);
        }
        for (i, &yi) in y_geq.iter().enumerate() {
            assert!(yi >= -1e-8, "geq dual y_geq[{}] negative", i);
        }
        // Complementary slackness is implied by the optimality checks below, so no direct check here.
        let mut dual_sum = vec![0.0; self.num_vars];
        for (row_idx, row) in self.equality_matrix.outer_iterator().enumerate() {
            let y_val = z[row_idx];
            if y_val == 0.0 {
                continue;
            }
            for (col_idx, coeff) in row.iter() {
                dual_sum[col_idx] += coeff * y_val;
            }
        }
        for (row_idx, row) in self.leq_matrix.outer_iterator().enumerate() {
            let y_val = y_leq[row_idx];
            if y_val == 0.0 {
                continue;
            }
            for (col_idx, coeff) in row.iter() {
                dual_sum[col_idx] += coeff * y_val;
            }
        }
        for (row_idx, row) in self.geq_matrix.outer_iterator().enumerate() {
            let y_val = y_geq[row_idx];
            if y_val == 0.0 {
                continue;
            }
            for (col_idx, coeff) in row.iter() {
                dual_sum[col_idx] += coeff * y_val;
            }
        }
        // Dual feasibility (minimization form): A^T y <= c
        for j in 0..self.num_vars {
            let aty = dual_sum[j];
            assert!(
                aty <= self.c[j] + 1e-8,
                "dual feasibility violated (A^T y > c) at j={}: A^T y = {}, c_j={}",
                j,
                aty,
                self.c[j]
            );
        }
        // Strong duality: b^T y = c^T x
        let primal_obj: f64 = self
            .c
            .iter()
            .zip(primal.iter())
            .map(|(cj, xj)| cj * xj)
            .sum();
        let dual_obj_leq: f64 = y_leq
            .iter()
            .zip(self.b_leq.iter())
            .map(|(y, b)| y * b)
            .sum();
        let dual_obj_geq: f64 = y_geq
            .iter()
            .zip(self.b_geq.iter())
            .map(|(y, b)| y * b)
            .sum();
        let dual_obj_eq: f64 = z.iter().zip(self.b_eq.iter()).map(|(zv, b)| zv * b).sum();
        let dual_obj: f64 = dual_obj_leq + dual_obj_geq + dual_obj_eq;
        assert!(
            (dual_obj - primal_obj).abs() <= 1e-8,
            "strong duality violated: dual={}, primal={}",
            dual_obj,
            primal_obj
        );
    }

    fn witness_width(&self) -> usize {
        1 + self.num_vars
            + self.num_eqs
            + self.num_leqs
            + self.num_geqs
            + self.num_vars
            + self.num_leqs
            + self.num_geqs
            + self.num_leqs
            + self.num_geqs
            + self.num_vars
    }

    pub fn get_metadata(&self) -> Metadata {
        let num_public_values = 1 + self.num_vars + self.num_eqs + self.num_leqs + self.num_geqs;
        let total_len = self.witness_width();
        Metadata {
            num_public_values,
            num_random_values: 0,
            num_witness_values: total_len - num_public_values,
            num_secondary_witness_values: 0,
            num_secondary_constraint_variables: 0,
            primary_output_labels: vec![],
            secondary_output_labels: vec![],
        }
    }

    pub fn get_ranges(&self) -> Vec<Range<usize>> {
        self.get_metadata().get_ranges()
    }

    // Generate A/B/C Spain R1CS matrices from LPSpain constraints
    pub fn generate_r1cs_matrices(
        &self,
        scale_factor: AFloat,
    ) -> (Matrix<i128>, Matrix<i128>, Matrix<i128>) {
        assert!(
            !self.cons.is_empty(),
            "generate_constraint must be called before exporting R1CS matrices"
        );
        let num_rows = self.cons.len();
        let width = self.witness_width();
        let mut a_entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut b_entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut c_entries: Vec<(usize, usize, f64)> = Vec::new();

        let push_entries = |dst: &mut Vec<(usize, usize, f64)>, row: usize, lc: &LinearCombo| {
            for &(col, coeff) in lc.iter() {
                if coeff != 0.0 {
                    dst.push((row, col, coeff));
                }
            }
        };

        for (row_idx, (a_lc, b_lc, c_lc)) in self.cons.iter().enumerate() {
            push_entries(&mut a_entries, row_idx, a_lc);
            push_entries(&mut b_entries, row_idx, b_lc);
            push_entries(&mut c_entries, row_idx, c_lc);
        }

        let build_matrix = |entries: Vec<(usize, usize, f64)>, label: &str| -> Matrix<i128> {
            let data = MatrixData::COO(BigVec::from_vec(entries));
            let mat_f64 = Matrix::new(data, width, num_rows, None, label.to_string());
            Matrix::<i128>::from_f64(&mat_f64, scale_factor.clone(), None)
        };

        let a = build_matrix(a_entries, "LPSpain A");
        let b = build_matrix(b_entries, "LPSpain B");
        let c = build_matrix(c_entries, "LPSpain C");

        (a, b, c)
    }

    pub fn get_witness_raw(&self, sols: &SolutionVectors) -> Witness {
        let (x, z_eq, z_leq, z_geq) = sols;
        let mut witness = Vec::with_capacity(
            1 + self.num_vars
                + self.num_eqs
                + self.num_leqs
                + self.num_geqs
                + self.num_vars
                + self.num_leqs
                + self.num_geqs
                + self.num_leqs
                + self.num_geqs
                + self.num_vars,
        );
        witness.push(1.0);
        witness.extend_from_slice(x);
        witness.extend_from_slice(z_eq);
        witness.extend_from_slice(z_leq);
        witness.extend_from_slice(z_geq);
        witness.extend(x.iter().map(|xi| xi.max(&0.0).sqrt()));
        witness.extend(z_leq.iter().map(|yi| (-yi).max(0.0).sqrt()));
        witness.extend(z_geq.iter().map(|yi| yi.max(&0.0).sqrt()));
        // Compute primal slacks
        let x_vec = CsVec::new(self.num_vars, (0..self.num_vars).collect(), x.clone());
        let ax_leq = (&self.leq_matrix * &x_vec).to_dense();
        let ax_geq = (&self.geq_matrix * &x_vec).to_dense();
        let slack_leq: Vec<f64> = self
            .b_leq
            .iter()
            .zip(ax_leq.iter())
            .map(|(b, val)| b - val)
            .collect();
        let slack_geq: Vec<f64> = ax_geq
            .iter()
            .zip(self.b_geq.iter())
            .map(|(val, b)| val - b)
            .collect();
        witness.extend(slack_leq.iter().map(|s| s.max(&0.0).sqrt()));
        witness.extend(slack_geq.iter().map(|s| s.max(&0.0).sqrt()));
        // Dual slacks
        let mut dual_sum = vec![0.0; self.num_vars];
        for (row_idx, row) in self.equality_matrix.outer_iterator().enumerate() {
            let y_val = z_eq[row_idx];
            if y_val == 0.0 {
                continue;
            }
            for (col_idx, coeff) in row.iter() {
                dual_sum[col_idx] += coeff * y_val;
            }
        }
        for (row_idx, row) in self.leq_matrix.outer_iterator().enumerate() {
            let y_val = z_leq[row_idx];
            if y_val == 0.0 {
                continue;
            }
            for (col_idx, coeff) in row.iter() {
                dual_sum[col_idx] += coeff * y_val;
            }
        }
        for (row_idx, row) in self.geq_matrix.outer_iterator().enumerate() {
            let y_val = z_geq[row_idx];
            if y_val == 0.0 {
                continue;
            }
            for (col_idx, coeff) in row.iter() {
                dual_sum[col_idx] += coeff * y_val;
            }
        }
        let dual_slack: Vec<f64> = (0..self.num_vars)
            .map(|j| self.c[j] - dual_sum[j])
            .collect();
        witness.extend(dual_slack.iter().map(|s| s.max(&0.0).sqrt()));
        witness
    }
    pub fn build_witness(&self, sols: &SolutionVectors, scale_factor: AFloat) -> Matrix<i128> {
        let witness_vec = self.get_witness_raw(sols);
        let witness_len = witness_vec.len();
        let witness_matrix = Matrix::new(
            MatrixData::Dense(BigVec::from_vec(witness_vec)),
            1,
            witness_len,
            None,
            "LPSpain witness".to_string(),
        );
        let mut scaled = Matrix::<i128>::from_f64(&witness_matrix, scale_factor, None);
        let ranges = self.get_ranges();
        scaled.set_ranges(&ranges);
        scaled
    }
}

impl R1CSInstance<AFloat, i128> for LPSpain {
    fn compute_commit_witness(&mut self, scale_factor: AFloat, batch_size: usize) -> Matrix<i128> {
        let mut ret = Vec::new();
        for _ in 0..batch_size {
            ret.push(self.build_witness(self.solution.as_ref().unwrap(), scale_factor.clone()));
        }
        let mut ret = Matrix::stack_dense_matrices_horizontally(ret.iter().collect());
        ret.set_ranges(&self.get_ranges());
        self.witness = Some(ret.clone());
        ret.extract_rows(&ret.ranges().unwrap()[1])
    }

    fn compute_full_witness(
        &mut self,
        _metadata: &Metadata,
        _random_values: Vec<AFloat>,
        _scale_factor: AFloat,
    ) -> Matrix<i128> {
        self.witness.take().unwrap()
    }

    fn get_matrices(
        &self,
        scale_factor: AFloat,
        randomness: Option<&Vec<i128>>,
    ) -> (R1CSMatrices<i128>, Option<InjectionInfo>) {
        assert!(randomness.is_none(), "no need for randomness for otti");
        let (a, b, c) = self.generate_r1cs_matrices(scale_factor);
        (R1CSMatrices { a, b, c }, None)
    }

    fn get_meta(&self) -> Metadata {
        self.get_metadata()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use model::FromPrimitive;
    use spain::simulate::stateful_simulate;

    use super::*;

    fn eval_lc(lc: &LinearCombo, witness: &[f64]) -> f64 {
        lc.iter()
            .map(|(idx, coeff)| coeff * witness[*idx])
            .sum::<f64>()
    }

    #[test]
    fn smoke_tests() {
        let datasets = vec![
            "adlittle", "afiro", "bnl1", "sc105", "sc50a", "sc50b", "scagr7", "scsd8",
        ];
        for dataset in datasets {
            smoke_test_lp(&format!("./datasets/{}.mps", dataset));
        }
    }

    #[test]
    fn smoke_test() {
        smoke_test_lp(&"./datasets/bnl1.mps");
    }

    #[test]
    fn test_stateful_simulate() {
        let dataset = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets/bnl1.mps");
        let lp = LPSpain::parse_mps(dataset.to_str().expect("dataset path should be valid"));
        stateful_simulate::<model::AFloat, _, i128>(lp, None);
    }

    fn smoke_test_lp(dataset: &str) {
        let mut lp = LPSpain::parse_mps(&dataset);
        let sols = lp.solve();
        lp.check_opt_certifiate(&sols);
        // Check R1CS constraints
        lp.generate_constraint();
        println!("Generated {} constraints", lp.cons.len());
        println!("Checking R1CS constraints...");
        let witness = lp.get_witness_raw(&sols);
        let mut total_squared_error: f64 = 0f64;
        for (idx, (a, b, c)) in lp.cons.iter().enumerate() {
            let lhs = eval_lc(a, &witness) * eval_lc(b, &witness);
            let rhs = eval_lc(c, &witness);
            let err = (lhs - rhs).abs();
            total_squared_error += err * err;
            assert!(
                (lhs - rhs).abs() <= 1e-8,
                "witness constraint {} violated: lhs={}, rhs={}",
                idx,
                lhs,
                rhs
            );
        }
        let witness_matrix = lp.build_witness(&sols, AFloat::from_f64(1.0).unwrap());
        println!("Error for {}:{:e}", dataset, total_squared_error.sqrt());
        assert_eq!(witness_matrix.height(), witness.len());
    }
}
