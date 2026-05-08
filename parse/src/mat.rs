use std::ops::Range;
use stream::bigvec::BigVec;

pub type SubRanges = Vec<Range<usize>>;

// TO DO, change these to usee BigVec
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixData<T: Clone + Default + PartialEq> {
    Dense(BigVec<T>),
    COO(BigVec<(usize, usize, T)>), // (row, col, value)
}

// TO DO, consider adding denominator directly to MatrixData (only for rational matrices)
#[derive(Debug, Clone)]
pub struct Matrix<T: Clone + Default + PartialEq> {
    data: MatrixData<T>,
    ranges: Option<SubRanges>,
    width: usize,
    height: usize,
    comment: String,
}

impl<T: Clone + Default + PartialEq> Matrix<T> {
    pub fn new(
        data: MatrixData<T>,
        width: usize,
        height: usize,
        ranges: Option<SubRanges>,
        comment: String,
    ) -> Self {
        Matrix {
            data,
            width,
            height,
            ranges,
            comment,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn ranges(&self) -> Option<&SubRanges> {
        self.ranges.as_ref()
    }

    pub fn num_entries(&self) -> usize {
        match &self.data {
            MatrixData::Dense(values) => values.len(),
            MatrixData::COO(entries) => entries.len(),
        }
    }

    pub fn comment(&self) -> &str {
        &self.comment
    }

    pub fn data(&self) -> &MatrixData<T> {
        &self.data
    }

    pub fn mut_data(&mut self) -> &mut MatrixData<T> {
        &mut self.data
    }

    pub fn into_data(self) -> MatrixData<T> {
        self.data
    }

    pub fn set_ranges(&mut self, ranges: &SubRanges) {
        self.ranges = Some(ranges.clone());
    }

    pub fn from_vec(values: Vec<T>) -> Self {
        let len = values.len();
        Matrix::new(
            MatrixData::Dense(BigVec::from_vec(values)),
            1,
            len,
            None,
            "Dense matrix from vec".to_string(),
        )
    }

    pub fn from_coo(entries: Vec<(usize, usize, T)>, width: usize, height: usize) -> Self {
        Matrix::new(
            MatrixData::COO(BigVec::from_vec(entries)),
            width,
            height,
            None,
            "COO matrix from vec".to_string(),
        )
    }
}

impl<T: Clone + Default + PartialEq> Matrix<T> {
    // get the ith row as a slice
    pub fn get_row(&self, i: usize) -> Vec<T> {
        let mut row = vec![T::default(); self.width];
        match &self.data {
            MatrixData::Dense(values) => {
                for j in 0..self.width {
                    row[j] = values[i * self.width + j].clone();
                }
            }
            _ => {
                panic!("Expected Dense matrix for get_row");
            }
        }
        row
    }
    // get the jth column as a slice
    pub fn get_col(&self, j: usize) -> Vec<T> {
        let mut column = vec![T::default(); self.height];
        match &self.data {
            MatrixData::Dense(values) => {
                for i in 0..self.height {
                    column[i] = values[i * self.width + j].clone();
                }
            }
            _ => {
                panic!("Expected Dense matrix for get_row");
            }
        }
        column
    }
    // take vector turn it into one where it is placed adjacent to itself `times` times
    pub fn repeat_column(&self, times: usize) -> Self {
        assert!(
            self.width == 1,
            "Matrix must have exactly one column to repeat it"
        );
        match &self.data {
            MatrixData::Dense(values) => {
                /*let mut new_values = Vec::with_capacity(self.height * times);
                for value in values.iter() {
                    for _ in 0..times {
                        new_values.push(value.clone());
                    }
                }*/
                let mut new_values = BigVec::new(self.height * times).unwrap();
                for (i, value) in values.iter().enumerate() {
                    for j in 0..times {
                        new_values[i * times + j] = value.clone();
                    }
                }
                Matrix::new(
                    MatrixData::Dense(new_values),
                    times,
                    self.height,
                    self.ranges.clone(),
                    self.comment.clone(),
                )
            }
            MatrixData::COO(_) => {
                unimplemented!("Repeating columns for COO matrices is not implemented yet");
            }
        }
    }
    // create a new matrix using only the rows in the given range
    // this is used to extract the witness rows for the commitment
    pub fn extract_rows(&self, rows: &Range<usize>) -> Self {
        match &self.data() {
            MatrixData::Dense(values) => {
                let mut new_values = BigVec::new(self.width() * rows.len()).unwrap();
                let start = rows.start * self.width();
                let end = rows.end * self.width();
                for i in start..end {
                    new_values[i - start] = values[i].clone();
                }
                Matrix::new(
                    MatrixData::Dense(new_values),
                    self.width(),
                    rows.len(),
                    self.ranges().cloned(),
                    self.comment().to_string(),
                )
            }
            _ => panic!("Expected Dense matrix for extract_rows"),
        }
    }
    pub fn as_dense_vector(&self) -> &BigVec<T> {
        match &self.data {
            MatrixData::Dense(values) => values,
            MatrixData::COO(_) => panic!("Expected Dense matrix, found COO"),
        }
    }
    pub fn fill_values_as_indices(a: &mut Matrix<i64>, shift: i64, z: &Matrix<i64>) {
        match &mut a.data {
            MatrixData::Dense(_) => panic!("Cannot fill values as indices in Dense matrix"),
            MatrixData::COO(entries) => {
                let lookup = match z.data() {
                    MatrixData::Dense(values) => values,
                    MatrixData::COO(_) => panic!("Expected Dense matrix for z"),
                };
                for (_, _, value) in entries.iter_mut() {
                    let idx = *value + shift;
                    if idx < 0 || idx as usize >= lookup.len() {
                        panic!(
                            "Index out of bounds: {} for lookup of size {}",
                            idx,
                            lookup.len()
                        );
                    }
                }
            }
        }
    }
    pub fn stack_sparse_matrices(mats: Vec<&Matrix<T>>) -> Matrix<T> {
        // create new matrix where height is sum of heights of mats
        let height: usize = mats.iter().map(|m| m.height()).sum();
        let width = mats[0].width();
        // check that all widths are the same
        for mat in mats.iter() {
            if mat.width() != width {
                panic!("All matrices must have the same width to stack them");
            }
        }
        // number of entries in the new matrix is sum of entries in all mats
        let num_entries: usize = mats.iter().map(|m| m.num_entries()).sum();
        let mut new_data = BigVec::new(num_entries).unwrap();
        let mut offset = 0;
        let mut i = 0;
        for mat in mats.iter() {
            match &mat.data {
                MatrixData::COO(entries) => {
                    for (r, c, value) in entries.iter() {
                        new_data[i] = (*r + offset, *c, value.clone());
                        i += 1;
                    }
                }
                _ => {
                    panic!("Only COO matrices can be stacked");
                }
            }
            offset += mat.height();
        }
        // print resulting dimensions
        eprintln!("Stacked matrix: {} rows, {} columns", height, width);
        // create new matrix
        Matrix::new(
            MatrixData::COO(new_data),
            width,
            height,
            None,
            "Stacked sparse matrix".to_string(),
        )
    }
    pub fn stack_dense_matrices(mats: Vec<&Matrix<T>>) -> Matrix<T> {
        // create new matrix where height is sum of heights of mats
        let height: usize = mats.iter().map(|m| m.height()).sum();
        let width = mats[0].width();
        // check that all widths are the same
        for mat in mats.iter() {
            if mat.width() != width {
                panic!("All matrices must have the same width to stack them");
            }
        }
        // number of entries in the new matrix is sum of entries in all mats
        let num_entries: usize = mats.iter().map(|m| m.num_entries()).sum();
        let mut new_data = BigVec::new(num_entries).unwrap();
        let mut i = 0;
        for mat in mats.iter() {
            match &mat.data {
                MatrixData::Dense(entries) => {
                    for value in entries.iter() {
                        new_data[i] = value.clone();
                        i += 1;
                    }
                }
                _ => {
                    panic!("Only dense matrices can be stacked");
                }
            }
        }
        // print resulting dimensions
        eprintln!("Stacked matrix: {} rows, {} columns", height, width);
        // create new matrix
        Matrix::new(
            MatrixData::Dense(new_data),
            width,
            height,
            None,
            "Stacked dense matrix".to_string(),
        )
    }
    pub fn stack_dense_matrices_horizontally(mats: Vec<&Matrix<T>>) -> Matrix<T> {
        // all matrices must have the same height
        let height = mats[0].height();
        for mat in mats.iter() {
            if mat.height() != height {
                panic!("All matrices must have the same height to stack horizontally");
            }
        }
        // total width is the sum of widths
        let total_width: usize = mats.iter().map(|m| m.width()).sum();

        // allocate output buffer
        let num_entries = height * total_width;
        let mut new_data = BigVec::new(num_entries).unwrap();

        // fill row-by-row, concatenating each matrix's row horizontally
        for r in 0..height {
            let mut col_offset = 0;
            for mat in mats.iter() {
                let w = mat.width();
                match mat.data() {
                    MatrixData::Dense(values) => {
                        let src_base = r * w;
                        let dst_base = r * total_width + col_offset;
                        for c in 0..w {
                            new_data[dst_base + c] = values[src_base + c].clone();
                        }
                    }
                    _ => {
                        panic!("Only dense matrices can be stacked");
                    }
                }
                col_offset += w;
            }
        }

        eprintln!("Stacked matrix: {} rows, {} columns", height, total_width);

        Matrix::new(
            MatrixData::Dense(new_data),
            total_width,
            height,
            None,
            "Horizontally stacked dense matrix".to_string(),
        )
    }
    // get square-like parameters
    pub fn square_like_params(&self) -> (usize, usize) {
        let nwidth = self.width.next_power_of_two();
        let nheight = self.height.next_power_of_two();
        let full_size = nwidth * nheight;
        let log2_size = full_size.ilog2();
        let height = 1 << ((log2_size / 2) - 2);
        let width = full_size / height;
        (width, height)
    }
    // turn an oblong matrix into one that is roughly square
    // sides are powers of 2
    pub fn to_square_like(&self) -> Self {
        let nwidth = self.width.next_power_of_two();
        let nheight = self.height.next_power_of_two();
        let full_size = nwidth * nheight;
        let mut padded = BigVec::new(full_size).unwrap();
        match &self.data {
            MatrixData::Dense(values) => {
                for i in 0..self.height {
                    for j in 0..self.width {
                        padded[i * nwidth + j] = values[i * self.width + j].clone();
                    }
                }
            }
            _ => panic!("Expected Dense matrix for to_square_like"),
        }
        // get log2 of full_size
        let (width, height) = self.square_like_params();
        Matrix::new(
            MatrixData::Dense(padded),
            width,
            height,
            self.ranges.clone(),
            "Square-like matrix".to_string(),
        )
    }
}

use std::collections::HashMap;
use std::hash::Hash;

impl<T: Clone + Default + PartialEq + Eq + Hash> Matrix<T> {
    // get the n most frequent values in the matrix and the percentage of the matrix that is them
    pub fn most_frq_values(&self, n: usize) -> Vec<(T, f64)> {
        let mut hist = HashMap::new();
        let num_elements;
        match &self.data() {
            MatrixData::Dense(values) => {
                num_elements = values.len();
                for val in values.iter() {
                    *hist.entry(val.clone()).or_insert(0) += 1;
                }
            }
            MatrixData::COO(entries) => {
                num_elements = entries.len();
                for (_, _, val) in entries.iter() {
                    *hist.entry(val.clone()).or_insert(0) += 1;
                }
            }
        };
        let mut hist_vec: Vec<_> = hist.into_iter().collect();
        hist_vec.sort_by(|a, b| b.1.cmp(&a.1)); // sort by count descending
        hist_vec.truncate(n); // keep only the top n
        // map to (value, percentage)
        hist_vec
            .into_iter()
            .map(|(val, count)| (val, count as f64 / num_elements as f64))
            .collect()
    }
}
