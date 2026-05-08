/* Parser for our bespoke file format.
 * The file format is as follows:
 * [1 byte: file type]
 * 0x00 - dense matrix
 * 0x01 - COO matrix (row, column, value)
 * [1 byte: indexing data type]
 * 0x00 - u32
 * 0x01 - u64
 * [1 byte: numeric data type]
 * 0x00 - f32
 * 0x01 - f64
 * 0x10 - u32
 * 0x11 - u64
 * [1 byte: log_2(denominator) if numeric data type is an integer, otherwise ignored]
 * [60 bytes: ASCII comment, padded to 60 bytes with spaces]
 * [8 bytes: number of data entries]
 * [8 bytes: width of matrix]
 * [8 bytes: height of matrix]
 * [entries]
 * The data format depends on the file type:
 * * For dense vector: width * height entries of the numeric data type.
 * * * row-major order.
 * * For COO matrix: list of:
 * * * (indexing data type row, indexing data type col, numeric data type) tuples.
 * * * presumed but not required to be sorted by row with ties broken by column.
 */
pub mod fs;
pub mod generalized;
pub mod mat;
pub mod matf64;
pub mod mati64;
pub mod matmont;
