use std::fs::File;
use std::io::{self, Read, Write};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::mat::{Matrix, MatrixData};
use stream::bigvec::BigVec;

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
fn read_index<R: Read>(reader: &mut R, index_type: u8) -> io::Result<usize> {
    match index_type {
        0 => Ok(read_u32(reader)? as usize),
        1 => Ok(read_u64(reader)? as usize),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid indexing data type",
        )),
    }
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
fn read_f64<R: Read>(reader: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}
fn read_float<R: Read>(reader: &mut R, value_type: u8) -> io::Result<f64> {
    match value_type {
        0x00 => Ok(read_f32(reader)? as f64),
        0x01 => Ok(read_f64(reader)?),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid numeric data type",
        )),
    }
}

pub trait NumericType: Clone + Default + Sized + PartialEq {
    fn read_val<R: Read>(reader: &mut R, value_type: u8) -> io::Result<Self>;
    fn to_le_bytes(&self) -> Vec<u8>;
    fn is_zero(&self) -> bool;
}

impl NumericType for f64 {
    fn read_val<R: Read>(reader: &mut R, value_type: u8) -> io::Result<Self> {
        read_float(reader, value_type)
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        f64::to_le_bytes(*self).to_vec()
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl NumericType for u64 {
    fn read_val<R: Read>(reader: &mut R, value_type: u8) -> io::Result<Self> {
        match value_type {
            0x10 => Ok(read_u32(reader)? as u64),
            0x11 => Ok(read_u64(reader)?),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid numeric data type",
            )),
        }
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        u64::to_le_bytes(*self).to_vec()
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl<T: NumericType> Matrix<T> {
    pub fn from_file(filename: &Path) -> io::Result<(Option<u8>, Self)> {
        // open the file and extract the header
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 88];
        reader.read_exact(&mut header)?;

        // Parse the header
        let file_type = header[0];
        let index_type = header[1];
        let value_type = header[2];
        let den_bits = if header[3] > 0 {
            Some(header[3])
        } else {
            None // If den_bits is 0, we treat it as None
        };
        let comment = String::from_utf8_lossy(&header[4..64]).trim().to_string();
        let num_entries = usize::from_le_bytes(header[64..72].try_into().unwrap());
        let width = usize::from_le_bytes(header[72..80].try_into().unwrap());
        let height = usize::from_le_bytes(header[80..88].try_into().unwrap());

        // Read data entries
        if file_type == 0 {
            // Dense matrix
            let mut values = BigVec::new(width * height).unwrap();
            for i in 0..(width * height) {
                values[i] = T::read_val(&mut reader, value_type)?;
            }
            let data = MatrixData::Dense(values);
            Ok((den_bits, Matrix::new(data, width, height, None, comment)))
        } else if file_type == 1 {
            // COO matrix
            // TO DO, later directly construct into BigVec without Vec intermediary
            // when we are assured no zeros are present
            let mut values = Vec::new();
            for _ in 0..num_entries {
                let row = read_index(&mut reader, index_type)?;
                let col = read_index(&mut reader, index_type)?;
                let value = T::read_val(&mut reader, value_type)?;
                values.push((row, col, value));
            }
            let data = MatrixData::COO(BigVec::from_vec(values));
            Ok((den_bits, Matrix::new(data, width, height, None, comment)))
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid file type",
            ))
        }
    }
    pub fn to_file(&self, filename: &Path, den_bits: Option<u8>) -> io::Result<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(&file);

        // Write header
        let file_type = match &self.data() {
            MatrixData::Dense(_) => 0u8,
            MatrixData::COO(_) => 1u8,
        };
        let index_type = if self.width() <= u32::MAX as usize {
            0u8 // u32
        } else {
            1u8 // u64
        };
        let value_type = 1u8; // f64
        let den_bits = den_bits.unwrap_or(0u8); // Default to 0 if not provided
        let comment = self.comment().as_bytes();
        let comment_padded = {
            let mut padded = vec![b' '; 60];
            let len = comment.len().min(60);
            padded[..len].copy_from_slice(&comment[..len]);
            padded
        };

        writer.write_all(&[file_type, index_type, value_type, den_bits])?;
        writer.write_all(&comment_padded)?;

        let num_entries = match &self.data() {
            MatrixData::Dense(values) => values.len(),
            MatrixData::COO(entries) => entries.len(),
        };
        writer.write_all(&(num_entries as u64).to_le_bytes())?;
        writer.write_all(&(self.width() as u64).to_le_bytes())?;
        writer.write_all(&(self.height() as u64).to_le_bytes())?;

        match &self.data() {
            MatrixData::Dense(values) => {
                for value in values.iter() {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            MatrixData::COO(entries) => {
                for (row, col, value) in entries.iter() {
                    if index_type == 0 {
                        writer.write_all(&(*row as u32).to_le_bytes())?;
                        writer.write_all(&(*col as u32).to_le_bytes())?;
                    } else {
                        writer.write_all(&(*row as u64).to_le_bytes())?;
                        writer.write_all(&(*col as u64).to_le_bytes())?;
                    }
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
        // Return success
        Ok(())
    }
}
