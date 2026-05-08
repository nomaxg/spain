// dynamic switching between slice and mmap backed big array
// optimized for sequential access
// initialized by size
use std::{
    fs::OpenOptions,
    io::{BufReader, Read},
    mem::size_of,
    ops::{Index, IndexMut},
    path::Path,
};

use memmap2::{Advice::Sequential, MmapMut, MmapOptions};

const THRESHOLD: usize = 1024 * 1024 * 1024; // 1 GiB

#[derive(Debug)]
enum Buffer<T> {
    VecBuffer(Vec<T>),
    MmapBuffer { mmap: MmapMut, len: usize },
}

#[derive(Debug)]
pub struct BigVec<T> {
    inner: Buffer<T>,
}

#[allow(clippy::len_without_is_empty)]
impl<T: Default + Clone> BigVec<T> {
    pub fn new(len: usize) -> Result<Self, String> {
        // Validate size
        if len == 0 {
            return Err("Len must be greater than zero".to_string());
        } else if len > usize::MAX / size_of::<T>() {
            return Err("Len is too large".to_string());
        }
        // get real size in bytes
        let num_bytes = len * size_of::<T>();
        if num_bytes < THRESHOLD {
            Ok(Self {
                inner: Buffer::VecBuffer(vec![T::default(); len]),
            })
        } else {
            let mmap = MmapOptions::new()
                .len(num_bytes)
                .map_anon()
                .map_err(|e| e.to_string())?;
            Ok(Self {
                inner: Buffer::MmapBuffer { mmap, len },
            })
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        // open file and check size
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        let num_bytes = file.metadata().map_err(|e| e.to_string())?.len() as usize;
        if !num_bytes.is_multiple_of(size_of::<T>()) {
            return Err("File size is not a multiple of element size".to_string());
        }
        let len = num_bytes / size_of::<T>();
        // if small enough, read into a Vec
        // otherwise, use mmap
        if num_bytes < THRESHOLD {
            let mut vec = vec![T::default(); len];
            let mut reader = BufReader::new(file);
            reader
                .read_exact(unsafe {
                    std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, num_bytes)
                })
                .map_err(|e| e.to_string())?;
            Ok(Self {
                inner: Buffer::VecBuffer(vec),
            })
        } else {
            let mmap = unsafe {
                MmapOptions::new()
                    .map_mut(&file)
                    .map_err(|e| e.to_string())?
            };
            Ok(Self {
                inner: Buffer::MmapBuffer { mmap, len },
            })
        }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        if vec.len() * size_of::<T>() < THRESHOLD {
            Self {
                inner: Buffer::VecBuffer(vec),
            }
        } else {
            let len = vec.len();
            let mmap = MmapOptions::new()
                .len(len * size_of::<T>())
                .map_anon()
                .expect("Failed to create mmap");
            let mut bigvec = Self {
                inner: Buffer::MmapBuffer { mmap, len },
            };
            for (i, value) in vec.into_iter().enumerate() {
                bigvec[i] = value;
            }
            bigvec
        }
    }

    // iterate
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match &self.inner {
            Buffer::VecBuffer(vec) => vec.iter(),
            Buffer::MmapBuffer { mmap, len } => unsafe {
                std::slice::from_raw_parts(mmap.as_ptr() as *const T, *len).iter()
            },
        }
    }

    // iterate mutable
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        match &mut self.inner {
            Buffer::VecBuffer(vec) => vec.iter_mut(),
            Buffer::MmapBuffer { mmap, len } => unsafe {
                std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut T, *len).iter_mut()
            },
        }
    }

    pub fn advise_seq(&mut self) {
        match &self.inner {
            Buffer::VecBuffer(_) => {}
            Buffer::MmapBuffer { mmap, len: _ } => mmap
                .advise(Sequential)
                .expect("Error, cannot set sequential access advise flag"),
        }
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            Buffer::VecBuffer(vec) => vec.len(),
            Buffer::MmapBuffer { len, .. } => *len,
        }
    }
}

// clone impl
impl<T: Clone> Clone for BigVec<T> {
    fn clone(&self) -> Self {
        match &self.inner {
            Buffer::VecBuffer(vec) => Self {
                inner: Buffer::VecBuffer(vec.clone()),
            },
            Buffer::MmapBuffer { mmap, len } => {
                let mut new_mmap = MmapOptions::new()
                    .len(*len * size_of::<T>())
                    .map_anon()
                    .expect("Failed to create mmap");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mmap.as_ptr(),
                        new_mmap.as_mut_ptr(),
                        *len * size_of::<T>(),
                    );
                }
                Self {
                    inner: Buffer::MmapBuffer {
                        mmap: new_mmap,
                        len: *len,
                    },
                }
            }
        }
    }
}

// index impl
impl<T> Index<usize> for BigVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.inner {
            Buffer::VecBuffer(vec) => &vec[index],
            Buffer::MmapBuffer { mmap, len } => {
                if index < *len {
                    unsafe { &*(mmap.as_ptr().add(index * size_of::<T>()) as *const T) }
                } else {
                    panic!("Index out of bounds for MmapBuffer");
                }
            }
        }
    }
}

// index mut impl
impl<T> IndexMut<usize> for BigVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.inner {
            Buffer::VecBuffer(vec) => &mut vec[index],
            Buffer::MmapBuffer { mmap, len } => {
                if index < *len {
                    unsafe { &mut *(mmap.as_mut_ptr().add(index * size_of::<T>()) as *mut T) }
                } else {
                    panic!("Index out of bounds for MmapBuffer");
                }
            }
        }
    }
}

// to apply ==
impl<T: Default + Clone + PartialEq> PartialEq for BigVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
