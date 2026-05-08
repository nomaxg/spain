pub mod bigvec;

#[cfg(test)]
mod tests {
    use std::fs::{File, remove_file};
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};

    use crate::bigvec::BigVec;

    fn data_file(filename: &str) -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir).join("data").join(filename)
    }

    // create vec of 1024*1024 elements to trigger mmap
    #[test]
    fn test_bigvec_mmap() {
        let len = 1024 * 1024;
        let mut bigvec = BigVec::<f64>::new(len).expect("Failed to create BigVec");
        // check length
        assert_eq!(bigvec.len(), len);
        // set and get values
        for i in 0..len {
            bigvec[i] = i as f64;
        }
        for i in 0..len {
            assert_eq!(bigvec[i], i as f64);
        }
    }

    // write f64 elements to a file, create BigVec from file directly
    #[test]
    fn test_bigvec_from_file() {
        // repeat twice with different file sizes
        for len in [1024, 1024 * 1024, 32 * 1024 * 1024] {
            let filename = data_file("bigvec_test.bin");
            let file = File::create(&filename).expect("Failed to create file");
            let mut writer = BufWriter::new(&file);
            for i in 0..len {
                writer
                    .write(&(i as f64).to_le_bytes())
                    .expect("Failed to write to file");
            }
            writer.flush().expect("Failed to flush writer");
            let bigvec =
                BigVec::<f64>::from_file(&filename).expect("Failed to create BigVec from file");
            // check length
            assert_eq!(bigvec.len(), len);
            // check values
            for i in 0..len {
                assert_eq!(bigvec[i], i as f64);
            }
            // clean up
            remove_file(&filename).expect("Failed to remove test file");
        }
    }
}
