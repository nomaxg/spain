pub mod inner;
pub mod inner2;
pub mod ops;
pub mod ops_128;
pub mod outer;
pub mod outer_eq;
pub mod poly;
pub mod prime;
pub mod prime_128;

// Aliases so downstream crates can switch Montgomery backends easily
// TODO: Design/implement a trait for Mont contexts so that we can swap easily
pub mod field {
    use i256::I256;
    use rug::Integer;

    pub type FieldElem = crate::ops_128::M128;
    pub type FieldMont = crate::ops_128::Mont;

    pub fn int_to_mont(val: &Integer, mont: &FieldMont) -> FieldElem {
        mont.from_bigint(val.clone())
    }

    pub fn i64_to_mont(val: &i64, mont: &FieldMont) -> FieldElem {
        mont.from_i128(*val as i128)
    }

    pub fn i128_to_mont(val: &i128, mont: &FieldMont) -> FieldElem {
        mont.from_i128(*val)
    }

    pub fn i256_to_mont(val: &I256, mont: &FieldMont) -> FieldElem {
        mont.from_i256(*val)
    }

    pub fn usize_to_mont(value: usize, mont: &FieldMont) -> FieldElem {
        mont.to_mont(value as u128)
    }
}

pub use field::{
    i128_to_mont, i256_to_mont, i64_to_mont, int_to_mont, usize_to_mont, FieldElem, FieldMont,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_prime_test() {
        // init rng
        let mut rng = rand::rng();
        // repeat for 100 iterations
        for _ in 0..100 {
            let p = prime::rand_prime(&mut rng);
            println!("Random prime: {}", p);
        }
    }

    #[test]
    fn mod_types_test() {
        // init rng
        let mut rng = rand::rng();
        // repeat for 100 iterations
        for _ in 0..100 {
            // get random prime modulus
            let p = prime::rand_prime(&mut rng);
            // get random element in the field
            let x = prime::rand_elem(p, &mut rng);
            // square using naive method and double
            let mut xr = ops::mul_mod(x, x, p);
            xr = ops::add_mod(xr, xr, p);
            // create a new montgomery context, convert, square, double and reduce
            let mont = ops::Mont::new(p);
            let mut xm = mont.to_mont(x);
            xm = mont.mul(xm, xm);
            xm = mont.add(xm, xm);
            let xn = mont.to_normal(xm);
            // check that the results are equal
            assert_eq!(xr, xn);
        }
    }
    #[test]
    fn mod_inv_test() {
        // init rng
        let mut rng = rand::rng();
        // repeat for 100 iterations
        for _ in 0..100 {
            // get random prime modulus
            let p = prime::rand_prime(&mut rng);
            // get random element in the field
            let x = prime::rand_elem(p, &mut rng);
            // create a new montgomery context, convert, invert, mul, and reduce
            let mont = ops::Mont::new(p);
            let xm = mont.to_mont(x);
            let xminv = mont.inv(xm);
            let mone = mont.mul(xm, xminv);
            let one = mont.to_normal(mone);
            // check that the result is 1
            assert_eq!(one, 1);
        }
    }
    #[test]
    fn mod_inv_test_128() {
        // init rng
        let mut rng = rand::rng();
        // repeat for 100 iterations
        for _ in 0..100 {
            // get random prime modulus
            let p = prime_128::rand_prime(&mut rng);
            // get random element in the field
            let x = prime_128::rand_elem(p, &mut rng);
            // create a new montgomery context, convert, invert, mul, and reduce
            let mont = ops_128::Mont::new(p);
            let xm = mont.to_mont(x);
            let xminv = mont.inv(xm);
            let mone = mont.mul(xm, xminv);
            let one = mont.to_normal(mone);
            // check that the result is 1
            assert_eq!(one, 1);
        }
    }
}
