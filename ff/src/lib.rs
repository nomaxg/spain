pub mod inner;
pub mod inner2;
pub mod ops;
pub mod ops_128;
pub mod outer;
pub mod poly;
pub mod prime;
pub mod prime_128;

// Aliases so downstream crates can switch Montgomery backends easily
// TODO: Design/implement a trait for Mont contexts so that we can swap easily
pub mod field {
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

    pub fn usize_to_mont(value: usize, mont: &FieldMont) -> FieldElem {
        mont.to_mont(value as u128)
    }
}

pub use field::{FieldElem, FieldMont, i64_to_mont, i128_to_mont, int_to_mont, usize_to_mont};

#[cfg(test)]
mod tests {
    use super::{field::usize_to_mont, *};
    use stream::bigvec::BigVec;

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

    #[test]
    fn test_mle_split_mont() {
        let mont = FieldMont::new(101);
        let n = 3;
        let size = 1 << n;
        let buf: Vec<_> = (0..size).map(|i| usize_to_mont(i, &mont)).collect();
        let bind_pts: Vec<_> = (0..n).map(|i| usize_to_mont(i * 7 + 3, &mont)).collect();
        let mut full = poly::mont::MLE::from_buffer(BigVec::from_vec(buf.clone()), n);
        // bind the original MLE
        for &x in &bind_pts {
            full.bind(x, &mont);
        }
        let expected = full.evals[0];

        let (mut f_l, mut f_r) =
            poly::mont::MLE::from_buffer(BigVec::from_vec(buf), n).split(&mont);

        // bind the split MLEs
        for &x in &bind_pts[1..] {
            f_l.bind(x, &mont);
            f_r.bind(x, &mont);
        }
        let x0 = bind_pts[0];
        let reconstructed = mont.add(f_l.evals[0], mont.mul(x0, f_r.evals[0]));

        assert_eq!(expected, reconstructed,);
    }
}
