use ff::{FieldElem, FieldMont, poly::mont::lagrange_interpolate, prime_128::rand_elem};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct VerifierState {
    num_vars: usize,
    degree: usize,
    challenges: Vec<FieldElem>,
    expected: FieldElem,
    mont: FieldMont,
}

impl VerifierState {
    pub fn new(num_vars: usize, degree: usize, expected: FieldElem, mont: FieldMont) -> Self {
        Self {
            num_vars,
            degree,
            challenges: Vec::new(),
            expected,
            mont,
        }
    }

    pub fn verify_round<R: Rng>(
        &mut self,
        p: &mut Vec<FieldElem>,
        rng: &mut R,
    ) -> Result<FieldElem, String> {
        // check if beyond last round
        if self.challenges.len() == self.num_vars {
            return Err("No more rounds to verify".to_string());
        }
        // check that p has the correct length
        if p.len() != self.degree {
            return Err(format!(
                "Expected polynomial of degree {}, got {}",
                self.degree,
                p.len()
            ));
        }
        // add self.expected - p[0] to the beginning of p
        p.insert(0, self.mont.sub(self.expected, p[0]));
        // generate a new challenge and add it to the list
        let r = self.mont.to_mont(rand_elem(self.mont.modulus(), rng));
        self.challenges.push(r);
        // interpolate to get the expected value
        self.expected = lagrange_interpolate(p, r, &self.mont);
        // return the challenge
        Ok(r)
    }
    pub fn final_verify(&self, c: FieldElem) -> Result<(), String> {
        // check that we are at the last round
        if self.challenges.len() != self.num_vars {
            panic!("Not at the last round, cannot verify final value");
        }
        // check that the final claim matches the expected value
        if c != self.expected {
            return Err(format!(
                "Final claim does not match expected value: {:?} != {:?}",
                c, self.expected
            ));
        }
        Ok(())
    }
    pub fn challenges(&self) -> &Vec<FieldElem> {
        &self.challenges
    }

    pub fn mont(&self) -> FieldMont {
        self.mont
    }
}
