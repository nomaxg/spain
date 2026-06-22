use ff::ops_128::M128;
use rand::Rng;
use rug::Complete;
use rug::{ops::Pow, Integer};
use serde::{Deserialize, Serialize};

use crate::bigrsa::{generate_rsa_group, RSAGroup};
use crate::prover::{ChunkedComm, FinalClaim, RoundClaim};
use crate::public::{PublicParams, MOD_BITS};
use crate::{mod_pow_u64, precompute_series_mod, sample_alpha};

#[derive(Clone, Debug, Default)]
pub struct VerifierState {
    pub car: Integer,
    pub comm: RSAGroup,
    pub chunked_comm: ChunkedComm,
    pub round: usize,
    pub y_claim: Option<M128>,
    pub eval_point: Vec<M128>,
    pub alpha_int: Integer,
    pub alpha_p: M128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundChallenge {
    pub alpha_int: Integer,
    pub alpha_p: M128,
}

impl VerifierState {
    pub fn new(public: &PublicParams) -> Self {
        // Generate the carmichael trapdoor for fast exponentiation
        let ((p_prime, q_prime), _) = generate_rsa_group(MOD_BITS);
        let pm1 = Integer::from(&p_prime - 1);
        let qm1 = Integer::from(&q_prime - 1);
        let car = Integer::lcm(pm1, &qm1);

        Self {
            car,
            comm: public.mont.one(),
            eval_point: vec![],
            chunked_comm: ChunkedComm::default(),
            round: 0,
            alpha_int: Integer::ZERO,
            alpha_p: public.small_mont.zero(),
            y_claim: None,
        }
    }

    // Verifier initiates a round, sampling claims for the prover
    pub fn start_round<R: Rng>(&mut self, public: &PublicParams, rng: &mut R) -> RoundChallenge {
        self.round += 1;
        self.sample_round_alpha(public, rng);
        RoundChallenge {
            alpha_int: self.alpha_int.clone(),
            alpha_p: self.alpha_p,
        }
    }

    // Validates prover claims for the current round and updates internal state
    pub fn verify_round(&mut self, round_claim: &RoundClaim, public: &PublicParams) {
        let verifier_in_head_round = self.round < public.num_verifier_in_head_rounds();

        // Derive or fetch commitment claims
        let (comm_fl, comm_fr) = if verifier_in_head_round {
            let (cl, cr) = self.get_derived_comms(public);
            (cl, cr)
        } else {
            (round_claim.comm_fl.unwrap(), round_claim.comm_fr.unwrap())
        };

        if !verifier_in_head_round {
            self.check_commitment_consistency(comm_fl, comm_fr, public);
        }
        // Verifier checks that y = y_l + X_2 * y_r
        self.check_y_claim(&round_claim.y_l, &round_claim.y_r, public);

        // Verifier folds commitments, updating internal state
        self.update_commitment_and_claim(
            round_claim.y_l,
            round_claim.y_r,
            comm_fl,
            comm_fr,
            public,
        );

        if self.round == public.total_rounds {
            let FinalClaim {
                final_constant,
                final_constant_int,
            } = round_claim.final_claim.as_ref().unwrap();
            self.final_check(final_constant, final_constant_int, public);
        }
    }

    // Set claim (set y claim and eval point)
    pub fn set_claim(&mut self, y_claim: M128, eval_point: Vec<M128>) {
        self.y_claim = Some(y_claim);
        self.eval_point = eval_point;
    }

    // Set commit for DARK eval protocol
    pub fn set_commit(&mut self, comm: ChunkedComm) {
        if comm.comms.len() == 1 {
            self.comm = comm.comms[0];
        }
        self.chunked_comm = comm;
    }

    // Samples round alpha
    pub fn sample_round_alpha<R: Rng>(&mut self, public: &PublicParams, rng: &mut R) {
        let (alpha_int, alpha_p) = sample_alpha(&public.small_mont, rng);
        self.alpha_int = alpha_int;
        self.alpha_p = alpha_p;
    }

    // Returns the precomputed constant commitments for verifier-in-head rounds, computing and caching them if not already done
    // Also caches the initial padding commitment
    pub fn compute_const_comms(&mut self, public: &mut PublicParams) {
        if !public.cached_const_comms.is_empty() {
            return;
        }
        let q = &public.q;
        let series = precompute_series_mod(q, &self.car, public.total_rounds);
        for i in 0..public.total_rounds {
            let c_i = mod_pow_u64(
                &public.uint_shift_base(),
                (i + 1).try_into().unwrap(),
                &self.car,
            );
            let num_vars = public.total_rounds - (i + 1);
            let s_k = (&series[num_vars] % &self.car).complete();
            let exp = (c_i * s_k) % &self.car;
            let comm = public.mont.fast_exp(&public.rsa_gen, &exp, &self.car);
            public.cached_const_comms.push(comm);
        }
        let pad_comm = self.chunk_padding_comm(public);
        public.initial_pad_comm = Some(pad_comm);
    }

    fn chunk_padding_comm(&self, public: &PublicParams) -> RSAGroup {
        let mut pow_q = Integer::from(1);
        let mut sum_q = Integer::from(0);
        for _ in 0..public.chunk_size() {
            sum_q += &pow_q;
            if sum_q >= self.car {
                sum_q -= &self.car;
            }
            pow_q *= &public.q;
            pow_q %= &self.car;
        }

        let shift_exp = (sum_q * public.int_shift_base()) % &self.car;
        public.mont.fast_exp(&public.rsa_gen, &shift_exp, &self.car)
    }

    pub fn combine_chunks(&self, chunked_comm: &ChunkedComm, public: &PublicParams) -> RSAGroup {
        let step = mod_pow_u64(&public.q, chunked_comm.chunk_size as u64, &self.car);
        let mut weight = Integer::from(1);
        let mut combined = public.mont.one();
        for comm in &chunked_comm.comms {
            public.mont.mul_assign(
                &mut combined,
                &public.mont.fast_exp(comm, &weight, &self.car),
            );
            weight = (weight * &step) % &self.car;
        }
        combined
    }

    // Fetch precomputed chunked for verifier-in-head rounds
    pub fn fetch_derived_chunks(&self, public: &PublicParams) -> (ChunkedComm, ChunkedComm) {
        let chunked_comm = &self.chunked_comm;
        let half = chunked_comm.comms.len() / 2;
        let fl = &chunked_comm.comms[0..half];
        let rhs = &chunked_comm.comms[half..];
        let fr = fl
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| {
                let mut res = *r;
                public.mont.div_assign(&mut res, l);
                res
            })
            .collect::<Vec<_>>();
        (
            ChunkedComm {
                comms: fl.to_vec(),
                chunk_size: chunked_comm.chunk_size,
            },
            ChunkedComm {
                comms: fr.to_vec(),
                chunk_size: chunked_comm.chunk_size,
            },
        )
    }

    // Compute the "chunked" fl and fr commitments for verifier-in-head rounds
    pub fn fetch_derived_comms(
        &self,
        fl: &ChunkedComm,
        fr: &ChunkedComm,
        public: &PublicParams,
    ) -> (RSAGroup, RSAGroup) {
        let const_comm = public.cached_const_comms[self.round - 1];
        let mut comm_fl = self.combine_chunks(fl, public);
        let mut comm_fr = self.combine_chunks(fr, public);
        public.mont.mul_assign(&mut comm_fl, &const_comm);
        public.mont.mul_assign(&mut comm_fr, &const_comm);
        (comm_fl, comm_fr)
    }

    // Fold chunks for verifier-in-head rounds
    pub fn fold_derived_chunks(
        &self,
        fl: &ChunkedComm,
        fr: &ChunkedComm,
        alpha_int: &Integer,
        public: &PublicParams,
    ) -> ChunkedComm {
        let mut folded_comms = Vec::with_capacity(fl.comms.len());
        for (cl, cr) in fl.comms.iter().zip(fr.comms.iter()) {
            let mut folded = *cl;
            let cr_alpha = public.mont.fast_exp(cr, alpha_int, &self.car);
            public.mont.mul_assign(&mut folded, &cr_alpha);
            folded_comms.push(folded);
        }
        ChunkedComm {
            comms: folded_comms,
            chunk_size: fl.chunk_size,
        }
    }

    // Get derived commitments from folding for verifier-in-head rounds
    pub fn get_derived_comms(&mut self, public: &PublicParams) -> (RSAGroup, RSAGroup) {
        let (fl, fr) = self.fetch_derived_chunks(public);
        let (cl, cr) = self.fetch_derived_comms(&fl, &fr, public);
        self.chunked_comm = self.fold_derived_chunks(&fl, &fr, &self.alpha_int, public);
        (cl, cr)
    }
    // Check the y claim for the current round, panics if claim is incorrect
    pub fn check_y_claim(&self, y_l: &M128, y_r: &M128, public: &PublicParams) {
        let mont = public.small_mont;
        let int_shift = public.uint_shift_base().pow(self.round as u32);
        let mont_shift = mont.from_bigint(int_shift.clone());
        let y_check = mont.add(
            mont.sub(*y_l, mont_shift),
            mont.mul(
                mont.sub(*y_r, mont_shift),
                self.eval_point[public.total_rounds - self.round],
            ),
        );
        assert_eq!(
            self.y_claim.unwrap(),
            y_check,
            "y does not match y_l + X_2 * y_r"
        );
    }

    // Verifier commitment consistency for non-head rounds, panics if claim is incorrect
    pub fn check_commitment_consistency(
        &self,
        comm_fl: RSAGroup,
        comm_fr: RSAGroup,
        public: &PublicParams,
    ) {
        let comm_const = &public.cached_const_comms[self.round - 1];
        let q_mu = mod_pow_u64(
            &public.q,
            1u64 << (public.total_rounds - self.round),
            &self.car,
        );
        let inv_const = public
            .mont
            .inverse(&mut comm_const.clone())
            .expect("inversion failed");
        let const_inv_sq = public
            .mont
            .fast_exp(&inv_const, &Integer::from(2u64), &self.car);
        let base_q = {
            let mut res = comm_fl;
            public.mont.mul_assign(&mut res, &comm_fr);
            public.mont.mul_assign(&mut res, &const_inv_sq);
            res
        };
        let powered = public.mont.fast_exp(&base_q, &q_mu, &self.car);
        let mut derived_comm = comm_fl;
        public.mont.mul_assign(&mut derived_comm, &powered);
        public.mont.mul_assign(&mut derived_comm, &inv_const);
        assert_eq!(
            derived_comm, self.comm,
            "Derived commitment does not match C"
        );
    }

    // Verifier updates the state of its commitment by folding the "left" and "right" commitments
    // and updates the state of its y claim for the next round
    pub fn update_commitment_and_claim(
        &mut self,
        y_l: M128,
        y_r: M128,
        comm_fl: RSAGroup,
        comm_fr: RSAGroup,
        public: &PublicParams,
    ) {
        // Fetch int/mont shift values that prevent negative coefficients
        let int_shift = public.uint_shift_base().pow(self.round as u32);
        let mont_shift = public.small_mont.from_bigint(int_shift.clone());

        // Calculate the new y claim
        self.y_claim = Some(
            public.small_mont.add(
                public.small_mont.sub(y_l, mont_shift),
                public
                    .small_mont
                    .mul(public.small_mont.sub(y_r, mont_shift), self.alpha_p),
            ),
        );

        // Calculate the new commitment
        let mut lhs = comm_fl;
        let comm_const = &public.cached_const_comms[self.round - 1];
        public.mont.div_assign(&mut lhs, comm_const);
        let mut rhs = comm_fr;
        public.mont.div_assign(&mut rhs, comm_const);
        let rhs = public.mont.fast_exp(&rhs, &self.alpha_int, &self.car);
        public.mont.mul_assign(&mut lhs, &rhs);
        self.comm = lhs;
    }

    // Final constant commitment + bounds check at the end of the protocol, panics if check fails
    pub fn final_check(
        &self,
        final_constant: &M128,
        final_constant_int: &Integer,
        public: &PublicParams,
    ) {
        let mut exp = final_constant_int.clone() % &self.car;
        let upper_bound = (public.small_mont.modulus() - 1)
            * Integer::from(2).pow((public.total_rounds as u32) * 128);

        // Ensure that final coefficient is positive
        if exp.is_negative() {
            exp += &self.car;
        }
        let derived_comm = public.mont.fast_exp(&public.rsa_gen, &exp, &self.car);

        // Check that the final constant polynomial is within bounds
        if final_constant_int.clone().abs() > upper_bound {
            eprintln!("WARNING: Final constant polynomial is out of bounds. This is expected for Otti-FE, as we do not use Otti's field.");
        }

        // Check that the final constant polynomial is the expected evaluation
        assert_eq!(
            self.y_claim.unwrap(),
            *final_constant,
            "y does not match final constant polynomial"
        );

        // Check the commitment consistency
        assert_eq!(
            self.comm, derived_comm,
            "comm does not match derived commitment from final polynomial"
        );
    }
}
