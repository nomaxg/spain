use ff::ops_128::M128;
use ff::poly::{int::MLE as IntMLE, mont::MLE};
use rug::Complete;
use rug::{ops::Pow, Integer};
use serde::{Deserialize, Serialize};

use crate::bigrsa::{pippenger_exp, RSAGroup};
use crate::public::PublicParams;
use crate::verifier::RoundChallenge;

// Represents a chunked commitment to a polynomial, where each commitment corresponds to a chunk of
// evaluations of an MLE
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ChunkedComm {
    pub comms: Vec<RSAGroup>,
    pub chunk_size: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ProverState {
    pub int_poly: Option<IntMLE>,
    pub mont_poly: Option<MLE>,
    pub eval_point: Vec<M128>,
    pub round: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundClaim {
    pub y_l: M128,
    pub y_r: M128,
    pub comm_fl: Option<RSAGroup>,
    pub comm_fr: Option<RSAGroup>,
    pub final_claim: Option<FinalClaim>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalClaim {
    pub final_constant: M128,
    pub final_constant_int: Integer,
}

// Commit to the int_poly in chunks for "verifier in the head" folding
// The number of chunks is determined by PublicParams
pub fn commit_chunked_pippenger(poly: IntMLE, public: &PublicParams) -> ChunkedComm {
    let len = poly.evals.len();
    let num_chunks = public.num_chunks;
    assert!(
        num_chunks.is_power_of_two(),
        "number of chunks must be a power of two"
    );
    assert!(
            len > num_chunks,
            "there cannot be more chunks than MLE evaluations, got {num_chunks} chunks for {len} evaluations"
        );

    let chunk_size = len / num_chunks;

    // Ensure we have enough precomputed bases for a single chunk:
    assert!(
            public.pippenger_bases.len() >= chunk_size,
            "pippenger_bases not precomputed for chunk_size={chunk_size}; \
             call build_pippenger_bases(log2(chunk_size)) or larger before commit_chunked_pippenger"
        );

    let mut comms = Vec::with_capacity(num_chunks);
    let shift = public.int_shift_base();
    let pad_comm = *public
        .initial_pad_comm
        .as_ref()
        .expect("initial_pad_comm not initialized; call verifier_compute_const_comms first");

    for chunk_start in (0..len).step_by(chunk_size) {
        let evals_chunk = &poly.evals[chunk_start..chunk_start + chunk_size];
        let padded_chunk: Vec<Integer> = evals_chunk
            .iter()
            .map(|v| {
                let res = (v + &shift).complete();
                assert!(
                    !res.is_negative(),
                    "DARK commit does not support negative coefficients after padding",
                );
                res
            })
            .collect();
        let c = pippenger_exp(
            &public.mont,
            &public.pippenger_bases[..chunk_size],
            &padded_chunk,
        );
        let mut unpadded = c;
        public.mont.div_assign(&mut unpadded, &pad_comm);
        comms.push(unpadded);
    }

    ChunkedComm { comms, chunk_size }
}

// Standard (non-chunked) commitment to int_poly
pub fn commit(poly: IntMLE, public: &PublicParams) -> RSAGroup {
    let len = poly.evals.len();
    assert!(
            public.pippenger_bases.len() >= len,
            "pippenger_bases not precomputed for enough terms; call build_pippenger_bases(log2({len})) first"
        );
    pippenger_exp(&public.mont, &public.pippenger_bases[..len], &poly.evals)
}

impl ProverState {
    // Commits to an int poly, and caches the int poly for later protocol case
    pub fn commit(&mut self, poly: IntMLE, public: &PublicParams) -> ChunkedComm {
        self.int_poly = Some(poly.clone());
        commit_chunked_pippenger(poly, public)
    }

    // Generate y claim, caching the eval point
    pub fn gen_y_claim(&mut self, eval_point: Vec<M128>, public: &PublicParams) -> M128 {
        self.poly_reduce(public);
        self.eval_point = eval_point.clone();
        let y = self
            .mont_poly
            .as_ref()
            .unwrap()
            .clone()
            .eval(&eval_point, &public.small_mont);
        y
    }

    // Returns the final constant int/mont claims
    pub fn final_claim(&self) -> FinalClaim {
        let final_constant = self.mont_poly.as_ref().unwrap().evals[0];
        let final_constant_int = self.int_poly.as_ref().unwrap().evals[0].clone();

        FinalClaim {
            final_constant,
            final_constant_int,
        }
    }

    // Respond to verifier challenges
    pub fn respond_to_challenge(
        &mut self,
        challenge: &RoundChallenge,
        public: &PublicParams,
    ) -> RoundClaim {
        self.round += 1;
        // Split the poly
        let (fl, fr, fl_int, fr_int) = self.poly_split(public);

        // Compute new comms
        let (comm_fl, comm_fr) = if self.round < public.num_verifier_in_head_rounds() {
            (None, None)
        } else {
            let (cl, cr) = self.poly_split_comm(&fl_int, &fr_int, public);
            (Some(cl), Some(cr))
        };
        let (y_l, y_r) = self.poly_split_eval(&fl, &fr, public);

        // Fold polys based on challenges
        self.update_polys(
            fl,
            fr,
            fl_int,
            fr_int,
            challenge.alpha_p,
            &challenge.alpha_int,
            public,
        );

        // Return round claim
        let final_claim = if self.round == public.total_rounds {
            Some(self.final_claim())
        } else {
            None
        };

        RoundClaim {
            y_l,
            y_r,
            comm_fl,
            comm_fr,
            final_claim,
        }
    }

    // Compute new evaluations by folding the "left" and "right" splits of the int_poly/mont_poly with the verifier's challenge alpha
    #[allow(clippy::too_many_arguments)]
    pub fn update_polys(
        &mut self,
        fl: MLE,
        fr: MLE,
        fl_int: IntMLE,
        fr_int: IntMLE,
        alpha_p: M128,
        alpha_int: &Integer,
        public: &PublicParams,
    ) {
        let mont = public.small_mont;
        let int_shift = public.uint_shift_base().pow(self.round as u32);
        let mont_shift = mont.from_bigint(int_shift.clone());
        self.mont_poly = Some(fl.fold(&mont, &fr, mont_shift, alpha_p));
        self.int_poly = Some(fl_int.fold(&fr_int, &int_shift, alpha_int));
    }

    // Poly split phase of the DARK protocol
    // Returns (fl, fr, fl_int, fr_int)
    pub fn poly_split(&self, public: &PublicParams) -> (MLE, MLE, IntMLE, IntMLE) {
        // Shift to prevent negative coefficients
        let mont = public.small_mont;
        let int_shift = public.uint_shift_base().pow(self.round as u32);
        let mont_shift = mont.from_bigint(int_shift.clone());
        // Prover computes fl and fr such that f = fl + X_2 * fr
        let (fl, fr) = self
            .mont_poly
            .as_ref()
            .unwrap()
            .split_msb(&mont, Some(mont_shift));
        let (fl_int, fr_int) = self.int_poly.as_ref().unwrap().split_msb(Some(&int_shift));
        (fl, fr, fl_int, fr_int)
    }

    // Returns the commitments to the "left" and "right" splits of an integer polynomial
    pub fn poly_split_comm(
        &self,
        left: &IntMLE,
        right: &IntMLE,
        public: &PublicParams,
    ) -> (RSAGroup, RSAGroup) {
        (commit(left.clone(), public), commit(right.clone(), public))
    }

    // Returns the evals of the "left" and "right" splits of an integer polynomial
    pub fn poly_split_eval(&self, left: &MLE, right: &MLE, public: &PublicParams) -> (M128, M128) {
        let mont = public.small_mont;
        (
            left.clone()
                .eval(&self.eval_point[0..left.num_vars()], &mont),
            right
                .clone()
                .eval(&self.eval_point[0..right.num_vars()], &mont),
        )
    }

    // Reduces the stored int poly to montgomery form
    pub fn poly_reduce(&mut self, public: &PublicParams) {
        self.mont_poly = Some(
            self.int_poly
                .as_ref()
                .unwrap()
                .reduce_to_mont(&public.small_mont),
        );
    }

    // sets the eval point for a protocol run
    pub fn set_eval_point(&mut self, eval_point: Vec<M128>) {
        self.eval_point = eval_point;
    }

    // sets the poly for protocol run
    pub fn set_poly(&mut self, poly: IntMLE) {
        self.int_poly = Some(poly);
    }
}
