HOST ?= 127.0.0.1
PORT ?= 9000
BATCH_SIZE ?= 1
SCALE_FACTOR_BITS ?= 70
MAX_EPSILON ?= 0.1
CHUNK_SIZE ?= 16
Q_BITS ?= 30000
PRECISION ?= 128
STEPS ?= 20

PROVER_ARGS = \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--steps $(STEPS)

VERIFIER_ARGS = \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--max-epsilon $(MAX_EPSILON) \
	--chunk-size $(CHUNK_SIZE) \
	--q-bits $(Q_BITS) \
	--precision $(PRECISION) \
	--steps $(STEPS)

build_bins:
	cargo build --release --bin physics-prover --bin physics-verifier

run_prover: build_bins
	ncat -l $(PORT) --exec "../target/release/physics-prover $(PROVER_ARGS)"

run_verifier: build_bins
	ncat $(HOST) $(PORT) --exec "../target/release/physics-verifier $(VERIFIER_ARGS)"
