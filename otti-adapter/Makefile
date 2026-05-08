HOST ?= 127.0.0.1
PORT ?= 9000
MPS_PATH ?= ./datasets/sc105.mps
BATCH_SIZE ?= 1
SCALE_FACTOR_BITS ?= 50
MAX_EPSILON ?= 0.1
CHUNK_SIZE ?= 4
Q_BITS ?= 30000
PRECISION ?= 128

PROVER_ARGS = \
	--mps-path $(MPS_PATH) \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS)

VERIFIER_ARGS = \
	--mps-path $(MPS_PATH) \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--max-epsilon $(MAX_EPSILON) \
	--chunk-size $(CHUNK_SIZE) \
	--q-bits $(Q_BITS) \
	--precision $(PRECISION)

build_bins:
	cargo build --release --bin otti-prover --bin otti-verifier

run_prover: build_bins
	ncat -l $(PORT) --exec "../target/release/otti-prover $(PROVER_ARGS)"

run_verifier: build_bins
	ncat $(HOST) $(PORT) --exec "../target/release/otti-verifier $(VERIFIER_ARGS)"
