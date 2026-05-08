HOST ?= 127.0.0.1
PORT ?= 9000
MODEL ?= layernorm_32x768
BATCH_SIZE ?= 1
SCALE_FACTOR_BITS ?= 34
MAX_EPSILON ?= 0.1
CHUNK_SIZE ?= 16
Q_BITS ?= 30000
PRECISION ?= 128
DATA_DIR ?= ../circuit/export
USE_SAME_INPUT ?= true
PHASE_BREAKDOWN ?= false

PROVER_ARGS = \
	--model $(MODEL) \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--data-dir $(DATA_DIR) \
	--use-same-input $(USE_SAME_INPUT) \
	--phase-breakdown $(PHASE_BREAKDOWN)

VERIFIER_ARGS = \
	--model $(MODEL) \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--max-epsilon $(MAX_EPSILON) \
	--chunk-size $(CHUNK_SIZE) \
	--q-bits $(Q_BITS) \
	--precision $(PRECISION) \
	--data-dir $(DATA_DIR) \
	--use-same-input $(USE_SAME_INPUT) \
	--phase-breakdown $(PHASE_BREAKDOWN)

build_bins:
	cargo build --release --bin prover --bin verifier

run_prover: build_bins
	ncat -l $(PORT) --exec "./measure.sh ../target/release/prover $(PROVER_ARGS)"

run_verifier: build_bins
	ncat $(HOST) $(PORT) --exec "../target/release/verifier $(VERIFIER_ARGS)"
