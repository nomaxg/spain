HOST ?= 127.0.0.1
PORT ?= 9000
BATCH_SIZE ?= 1
SCALE_FACTOR_BITS ?= 70
MAX_EPSILON ?= 0.00000000005
ZKLP_MAX_EPSILON ?= 0.001
CHUNK_SIZE ?= 16
Q_BITS ?= 30000
PRECISION ?= 128
STEPS ?= 10
GRID_SIZE ?= 16
PHASE_BREAKDOWN ?= true

PROVER_ARGS = \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--phase-breakdown $(PHASE_BREAKDOWN) \
	--steps $(STEPS) \
	--grid-size $(GRID_SIZE)

VERIFIER_ARGS = \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--max-epsilon $(MAX_EPSILON) \
	--chunk-size $(CHUNK_SIZE) \
	--q-bits $(Q_BITS) \
	--precision $(PRECISION) \
	--phase-breakdown $(PHASE_BREAKDOWN) \
	--steps $(STEPS) \
	--grid-size $(GRID_SIZE)

build_bins:
	cargo build --release --bin physics-prover --bin physics-verifier
	
build_zklp_bins:
	cargo build --release --bin zklp-prover --bin zklp-verifier

run_prover: build_bins
	ncat -l $(PORT) --exec "../target/release/physics-prover $(PROVER_ARGS)"

run_verifier: build_bins
	ncat $(HOST) $(PORT) --exec "../target/release/physics-verifier $(VERIFIER_ARGS)"


PROVER_ARGS_ZKLP = \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--phase-breakdown $(PHASE_BREAKDOWN) \

VERIFIER_ARGS_ZKLP = \
	--batch-size $(BATCH_SIZE) \
	--scale-factor-bits $(SCALE_FACTOR_BITS) \
	--max-epsilon $(ZKLP_MAX_EPSILON) \
	--chunk-size $(CHUNK_SIZE) \
	--q-bits $(Q_BITS) \
	--precision $(PRECISION) \
	--phase-breakdown $(PHASE_BREAKDOWN) \

run_prover_zklp: build_zklp_bins
	ncat -l $(PORT) --exec "../target/release/zklp-prover $(PROVER_ARGS_ZKLP)"

run_verifier_zklp: build_zklp_bins
	ncat $(HOST) $(PORT) --exec "../target/release/zklp-verifier $(VERIFIER_ARGS_ZKLP)"
