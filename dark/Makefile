build_bins:
	cargo build --release --bin actor

run_prover: build_bins
	ncat -l 9000 --exec "../target/release/actor --role prover"

run_verifier: build_bins
	ncat 127.0.0.1 9000 --exec "../target/release/actor  --role verifier"

