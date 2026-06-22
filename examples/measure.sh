#! /bin/bash
batch_sizes="1 1024"
sleep_time="10s"
cargo build --release --bins
for batch_size in $batch_sizes; do
	echo $batch_size
	mkdir -p eval/${batch_size}
	for i in {1..5}; do
		if [[ $1 == "prover" ]]; then
			make run_prover_zklp BATCH_SIZE=$batch_size 2> "eval/${batch_size}/prover_${i}.txt"
		elif [[ $1 == "verifier" ]]; then
			make run_verifier_zklp BATCH_SIZE=$batch_size 2> "eval/${batch_size}/verifier_${i}.txt"
			sleep $sleep_time
		else
			exit 1
		fi
	done
done
