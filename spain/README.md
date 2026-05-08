
To run over network (remote prover, local verifier):
```
// Run prover on cluster
ssh access.cims.nyu.edu
cd approx_sum_check/code/spain
make DATA_DIR=${wherever models are} run_prover

// Run verifier locally
cd approx_sum_check/code/spain
ssh -L 9000:access.nyu.edu:9000 access.nyu.edu
make run_verifier
```
