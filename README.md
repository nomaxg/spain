# Spain: Succinct proofs for numerical computations
Note to artifact evaluators: This README is with reference to the submitted version of the paper, as opposed to the camera ready (we will update the README after the camera ready is finalized).

Note to everyone else: this is a README for a code artifact that accompanies a research paper. The research paper itself is under preparation and will be available by the beginning of June; for a preliminary version, please write to the authors. Along those lines, please note the following Warning.

Warning! Spain is a research project. This code has not been audited. Do not use Spain in production environments or anywhere else that security is necessary.

This repository includes instructions for reproducing the results in the submitted version of the paper, figure by figure. 

# Getting started

## Requirements
- Ensure that you have [Docker](https://docs.docker.com/get-started/get-docker/) installed.
- Reproducing basic experimental results requires >= 18 GB RAM. 
  - Running Spain on GPT-2 _seq_= 32 requires >= 32 GB RAM.
  - 18 GB RAM is sufficient for GPT-2 _seq_ = 2. 

## Docker container
Build the docker container:
```
./build-docker.sh
```
Run the docker container:
```
./run-docker.sh
```
Build Spain: 
```
cargo build --release --bins
```
We provide a single Python script, ```run.py``` to reproduce experimental results. To quickly ensure that your setup is working properly, run:
```
python run.py --test
```
This will run Spain on a small circuit. If the ```--test``` command succeeded, you may safely move on. 

# Detailed instructions

## Building ONNX circuits (25-45 minutes)
The circuits for the ONNX computations (LayerNorm, Softmax, GPT-2, etc.) are larger and must be built and exported before use (this ensures that re-runs involving these models do not take too much time). 

All ONNX files are included in the repo, with the exception of GPT-2, which must downloaded from Zenodo (see "Files" of the latest version under the [Zenodo record](https://zenodo.org/records/20090538).
```
cp /PATH/TO/DOWNLOAD/gpt2-seq-2.onnx circuit/onnx/gpt2-seq-2.onnx
cp /PATH/TO/DOWNLOAD/gpt2-seq-32.onnx circuit/onnx/gpt2-seq-32.onnx
```
To build all ONNX circuits, run the following command:
```
python run.py --build-onnx-circuits
```

## Figure 1 (15-30 minutes)
Derive the constraint counts for FP-Spartan based on [ZKLP](https://eprint.iacr.org/2024/1842.pdf) arithmetizations: 
```
python run.py --fp-spartan-estimates
```
Run FP-Spartan on ONNX benchmarks:
```
python run.py --eval-fp-spartan
```
_Notes_: Constraint counts for FP-Spartan may be lower than the ZKLP estimates. The Spartan backend requires that synthetic constraint counts are a power of two, so we round down. Due to high resource requirements and long runtimes, timings for all benchmarks (besides Softmax) are estimated using the FP-Spartan cost model. 

Run Spain on ONNX benchmarks: 
```
python run.py <onnx-benchmark>
```

Possible ONNX benchmarks are:
- softmax-32x32
- layernorm-32x768
- gelu-32x3072
- gpt2-seq-2
- gpt2-seq-32
  
Measure native times for ONNX benchmarks:
```
python run.py --native <onnx-benchmark>
```
**Claim**: Spain shows 1-4 orders of magnitude of improvement over FP-Spartan in constraint count and prover time. For the small ONNX benchmarks, Spain's verifier is 2-10x faster. For GPT-2 benchmarks, our verifier is slower than FP-Spartan's (this discrepancy from the submitted paper is due to the fact that Spain now uses a larger field fo allow smaller epsilon values). 


## Figure 2 (5 minutes) 
Run Spain on LP benchmarks: 
```
python run.py <lp-benchmark>
```

Possible values for lp-benchmark are:
- adlittle
- afiro
- bnl1
- sc50a

To reproduce Otti's experimental results, refer to their [repo](https://github.com/eniac/otti). 


**Claim**: Against Otti, Spain improves constraint counts by roughly 300–600×, prover time by roughly 5–15×, and verifier time by roughly 5–50x (again, discrepancies from submitted paper are due to the fact that Spain now uses a larger field).  

## Figure 3 + 4 (Varies based on benchmark and _passes_)
Run Spain on ONNX benchmarks for varying _passes_: 
```
python run.py --passes 2 <onnx-benchmark>
```
**Claim**: The runtime of Spain's verifier per instance drops with increasing _passes_. The per instance cost of the prover remains relatively constant. 

## Figure 4 + 5 (5-10 minutes)
See the cost breakdown by prover/verifier for ONNX benchmarks: 
```
python run.py --phases <onnx-benchmark>
```
**Claim** The verifier is bottlenecked by A/B/C matrix evaluation and DARK evaluation. The prover is bottlenecked by DARK commitment costs and DARK evaluation. 


# Extensions
Feel free to experiment with other ONNX files, support for ONNX is relatively limited (although we will welcome open source extensions in the future). 

# Directory structure
1. ```circuit```: Circuit compiler from ONNX to R1CS.
2. ```dark```: Implementation of Spain's modified DARK PCS scheme.
3. ```ff```: Finite-field utilities.
4. ```fp-spartan-exp```: Evaluation code for the FP-Spartan synthetic benchmark.
5. ```iop```: Sum-check implementations.
6. ```model```: High-precision ONNX executor for generating ONNX benchmark witnesses.
7. ```otti-adapter```: Compiler from LP instance to R1CS.
8. ```parse```: Utilities for importing R1CS matrices.
9. ```protocol```: Generic utilities for interactive protocol message passing. 
10. ```spain```: Spain protocol implementation. 
11. ```stream```: Implementation of efficient memory-mapped vectors.




