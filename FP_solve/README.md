# Solving Fokker-Planck Hamiltonian

```
.
├── compute_rt.sh       <- Slrum script to rund the code
└── fast_Rt_cluster.py  <- Finite box FP Hamiltonian solver
```

## Dependencies
- numpy
- scipy-1.12.0
- joblib

## Running the code
Modify `compute_rt.sh' to set relevant environment variables and paths.

--input: set of 3-vectors (Q)

--output: where to save the results

--tau: time step for the evolution (in integrated time)

--observable: "constant" for sanity check, "thrust_tensor" for thrust overlap

--max-reference-events: How many 3-vectors to use from P(Q)

--n-mc: MC samples

On the cluster, run the code with:
```
sbatch <path to scritps>/compute_rt.sh
```
