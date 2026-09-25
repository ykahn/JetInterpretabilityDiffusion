# ImprovedModel: distributed-training copy of the singular-distribution model of record

Copy of `PhaseSpaceDiffusion/improved/record/{model_singular.py, train_singular.py, utils.py}` (the qqg / APS model
of record, Sept 2026) with data-parallel multi-GPU training added, following the DDP pattern of the original
`phasespace_diffusion/train.py`.  Everything else (network, schedule, loss, sampler, checkpoint format) is identical
to `improved/record`; use its `generate.py` / `sample.py` / `evolution.py` on the checkpoints.

| file | change |
|---|---|
| `model_singular.py` | `train()` shards the events over the ranks, caches only the rank's shard, wraps the network in `DistributedDataParallel`, averages the epoch loss over ranks, writes checkpoints from rank 0; `raw_net` / `dist_info()` helpers. Single-process behaviour unchanged. |
| `train_singular.py` | initialises the process group from the `torchrun` environment, binds each rank to its GPU, writes files from rank 0 only, records `world_size` in `args.json`. |
| `utils.py` | unchanged. |
| `slurm/train_singular.sh` | one node, 4 GPUs, `torchrun --standalone --nproc_per_node=4`. |

Launch:
```bash
DATA=/project/6104763/yfkahn/PhaseSpaceDiffusion/datasets/SARGE_N10_xi40_1M.pt NAME=aps_xi40 sbatch slurm/train_singular.sh
python train_singular.py --data-file ... --output-dir runs/x        # single GPU, as before
torchrun --standalone --nproc_per_node=4 train_singular.py ...       # interactive multi-GPU
```

What it buys: memory.  The forward cache is the only memory that grows with the dataset; with 4 ranks each holds a
quarter of it, so 2M events at N = 20 fit with the default dense cache (23 GB per GPU).  Per optimiser step the
time stays ~49 ms on an L40S (latency-bound at batch 1024 whatever N), so with `--batch-size 1024` per rank an
epoch takes about a quarter of the single-GPU time but also contains a quarter of the optimiser updates (global
batch 4096).  The model has only been validated at global batch 1024 (single GPU); to keep that, use
`--batch-size 256`, which keeps the per-epoch update count and the memory benefit but not the speed-up.

Test: `DDP_BACKEND=gloo torchrun --standalone --nproc_per_node=2 train_singular.py ...` runs two ranks on one GPU
(NCCL needs one GPU per rank); a 2-rank smoke test on 4096 events gives a checkpoint that `generate.py` samples.
