#!/bin/bash
# Distributed (4-GPU, one L40S node) training of the singular-distribution model of record.
#
#   cd /project/6104763/yfkahn/JetDiffusion/ImprovedModel
#   DATA=/project/6104763/yfkahn/PhaseSpaceDiffusion/datasets/SARGE_N10_xi40_1M.pt NAME=aps_xi40 sbatch slurm/train_singular.sh [extra train_singular.py options]
#
# Data-parallel over the 4 GPUs (torchrun): the forward cache is sharded, so the GPU memory per rank is 1/4 of the
# single-GPU cache (2M events x N=20 with the default dense cache: 91 GB total = 23 GB per GPU, fits an L40S), and the
# GLOBAL batch is 4 x --batch-size events per optimiser step.  Per-step time stays ~49 ms (latency-bound), so an epoch
# takes ~1/4 of the single-GPU time and has 1/4 of the optimiser updates.  Outputs: runs/$NAME/ (checkpoint format
# identical to improved/record; sample with improved/record/generate.py or sample.py).
#SBATCH --job-name=train_singular_ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=11:59:00
#SBATCH --output=slurm/logs/train_%j.out
module load scipy-stack
source /project/6104763/yfkahn/JetDiffusion/venv/bin/activate
export PYTHONUNBUFFERED=1
cd /project/6104763/yfkahn/JetDiffusion/ImprovedModel
: "${DATA:?set DATA=path/to/events.pt}"; : "${NAME:?set NAME=run_name}"
NGPU=$(nvidia-smi -L | wc -l)
torchrun --standalone --nproc_per_node=$NGPU train_singular.py --data-file "$DATA" --output-dir runs/"$NAME" "$@"
