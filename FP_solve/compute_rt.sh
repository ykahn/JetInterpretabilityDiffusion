#!/bin/bash
#SBATCH --job-name=compute_Rt_thrust
#SBATCH --array=0-99%10
#SBATCH --time=1-00:00:00
#SBATCH --account=<your-project-account>
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --nodes=1

# below change to whatever you need to get your environment to work

module load python/3.12
source $HOME/jupyter_py3/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECTDIR=$HOME/projects/aip-yfkahn/cjpark/
BINDIR=$PROJECTDIR/bins/
DATADIR=$PROJECTDIR/data/
OUTDIR=$PROJECTDIR/results/thrust_earlytime/
LOGDIR=$PROJECTDIR/logs/

mkdir -p $OUTDIR
mkdir -p $LOGDIR

# OBSERVABLE="constant"
OBSERVABLE="thrust_tensor"

mkdir -p $OUTDIR

srun --output ${LOGDIR}/fast_Rt_%j.out --error ${LOGDIR}/fast_Rt_%j.err \
  python ${BINDIR}/fast_Rt_cluster.py \
  --input ${DATADIR}/q_sph.npz \
  --output ${OUTDIR}/thrust_results.npz \
  --tau $(python -c "import numpy as np; print(*np.logspace(-3, 1, 100))") \
  --observable ${OBSERVABLE} \
  --max-reference-events 65536 \
  --n-mc 8192 \
  --batch-size 64 \
  --tau-index "$SLURM_ARRAY_TASK_ID" \
  --n-jobs ${SLURM_CPUS_PER_TASK}
