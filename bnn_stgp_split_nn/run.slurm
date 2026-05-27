#!/bin/bash
#SBATCH --job-name=bnn_stgp_split_nn
#SBATCH --time=2:00:00
#SBATCH --mail-type=end,fail
#SBATCH --account=jiankang0
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-20%5
#SBATCH --output=/home/%u/slurm_output_5_15/bnn_stgp_split_nn-%A_%a.out
#SBATCH --error=/home/%u/slurm_output_5_15/bnn_stgp_split_nn-%A_%a.err

module purge
module load python3.10-anaconda/2023.03
source /home/${USER}/.bashrc

cd "${SLURM_SUBMIT_DIR}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${TASK_ID}
EPOCHS=600
NUM_CONFIGS=3

echo "SLURM job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Running seed=${SEED}, epochs=${EPOCHS}, configs=${NUM_CONFIGS}"
echo "R2 sweep: true_r2=0.8 0.5 0.3, lr=5e-6, burnin_ratio=0.9, poly_degree=5"

srun python run.py
