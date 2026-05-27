#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --time=8:00:00
#SBATCH --mail-type=end,fail
#SBATCH --account=jiankang0
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-20%5
#SBATCH --output=slurm_results_5_15/cnn-%A_%a.out
#SBATCH --error=slurm_results_5_15/cnn-%A_%a.err

module purge
module load python3.10-anaconda/2023.03
source /home/${USER}/.bashrc

cd "${SLURM_SUBMIT_DIR}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${TASK_ID}
EPOCHS=150
R2_VALUES="0.8 0.5 0.3"

echo "SLURM job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Running CNN seed=${SEED}, epochs=${EPOCHS}, true_r2 values=${R2_VALUES}"
echo "Parameters: sample_size=2000, val_ratio=0.5, in_feature_list=[50,100], lr=1e-3"

export CNN_EPOCHS=${EPOCHS}
export CNN_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results"

srun python run.py
