#!/bin/bash
#SBATCH --job-name=bnn_stgp_single_nn
#SBATCH --time=0:15:00
#SBATCH --mail-type=end,fail
#SBATCH --account=jiankang0
#SBATCH --partition=standard
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-120%5
#SBATCH --output=slurm_results/bnn_stgp_single_nn-%A_%a.out
#SBATCH --error=slurm_results/bnn_stgp_single_nn-%A_%a.err

module purge
module load python3.10-anaconda/2023.03
source /home/${USER}/.bashrc

cd "${SLURM_SUBMIT_DIR}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
EPOCHS=600
BURNIN_RATIO=0.9
TRUE_R2_VALUES="0.3 0.5 0.8"
A_FOR_SIGMA_SQUARED=1000
CR_CW_PRIOR_VALUES="0.01 1"

mkdir -p slurm_results

echo "SLURM job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Running one (cr/cw prior, true_r2, seed) combination per task; seeds=1-20, true_r2 values=${TRUE_R2_VALUES}, cr/cw prior values=${CR_CW_PRIOR_VALUES}"
echo "Each task runs epochs=${EPOCHS}, a_for_sigma_squared=${A_FOR_SIGMA_SQUARED}"
echo "Parameters: sample_size=2000, lr=5e-6, poly_degree=5, burnin_ratio=${BURNIN_RATIO}, hidden_unit_list=[4]"

export BNN_STGP_EPOCHS=${EPOCHS}
export BNN_STGP_BURNIN_RATIOS=${BURNIN_RATIO}
export BNN_STGP_A_FOR_SIGMA_SQUARED=${A_FOR_SIGMA_SQUARED}
export BNN_STGP_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results"

srun python run.py
