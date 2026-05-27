#!/bin/bash
#SBATCH --job-name=bnn_stgp_split_nn_view
#SBATCH --time=1:00:00
#SBATCH --mail-type=end,fail
#SBATCH --account=jiankang0
#SBATCH --partition=standard
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=../slurm_output_5_11/view_results-%j.out
#SBATCH --error=../slurm_output_5_11/view_results-%j.err

set -eo pipefail

module purge
module load python3.10-anaconda/2023.03
source /home/${USER}/.bashrc

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}"

echo "SLURM job ${SLURM_JOB_ID}"
echo "Running run_view_results.py from ${SCRIPT_DIR}"
echo "Input folder: ${SCRIPT_DIR}/newest_results"
echo "Output folder: ${SCRIPT_DIR}/view_results_outputs"
echo "Exporting results_summary.csv, per-polydegree train/test R2 grids, modality 1/2 selection grids, and aggregate selections"

srun python "${SCRIPT_DIR}/run_view_results.py"
