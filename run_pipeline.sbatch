#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=haimonti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --qos=scavenger
#SBATCH --job-name=HSAL_GNN_STEP1
#SBATCH --output=run_pipeline.out
#SBATCH --error=run_pipeline.err

set -e   # stop if any script fails

echo "Running on $(hostname)"
nvidia-smi

# Use node-local cache for apptainer
export APPTAINER_CACHEDIR=$SLURMTMPDIR/apptainer-cache
mkdir -p $APPTAINER_CACHEDIR

PROJECT_PATH=/projects/academic/haimonti/prahithd/HSAL_GNN

# -------- Step 1: Generate Negative Samples --------
apptainer exec --nv --overlay pytorch_overlay.img \
    -B /projects/academic/haimonti:/projects \
    pytorch.sif python $PROJECT_PATH/generate_neg.py


# -------- Step 2: Sinusoidal Processing --------
# apptainer exec --nv --overlay pytorch_overlay.img \
#     -B /projects/academic/haimonti:/projects \
#     pytorch.sif python $PROJECT_PATH/sinusoidal.py


# -------- Step 3: Create New Data Splits --------
# apptainer exec --nv --overlay pytorch_overlay.img \
#     -B /projects/academic/haimonti:/projects \
#     pytorch.sif python $PROJECT_PATH/new_data.py


# -------- Step 4: Training / Main Execution --------
# apptainer exec --nv --overlay pytorch_overlay.img \
#     -B /projects/academic/haimonti:/projects \
#     pytorch.sif python $PROJECT_PATH/new_main.py


echo "Step 1 finished successfully"

