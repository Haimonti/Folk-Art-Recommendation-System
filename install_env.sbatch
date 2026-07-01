#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=haimonti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --qos=scavenger
#SBATCH --job-name=INSTALL_ENV
#SBATCH --output=install_%j.out
#SBATCH --error=install_%j.err

echo "Installing Python dependencies inside container"

export APPTAINER_CACHEDIR=$SLURMTMPDIR/apptainer-cache
mkdir -p $APPTAINER_CACHEDIR

apptainer exec --nv \
    --overlay pytorch_overlay.img \
    -B /projects/academic/haimonti:/projects \
    pytorch.sif \
    bash -c "
        cd /projects/academic/haimonti/prahithd/HSAL_GNN &&
        python -m pip install --upgrade pip &&
        pip install -r requirements_CCR.txt
    "

echo "Installation complete"

