#!/bin/bash
#SBATCH --cluster=faculty
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --job-name=lrd_chunked
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=snirwade@buffalo.edu
#SBATCH --mail-type=ALL

# Generic chunked LRD job. Pick the dataset (and memory) when submitting, e.g.
#   sbatch --export=ALL,DATASET=Goodreads_Romance_HSAL_Final --job-name=LRDchunk_Romance run_lrd_chunked.sbatch
# Scavenger jobs can be stopped: --requeue restarts the job and --resume continues training.


WORKDIR=/vscratch/grp-haimonti/snirwade/HSAL-Chunking
echo "Job started : $(date)   host: $(hostname)"
nvidia-smi
export APPTAINER_CACHEDIR=/vscratch/grp-haimonti/$USER/apptainer_cache
export APPTAINER_TMPDIR=/vscratch/grp-haimonti/$USER/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
cd $WORKDIR
# -B: venv links to ../HSAL-GNN-Research/venv, which links to /projects/academic/haimonti/snirwade/venv;
# the container must see both folders
apptainer exec --nv --writable-tmpfs -B /vscratch/grp-haimonti/snirwade,/projects/academic/haimonti pytorch.sif bash lrd_chunked_steps.sh
status=$?
echo "Job finished : $(date)   exit code: $status"
exit $status
