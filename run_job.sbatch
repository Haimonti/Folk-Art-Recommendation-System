#!/bin/bash
#SBATCH --cluster=ub-hpc
#SBATCH --account=haimonti
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=ML1M
#SBATCH --output=run_job.out
#SBATCH --error=run_job.err
#SBATCH --mail-user=prahithd@buffalo.edu
#SBATCH --mail-type=ALL

echo "=============================="
echo "Job started on: $(hostname)"
echo "GPUs allocated:"
nvidia-smi
echo "=============================="

# Set Apptainer directories
export APPTAINER_CACHEDIR=/vscratch/grp-haimonti/$USER/apptainer_cache
export APPTAINER_TMPDIR=/vscratch/grp-haimonti/$USER/apptainer_tmp

mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR

echo "Starting container..."

apptainer exec --nv --overlay pytorch_overlay.img:rw pytorch.sif bash -c "

    echo 'Inside container'
    source venv/bin/activate


    echo 'Running new_main_BPR.py'
    python -u new_main_BPR.py --data MovieLens_all --batchSize 256 --item_max_length 50 --user_max_length 50 --k_hop 2 --lr 0.001 --l2 0.0001 --layer_num 3 --hidden_size 50 --gpu 0 --epoch 50 --model_record --resume 



    echo 'All jobs completed successfully.'
"

echo "=============================="
echo "Job finished at: $(date)"
echo "=============================="

