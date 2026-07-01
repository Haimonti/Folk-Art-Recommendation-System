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
#SBATCH --time=4:00:00
#SBATCH --job-name=MAL_Anime
#SBATCH --output=run_job_MAL_Anime.out
#SBATCH --error=run_job_MAL_Anime.err
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
    export LD_LIBRARY_PATH='/.singularity.d/libs'
   
    #echo 'Running new_data.py'
    #python -u new_data.py

    #echo 'new_data.py ran successfully'
    mkdir -p save_models
    
    echo 'Running new_main_BPR.py'
    python -u new_main_BPR.py --data MAL_HSAL_Final  --model_record --resume 


    echo 'All jobs completed successfully.'
"

echo "=============================="
echo "Job finished at: $(date)"
echo "=============================="

