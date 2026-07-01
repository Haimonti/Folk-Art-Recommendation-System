#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=haimonti
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --qos=scavenger

apptainer overlay create --size 5000 pytorch_overlay.img


