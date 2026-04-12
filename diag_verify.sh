#!/bin/bash
#SBATCH --job-name=agama_verify
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/home/arora125/diag_verify_out.txt
#SBATCH --error=/gpfs/home/arora125/diag_verify_err.txt

source ~/.bashrc
mamba activate Nbodystream

python /gpfs/home/arora125/libs/AGAMA_GPU/diag_verify_inner.py
