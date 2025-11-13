#!/bin/bash
#SBATCH --job-name=qwenmem_train
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a5000:4
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/tang.ri/slurm-%j.out
#SBATCH --export=ALL

echo "Job started on $(date)"
echo "Current directory: $(pwd)"

source ~/.bashrc

conda activate conda_env/qwenmem/
cd QwenMem/LLaMA-Factory/

lmf train ../configs/qwen2_5vl_vggt_sft_nframes_16.yaml