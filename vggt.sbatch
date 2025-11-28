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

mkdir -p /scratch/tang.ri/cache
mkdir -p /scratch/tang.ri/triton_cache

export HF_HOME=/scratch/tang.ri/cache
export TRITON_CACHE_DIR=/scratch/tang.ri/triton_cache
export FORCE_TORCHRUN=1

conda activate conda_env/qwenmem/
cd QwenMem/LLaMA-Factory/

lmf train ../configs/vggt/lora/qwen2_5vl_vggt_sft_nframes_16.yaml