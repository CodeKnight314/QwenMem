#!/bin/bash
#SBATCH --job-name=qwenmem_train
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/tang.ri/qwenmem-%j.out
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

cd QwenMem/
cp -f configs/models/config_m1.json LLaMA-Factory/models/Qwen2_5_VL-3B-WithMemory/config.json
cd LLaMA-Factory/

echo "Starting training at $(date)"

lmf train ../configs/training/qwen2_5vl_mem_sft_nframes_16_m1.yaml
