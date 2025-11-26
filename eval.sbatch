#!/bin/bash
#SBATCH --job-name=qwenmem_eval
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
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
cd QwenMem/scripts/

torchrun --nproc_per_node=4 -- evaluate.py \
    --m RichardGTang/Qwen2_5_VL-3B-WithMemory \
    --o vsi_bench_outputs/ \
    --n 16 \
    --distributed

torchrun --nproc_per_node=4 -- evaluate.py \
    --m RichardGTang/Qwen2_5_VL-3B-WithVGGT \
    --o vsi_bench_outputs/ \
    --n 16 \
    --distributed