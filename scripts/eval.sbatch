#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/tang.ri/qwenmem_eval-%j.out
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

torchrun --nproc_per_node=4 -- evaluate.py \
    --m RichardGTang/QwenMemM1 \
    --o vsi_bench_outputs/m1/ \
    --n 16 \
    --distributed

torchrun --nproc_per_node=4 -- evaluate.py \
    --m RichardGTang/QwenMemM2 \
    --o vsi_bench_outputs/m2/ \
    --n 16 \
    --distributed

torchrun --nproc_per_node=4 -- evaluate.py \
    --m RichardGTang/QwenMemM3 \
    --o vsi_bench_outputs/m3/ \
    --n 16 \
    --distributed