#! /bin/bash

python scripts/utils/dataset_jsons.py --nframes 32 --nsamples 15000
python scripts/utils/dataset_jsons.py --nframes 16 --nsamples 15000
python scripts/utils/dataset_jsons.py --nframes 8 --nsamples 15000

mv qwenmem_nframes_32.json LLaMA-Factory/data/qwenmem_nframes_32.json
mv qwenmem_nframes_16.json LLaMA-Factory/data/qwenmem_nframes_16.json
mv qwenmem_nframes_8.json LLaMA-Factory/data/qwenmem_nframes_8.json

cd LLaMA-Factory/
pip install -e ".[torch,metrics]" --no-build-isolation

cd ..

git clone https://huggingface.co/Codeknight314/Qwen2_5_VL-3B-WithMemory
git clone https://huggingface.co/Codeknight314/Qwen2_5_VL-3B-WithVGGT

mkdir -p LLaMA-Factory/models

mv Qwen2_5_VL-3B-WithMemory LLaMA-Factory/models/Qwen2_5_VL-3B-WithMemory/
mv Qwen2_5_VL-3B-WithVGGT   LLaMA-Factory/models/Qwen2_5_VL-3B-WithVGGT/