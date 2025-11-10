#! /bin/bash

python scripts/dataset_jsons.py --nframes 32
python scripts/dataset_jsons.py --nframes 16
python scripts/dataset_jsons.py --nframes 8

mv qwenmem_nframes_32.json LLaMA-Factory/data/qwenmem_nframes_32.json
mv qwenmem_nframes_16.json LLaMA-Factory/data/qwenmem_nframes_16.json
mv qwenmem_nframes_8.json LLaMA-Factory/data/qwenmem_nframes_8.json

cd LLaMA-Factory/
pip install -e ".[torch,metrics]" --no-build-isolation

cd ..

mv src/QwenMem/ LLaMA-Factory/src/qwenmem/