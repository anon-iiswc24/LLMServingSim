# LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving Systems at Scale

## Build LLMServingSim

1. Install Dependency (tested in python 3.9, GCC, G++ 7.5.0)

```bash
# using conda
conda create -n env_name python=3.9
conda activate env_name
conda install libprotobuf=3.6.1
conda install cmake=3.15
conda install boost-cpp=1.7.4
```

2. Install Python Dependency

```bash
pip install -r requirements.txt
```

3. Build ASTRA-Sim, Chakra, Polymath

```bash
./build/astra_analytical/build.sh
cd extern/graph_frontend/chakra
pip install -e .
cd ../../../execution_engine/polymath
pip install -e .
cd ../..
```

## Run LLMServingSim

1. Set Configurations

Examples:

- Network config: “astra-sim/inputs/network/analytical/fully_connected_{network_dim}d_{number_of_NPUs}d.json”
- NPU config: “execution_engine/codelets_src/codelets/examples/genesys/configs/benchmark_128x128.json”
2. Run LLMServingSim

Short Test Run

```bash
python3 main.py --model_name 'gpt3-175b' --npu_num 16 --npu_group 1
```

Full Test Run

```bash
python3 main.py --model_name 'llama-7b' --npu_num 1 --npu_group 1 --npu_mem 24 --dataset 'dataset/share-gpt-req100-rate10.tsv'
```

Arguments of main.py

| Arguments | Supporting Options | Default Value | Notes |
| --- | --- | --- | --- |
| --model_name | 'gpt2', 'gpt3-6.7b', 'gpt3-125m', 'gpt3-350m', 'gpt3-760m', 'gpt3-1.3bm', 'gpt3-2.7b', 'gpt3-6.7b', 'gpt3-13b', 'gpt3-175b', 'opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b', 'opt-66b', 'opt-175b', 'llama-7b', 'llama-30b', 'llama-70b' | 'gpt2' |  |
| --npu_num  | Integer | 16 |  |
| --max_batch | Integer | 0 | 0: no limit |
| --batch_delay  | Integer | 0 |  |
| --scheduling | 'none', 'orca' | 'orca' |  |
| --parallel | 'pipeline', 'tensor', 'hybrid' | 'hybrid' |  |
| --npu_group | Integer | 1 |  |
| --npu_mem | Integer | 40 |  |
| --kv_manage | 'max', 'pow2', 'oracle', 'vllm' | 'vllm' |  |
| --block_size | Integer | 8 |  |
| --pim_type | 'none', 'local', 'pool' | 'none' |  |
| --sub_batch |  | False | Sub-batch Scheduling On/Off |
| --dataset | 'dataset_path.tsv' | None | None: manually add requests in main.py |