#!/bin/bash

model_path=$1
output_path=$2
n_gpus=$3
benchmark_config=$4
generator=$5
num_generations=$6
num_seeds=$7

declare -a seeds=("1234" "2345" "3456" "4567" "5678" "6789" "7890" "8901")

echo "starting jobs"

for i in $(seq 0 $n_gpus)
do
    seed=${seeds[$i]}
    echo run device $i with seed $seed
    docker run -it -d -e MODEL_PATH=$model_path -e OUTPUT_PATH=$output_path -e SEED=$seed -e GENERATOR=$generator -e NUM_GENERATIONS=$num_generations -e NUM_SEEDS=$num_seeds --gpus device=$i --shm-size=128G --env-file=$benchmark_config --name=optimization_job_$i --mount type=bind,source=./run_production_optimization.sh,target=/workspace/bionemo/run_production_optimization.sh,readonly alphabind:latest ./run_production_optimization.sh
done
