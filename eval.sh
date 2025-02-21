#!/bin/bash
# conflict_degrees=("no_conflict" "mild_conflict" "significant_conflict")
conflict_degrees=("no_conflict")
model_types=("ControlNet" "SmartControl" "ControlAttend")

for conflict in "${conflict_degrees[@]}"; do
    for model in "${model_types[@]}"; do
		python3 eval.py --CUDA_VISIBLE_DEVICES="2" --conflict_degree="$conflict" --model_type="$model"
    done
done
