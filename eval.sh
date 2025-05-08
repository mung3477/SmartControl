#!/bin/bash
# controlnet_alphas=("1.0" "0.4")
controlnet_alphas=("1.0" "0.4")
# attn_bias=("0.0" "1.0" "3.0")
attn_bias=("0.0")
# conflict_degrees=("no_conflict" "mild_conflict" "significant_conflict")
conflict_degrees=("no_conflict" "mild_conflict" "significant_conflict")
# model_types=("ControlNet" "SmartControl" "ControlAttend")
model_types=("ControlNet")

for conflict in "${conflict_degrees[@]}"; do
    for model in "${model_types[@]}"; do
        for alpha in "${controlnet_alphas[@]}"; do
            for bias in "${attn_bias[@]}"; do
                python3 eval.py --CUDA_VISIBLE_DEVICES="2" --conflict_degree="$conflict" --model_type="$model" --controlnet_alpha="$alpha"
                # python3 eval.py --CUDA_VISIBLE_DEVICES="1" --conflict_degree="$conflict" --model_type="$model" --controlnet_alpha="$alpha" --attn_bias="$bias"
            done
        done
    done
done
