#!/bin/bash

control="depth"

# ControlNet
if [[ "$1" == "ControlNet" ]]; then
	if [[ "$2" == "distribute_seed" ]]; then
		python3 sample.py --CUDA_VISIBLE_DEVICES="0" --control="$control" --mode="ControlNet" --seed_idx=0
		python3 sample.py --CUDA_VISIBLE_DEVICES="1" --control="$control" --mode="ControlNet" --seed_idx=1
		python3 sample.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="ControlNet" --seed_idx=2
	else
		python3 sample.py --CUDA_VISIBLE_DEVICES="0" --control="$control" --mode="ControlNet" --seed_idx=2 --for_loop_idx=0 --init_subject_idx=4
	fi
elif [[ "$1" == "SmartControl" ]]; then
	if [[ "$2" == "distribute_seed" ]]; then
		python3 sample.py --CUDA_VISIBLE_DEVICES="0" --control="$control" --mode="SmartControl" --seed_idx=0
		python3 sample.py --CUDA_VISIBLE_DEVICES="1" --control="$control" --mode="SmartControl" --seed_idx=1
		python3 sample.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="SmartControl" --seed_idx=2
	else
		python3 sample.py --CUDA_VISIBLE_DEVICES="1" --control="$control" --mode="SmartControl"
	fi
else
	if [[ "$2" == "distribute_seed" ]]; then
		python3 sample.py --CUDA_VISIBLE_DEVICES="0" --control="$control" --mode="ControlAttend" --seed_idx=0
		python3 sample.py --CUDA_VISIBLE_DEVICES="1" --control="$control" --mode="ControlAttend" --seed_idx=1
		python3 sample.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="ControlAttend" --seed_idx=2
	else
		python3 sample.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="ControlAttend"
	fi
fi
