#!/bin/bash

control="depth"

# ControlNet
if [[ "$1" == "ControlNet" ]]; then
	python3 sample.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="ControlNet" --seed_idx=2 --for_loop_idx=1
elif [[ "$1" == "SmartControl" ]]; then
	python3 sample.py --CUDA_VISIBLE_DEVICES="1" --control="$control" --mode="SmartControl"
else
	python3 sample.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="ControlAttend"
fi
