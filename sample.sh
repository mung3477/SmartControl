#!/bin/bash

control="depth"

# ControlNet
if [[ "$1" == "ControlNet" ]]; then
	python3 test.py --CUDA_VISIBLE_DEVICES="3" --control="$control" --mode="ControlNet"
elif [[ "$1" == "SmartControl" ]]; then
	python3 test.py --CUDA_VISIBLE_DEVICES="1" --control="$control" --mode="SmartControl"
else
	python3 test.py --CUDA_VISIBLE_DEVICES="2" --control="$control" --mode="ControlAttend"
fi
