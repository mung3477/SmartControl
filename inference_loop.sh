#!/bin/bash

: << 'END'

subjects=("girl" "hulk" "mickey" "bassethound" "monster toy")
behaviors=("riding a bike" "doing deadlift" "doing bicycle kick")

for behavior in "${behaviors[@]}"
do
	for subject in "${subjects[@]}"
	do
		CUDA_VISIBLE_DEVICES="6" python3 smartcontrol_ipadapter_demo.py \
			--prompt="$subject $behavior" \
			--ref="$behavior.png" \
			--ip="$subject.png"
	done
done

END


# alphas=(0.0 0.2 0.4 0.6 0.8 1.0)
alphas=(1.0)
for alpha in "${alphas[@]}"
do
	CUDA_VISIBLE_DEVICES="3" python3 smartcontrol_demo.py \
		--prompt="a photo of tiger" \
		--gen_phrase="a photo of tiger" \
		--ref="deer.png" \
		--cond_prompt="a photo of deer" \
		--cond_phrase="a photo of deer" \
		--cntl="depth" \
		--seed=12345 \
		--alpha_mask=$alpha \
		--alpha_attn_diff


done
