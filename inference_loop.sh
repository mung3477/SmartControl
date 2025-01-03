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
# for alpha in "${alphas[@]}"

# subjects=("A photo of tiger" "girl with a purse in anime style" "A man with short hair" "High-heeled shoe encrusted with diamonds" "A dog doing deadlift")
# ref=("deer.png" "hulk.png" "long hair woman.png" "shoes.png" "doing deadlift.png")
# cntl=("depth" "depth" "canny" "canny" "depth")
# mask_prompt=("A photo of tiger with a horn" "muscular girl with a purse" "A man with long hair" "enamel ankle boots with a strap" "a man doing deadlift")
# focus_prompt=("tiger" "girl" "man" "boots" "man")

subjects=("A bear doing deadlift")
mask_prompt=("a man doing deadlift")
focus_prompt=("deadlift")
ref=("doing deadlift.png")
cntl=("depth")

for index in "${!subjects[@]}"
do

	# my control
	# --ignore_special_tkns
	CUDA_VISIBLE_DEVICES="0" python3 smartcontrol_demo.py \
		--prompt="${subjects[$index]}" \
		--cond_prompt="${mask_prompt[$index]}" \
		--focus_prompt="${focus_prompt[$index]}" \
		--ref="${ref[$index]}" \
		--cntl="${cntl[$index]}" \
		--seed=12345 \
		--alpha_mask=1 \
		--alpha_attn_prev \
		--alpha_fixed \
		--ignore_special_tkns \

: << 'END'
	# SmartControl
	CUDA_VISIBLE_DEVICES="0" python3 smartcontrol_demo.py \
		--prompt="${subjects[$index]}" \
		--ref="${ref[$index]}" \
		--cntl="${cntl[$index]}" \
		--seed=12345 \
		--alpha_mask=1 \
		--alpha_fixed
END

done
