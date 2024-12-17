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

# subjects=("A photo of tiger", "girl with a purse in anime style" "A man with short hair" "High-heeled shoe encrusted with diamonds")
# subj_phrases=("tiger", "girl purse anime" "man short hair" "High-heeled shoe encrusted")
# ref=("deer.png" "hulk.png" "long hair woman.png" "shoes.png")
# cntl=("depth" "depth" "canny" "canny")
# cond_prompts=("a photo of deer" "a photo of hulk" "A woman with long hair" "Enamel ankle boots with a strap")
# cond_phrases=("deer" "photo hulk" "woman long hair" "ankle boots strap")


subjects=("A photo of tiger")
cond_prompts=("A photo of deer")
ref=("deer.png")
cntl=("depth")

for index in "${!subjects[@]}"
do

	# my control
	# --gen_phrase="${subj_phrases[$index]}" \
	# --cond_phrase="${cond_phrases[$index]}" \
	# --ignore_special_tkns
	CUDA_VISIBLE_DEVICES="0" python3 smartcontrol_demo.py \
		--prompt="${subjects[$index]}" \
		--ref="${ref[$index]}" \
		--cond_prompt="${cond_prompts[$index]}" \
		--cntl="${cntl[$index]}" \
		--seed=12345 \
		--alpha_mask=1 \
		--alpha_attn_diff \
		--attn_diff_threshold=0.2

: << 'END'
	# SmartControl
	CUDA_VISIBLE_DEVICES="0" python3 smartcontrol_demo.py \
		--prompt="${subjects[$index]}" \
		--ref="${ref[$index]}" \
		--cntl="${cntl[$index]}" \
		--seed=12345 \
		--alpha_mask=1
END

done
