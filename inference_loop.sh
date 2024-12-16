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

# subjects=("girl with a purse in anime style" "A man with short hair" "High-heeled shoe encrusted with diamonds")
# subj_phrases=("girl purse anime" "man short hair" "High-heeled shoe encrusted")
# ref=("hulk.png" "long hair woman.png" "shoes.png")
# cntl=("depth" "canny" "canny")
# cond_prompts=("a photo of hulk" "A woman with long hair" "Enamel ankle boots with a strap")
# cond_phrases=("photo hulk" "woman long hair" "ankle boots strap")


subjects=("High-heeled shoe encrusted with diamonds")
subj_phrases=("High-heeled shoe diamonds")

ref=("shoes.png")
cntl=("canny")
cond_prompts=("Enamel ankle boots with a strap")
cond_phrases=("Enamel boots")

for index in "${!subjects[@]}"
do
	CUDA_VISIBLE_DEVICES="0" python3 smartcontrol_demo.py \
		--prompt="${subjects[$index]}" \
		--gen_phrase="${subj_phrases[$index]}" \
		--ref="${ref[$index]}" \
		--cond_prompt="${cond_prompts[$index]}" \
		--cond_phrase="${cond_phrases[$index]}" \
		--cntl="${cntl[$index]}" \
		--seed=12345 \
		--alpha_mask=1 \
		--alpha_attn_diff \
		--ignore_special_tkns


done
