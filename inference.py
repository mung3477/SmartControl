import argparse
import os
from typing import List, TypedDict, Optional
from warnings import warn

subjects = ["dog"]

class Action(TypedDict):
	prompt: str
	mask_prompt: str
	focus_prompt: str
	reference: str
	control: str
	alpha_mask: Optional[float]

actions: List[Action] = [{
	"prompt": "A {subject} playing a trumpet",
	"mask_prompt": "A man playing a trumpet",
	"focus_prompt": "playing trumpet",
	"reference": "Trumpet.png",
	"control": "depth"
},{
	"prompt": "A {subject} doing handstand exercise",
	"mask_prompt": "A woman doing handstand exercise",
	"focus_prompt": "doing handstand",
	"reference": "Handstand.jpg",
	"control": "depth"
},{
	"prompt": "A {subject} doing deadlift",
	"mask_prompt": "A man doing deadlift",
	"focus_prompt": "deadlift",
	"reference": "doing deadlift.png",
	"control": "pose"
},{
	"prompt": "A {subject} riding a bicycle",
	"mask_prompt": "A man riding a bicycle",
	"focus_prompt": "riding bicycle",
	"reference": "riding a bike.png",
	"control": "pose"
},{
	"prompt": "A {subject} with both arms gesture",
	"mask_prompt": "A man with both arms gesture",
	"focus_prompt": "both arms gesture",
	"reference": "Cheer.jpg",
	"control": "pose"
},{
	"prompt": "A {subject} holding a clarinet",
	"mask_prompt": "A man holding a clarinet",
	"focus_prompt": "holding",
	"reference": "Clarinet.png",
	"control": "depth"
},{
	"prompt": "A {subject} holding a guitar",
	"mask_prompt": "A man holding a guitar",
	"focus_prompt": "holding",
	"reference": "Guitar.png",
	"control": "depth"
},{
	"prompt": "A {subject} praying with clasped hands",
	"mask_prompt": "A woman praying with clasped hands",
	"focus_prompt": "praying with clasped hands",
	"reference": "Pray.jpg",
	"control": "pose"
}]

def inference_loop(mode: str, CUDA_VISIBLE_DEVICES: str):
	for subject in subjects:
		for action in actions:
			if mode == "Mine":
				os.system(f'CUDA_VISIBLE_DEVICES="{CUDA_VISIBLE_DEVICES}" python3 smartcontrol_demo.py \
									--prompt="{action["prompt"].format(subject=subject)}" \
									--cond_prompt="{action["mask_prompt"]}" \
									--focus_prompt="{action["focus_prompt"]}" \
									--ref="{action["reference"]}" \
									--cntl="{action["control"]}" \
									--seed=12345 \
									--save_attn \
									--alpha_mask=1 \
									--alpha_attn_prev \
									--alpha_fixed \
									--ignore_special_tkns \
									--editing_prompt "dog" "man" \
									--reverse_edit_direction 0 0 \
									--edit_warmup_steps 10 10 \
									--edit_guidance_scale 5 5 \
									--edit_threshold 0.90 0.90 \
									--edit_weights 1 1 \
				')

			elif mode == "SmartControl":
				os.system(f'CUDA_VISIBLE_DEVICES="{CUDA_VISIBLE_DEVICES}" python3 smartcontrol_demo.py \
									--prompt="{action["prompt"].format(subject=subject)}" \
									--ref="{action["reference"]}" \
									--cntl="{action["control"]}" \
									--seed=12345 \
									--alpha_mask=1 \
									--editing_prompt "animal, dog" \
									--reverse_edit_direction 0 \
									--edit_warmup_steps 10 \
									--edit_guidance_scale 5 \
									--edit_threshold 0.90 \
									--edit_weights 1 \
				')

			elif mode == "ControlNet":
				os.system(f'CUDA_VISIBLE_DEVICES="{CUDA_VISIBLE_DEVICES}" python3 smartcontrol_demo.py \
									--prompt="{action["prompt"].format(subject=subject)}" \
									--ref="{action["reference"]}" \
									--cntl="{action["control"]}" \
									--seed=12345 \
									--alpha_mask=0.4 \
									--alpha_fixed \
				')

			else:
				warn(f"{mode} is not a valid inference mode.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default="Mine", help="ControlNet | SmartControl | Mine")
	parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")

	args = parser.parse_args()
	inference_loop(args.mode, args.CUDA_VISIBLE_DEVICES)




"""
,{
	"prompt": "A {subject} raising finger",
	"mask_prompt": "A woman is raising a finger",
	"focus_prompt": "raising",
	"reference": "Gesture.jpg",
	"control": "depth"
},{
	"prompt": "A {subject} doing squat exercise",
	"mask_prompt": "A woman doing squat exercise",
	"focus_prompt": "squat",
	"reference": "Squat.jpg",
	"control": "depth"
},{
	"prompt": "A {subject} doing bicycle kick",
	"mask_prompt": "A man doing bicycle kick",
	"focus_prompt": "bicycle kick",
	"reference": "doing bicycle kick.png",
	"control": "depth"
},{
	"prompt": "A {subject} doing meditation",
	"mask_prompt": "A woman doing meditation",
	"focus_prompt": "doing meditation",
	"reference": "Meditate.jpg",
	"control": "pose"
},{
	"prompt": "A saluting {subject}",
	"mask_prompt": "A saluting soldier",
	"focus_prompt": "saluting",
	"reference": "Salute.jpg",
	"control": "pose"
},{
	"prompt": "A {subject} sitting on a cube",
	"mask_prompt": "A man sitting on a cube",
	"focus_prompt": "sitting",
	"reference": "Sit.jpg",
	"control": "depth"
}
"""
