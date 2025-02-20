import argparse
import os
from test.inference import EvalModel, ModelType
from test.test_set import (animal_prompts, animal_subjects, human_prompts,
                           human_subjects, seeds)
from typing import Callable


def test_no_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = "/root/Desktop/workspace/SmartControl/test/output/no_conflict",
	alpha_mask: str = "1",
	save_attn: bool = False
):
	eval.set_output_dir(output_dir)

	for seed in seeds:
		for subject in human_subjects:
			for prompt, mask_prompt, focus_tokens in human_prompts:
				prompt = prompt.format(subject=subject)
				mask_prompt = mask_prompt.format(condition_subject=subject)
				reference = f"/root/Desktop/workspace/SmartControl/assets/test/{subject}/{' '.join(prompt.split(' ')[2:])}.png"

				output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
				eval.postprocess(output, output_name, save_attn=save_attn)

		for subject in animal_subjects:
			for prompt, mask_prompt, focus_tokens in animal_prompts:
				prompt = prompt.format(subject=subject)
				mask_prompt = mask_prompt.format(condition_subject=subject)
				reference = f"/root/Desktop/workspace/SmartControl/assets/test/{subject}/{' '.join(prompt.split(' ')[2:])}.png"

				output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
				eval.postprocess(output, output_name, save_attn=save_attn)

	# CLIP score
	# ImageReward
	# Picscore

def test_mild_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = "/root/Desktop/workspace/SmartControl/test/output/no_conflict",
	alpha_mask: str = "1",
	mask_prompt: str = None,
	focus_tokens: str = None,
	save_attn: bool = False
):
	eval.set_output_dir(output_dir)

	for seed in seeds:
		for subject in human_subjects:
			for subject2 in human_subjects:
				if subject == subject2:
					continue
				for prompt, mask_prompt, focus_tokens in human_prompts:
					prompt = prompt.format(subject=subject)
					mask_prompt = mask_prompt.format(condition_subject=subject2)
					reference = f"/root/Desktop/workspace/SmartControl/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

					output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
					eval.postprocess(output, output_name, save_attn=save_attn)

		for subject in animal_subjects:
			for subject2 in animal_subjects:
				if subject == subject2:
					continue
				for prompt, mask_prompt, focus_tokens in animal_prompts:
					prompt = prompt.format(subject=subject)
					mask_prompt = mask_prompt.format(condition_subject=subject2)
					reference = f"/root/Desktop/workspace/SmartControl/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

					output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
					eval.postprocess(output, output_name, save_attn=save_attn)

	# CLIP score
	# ImageReward
	# Picscore

def test_significant_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = "/root/Desktop/workspace/SmartControl/test/output/no_conflict",
	alpha_mask: str = "1",
	save_attn: bool = False
):
	eval.set_output_dir(output_dir)

	for seed in seeds:
		for subject in human_subjects:
			for subject2 in animal_subjects:
				for prompt, mask_prompt, focus_tokens in animal_prompts:
					prompt = prompt.format(subject=subject)
					mask_prompt = mask_prompt.format(condition_subject=subject2)
					reference = f"/root/Desktop/workspace/SmartControl/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

					output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
					eval.postprocess(output, output_name, save_attn=save_attn)

		for subject in animal_subjects:
			for subject2 in human_subjects:
				for prompt, mask_prompt, focus_tokens in human_prompts:
					prompt = prompt.format(subject=subject)
					mask_prompt = mask_prompt.format(condition_subject=subject2)
					reference = f"/root/Desktop/workspace/SmartControl/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

					output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
					eval.postprocess(output, output_name, save_attn=save_attn)

	# CLIP score
	# ImageReward
	# Picscore

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
	parser.add_argument('--control', type=str, required=True, help="depth | pose | canny")
	parser.add_argument('--model', type=str, default="ControlAttend", help="ControlNet | SmartControl | ControlAttend")
	parser.add_argument('--alpha_mask', nargs="*", type=float, default=[1], help="Mask applied on inferred alpha. [1, 0, 0, 0] means only upper left is used with 1. None uses SmartControl's inferred alpha_mask.")

	args = parser.parse_args()

	args.modelType = ModelType.str2enum(args.model)

	return args

def main():
	args = parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

	eval = EvalModel(args.control)
	inference = eval.get_inference_func(args.modelType)

	test_no_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask)
	test_mild_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask)
	test_significant_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask)


if __name__ == "__main__":
    main()
