import argparse
import os
from test.inference import EvalModel
from test.types import ModelType
from test.test_set import (animal_prompts, animal_subjects, human_prompts,
                           human_subjects, selected, seeds)
from typing import Callable, Optional
from tqdm import tqdm

def test_no_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = f"{os.getcwd()}/test/output/no_conflict",
	alpha_mask: str = "1",
	save_attn: bool = False,
	seed_idx: Optional[int] = None,
  	for_loop_idx: Optional[int] = None,
   	init_subject_idx: Optional[int] = None,
):
	eval.set_output_dir(output_dir)

	for i, seed in tqdm(enumerate(seeds), desc="For all seeds"):
		if seed_idx is not None and i != seed_idx:
			continue
		if for_loop_idx is None or for_loop_idx == 0:
			for subj_idx, subject in tqdm(enumerate(human_subjects), desc="For all human subjects"):
				if init_subject_idx is None or subj_idx >= init_subject_idx:
					for prompt, mask_prompt, focus_tokens in human_prompts:
						prompt = prompt.format(subject=subject)
						mask_prompt = mask_prompt.format(condition_subject=subject)
						focus_tokens = focus_tokens.format(condition_subject=subject.split(" ")[-1])
						reference = f"{os.getcwd()}/assets/test/{subject}/{' '.join(prompt.split(' ')[2:])}.png"

						output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
						eval.postprocess(output, output_name, save_attn=save_attn)

		if for_loop_idx is None or for_loop_idx == 1:
			for subj_idx, subject in tqdm(enumerate(animal_subjects), desc="For all human subjects"):
				if init_subject_idx is None or subj_idx >= init_subject_idx:
					for prompt, mask_prompt, focus_tokens in animal_prompts:
						prompt = prompt.format(subject=subject)
						mask_prompt = mask_prompt.format(condition_subject=subject)
						focus_tokens = focus_tokens.format(condition_subject=subject.split(" ")[-1])
						reference = f"{os.getcwd()}/assets/test/{subject}/{' '.join(prompt.split(' ')[2:])}.png"

						output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
						eval.postprocess(output, output_name, save_attn=save_attn)

def test_mild_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = f"{os.getcwd()}/test/output/mild_conflict",
	alpha_mask: str = "1",
	mask_prompt: str = None,
	focus_tokens: str = None,
	save_attn: bool = False,
	seed_idx: Optional[int] = None,
 	for_loop_idx: Optional[int] = None,
  	init_subject_idx: Optional[int] = None,
):
	eval.set_output_dir(output_dir)

	for i, seed in tqdm(enumerate(seeds), desc="For all seeds"):
		if seed_idx is not None and i != seed_idx:
			continue

		if for_loop_idx is None or for_loop_idx == 0:
			for subj_idx, subject in tqdm(enumerate(human_subjects), desc="For all human subjects"):
				if init_subject_idx is None or subj_idx >= init_subject_idx:
					for subject2 in human_subjects:
						if subject == subject2:
							continue
						for prompt, mask_prompt, focus_tokens in human_prompts:
							prompt = prompt.format(subject=subject)
							mask_prompt = mask_prompt.format(condition_subject=subject2)
							focus_tokens = focus_tokens.format(condition_subject=subject2.split(" ")[-1])
							reference = f"{os.getcwd()}/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

							output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
							eval.postprocess(output, output_name, save_attn=save_attn)

		if for_loop_idx is None or for_loop_idx == 1:
			for subj_idx, subject in tqdm(enumerate(animal_subjects), desc="For all animal subjects"):
				if init_subject_idx is None or subj_idx >= init_subject_idx:
					for subject2 in animal_subjects:
						if subject == subject2:
							continue
						for prompt, mask_prompt, focus_tokens in animal_prompts:
							prompt = prompt.format(subject=subject)
							mask_prompt = mask_prompt.format(condition_subject=subject2)
							focus_tokens = focus_tokens.format(condition_subject=subject2.split(" ")[-1])
							reference = f"{os.getcwd()}/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

							output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
							eval.postprocess(output, output_name, save_attn=save_attn)

def test_significant_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = f"{os.getcwd()}/test/output/significant_conflict",
	alpha_mask: str = "1",
	save_attn: bool = False,
	seed_idx: Optional[int] = None,
	for_loop_idx: Optional[int] = None,
	init_subject_idx: Optional[int] = None,
):
	eval.set_output_dir(output_dir)

	for i, seed in tqdm(enumerate(seeds), desc="For all seeds"):
		if seed_idx is not None and i != seed_idx:
			continue

		if for_loop_idx is None or for_loop_idx == 0:
			for subj_idx, subject in tqdm(enumerate(human_subjects), desc="For all human subjects"):
				if init_subject_idx is None or subj_idx >= init_subject_idx:
					for subject2 in animal_subjects:
						for prompt, mask_prompt, focus_tokens in animal_prompts:
							prompt = prompt.format(subject=subject)
							mask_prompt = mask_prompt.format(condition_subject=subject2)
							focus_tokens = focus_tokens.format(condition_subject=subject2.split(" ")[-1])
							reference = f"{os.getcwd()}/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

							output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
							eval.postprocess(output, output_name, save_attn=save_attn)

		if for_loop_idx is None or for_loop_idx == 1:
			for subj_idx, subject in tqdm(enumerate(animal_subjects), desc="For all animal subjects"):
				if init_subject_idx is None or subj_idx >= init_subject_idx:
					for subject2 in human_subjects:
						for prompt, mask_prompt, focus_tokens in human_prompts:
							prompt = prompt.format(subject=subject)
							mask_prompt = mask_prompt.format(condition_subject=subject2)
							focus_tokens = focus_tokens.format(condition_subject=subject2.split(" ")[-1])
							reference = f"{os.getcwd()}/assets/test/{subject2}/{' '.join(prompt.split(' ')[2:])}.png"

							output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
							eval.postprocess(output, output_name, save_attn=save_attn)


def test_selected_mild_conflict(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = f"{os.getcwd()}/test/output/selected/mild_conflict",
	alpha_mask: str = "1",
	save_attn: bool = False,
	seed_idx: Optional[int] = None,
	for_loop_idx: Optional[int] = None,
	init_subject_idx: Optional[int] = None,
):
	eval.set_output_dir(output_dir)

	for i, seed in tqdm(enumerate(seeds), desc="For all seeds"):
		if seed_idx is not None and i != seed_idx:
			continue

		if for_loop_idx is None or for_loop_idx == 0:
					for situation in selected['mild']:
						for subject in situation['subjects']:
							prompt = situation['prompt'].format(subject=subject)
							mask_prompt = situation['mask_prompt']
							focus_tokens = situation['focus_tokens']
							reference = f"{os.getcwd()}/assets/test/{situation['ref']}"

							output, output_name = inference(prompt, reference, seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens)
							eval.postprocess(output, output_name, save_attn=save_attn)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
	parser.add_argument('--control', type=str, required=True, help="depth | pose | canny")
	parser.add_argument('--model', type=str, default="ControlAttend", help="ControlNet | SmartControl | ControlAttend")
	parser.add_argument('--alpha_mask', nargs="*", type=float, default=[1], help="Mask applied on inferred alpha. [1, 0, 0, 0] means only upper left is used with 1. None uses SmartControl's inferred alpha_mask.")
	parser.add_argument('--seed_idx', type=int, default=None)
	parser.add_argument('--for_loop_idx', type=int, default=None)
	parser.add_argument('--init_subject_idx', type=int, default=None)

	args = parser.parse_args()

	args.modelType = ModelType.str2enum(args.model)

	return args

def main():
	args = parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

	eval = EvalModel(args.control)
	inference = eval.get_inference_func(args.modelType)

	# test_no_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask, seed_idx=args.seed_idx, for_loop_idx=args.for_loop_idx)
	# test_mild_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask, seed_idx=args.seed_idx, for_loop_idx=args.for_loop_idx)
	# test_significant_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask, seed_idx=args.seed_idx, for_loop_idx=args.for_loop_idx, init_subject_idx=args.init_subject_idx)
	# test_selected_significant_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask, seed_idx=args.seed_idx, for_loop_idx=args.for_loop_idx, init_subject_idx=args.init_subject_idx)
	test_selected_mild_conflict(eval=eval, inference=inference, alpha_mask=args.alpha_mask, seed_idx=args.seed_idx, for_loop_idx=args.for_loop_idx, init_subject_idx=args.init_subject_idx)
if __name__ == "__main__":
    main()
