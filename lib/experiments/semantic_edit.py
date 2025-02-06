import argparse
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class SemanticStableDiffusionPipelineArgs:
	editing_prompt: List[str]
	reverse_edit_direction: List[bool]
	edit_warmup_steps: List[int]
	edit_guidance_scale: List[int]
	edit_threshold: List[float]
	edit_weights: List[float]
	edit_mom_scale: float
	edit_mom_beta: float

	def init_call_params(self):
		if self.editing_prompt:
			self.enable_edit_guidance = True
			self.enabled_editing_prompts = len(self.editing_prompt)
		else:
			self.enable_edit_guidance = False
			self.enabled_editing_prompts = 0

class EditGuidance:
	def __init__(self, device: torch.device, edit_args: SemanticStableDiffusionPipelineArgs):
		self.device = device
		self.edit_args = edit_args

		self.edit_momentum = None

		self.edit_estimates = None
		self.sem_guidance = None

	def init_components(self, num_inference_steps: int, noise_pred_edit_concepts, noise_pred_text, noise_guidance):
		if self.edit_estimates is None:
			self.edit_estimates = torch.zeros(
				(num_inference_steps + 1, len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
			)

		if self.sem_guidance is None:
			self.sem_guidance = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))

		if self.edit_momentum is None:
			self.edit_momentum = torch.zeros_like(noise_guidance)

		self.concept_weights = torch.zeros(
			(len(noise_pred_edit_concepts), noise_guidance.shape[0]),
			device=self.device,
			dtype=noise_guidance.dtype,
		)
		self.noise_guidance_edit = torch.zeros(
			(len(noise_pred_edit_concepts), *noise_guidance.shape),
			device=self.device,
			dtype=noise_guidance.dtype,
		)

	def _warmup(self, infer_step: int, noise_pred_edit_concepts, noise_pred_uncond, noise_guidance):
		warmup_inds = []
		for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
			self.edit_estimates[infer_step, c] = noise_pred_edit_concept
			if isinstance(self.edit_args.edit_guidance_scale, list):
				edit_guidance_scale_c = self.edit_args.edit_guidance_scale[c]
			else:
				edit_guidance_scale_c = self.edit_args.edit_guidance_scale

			if isinstance(self.edit_args.edit_threshold, list):
				edit_threshold_c = self.edit_args.edit_threshold[c]
			else:
				edit_threshold_c = self.edit_args.edit_threshold
			if isinstance(self.edit_args.reverse_edit_direction, list):
				reverse_editing_direction_c = self.edit_args.reverse_edit_direction[c]
			else:
				reverse_editing_direction_c = self.edit_args.reverse_edit_direction
			if self.edit_args.edit_weights:
				edit_weight_c = self.edit_args.edit_weights[c]
			else:
				edit_weight_c = 1.0
			if isinstance(self.edit_args.edit_warmup_steps, list):
				edit_warmup_steps_c = self.edit_args.edit_warmup_steps[c]
			else:
				edit_warmup_steps_c = self.edit_args.edit_warmup_steps

			if infer_step >= edit_warmup_steps_c:
				warmup_inds.append(c)

			noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
			# tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
			tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

			tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
			if reverse_editing_direction_c:
				noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
			self.concept_weights[c, :] = tmp_weights

			noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

			# torch.quantile function expects float32
			if noise_guidance_edit_tmp.dtype == torch.float32:
				tmp = torch.quantile(
					torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2),
					edit_threshold_c,
					dim=2,
					keepdim=False,
				)
			else:
				tmp = torch.quantile(
					torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2).to(torch.float32),
					edit_threshold_c,
					dim=2,
					keepdim=False,
				).to(noise_guidance_edit_tmp.dtype)

			noise_guidance_edit_tmp = torch.where(
				torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
				noise_guidance_edit_tmp,
				torch.zeros_like(noise_guidance_edit_tmp),
			)
			self.noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

			# noise_guidance_edit = noise_guidance_edit + noise_guidance_edit_tmp

		warmup_inds = torch.tensor(warmup_inds).to(self.device)
		self._warmup_postprocess(infer_step, noise_pred_edit_concepts, warmup_inds, noise_guidance)
		return warmup_inds

	def _warmup_postprocess(self, infer_step, noise_pred_edit_concepts, warmup_inds, noise_guidance):
		if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
			concept_weights_tmp = self.concept_weights.to("cpu")  # Offload to cpu
			noise_guidance_edit_tmp: torch.Tensor = self.noise_guidance_edit.to("cpu")

			concept_weights_tmp = torch.index_select(concept_weights_tmp.to(self.device), 0, warmup_inds)
			concept_weights_tmp = torch.where(
				concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
			)
			concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
			# concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

			noise_guidance_edit_tmp = torch.index_select(
				noise_guidance_edit_tmp.to(self.device), 0, warmup_inds
			)
			noise_guidance_edit_tmp = torch.einsum(
				"cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
			)
			noise_guidance_edit_tmp = noise_guidance_edit_tmp
			noise_guidance += noise_guidance_edit_tmp

			self.sem_guidance[infer_step] = noise_guidance_edit_tmp.detach().cpu()

			del noise_guidance_edit_tmp
			del concept_weights_tmp


	def calc_guidance(self, infer_step: int, noise_pred_edit_concepts, noise_pred_uncond, noise_guidance):
		warmup_inds = self._warmup(infer_step, noise_pred_edit_concepts, noise_pred_uncond, noise_guidance)

		self.concept_weights = torch.where(
			self.concept_weights < 0, torch.zeros_like(self.concept_weights), self.concept_weights
		)

		self.concept_weights = torch.nan_to_num(self.concept_weights)

		self.noise_guidance_edit = torch.einsum("cb,cbijk->bijk", self.concept_weights, self.noise_guidance_edit)

		self.noise_guidance_edit = self.noise_guidance_edit + self.edit_args.edit_mom_scale * self.edit_momentum

		self.edit_momentum = self.edit_args.edit_mom_beta * self.edit_momentum + (1 - self.edit_args.edit_mom_beta) * self.noise_guidance_edit

		if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
			noise_guidance = noise_guidance + self.noise_guidance_edit
			self.sem_guidance[infer_step] = self.noise_guidance_edit.detach().cpu()


def _check_edit_args(args: argparse.Namespace):
	if args.editing_prompt is not None:
		args.edit_args = SemanticStableDiffusionPipelineArgs(
			editing_prompt=args.editing_prompt,
			reverse_edit_direction=args.reverse_edit_direction,
			edit_warmup_steps=args.edit_warmup_steps,
			edit_guidance_scale=args.edit_guidance_scale,
			edit_threshold=args.edit_threshold,
			edit_weights=args.edit_weights,
			edit_mom_scale=args.edit_mom_scale,
			edit_mom_beta=args.edit_mom_beta
		)
		args.edit_args.init_call_params()

def _SEGA_image_name(args: argparse.Namespace):
	suffix = ""

	if args.editing_prompt is not None:
		suffix += "-SEGA"
		for i, prompt in enumerate(args.editing_prompt):
			suffix += f" {'-' if args.reverse_edit_direction[i] else '+'}{args.edit_guidance_scale[i]}{prompt}"

	return suffix
