import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models import (AutoencoderKL, ControlNetModel, ImageProjection,
                              UNet2DConditionModel)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet import \
    retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_output import \
    StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from PIL import Image, ImageChops
from torchvision.transforms import ToPILImage
from transformers import (CLIPImageProcessor, CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection)

from lib import (COND_BLOCKS, AttnSaveOptions, assert_path, default_option,
                 save_attention_maps, tokenize_and_mark_prompts)

from .types import AttnDiffTrgtTokens

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class SmartControlPipeline(StableDiffusionControlNetPipeline):
	def __init__(
		self,
		vae: AutoencoderKL,
		text_encoder: CLIPTextModel,
		tokenizer: CLIPTokenizer,
		unet: UNet2DConditionModel,
		controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
		scheduler: KarrasDiffusionSchedulers,
		safety_checker: StableDiffusionSafetyChecker,
		feature_extractor: CLIPImageProcessor,
		image_encoder: CLIPVisionModelWithProjection = None,
		requires_safety_checker: bool = True,
		ignore_special_tkns: bool = True,
		diff_threshold: float = 0.0
	):
		super().__init__(
			vae=vae,
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			unet=unet,
			controlnet=controlnet,
			scheduler=scheduler,
			safety_checker=safety_checker,
			feature_extractor=feature_extractor,
			image_encoder=image_encoder,
			requires_safety_checker=requires_safety_checker
		)
		self.ignore_special_tkns = ignore_special_tkns
		self.diff_threshold = diff_threshold

	def control_branch_forward(
		self,
		sample: torch.FloatTensor,
		timestep: Union[torch.Tensor, float, int],
		encoder_hidden_states: torch.Tensor,
		controlnet_cond: torch.FloatTensor,
		conditioning_scale: float = 1.0,
		guess_mode: bool = False,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		return_dict: bool = True,
		prompt: str = None,
		output_name: str = None,
		attn_options: AttnSaveOptions = default_option,
	):
		last_idx = len(self.tokenizer(prompt)['input_ids']) - 1

		if cross_attention_kwargs is None:
			cross_attention_kwargs = {'timestep' : timestep}
		else:
			cross_attention_kwargs['timestep'] = timestep

		if self.ignore_special_tkns:
			cross_attention_kwargs['token_last_idx'] = last_idx

		down_block_res_samples, mid_block_res_sample = self.controlnet(
			sample,
			timestep,
			encoder_hidden_states=encoder_hidden_states,
			controlnet_cond=controlnet_cond,
			conditioning_scale=conditioning_scale,
			cross_attention_kwargs=cross_attention_kwargs,
			guess_mode=guess_mode,
			return_dict=return_dict,
		)

		timestep_key = timestep.item()
		organized = save_attention_maps(
			{timestep_key: self.controlnet.attn_maps[timestep_key]},
			self.tokenizer,
			base_dir=f"log/attn_maps/{output_name}",
			prompts=[prompt],
			options=attn_options
		)
		del self.controlnet.attn_maps[timestep_key]

		return down_block_res_samples, mid_block_res_sample, organized

	def infer_alpha_mask(self, output_name: str, timestep: int, cond_prompt_attns: dict, gen_prompt_attns: dict, trgt_tokens: AttnDiffTrgtTokens):
		to_pil = ToPILImage()
		masks = {}
		save_dir = f"log/alpha_masks/inferred/{output_name}/{timestep}"

		assert_path(save_dir)

		cond_tokens = tokenize_and_mark_prompts(
			prompts=[trgt_tokens["cond"]],
			tokenizer=self.tokenizer,
			ignore_special_tokens=self.ignore_special_tkns
		)[0]
		gen_tokens = tokenize_and_mark_prompts(
			prompts=[trgt_tokens["gen"]],
			tokenizer=self.tokenizer,
			ignore_special_tokens=self.ignore_special_tkns
		)[0]

		def _filter_dict_with_key(d: dict, substr: str):
			return list(dict(filter(
				lambda item: substr in item[0],
				d.items()
			)).values())

		def _filter_attns(attns: dict, trgt_block: str, trgt_token: str):
			blocks = _filter_dict_with_key(attns, trgt_block)
			filtered_attns = [_filter_dict_with_key(block, trgt_token) for block in blocks]
			return [attn for filtered in filtered_attns for attn in filtered]

		def _aggregate_attns(attns: dict, trgt_block: str, tokens: List[str]):
			aggregated = None

			for token in tokens:
				attn = _filter_attns(attns=attns, trgt_block=trgt_block, trgt_token=token)
				block_avg_attn = np.array(attn).sum(axis=0) / len(attn)
				if aggregated is None:
					aggregated = block_avg_attn
				else:
					aggregated += block_avg_attn

			aggregated /= len(tokens)
			return torch.from_numpy(aggregated)

		for trgt_block in COND_BLOCKS:
			cond_attn = _aggregate_attns(cond_prompt_attns, trgt_block=trgt_block, tokens=cond_tokens)
			gen_attn = _aggregate_attns(gen_prompt_attns, trgt_block=trgt_block, tokens=gen_tokens)

			avg_diff = cond_attn - gen_attn

			masks[trgt_block] = 1 - avg_diff
			# masks[trgt_block] = gen_attn
			# masks[trgt_block][avg_diff > self.diff_threshold] = 0

			to_pil(masks[trgt_block]).save(f"{save_dir}/{trgt_block}-{trgt_tokens}.png")

		return masks

	@torch.no_grad()
	def __call__(
		self,
		prompt: Union[str, List[str]] = None,
		condition_prompt: Optional[str] = None,
		diff_phrases: Optional[Dict[str, List[str]]] = None,
		image: PipelineImageInput = None,
		output_name: str = None,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		timesteps: List[int] = None,
		guidance_scale: float = 7.5,
		negative_prompt: Optional[Union[str, List[str]]] = None,
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		latents: Optional[torch.FloatTensor] = None,
		prompt_embeds: Optional[torch.FloatTensor] = None,
		negative_prompt_embeds: Optional[torch.FloatTensor] = None,
		ip_adapter_image: Optional[PipelineImageInput] = None,
		output_type: Optional[str] = "pil",
		return_dict: bool = True,
		use_attn_diff: bool = False,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
		guess_mode: bool = False,
		control_guidance_start: Union[float, List[float]] = 0.0,
		control_guidance_end: Union[float, List[float]] = 1.0,
		clip_skip: Optional[int] = None,
		callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
		callback_on_step_end_tensor_inputs: List[str] = ["latents"],
		**kwargs,
	):
		r"""
		The call function to the pipeline for generation.

		Args:
			prompt (`str` or `List[str]`, *optional*):
				The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
			image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
					`List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
				The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
				specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
				accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
				and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
				`init`, images must be passed as a list such that each element of the list can be correctly batched for
				input to a single ControlNet.
			height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
				The height in pixels of the generated image.
			width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
				The width in pixels of the generated image.
			num_inference_steps (`int`, *optional*, defaults to 50):
				The number of denoising steps. More denoising steps usually lead to a higher quality image at the
				expense of slower inference.
			timesteps (`List[int]`, *optional*):
				Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
				in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
				passed will be used. Must be in descending order.
			guidance_scale (`float`, *optional*, defaults to 7.5):
				A higher guidance scale value encourages the model to generate images closely linked to the text
				`prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
			negative_prompt (`str` or `List[str]`, *optional*):
				The prompt or prompts to guide what to not include in image generation. If not defined, you need to
				pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
			num_images_per_prompt (`int`, *optional*, defaults to 1):
				The number of images to generate per prompt.
			eta (`float`, *optional*, defaults to 0.0):
				Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
				to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
			generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
				A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
				generation deterministic.
			latents (`torch.FloatTensor`, *optional*):
				Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
				generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
				tensor is generated by sampling using the supplied random `generator`.
			prompt_embeds (`torch.FloatTensor`, *optional*):
				Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
				provided, text embeddings are generated from the `prompt` input argument.
			negative_prompt_embeds (`torch.FloatTensor`, *optional*):
				Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
				not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
			ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
			output_type (`str`, *optional*, defaults to `"pil"`):
				The output format of the generated image. Choose between `PIL.Image` or `np.array`.
			return_dict (`bool`, *optional*, defaults to `True`):
				Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
				plain tuple.
			callback (`Callable`, *optional*):
				A function that calls every `callback_steps` steps during inference. The function is called with the
				following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
			callback_steps (`int`, *optional*, defaults to 1):
				The frequency at which the `callback` function is called. If not specified, the callback is called at
				every step.
			cross_attention_kwargs (`dict`, *optional*):
				A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
				[`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
			controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
				The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
				to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
				the corresponding scale as a list.
			guess_mode (`bool`, *optional*, defaults to `False`):
				The ControlNet encoder tries to recognize the content of the input image even if you remove all
				prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
			control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
				The percentage of total steps at which the ControlNet starts applying.
			control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
				The percentage of total steps at which the ControlNet stops applying.
			clip_skip (`int`, *optional*):
				Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
				the output of the pre-final layer will be used for computing the prompt embeddings.
			callback_on_step_end (`Callable`, *optional*):
				A function that calls at the end of each denoising steps during the inference. The function is called
				with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
				callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
				`callback_on_step_end_tensor_inputs`.
			callback_on_step_end_tensor_inputs (`List`, *optional*):
				The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
				will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
				`._callback_tensor_inputs` attribute of your pipeine class.

		Examples:

		Returns:
			[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
				If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
				otherwise a `tuple` is returned where the first element is a list with the generated images and the
				second element is a list of `bool`s indicating whether the corresponding generated image contains
				"not-safe-for-work" (nsfw) content.
		"""

		callback = kwargs.pop("callback", None)
		callback_steps = kwargs.pop("callback_steps", None)

		if callback is not None:
			deprecate(
				"callback",
				"1.0.0",
				"Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
			)
		if callback_steps is not None:
			deprecate(
				"callback_steps",
				"1.0.0",
				"Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
			)

		controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

		# align format for control guidance
		if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
			control_guidance_start = len(control_guidance_end) * [control_guidance_start]
		elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
			control_guidance_end = len(control_guidance_start) * [control_guidance_end]
		elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
			mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
			control_guidance_start, control_guidance_end = (
				mult * [control_guidance_start],
				mult * [control_guidance_end],
			)

		# 1. Check inputs. Raise error if not correct
		self.check_inputs(
			prompt,
			image,
			callback_steps,
			negative_prompt,
			prompt_embeds,
			negative_prompt_embeds,
			controlnet_conditioning_scale,
			control_guidance_start,
			control_guidance_end,
			callback_on_step_end_tensor_inputs,
		)

		self._guidance_scale = guidance_scale
		self._clip_skip = clip_skip
		self._cross_attention_kwargs = cross_attention_kwargs

		# 2. Define call parameters
		if prompt is not None and isinstance(prompt, str):
			batch_size = 1
		elif prompt is not None and isinstance(prompt, list):
			batch_size = len(prompt)
		else:
			batch_size = prompt_embeds.shape[0]

		device = self._execution_device

		if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
			controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

		global_pool_conditions = (
			controlnet.config.global_pool_conditions
			if isinstance(controlnet, ControlNetModel)
			else controlnet.nets[0].config.global_pool_conditions
		)
		guess_mode = guess_mode or global_pool_conditions

		# 3. Encode input prompt
		text_encoder_lora_scale = (
			self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
		)

		prompt_embeds, negative_prompt_embeds = self.encode_prompt(
			prompt,
			device,
			num_images_per_prompt,
			self.do_classifier_free_guidance,
			negative_prompt,
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_prompt_embeds,
			lora_scale=text_encoder_lora_scale,
			clip_skip=self.clip_skip,
		)

		################################################################################
		condition_prompt = condition_prompt or prompt
		cond_prompt_embeds, neg_cond_prompt_embeds = self.encode_prompt(
			condition_prompt,
			device,
			num_images_per_prompt,
			self.do_classifier_free_guidance,
			negative_prompt,
			prompt_embeds=None,
			negative_prompt_embeds=None,
			lora_scale=text_encoder_lora_scale,
			clip_skip=self.clip_skip,
		)
		################################################################################

		# For classifier free guidance, we need to do two forward passes.
		# Here we concatenate the unconditional and text embeddings into a single batch
		# to avoid doing two forward passes
		if self.do_classifier_free_guidance:
			prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
			##########################################################################################
			cond_prompt_embeds = torch.cat([neg_cond_prompt_embeds, cond_prompt_embeds])
			##########################################################################################

		if ip_adapter_image is not None:
			output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
			image_embeds, negative_image_embeds = self.encode_image(
				ip_adapter_image, device, num_images_per_prompt, output_hidden_state
			)
			if self.do_classifier_free_guidance:
				image_embeds = torch.cat([negative_image_embeds, image_embeds])

		# 4. Prepare image
		if isinstance(controlnet, ControlNetModel):
			image = self.prepare_image(
				image=image,
				width=width,
				height=height,
				batch_size=batch_size * num_images_per_prompt,
				num_images_per_prompt=num_images_per_prompt,
				device=device,
				dtype=controlnet.dtype,
				do_classifier_free_guidance=self.do_classifier_free_guidance,
				guess_mode=guess_mode,
			)
			height, width = image.shape[-2:]
		elif isinstance(controlnet, MultiControlNetModel):
			images = []

			for image_ in image:
				image_ = self.prepare_image(
					image=image_,
					width=width,
					height=height,
					batch_size=batch_size * num_images_per_prompt,
					num_images_per_prompt=num_images_per_prompt,
					device=device,
					dtype=controlnet.dtype,
					do_classifier_free_guidance=self.do_classifier_free_guidance,
					guess_mode=guess_mode,
				)

				images.append(image_)

			image = images
			height, width = image[0].shape[-2:]
		else:
			assert False

		# 5. Prepare timesteps
		timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
		self._num_timesteps = len(timesteps)

		# 6. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents(
			batch_size * num_images_per_prompt,
			num_channels_latents,
			height,
			width,
			prompt_embeds.dtype,
			device,
			generator,
			latents,
		)

		# 6.5 Optionally get Guidance Scale Embedding
		timestep_cond = None
		if self.unet.config.time_cond_proj_dim is not None:
			guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
			timestep_cond = self.get_guidance_scale_embedding(
				guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
			).to(device=device, dtype=latents.dtype)

		# 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 7.1 Add image embeds for IP-Adapter
		added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

		# 7.2 Create tensor stating which controlnets to keep
		controlnet_keep = []
		for i in range(len(timesteps)):
			keeps = [
				1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
				for s, e in zip(control_guidance_start, control_guidance_end)
			]
			controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

		# 8. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		is_unet_compiled = is_compiled_module(self.unet)
		is_controlnet_compiled = is_compiled_module(self.controlnet)
		is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):
				# Relevant thread:
				# https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
				if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
					torch._inductor.cudagraph_mark_step_begin()
				# expand the latents if we are doing classifier free guidance
				latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
				latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

				# controlnet(s) inference
				if guess_mode and self.do_classifier_free_guidance:
					# Infer ControlNet only for the conditional batch.
					control_model_input = latents
					control_model_input = self.scheduler.scale_model_input(control_model_input, t)
					controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
				else:
					control_model_input = latent_model_input
					controlnet_prompt_embeds = prompt_embeds

				if isinstance(controlnet_keep[i], list):
					cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
				else:
					controlnet_cond_scale = controlnet_conditioning_scale
					if isinstance(controlnet_cond_scale, list):
						controlnet_cond_scale = controlnet_cond_scale[0]
					cond_scale = controlnet_cond_scale * controlnet_keep[i]

				############################################################################
				down_block_res_samples, mid_block_res_sample, gen_prompt_attn = self.control_branch_forward(
					control_model_input,
					t,
					prompt=prompt,
					encoder_hidden_states=controlnet_prompt_embeds,
					controlnet_cond=image,
					conditioning_scale=cond_scale,
					guess_mode=guess_mode,
					cross_attention_kwargs=self.cross_attention_kwargs,
					return_dict=False,
					output_name=output_name,
					attn_options={
						"prefix": "",
						"return_dict": True,
						"ignore_special_tkns": self.ignore_special_tkns
					}
				)
				inferred_masks = None

				if use_attn_diff is True:
					_, _2, cond_prompt_attn = self.control_branch_forward(
						control_model_input,
						t,
						prompt=condition_prompt,
						encoder_hidden_states=cond_prompt_embeds,
						controlnet_cond=image,
						conditioning_scale=cond_scale,
						guess_mode=guess_mode,
						return_dict=False,
						output_name=output_name,
						attn_options={
							"prefix": "sub-",
							"return_dict": True,
							"ignore_special_tkns": self.ignore_special_tkns
						}
					)

					inferred_masks = self.infer_alpha_mask(
						output_name=output_name,
						timestep=t,
						cond_prompt_attns=cond_prompt_attn,
						gen_prompt_attns=gen_prompt_attn,
						trgt_tokens=diff_phrases
					)
				############################################################################

				if guess_mode and self.do_classifier_free_guidance:
					# Infered ControlNet only for the conditional batch.
					# To apply the output of ControlNet to both the unconditional and conditional batches,
					# add 0 to the unconditional batch to keep it unchanged.
					down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
					mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

				# predict the noise residual
				noise_pred = self.unet(
					latent_model_input,
					t,
					encoder_hidden_states=prompt_embeds,
					timestep_cond=timestep_cond,
					cross_attention_kwargs=self.cross_attention_kwargs,
					down_block_additional_residuals=down_block_res_samples,
					mid_block_additional_residual=mid_block_res_sample,
					inferred_masks=inferred_masks,
					added_cond_kwargs=added_cond_kwargs,
					return_dict=False,
				)[0]

				# perform guidance
				if self.do_classifier_free_guidance:
					noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
					noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

				# compute the previous noisy sample x_t -> x_t-1
				latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

				if callback_on_step_end is not None:
					callback_kwargs = {}
					for k in callback_on_step_end_tensor_inputs:
						callback_kwargs[k] = locals()[k]
					callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

					latents = callback_outputs.pop("latents", latents)
					prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
					negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						step_idx = i // getattr(self.scheduler, "order", 1)
						callback(step_idx, t, latents)

		# If we do sequential model offloading, let's offload unet and controlnet
		# manually for max memory savings
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.unet.to("cpu")
			self.controlnet.to("cpu")
			torch.cuda.empty_cache()

		if not output_type == "latent":
			image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
				0
			]
			image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
		else:
			image = latents
			has_nsfw_concept = None

		if has_nsfw_concept is None:
			do_denormalize = [True] * image.shape[0]
		else:
			do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

		image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

		# Offload all models
		self.maybe_free_model_hooks()

		if not return_dict:
			return (image, has_nsfw_concept)

		return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
