import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import (USE_PEFT_BACKEND, deprecate, is_torch_version,
                             logger, scale_lora_layers, unscale_lora_layers)


def UNet2DConditionModelForward(
	self,
	sample: torch.Tensor,
	timestep: Union[torch.Tensor, float, int],
	encoder_hidden_states: torch.Tensor,
	class_labels: Optional[torch.Tensor] = None,
	timestep_cond: Optional[torch.Tensor] = None,
	attention_mask: Optional[torch.Tensor] = None,
	cross_attention_kwargs: Optional[Dict[str, Any]] = None,
	added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
	down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
	mid_block_additional_residual: Optional[torch.Tensor] = None,
	down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
	encoder_attention_mask: Optional[torch.Tensor] = None,
	return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
	r"""
	The [`UNet2DConditionModel`] forward method.

	Args:
		sample (`torch.Tensor`):
			The noisy input tensor with the following shape `(batch, channel, height, width)`.
		timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
		encoder_hidden_states (`torch.Tensor`):
			The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
		class_labels (`torch.Tensor`, *optional*, defaults to `None`):
			Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
		timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
			Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
			through the `self.time_embedding` layer to obtain the timestep embeddings.
		attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
			An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
			is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
			negative values to the attention scores corresponding to "discard" tokens.
		cross_attention_kwargs (`dict`, *optional*):
			A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
			`self.processor` in
			[diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
		added_cond_kwargs: (`dict`, *optional*):
			A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
			are passed along to the UNet blocks.
		down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
			A tuple of tensors that if specified are added to the residuals of down unet blocks.
		mid_block_additional_residual: (`torch.Tensor`, *optional*):
			A tensor that if specified is added to the residual of the middle unet block.
		down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
			additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
		encoder_attention_mask (`torch.Tensor`):
			A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
			`True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
			which adds large negative values to the attention scores corresponding to "discard" tokens.
		return_dict (`bool`, *optional*, defaults to `True`):
			Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
			tuple.

	Returns:
		[`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
			If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
			otherwise a `tuple` is returned where the first element is the sample tensor.
	"""
	# By default samples have to be AT least a multiple of the overall upsampling factor.
	# The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
	# However, the upsampling interpolation output size can be forced to fit any upsampling size
	# on the fly if necessary.
	default_overall_up_factor = 2**self.num_upsamplers

	# upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
	forward_upsample_size = False
	upsample_size = None

	for dim in sample.shape[-2:]:
		if dim % default_overall_up_factor != 0:
			# Forward upsample size to force interpolation output size.
			forward_upsample_size = True
			break

	# ensure attention_mask is a bias, and give it a singleton query_tokens dimension
	# expects mask of shape:
	#   [batch, key_tokens]
	# adds singleton query_tokens dimension:
	#   [batch,                    1, key_tokens]
	# this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
	#   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
	#   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
	if attention_mask is not None:
		# assume that mask is expressed as:
		#   (1 = keep,      0 = discard)
		# convert mask into a bias that can be added to attention scores:
		#       (keep = +0,     discard = -10000.0)
		attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
		attention_mask = attention_mask.unsqueeze(1)

	# convert encoder_attention_mask to a bias the same way we do for attention_mask
	if encoder_attention_mask is not None:
		encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
		encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

	# 0. center input if necessary
	if self.config.center_input_sample:
		sample = 2 * sample - 1.0

	# 1. time
	t_emb = self.get_time_embed(sample=sample, timestep=timestep)
	emb = self.time_embedding(t_emb, timestep_cond)
	aug_emb = None

	class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
	if class_emb is not None:
		if self.config.class_embeddings_concat:
			emb = torch.cat([emb, class_emb], dim=-1)
		else:
			emb = emb + class_emb

	aug_emb = self.get_aug_embed(
		emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
	)
	if self.config.addition_embed_type == "image_hint":
		aug_emb, hint = aug_emb
		sample = torch.cat([sample, hint], dim=1)

	emb = emb + aug_emb if aug_emb is not None else emb

	if self.time_embed_act is not None:
		emb = self.time_embed_act(emb)

	encoder_hidden_states = self.process_encoder_hidden_states(
		encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
	)

	# 2. pre-process
	sample = self.conv_in(sample)

	# 2.5 GLIGEN position net
	if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
		cross_attention_kwargs = cross_attention_kwargs.copy()
		gligen_args = cross_attention_kwargs.pop("gligen")
		cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

	# 3. down
	# we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
	# to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
	################################################################################
	if cross_attention_kwargs is None:
		cross_attention_kwargs = {'timestep' : timestep}
	else:
		cross_attention_kwargs['timestep'] = timestep
	################################################################################


	if cross_attention_kwargs is not None:
		cross_attention_kwargs = cross_attention_kwargs.copy()
		lora_scale = cross_attention_kwargs.pop("scale", 1.0)
	else:
		lora_scale = 1.0

	if USE_PEFT_BACKEND:
		# weight the lora layers by setting `lora_scale` for each PEFT layer
		scale_lora_layers(self, lora_scale)

	is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
	# using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
	is_adapter = down_intrablock_additional_residuals is not None
	# maintain backward compatibility for legacy usage, where
	#       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
	#       but can only use one or the other
	if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
		deprecate(
			"T2I should not use down_block_additional_residuals",
			"1.3.0",
			"Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
					and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
					for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
			standard_warn=False,
		)
		down_intrablock_additional_residuals = down_block_additional_residuals
		is_adapter = True

	down_block_res_samples = (sample,)
	for downsample_block in self.down_blocks:
		if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
			# For t2i-adapter CrossAttnDownBlock2D
			additional_residuals = {}
			if is_adapter and len(down_intrablock_additional_residuals) > 0:
				additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

			sample, res_samples = downsample_block(
				hidden_states=sample,
				temb=emb,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=attention_mask,
				cross_attention_kwargs=cross_attention_kwargs,
				encoder_attention_mask=encoder_attention_mask,
				**additional_residuals,
			)
		else:
			sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
			if is_adapter and len(down_intrablock_additional_residuals) > 0:
				sample += down_intrablock_additional_residuals.pop(0)

		down_block_res_samples += res_samples

	if is_controlnet:
		new_down_block_res_samples = ()

		for down_block_res_sample, down_block_additional_residual in zip(
			down_block_res_samples, down_block_additional_residuals
		):
			down_block_res_sample = down_block_res_sample + down_block_additional_residual
			new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

		down_block_res_samples = new_down_block_res_samples

	# 4. mid
	if self.mid_block is not None:
		if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
			sample = self.mid_block(
				sample,
				emb,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=attention_mask,
				cross_attention_kwargs=cross_attention_kwargs,
				encoder_attention_mask=encoder_attention_mask,
			)
		else:
			sample = self.mid_block(sample, emb)

		# To support T2I-Adapter-XL
		if (
			is_adapter
			and len(down_intrablock_additional_residuals) > 0
			and sample.shape == down_intrablock_additional_residuals[0].shape
		):
			sample += down_intrablock_additional_residuals.pop(0)

	if is_controlnet:
		sample = sample + mid_block_additional_residual

	# 5. up
	for i, upsample_block in enumerate(self.up_blocks):
		is_final_block = i == len(self.up_blocks) - 1

		res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
		down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

		# if we have not reached the final block and need to forward the
		# upsample size, we do it here
		if not is_final_block and forward_upsample_size:
			upsample_size = down_block_res_samples[-1].shape[2:]

		if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
			sample = upsample_block(
				hidden_states=sample,
				temb=emb,
				res_hidden_states_tuple=res_samples,
				encoder_hidden_states=encoder_hidden_states,
				cross_attention_kwargs=cross_attention_kwargs,
				upsample_size=upsample_size,
				attention_mask=attention_mask,
				encoder_attention_mask=encoder_attention_mask,
			)
		else:
			sample = upsample_block(
				hidden_states=sample,
				temb=emb,
				res_hidden_states_tuple=res_samples,
				upsample_size=upsample_size,
			)

	# 6. post-process
	if self.conv_norm_out:
		sample = self.conv_norm_out(sample)
		sample = self.conv_act(sample)
	sample = self.conv_out(sample)

	if USE_PEFT_BACKEND:
		# remove `lora_scale` from each PEFT layer
		unscale_lora_layers(self, lora_scale)

	if not return_dict:
		return (sample,)

	return UNet2DConditionOutput(sample=sample)

def Transformer2DModelForward(
	self,
	hidden_states: torch.Tensor,
	encoder_hidden_states: Optional[torch.Tensor] = None,
	timestep: Optional[torch.LongTensor] = None,
	added_cond_kwargs: Dict[str, torch.Tensor] = None,
	class_labels: Optional[torch.LongTensor] = None,
	cross_attention_kwargs: Dict[str, Any] = None,
	attention_mask: Optional[torch.Tensor] = None,
	encoder_attention_mask: Optional[torch.Tensor] = None,
	return_dict: bool = True,
):
	"""
	The [`Transformer2DModel`] forward method.

	Args:
		hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
			Input `hidden_states`.
		encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
			Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
			self-attention.
		timestep ( `torch.LongTensor`, *optional*):
			Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
		class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
			Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
			`AdaLayerZeroNorm`.
		cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
			A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
			`self.processor` in
			[diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
		attention_mask ( `torch.Tensor`, *optional*):
			An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
			is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
			negative values to the attention scores corresponding to "discard" tokens.
		encoder_attention_mask ( `torch.Tensor`, *optional*):
			Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

				* Mask `(batch, sequence_length)` True = keep, False = discard.
				* Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

			If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
			above. This bias will be added to the cross-attention scores.
		return_dict (`bool`, *optional*, defaults to `True`):
			Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
			tuple.

	Returns:
		If `return_dict` is True, an [`~models.transformers.transformer_2d.Transformer2DModelOutput`] is returned,
		otherwise a `tuple` where the first element is the sample tensor.
	"""
	if cross_attention_kwargs is not None:
		if cross_attention_kwargs.get("scale", None) is not None:
			logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
	# ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
	#   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
	#   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
	# expects mask of shape:
	#   [batch, key_tokens]
	# adds singleton query_tokens dimension:
	#   [batch,                    1, key_tokens]
	# this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
	#   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
	#   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
	if attention_mask is not None and attention_mask.ndim == 2:
		# assume that mask is expressed as:
		#   (1 = keep,      0 = discard)
		# convert mask into a bias that can be added to attention scores:
		#       (keep = +0,     discard = -10000.0)
		attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
		attention_mask = attention_mask.unsqueeze(1)

	# convert encoder_attention_mask to a bias the same way we do for attention_mask
	if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
		encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
		encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

	# Retrieve lora scale.
	lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

	# 1. Input
	if self.is_input_continuous:
		batch, _, height, width = hidden_states.shape
		residual = hidden_states
		hidden_states = self.norm(hidden_states)

		if not self.use_linear_projection:
			hidden_states = (
				self.proj_in(hidden_states, scale=lora_scale)
				if not USE_PEFT_BACKEND
				else self.proj_in(hidden_states)
			)
			inner_dim = hidden_states.shape[1]
			hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
		else:
			inner_dim = hidden_states.shape[1]
			hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
			hidden_states = (
				self.proj_in(hidden_states, scale=lora_scale)
				if not USE_PEFT_BACKEND
				else self.proj_in(hidden_states)
			)
	elif self.is_input_vectorized:
		hidden_states = self.latent_image_embedding(hidden_states)
	elif self.is_input_patches:
		height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
		hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
			hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
		)

	####################################################################################################
	cross_attention_kwargs['height'] = height
	cross_attention_kwargs['width'] = width
	####################################################################################################

	# 2. Blocks
	for block in self.transformer_blocks:
		if self.training and self.gradient_checkpointing:

			def create_custom_forward(module, return_dict=None):
				def custom_forward(*inputs):
					if return_dict is not None:
						return module(*inputs, return_dict=return_dict)
					else:
						return module(*inputs)

				return custom_forward

			ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
			hidden_states = torch.utils.checkpoint.checkpoint(
				create_custom_forward(block),
				hidden_states,
				attention_mask,
				encoder_hidden_states,
				encoder_attention_mask,
				timestep,
				cross_attention_kwargs,
				class_labels,
				**ckpt_kwargs,
			)
		else:
			hidden_states = block(
				hidden_states,
				attention_mask=attention_mask,
				encoder_hidden_states=encoder_hidden_states,
				encoder_attention_mask=encoder_attention_mask,
				timestep=timestep,
				cross_attention_kwargs=cross_attention_kwargs,
				class_labels=class_labels,
			)

	# 3. Output
	if self.is_input_continuous:
			if not self.use_linear_projection:
				hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
				hidden_states = (
					self.proj_out(hidden_states, scale=lora_scale)
					if not USE_PEFT_BACKEND
					else self.proj_out(hidden_states)
				)
			else:
				hidden_states = (
					self.proj_out(hidden_states, scale=lora_scale)
					if not USE_PEFT_BACKEND
					else self.proj_out(hidden_states)
				)
				hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

			output = hidden_states + residual
	elif self.is_input_vectorized:
		output = self._get_output_for_vectorized_inputs(hidden_states)
	elif self.is_input_patches:
		output = self._get_output_for_patched_inputs(
			hidden_states=hidden_states,
			timestep=timestep,
			class_labels=class_labels,
			embedded_timestep=embedded_timestep,
			height=height,
			width=width,
		)

	if not return_dict:
		return (output,)

	return Transformer2DModelOutput(sample=output)

def BasicTransformerBlockForward(
	self,
	hidden_states: torch.Tensor,
	attention_mask: Optional[torch.Tensor] = None,
	encoder_hidden_states: Optional[torch.Tensor] = None,
	encoder_attention_mask: Optional[torch.Tensor] = None,
	timestep: Optional[torch.LongTensor] = None,
	cross_attention_kwargs: Dict[str, Any] = None,
	class_labels: Optional[torch.LongTensor] = None,
	added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
	if cross_attention_kwargs is not None:
		if cross_attention_kwargs.get("scale", None) is not None:
			logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

	# Notice that normalization is always applied before the real computation in the following blocks.
	# 0. Self-Attention
	batch_size = hidden_states.shape[0]

	if self.norm_type == "ada_norm":
		norm_hidden_states = self.norm1(hidden_states, timestep)
	elif self.norm_type == "ada_norm_zero":
		norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
			hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
		)
	elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
		norm_hidden_states = self.norm1(hidden_states)
	elif self.norm_type == "ada_norm_continuous":
		norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
	elif self.norm_type == "ada_norm_single":
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
			self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
		).chunk(6, dim=1)
		norm_hidden_states = self.norm1(hidden_states)
		norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
		norm_hidden_states = norm_hidden_states.squeeze(1)
	else:
		raise ValueError("Incorrect norm used")

	if self.pos_embed is not None:
		norm_hidden_states = self.pos_embed(norm_hidden_states)

	# 1. Prepare GLIGEN inputs
	cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
	gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

	################################################################################
	attn_parameters = set(inspect.signature(self.attn1.processor.__call__).parameters.keys())
	################################################################################

	attn_output = self.attn1(
		norm_hidden_states,
		encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
		attention_mask=attention_mask,
		################################################################################
		**{k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters},
		################################################################################
	)
	if self.norm_type == "ada_norm_zero":
		attn_output = gate_msa.unsqueeze(1) * attn_output
	elif self.norm_type == "ada_norm_single":
		attn_output = gate_msa * attn_output

	hidden_states = attn_output + hidden_states
	if hidden_states.ndim == 4:
		hidden_states = hidden_states.squeeze(1)

	# 1.2 GLIGEN Control
	if gligen_kwargs is not None:
		hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

	# 3. Cross-Attention
	if self.attn2 is not None:
		if self.norm_type == "ada_norm":
			norm_hidden_states = self.norm2(hidden_states, timestep)
		elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
			norm_hidden_states = self.norm2(hidden_states)
		elif self.norm_type == "ada_norm_single":
			# For PixArt norm2 isn't applied here:
			# https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
			norm_hidden_states = hidden_states
		elif self.norm_type == "ada_norm_continuous":
			norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
		else:
			raise ValueError("Incorrect norm")

		if self.pos_embed is not None and self.norm_type != "ada_norm_single":
			norm_hidden_states = self.pos_embed(norm_hidden_states)

		attn_output = self.attn2(
			norm_hidden_states,
			encoder_hidden_states=encoder_hidden_states,
			attention_mask=encoder_attention_mask,
			**cross_attention_kwargs,
		)
		hidden_states = attn_output + hidden_states

	# 4. Feed-forward
	# i2vgen doesn't have this norm 🤷‍♂️
	if self.norm_type == "ada_norm_continuous":
		norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
	elif not self.norm_type == "ada_norm_single":
		norm_hidden_states = self.norm3(hidden_states)

	if self.norm_type == "ada_norm_zero":
		norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

	if self.norm_type == "ada_norm_single":
		norm_hidden_states = self.norm2(hidden_states)
		norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

	if self._chunk_size is not None:
		# "feed_forward_chunk_size" can be used to save memory
		ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
	else:
		ff_output = self.ff(norm_hidden_states)

	if self.norm_type == "ada_norm_zero":
		ff_output = gate_mlp.unsqueeze(1) * ff_output
	elif self.norm_type == "ada_norm_single":
		ff_output = gate_mlp * ff_output

	hidden_states = ff_output + hidden_states
	if hidden_states.ndim == 4:
		hidden_states = hidden_states.squeeze(1)

	return hidden_states
