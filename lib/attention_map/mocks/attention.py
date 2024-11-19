
from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
from einops import rearrange


def attn_call(
		self,
		attn: Attention,
		hidden_states: torch.Tensor,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		temb: Optional[torch.Tensor] = None,
		height: int = None,
		width: int = None,
		timestep: Optional[torch.Tensor] = None,
		*args,
		**kwargs,
	) -> torch.Tensor:
		r"""
			To save attention score as an image, height parameter should be added.
		"""
		if len(args) > 0 or kwargs.get("scale", None) is not None:
			deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
			deprecate("scale", "1.0.0", deprecation_message)

		residual = hidden_states

		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)

		input_ndim = hidden_states.ndim

		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)
		attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)
		####################################################################################################
		# attention score should be saved as an image
		if hasattr(self, "store_attn_map"):
			self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height)
			self.timestep = int(timestep.item())
		####################################################################################################
		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

		if attn.residual_connection:
			hidden_states = hidden_states + residual

		hidden_states = hidden_states / attn.rescale_output_factor

		return hidden_states
