# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
	# Efficient implementation equivalent to the following:
	L, S = query.size(-2), key.size(-2)
	scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
	attn_bias = torch.zeros(L, S, dtype=query.dtype)
	if is_causal:
		assert attn_mask is None
		temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
		attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
		attn_bias.to(query.dtype)

	if attn_mask is not None:
		if attn_mask.dtype == torch.bool:
			attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
		else:
			attn_bias += attn_mask
	attn_weight = query @ key.transpose(-2, -1) * scale_factor
	attn_weight += attn_bias.to(attn_weight.device)
	attn_weight = torch.softmax(attn_weight, dim=-1)

	return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight



class AttnProcessor(nn.Module):
	r"""
	Default processor for performing attention-related computations.
	"""

	def __init__(
		self,
		hidden_size=None,
		cross_attention_dim=None,
	):
		super().__init__()

	def __call__(
		self,
		attn,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
	):
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


class IPAttnProcessor(nn.Module):
	r"""
	Attention processor for IP-Adapater.
	Args:
		hidden_size (`int`):
			The hidden size of the attention layer.
		cross_attention_dim (`int`):
			The number of channels in the `encoder_hidden_states`.
		scale (`float`, defaults to 1.0):
			the weight scale of image prompt.
		num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
			The context length of the image features.
	"""

	def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
		super().__init__()

		self.hidden_size = hidden_size
		self.cross_attention_dim = cross_attention_dim
		self.scale = scale
		self.num_tokens = num_tokens

		self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
		self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

	def __call__(
		self,
		attn,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
	):
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
		else:
			# get encoder_hidden_states, ip_hidden_states
			end_pos = encoder_hidden_states.shape[1] - self.num_tokens
			encoder_hidden_states, ip_hidden_states = (
				encoder_hidden_states[:, :end_pos, :],
				encoder_hidden_states[:, end_pos:, :],
			)
			if attn.norm_cross:
				encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)
		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)

		# for ip-adapter
		ip_key = self.to_k_ip(ip_hidden_states)
		ip_value = self.to_v_ip(ip_hidden_states)

		ip_key = attn.head_to_batch_dim(ip_key)
		ip_value = attn.head_to_batch_dim(ip_value)

		ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
		# self.attn_map = ip_attention_probs
		ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
		ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

		hidden_states = hidden_states + self.scale * ip_hidden_states

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


class AttnProcessor2_0(torch.nn.Module):
	r"""
	Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
	"""

	def __init__(
		self,
		hidden_size=None,
		cross_attention_dim=None,
	):
		super().__init__()
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

	def __call__(
		self,
		attn,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
		timestep: Optional[torch.Tensor] = None,
	):
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

		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		# the output of sdp = (batch, num_heads, seq_len, head_dim)
		# TODO: add support for attn.scale when we move to Torch 2.1
		hidden_states = F.scaled_dot_product_attention(
			query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
		)

		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = hidden_states.to(query.dtype)

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


class IPAttnProcessor2_0(torch.nn.Module):
	r"""
	Attention processor for IP-Adapater for PyTorch 2.0.
	Args:
		hidden_size (`int`):
			The hidden size of the attention layer.
		cross_attention_dim (`int`):
			The number of channels in the `encoder_hidden_states`.
		scale (`float`, defaults to 1.0):
			the weight scale of image prompt.
		num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
			The context length of the image features.
	"""

	def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
		super().__init__()

		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

		self.hidden_size = hidden_size
		self.cross_attention_dim = cross_attention_dim
		self.scale = scale
		self.num_tokens = num_tokens

		self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
		self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

	def __call__(
		self,
		attn,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
		height: int = None,
		width: int = None,
		timestep: Optional[torch.Tensor] = None,
		*args,
		**kwargs,
	):
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

		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		else:
			# get encoder_hidden_states, ip_hidden_states
			end_pos = encoder_hidden_states.shape[1] - self.num_tokens
			encoder_hidden_states, ip_hidden_states = (
				encoder_hidden_states[:, :end_pos, :],
				encoder_hidden_states[:, end_pos:, :],
			)
			if attn.norm_cross:
				encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		# the output of sdp = (batch, num_heads, seq_len, head_dim)
		# TODO: add support for attn.scale when we move to Torch 2.1
		####################################################################################################
		if hasattr(self, "store_attn_map"):
			hidden_states, attention_probs = scaled_dot_product_attention(
				query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
			)

			if "token_last_idx" in kwargs:
				""" re-weight the attention values by ignoring the attention of ⟨𝑠𝑜𝑡⟩ and <eot> """
				attention_probs = attention_probs[:, :, :, 1:kwargs["token_last_idx"]]
				attention_probs *= 1000
				attention_probs = torch.nn.functional.softmax(attention_probs, dim=-1)

			attention_probs = rearrange(attention_probs, 'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ', h=height) # detach height*width
			if attention_probs.dim() != 5:
				import pudb; pudb.set_trace()

			self.attn_map = attention_probs
			self.timestep = int(timestep.item())

		else:
			hidden_states = F.scaled_dot_product_attention(
				query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
			)
		####################################################################################################

		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = hidden_states.to(query.dtype)

		# for ip-adapter
		ip_key = self.to_k_ip(ip_hidden_states)
		ip_value = self.to_v_ip(ip_hidden_states)

		ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		# the output of sdp = (batch, num_heads, seq_len, head_dim)
		# TODO: add support for attn.scale when we move to Torch 2.1
		ip_hidden_states = F.scaled_dot_product_attention(
			query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
		)
		# with torch.no_grad():
			# self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
			#print(self.attn_map.shape)

		ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		ip_hidden_states = ip_hidden_states.to(query.dtype)

		hidden_states = hidden_states + self.scale * ip_hidden_states

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


## for controlnet
class CNAttnProcessor:
	r"""
	Default processor for performing attention-related computations.
	"""

	def __init__(self, num_tokens=4):
		self.num_tokens = num_tokens

	def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
				height: int = None,
				width: int = None,
				timestep: Optional[torch.Tensor] = None,
				*args,
				**kwargs,
		):
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
		else:
			end_pos = encoder_hidden_states.shape[1] - self.num_tokens
			encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
			if attn.norm_cross:
				encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)

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


class CNAttnProcessor2_0:
	r"""
	Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
	"""

	def __init__(self, num_tokens=4):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
		self.num_tokens = num_tokens

	def __call__(
		self,
		attn,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
		height: int = None,
		width: int = None,
		timestep: Optional[torch.Tensor] = None,
		*args,
		**kwargs,
	):
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

		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		else:
			end_pos = encoder_hidden_states.shape[1] - self.num_tokens
			encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
			if attn.norm_cross:
				encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		# the output of sdp = (batch, num_heads, seq_len, head_dim)
		# TODO: add support for attn.scale when we move to Torch 2.1
		"""
		hidden_states = F.scaled_dot_product_attention(
			query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
		)
		"""
		####################################################################################################
		if hasattr(self, "store_attn_map"):
			hidden_states, attention_probs = scaled_dot_product_attention(
				query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
			)

			if "token_last_idx" in kwargs:
				""" re-weight the attention values by ignoring the attention of ⟨𝑠𝑜𝑡⟩ and <eot> """
				attention_probs = attention_probs[:, :, :, 1:kwargs["token_last_idx"]]
				attention_probs *= 1000
				attention_probs = torch.nn.functional.softmax(attention_probs, dim=-1)

			attention_probs = rearrange(attention_probs, 'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ', h=height) # detach height*width
			if len(attention_probs.shape) != 5:
				import pudb; pudb.set_trace()

			self.attn_map = attention_probs
			self.timestep = int(timestep.item())

		else:
			hidden_states = F.scaled_dot_product_attention(
				query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
			)
		####################################################################################################


		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = hidden_states.to(query.dtype)

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
