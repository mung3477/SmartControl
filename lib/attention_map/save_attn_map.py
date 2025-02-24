import os
from typing import List, TypedDict

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from transformers import CLIPTokenizer

from ..utils.file import assert_path


class AttnSaveOptions(TypedDict):
	prefix: str = ""
	return_dict: bool = False
	ignore_special_tkns: bool = True
	enabled_editing_prompts: int = 0

default_option: AttnSaveOptions = {
	'prefix': "",
	'return_dict': False,
	'ignore_special_tkns': True,
	'enabled_editing_prompts': 0
}

def tokenize_prompts(prompts: List[str], tokenizer: CLIPTokenizer, ignore_special_tokens: bool):
	token_ids = tokenizer(prompts)['input_ids']

	tokens_list = []
	for token_id in token_ids:
		if ignore_special_tokens:
			token_id = token_id[1:-1]
		tokens_list.append(tokenizer.convert_ids_to_tokens(token_id))
	return tokens_list

def mark_words(tokens_list: List[List[str]]):
	marked_tokens_list = []
	for tokens in tokens_list:
		marked_tokens = []
		startofword = True
		for token in tokens:
			if '</w>' in token:
				token = token.replace('</w>', '')
				if startofword:
					token = '<' + token + '>'
				else:
					token = '-' + token + '>'
					startofword = True

			elif token != '<|startoftext|>' and token != '<|endoftext|>':
				if startofword:
					token = '<' + token + '-'
					startofword = False
				else:
					token = '-' + token + '-'
			marked_tokens.append(token)
		marked_tokens_list.append(marked_tokens)
	return marked_tokens_list

def tokenize_and_mark_prompts(prompts: List[str], tokenizer: CLIPTokenizer, ignore_special_tokens: bool):
	total_tokens = tokenize_prompts(prompts=prompts, tokenizer=tokenizer, ignore_special_tokens=ignore_special_tokens)
	return mark_words(total_tokens)

def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='log/attn_maps', unconditional=True, options: AttnSaveOptions = default_option):
	to_pil = ToPILImage()

	bsz = options["enabled_editing_prompts"] + int(unconditional) + 1
	total_marked_tokens = tokenize_and_mark_prompts(prompts=prompts, tokenizer=tokenizer, ignore_special_tokens=options["ignore_special_tkns"])
	organized_attn_maps = {}

	assert_path(base_dir)

	total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1) # If we use AttnProcessor2.0, batch attn_head h w attn_dim -> batch h w attn_dim
	if unconditional:
		total_attn_map = total_attn_map.chunk(bsz)[1]  # (batch, height, width, attn_dim)


	total_attn_map = total_attn_map.permute(0, 3, 1, 2)
	total_attn_map = torch.zeros_like(total_attn_map)
	total_attn_map_shape = total_attn_map.shape[-2:]
	total_attn_map_number = 0

	for timestep, layers in attn_maps.items():
		timestep_dir = os.path.join(base_dir, f'{timestep}')
		assert_path(timestep_dir)

		for layer, attn_map in layers.items():
			layer_dir = os.path.join(timestep_dir, f'{layer}')
			assert_path(layer_dir)

			attn_map = attn_map.sum(1) / attn_map.shape[1]
			attn_map = attn_map.permute(0, 3, 1, 2)

			if unconditional:
				attn_map = attn_map.chunk(bsz)[1]

			resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
			total_attn_map += resized_attn_map
			total_attn_map_number += 1

			for batch, (tokens, attn) in enumerate(zip(total_marked_tokens, attn_map)):
				batch_dir = os.path.join(layer_dir, f'batch-{batch}')
				if not os.path.exists(batch_dir):
					os.mkdir(batch_dir)

				organized_attn_maps[batch_dir] = {}

				for i, (token, a) in enumerate(zip(tokens, attn[:len(tokens)])):
					filename = f'{options["prefix"]}{i}-{token}.png'
					organized_attn_maps[batch_dir][filename] = a

					a_img = to_pil(a.to(torch.float32))
					a_img.save(os.path.join(batch_dir, filename))


	total_attn_map /= total_attn_map_number
	for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_marked_tokens)):
		batch_dir = os.path.join(base_dir, f'batch-{batch}')
		if not os.path.exists(batch_dir):
			os.mkdir(batch_dir)

		for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
			a_img = to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))

	if options["return_dict"]:
		return organized_attn_maps
	return None
